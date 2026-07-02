"""Track B (Phase 4): LightRAG GRAPH layer for MULTI-HOP questions.

Builds an entity-relation graph over the SAME chunks indexed in Qdrant (LightRAG file
storage, no new infra) and exposes `graph_search(query) -> list[payload]`, so the rest of the
pipeline (generate -> validate -> evidence, citations included) is UNCHANGED. LightRAG only
SELECTS source chunks via traversal; our gpt-4o generator writes the answer. This is what
preserves grounding: the synthetic entity/relation descriptions are never cited, only the
original chunks they point to. The LLM/embedding are isolated in LLM_COMPLETE / _embed so
switching to Azure OpenAI (EU, for GDPR) is a one-line change.

Usage:
    1. Build the graph once:   .venv\\Scripts\\python.exe -m graph.lightrag_track
    2. Evaluate it:            set PIPELINE = "graph" in evaluation.py and run it.
"""
import os
import sys
import json
import atexit
import asyncio

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: keep accents/boxes intact
except (AttributeError, ValueError):
    pass

# Project root = parent of this package, so the shared parent modules/files resolve no
# matter the cwd (and `python graph/lightrag_track.py` finds `rag` on sys.path too).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete

from concurrent.futures import ThreadPoolExecutor

from rag import (rerank, retrieve_hybrid, rephrase,  # shared retrieval primitives (parent)
                 _get_reranker, _get_bm25)

# --- Config ---------------------------------------------------------------
WORKING_DIR = os.path.join(ROOT, "lightrag_store")          # file-based graph + vector store
CHUNKS_PATH = os.path.join(ROOT, "chunks", "chunks.jsonl")  # shared corpus (parent folder)
EMBEDDING_MODEL = "text-embedding-3-large"                  # same as Qdrant -> 3072 dims
EMBEDDING_DIM = 3072

# LLM for entity/relation extraction (indexing) and keyword extraction (query). gpt-4o-mini
# is enough and cheap for a small, stable corpus. Swap for azure_openai_* when GDPR requires it.
LLM_COMPLETE = gpt_4o_mini_complete

# Build-time concurrency (one-time index): the defaults (2 / 4) make the build crawl. These
# do not affect query latency.
MAX_PARALLEL_INSERT = 8
LLM_MAX_ASYNC = 16
EMBED_MAX_ASYNC = 16


async def _embed(texts):
    return await openai_embed(texts, model=EMBEDDING_MODEL)


def _traceable_llm(func):
    """Wrap LightRAG's internal keyword-extraction LLM with LangSmith's @traceable so it nests
    under the graph run (LightRAG uses its own OpenAI client, which main.py's wrap_openai does
    not reach). No-op without LANGSMITH_API_KEY; not applied during the index build."""
    if not os.environ.get("LANGSMITH_API_KEY"):
        return func
    try:
        from langsmith import traceable
        return traceable(name="lightrag_llm", run_type="llm")(func)
    except Exception:
        return func


def _make_rag(trace_llm: bool = False) -> LightRAG:
    llm_func = _traceable_llm(LLM_COMPLETE) if trace_llm else LLM_COMPLETE
    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM, max_token_size=8192, func=_embed
        ),
        max_parallel_insert=MAX_PARALLEL_INSERT,
        llm_model_max_async=LLM_MAX_ASYNC,
        embedding_func_max_async=EMBED_MAX_ASYNC,
    )


# ---------------------------------------------------------------------------
# INDEX BUILD  (run as a module:  python -m graph.lightrag_track)
# ---------------------------------------------------------------------------
def _load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


async def _build_index() -> None:
    os.makedirs(WORKING_DIR, exist_ok=True)
    rag = _make_rag()
    await rag.initialize_storages()
    await initialize_pipeline_status()

    chunks = _load_chunks()
    # Each chunk is its own document (all < chunk_token_size, so each stays one LightRAG
    # chunk). Re-running resumes: processed docs are skipped and the LLM cache makes redo cheap.
    await rag.ainsert(
        input=[c["text"] for c in chunks],
        ids=[c["chunk_id"] for c in chunks],
        file_paths=[c["source_file"] for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks into {WORKING_DIR}/ (entity-relation graph built).")
    await rag.finalize_storages()


# ---------------------------------------------------------------------------
# RETRIEVAL  (imported by evaluation.py and by the graph node in main.py)
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    """Whitespace-collapsed key for matching LightRAG's returned content to our chunks."""
    return " ".join((text or "").split())


_chunk_lookup: dict | None = None
def _get_chunk_lookup() -> dict:
    """Map normalized chunk text -> our payload (with full metadata), built once."""
    global _chunk_lookup
    if _chunk_lookup is None:
        _chunk_lookup = {_norm(c["text"]): c for c in _load_chunks()}
    return _chunk_lookup


# Persistent LightRAG instance + event loop so the eval run does not re-init per query.
_loop = None
_rag = None
def _ensure_rag():
    global _loop, _rag
    if _rag is None:
        if not os.path.isdir(WORKING_DIR):
            raise SystemExit(
                f"{WORKING_DIR}/ does not exist. Build the graph first: "
                f".venv\\Scripts\\python.exe -m graph.lightrag_track"
            )
        _loop = asyncio.new_event_loop()
        _rag = _make_rag(trace_llm=True)  # query path: trace LightRAG's internal LLM
        _loop.run_until_complete(_rag.initialize_storages())
        _loop.run_until_complete(initialize_pipeline_status())
        atexit.register(_shutdown)
    return _loop, _rag


def _shutdown() -> None:
    """Clean teardown at process exit: finalize storages, cancel LightRAG's leftover background
    workers, then close the loop (otherwise they raise 'Event loop is closed' noise)."""
    global _loop, _rag
    if _loop is None:
        return
    try:
        if _rag is not None:
            _loop.run_until_complete(_rag.finalize_storages())
        pending = asyncio.all_tasks(_loop)
        for t in pending:
            t.cancel()
        if pending:
            _loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
        pass
    finally:
        _loop.close()
        _loop, _rag = None, None


def _map_to_payloads(chunks: list[dict]) -> list[dict]:
    """Map LightRAG's selected chunks (by exact content) back to our payloads. Content is
    inserted and returned verbatim, so an exact normalized match is expected; a prefix
    fallback covers any edge truncation."""
    lookup = _get_chunk_lookup()
    prefix_index = None
    out, seen = [], set()
    for ch in chunks:
        content = ch.get("content", "")
        payload = lookup.get(_norm(content))
        if payload is None:
            # prefix fallback (build the prefix index lazily, only if needed)
            if prefix_index is None:
                prefix_index = {k[:120]: v for k, v in lookup.items()}
            payload = prefix_index.get(_norm(content)[:120])
        if payload is not None:
            key = payload["chunk_id"]
            if key not in seen:
                seen.add(key)
                out.append(payload)
    return out


def _merge_dedup(*lists) -> list:
    """Concatenate payload lists, dedup by chunk_id (fallback text), preserving order
    (earlier lists win — graph chunks first, then the hybrid complement)."""
    out, seen = [], set()
    for lst in lists:
        for p in lst:
            key = p.get("chunk_id") or p.get("text", "")
            if key and key not in seen:
                seen.add(key)
                out.append(p)
    return out


def graph_traverse(query: str, chunk_top_k: int = 20) -> list:
    """LightRAG's ENTITY + RELATION traversal (mode='hybrid' = entity+relation, NOT vector
    hybrid; it drops LightRAG's naive dense chunk search so we plug in our own), mapping the
    reached source chunks back to our payloads. Exposed on its own so the expanded graph can
    show the traversal as a distinct step."""
    loop, rag = _ensure_rag()
    data = loop.run_until_complete(rag.aquery_data(
        query,
        param=QueryParam(mode="hybrid", chunk_top_k=chunk_top_k, enable_rerank=False),
    ))
    graph_chunks = (data.get("data") or {}).get("chunks") or []
    return _map_to_payloads(graph_chunks)


def graph_extract_keywords(query: str, mode: str = "hybrid") -> dict:
    """The LLM step of the traversal: extract HIGH-LEVEL keywords (-> matched to RELATION
    vectors) and LOW-LEVEL keywords (-> ENTITY vectors). Exposed on its own so the expanded
    graph can show this step; replicates how LightRAG calls it internally."""
    from dataclasses import asdict
    from lightrag.operate import extract_keywords_only
    loop, rag = _ensure_rag()
    hl, ll = loop.run_until_complete(
        extract_keywords_only(query, QueryParam(mode=mode),
                              asdict(rag), hashing_kv=rag.llm_response_cache)
    )
    return {"hl_keywords": hl, "ll_keywords": ll}


def graph_select(query: str, hl_keywords: list, ll_keywords: list,
                 chunk_top_k: int = 20) -> dict:
    """Vector search + graph walk (NO LLM). With the keywords pre-extracted, LightRAG cosine-
    matches low-level vs the ENTITY store and high-level vs the RELATION store, walks the graph
    and gathers the source chunks via source_id. Passing the keywords makes it skip its own
    extraction. Returns the mapped payloads + the selected entities/relationships (for the state)."""
    loop, rag = _ensure_rag()
    param = QueryParam(mode="hybrid", chunk_top_k=chunk_top_k, enable_rerank=False,
                       hl_keywords=hl_keywords or [], ll_keywords=ll_keywords or [])
    data = loop.run_until_complete(rag.aquery_data(query, param=param))
    d = data.get("data") or {}
    return {
        "payloads": _map_to_payloads(d.get("chunks") or []),
        "entities": d.get("entities") or [],
        "relationships": d.get("relationships") or [],
    }


def graph_search(query: str, top_k: int = 8, chunk_top_k: int = 20, hybrid_k: int = 10,
                 hybrid_query: str | None = None) -> list:
    """Track B retriever (collapsed single-call API, used by the eval and the combined graph).
    Two complementary chunk sources, merged (dedup by chunk_id) and reranked to top_k:
      1. GRAPH traversal (graph_traverse) — the multi-hop signal.
      2. our HYBRID search (dense + BM25 RRF) on the rephrased query — replaces LightRAG's
         dense-only complement (BM25 helps with the guides' heavy abbreviation use).
    The two are independent, so they run IN PARALLEL (graph on LightRAG's loop, hybrid on a
    thread hitting Qdrant/OpenAI)."""
    _get_reranker(); _get_bm25()  # pre-warm local models before the parallel branches
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_graph = ex.submit(graph_traverse, query, chunk_top_k)
        # reuse the caller's rephrased query when available; else rephrase inside the thread
        fut_hybrid = ex.submit(
            lambda: retrieve_hybrid(hybrid_query or rephrase(query),
                                    top_k=hybrid_k, prefetch_limit=30))
        graph_payloads = fut_graph.result()
        hybrid_payloads = fut_hybrid.result()

    merged = _merge_dedup(graph_payloads, hybrid_payloads)
    if not merged:
        return []
    return rerank(query, merged, top_k=top_k)


if __name__ == "__main__":
    asyncio.run(_build_index())
