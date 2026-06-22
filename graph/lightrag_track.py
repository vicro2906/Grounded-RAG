"""
graph/lightrag_track.py — Track B (Phase 4): LightRAG GRAPH layer for MULTI-HOP questions.

Builds an entity-relation knowledge graph over the SAME chunks already indexed in Qdrant
(LightRAG's default FILE storage, no new infra) and exposes:

    graph_search(query) -> list[payload]   # our chunk dicts, with metadata

so the rest of the pipeline (generate -> validate -> evidence, including the literal-
citation panel) keeps working UNCHANGED. LightRAG is used ONLY to SELECT source chunks
via graph traversal (aquery_data); it does NOT write the final answer — our gpt-4o
generator + validator do, exactly as in the baseline and the agentic track. This is the
design decision that preserves grounding/citations: the synthetic entity/relation
descriptions are never cited, only the original chunks they point to.

Shared resources come from the PARENT folder: the reranker from `rag.py`, the corpus from
`chunks/chunks.jsonl`, and the graph store at `lightrag_store/` (all at project root).

Usage:
    1. Build the graph once:   .venv\\Scripts\\python.exe -m graph.lightrag_track
    2. Evaluate it:            set PIPELINE = "graph" in evaluation.py and run it.

RGPD / encapsulation: the LLM and embedding are the SAME providers as the rest of the
system and are isolated in LLM_COMPLETE / _embed below, so switching to Azure OpenAI (EU)
is a one-line change (azure_openai_complete / azure_openai_embed + AZURE_* env vars).
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

from rag import rerank  # shared cross-encoder (parent module)

# --- Config ---------------------------------------------------------------
WORKING_DIR = os.path.join(ROOT, "lightrag_store")          # file-based graph + vector store
CHUNKS_PATH = os.path.join(ROOT, "chunks", "chunks.jsonl")  # shared corpus (parent folder)
EMBEDDING_MODEL = "text-embedding-3-large"                  # same as Qdrant -> 3072 dims
EMBEDDING_DIM = 3072

# LLM used by LightRAG for entity/relation extraction (indexing) and keyword extraction
# (query). gpt-4o-mini is enough and cheap for a small, stable corpus. Swap both funcs
# for the azure_openai_* variants when RGPD requires a private EU model.
LLM_COMPLETE = gpt_4o_mini_complete

# Build-time concurrency (one-time index): defaults are tiny (max_parallel_insert=2,
# llm_model_max_async=4) which makes the build crawl. These do not affect query latency.
MAX_PARALLEL_INSERT = 8
LLM_MAX_ASYNC = 16
EMBED_MAX_ASYNC = 16


async def _embed(texts):
    return await openai_embed(texts, model=EMBEDDING_MODEL)


def _make_rag() -> LightRAG:
    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=LLM_COMPLETE,
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
    # Insert each chunk as its own document (they are < chunk_token_size, so each stays a
    # single LightRAG chunk). We pass our chunk_id as the id and source_file as file_path
    # for traceability; mapping back at query time is by exact content anyway. Re-running
    # resumes: already-processed docs are skipped and the LLM cache makes redo cheap.
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
                f"No existe {WORKING_DIR}/. Construye el grafo primero: "
                f".venv\\Scripts\\python.exe -m graph.lightrag_track"
            )
        _loop = asyncio.new_event_loop()
        _rag = _make_rag()
        _loop.run_until_complete(_rag.initialize_storages())
        _loop.run_until_complete(initialize_pipeline_status())
        atexit.register(_shutdown)
    return _loop, _rag


def _shutdown() -> None:
    """Clean teardown at process exit: LightRAG leaves background workers (priority-limit
    queues + a health check) on the loop; without this they raise 'Event loop is closed'
    noise when the interpreter tears down. We finalize storages, cancel any pending tasks,
    then close the loop."""
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


def graph_search(query: str, top_k: int = 8, chunk_top_k: int = 20) -> list:
    """Track B retriever: LightRAG selects source chunks via graph traversal (mix mode:
    entities + relations + vector), we map them back to our payloads and apply the same
    cross-encoder reranker as the other tracks for a comparable top_k. LightRAG's own
    rerank is disabled (no rerank model configured there; we rerank ourselves)."""
    loop, rag = _ensure_rag()
    data = loop.run_until_complete(rag.aquery_data(
        query,
        param=QueryParam(mode="mix", chunk_top_k=chunk_top_k, enable_rerank=False),
    ))
    chunks = (data.get("data") or {}).get("chunks") or []
    payloads = _map_to_payloads(chunks)
    if not payloads:
        return []
    return rerank(query, payloads, top_k=top_k)


if __name__ == "__main__":
    asyncio.run(_build_index())
