"""HippoRAG 2: memory-style retrieval over an open KG with Personalized PageRank (arXiv 2502.14802).

The neurobiological framing: an LLM acts as the neocortex (extracting open triples offline and
filtering them online), a knowledge graph as the hippocampal index, and Personalized PageRank
as the associative recall that spreads activation from what the query names to the passages
that answer it. Its distinguishing result is that, unlike GraphRAG/LightRAG, it improves
multi-hop WITHOUT degrading simple questions — which is why it is worth an A/B here, where the
question mix runs from a single dose lookup to a three-guide interaction.

Three design notes specific to this project:

  1. NATIVE implementation rather than the `hipporag` package. That package pins
     openai==1.91 / tiktoken==0.7 / vllm, which conflicts with this project's stack and does
     not install on Windows. The algorithm is small on top of what we already have —
     embeddings, an LLM, and networkx's PPR — and reimplementing keeps every LLM call
     encapsulated for the eventual Azure/EU swap.
  2. PASSAGE nodes are our own chunk_ids, so what PPR ranks IS what we cite: no mapping
     heuristics, and the grounding contract holds by construction.
  3. Phrases are canonicalized (`canonical_key`), so «DTG» and «dolutegravir» activate ONE
     node. Fragmented phrasing splits the PageRank mass that is the whole point of the method.

Usage:
    Build the index once:  .venv\\Scripts\\python.exe -m retrieval.hipporag
    (resumable: re-running only extracts the chunks still missing)
    Inspect a demo query:  .venv\\Scripts\\python.exe -m retrieval.hipporag --smoke
"""
import asyncio
import json
import os
import sys
import threading
from dataclasses import dataclass
from typing import cast

import numpy as np
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: keep accents intact
except (AttributeError, ValueError):
    pass

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import corpus
from rag import (EMBEDDING_MODEL, REPHRASE_MODEL, _ABBREV_LIST, chat_model, client,
                 get_embedding)

from ._common import canonical_key, house_tail, load_chunks, map_chunk_ids_to_payloads

# --- Config ---------------------------------------------------------------
STORE_DIR = corpus.hipporag_dir()
OPENIE_PATH = os.path.join(STORE_DIR, "openie.jsonl")        # resumability unit (append-only)
GRAPH_PATH = os.path.join(STORE_DIR, "graph.graphml")
PHRASES_PATH = os.path.join(STORE_DIR, "phrases.json")
PHRASE_EMB_PATH = os.path.join(STORE_DIR, "phrase_emb.npy")
TRIPLES_PATH = os.path.join(STORE_DIR, "triples.json")
TRIPLE_EMB_PATH = os.path.join(STORE_DIR, "triple_emb.npy")
PASSAGES_PATH = os.path.join(STORE_DIR, "passages.json")
PASSAGE_EMB_PATH = os.path.join(STORE_DIR, "passage_emb.npy")
MANIFEST_PATH = os.path.join(STORE_DIR, "manifest.json")

EMBEDDING_DIM = 3072                         # EMBEDDING_MODEL (rag.py): same space as Qdrant
EMBED_BATCH = 256                            # texts per embeddings request during the build
OPENIE_CONCURRENCY = 8                       # parallel extraction calls (build only)

# Paper hyperparameters.
SYNONYM_THRESHOLD = 0.85   # cosine above which two phrases get a synonym edge
PASSAGE_WEIGHT = 0.05      # reset-probability weight of passage vs phrase nodes (paper §6.2)
TRIPLE_TOP_K = 5           # triples retrieved for the recognition-memory filter
PPR_ALPHA = 0.85           # PageRank damping
PASSAGE_TOP_N = 20         # passages taken from the PPR ranking into the rerank

PASSAGE_PREFIX = "passage::"   # namespaces passage nodes against phrase nodes in one graph


_OPENIE_SYS = f"""Eres un extractor de TRIPLETAS (OpenIE) para construir un grafo de conocimiento sobre las guías clínicas de VIH (GeSIDA). NO respondes preguntas ni resumes.

Extrae del fragmento TODAS las relaciones factuales relevantes en forma de tripletas (sujeto, relación, objeto):
- El sujeto y el objeto son FRASES CORTAS y autocontenidas: fármacos, patologías, coinfecciones, situaciones del paciente (gestación, insuficiencia renal…), pruebas, parámetros, pautas o recomendaciones.
- La relación es un verbo o locución verbal breve que exprese el hecho («está recomendado en», «está contraindicado en», «reduce», «requiere ajuste de dosis en»…).
- Extrae SOLO lo que el fragmento afirma. No añadas conocimiento externo ni inferencias.
- Conserva los datos clínicos precisos (dosis, grados de recomendación, umbrales) dentro de la frase cuando formen parte del hecho.
- Entre 5 y 20 tripletas según la densidad del fragmento.

NORMALIZACIÓN: para cualquier fármaco/término de la lista (sigla o nombre), escríbelo SIEMPRE como «nombre completo (SIGLA)». Solo términos de la lista; no inventes siglas.

LISTA DE ABREVIATURAS (SIGLA = nombre):
{_ABBREV_LIST}

Devuelve triples: lista de objetos con subject, relation y object (strings en español)."""


_FILTER_SYS = """Eres la MEMORIA DE RECONOCIMIENTO de un asistente sobre guías clínicas de VIH: decides qué hechos recuperados son REALMENTE pertinentes para la pregunta. NO respondes la pregunta.

Te doy una PREGUNTA y una lista numerada de TRIPLETAS (hechos extraídos de las guías). Devuelve los índices (empezando en 0) de las tripletas que ayudan a responder esa pregunta concreta: las que mencionan las entidades de la pregunta o un paso intermedio necesario para llegar a la respuesta.

Sé estricto: descarta las tripletas genéricas o de otro tema. Si ninguna es pertinente, devuelve una lista vacía.

Devuelve relevant_indices (lista de enteros)."""


class _Triple(BaseModel):
    subject: str
    relation: str
    object: str


class _OpenIE(BaseModel):
    triples: list[_Triple]


class _Filter(BaseModel):
    relevant_indices: list[int]


# ---------------------------------------------------------------------------
# INDEX BUILD  (run as a module:  python -m retrieval.hipporag)
# ---------------------------------------------------------------------------
def _openie_schema() -> dict:
    """JSON schema for the extraction call (the build uses the raw OpenAI client, so the
    structured-output contract is declared here instead of through LangChain)."""
    return {
        "name": "openie",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["triples"],
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["subject", "relation", "object"],
                        "properties": {
                            "subject": {"type": "string"},
                            "relation": {"type": "string"},
                            "object": {"type": "string"},
                        },
                    },
                }
            },
        },
    }


async def _extract_chunk(chunk: dict, semaphore: asyncio.Semaphore) -> dict:
    """OpenIE over one chunk. A failure yields no triples rather than aborting the build: the
    chunk simply contributes no phrase nodes (it stays reachable through hybrid search)."""
    async with semaphore:
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=REPHRASE_MODEL,
                temperature=0,
                messages=[{"role": "system", "content": _OPENIE_SYS},
                          {"role": "user", "content": chunk["text"]}],
                response_format={"type": "json_schema", "json_schema": _openie_schema()},
            )
            data = json.loads(response.choices[0].message.content)
            triples = [[t["subject"], t["relation"], t["object"]] for t in data["triples"]]
        except Exception as exc:
            print(f"  ! extraction failed for {chunk['chunk_id']}: {type(exc).__name__}")
            triples = []
        return {"chunk_id": chunk["chunk_id"], "triples": triples}


def _load_openie() -> dict[str, list]:
    if not os.path.isfile(OPENIE_PATH):
        return {}
    with open(OPENIE_PATH, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return {r["chunk_id"]: r["triples"] for r in records}


async def _run_openie(chunks: list[dict]) -> dict[str, list]:
    """Extract triples for the chunks not already in openie.jsonl, appending as they finish so
    an interrupted build resumes where it stopped (this is the only paid, slow step)."""
    done = _load_openie()
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"OpenIE: {len(done)} chunks already extracted, {len(todo)} to go.")
    if not todo:
        return done

    semaphore = asyncio.Semaphore(OPENIE_CONCURRENCY)
    tasks = [asyncio.create_task(_extract_chunk(c, semaphore)) for c in todo]
    with open(OPENIE_PATH, "a", encoding="utf-8") as fh:
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            record = await task
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            done[record["chunk_id"]] = record["triples"]
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} chunks extracted")
    return done


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts in batches, L2-normalized (so cosine is a dot product)."""
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(d.embedding for d in response.data)
        print(f"  embedded {min(start + EMBED_BATCH, len(texts))}/{len(texts)}")
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _linearize(triple: list[str]) -> str:
    return " ".join(part.strip() for part in triple)


def _build_graph(openie: dict[str, list], phrase_ids: dict[str, str],
                 phrase_emb: np.ndarray, phrase_keys: list[str]):
    """Assemble the hippocampal index: phrase nodes joined by relation edges, synonym edges
    between near-identical phrases, and passage nodes joined to the phrases they contain.

    The passage nodes are HippoRAG 2's key addition over HippoRAG: with only phrase nodes the
    graph loses the context a concept appeared in, which is what made the original degrade on
    simple questions."""
    import networkx as nx

    graph = nx.Graph()
    for key, label in phrase_ids.items():
        graph.add_node(key, label=label, kind="phrase")

    for chunk_id, triples in openie.items():
        passage = PASSAGE_PREFIX + chunk_id
        graph.add_node(passage, kind="passage")
        for subject, relation, obj in triples:
            s_key, o_key = canonical_key(subject), canonical_key(obj)
            if not s_key or not o_key:
                continue
            if s_key != o_key:
                graph.add_edge(s_key, o_key, kind="relation", relation=relation)
            # Context edges: the passage "contains" every phrase derived from it.
            for key in (s_key, o_key):
                graph.add_edge(passage, key, kind="context")

    # Synonym edges: phrases whose embeddings are near-identical but that canonicalization
    # could not merge (e.g. «insuficiencia renal» vs «deterioro de la función renal»).
    synonyms = 0
    for start in range(0, len(phrase_keys), 512):
        block = phrase_emb[start:start + 512]
        sims = block @ phrase_emb.T
        for local, row in enumerate(sims):
            i = start + local
            for j in np.where(row >= SYNONYM_THRESHOLD)[0]:
                if int(j) > i:
                    graph.add_edge(phrase_keys[i], phrase_keys[int(j)], kind="synonym")
                    synonyms += 1
    print(f"  synonym edges: {synonyms}")
    return graph


async def _build_index() -> None:
    import networkx as nx

    os.makedirs(STORE_DIR, exist_ok=True)
    chunks = load_chunks()
    openie = await _run_openie(chunks)

    # Phrase vocabulary: canonical key -> a readable label (the longest surface form seen).
    phrase_ids: dict[str, str] = {}
    for triples in openie.values():
        for subject, _relation, obj in triples:
            for surface in (subject, obj):
                key = canonical_key(surface)
                if key and len(surface) > len(phrase_ids.get(key, "")):
                    phrase_ids[key] = surface.strip()
    phrase_keys = list(phrase_ids)
    print(f"Phrases: {len(phrase_keys)} distinct concepts.")

    print("Embedding phrases…")
    phrase_emb = _embed_texts([phrase_ids[k] for k in phrase_keys])

    triples_flat = [{"chunk_id": cid, "triple": t}
                    for cid, ts in openie.items() for t in ts]
    print(f"Triples: {len(triples_flat)}. Embedding…")
    triple_emb = _embed_texts([_linearize(t["triple"]) for t in triples_flat])

    # Passage embeddings back every query's reset vector, so they are computed once here
    # rather than on the first query of every process.
    print("Embedding passages…")
    passage_ids = [c["chunk_id"] for c in chunks]
    passage_emb = _embed_texts([c["text"] for c in chunks])

    print("Assembling the graph…")
    graph = _build_graph(openie, phrase_ids, phrase_emb, phrase_keys)

    nx.write_graphml(graph, GRAPH_PATH)
    with open(PHRASES_PATH, "w", encoding="utf-8") as fh:
        json.dump({"keys": phrase_keys, "labels": phrase_ids}, fh, ensure_ascii=False)
    np.save(PHRASE_EMB_PATH, phrase_emb)
    with open(TRIPLES_PATH, "w", encoding="utf-8") as fh:
        json.dump(triples_flat, fh, ensure_ascii=False)
    np.save(TRIPLE_EMB_PATH, triple_emb)
    with open(PASSAGES_PATH, "w", encoding="utf-8") as fh:
        json.dump(passage_ids, fh)
    np.save(PASSAGE_EMB_PATH, passage_emb)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump({"embedding_model": EMBEDDING_MODEL, "embedding_dim": EMBEDDING_DIM,
                   "n_chunks": len(chunks), "n_phrases": len(phrase_keys),
                   "n_triples": len(triples_flat),
                   "synonym_threshold": SYNONYM_THRESHOLD,
                   "passage_weight": PASSAGE_WEIGHT}, fh, indent=1)

    print(f"\nIndexed into {STORE_DIR}/: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges ({len(chunks)} passages, "
          f"{len(phrase_keys)} phrases, {len(triples_flat)} triples).")


# ---------------------------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------------------------
@dataclass
class _HippoStore:
    graph: "object"                 # networkx.Graph (phrase + passage nodes)
    triples: list[dict]             # [{chunk_id, triple}], row-aligned with triple_emb
    triple_emb: np.ndarray          # (n_triples, 3072), L2-normalized
    passage_ids: list[str]          # our chunk_ids, row-aligned with passage_emb
    passage_emb: np.ndarray         # (n_chunks, 3072), L2-normalized


def _build_store() -> _HippoStore:
    import networkx as nx

    if not os.path.isfile(GRAPH_PATH):
        raise SystemExit(
            f"{STORE_DIR}/ is missing or incomplete. Build the index first: "
            f".venv\\Scripts\\python.exe -m retrieval.hipporag"
        )
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("embedding_dim") != EMBEDDING_DIM:
        raise SystemExit(
            f"{MANIFEST_PATH}: the index was built with {manifest.get('embedding_model')} "
            f"({manifest.get('embedding_dim')}d); rebuild it for {EMBEDDING_MODEL}."
        )
    with open(TRIPLES_PATH, encoding="utf-8") as fh:
        triples = json.load(fh)
    with open(PASSAGES_PATH, encoding="utf-8") as fh:
        passage_ids = json.load(fh)
    return _HippoStore(graph=nx.read_graphml(GRAPH_PATH), triples=triples,
                       triple_emb=np.load(TRIPLE_EMB_PATH),
                       passage_ids=passage_ids, passage_emb=np.load(PASSAGE_EMB_PATH))


_store: _HippoStore | None = None
_store_lock = threading.Lock()
def _get_store() -> _HippoStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = _build_store()
    return _store


_filter_llm = None
def _get_filter_llm():
    global _filter_llm
    if _filter_llm is None:
        _filter_llm = chat_model(REPHRASE_MODEL, temperature=0).with_structured_output(
            _Filter, method="json_schema", strict=True
        )
    return _filter_llm


def _embed_query(query: str) -> np.ndarray:
    vec = np.asarray(get_embedding(query), dtype=np.float32)
    return vec / (np.linalg.norm(vec) or 1.0)


def retrieve_triples(query_vec: np.ndarray, top_k: int = TRIPLE_TOP_K) -> list[tuple[int, float]]:
    """Query-to-triple linking (paper §3.3): match the WHOLE query against triple embeddings.
    Matching the query rather than extracted entities is what gives HippoRAG 2 its contextual
    grounding — «¿qué TAR en gestante con VHB?» scores triples about that combination, not
    every triple mentioning TAR."""
    store = _get_store()
    sims = store.triple_emb @ query_vec
    order = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in order]


def filter_triples(query: str, candidates: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Recognition memory (paper §3.4): an LLM drops the triples that merely look similar.
    These become the PPR seeds, so a bad seed misdirects the whole walk. On LLM failure we
    keep the candidates — degrading to plain HippoRAG behaviour beats losing retrieval."""
    if not candidates:
        return []
    store = _get_store()
    listing = "\n".join(f"{n}. {_linearize(store.triples[i]['triple'])}"
                        for n, (i, _s) in enumerate(candidates))
    try:
        verdict = cast(_Filter, _get_filter_llm().invoke(
            [("system", _FILTER_SYS), ("human", f"PREGUNTA:\n{query}\n\nTRIPLETAS:\n{listing}")]))
        keep = {n for n in verdict.relevant_indices if 0 <= n < len(candidates)}
        return [c for n, c in enumerate(candidates) if n in keep]
    except Exception:
        return candidates


def _reset_probabilities(query_vec: np.ndarray,
                         filtered: list[tuple[int, float]]) -> dict[str, float]:
    """Seed the PPR walk (paper §3.5 and appendix G.1).

    Phrase nodes from the surviving triples get mass proportional to their triple's score;
    ALL passage nodes get mass proportional to their similarity to the query, scaled by
    PASSAGE_WEIGHT. Seeding every passage (not just matched ones) is deliberate: it keeps a
    dense-retrieval prior underneath the graph walk, which is why the method does not
    degrade simple questions the way a purely entity-seeded walk does."""
    store = _get_store()
    reset: dict[str, float] = {}

    for index, score in filtered:
        record = store.triples[index]
        weight = max(score, 0.0)
        for surface in (record["triple"][0], record["triple"][2]):
            key = canonical_key(surface)
            if key in store.graph:
                reset[key] = reset.get(key, 0.0) + weight

    sims = store.passage_emb @ query_vec
    for chunk_id, sim in zip(store.passage_ids, sims):
        node = PASSAGE_PREFIX + chunk_id
        if node in store.graph:
            reset[node] = reset.get(node, 0.0) + PASSAGE_WEIGHT * max(float(sim), 0.0)

    total = sum(reset.values())
    return {k: v / total for k, v in reset.items()} if total > 0 else {}


def rank_passages(query: str, top_n: int = PASSAGE_TOP_N) -> list[str]:
    """The full online path: query -> triples -> recognition filter -> PPR -> ranked chunk_ids.
    Returns [] when there is nothing to seed with, letting the caller fall back to dense."""
    import networkx as nx

    store = _get_store()
    query_vec = _embed_query(query)
    filtered = filter_triples(query, retrieve_triples(query_vec))
    reset = _reset_probabilities(query_vec, filtered)
    if not reset:
        return []
    try:
        scores = nx.pagerank(store.graph, alpha=PPR_ALPHA, personalization=reset)
    except nx.PowerIterationFailedConvergence:
        return []
    passages = [(node[len(PASSAGE_PREFIX):], score) for node, score in scores.items()
                if node.startswith(PASSAGE_PREFIX)]
    passages.sort(key=lambda kv: -kv[1])
    return [chunk_id for chunk_id, _score in passages[:top_n]]


def hipporag_search(query: str, top_k: int = 8, rewritten_query: str | None = None,
                    scope: corpus.Scope | None = None) -> list:
    """HippoRAG 2 retriever, honouring the shared retrieval contract. The PPR ranking selects
    the primary passages; `house_tail` adds the dense+BM25 complement and reranks — so an
    empty graph result (no triple survived the filter) degrades to the dense fallback the
    paper prescribes rather than to no evidence at all."""
    primary = map_chunk_ids_to_payloads(rank_passages(query))
    return house_tail(query, primary, rewritten_query, top_k=top_k, scope=scope)


def _smoke() -> None:
    """Inspect the index and the online stages of a demo query. Everything up to the final
    rerank is checked here, so the graph side can be validated without Qdrant."""
    store = _get_store()
    graph = store.graph
    passages = [n for n in graph.nodes if n.startswith(PASSAGE_PREFIX)]
    print(f"Index: {graph.number_of_nodes()} nodes ({len(passages)} passages, "
          f"{graph.number_of_nodes() - len(passages)} phrases), "
          f"{graph.number_of_edges()} edges, {len(store.triples)} triples.")

    demo = ("En un paciente con coinfección por VHB que inicia TAR, "
            "¿qué fármacos debe incluir el régimen?")
    print(f"\nQuery: {demo}")
    query_vec = _embed_query(demo)
    candidates = retrieve_triples(query_vec)
    print("Retrieved triples:")
    for i, score in candidates:
        print(f"  {score:.3f}  {_linearize(store.triples[i]['triple'])}")
    filtered = filter_triples(demo, candidates)
    print(f"Kept by recognition memory: {len(filtered)}/{len(candidates)}")
    for chunk_id in rank_passages(demo)[:8]:
        payload = map_chunk_ids_to_payloads([chunk_id])[0]
        print(f"  {chunk_id} | {payload['source_file']} | {(payload.get('heading') or '')[:60]}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        asyncio.run(_build_index())
