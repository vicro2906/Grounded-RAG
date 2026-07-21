"""PathRAG: relational-path retrieval with flow-based pruning (arXiv 2502.14902).

PathRAG's premise is that graph RAG's problem is REDUNDANCY, not insufficiency: LightRAG
feeds the LLM every immediate neighbour of the matched nodes, and the noise degrades the
answer. Instead it keeps only the key relational PATHS between the query-related nodes,
scored by a flow-based pruning algorithm with distance decay.

Two adaptations to this project, both driven by priority #1 (do not hallucinate):

  1. It reads the EXISTING LightRAG index (`data/lightrag_store/`) instead of building its
     own. PathRAG needs exactly what LightRAG already produced — an entity graph with
     described edges and per-node source chunks — so a second index would cost 1.5 h and
     duplicate the corpus for nothing. All the store-format knowledge lives in `_PathStore`,
     the single point coupled to lightrag-hku's on-disk layout.
  2. The paths are used TWICE, with different privileges. As a chunk SELECTOR they decide
     which literal chunks reach the prompt (those stay the only citable evidence, mapped back
     through `source_id`). As a CONCEPT MAP their textual form also reaches the generator, but
     explicitly marked non-citable — it may guide the reasoning across hops, never support a
     quote. The paper's path-based prompting, made safe for a clinical setting.

Usage:
    Smoke the store and a demo query:  .venv\\Scripts\\python.exe -m retrieval.pathrag
    (the LightRAG index must exist:    .venv\\Scripts\\python.exe -m retrieval.graph)
"""
import base64
import heapq
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag import REPHRASE_MODEL, _ABBREV_LIST, get_embedding, rephrase

from ._common import canonical_key, expand_abbrevs, house_tail, map_chunk_ids_to_payloads

# --- Config ---------------------------------------------------------------
STORE_DIR = os.path.join(ROOT, "data", "lightrag_store")
GRAPHML = os.path.join(STORE_DIR, "graph_chunk_entity_relation.graphml")
VDB_ENTITIES = os.path.join(STORE_DIR, "vdb_entities.json")
TEXT_CHUNKS = os.path.join(STORE_DIR, "kv_store_text_chunks.json")

EMBEDDING_DIM = 3072          # text-embedding-3-large, as indexed by retrieval.graph
SEP = "<SEP>"                 # LightRAG's separator inside multi-valued source_id fields

# Paper defaults (N=40 retrieved nodes, K=15 kept paths, decay alpha=0.7).
NODE_TOP_N = 40
MAX_PATHS = 15
DECAY = 0.7
PRUNE_THRESHOLD = 0.01        # early stop: below this a node contributes nothing downstream
MAX_HOPS = 4                  # safety cap; the decay reaches the threshold at ~4 hops anyway
DESC_CHARS = 220              # per-item truncation in the concept map (keeps tokens bounded)
# The paths of a broad question touch well over a hundred chunks; keeping the most reliable
# ones bounds the rerank (its cost is linear in candidates) and matches the graph mode's
# chunk_top_k=20 budget, so both graph modes hand the reranker a comparable pool.
PRIMARY_CHUNK_CAP = 20


_KEYWORDS_SYS = f"""Eres un extractor de PALABRAS CLAVE para buscar en un grafo de conceptos construido sobre las guías clínicas de VIH (GeSIDA). NO respondes la pregunta.

Extrae de la pregunta las ENTIDADES y CONCEPTOS clínicos concretos que habrá que localizar en el grafo: fármacos, patologías, coinfecciones, situaciones del paciente (gestación, insuficiencia renal…), pruebas y parámetros. De 3 a 8 palabras clave, cada una un término corto y autocontenido (no frases largas).

NORMALIZACIÓN: para cualquier fármaco/término de la lista (sigla o nombre), escríbelo en la forma «nombre completo (SIGLA)». Solo términos de la lista; no inventes siglas.

LISTA DE ABREVIATURAS (SIGLA = nombre):
{_ABBREV_LIST}

Devuelve keywords (lista de strings en español)."""


class _Keywords(BaseModel):
    keywords: list[str]


_kw_llm = None
def _get_keywords_llm():
    global _kw_llm
    if _kw_llm is None:
        _kw_llm = ChatOpenAI(model=REPHRASE_MODEL, temperature=0).with_structured_output(
            _Keywords, method="json_schema", strict=True
        )
    return _kw_llm


def extract_keywords(query: str) -> list[str]:
    """Stage 1 of the paper: the LLM names what to look for in the graph. On LLM failure it
    degrades to the rephrased query as a single keyword, so retrieval never blocks."""
    try:
        k = cast(_Keywords, _get_keywords_llm().invoke(
            [("system", _KEYWORDS_SYS), ("human", query)]))
        kws = [s.strip() for s in k.keywords if s and s.strip()]
        if kws:
            return kws
    except Exception:
        pass
    return [rephrase(query)]


# ---------------------------------------------------------------------------
# STORE  (the only code that knows lightrag-hku's on-disk layout)
# ---------------------------------------------------------------------------
@dataclass
class _PathStore:
    """The LightRAG index as PathRAG needs it: a walkable graph, node embeddings to match the
    query against, and the mapping back to our citable chunks."""
    graph: "object"                       # networkx.Graph (undirected entity graph)
    node_names: list[str]                 # row i of `node_emb` describes node_names[i]
    node_emb: np.ndarray                  # (n_nodes, 3072), L2-normalized
    node_row: dict[str, int] = field(default_factory=dict)
    lr_to_chunk: dict[str, str] = field(default_factory=dict)  # LightRAG chunk id -> our chunk_id


def _decode_matrix(vdb: dict) -> np.ndarray:
    """LightRAG stores the vectors as ONE base64 float32 blob, row-aligned with `data`."""
    dim = int(vdb["embedding_dim"])
    rows = len(vdb["data"])
    raw = base64.b64decode(vdb["matrix"])
    expected = rows * dim * 4
    if len(raw) != expected:
        raise SystemExit(
            f"{VDB_ENTITIES}: unexpected matrix size ({len(raw)} bytes, expected {expected} "
            f"for {rows}x{dim} float32). The lightrag-hku store layout changed — update "
            f"_PathStore in retrieval/pathrag.py."
        )
    return np.frombuffer(raw, dtype=np.float32).reshape(rows, dim)


def _build_store() -> _PathStore:
    import networkx as nx

    for path in (GRAPHML, VDB_ENTITIES, TEXT_CHUNKS):
        if not os.path.isfile(path):
            raise SystemExit(
                f"{path} not found. PathRAG reads the LightRAG index; build it first: "
                f".venv\\Scripts\\python.exe -m retrieval.graph"
            )

    graph = nx.read_graphml(GRAPHML)

    with open(VDB_ENTITIES, encoding="utf-8") as fh:
        vdb = json.load(fh)
    if int(vdb["embedding_dim"]) != EMBEDDING_DIM:
        raise SystemExit(
            f"{VDB_ENTITIES}: embedding_dim {vdb['embedding_dim']} != {EMBEDDING_DIM}. "
            f"The index was built with a different embedding model."
        )
    emb = _decode_matrix(vdb)                       # already L2-normalized by LightRAG
    names = [d.get("entity_name", "") for d in vdb["data"]]

    # Each chunk was inserted as its own LightRAG document keyed by OUR chunk_id, so a
    # LightRAG chunk's `full_doc_id` IS our chunk_id — this is the bridge from the graph's
    # source_id back to citable payloads. (535 LightRAG chunks -> 517 of ours: a few long
    # chunks were split, several mapping to the same payload.)
    with open(TEXT_CHUNKS, encoding="utf-8") as fh:
        text_chunks = json.load(fh)
    lr_to_chunk = {lr_id: v["full_doc_id"] for lr_id, v in text_chunks.items()
                   if v.get("full_doc_id")}

    return _PathStore(graph=graph, node_names=names, node_emb=emb,
                      node_row={n: i for i, n in enumerate(names)},
                      lr_to_chunk=lr_to_chunk)


_store: _PathStore | None = None
_store_lock = threading.Lock()
def _get_store() -> _PathStore:
    """Load the store once (double-checked locking: retrieval runs branches in parallel)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = _build_store()
    return _store


def _chunk_ids_of(attrs: dict) -> list[str]:
    """Our chunk_ids behind a node/edge, via its LightRAG source_id field."""
    store = _get_store()
    raw = attrs.get("source_id") or ""
    return [store.lr_to_chunk[p] for p in raw.split(SEP) if p in store.lr_to_chunk]


# ---------------------------------------------------------------------------
# STAGE 2: node retrieval
# ---------------------------------------------------------------------------
def retrieve_nodes(keywords: list[str], top_n: int = NODE_TOP_N) -> list[str]:
    """Match the keywords against the entity embeddings and return the top_n distinct CONCEPTS.

    Distinct concepts, not distinct nodes: the index holds up to seven spellings of «TAR», and
    without collapsing them by `canonical_key` they alone would fill a sixth of the budget,
    starving the concepts that actually discriminate the question (measured: the top-40 for an
    HBV question was mostly TAR variants). Keywords are abbreviation-expanded so «DTG» matches
    a node written «dolutegravir»."""
    store = _get_store()
    vecs = []
    for kw in keywords:
        try:
            v = np.asarray(get_embedding(expand_abbrevs(kw)), dtype=np.float32)
            vecs.append(v / (np.linalg.norm(v) or 1.0))
        except Exception:
            continue
    if not vecs:
        return []
    # Best score across keywords per node (a node is relevant if ANY keyword matches it),
    # rather than an average that would dilute a strong single-keyword hit.
    sims = (store.node_emb @ np.vstack(vecs).T).max(axis=1)
    out, seen = [], set()
    for i in np.argsort(-sims):
        name = store.node_names[i]
        key = canonical_key(name)
        if name in store.graph and key not in seen:
            seen.add(key)
            out.append(name)
            if len(out) >= top_n:
                break
    return out


# ---------------------------------------------------------------------------
# STAGE 3: path retrieval (flow-based pruning with distance awareness)
# ---------------------------------------------------------------------------
@dataclass
class _Path:
    nodes: list[str]
    reliability: float


def _flow_paths(start: str, targets: set[str], decay: float, threshold: float,
                max_hops: int) -> list[_Path]:
    """Propagate one unit of resource from `start` and collect the paths that reach another
    retrieved node (eq. 2-4 of the paper).

    Resource flowing into a node is its predecessor's resource, decayed by `decay` and split
    across that predecessor's neighbours — so distant and highly-connected (i.e. generic)
    nodes contribute little, which is exactly the redundancy PathRAG set out to prune. A
    branch dies as soon as its resource falls below `threshold`. Path reliability is the mean
    resource over its nodes, normalized by the number of edges (eq. 4)."""
    graph = _get_store().graph
    paths: list[_Path] = []
    # Each stack item is a partial path: (nodes so far, resource at the tip, resource sum).
    stack = [([start], 1.0, 1.0)]
    while stack:
        nodes, resource, total = stack.pop()
        if len(nodes) - 1 >= max_hops:
            continue
        tip = nodes[-1]
        degree = graph.degree(tip) or 1
        flow = decay * resource / degree
        if flow < threshold:                    # early stopping (eq. 3)
            continue
        # Deterministic order: strongest edge first, ties broken by node id.
        neighbours = sorted(
            graph[tip].items(),
            key=lambda kv: (-float(kv[1].get("weight", 1.0) or 1.0), kv[0]),
        )
        for nxt, _attrs in neighbours:
            if nxt in nodes:                    # simple paths only
                continue
            extended = nodes + [nxt]
            acc = total + flow
            if nxt in targets:
                paths.append(_Path(nodes=extended, reliability=acc / (len(extended) - 1)))
            stack.append((extended, flow, acc))
    return paths


def retrieve_paths(nodes: list[str], max_paths: int = MAX_PATHS, decay: float = DECAY,
                   threshold: float = PRUNE_THRESHOLD, max_hops: int = MAX_HOPS) -> list[_Path]:
    """Key relational paths among the retrieved nodes, most reliable first. Only the best path
    per node PAIR enters the pool (the paper's per-pair selection), so one densely connected
    pair cannot crowd out the rest of the query's structure."""
    targets = set(nodes)
    best_per_pair: dict[tuple[str, str], _Path] = {}
    for start in nodes:
        for path in _flow_paths(start, targets - {start}, decay, threshold, max_hops):
            pair = tuple(sorted((path.nodes[0], path.nodes[-1])))
            current = best_per_pair.get(pair)
            if current is None or path.reliability > current.reliability:
                best_per_pair[pair] = path
    return heapq.nlargest(max_paths, best_per_pair.values(), key=lambda p: p.reliability)


def _path_chunk_ids(paths: list[_Path]) -> list[str]:
    """Chunk ids behind the paths, ordered by path reliability (most reliable first). Both the
    nodes and the EDGES contribute: an edge's source chunk is the passage that stated the
    relation, which is often the one that actually answers a multi-hop question."""
    graph = _get_store().graph
    ordered: list[str] = []
    for path in paths:
        for i, node in enumerate(path.nodes):
            ordered.extend(_chunk_ids_of(graph.nodes[node]))
            if i + 1 < len(path.nodes):
                ordered.extend(_chunk_ids_of(graph.edges[node, path.nodes[i + 1]]))
    return ordered


def _describe(attrs: dict) -> str:
    """One readable description for a node/edge. LightRAG concatenates every extraction of the
    same item with its own separator, so we keep the first and trim it."""
    raw = (attrs.get("description") or "").split(SEP)[0]
    return " ".join(raw.split())[:DESC_CHARS]


def format_paths(paths: list[_Path]) -> str:
    """Render the paths as the non-citable concept map (Spanish: it goes into the generation
    prompt). Ordered by ASCENDING reliability so the most reliable path sits closest to the
    question at the end of the prompt — the paper's answer to the "lost in the middle" effect,
    adapted to our template (context first, question last)."""
    if not paths:
        return ""
    graph = _get_store().graph
    lines = []
    for path in sorted(paths, key=lambda p: p.reliability):
        parts = []
        for i, node in enumerate(path.nodes):
            desc = _describe(graph.nodes[node])
            parts.append(f"{node}: {desc}" if desc else node)
            if i + 1 < len(path.nodes):
                parts.append(f"  --[{_describe(graph.edges[node, path.nodes[i + 1]])}]-->")
        lines.append("- " + "\n  ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RETRIEVAL  (imported by the registry, the pipeline node and the evaluation)
# ---------------------------------------------------------------------------
def pathrag_search_with_paths(query: str, top_k: int = 8, node_top_n: int = NODE_TOP_N,
                              max_paths: int = MAX_PATHS, decay: float = DECAY,
                              rewritten_query: str | None = None) -> tuple[list, str]:
    """Full PathRAG retrieval: keywords -> nodes -> pruned paths -> chunks + concept map.

    Returns (payloads, concept_map). The payloads are ordinary citable chunks; the concept map
    is the textual form of the same paths, for the generator to reason over but never quote.
    An empty graph result is not an error — `house_tail`'s hybrid complement still answers."""
    keywords = extract_keywords(query)
    nodes = retrieve_nodes(keywords, top_n=node_top_n)
    paths = retrieve_paths(nodes, max_paths=max_paths, decay=decay) if nodes else []
    primary = map_chunk_ids_to_payloads(_path_chunk_ids(paths))[:PRIMARY_CHUNK_CAP]
    return house_tail(query, primary, rewritten_query, top_k=top_k), format_paths(paths)


def pathrag_search(query: str, top_k: int = 8, rewritten_query: str | None = None) -> list:
    """Chunk-only entry point, honouring the retrieval contract shared by every mode (this is
    what the evaluation A/B calls, so all modes are compared on chunks alone)."""
    payloads, _ = pathrag_search_with_paths(query, top_k=top_k, rewritten_query=rewritten_query)
    return payloads


if __name__ == "__main__":
    # Smoke: store integrity + the graph stages of a demo query. Only the final house_tail
    # needs Qdrant, so everything above it can be checked offline.
    store = _get_store()
    print(f"Store: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} "
          f"edges, embeddings {store.node_emb.shape}, "
          f"{len(store.lr_to_chunk)} LightRAG chunks mapped to our payloads.")

    demo = ("En un paciente con coinfección por VHB que inicia TAR, "
            "¿qué fármacos debe incluir el régimen?")
    keywords = extract_keywords(demo)
    print(f"\nQuery: {demo}\nKeywords: {keywords}")
    nodes = retrieve_nodes(keywords)
    print(f"Nodes ({len(nodes)}): {nodes[:10]}")
    paths = retrieve_paths(nodes)
    print(f"Paths ({len(paths)}), most reliable first:")
    for p in paths[:5]:
        print(f"  {p.reliability:.4f}  {' -> '.join(p.nodes)}")
    selected = map_chunk_ids_to_payloads(_path_chunk_ids(paths))
    payloads = selected[:PRIMARY_CHUNK_CAP]
    print(f"\nChunks selected by the paths: {len(selected)} -> keeping {len(payloads)}")
    for p in payloads:
        print(f"  {p['chunk_id']} | {p['source_file']} | {(p.get('heading') or '')[:60]}")
    print("\nConcept map (ascending reliability, non-citable):")
    print(format_paths(paths)[:1200])
