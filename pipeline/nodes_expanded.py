"""Expanded retrieval nodes for the dedicated Studio graphs (teaching views).

Same logic as iterative_search / graph_search, but split into one node per real step so the
dedicated graphs SHOW how each architecture works (plan/hop/reflect/rerank; keywords/select/
hybrid/merge/rerank) instead of hiding it behind a single retrieve node. They reuse the exact
same primitives, so behaviour is unchanged. Used only by build_graph (the dedicated graphs);
the combined graph keeps the collapsed nodes in nodes.py.
"""
from langgraph.types import Send

from rag import retrieve_hybrid, rerank, build_context
from retrieval.baseline import retrieve_rerank
from retrieval.iterative import _plan, _reflect, MAX_HOPS, PER_HOP

from .state import IterativeState, GraphState


# --- Track A (iterative), expanded with a per-sub-question fan-out (Send) ---
#   iter_generate_subquestions ─┬─ (single-hop) ─ iter_single ─────────────────────────────┐
#                               └─ (multi-hop) ─ Send×N ▶ iter_retrieve_one ─ iter_reflect ─┬─ Send×1 ▶ (loop)
#                                                (parallel, one per sub-question)           └─ iter_rerank ─┘
# pool / hops accumulate via the state reducers.
def node_iter_generate_subquestions(state: IterativeState) -> dict:
    """Decompose the question into normalized sub-queries (or mark it single-hop). This node
    ONLY generates; retrieval happens later in iter_retrieve_one."""
    p = _plan(state["question"])
    return {"is_multihop": p["is_multihop"], "planned": list(p["sub_queries"][:MAX_HOPS]),
            "retrieval_mode": "iterative"}


def route_iter_generate_subquestions(state: IterativeState):
    """Single-hop -> one baseline shot. Multi-hop -> fan out one iter_retrieve_one per
    generated sub-question (parallel), each carrying just its own sub-query."""
    if not state.get("is_multihop"):
        return "iter_single"
    return [Send("iter_retrieve_one", {"subquery": sq}) for sq in (state.get("planned") or [])]


def node_iter_single(state: IterativeState) -> dict:
    """Single-hop fallback: one baseline shot (hybrid + rerank) reusing the query that
    node_rephrase already rewrote (no duplicate rephrase LLM call)."""
    contexts = retrieve_rerank(state.get("search_query") or state["question"], top_k=8)
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context}


def node_iter_retrieve_one(state: IterativeState) -> dict:
    """Retrieve + rerank ONE sub-question (one fan-out branch). The reducers merge its hits
    into the shared pool and add 1 to the hop count. Executed N times in the trace."""
    hits = retrieve_rerank(state["subquery"], top_k=PER_HOP)
    return {"pool": hits, "hops": 1}


def node_iter_reflect(state: IterativeState) -> dict:
    """Self-ask whether the pooled evidence covers the question. Emits next_query="" when it
    is enough or the hop budget is spent; otherwise the follow-up sub-question for another
    round (skips the LLM call once MAX_HOPS is reached)."""
    if state.get("hops", 0) >= MAX_HOPS:
        return {"next_query": ""}
    pool_dict = {(p.get("chunk_id") or p.get("text", "")): p for p in (state.get("pool") or [])}
    r = _reflect(state["question"], pool_dict)
    return {"next_query": r["next_query"] if (not r["sufficient"] and r["next_query"]) else ""}


def route_iter_reflect(state: IterativeState):
    """A follow-up was requested -> fan out one more iter_retrieve_one (loop); else -> rerank."""
    return [Send("iter_retrieve_one", {"subquery": state["next_query"]})] \
        if state.get("next_query") else "iter_rerank"


def node_iter_rerank(state: IterativeState) -> dict:
    """Final precision pass: rerank the pooled union against the ORIGINAL question."""
    contexts = rerank(state["question"], state.get("pool") or [], top_k=8)
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context}


# --- Track B (graph), expanded: keywords -> select -> hybrid -> merge -> rerank ---
#   rephrase ─┬─ graph_keywords (LLM) ─ graph_select (vector+graph) ─┬─ graph_merge ─ graph_rerank
#             └─ graph_hybrid (dense+BM25) ───────────────────────────┘   (defer=True)
# The hybrid branch only needs search_query, so it runs in PARALLEL with the keyword->select
# chain. graph_merge uses defer=True so it fires ONCE, after both (uneven-length) branches.
def node_graph_keywords(state: GraphState) -> dict:
    """LightRAG's LLM extracts high-level keywords (-> RELATIONS) and low-level keywords
    (-> ENTITIES) from the question."""
    from retrieval.graph import graph_extract_keywords
    kw = graph_extract_keywords(state["question"])
    return {"graph_hl": kw["hl_keywords"], "graph_ll": kw["ll_keywords"],
            "retrieval_mode": "graph"}


def node_graph_select(state: GraphState) -> dict:
    """Vector search + graph walk (NO LLM): cosine of the low-level keywords vs the ENTITY
    store and high-level vs the RELATION store, walk the graph, gather the source chunks. Also
    surfaces the selected entities/relations so the traversal is visible in the state."""
    from retrieval.graph import graph_select
    res = graph_select(state["question"], state.get("graph_hl") or [], state.get("graph_ll") or [])
    return {"graph_payloads": res["payloads"], "graph_entities": res["entities"],
            "graph_relationships": res["relationships"]}


def node_graph_hybrid(state: GraphState) -> dict:
    """Hybrid complement: our dense + BM25 search (reuses the rephrased query) — replaces
    LightRAG's dropped dense-only chunk search."""
    hq = state.get("search_query") or state["question"]
    return {"hybrid_payloads": retrieve_hybrid(hq, top_k=10, prefetch_limit=30)}


def node_graph_merge(state: GraphState) -> dict:
    """Union the two chunk sources, deduped by chunk_id (graph chunks first)."""
    from retrieval.graph import _merge_dedup
    merged = _merge_dedup(state.get("graph_payloads") or [], state.get("hybrid_payloads") or [])
    return {"merged": merged}


def node_graph_rerank(state: GraphState) -> dict:
    """Cross-encoder over the merged union against the question -> top 8."""
    merged = state.get("merged") or []
    contexts = rerank(state["question"], merged, top_k=8) if merged else []
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context}
