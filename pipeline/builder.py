"""Graph assembly.

The head (rephrase + domain guardrail) and the tail (generate <-> validate -> evidence/
fallback) are identical for every architecture; only the retrieval section changes. So we
factor head/tail into `_add_common` and let each mode plug its own retrieval node(s).

    START -> rephrase ─┬─ out of domain -> out_of_domain -> END
                       └─ in domain -> [retrieval] -> assess_context -> re_retrieve -> generate
                                       generate -> validate ─┬─ evidence -> refine_offer
                                                             ├─ refocus_retrieve -> generate ↺
                                                             └─ fallback -> END
                       refine_offer (interrupt, OPTIONAL) ─┬─ declined -> END
                                                           └─ answered -> re_retrieve ↺

Two assemblies:
  - build_graph(mode): a DEDICATED graph with only that mode's retrieval path, EXPANDED into
    its real steps (cleanest to visualize in Studio). Exposed as app_baseline/_iterative/_graph.
    Only the modes with an expanded breakdown have one; the rest are reached through `app`.
  - build_combined_graph(): every path in one graph, each COLLAPSED to a single node; the
    active one is chosen at runtime via the `retrieval_mode` context field. Exposed as `app`.

Both take an optional `checkpointer`, which is what makes `clarify`'s `interrupt()` RESUMABLE.
The LangGraph platform (Studio / `langgraph dev`) injects its own, so the graphs it imports are
compiled without one; any plain embedding — the CLI, a future web app — MUST pass a checkpointer
or the run pauses with no way to resume it (invoke returns `__interrupt__` and no `output`).
"""
from langgraph.graph import StateGraph, START, END

from .config import VALID_MODES, ConfigSchema
from .state import InputState, RAGState, IterativeState, GraphState
from . import nodes as N
from . import nodes_expanded as X


def _add_common(builder: StateGraph) -> None:
    """Add the head and the tail shared by every architecture. The retrieval section and the
    rephrase routing are added by the caller (the only mode-dependent parts)."""
    builder.add_node("rephrase", N.node_rephrase)
    builder.add_node("out_of_domain", N.node_out_of_domain)
    builder.add_node("assess_context", N.node_assess_context)
    builder.add_node("refine_offer", N.node_refine_offer)
    builder.add_node("re_retrieve", N.node_re_retrieve)
    builder.add_node("generate", N.node_generate)
    builder.add_node("validate", N.node_validate)
    builder.add_node("refocus_retrieve", N.node_refocus_retrieve)
    builder.add_node("evidence", N.node_evidence)
    builder.add_node("fallback", N.node_fallback)

    builder.add_edge(START, "rephrase")
    builder.add_edge("out_of_domain", END)
    # Clarification, ANSWER-FIRST: assess only labels the dimensions the doctor has not pinned
    # down (they reach generate as an explicit "unknown" block, so the answer presents the
    # branches instead of assuming one) and the run goes straight on to answer. The refinement
    # is offered AFTER `evidence`, with the answer already on screen; only if the doctor takes
    # it up does the run loop back through re_retrieve (which folds the new facts into the
    # query, so the conditional passages are there to cite) and generate again.
    builder.add_edge("assess_context", "re_retrieve")
    builder.add_edge("re_retrieve", "generate")
    builder.add_edge("generate", "validate")
    # A grounding rejection is usually a RETRIEVAL miss, so the retry does not loop straight
    # back to generate: refocus_retrieve first chases the validator's unsupported claims, and
    # only then is the answer regenerated over evidence that may actually support it.
    builder.add_conditional_edges("validate", N.route_validation,
                                  {"evidence": "evidence", "fallback": "fallback",
                                   "refocus_retrieve": "refocus_retrieve"})
    builder.add_edge("refocus_retrieve", "generate")
    builder.add_edge("evidence", "refine_offer")
    builder.add_conditional_edges("refine_offer", N.route_refinement,
                                  {"re_retrieve": "re_retrieve", "end": END})
    builder.add_edge("fallback", END)


def _add_retrieval_collapsed(builder: StateGraph, mode: str) -> None:
    """COMBINED graph: add `mode`'s retrieval as a SINGLE node wired into assess_context
    (baseline is the exception — already two explicit nodes: retrieve + rerank)."""
    if mode == "baseline":
        builder.add_node("retrieve", N.node_retrieve)
        builder.add_node("rerank", N.node_rerank)
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "assess_context")
        return
    entry = N.retrieval_entry(mode)
    builder.add_node(entry, N.RETRIEVAL_NODES[mode])
    builder.add_edge(entry, "assess_context")


# Modes with a hand-built "teaching" breakdown for Studio. A mode is added here once it has
# earned the extra surface (graph did, after winning the Phase-4 A/B); until then it is
# reachable through the combined graph, collapsed to one node.
EXPANDED_MODES = ("baseline", "iterative", "graph")


def _add_retrieval_expanded(builder: StateGraph, mode: str):
    """DEDICATED graph: add `mode`'s retrieval as its real sequence of nodes, wired into
    assess_context. Returns the entry node name (a list for graph, which fans out to two)."""
    if mode == "baseline":
        builder.add_node("retrieve", N.node_retrieve)
        builder.add_node("rerank", N.node_rerank)
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "assess_context")
        return "retrieve"
    if mode == "iterative":
        builder.add_node("iter_generate_subquestions", X.node_iter_generate_subquestions)
        builder.add_node("iter_single", X.node_iter_single)
        builder.add_node("iter_retrieve_one", X.node_iter_retrieve_one)
        builder.add_node("iter_reflect", X.node_iter_reflect)
        builder.add_node("iter_rerank", X.node_iter_rerank)
        # route functions return Send objects, so we list the reachable targets explicitly.
        builder.add_conditional_edges("iter_generate_subquestions",
                                      X.route_iter_generate_subquestions,
                                      ["iter_single", "iter_retrieve_one"])
        builder.add_edge("iter_single", "assess_context")
        builder.add_edge("iter_retrieve_one", "iter_reflect")   # branches converge on reflect
        builder.add_conditional_edges("iter_reflect", X.route_iter_reflect,
                                      ["iter_retrieve_one", "iter_rerank"])
        builder.add_edge("iter_rerank", "assess_context")
        return "iter_generate_subquestions"
    # graph: two parallel retrieval branches from rephrase, merged then reranked.
    builder.add_node("graph_keywords", X.node_graph_keywords)
    builder.add_node("graph_select", X.node_graph_select)
    builder.add_node("graph_hybrid", X.node_graph_hybrid)
    builder.add_node("graph_merge", X.node_graph_merge, defer=True)
    builder.add_node("graph_rerank", X.node_graph_rerank)
    builder.add_edge("graph_keywords", "graph_select")
    builder.add_edge("graph_select", "graph_merge")
    builder.add_edge("graph_hybrid", "graph_merge")
    builder.add_edge("graph_merge", "graph_rerank")
    builder.add_edge("graph_rerank", "assess_context")
    return ["graph_keywords", "graph_hybrid"]   # rephrase fans out to BOTH entries


_STATE_SCHEMA = {"iterative": IterativeState, "graph": GraphState}


def build_graph(mode: str = "graph", checkpointer=None):
    """Dedicated single-architecture graph: head + only `mode`'s retrieval path (expanded
    into its real steps) + tail. Restricted to the modes with an expanded breakdown."""
    if mode not in EXPANDED_MODES:
        raise ValueError(
            f"{mode!r} has no expanded graph (available: {', '.join(EXPANDED_MODES)}). "
            f"Run it through the combined graph with retrieval_mode={mode!r}."
        )
    builder = StateGraph(_STATE_SCHEMA.get(mode, RAGState), input_schema=InputState)
    _add_common(builder)
    entry = _add_retrieval_expanded(builder, mode)
    entries = entry if isinstance(entry, list) else [entry]
    builder.add_conditional_edges("rephrase", N.make_route_in_domain(entries),
                                  entries + ["out_of_domain"])
    return builder.compile(checkpointer=checkpointer)


def build_combined_graph(checkpointer=None):
    """EVERY retrieval path in one graph (each collapsed to a single node); route_domain picks
    the active one at runtime from the `retrieval_mode` context field / config /
    RETRIEVAL_MODE. This is the graph the CLI uses. Both the nodes and the routing targets
    come from the mode catalogue, so registering a mode is enough to make it selectable."""
    builder = StateGraph(RAGState, input_schema=InputState, context_schema=ConfigSchema)
    _add_common(builder)
    for m in VALID_MODES:
        _add_retrieval_collapsed(builder, m)
    targets = {N.retrieval_entry(m): N.retrieval_entry(m) for m in VALID_MODES}
    builder.add_conditional_edges("rephrase", N.route_domain,
                                  {**targets, "out_of_domain": "out_of_domain"})
    return builder.compile(checkpointer=checkpointer)
