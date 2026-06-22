"""
main.py — Entry point: orchestrates the clinical HIV RAG with LangGraph + LangSmith.

Chains the pipeline functions (rag.py) and the evidence formatting (evidence.py) as
graph nodes so every step can be audited in LangSmith.

Flow:   question -> rephrase -> [retrieval mode] -> generate <-> validate -> evidence -> output
        - rephrase classifies the domain: if the question is unrelated to HIV, it stops
          here (direct message) and skips retrieve/rerank/generate/validate.
        - the retrieval step is one of three interchangeable modes (baseline / iterative /
          graph); they all feed the SAME generate -> validate -> evidence tail.
        - validate retries generate when not valid, up to MAX_ITER; if exhausted, safe exit.

Choosing the retrieval mode:
    - In code / CLI: the default is RETRIEVAL_MODE (env var, falls back to "graph").
      `python main.py iterative` runs that one shot for a quick manual test.
    - In LangGraph Studio: the graph exposes a `retrieval_mode` context field (a dropdown
      with baseline/iterative/graph). Pick it in the run config panel and each run takes
      the chosen path live — no code change, all three traced side by side.

Tracing (LangSmith):
    Enabled automatically ONLY if LANGSMITH_API_KEY is set in .env. Then each graph run
    and each OpenAI call (chat and embeddings) shows up at https://smith.langchain.com
    inside the LANGSMITH_PROJECT project. Without that key the graph works the same but
    sends no traces. The chosen retrieval mode is recorded on every run (in the graph
    state, as a run tag and in metadata) so the three architectures are filterable there.

Usage:
    python main.py             # interactive, uses RETRIEVAL_MODE (default "graph")
    python main.py iterative   # interactive, forces the iterative mode for this run
"""
import os
import sys

# The Windows console uses cp1252 by default and breaks accents/boxes. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from typing import Literal, TypedDict, cast

from dotenv import load_dotenv
load_dotenv()

# --- LangSmith: enable tracing only if there is an API key (otherwise stays out of the way) ---
TRACING = bool(os.environ.get("LANGSMITH_API_KEY"))
if TRACING:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "chatbot_vih")  # matches .env / CLAUDE.md

# --- RAG pipeline (retrieval/generation functions and clients) ---
import rag  # creates the OpenAI/Qdrant clients on import
from rag import (refine, retrieve_hybrid, rerank, retrieve_rerank, search, build_context,
                 validate, SYS_PROMPT, build_user_prompt, GENERATION_MODEL)
from agentic.iterative import iterative_search   # Track A (agentic, multi-hop) — collapsed API
# Track A primitives, reused to expose the iterative loop as visible graph nodes:
from agentic.iterative import _plan, _reflect, MAX_HOPS, PER_HOP
from evidence import format_answer

# --- Fine-grained tracing of OpenAI calls (tokens, latency) WITHOUT modifying rag.py ---
# rag's functions read `rag.client` at call time, so reassigning here its LangSmith-wrapped
# version is enough for retrieve() and the embeddings to be traced inside the graph run.
if TRACING:
    try:
        from langsmith.wrappers import wrap_openai
        rag.client = wrap_openai(rag.client)
    except Exception:
        pass

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


# ===========================================================================
# STRUCTURED GENERATION (LangChain) — replaces the raw response_format. Under the
# hood it uses the SAME OpenAI mechanism (Structured Outputs, strict json_schema),
# but with Pydantic validation and native LangSmith tracing. We return .model_dump()
# so the rest of the pipeline keeps receiving the same dict as before.
# ===========================================================================
class SourceUsed(BaseModel):
    ref: int
    quote: str


class ClinicalAnswer(BaseModel):
    sufficient_information: bool
    answer: str
    sources_used: list[SourceUsed]
    follow_up_questions: list[str]


_structured_llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0.2).with_structured_output(
    ClinicalAnswer, method="json_schema", strict=True
)

# Validation loop: max number of generations (initial + retries).
MAX_ITER = 2

# Retrieval strategy (Phase 4) — three interchangeable modes:
#   "baseline"  -> hybrid + reranker (single-shot, current system)
#   "iterative" -> Track A (agentic): self-ask / reflect-retrieve loop for multi-hop
#                  (single-hop questions fall back to baseline inside iterative_search)
#   "graph"     -> Track B: LightRAG entity-relation graph retrieval (needs the index
#                  built once: python -m graph.lightrag_track)
# All three feed the SAME generate -> validate -> evidence, so citations and the
# anti-hallucination validator are identical across strategies.
# F4 A/B decision: "graph" wins context_recall on the multi-hop set (0.98 vs 0.86
# iterative vs 0.84 baseline) at baseline-level latency (~11s), so it is the default.
#
# How the mode is chosen at runtime (resolved by _resolve_mode, in order):
#   1. LangGraph Studio / app.invoke(context=...): the `retrieval_mode` context field.
#   2. app.invoke(config={"configurable": {"retrieval_mode": ...}}): programmatic override.
#   3. RETRIEVAL_MODE below (env var RETRIEVAL_MODE, else "graph"): the default.
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "graph")
VALID_MODES = ("baseline", "iterative", "graph")


# Context schema exposed to LangGraph Studio: renders `retrieval_mode` as a dropdown
# (baseline/iterative/graph) in the run config panel, so the three architectures can be
# picked and traced live without touching the code. Optional (total=False): when unset we
# fall back to RETRIEVAL_MODE.
class ConfigSchema(TypedDict, total=False):
    retrieval_mode: Literal["baseline", "iterative", "graph"]

# --- User-facing messages (kept in Spanish: shown to the user) ---
MSG_NOT_VALIDATED = (
    "No he podido elaborar una respuesta suficientemente fundamentada en las guías "
    "para esta consulta. Te sugiero reformular la pregunta o revisar directamente las "
    "guías; es posible que la información no esté disponible en ellas."
)
MSG_VALIDATION_ERROR = (
    "No he podido validar la respuesta por un problema técnico (no se pudo contactar "
    "con el servicio de validación). Por seguridad no la muestro sin verificar; "
    "inténtalo de nuevo en unos momentos."
)
MSG_OUT_OF_DOMAIN = (
    "Soy un asistente centrado en las guías clínicas de VIH (GeSIDA) y solo puedo "
    "ayudarte con consultas sobre el manejo clínico del VIH. ¿Tienes alguna pregunta "
    "sobre ese tema?"
)


# ===========================================================================
# STATE
#   InputState: the only thing to pass to app.invoke() -> the question.
#   RAGState:   full internal state that flows through the graph. Keys are required
#               (each node fills them in) so that accessing state["..."] is safe for
#               the type checker. Nodes return only the keys they produce and LangGraph
#               merges them.
# ===========================================================================
class InputState(TypedDict):
    question: str            # user input


class RAGState(TypedDict):
    question: str            # user input (original; used for generation)
    retrieval_mode: str      # which retrieval architecture ran (set by the retrieval node)
    in_domain: bool          # is the question within the HIV domain? (classified in rephrase)
    search_query: str        # rewritten/normalized query for the retriever (rephrase)
    candidates: list         # retrieved payloads (hybrid, ~20) before reranking
    contexts: list           # final payloads after the reranker (top 5)
    chunk_index: dict        # {n: chunk} for source citation (build_context)
    formatted_context: str   # numbered context [1]/[2]… for the LLM
    answer: dict             # structured answer from the LLM (generate_answer)
    attempts: int            # number of generations done (for the validation loop)
    validation: dict         # validator verdict {is_valid, reason, ...}
    output: str              # final formatted text with sources (format_answer)


# Extra state for the DEDICATED expanded graphs (the teaching views). Subclassing keeps the
# base RAGState clean: these intermediate keys only exist in the iterative / graph graphs.
class IterativeState(RAGState, total=False):
    is_multihop: bool        # did the planner decide the question is multi-hop?
    pending: list            # queue of sub-queries still to retrieve (plan + reflect refills)
    pool: list               # payloads accumulated across hops (deduped), before final rerank
    hops: int                # retrieval rounds done so far (capped at MAX_HOPS)


class GraphState(RAGState, total=False):
    graph_payloads: list     # chunks from LightRAG entity/relation traversal
    hybrid_payloads: list    # chunks from our dense+BM25 hybrid complement
    merged: list             # the two sources merged & deduped, before final rerank


# ===========================================================================
# NODES — each one wraps a pipeline function.
#   They receive the full state and return ONLY the keys they produce.
# ===========================================================================
def node_rephrase(state: RAGState) -> dict:
    """question -> rewritten/normalized query + domain classification (single LLM call).
    If the question is unrelated to HIV, the pipeline is short-circuited (see
    route_domain): retrieve, rerank, generate and validate are skipped."""
    r = refine(state["question"])
    return {"search_query": r["query"], "in_domain": r["in_domain"]}


def node_out_of_domain(state: RAGState) -> dict:
    """Question outside the HIV domain: direct message, no retrieval or generation."""
    return {"output": MSG_OUT_OF_DOMAIN}


def _resolve_mode(config) -> str:
    """Pick the retrieval mode for this run, in order: Studio/context field ->
    config["configurable"] -> RETRIEVAL_MODE default. Unknown values fall back to the
    default, never crash."""
    mode = None
    try:  # Studio and app.invoke(context=...) deliver it via the runtime context.
        from langgraph.runtime import get_runtime
        mode = (get_runtime(ConfigSchema).context or {}).get("retrieval_mode")
    except Exception:
        pass
    if not mode:  # programmatic app.invoke(config={"configurable": {...}})
        mode = ((config or {}).get("configurable") or {}).get("retrieval_mode")
    return mode if mode in VALID_MODES else RETRIEVAL_MODE


def route_domain(state: RAGState, config) -> str:
    """Combined graph only — after rephrase: out of domain -> message and end; in domain ->
    the selected retrieval strategy (baseline single-shot, Track A iterative, Track B graph)."""
    if not state.get("in_domain", True):
        return "out_of_domain"
    return {
        "iterative": "iterative_retrieve",
        "graph": "graph_retrieve",
    }.get(_resolve_mode(config), "retrieve")


def route_in_domain(state: RAGState) -> str:
    """Dedicated graphs only — there is a single retrieval path, so the rephrase branch just
    decides in-domain vs out-of-domain. The "retrieve" key is mapped by the caller to that
    graph's actual retrieval entry node."""
    return "out_of_domain" if not state.get("in_domain", True) else "retrieve"


def node_retrieve(state: RAGState) -> dict:
    """search_query -> ~20 candidates via hybrid search (dense + BM25)."""
    candidates = retrieve_hybrid(state["search_query"], top_k=20, prefetch_limit=30)
    return {"candidates": candidates, "retrieval_mode": "baseline"}


def node_rerank(state: RAGState) -> dict:
    """candidates -> top 5 reordered by the cross-encoder + numbered context."""
    contexts = rerank(state["search_query"], state["candidates"], top_k=5)
    chunk_index, formatted_context = build_context(contexts)
    return {
        "contexts": contexts,
        "chunk_index": chunk_index,
        "formatted_context": formatted_context,
    }


def node_iterative_retrieve(state: RAGState) -> dict:
    """Track A: multi-hop retrieval (plan -> hop retrieve -> reflect) producing the same
    numbered-context shape the baseline rerank node yields, so generate/validate/evidence
    (and the literal-citation panel) work unchanged. Uses the ORIGINAL question; the
    iterative search does its own decomposition and normalization."""
    contexts = iterative_search(state["question"], top_k=8)
    chunk_index, formatted_context = build_context(contexts)
    return {
        "contexts": contexts,
        "chunk_index": chunk_index,
        "formatted_context": formatted_context,
        "retrieval_mode": "iterative",
    }


def node_graph_retrieve(state: RAGState) -> dict:
    """Track B: LightRAG graph retrieval. Selects source chunks via entity/relation
    traversal and maps them back to our payloads, yielding the same numbered-context shape
    as the other retrieval nodes so generate/validate/evidence are unchanged. Imported
    lazily so baseline/iterative runs do not load LightRAG."""
    from graph.lightrag_track import graph_search
    # Reuse the query rephrased by node_rephrase for the hybrid-vector complement (BM25
    # benefits from the normalized abbreviations); avoids a second rephrase call.
    contexts = graph_search(state["question"], top_k=8, hybrid_query=state.get("search_query"))
    chunk_index, formatted_context = build_context(contexts)
    return {
        "contexts": contexts,
        "chunk_index": chunk_index,
        "formatted_context": formatted_context,
        "retrieval_mode": "graph",
    }


# ---------------------------------------------------------------------------
# EXPANDED retrieval nodes — the SAME logic as iterative_search / graph_search above, but
# split into one node per real step so the dedicated graphs SHOW how each architecture works
# under the hood (plan/hop/reflect/rerank; traverse/hybrid/merge/rerank) instead of hiding it
# behind a single "retrieve" node. They reuse the exact same primitives, so behaviour is
# unchanged. Used only by the dedicated graphs (build_graph); the combined graph keeps the
# collapsed nodes above.
# ---------------------------------------------------------------------------

# --- Track A (iterative), expanded: plan -> hop(s) <-> reflect -> rerank ---
def _dedup_extend(pool: list, new: list) -> list:
    """Append payloads not already in pool (dedup by chunk_id, fallback text)."""
    seen = {(p.get("chunk_id") or p.get("text", "")) for p in pool}
    for p in new:
        key = p.get("chunk_id") or p.get("text", "")
        if key and key not in seen:
            seen.add(key)
            pool.append(p)
    return pool


def node_iter_plan(state: IterativeState) -> dict:
    """PLAN: decompose the question into normalized sub-queries (or mark it single-hop)."""
    p = _plan(state["question"])
    return {"is_multihop": p["is_multihop"], "pending": list(p["sub_queries"][:MAX_HOPS]),
            "pool": [], "hops": 0, "retrieval_mode": "iterative"}


def route_iter_plan(state: IterativeState) -> str:
    """Single-hop -> one baseline-style shot; multi-hop -> enter the hop/reflect loop."""
    return "hop" if state.get("is_multihop") else "single"


def node_iter_single(state: IterativeState) -> dict:
    """Single-hop fallback: the baseline one-shot (rephrase + hybrid + rerank)."""
    contexts = search(state["question"], top_k=8)
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context}


def node_iter_hop(state: IterativeState) -> dict:
    """HOP: take the next pending sub-query, retrieve+rerank it, accumulate into the pool."""
    pending = list(state.get("pending") or [])
    sq = pending.pop(0)
    pool = _dedup_extend(list(state.get("pool") or []), retrieve_rerank(sq, top_k=PER_HOP))
    return {"pending": pending, "pool": pool, "hops": state.get("hops", 0) + 1}


def route_iter_hop(state: IterativeState) -> str:
    """After a hop: stop if we hit MAX_HOPS; else drain remaining planned sub-queries; once
    the planned queue is empty, hand over to reflect (self-ask for what is still missing)."""
    if state.get("hops", 0) >= MAX_HOPS:
        return "rerank"
    return "hop" if state.get("pending") else "reflect"


def node_iter_reflect(state: IterativeState) -> dict:
    """REFLECT: judge whether the pooled evidence suffices; if not, queue the next sub-query."""
    pool_dict = {(p.get("chunk_id") or p.get("text", "")): p for p in (state.get("pool") or [])}
    r = _reflect(state["question"], pool_dict)
    pending = list(state.get("pending") or [])
    if not r["sufficient"] and r["next_query"]:
        pending.append(r["next_query"])
    return {"pending": pending}


def route_iter_reflect(state: IterativeState) -> str:
    """Loop back to hop if reflect asked for more (and budget remains); else go rerank."""
    if state.get("hops", 0) >= MAX_HOPS:
        return "rerank"
    return "hop" if state.get("pending") else "rerank"


def node_iter_rerank(state: IterativeState) -> dict:
    """Final precision pass: rerank the pooled union against the ORIGINAL question -> top 8."""
    contexts = rerank(state["question"], state.get("pool") or [], top_k=8)
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context}


# --- Track B (graph), expanded: traverse -> hybrid -> merge -> rerank ---
def node_graph_traverse(state: GraphState) -> dict:
    """TRAVERSE: LightRAG entity/relation traversal (this is where LightRAG's internal LLM
    extracts keywords) -> source chunks mapped to our payloads. Imported lazily so the other
    architectures never load LightRAG."""
    from graph.lightrag_track import graph_traverse
    return {"graph_payloads": graph_traverse(state["question"]), "retrieval_mode": "graph"}


def node_graph_hybrid(state: GraphState) -> dict:
    """HYBRID COMPLEMENT: our dense + BM25 search (reuses the rephrased query; BM25 helps with
    the guides' abbreviations) — replaces LightRAG's dropped dense-only chunk search."""
    hq = state.get("search_query") or state["question"]
    return {"hybrid_payloads": retrieve_hybrid(hq, top_k=10, prefetch_limit=30)}


def node_graph_merge(state: GraphState) -> dict:
    """MERGE: union the two chunk sources, deduped by chunk_id (graph chunks first)."""
    from graph.lightrag_track import _merge_dedup
    merged = _merge_dedup(state.get("graph_payloads") or [], state.get("hybrid_payloads") or [])
    return {"merged": merged}


def node_graph_rerank(state: GraphState) -> dict:
    """RERANK: cross-encoder over the merged union against the question -> top 8."""
    merged = state.get("merged") or []
    contexts = rerank(state["question"], merged, top_k=8) if merged else []
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context}


def node_generate(state: RAGState) -> dict:
    """context + question -> structured answer from the LLM. If a previous attempt was
    rejected by the validator, inject its feedback to correct it."""
    user = build_user_prompt(state["question"], state["formatted_context"])
    val = state.get("validation")
    if val and not val.get("is_valid", True):
        user += (
            f"\n\n    REINTENTO: tu respuesta anterior fue RECHAZADA por el validador. "
            f"Motivo: {val.get('reason', '')}. Corrige la respuesta para que cada afirmación "
            f"esté respaldada por el contexto y aborde la pregunta. Si el contexto no respalda "
            f"la respuesta, marca informacion_suficiente=false."
        )
    messages = [("system", SYS_PROMPT), ("human", user)]
    answer = cast(ClinicalAnswer, _structured_llm.invoke(messages))  # validated by Pydantic
    return {"answer": answer.model_dump(), "attempts": state.get("attempts", 0) + 1}


def node_validate(state: RAGState) -> dict:
    """Relevance + grounding judge over the answer. Marks valid / not valid."""
    verdict = validate(state["question"], state["answer"], state["formatted_context"])
    return {"validation": verdict}


def node_evidence(state: RAGState) -> dict:
    """answer + index -> final text with the sources and follow-up panel."""
    output = format_answer(state["answer"], state["chunk_index"])
    return {"output": output}


def node_fallback(state: RAGState) -> dict:
    """The answer is not shown. The message depends on the reason: technical error of
    the validator vs. no valid answer reached after the retries."""
    if state["validation"].get("error", False):
        return {"output": MSG_VALIDATION_ERROR}
    return {"output": MSG_NOT_VALIDATED}


def route_validation(state: RAGState) -> str:
    """After validating:
      - technical error of the validator -> fallback (do not retry; the judge is down).
      - valid -> format (evidence).
      - not valid and attempts left -> regenerate with feedback.
      - not valid and exhausted -> fallback (safe exit)."""
    v = state["validation"]
    if v.get("error", False):
        return "fallback"
    if v.get("is_valid", False):
        return "evidence"
    if state.get("attempts", 0) >= MAX_ITER:
        return "fallback"
    return "generate"


# ===========================================================================
# GRAPH ASSEMBLY
#   The HEAD (rephrase + domain guardrail) and the TAIL (generate <-> validate ->
#   evidence/fallback) are IDENTICAL for every architecture; only the retrieval section in
#   the middle changes. So we factor head/tail into helpers and let each retrieval mode plug
#   its own node(s). This is exactly why the pipeline is retrieval-agnostic: each retrieval
#   node honours the SAME state contract (it fills contexts / chunk_index / formatted_context),
#   and the tail reads only that contract — never anything mode-specific.
#
# Two ways to assemble it:
#   - build_graph(mode): a DEDICATED graph with only that mode's retrieval path. Cleanest to
#     read and to visualize in Studio (no dead branches). Exposed as app_baseline/_iterative/
#     _graph and selectable as separate graphs in the Studio graph dropdown.
#   - build_combined_graph(): all three paths in one graph; the active one is chosen at
#     runtime by route_domain via the `retrieval_mode` context field. Exposed as `app`.
#
# Shape:  START -> rephrase -┬- out of domain -> out_of_domain -> END
#                            └- in domain -> [retrieval] -> generate -> validate -┬-> evidence -> END
#                                                            ▲ (retry)  │          ├-> generate (loop)
#                                                            └──────────┘          └-> fallback -> END
# ===========================================================================
def _add_common(builder: StateGraph) -> None:
    """Add the head (rephrase, out_of_domain) and the tail (generate/validate/evidence/
    fallback) shared by every architecture. The retrieval section and the rephrase routing
    are added by the caller (they are the only mode-dependent parts)."""
    builder.add_node("rephrase", node_rephrase)
    builder.add_node("out_of_domain", node_out_of_domain)
    builder.add_node("generate", node_generate)
    builder.add_node("validate", node_validate)
    builder.add_node("evidence", node_evidence)
    builder.add_node("fallback", node_fallback)

    builder.add_edge(START, "rephrase")
    builder.add_edge("out_of_domain", END)
    builder.add_edge("generate", "validate")
    # Loop: validate -> evidence (valid) | generate (retry) | fallback (exhausted)
    builder.add_conditional_edges("validate", route_validation,
                                  {"evidence": "evidence", "generate": "generate",
                                   "fallback": "fallback"})
    builder.add_edge("evidence", END)
    builder.add_edge("fallback", END)


def _add_retrieval_collapsed(builder: StateGraph, mode: str) -> str:
    """COMBINED graph: add `mode`'s retrieval as a SINGLE node (its internals stay inside
    iterative_search / graph_search). Returns the entry node name. baseline is the exception
    — it is already two explicit nodes (retrieve + rerank)."""
    if mode == "baseline":
        builder.add_node("retrieve", node_retrieve)
        builder.add_node("rerank", node_rerank)
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "generate")
        return "retrieve"
    if mode == "iterative":
        builder.add_node("iterative_retrieve", node_iterative_retrieve)
        builder.add_edge("iterative_retrieve", "generate")
        return "iterative_retrieve"
    builder.add_node("graph_retrieve", node_graph_retrieve)
    builder.add_edge("graph_retrieve", "generate")
    return "graph_retrieve"


def _add_retrieval_expanded(builder: StateGraph, mode: str) -> str:
    """DEDICATED graph: add `mode`'s retrieval as its real sequence of nodes (the teaching
    view), wired into `generate`. Returns the entry node name."""
    if mode == "baseline":
        # Already explicit: hybrid retrieve -> cross-encoder rerank.
        builder.add_node("retrieve", node_retrieve)
        builder.add_node("rerank", node_rerank)
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "generate")
        return "retrieve"
    if mode == "iterative":
        # plan -> hop(s) <-> reflect -> rerank  (self-ask loop, capped at MAX_HOPS).
        builder.add_node("iter_plan", node_iter_plan)
        builder.add_node("iter_single", node_iter_single)
        builder.add_node("iter_hop", node_iter_hop)
        builder.add_node("iter_reflect", node_iter_reflect)
        builder.add_node("iter_rerank", node_iter_rerank)
        builder.add_conditional_edges("iter_plan", route_iter_plan,
                                      {"single": "iter_single", "hop": "iter_hop"})
        builder.add_edge("iter_single", "generate")
        builder.add_conditional_edges("iter_hop", route_iter_hop,
                                      {"hop": "iter_hop", "reflect": "iter_reflect",
                                       "rerank": "iter_rerank"})
        builder.add_conditional_edges("iter_reflect", route_iter_reflect,
                                      {"hop": "iter_hop", "rerank": "iter_rerank"})
        builder.add_edge("iter_rerank", "generate")
        return "iter_plan"
    # graph: traverse -> hybrid complement -> merge -> rerank.
    builder.add_node("graph_traverse", node_graph_traverse)
    builder.add_node("graph_hybrid", node_graph_hybrid)
    builder.add_node("graph_merge", node_graph_merge)
    builder.add_node("graph_rerank", node_graph_rerank)
    builder.add_edge("graph_traverse", "graph_hybrid")
    builder.add_edge("graph_hybrid", "graph_merge")
    builder.add_edge("graph_merge", "graph_rerank")
    builder.add_edge("graph_rerank", "generate")
    return "graph_traverse"


# Per-mode state schema for the dedicated graphs (extra intermediate keys; see classes above).
_STATE_SCHEMA = {"iterative": IterativeState, "graph": GraphState}


def build_graph(mode: str = "graph"):
    """Dedicated single-architecture graph: head + ONLY `mode`'s retrieval path (expanded
    into its real steps) + tail."""
    builder = StateGraph(_STATE_SCHEMA.get(mode, RAGState), input_schema=InputState)
    _add_common(builder)
    entry = _add_retrieval_expanded(builder, mode)
    # Domain guardrail: in -> the retrieval entry node; out -> direct message.
    builder.add_conditional_edges("rephrase", route_in_domain,
                                  {"retrieve": entry, "out_of_domain": "out_of_domain"})
    return builder.compile()


def build_combined_graph():
    """All three retrieval paths in one graph (each collapsed to a single node); route_domain
    picks the active one at runtime from the `retrieval_mode` context field (Studio dropdown)
    / config / RETRIEVAL_MODE."""
    builder = StateGraph(RAGState, input_schema=InputState, context_schema=ConfigSchema)
    _add_common(builder)
    for m in VALID_MODES:
        _add_retrieval_collapsed(builder, m)
    builder.add_conditional_edges("rephrase", route_domain,
                                  {"retrieve": "retrieve",
                                   "iterative_retrieve": "iterative_retrieve",
                                   "graph_retrieve": "graph_retrieve",
                                   "out_of_domain": "out_of_domain"})
    return builder.compile()


# Compiled graphs. The three dedicated ones (expanded, teaching views) are selectable as
# separate graphs in LangGraph Studio (see langgraph.json); `app` is the combined one with
# the live retrieval_mode dropdown and is what the CLI uses.
app_baseline  = build_graph("baseline")
app_iterative = build_graph("iterative")
app_graph     = build_graph("graph")
app = build_combined_graph()


def main_cli():
    # Optional first arg picks the retrieval mode for this run (e.g. `python main.py graph`).
    mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in VALID_MODES else RETRIEVAL_MODE
    question = input("¿Cuál es tú pregunta?: ")
    # context -> selects the mode; tags/metadata -> make the run filterable in LangSmith.
    result = app.invoke(
        {"question": question},
        context={"retrieval_mode": mode},
        config={"tags": [f"mode:{mode}"], "metadata": {"retrieval_mode": mode}},
    )
    print(result["output"])


if __name__ == "__main__":
    main_cli()
