"""Graph nodes and routing for the combined pipeline.

Each node receives the full state and returns ONLY the keys it produces (LangGraph merges
them). The head (rephrase + domain guardrail) and the tail (assess -> generate <-> validate
-> evidence/fallback) are shared by every retrieval mode; only the retrieval node in between
changes. The expanded "teaching" nodes for the dedicated Studio graphs live in nodes_expanded.
"""
from typing import cast

from langgraph.types import interrupt

from rag import (refine, retrieve_hybrid, rerank, build_context, validate, assess,
                 SYS_PROMPT, build_user_prompt)
from retrieval import merge_dedup
from retrieval.registry import MODES
from evidence import format_answer

from .config import (MAX_ITER, CLARIFY_MAX_ROUNDS, CLARIFY_QUESTIONS_PER_ROUND,
                     RETRIEVAL_MODE, VALID_MODES, ConfigSchema,
                     MSG_NOT_VALIDATED, MSG_VALIDATION_ERROR, MSG_OUT_OF_DOMAIN)
from .state import RAGState
from .generation import ClinicalAnswer, structured_llm


# --- Head: rephrase + domain guardrail -------------------------------------
def node_rephrase(state: RAGState) -> dict:
    """question -> rewritten/normalized query + domain classification (single LLM call), plus
    the cheap half of the clarification step (seeds clinical_facts / candidate_modifiers).

    Runs on EVERY question, so it is also the RESET point for every per-question field. A
    thread outlives the question (Studio, and now the CLI, keep one across turns), so anything
    not reset here leaks into the NEXT question. Two loops depend on this: the clarification
    one (clarify_rounds / asked_questions / clinical_facts — a spent budget or the previous
    patient's data) and the validation one (attempts / validation — a carried-over `attempts`
    silently costs the next question its retry, and a carried-over rejected `validation` makes
    node_generate open with "your previous answer was REJECTED" on a first attempt)."""
    r = refine(state["question"])
    return {"search_query": r["query"], "in_domain": r["in_domain"],
            "clinical_facts": r.get("known_facts", {}),
            "candidate_modifiers": r.get("candidate_modifiers", []),
            "pending_clarifications": [], "clarify_rounds": 0, "assessment": {},
            "asked_questions": [], "concept_map": "",
            "attempts": 0, "validation": {}, "refocus_query": ""}


def node_out_of_domain(state: RAGState) -> dict:
    """Question outside the HIV domain: direct message, no retrieval or generation."""
    return {"output": MSG_OUT_OF_DOMAIN}


def _resolve_mode(config) -> str:
    """Pick the retrieval mode for this run: Studio/context field -> config["configurable"]
    -> RETRIEVAL_MODE default. Unknown values fall back to the default, never crash."""
    mode = None
    try:  # Studio and app.invoke(context=...) deliver it via the runtime context.
        from langgraph.runtime import get_runtime
        mode = (get_runtime(ConfigSchema).context or {}).get("retrieval_mode")
    except Exception:
        pass
    if not mode:  # programmatic app.invoke(config={"configurable": {...}})
        mode = ((config or {}).get("configurable") or {}).get("retrieval_mode")
    return mode if mode in VALID_MODES else RETRIEVAL_MODE


def retrieval_entry(mode: str) -> str:
    """Name of the node that runs `mode`'s retrieval in the combined graph. Baseline is the
    one mode split into two nodes (retrieve -> rerank), so it names the first of them."""
    return "retrieve" if mode == "baseline" else f"{mode}_retrieve"


def route_domain(state: RAGState, config) -> str:
    """Combined graph: after rephrase, route out-of-domain to a message, in-domain to the
    selected retrieval strategy."""
    if not state.get("in_domain", True):
        return "out_of_domain"
    return retrieval_entry(_resolve_mode(config))


def make_route_in_domain(entries: list):
    """Dedicated graphs: route out-of-domain to a message, in-domain to that graph's retrieval
    entry node(s) (a single node, or both parallel entries for the graph mode)."""
    def route(state: RAGState):
        return "out_of_domain" if not state.get("in_domain", True) else entries
    return route


# --- Collapsed retrieval nodes (combined graph) ----------------------------
def node_retrieve(state: RAGState) -> dict:
    """search_query -> ~20 candidates via hybrid search (dense + BM25)."""
    candidates = retrieve_hybrid(state["search_query"], top_k=20, prefetch_limit=30)
    return {"candidates": candidates, "retrieval_mode": "baseline"}


def node_rerank(state: RAGState) -> dict:
    """candidates -> top 5 reordered by the cross-encoder + numbered context."""
    contexts = rerank(state["search_query"], state["candidates"], top_k=5)
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context}


def _retrieve_with(mode: str, question: str, rewritten_query: str | None) -> tuple[list, str]:
    """Run `mode`'s retrieval, returning (payloads, concept_map).

    Every mode is called the same way — the ORIGINAL question plus the already-rephrased query
    so it need not rephrase again — and the registry loads it lazily, so a baseline run never
    imports LightRAG or a graph store. A mode that also produces a non-citable concept map
    (PathRAG's relational paths) returns it here; the others return "" and the prompt is
    unchanged."""
    m = MODES[mode]
    with_map = m.search_with_concept_map()
    if with_map is not None:
        return with_map(question, top_k=8, rewritten_query=rewritten_query)
    return m.search()(question, top_k=8, rewritten_query=rewritten_query), ""


def _make_retrieval_node(mode: str):
    """Build the collapsed retrieval node for `mode` (combined graph). One factory instead of
    one hand-written node per mode: they differ only in which retriever they call."""
    def node(state: RAGState) -> dict:
        contexts, concept_map = _retrieve_with(mode, state["question"],
                                               state.get("search_query"))
        chunk_index, formatted_context = build_context(contexts)
        return {"contexts": contexts, "chunk_index": chunk_index,
                "formatted_context": formatted_context, "concept_map": concept_map,
                "retrieval_mode": mode}
    node.__name__ = f"node_{mode}_retrieve"
    return node


# Collapsed node per non-baseline mode (baseline keeps its explicit retrieve -> rerank pair).
RETRIEVAL_NODES = {mode: _make_retrieval_node(mode)
                   for mode in VALID_MODES if mode != "baseline"}


# --- Clarification gate (between retrieval and generate) --------------------
def node_assess_context(state: RAGState) -> dict:
    """Evidence-grounded half of the clarification: decide whether the answer hinges on a
    clinical modifier the guides branch on (gestation, renal function, coinfection…) that the
    doctor has not provided, and if so propose questions. Once the budget is spent it stops
    asking. The datum is never cited — it only steers the branch and re-triggers retrieval."""
    if state.get("clarify_rounds", 0) >= CLARIFY_MAX_ROUNDS:
        return {"pending_clarifications": []}   # budget spent: do not ask again
    a = assess(state["question"], state["formatted_context"],
               known_facts=state.get("clinical_facts"),
               candidate_modifiers=state.get("candidate_modifiers"),
               asked_questions=state.get("asked_questions"),
               max_questions=CLARIFY_QUESTIONS_PER_ROUND)
    return {"pending_clarifications": a["questions"] if a["needs_clarification"] else [],
            "assessment": {k: a.get(k) for k in
                           ("clinically_relevant", "branches_on", "already_covered")}}


def node_clarify(state: RAGState) -> dict:
    """Pause the run (interrupt) and ask the doctor the pending questions. On resume, fold the
    answers into clinical_facts and bump the round counter (by hand — these fields carry no
    reducer), then return to assess_context. Retrieval is not re-run here; the loop stays on
    the initial context and the single re_retrieve happens on exit."""
    pending = state.get("pending_clarifications", [])
    answers = interrupt({"questions": pending})
    if isinstance(answers, dict):
        new_facts = answers                        # structured answer: merge as-is
    else:
        text = str(answers).strip()
        # Key a plain answer by the QUESTION asked (not a fixed key), else each round would
        # overwrite the previous one and assess would re-ask.
        new_facts = ({pending[0]: text} if (len(pending) == 1 and text)
                     else ({"respuesta_medico": text} if text else {}))
    merged = {**(state.get("clinical_facts") or {}), **new_facts}
    asked = list(state.get("asked_questions") or []) + list(pending)
    return {"clinical_facts": merged, "clarify_rounds": state.get("clarify_rounds", 0) + 1,
            "pending_clarifications": [], "asked_questions": asked}


def route_assess(state: RAGState) -> str:
    """Ask for clarification only if assess flagged missing data AND the budget is not spent."""
    if state.get("pending_clarifications") and state.get("clarify_rounds", 0) < CLARIFY_MAX_ROUNDS:
        return "clarify"
    return "generate"


def _retrieve_for_mode(mode: str, question: str, query: str) -> tuple[list, str]:
    """Run `mode`'s retrieval over an already-enriched query, returning (payloads, concept_map).

    The single place that knows baseline is the mode without one search function (the graph
    splits it into retrieve -> rerank), so every node that re-runs retrieval dispatches through
    here instead of repeating the special case."""
    if mode == "baseline":
        return rerank(query, retrieve_hybrid(query, top_k=20, prefetch_limit=30), top_k=5), ""
    return _retrieve_with(mode, question, query)


def _facts_phrase(clinical_facts: dict | None) -> str:
    """Compact inline rendering of the patient data, to enrich the retrieval query."""
    facts = {k: v for k, v in (clinical_facts or {}).items() if k}
    return "; ".join(f"{k}: {v}" if v else k for k, v in facts.items())


def node_re_retrieve(state: RAGState, config) -> dict:
    """Runs ONCE, right before generate, after the clarification loop gathered all the patient
    data. Re-runs retrieval with those facts folded into the query so the doctor's answers PULL
    the conditional passages (e.g. the HBV-coinfection or first-trimester branch) for generate
    to cite. Dispatches on the mode that already ran; no-ops unless clarification ADDED facts —
    facts that came in the question itself were already in the initial retrieval query."""
    facts = _facts_phrase(state.get("clinical_facts"))
    if not facts or not state.get("clarify_rounds"):
        return {}
    mode = state.get("retrieval_mode") or _resolve_mode(config)
    patient = f"Contexto del paciente: {facts}"
    enriched = f"{state.get('search_query') or state['question']}\n{patient}"
    contexts, concept_map = _retrieve_for_mode(mode, f"{state['question']}\n{patient}", enriched)
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context, "concept_map": concept_map}


# --- Tail: generate <-> validate -> evidence/fallback ----------------------
def node_generate(state: RAGState) -> dict:
    """context + question -> structured answer. Two non-citable blocks may accompany the
    numbered context: the clarified patient data (clinical_facts), which selects the applicable
    branch of the guides, and the retrieval mode's concept map, which shows how the concepts
    connect. Neither can be cited — `validate` still requires every claim to be grounded in the
    chunks. On a validator rejection, its feedback is injected so the retry can correct it."""
    user = build_user_prompt(state["question"], state["formatted_context"],
                             clinical_facts=state.get("clinical_facts"),
                             concept_map=state.get("concept_map"))
    val = state.get("validation")
    if val and not val.get("is_valid", True):
        claims = "".join(f"\n      - {c}" for c in val.get("unsupported_claims") or [])
        user += (
            f"\n\n    REINTENTO: tu respuesta anterior fue RECHAZADA por el validador. "
            f"Motivo: {val.get('reason', '')}."
            + (f" Afirmaciones sin respaldo:{claims}" if claims else "")
            + f"\n    Se ha vuelto a buscar evidencia sobre esos puntos, así que el CONTEXTO "
              f"de arriba puede haber cambiado: reléelo antes de responder. Corrige la "
              f"respuesta para que cada afirmación esté respaldada por el contexto y aborde la "
              f"pregunta. Si el contexto sigue sin respaldarla, marca "
              f"informacion_suficiente=false."
        )
    messages = [("system", SYS_PROMPT), ("human", user)]
    answer = cast(ClinicalAnswer, structured_llm.invoke(messages))
    return {"answer": answer.model_dump(), "attempts": state.get("attempts", 0) + 1}


def node_validate(state: RAGState) -> dict:
    """Relevance + grounding judge over the answer. Marks valid / not valid."""
    verdict = validate(state["question"], state["answer"], state["formatted_context"])
    return {"validation": verdict}


def node_refocus_retrieve(state: RAGState, config) -> dict:
    """The answer was rejected for lack of grounding: go back for EVIDENCE, not for a rewording.

    The usual cause of a grounding rejection is a retrieval miss, and regenerating over the
    same chunks cannot fix that — the retry would only rephrase an unsupported claim until the
    budget runs out and the doctor gets `MSG_NOT_VALIDATED`. So the validator's
    `unsupported_claims`, which until now were only injected as prose feedback, become the
    retrieval query: the pipeline chases exactly what it could not back.

    The new pass is MERGED with the context already in hand rather than replacing it, so the
    claims that WERE grounded keep their support, and reranked back to the same size. With no
    claims to chase this is a no-op and generate simply retries as before."""
    claims = [c.strip() for c in (state.get("validation") or {}).get("unsupported_claims") or []
              if isinstance(c, str) and c.strip()]
    if not claims:
        return {}
    mode = state.get("retrieval_mode") or _resolve_mode(config)
    focus = " ".join(claims)
    previous = state.get("contexts") or []
    found, concept_map = _retrieve_for_mode(
        mode, f"{state['question']}\n{focus}",
        f"{state.get('search_query') or state['question']}\n{focus}")
    contexts = rerank(state["question"], merge_dedup(found, previous),
                      top_k=len(previous) or 8)
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context, "refocus_query": focus,
            "concept_map": concept_map or state.get("concept_map", "")}


def node_evidence(state: RAGState) -> dict:
    """answer + index -> final text with the sources and follow-up panel."""
    return {"output": format_answer(state["answer"], state["chunk_index"])}


def node_fallback(state: RAGState) -> dict:
    """The answer is not shown: technical validator error vs. no valid answer after retries."""
    if state["validation"].get("error", False):
        return {"output": MSG_VALIDATION_ERROR}
    return {"output": MSG_NOT_VALIDATED}


def route_validation(state: RAGState) -> str:
    """valid -> evidence; not valid with attempts left -> chase the missing evidence, then
    regenerate; exhausted or technical validator error -> fallback (never 'fail open')."""
    v = state["validation"]
    if v.get("error", False):
        return "fallback"
    if v.get("is_valid", False):
        return "evidence"
    if state.get("attempts", 0) >= MAX_ITER:
        return "fallback"
    return "refocus_retrieve"
