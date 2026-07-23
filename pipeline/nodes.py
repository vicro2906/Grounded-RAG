"""Graph nodes and routing for the combined pipeline.

Each node receives the full state and returns ONLY the keys it produces (LangGraph merges
them). The head (rephrase + domain guardrail) and the tail (assess -> generate <-> validate
-> evidence/fallback) are shared by every retrieval mode; only the retrieval node in between
changes. The expanded "teaching" nodes for the dedicated Studio graphs live in nodes_expanded.
"""
import functools
from typing import cast

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from rag import (refine, retrieve_hybrid, rerank, build_context, validate, assess,
                 SYS_PROMPT, build_user_prompt)
from retrieval import merge_dedup
from retrieval.registry import MODES
from evidence import format_answer

from .config import (MAX_ITER, CLARIFY_MAX_ROUNDS, CLARIFY_QUESTIONS_PER_ROUND,
                     RETRIEVAL_MODE, VALID_MODES, ConfigSchema,
                     MSG_NOT_VALIDATED, MSG_VALIDATION_ERROR, MSG_OUT_OF_DOMAIN,
                     MSG_TECHNICAL_ERROR, STEP_FORMATTING, STEP_GENERATION, STEP_RETRIEVAL)
from .state import RAGState
from .generation import ClinicalAnswer, structured_llm


# --- Technical failures: a message, never a traceback ----------------------
def guarded(step: str):
    """Wrap a node so an unreachable service becomes a MESSAGE instead of a crash.

    Without this, an OpenAI outage or a Qdrant timeout propagates out of `invoke` and the
    doctor gets a Python traceback. The message matters as much as the catch: a technical
    failure must never be dressed up as a clinical result ("no está en las guías"), because
    that is a statement about the guidelines that could change a decision.

    `GraphBubbleUp` is re-raised on purpose: LangGraph signals its own control flow with
    exceptions, and `interrupt()` raises one of them. A bare `except Exception` here would
    swallow the refinement pause and turn it into a fake outage.
    """
    def decorator(fn):
        # functools.wraps keeps the wrapped signature visible, which LangGraph inspects to
        # decide whether to hand the node a `config` — so the wrapper has to forward keyword
        # arguments too, not just positional ones.
        @functools.wraps(fn)
        def node(state: RAGState, *args, **kwargs):
            try:
                return fn(state, *args, **kwargs)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                return {"technical_error": step, "technical_detail": f"{type(exc).__name__}: {exc}"}
        return node
    return decorator


def route_on_error(ok: str):
    """Conditional edge for every fallible step: on a technical failure jump straight to the
    message, otherwise carry on to `ok`."""
    def route(state: RAGState) -> str:
        return "technical_error" if state.get("technical_error") else ok
    return route


def node_technical_error(state: RAGState) -> dict:
    """Terminal node for a step that could not run. Names the step so a persistent failure is
    reportable, without leaking the exception."""
    return {"output": MSG_TECHNICAL_ERROR.format(step=state.get("technical_error", ""))}


# --- Head: rephrase + domain guardrail -------------------------------------
def node_rephrase(state: RAGState) -> dict:
    """question -> rewritten/normalized query + domain classification (single LLM call), plus
    the cheap half of the clarification step (seeds clinical_facts / candidate_modifiers).

    Runs on EVERY question, so it is also the RESET point for every PER-QUESTION field. A thread
    outlives the question (Studio, and the CLI, keep one across turns), so anything reset here
    starts each question clean: the clarification budget (clarify_rounds / asked_questions), the
    validation loop (attempts / validation — a carried-over `attempts` costs the next question
    its retry, a carried-over rejected `validation` opens generate with "your previous answer
    was REJECTED"), and the error/refocus flags.

    `patient_facts` and `prev_question` are the SESSION-scoped exceptions: instead of resetting
    them this node ACCUMULATES facts into the first (so a follow-up keeps the VHB coinfection
    mentioned two questions ago) and hands the second — the PREVIOUS turn's question — to
    `refine`, so «¿y en embarazo?» is rewritten into a standalone query carrying the earlier
    topic, then records THIS question as the previous one for next turn. Both are cleared only
    on an explicit new patient (never here)."""
    prior = state.get("patient_facts") or {}
    r = refine(state["question"], prev_question=state.get("prev_question"), patient_facts=prior)
    turn_facts = r.get("known_facts", {})
    return {"search_query": r["query"], "in_domain": r["in_domain"],
            "patient_facts": {**prior, **turn_facts}, "prev_question": state["question"],
            "turn_facts": turn_facts, "possible_new_patient": r.get("possible_new_patient", False),
            "candidate_modifiers": r.get("candidate_modifiers", []),
            "pending_clarifications": [], "clarify_rounds": 0, "assessment": {},
            "asked_questions": [], "concept_map": "",
            "attempts": 0, "validation": {}, "refocus_query": "", "refining": False,
            "technical_error": "", "technical_detail": "", "output": ""}


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
    """After rephrase: out-of-domain to a message, in-domain to the patient-switch gate (which
    passes straight through unless refine flagged a probable different patient)."""
    return "confirm_patient" if state.get("in_domain", True) else "out_of_domain"


def route_after_confirm(state: RAGState, config) -> str:
    """Combined graph: from the gate on to the selected retrieval strategy."""
    return retrieval_entry(_resolve_mode(config))


def make_route_in_domain():
    """Dedicated graphs: route out-of-domain to a message, in-domain to the patient-switch gate."""
    def route(state: RAGState):
        return "confirm_patient" if state.get("in_domain", True) else "out_of_domain"
    return route


def make_route_to(entries: list):
    """Dedicated graphs: from the gate on to that graph's retrieval entry node(s) — a single
    node, or both parallel entries for the graph mode's fan-out."""
    def route(state: RAGState):
        return entries
    return route


# --- Patient-switch gate: the ONE blocking pause, and only on contradiction ---
def _is_affirmative(answer) -> bool:
    """Parse the doctor's yes/no to "is this a different patient?". Clearing the remembered
    data needs an EXPLICIT yes — an empty reply keeps the patient, because auto-forgetting a
    patient on a stray Enter is worse than answering one extra question about the same one (and
    a genuine switch the doctor waves past can still be corrected with /nuevo)."""
    if isinstance(answer, bool):
        return answer
    return str(answer or "").strip().lower()[:2] in ("s", "sí", "si", "y", "ye", "1", "tr", "ok")


def node_confirm_patient(state: RAGState) -> dict:
    """Blocking safety gate. If refine flagged that this question likely concerns a DIFFERENT
    patient than the accumulated facts, pause and ASK before answering — answering it with the
    previous patient's renal function or gestation folded in would put a wrong recommendation on
    screen, and that is the exact harm the whole system guards against. This is the deliberate
    exception to answer-first: the clarification is offered after the answer because a generic
    answer is safe, but a cross-patient answer is not.

    It fires ONLY on a flagged contradiction AND only when there is remembered data to
    contradict, so a normal follow-up never pauses. On "yes, new patient" it drops the previous
    patient's facts, keeping only this question's own (`turn_facts`); `prev_question` was already
    set to this question by rephrase, so the next follow-up resolves against it correctly."""
    if not state.get("possible_new_patient") or not (state.get("patient_facts") or {}):
        return {"possible_new_patient": False}
    answer = interrupt({"confirm_new_patient": True,
                        "facts": state.get("patient_facts"), "question": state["question"]})
    if _is_affirmative(answer):
        return {"patient_facts": dict(state.get("turn_facts") or {}),
                "possible_new_patient": False}
    return {"possible_new_patient": False}


# --- Collapsed retrieval nodes (combined graph) ----------------------------
@guarded(STEP_RETRIEVAL)
def node_retrieve(state: RAGState) -> dict:
    """search_query (+ carried-over patient facts) -> ~20 candidates via hybrid (dense + BM25)."""
    candidates = retrieve_hybrid(_with_facts(state["search_query"], state),
                                 top_k=20, prefetch_limit=30)
    return {"candidates": candidates, "retrieval_mode": "baseline"}


@guarded(STEP_RETRIEVAL)
def node_rerank(state: RAGState) -> dict:
    """candidates -> top 5 reordered by the cross-encoder + numbered context."""
    contexts = rerank(_with_facts(state["search_query"], state), state["candidates"], top_k=5)
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
    @guarded(STEP_RETRIEVAL)
    def node(state: RAGState) -> dict:
        rewritten = _with_facts(state.get("search_query") or state["question"], state)
        contexts, concept_map = _retrieve_with(mode, state["question"], rewritten)
        chunk_index, formatted_context = build_context(contexts)
        return {"contexts": contexts, "chunk_index": chunk_index,
                "formatted_context": formatted_context, "concept_map": concept_map,
                "retrieval_mode": mode}
    node.__name__ = f"node_{mode}_retrieve"
    return node


# Collapsed node per non-baseline mode (baseline keeps its explicit retrieve -> rerank pair).
RETRIEVAL_NODES = {mode: _make_retrieval_node(mode)
                   for mode in VALID_MODES if mode != "baseline"}


# --- Clarification: assess before answering, OFFER the refinement after -----
def node_assess_context(state: RAGState) -> dict:
    """Decide which clinical modifiers the answer hinges on (gestation, renal function,
    coinfection…) and that the doctor has not provided.

    This no longer GATES the answer. It runs before `generate` because its output is what makes
    a non-blocking flow safe: the pending dimensions go into the prompt as explicitly UNKNOWN,
    so the answer lays out the branches instead of silently picking one. The same list is then
    offered as an optional refinement once the answer is on screen. Once the budget is spent it
    stops proposing anything."""
    if state.get("clarify_rounds", 0) >= CLARIFY_MAX_ROUNDS:
        return {"pending_clarifications": []}   # budget spent: do not ask again
    a = assess(state["question"], state["formatted_context"],
               known_facts=state.get("patient_facts"),
               candidate_modifiers=state.get("candidate_modifiers"),
               asked_questions=state.get("asked_questions"),
               max_questions=CLARIFY_QUESTIONS_PER_ROUND)
    return {"pending_clarifications": a["questions"] if a["needs_clarification"] else [],
            "assessment": {k: a.get(k) for k in
                           ("clinically_relevant", "branches_on", "already_covered")}}


def _fold_answers(state: RAGState, pending: list, answers) -> dict:
    """Merge the doctor's refinement answers into patient_facts and close the round.

    Plain text is keyed by the QUESTION it answers (not by a fixed key) so several answers do
    not overwrite each other, and every question asked is recorded even when left blank — a
    skipped question must not come back. patient_facts carries no reducer (so it can be cleared
    per patient), so the merge is done by hand here; clarify_rounds is per-question."""
    if isinstance(answers, dict):
        new_facts = {k: v for k, v in answers.items() if k and str(v).strip()}
    else:
        text = str(answers or "").strip()
        new_facts = {pending[0]: text} if (len(pending) == 1 and text) else {}
        if text and len(pending) != 1:
            new_facts = {"respuesta_medico": text}
    # Whatever was NOT answered stays pending: it still feeds the "unknown data" block of the
    # prompt, so the refined answer keeps laying out the branches for the dimensions the doctor
    # skipped instead of silently assuming them. (It will not be offered again — the round
    # budget is spent.) Free text answering several questions at once cannot be attributed to
    # any of them, so they all stay open.
    return {"patient_facts": {**(state.get("patient_facts") or {}), **new_facts},
            "clarify_rounds": state.get("clarify_rounds", 0) + 1,
            "pending_clarifications": [q for q in pending if q not in new_facts],
            "asked_questions": list(state.get("asked_questions") or []) + list(pending),
            "refining": bool(new_facts)}


def node_refine_offer(state: RAGState) -> dict:
    """Answer first, refine after: the run pauses here with the ANSWER ALREADY PRODUCED.

    The clarification used to sit before `generate`, so a doctor asking a normal clinical
    question faced up to three sequential pauses before reading a single word — at the point of
    care that is worse than a slightly generic answer. Now `evidence` has already written
    `output`, and this pause only OFFERS to narrow it down; ignoring it (an empty answer) ends
    the run with the answer the doctor already has.

    If they do supply something, the facts are folded in and the run loops back through
    re_retrieve -> generate, so the datum PULLS the conditional passages and the refined answer
    cites them. `attempts`/`validation` are reset because that regeneration is a NEW answer over
    a NEW context, not a retry of the rejected one."""
    pending = state.get("pending_clarifications") or []
    if not pending or state.get("clarify_rounds", 0) >= CLARIFY_MAX_ROUNDS:
        return {"refining": False}
    folded = _fold_answers(state, pending, interrupt({"questions": pending, "optional": True}))
    if not folded["refining"]:
        return folded
    return {**folded, "attempts": 0, "validation": {}}


def route_refinement(state: RAGState) -> str:
    """Loop back to re-retrieve + regenerate only if the doctor actually supplied a datum."""
    return "re_retrieve" if state.get("refining") else "end"


def _retrieve_for_mode(mode: str, question: str, query: str) -> tuple[list, str]:
    """Run `mode`'s retrieval over an already-enriched query, returning (payloads, concept_map).

    The single place that knows baseline is the mode without one search function (the graph
    splits it into retrieve -> rerank), so every node that re-runs retrieval dispatches through
    here instead of repeating the special case."""
    if mode == "baseline":
        return rerank(query, retrieve_hybrid(query, top_k=20, prefetch_limit=30), top_k=5), ""
    return _retrieve_with(mode, question, query)


def _facts_phrase(patient_facts: dict | None) -> str:
    """Compact inline rendering of the patient data, to enrich the retrieval query."""
    facts = {k: v for k, v in (patient_facts or {}).items() if k}
    return "; ".join(f"{k}: {v}" if v else k for k, v in facts.items())


def _with_facts(base: str, state: RAGState) -> str:
    """Append the accumulated patient facts to a retrieval query.

    This is what carries a datum ACROSS turns: on the second question the facts are no longer in
    the question text, so without this the VHB coinfection mentioned earlier would steer
    generation (it is in the prompt) but not retrieval — and generate would be told to use a
    branch whose chunk was never fetched. Enriching the query keeps the two in step. On the
    first turn, where the fact is still in the question, it is merely redundant."""
    facts = _facts_phrase(state.get("patient_facts"))
    return f"{base}\nContexto del paciente: {facts}" if facts else base


@guarded(STEP_RETRIEVAL)
def node_re_retrieve(state: RAGState, config) -> dict:
    """After a refinement added patient data MID-TURN, re-run retrieval so the new datum pulls
    the conditional passages (the HBV-coinfection or first-trimester branch) for generate to
    cite. No-op unless THIS turn's refinement added something (clarify_rounds > 0): the initial
    retrieval already enriched with every fact known at the start of the turn, including data
    carried over from earlier turns, so there is nothing new to fetch otherwise."""
    if not state.get("clarify_rounds") or not _facts_phrase(state.get("patient_facts")):
        return {}
    mode = state.get("retrieval_mode") or _resolve_mode(config)
    rewritten = _with_facts(state.get("search_query") or state["question"], state)
    contexts, concept_map = _retrieve_for_mode(mode, state["question"], rewritten)
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context, "concept_map": concept_map}


# --- Tail: generate <-> validate -> evidence/fallback ----------------------
@guarded(STEP_GENERATION)
def node_generate(state: RAGState) -> dict:
    """context + question -> structured answer. Three non-citable blocks may accompany the
    numbered context: the clarified patient data (clinical_facts), which selects the applicable
    branch of the guides; the dimensions still UNKNOWN (pending_clarifications), so the answer
    lays out the branches rather than assuming one; and the retrieval mode's concept map, which
    shows how the concepts connect. None can be cited — `validate` still requires every claim to
    be grounded in the chunks. On a validator rejection, its feedback is injected so the retry
    can correct it."""
    user = build_user_prompt(state["question"], state["formatted_context"],
                             clinical_facts=state.get("patient_facts"),
                             concept_map=state.get("concept_map"),
                             open_questions=state.get("pending_clarifications"))
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


@guarded(STEP_RETRIEVAL)
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
        f"{_with_facts(state.get('search_query') or state['question'], state)}\n{focus}")
    contexts = rerank(state["question"], merge_dedup(found, previous),
                      top_k=len(previous) or 8)
    chunk_index, formatted_context = build_context(contexts)
    return {"contexts": contexts, "chunk_index": chunk_index,
            "formatted_context": formatted_context, "refocus_query": focus,
            "concept_map": concept_map or state.get("concept_map", "")}


@guarded(STEP_FORMATTING)
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
