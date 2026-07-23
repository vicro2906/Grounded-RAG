"""Routing decisions and the parsing that feeds them.

Pure functions, no graph and no LLM: these are the branches that decide whether an unvalidated
answer reaches the doctor, so they are worth pinning down on their own.
"""
import pytest

from langgraph.errors import GraphInterrupt

from pipeline.config import MAX_ITER, STEP_GENERATION, STEP_RETRIEVAL, VALID_MODES
from pipeline.nodes import (_facts_phrase, _fold_answers, _resolve_mode, guarded,
                            node_re_retrieve, retrieval_entry, route_on_error,
                            route_refinement, route_validation)
from rag import _facts_to_dict


# --- route_validation: never fail open -------------------------------------
def test_valid_answer_goes_to_evidence():
    assert route_validation({"validation": {"is_valid": True}, "attempts": 1}) == "evidence"


def test_technical_judge_error_never_shows_the_answer():
    """A judge that could not run must fall back, not pass the answer through."""
    state = {"validation": {"is_valid": True, "error": True}, "attempts": 1}
    assert route_validation(state) == "fallback"


def test_invalid_answer_chases_the_missing_evidence_before_retrying():
    """The retry must go through retrieval, not straight back to generate: regenerating over
    the same context cannot fix a retrieval miss, the usual cause of a rejection."""
    assert route_validation({"validation": {"is_valid": False},
                             "attempts": 1}) == "refocus_retrieve"


def test_retry_budget_is_capped():
    assert route_validation({"validation": {"is_valid": False},
                             "attempts": MAX_ITER}) == "fallback"


# --- route_refinement: only re-answer if a datum actually arrived -----------
def test_supplied_datum_re_answers():
    assert route_refinement({"refining": True}) == "re_retrieve"


def test_declined_refinement_ends_the_run():
    assert route_refinement({"refining": False}) == "end"
    assert route_refinement({}) == "end"


# --- route_on_error: a failed step never continues down the pipeline -------
def test_a_failed_step_diverts_to_the_message():
    assert route_on_error("generate")({"technical_error": STEP_RETRIEVAL}) == "technical_error"


def test_a_healthy_step_carries_on():
    assert route_on_error("generate")({}) == "generate"
    assert route_on_error("generate")({"technical_error": ""}) == "generate"


# --- folding the doctor's answers into the patient facts --------------------
def test_single_answer_is_keyed_by_the_question_it_answers():
    folded = _fold_answers({}, ["¿Hay coinfección por VHB?"], "sí")
    assert folded["patient_facts"] == {"¿Hay coinfección por VHB?": "sí"}
    assert folded["refining"] is True


def test_blank_answer_declines_without_losing_the_question():
    """An unanswered question must still count as asked, or assess would offer it again."""
    folded = _fold_answers({}, ["¿Función renal?"], "")
    assert folded["patient_facts"] == {}
    assert folded["refining"] is False
    assert folded["asked_questions"] == ["¿Función renal?"]


def test_structured_answers_merge_and_drop_the_blank_ones():
    folded = _fold_answers({"patient_facts": {"CD4": "200"}},
                           ["¿VHB?", "¿Función renal?"],
                           {"¿VHB?": "sí", "¿Función renal?": "  "})
    assert folded["patient_facts"] == {"CD4": "200", "¿VHB?": "sí"}


def test_unanswered_dimensions_stay_flagged_as_unknown():
    """Answering one of the offered questions must not make the others disappear: they still
    have to reach the refined answer as UNKNOWN so it keeps presenting their branches."""
    folded = _fold_answers({}, ["¿VHB?", "¿Función renal?"],
                           {"¿VHB?": "sí", "¿Función renal?": ""})
    assert folded["pending_clarifications"] == ["¿Función renal?"]


def test_previous_facts_are_preserved():
    folded = _fold_answers({"patient_facts": {"embarazo": "sí"}}, ["¿VHB?"], "no")
    assert folded["patient_facts"] == {"embarazo": "sí", "¿VHB?": "no"}


# --- mode resolution: unknown input must not crash a run --------------------
@pytest.mark.parametrize("mode", VALID_MODES)
def test_every_registered_mode_resolves_and_has_an_entry_node(mode):
    assert _resolve_mode({"configurable": {"retrieval_mode": mode}}) == mode
    assert retrieval_entry(mode)


def test_unknown_mode_falls_back_to_the_default():
    from pipeline.config import RETRIEVAL_MODE
    assert _resolve_mode({"configurable": {"retrieval_mode": "inventado"}}) == RETRIEVAL_MODE
    assert _resolve_mode(None) == RETRIEVAL_MODE


# --- clinical facts parsing -------------------------------------------------
def test_facts_are_parsed_into_attribute_value_pairs():
    assert _facts_to_dict(["embarazo: sí", "CD4: 200"]) == {"embarazo": "sí", "CD4": "200"}


def test_value_containing_a_colon_is_preserved():
    assert _facts_to_dict(["pauta_actual: BIC/FTC/TAF: desde 2023"]) == {
        "pauta_actual": "BIC/FTC/TAF: desde 2023"}


def test_malformed_facts_are_dropped_not_crashed():
    assert _facts_to_dict(["", "   ", None, "sin_valor"]) == {"sin_valor": ""}


# --- re-retrieval only when a refinement actually added something ----------
# This is a latency guarantee, and the kind that erodes silently: re-running retrieval "just in
# case" costs a full extra pass (in graph mode, another LLM call too) and nothing breaks, so
# nobody would notice it came back.
def test_no_refinement_means_no_second_retrieval():
    """Facts that arrived inside the question were already in the first retrieval query."""
    state = {"question": "¿Pauta?", "search_query": "¿Pauta?",
             "patient_facts": {"embarazo": "sí"}, "clarify_rounds": 0}
    assert node_re_retrieve(state, None) == {}


def test_no_facts_means_no_second_retrieval():
    state = {"question": "¿Pauta?", "search_query": "¿Pauta?",
             "patient_facts": {}, "clarify_rounds": 1}
    assert node_re_retrieve(state, None) == {}


# --- the guard around every step that calls out to a service ---------------
def test_a_failing_step_becomes_a_labelled_error_not_an_exception():
    @guarded(STEP_RETRIEVAL)
    def node(state):
        raise ConnectionError("qdrant unreachable")

    out = node({})
    assert out["technical_error"] == STEP_RETRIEVAL
    assert "ConnectionError" in out["technical_detail"]
    assert "qdrant unreachable" in out["technical_detail"]


def test_a_working_step_is_untouched():
    @guarded(STEP_RETRIEVAL)
    def node(state):
        return {"contexts": ["algo"]}

    assert node({}) == {"contexts": ["algo"]}


def test_the_guard_forwards_the_config_langgraph_passes_by_keyword():
    """Nodes that need the run config receive it as a KEYWORD argument, so a wrapper that only
    forwarded positional ones would break them the moment they were guarded."""
    @guarded(STEP_RETRIEVAL)
    def node(state, config):
        return {"seen": config}

    assert node({}, config={"configurable": {}}) == {"seen": {"configurable": {}}}


def test_the_guard_lets_the_pause_through():
    """`interrupt()` signals itself with an exception. If the guard swallowed it, a pause would
    be reported to the doctor as a service outage and the run could never resume — so
    LangGraph's own control-flow exceptions must pass straight through."""
    @guarded(STEP_GENERATION)
    def node(state):
        raise GraphInterrupt(("pausa",))

    with pytest.raises(GraphInterrupt):
        node({})


def test_patient_data_is_rendered_compactly_for_the_query():
    assert _facts_phrase({"embarazo": "sí", "CD4": "200"}) == "embarazo: sí; CD4: 200"
    assert _facts_phrase({"gestacion": ""}) == "gestacion"
    assert _facts_phrase({}) == "" and _facts_phrase(None) == ""
