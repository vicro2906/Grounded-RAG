"""Routing decisions and the parsing that feeds them.

Pure functions, no graph and no LLM: these are the branches that decide whether an unvalidated
answer reaches the doctor, so they are worth pinning down on their own.
"""
import pytest

from pipeline.config import MAX_ITER, VALID_MODES
from pipeline.nodes import (_facts_phrase, _fold_answers, _resolve_mode, node_re_retrieve,
                            retrieval_entry, route_refinement, route_validation)
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


# --- folding the doctor's answers into the patient facts --------------------
def test_single_answer_is_keyed_by_the_question_it_answers():
    folded = _fold_answers({}, ["¿Hay coinfección por VHB?"], "sí")
    assert folded["clinical_facts"] == {"¿Hay coinfección por VHB?": "sí"}
    assert folded["refining"] is True


def test_blank_answer_declines_without_losing_the_question():
    """An unanswered question must still count as asked, or assess would offer it again."""
    folded = _fold_answers({}, ["¿Función renal?"], "")
    assert folded["clinical_facts"] == {}
    assert folded["refining"] is False
    assert folded["asked_questions"] == ["¿Función renal?"]


def test_structured_answers_merge_and_drop_the_blank_ones():
    folded = _fold_answers({"clinical_facts": {"CD4": "200"}},
                           ["¿VHB?", "¿Función renal?"],
                           {"¿VHB?": "sí", "¿Función renal?": "  "})
    assert folded["clinical_facts"] == {"CD4": "200", "¿VHB?": "sí"}


def test_unanswered_dimensions_stay_flagged_as_unknown():
    """Answering one of the offered questions must not make the others disappear: they still
    have to reach the refined answer as UNKNOWN so it keeps presenting their branches."""
    folded = _fold_answers({}, ["¿VHB?", "¿Función renal?"],
                           {"¿VHB?": "sí", "¿Función renal?": ""})
    assert folded["pending_clarifications"] == ["¿Función renal?"]


def test_previous_facts_are_preserved():
    folded = _fold_answers({"clinical_facts": {"embarazo": "sí"}}, ["¿VHB?"], "no")
    assert folded["clinical_facts"] == {"embarazo": "sí", "¿VHB?": "no"}


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
             "clinical_facts": {"embarazo": "sí"}, "clarify_rounds": 0}
    assert node_re_retrieve(state, None) == {}


def test_no_facts_means_no_second_retrieval():
    state = {"question": "¿Pauta?", "search_query": "¿Pauta?",
             "clinical_facts": {}, "clarify_rounds": 1}
    assert node_re_retrieve(state, None) == {}


def test_patient_data_is_rendered_compactly_for_the_query():
    assert _facts_phrase({"embarazo": "sí", "CD4": "200"}) == "embarazo: sí; CD4: 200"
    assert _facts_phrase({"gestacion": ""}) == "gestacion"
    assert _facts_phrase({}) == "" and _facts_phrase(None) == ""
