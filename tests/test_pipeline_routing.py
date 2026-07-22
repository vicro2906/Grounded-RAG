"""Routing decisions and the parsing that feeds them.

Pure functions, no graph and no LLM: these are the branches that decide whether an unvalidated
answer reaches the doctor, so they are worth pinning down on their own.
"""
import pytest

from pipeline.config import MAX_ITER, VALID_MODES
from pipeline.nodes import _resolve_mode, retrieval_entry, route_assess, route_validation
from rag import _facts_to_dict


# --- route_validation: never fail open -------------------------------------
def test_valid_answer_goes_to_evidence():
    assert route_validation({"validation": {"is_valid": True}, "attempts": 1}) == "evidence"


def test_technical_judge_error_never_shows_the_answer():
    """A judge that could not run must fall back, not pass the answer through."""
    state = {"validation": {"is_valid": True, "error": True}, "attempts": 1}
    assert route_validation(state) == "fallback"


def test_invalid_answer_retries_until_the_budget_is_spent():
    assert route_validation({"validation": {"is_valid": False}, "attempts": 1}) == "generate"
    assert route_validation({"validation": {"is_valid": False},
                             "attempts": MAX_ITER}) == "fallback"


# --- route_assess: clarify only with a live budget --------------------------
def test_pending_questions_route_to_clarify():
    assert route_assess({"pending_clarifications": ["¿VHB?"], "clarify_rounds": 0}) == "clarify"


def test_no_pending_questions_routes_to_generate():
    assert route_assess({"pending_clarifications": [], "clarify_rounds": 0}) == "generate"


def test_spent_budget_answers_instead_of_asking_forever():
    from pipeline.config import CLARIFY_MAX_ROUNDS
    state = {"pending_clarifications": ["¿VHB?"], "clarify_rounds": CLARIFY_MAX_ROUNDS}
    assert route_assess(state) == "generate"


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
