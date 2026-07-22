"""End-to-end wiring of the combined graph, with every LLM/network boundary stubbed.

These cover the two loops that carry state (clarification and validation) and the routing
between them — the parts where a regression is silent rather than loud.
"""
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from pipeline import build_combined_graph
from pipeline.config import MSG_NOT_VALIDATED, MSG_OUT_OF_DOMAIN


BASELINE = {"retrieval_mode": "baseline"}


@pytest.fixture
def app():
    """The graph as the CLI compiles it: with persistence, so interrupts can be resumed."""
    return build_combined_graph(checkpointer=InMemorySaver())


def thread():
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


def test_answers_without_clarification(app, graph_env):
    result = app.invoke({"question": "¿Qué TAR en coinfección por VHB?"},
                        context=BASELINE, config=thread())
    assert "__interrupt__" not in result
    assert "RESPUESTA" in result["output"]


def test_out_of_domain_short_circuits(app, graph_env):
    result = app.invoke({"question": "una pregunta fuera de dominio"},
                        context=BASELINE, config=thread())
    assert result["output"] == MSG_OUT_OF_DOMAIN
    assert not graph_env.llm.calls, "an out-of-domain question must never reach generation"


def test_clarification_pauses_and_resumes(app, graph_env):
    """The CLI contract: invoke returns __interrupt__ with the pending questions, and resuming
    with Command produces the final answer. Without a checkpointer this run could not resume —
    which is exactly the bug this guards."""
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"], []]
    cfg = thread()

    paused = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=cfg)
    assert "output" not in paused
    interrupts = paused["__interrupt__"]
    assert interrupts[0].value["questions"] == ["¿Hay coinfección por VHB?"]

    done = app.invoke(Command(resume="sí, VHB positivo"), context=BASELINE, config=cfg)
    assert "RESPUESTA" in done["output"]
    # The doctor's answer must reach generation as non-citable context, not as a source.
    assert "DATOS APORTADOS POR EL MÉDICO" in graph_env.llm.last_user_prompt
    assert "VHB positivo" in graph_env.llm.last_user_prompt


def test_clarification_budget_is_capped(app, graph_env):
    """assess keeps asking, but CLARIFY_MAX_ROUNDS must stop the loop and answer anyway."""
    from pipeline.config import CLARIFY_MAX_ROUNDS

    graph_env.assess_questions = [[f"¿Dato {i}?"] for i in range(CLARIFY_MAX_ROUNDS + 3)]
    cfg = thread()

    result = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=cfg)
    rounds = 0
    while "__interrupt__" in result:
        rounds += 1
        assert rounds <= CLARIFY_MAX_ROUNDS, "the clarification loop is not capped"
        result = app.invoke(Command(resume="no lo sé"), context=BASELINE, config=cfg)
    assert rounds == CLARIFY_MAX_ROUNDS
    assert "RESPUESTA" in result["output"]


def test_invalid_answer_is_not_shown(app, graph_env):
    graph_env.valid = False
    result = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())
    assert result["output"] == MSG_NOT_VALIDATED
    assert len(graph_env.llm.calls) == 2, "the validation loop must retry exactly MAX_ITER times"
    assert "REINTENTO" in graph_env.llm.last_user_prompt


def test_rejection_re_retrieves_the_unsupported_claims_before_retrying(app, graph_env):
    """The retry must chase EVIDENCE, not just reword: regenerating over the same chunks
    cannot fix a retrieval miss, which is the usual cause of a grounding rejection."""
    graph_env.valid = False
    app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())

    assert len(graph_env.retrieval_queries) == 2, "the rejection must trigger a new retrieval"
    assert "afirmación sin respaldo" in graph_env.retrieval_queries[1]
    # The prompt of the retry must name what was unsupported and warn that the context moved.
    retry_prompt = graph_env.llm.last_user_prompt
    assert "afirmación sin respaldo" in retry_prompt
    assert "puede haber cambiado" in retry_prompt


def test_refocus_keeps_the_already_grounded_context_as_a_candidate(app, graph_env):
    """Merging rather than replacing: the claims that WERE grounded must not lose their
    support just because another claim sent us back to the index."""
    graph_env.valid = False
    app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())

    _, candidates, _ = graph_env.rerank_calls[-1]
    assert {c["chunk_id"] for c in candidates} == {"c1", "c2"}


def test_no_unsupported_claims_means_no_extra_retrieval(app, graph_env, monkeypatch):
    """A rejection for RELEVANCE (nothing specific to chase) must not pay for a retrieval."""
    from pipeline import nodes
    monkeypatch.setattr(nodes, "validate", lambda *a, **k: {
        "is_valid": False, "error": False, "reason": "no aborda la pregunta",
        "unsupported_claims": []})

    app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())
    assert len(graph_env.retrieval_queries) == 1


def test_validation_state_does_not_leak_into_the_next_question(app, graph_env):
    """Regression: `attempts` and `validation` are per-question. A thread outlives the question
    (Studio, and the CLI), so without resetting them the next question opened with "your
    previous answer was REJECTED" and had already spent its retry budget."""
    cfg = thread()
    graph_env.valid = False
    app.invoke({"question": "primera pregunta"}, context=BASELINE, config=cfg)

    graph_env.valid = True
    graph_env.llm.calls.clear()
    result = app.invoke({"question": "segunda pregunta"}, context=BASELINE, config=cfg)

    assert len(graph_env.llm.calls) == 1, "the second question must start with a clean budget"
    assert "REINTENTO" not in graph_env.llm.last_user_prompt
    assert "RESPUESTA" in result["output"]


def test_clarification_state_does_not_leak_into_the_next_question(app, graph_env):
    """Same guarantee for the clarification side: a spent round budget must not silently skip
    the gate on the following question."""
    cfg = thread()
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"], []]
    app.invoke({"question": "primera pregunta"}, context=BASELINE, config=cfg)
    app.invoke(Command(resume="sí"), context=BASELINE, config=cfg)

    graph_env.assess_calls = 0
    graph_env.assess_questions = [["¿Función renal?"], []]
    paused = app.invoke({"question": "segunda pregunta"}, context=BASELINE, config=cfg)
    assert paused["__interrupt__"][0].value["questions"] == ["¿Función renal?"]

    done = app.invoke(Command(resume="aclaramiento 30 ml/min"), context=BASELINE, config=cfg)
    prompt = graph_env.llm.last_user_prompt
    assert "aclaramiento 30 ml/min" in prompt
    assert "¿Hay coinfección por VHB?" not in prompt, "previous patient's data leaked in"
