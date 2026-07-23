"""End-to-end wiring of the combined graph, with every LLM/network boundary stubbed.

These cover the two loops that carry state (clarification and validation) and the routing
between them — the parts where a regression is silent rather than loud.
"""
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from conftest import CHUNK
from pipeline import build_combined_graph
from pipeline.config import (MSG_NOT_VALIDATED, MSG_OUT_OF_DOMAIN, MSG_VALIDATION_ERROR,
                             STEP_FORMATTING, STEP_GENERATION, STEP_RETRIEVAL)


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


def test_answer_comes_before_the_refinement_offer(app, graph_env):
    """Answer-first: the pause must arrive WITH the answer already produced, never instead of
    it. The clarification used to gate generation, so the doctor faced up to three sequential
    pauses before reading a single word."""
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"]]
    paused = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())

    assert "RESPUESTA" in paused["output"], "the answer must exist before anything is asked"
    assert paused["__interrupt__"][0].value["questions"] == ["¿Hay coinfección por VHB?"]
    assert paused["__interrupt__"][0].value["optional"] is True


def test_unknown_dimensions_reach_generation_as_explicit_unknowns(app, graph_env):
    """What makes the non-blocking flow safe: unresolved dimensions are NAMED in the prompt, so
    the answer lays out the branches instead of quietly picking one."""
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"]]
    app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())

    prompt = graph_env.llm.last_user_prompt
    assert "DATOS DEL PACIENTE NO APORTADOS" in prompt
    assert "¿Hay coinfección por VHB?" in prompt


def test_declining_the_refinement_ends_with_the_answer_already_given(app, graph_env):
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"]]
    cfg = thread()
    paused = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=cfg)
    generations = len(graph_env.llm.calls)

    done = app.invoke(Command(resume=""), context=BASELINE, config=cfg)
    assert "__interrupt__" not in done
    assert done["output"] == paused["output"]
    assert len(graph_env.llm.calls) == generations, "declining must not cost another answer"


def test_accepting_the_refinement_re_answers_with_the_datum(app, graph_env):
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"], []]
    cfg = thread()
    app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=cfg)

    done = app.invoke(Command(resume="sí, VHB positivo"), context=BASELINE, config=cfg)
    assert "__interrupt__" not in done
    assert "RESPUESTA" in done["output"]
    # The datum must reach generation as non-citable context, and must have re-run retrieval so
    # the conditional passages are there to cite.
    prompt = graph_env.llm.last_user_prompt
    assert "DATOS APORTADOS POR EL MÉDICO" in prompt
    assert "VHB positivo" in prompt
    assert any("VHB positivo" in q for q in graph_env.retrieval_queries)


def test_only_one_refinement_offer_per_question(app, graph_env):
    """assess would keep proposing dimensions; CLARIFY_MAX_ROUNDS must stop the loop so the
    refinement cannot turn back into an interrogation."""
    from pipeline.config import CLARIFY_MAX_ROUNDS

    graph_env.assess_questions = [[f"¿Dato {i}?"] for i in range(CLARIFY_MAX_ROUNDS + 3)]
    cfg = thread()
    result = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=cfg)
    rounds = 0
    while "__interrupt__" in result:
        rounds += 1
        assert rounds <= CLARIFY_MAX_ROUNDS, "the refinement loop is not capped"
        result = app.invoke(Command(resume="un dato"), context=BASELINE, config=cfg)
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


# --- technical failures: a message, never a traceback ----------------------
# Everything below asserts the same two things: the doctor gets an intelligible message instead
# of a stack trace, AND that message is not mistakable for a clinical statement. Saying "no
# está en las guías" when the truth is "OpenAI is down" is a claim about the guidelines.
def test_retrieval_outage_ends_in_a_message(app, graph_env, monkeypatch):
    from pipeline import nodes
    monkeypatch.setattr(nodes, "retrieve_hybrid",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("qdrant down")))

    result = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())

    assert STEP_RETRIEVAL in result["output"]
    assert "problema técnico" in result["output"]
    assert "no significa que las guías no cubran" in result["output"]
    assert not graph_env.llm.calls, "an unreachable index must not reach generation"


def test_generation_outage_ends_in_a_message(app, graph_env, monkeypatch):
    from pipeline import nodes

    class Down:
        def invoke(self, messages):
            raise RuntimeError("openai unavailable")

    monkeypatch.setattr(nodes, "structured_llm", Down())
    result = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())

    assert STEP_GENERATION in result["output"]
    assert "__interrupt__" not in result


def test_formatting_failure_ends_in_a_message(app, graph_env, monkeypatch):
    """The last step is pure Python, but if it ever throws we still have a validated answer and
    no way to render its sources — showing it without them would drop the citations that are
    the whole promise."""
    from pipeline import nodes
    monkeypatch.setattr(nodes, "format_answer",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("payload")))

    result = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())
    assert STEP_FORMATTING in result["output"]


def test_the_failing_step_is_named_and_the_cause_kept_for_the_trace(app, graph_env, monkeypatch):
    """The doctor gets the step; the exception stays in the state so an outage is diagnosable
    instead of becoming a silent mystery."""
    from pipeline import nodes
    monkeypatch.setattr(nodes, "retrieve_hybrid",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("qdrant down")))

    result = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())

    assert result["technical_error"] == STEP_RETRIEVAL
    assert "qdrant down" in result["technical_detail"]
    assert "qdrant down" not in result["output"], "the exception must not reach the doctor"


def test_a_judge_that_cannot_run_never_shows_the_answer(app, graph_env, monkeypatch):
    """The validator has its own message because its case is different: an answer EXISTS, we
    just could not verify it. It must not be shown."""
    from pipeline import nodes
    monkeypatch.setattr(nodes, "validate", lambda *a, **k: {
        "is_valid": False, "error": True, "reason": "técnico", "unsupported_claims": []})

    result = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())
    assert result["output"] == MSG_VALIDATION_ERROR


def test_guarding_the_pipeline_did_not_break_the_refinement_pause(app, graph_env):
    """Integration check that the guards left the pause intact. What actually protects it is
    the GraphBubbleUp re-raise inside `guarded`, pinned directly in test_pipeline_routing."""
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"]]
    paused = app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=thread())

    assert "__interrupt__" in paused
    assert not paused.get("technical_error")


def test_a_recovered_service_does_not_poison_the_next_question(app, graph_env, monkeypatch):
    """technical_error is per-question state, like attempts and validation."""
    from pipeline import nodes
    cfg = thread()
    monkeypatch.setattr(nodes, "retrieve_hybrid",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    app.invoke({"question": "primera"}, context=BASELINE, config=cfg)

    monkeypatch.setattr(nodes, "retrieve_hybrid", lambda *a, **k: [CHUNK])
    result = app.invoke({"question": "segunda"}, context=BASELINE, config=cfg)

    assert "RESPUESTA" in result["output"]
    assert not result.get("technical_error")


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


def test_the_round_budget_resets_but_the_patient_is_remembered(app, graph_env):
    """The two things must move in OPPOSITE directions across a turn, which is the whole point
    of splitting session-scoped from per-question state: the clarification BUDGET resets (the
    second question can offer its own refinement), while the patient DATA carries (the VHB
    coinfection supplied for the first question still steers the second)."""
    cfg = thread()
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"], []]
    app.invoke({"question": "primera pregunta"}, context=BASELINE, config=cfg)
    app.invoke(Command(resume="sí, VHB positivo"), context=BASELINE, config=cfg)

    graph_env.assess_calls = 0
    graph_env.assess_questions = [["¿Función renal?"], []]
    # Budget reset -> the second question is free to raise its own refinement.
    paused = app.invoke({"question": "segunda pregunta"}, context=BASELINE, config=cfg)
    assert paused["__interrupt__"][0].value["questions"] == ["¿Función renal?"]
    # Patient remembered -> the first question's datum is already in the second answer's prompt.
    assert "VHB positivo" in graph_env.llm.last_user_prompt

    app.invoke(Command(resume="aclaramiento 30 ml/min"), context=BASELINE, config=cfg)
    prompt = graph_env.llm.last_user_prompt
    assert "aclaramiento 30 ml/min" in prompt and "VHB positivo" in prompt


def test_each_question_is_rewritten_against_the_previous_one(app, graph_env):
    """Contextual rewriting: turn 2's refine must receive turn 1's question, so an elliptical
    follow-up can be resolved. Turn 1 has nothing before it."""
    cfg = thread()
    graph_env.assess_questions = [[], []]
    app.invoke({"question": "¿qué TAR de inicio se recomienda?"}, context=BASELINE, config=cfg)
    app.invoke({"question": "¿y en embarazo?"}, context=BASELINE, config=cfg)

    assert graph_env.refine_prev == [None, "¿qué TAR de inicio se recomienda?"]


def test_a_new_patient_also_resets_the_conversation_context(app, graph_env):
    """After /nuevo the next question must not resolve against the previous patient's question."""
    cfg = thread()
    graph_env.assess_questions = [[], []]
    app.invoke({"question": "¿qué TAR de inicio se recomienda?"}, context=BASELINE, config=cfg)
    app.update_state(cfg, {"patient_facts": {}, "prev_question": ""})
    app.invoke({"question": "¿y en embarazo?"}, context=BASELINE, config=cfg)

    assert graph_env.refine_prev[-1] == ""      # no prior question carried across the reset


def test_a_carried_over_fact_steers_the_follow_up_retrieval(app, graph_env):
    """A remembered datum must reach the NEXT question's retrieval, not only its generation:
    otherwise generate is told to use the VHB branch whose chunk was never fetched."""
    cfg = thread()
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"], []]
    app.invoke({"question": "primera pregunta"}, context=BASELINE, config=cfg)
    app.invoke(Command(resume="sí, VHB positivo"), context=BASELINE, config=cfg)

    graph_env.assess_questions = [[]]
    graph_env.retrieval_queries.clear()
    app.invoke({"question": "¿y la monitorización?"}, context=BASELINE, config=cfg)
    assert any("VHB positivo" in q for q in graph_env.retrieval_queries), \
        "the carried datum must enrich the follow-up's retrieval query"


def test_a_new_patient_clears_the_remembered_data(app, graph_env):
    """update_state({patient_facts: {}}) is what the CLI's /nuevo runs — the carried data must
    be gone from the next question so a new patient never inherits the previous one's."""
    cfg = thread()
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"], []]
    app.invoke({"question": "primera pregunta"}, context=BASELINE, config=cfg)
    app.invoke(Command(resume="sí, VHB positivo"), context=BASELINE, config=cfg)

    app.update_state(cfg, {"patient_facts": {}})

    graph_env.assess_questions = [[]]
    app.invoke({"question": "otro paciente"}, context=BASELINE, config=cfg)
    assert "VHB positivo" not in graph_env.llm.last_user_prompt
