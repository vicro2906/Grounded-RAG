"""End-to-end wiring of the combined graph, with every LLM/network boundary stubbed.

These cover the two loops that carry state (clarification and validation) and the routing
between them — the parts where a regression is silent rather than loud.
"""
import asyncio
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import corpus
from conftest import CHUNK
from pipeline import build_combined_graph, refinement_reply
from pipeline.config import (MSG_NOT_VALIDATED, msg_out_of_domain, MSG_STEP_LABELS,
                             MSG_VALIDATION_ERROR, STEP_FORMATTING, STEP_GENERATION,
                             STEP_RETRIEVAL)


BASELINE = {"retrieval_mode": "baseline"}


@pytest.fixture
def app():
    """The graph as the CLI compiles it: with persistence, so interrupts can be resumed."""
    return build_combined_graph(checkpointer=InMemorySaver())


def thread():
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


# --- Specialty scoping: what keeps one area's guidelines out of another's answer ---
def test_every_retrieval_is_confined_to_the_active_specialty(app, graph_env):
    """The guarantee is not "the first search is scoped" — it is that NO search escapes. A
    single unscoped call (the hybrid complement, a validator-driven re-retrieval) would put
    another area's guidance into a context the doctor is told is theirs."""
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "specialty": "vih"}}
    app.invoke({"question": "¿Qué TAR en coinfección por VHB?"}, context=BASELINE, config=cfg)

    assert graph_env.retrieval_scopes, "no retrieval ran"
    assert all(s is not None and s.specialty == "vih" for s in graph_env.retrieval_scopes)


def test_the_specialty_is_pinned_on_the_first_turn_and_survives_the_conversation(app, graph_env):
    """Resolved once and written into the state, so a later turn cannot silently answer from a
    different area because a config default disagreed."""
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "specialty": "vih"}}
    app.invoke({"question": "¿Qué TAR de inicio?"}, context=BASELINE, config=cfg)
    assert app.get_state(cfg).values["specialty"] == "vih"

    # Second turn on the same thread, with the configurable gone: the state must still decide.
    bare = {"configurable": {"thread_id": cfg["configurable"]["thread_id"]}}
    app.invoke({"question": "¿y en embarazo?"}, context=BASELINE, config=bare)
    assert app.get_state(bare).values["specialty"] == "vih"
    assert all(s.specialty == "vih" for s in graph_env.retrieval_scopes)


def test_an_unknown_specialty_widens_instead_of_emptying_the_corpus(app, graph_env):
    """A typo must not silently produce «no está en las guías» over an empty filtered corpus —
    that reads as a clinical statement and it would be false."""
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "specialty": "no_existe"}}
    app.invoke({"question": "¿Qué TAR de inicio?"}, context=BASELINE, config=cfg)
    assert app.get_state(cfg).values["specialty"] == corpus.default_specialty()


def test_the_prompts_follow_the_specialty_too(app, graph_env):
    """Scoping retrieval without scoping the prompt would leave the guardrail and the clinical
    modifiers describing an area the answer no longer comes from."""
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "specialty": "vih"}}
    app.invoke({"question": "¿Qué TAR de inicio?"}, context=BASELINE, config=cfg)
    assert graph_env.refine_specialties == ["vih"]


def test_answers_without_clarification(app, graph_env):
    result = app.invoke({"question": "¿Qué TAR en coinfección por VHB?"},
                        context=BASELINE, config=thread())
    assert "__interrupt__" not in result
    assert "RESPUESTA" in result["output"]


def test_out_of_domain_short_circuits(app, graph_env):
    result = app.invoke({"question": "una pregunta fuera de dominio"},
                        context=BASELINE, config=thread())
    assert result["output"] == msg_out_of_domain(corpus.default_specialty())
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


# --- declining an offer of SEVERAL questions -------------------------------
# The default offer carries three questions (CLARIFY_QUESTIONS_PER_ROUND), and Enter on all of
# them is the documented way out. It did not work: the CLI encoded "nothing answered" as {},
# LangGraph does not accept an empty dict as a resume value, and the interrupt fired again with
# the budget never spent — the same three questions, forever. Only the real graph shows this;
# the fake one in test_cli.py resumes on anything.
def test_leaving_every_question_blank_ends_the_run(app, graph_env):
    graph_env.assess_questions = [["¿Serología?", "¿Carga viral?", "¿CD4?"]]
    cfg = thread()
    app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=cfg)

    reply = refinement_reply(["¿Serología?", "¿Carga viral?", "¿CD4?"],
                             {"¿Serología?": "", "¿Carga viral?": "", "¿CD4?": ""})
    done = app.invoke(Command(resume=reply), context=BASELINE, config=cfg)

    assert "__interrupt__" not in done, "declining must end the run, not re-ask"
    assert "RESPUESTA" in done["output"]


def test_an_empty_dict_is_never_what_a_decline_looks_like(app, graph_env):
    """Pinning the trap itself rather than only its symptom: if some frontend ever sends {},
    the run silently loops. `refinement_reply` is the one place that must not produce it."""
    assert refinement_reply(["¿A?", "¿B?"], {"¿A?": "", "¿B?": "   "}) == ""
    assert refinement_reply(["¿A?", "¿B?"], {}) == ""
    assert refinement_reply(["¿A?"], {"¿A?": ""}) == ""


def test_answering_only_some_questions_still_refines(app, graph_env):
    graph_env.assess_questions = [["¿Serología?", "¿Carga viral?", "¿CD4?"], []]
    cfg = thread()
    app.invoke({"question": "¿Qué pauta inicio?"}, context=BASELINE, config=cfg)

    reply = refinement_reply(["¿Serología?", "¿Carga viral?", "¿CD4?"],
                             {"¿Serología?": "", "¿Carga viral?": "indetectable", "¿CD4?": ""})
    done = app.invoke(Command(resume=reply), context=BASELINE, config=cfg)

    assert "__interrupt__" not in done
    assert "indetectable" in graph_env.llm.last_user_prompt
    # The two left blank stay pending, so the refined answer keeps laying out their branches.
    assert app.get_state(cfg).values["pending_clarifications"] == ["¿Serología?", "¿CD4?"]


# --- the stream contract the CLI depends on --------------------------------
# The CLI reads the run as a stream so the doctor sees the work instead of ~20 s of silence.
# Everything above drives the graph with `invoke`, and tests/test_cli.py drives a FAKE graph —
# so without these, the contract between the two is asserted by nobody.
def _stream(app, payload, cfg):
    return list(app.stream(payload, context=BASELINE, config=cfg,
                           stream_mode=["updates", "custom"]))


def _outputs(chunks):
    """(node, output) for every update that put text on screen, in arrival order."""
    return [(node, (update or {}).get("output"))
            for channel, chunk in chunks if channel == "updates"
            for node, update in chunk.items()
            if node != "__interrupt__" and (update or {}).get("output")]


def test_the_answer_arrives_as_the_update_of_the_node_that_wrote_it(app, graph_env):
    """The CLI prints the last non-empty `output` of the stream instead of reading a final
    state. That is only equivalent because exactly ONE node writes a visible output per run."""
    chunks = _stream(app, {"question": "¿Qué TAR en coinfección por VHB?"}, thread())
    written = _outputs(chunks)

    assert [node for node, _ in written] == ["evidence"]
    assert "RESPUESTA" in written[0][1]


def test_the_pause_arrives_as_its_own_chunk(app, graph_env):
    """`__interrupt__` is a key in an updates chunk, not a returned value — if it stopped
    arriving that way the refinement would never be offered, silently."""
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"]]
    chunks = _stream(app, {"question": "¿Qué pauta inicio?"}, thread())

    pauses = [update for channel, chunk in chunks if channel == "updates"
              for node, update in chunk.items() if node == "__interrupt__"]
    assert pauses and pauses[0][0].value["questions"] == ["¿Hay coinfección por VHB?"]


def test_a_paused_run_resumes_through_the_stream(app, graph_env):
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"], []]
    cfg = thread()
    _stream(app, {"question": "¿Qué pauta inicio?"}, cfg)
    resumed = _stream(app, Command(resume="sí, VHB positivo"), cfg)

    assert [node for node, _ in _outputs(resumed)] == ["evidence"]
    assert "VHB positivo" in graph_env.llm.last_user_prompt


def test_a_no_op_node_streams_an_empty_update(app, graph_env):
    """re_retrieve does nothing on the first pass. The CLI must survive reading `output` off
    whatever it yields — an AttributeError here would swallow the whole answer."""
    chunks = _stream(app, {"question": "¿Qué TAR en coinfección por VHB?"}, thread())
    updates = {node: update for channel, chunk in chunks if channel == "updates"
               for node, update in chunk.items()}

    assert "re_retrieve" in updates and not updates["re_retrieve"]


def test_the_sections_read_reach_the_stream_before_the_answer(app, graph_env):
    """The one piece of progress worth keeping on screen. It has to arrive EARLY — its whole
    point is filling the wait — and it has to be real: these are the sections the answer is
    being written from, which is why showing them can never have to be taken back."""
    chunks = _stream(app, {"question": "¿Qué TAR en coinfección por VHB?"}, thread())
    kinds = [chunk.get("kind") for channel, chunk in chunks if channel == "custom"]
    sources = [chunk for channel, chunk in chunks
               if channel == "custom" and chunk.get("kind") == "sources"]

    assert sources and any("7.4.4" in item for item in sources[0]["items"])
    assert kinds.index("sources") == 0
    answered = [i for i, (channel, chunk) in enumerate(chunks)
                if channel == "updates" and (chunk.get("evidence") or {}).get("output")]
    seen_at = [i for i, (channel, chunk) in enumerate(chunks)
               if channel == "custom" and chunk.get("kind") == "sources"]
    assert seen_at[0] < answered[0]


def test_the_same_contract_holds_through_the_async_api(app, graph_env):
    """The web frontend consumes `astream`, so the sync nodes run in LangGraph's executor and the
    event loop stays free to paint. Nothing else exercises that path, and the failure would not
    be subtle: no answer at all, or a pause that can never be resumed."""
    cfg = thread()
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"], []]

    async def collect(payload):
        out = []
        async for chunk in app.astream(payload, context=BASELINE, config=cfg,
                                       stream_mode=["updates", "custom"]):
            out.append(chunk)
        return out

    paused = asyncio.run(collect({"question": "¿Qué pauta inicio?"}))
    assert [node for node, _ in _outputs(paused)] == ["evidence"]
    assert any(channel == "updates" and "__interrupt__" in chunk for channel, chunk in paused)
    assert any(channel == "custom" and chunk.get("kind") == "sources"
               for channel, chunk in paused)

    resumed = asyncio.run(collect(Command(resume="sí, VHB positivo")))
    assert [node for node, _ in _outputs(resumed)] == ["evidence"]
    assert "VHB positivo" in graph_env.llm.last_user_prompt


def test_the_answer_and_its_chunks_arrive_before_the_text(app, graph_env):
    """The web renders sources from `answer` + `chunk_index` collected FROM THE STREAM rather
    than asking the checkpointer mid-run. That only works if both are already in hand when
    `evidence` emits its text."""
    chunks = _stream(app, {"question": "¿Qué TAR en coinfección por VHB?"}, thread())
    seen = {}
    for channel, chunk in chunks:
        if channel != "updates":
            continue
        for node, update in chunk.items():
            if node == "__interrupt__" or not update:
                continue
            for key in ("answer", "chunk_index"):
                if update.get(key) and key not in seen:
                    seen[key] = node
            if update.get("output"):
                assert set(seen) == {"answer", "chunk_index"}, f"missing {seen} at {node}"
                return
    raise AssertionError("no visible output in the stream")


def test_every_progress_label_names_a_real_node(app):
    """MSG_STEP_LABELS is keyed by node name. A renamed node would not fail anything — the
    progress line would just go quiet for that step, which is invisible in a passing suite."""
    nodes = set(app.get_graph().nodes)
    unknown = sorted(set(MSG_STEP_LABELS) - nodes)
    assert not unknown, f"labels for nodes that do not exist: {unknown}"


# --- the blocking patient-switch gate --------------------------------------
def test_a_normal_follow_up_never_pauses_to_confirm(app, graph_env):
    """The gate must stay invisible unless refine actually flags a contradiction: a normal
    follow-up about the same patient answers straight through."""
    cfg = thread()
    graph_env.assess_questions = [["¿VHB?"], []]
    app.invoke({"question": "primera"}, context=BASELINE, config=cfg)
    app.invoke(Command(resume="sí, VHB positivo"), context=BASELINE, config=cfg)

    graph_env.assess_questions = [[]]
    result = app.invoke({"question": "¿y la dosis?"}, context=BASELINE, config=cfg)
    assert "RESPUESTA" in result["output"]      # answered, never paused


def test_a_flagged_switch_pauses_before_answering(app, graph_env):
    """When refine flags a probable different patient, the gate must interrupt BEFORE any
    answer — a cross-patient recommendation is the harm, so this one pause is allowed to block."""
    cfg = thread()
    graph_env.assess_questions = [["¿VHB?"], []]
    app.invoke({"question": "paciente gestante"}, context=BASELINE, config=cfg)
    app.invoke(Command(resume="embarazo: sí"), context=BASELINE, config=cfg)  # remember a fact

    graph_env.new_patient = True
    graph_env.assess_questions = [[]]
    paused = app.invoke({"question": "¿y en un varón de 70 años?"}, context=BASELINE, config=cfg)
    assert not paused.get("output"), "no answer must be on screen when the gate pauses"
    assert paused["__interrupt__"][0].value["confirm_new_patient"] is True


def test_confirming_a_switch_drops_the_previous_patients_data(app, graph_env):
    cfg = thread()
    graph_env.assess_questions = [["¿VHB?"], []]
    app.invoke({"question": "primera"}, context=BASELINE, config=cfg)
    app.invoke(Command(resume="sí, VHB positivo"), context=BASELINE, config=cfg)

    graph_env.new_patient = True
    graph_env.assess_questions = [[]]
    app.invoke({"question": "otro paciente distinto"}, context=BASELINE, config=cfg)
    app.invoke(Command(resume="sí"), context=BASELINE, config=cfg)   # yes, new patient

    assert "VHB positivo" not in graph_env.llm.last_user_prompt


def test_declining_a_switch_keeps_the_patient(app, graph_env):
    """An empty (or negative) reply means "same patient": the data must survive so the answer
    still uses it. Clearing on a stray Enter would silently lose the patient's history."""
    cfg = thread()
    graph_env.assess_questions = [["¿VHB?"], []]
    app.invoke({"question": "primera"}, context=BASELINE, config=cfg)
    app.invoke(Command(resume="sí, VHB positivo"), context=BASELINE, config=cfg)

    graph_env.new_patient = True
    graph_env.assess_questions = [[]]
    app.invoke({"question": "misma persona, otra duda"}, context=BASELINE, config=cfg)
    app.invoke(Command(resume=""), context=BASELINE, config=cfg)     # Enter = keep

    assert "VHB positivo" in graph_env.llm.last_user_prompt


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
