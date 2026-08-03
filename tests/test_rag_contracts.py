"""The promises rag.py makes to the pipeline around each LLM call.

Two families, and both are about what happens when things are NOT ideal:

  1. DEGRADATION — every LLM step is wrapped in a try/except whose branch decides whether a
     failure blocks the doctor. Those branches are the difference between "the service hiccuped
     and the answer came out a bit generic" and "the service hiccuped and an unverified answer
     reached a clinician". They never run in a happy-path manual test.
  2. PROMPT ASSEMBLY — where each fragment comes from, and the non-citable blocks. Their exact
     wording is what separates "data that selects a branch" from "data the model will quote as
     if it were a guideline".
"""
import pytest

import corpus
import rag
from conftest import CHUNK

# The prompts are built per specialty now, so every assertion about their wording has to say
# WHICH one it is reading. The manifest default is the one the app answers from.
SPECIALTY = corpus.default_specialty()


class _Boom:
    """An LLM that fails, to exercise the except branch."""

    def invoke(self, *args, **kwargs):
        raise RuntimeError("service unavailable")


class _Canned:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.value


# --- refine: a failure must not block the question -------------------------
def test_refine_failure_lets_the_question_through(monkeypatch):
    """The rephrase step is an optimization, not a gate. If it dies, the ORIGINAL question must
    still reach retrieval — and in_domain must default to True, because refusing a clinical
    question because a helper LLM hiccuped is the worse error."""
    monkeypatch.setattr(rag, "_get_refine_llm", lambda: _Boom())

    out = rag.refine("¿Qué pauta inicio en coinfección por VHB?")

    assert out["query"] == "¿Qué pauta inicio en coinfección por VHB?"
    assert out["in_domain"] is True
    assert out["known_facts"] == {} and out["candidate_modifiers"] == []


def test_refine_falls_back_to_the_original_when_the_rewrite_is_empty(monkeypatch):
    monkeypatch.setattr(rag, "_get_refine_llm", lambda: _Canned(rag._Refined(
        in_domain=True, rewritten_query="   ", known_facts=[], candidate_modifiers=[], possible_new_patient=False)))

    assert rag.refine("pregunta original")["query"] == "pregunta original"


def test_refine_passes_the_previous_question_for_follow_up_resolution(monkeypatch):
    """A follow-up like «¿y en embarazo?» is only resolvable if refine sees the previous
    question, so it must reach the LLM's human message."""
    llm = _Canned(rag._Refined(in_domain=True, rewritten_query="¿qué TAR en embarazo?",
                               known_facts=[], candidate_modifiers=[], possible_new_patient=False))
    monkeypatch.setattr(rag, "_get_refine_llm", lambda: llm)

    rag.refine("¿y en embarazo?", prev_question="¿qué TAR de inicio se recomienda?")

    human = llm.calls[-1][-1][1]
    assert "¿qué TAR de inicio se recomienda?" in human
    assert "¿y en embarazo?" in human


def test_refine_without_a_previous_question_sends_only_the_question(monkeypatch):
    """The first turn has no prior context: the human message is just the question, not an
    empty "PREGUNTA ANTERIOR" block that would only confuse the model."""
    llm = _Canned(rag._Refined(in_domain=True, rewritten_query="reescrita",
                               known_facts=[], candidate_modifiers=[], possible_new_patient=False))
    monkeypatch.setattr(rag, "_get_refine_llm", lambda: llm)

    rag.refine("¿qué es el VIH?")

    assert llm.calls[-1][-1][1] == "¿qué es el VIH?"
    assert "PREGUNTA ANTERIOR" not in llm.calls[-1][-1][1]


def test_refine_shows_the_accumulated_facts_so_it_can_flag_a_switch(monkeypatch):
    """The patient-switch flag is only meaningful if refine sees what to contradict, so the
    accumulated facts must reach the LLM's human message."""
    llm = _Canned(rag._Refined(in_domain=True, rewritten_query="r", known_facts=[],
                               candidate_modifiers=[], possible_new_patient=True))
    monkeypatch.setattr(rag, "_get_refine_llm", lambda: llm)

    out = rag.refine("¿y en un varón?", patient_facts={"embarazo": "sí"})

    assert "embarazo: sí" in llm.calls[-1][-1][1]
    assert out["possible_new_patient"] is True


def test_refine_reports_no_switch_by_default(monkeypatch):
    llm = _Canned(rag._Refined(in_domain=True, rewritten_query="r", known_facts=[],
                               candidate_modifiers=[], possible_new_patient=False))
    monkeypatch.setattr(rag, "_get_refine_llm", lambda: llm)
    assert rag.refine("pregunta")["possible_new_patient"] is False


def test_refine_failure_flags_no_switch(monkeypatch):
    """A technical hiccup must not invent a patient switch and pause the run."""
    monkeypatch.setattr(rag, "_get_refine_llm", lambda: _Boom())
    assert rag.refine("p", patient_facts={"embarazo": "sí"})["possible_new_patient"] is False


def test_refine_parses_the_patient_data_it_screened(monkeypatch):
    monkeypatch.setattr(rag, "_get_refine_llm", lambda: _Canned(rag._Refined(
        in_domain=True, rewritten_query="reescrita",
        known_facts=["embarazo: sí", "CD4: 200"], candidate_modifiers=["funcion_renal"],
        possible_new_patient=False)))

    out = rag.refine("da igual")
    assert out["known_facts"] == {"embarazo": "sí", "CD4": "200"}
    assert out["candidate_modifiers"] == ["funcion_renal"]


# --- validate: a failure must NOT fail open --------------------------------
def test_validation_failure_is_reported_as_an_error_not_as_valid(monkeypatch):
    """The one place where degrading gracefully would be WRONG. If the judge cannot run we do
    not know whether the answer is grounded, and the pipeline must be told so (error=True) so it
    routes to the safety message instead of showing an unverified clinical answer."""
    monkeypatch.setattr(rag, "_get_validate_llm", lambda: _Boom())

    verdict = rag.validate("pregunta", {"sufficient_information": True, "answer": "…"}, "ctx")

    assert verdict["error"] is True
    assert verdict["is_valid"] is False


def test_nothing_to_validate_when_the_model_declared_insufficient_information():
    """No clinical claim was made, so there is nothing to ground — and no judge call to pay."""
    verdict = rag.validate("pregunta", {"sufficient_information": False, "answer": "…"}, "ctx")

    assert verdict["is_valid"] is True and verdict["error"] is False


def test_validation_passes_the_rejected_claims_through(monkeypatch):
    """refocus_retrieve turns these into its search query, so they must survive intact."""
    monkeypatch.setattr(rag, "_get_validate_llm", lambda: _Canned(rag._Validation(
        is_valid=False, reason="no está en el contexto",
        unsupported_claims=["la dosis es de 600 mg"])))

    verdict = rag.validate("pregunta", {"sufficient_information": True, "answer": "…"}, "ctx")

    assert verdict["unsupported_claims"] == ["la dosis es de 600 mg"]
    assert verdict["error"] is False


# --- assess: a failure must not cost the doctor the answer -----------------
def test_assess_failure_asks_nothing(monkeypatch):
    monkeypatch.setattr(rag, "_get_assess_llm", lambda: _Boom())

    assert rag.assess("pregunta", "contexto")["needs_clarification"] is False


def test_assess_ignores_a_flag_with_no_questions_behind_it(monkeypatch):
    """The model can say "I need to clarify" and then list nothing. Trusting the flag alone
    would pause the run with an empty question."""
    monkeypatch.setattr(rag, "_get_assess_llm", lambda: _Canned(rag._Assessment(
        clinically_relevant=["gestacion"], branches_on=[], already_covered=[],
        questions=[], needs_clarification=True)))

    assert rag.assess("pregunta", "contexto")["needs_clarification"] is False


def test_assess_caps_the_questions_it_returns(monkeypatch):
    monkeypatch.setattr(rag, "_get_assess_llm", lambda: _Canned(rag._Assessment(
        clinically_relevant=["a", "b", "c", "d"], branches_on=[], already_covered=[],
        questions=["¿1?", "¿2?", "¿3?", "¿4?"], needs_clarification=True)))

    assert rag.assess("p", "c", max_questions=2)["questions"] == ["¿1?", "¿2?"]


def test_assess_reasoning_is_kept_for_the_trace(monkeypatch):
    """The three reasoning fields are what let a trace show WHY a question was asked."""
    monkeypatch.setattr(rag, "_get_assess_llm", lambda: _Canned(rag._Assessment(
        clinically_relevant=["gestacion"], branches_on=["funcion_renal"],
        already_covered=["cd4"], questions=["¿Está embarazada?"], needs_clarification=True)))

    out = rag.assess("p", "c")
    assert out["clinically_relevant"] == ["gestacion"]
    assert out["branches_on"] == ["funcion_renal"]
    assert out["already_covered"] == ["cd4"]


# --- prompt assembly: where each fragment comes from ------------------------
# Seven guides spanning 2013-2025 sit in this corpus, and SYS_PROMPT asks the model to decide
# which version applies. It could not: the guide and the year reached the doctor (the sources
# panel renders them) but never the prompt. Nothing failed — the answer just arbitrated blind.
def test_each_fragment_is_headed_by_its_guide_and_year():
    _, formatted = rag.build_context([CHUNK])

    assert formatted.startswith(f"[1] {CHUNK['doc_title']} ({CHUNK['year']})\n")
    assert CHUNK["text"] in formatted


def test_the_payload_is_left_untouched_by_the_header():
    """Only the PROMPT string carries the provenance. If it leaked into the payload,
    `evidence.attribute` would fuzzy-match quotes against text the guide does not contain."""
    chunk = dict(CHUNK)
    index, _ = rag.build_context([chunk])

    assert index[1]["text"] == CHUNK["text"]
    assert CHUNK["doc_title"] not in index[1]["text"]


def test_a_fragment_without_metadata_still_gets_numbered():
    """Metadata is uneven across the guides; a missing title must not cost the fragment its
    place in the context, only its provenance line."""
    _, formatted = rag.build_context([{"text": "texto suelto"}])
    assert formatted == "[1] texto suelto\n\n"


def test_the_conflict_rule_can_actually_be_applied():
    """Rule 5 tells the model which version prevails. It is only actionable because the year is
    now in front of it — the pairing is the whole point, so both are asserted together."""
    _, formatted = rag.build_context([CHUNK, dict(CHUNK, year=2013, doc_title="Guía antigua")])

    assert "(2022)" in formatted and "(2013)" in formatted
    assert "PREVALECE LA MÁS RECIENTE" in rag.sys_prompt(SPECIALTY)
    assert "PREVALECE LA ESPECÍFICA" in rag.sys_prompt(SPECIALTY)


def test_the_conflict_rule_names_the_scopes_the_corpus_actually_covers():
    """The «specific beats general» half needs to know WHICH situations have dedicated guidance,
    and that list is derived from the manifest. It used to be typed into the prompt, so it aged
    silently every time the corpus changed."""
    prompt = rag.sys_prompt(SPECIALTY)
    for topic in corpus.topics_for(SPECIALTY):
        assert topic in prompt


def test_the_provenance_line_is_declared_non_citable():
    """It sits inside the numbered fragment, so without this the model could quote it as if it
    were guideline text."""
    assert 'no la cites nunca en "cita_textual"' in rag.sys_prompt(SPECIALTY)


def test_every_prompt_speaks_the_active_specialtys_vocabulary():
    """The four system prompts are built per specialty. What must reach ALL of them is the
    abbreviation dictionary: it is what makes a sigla and its full name the same term to the
    rewriter, the generator, the judge and the clarifier alike."""
    example = next(iter(corpus.specialty(SPECIALTY).abbreviations))
    for build in (rag.rephrase_sys, rag.sys_prompt, rag.validate_sys, rag.assess_sys):
        assert f"{example} = " in build(SPECIALTY)


def test_the_clarifier_and_the_screener_name_the_same_modifiers():
    """One profile entry yields both spellings. They were two hand-maintained lists in two
    prompts and had already drifted: the screener knew twelve slugs, the clarifier a different,
    longer prose list."""
    profile = corpus.specialty(SPECIALTY)
    screener, clarifier = rag.rephrase_sys(SPECIALTY), rag.assess_sys(SPECIALTY)
    for modifier in profile.modifiers:
        assert f'"{modifier.slug}"' in screener
        assert modifier.label in clarifier


# --- prompt assembly: the non-citable blocks -------------------------------
def test_no_blocks_when_there_is_nothing_to_add():
    """A question with no patient data and no concept map must produce the plain prompt: the
    blocks are conditional, not decoration."""
    prompt = rag.build_user_prompt("¿Pauta de inicio?", "[1] texto")

    assert "DATOS APORTADOS" not in prompt
    assert "NO APORTADOS" not in prompt
    assert "MAPA CONCEPTUAL" not in prompt


def test_supplied_patient_data_is_marked_non_citable():
    prompt = rag.build_user_prompt("¿Pauta?", "[1] texto",
                                   clinical_facts={"coinfeccion_VHB": "sí"})

    assert "DATOS APORTADOS POR EL MÉDICO" in prompt
    assert "coinfeccion_VHB: sí" in prompt
    assert "NO citable" in prompt
    assert "nunca como fuente ni en cita_textual" in prompt


def test_missing_patient_data_is_named_and_forbidden_to_assume():
    """This block is what makes the answer-first design safe: told the datum is unknown, the
    model presents the branches instead of silently choosing one."""
    prompt = rag.build_user_prompt("¿Pauta?", "[1] texto",
                                   open_questions=["¿Hay coinfección por VHB?"])

    assert "DATOS DEL PACIENTE NO APORTADOS" in prompt
    assert "¿Hay coinfección por VHB?" in prompt
    assert "no los supongas" in prompt


def test_concept_map_sits_closest_to_the_question():
    """Ordering is deliberate: the map is printed worst-to-best and placed last, so the most
    reliable path lands next to the question, where long-context models attend best."""
    prompt = rag.build_user_prompt("¿Pauta?", "[1] texto",
                                   clinical_facts={"embarazo": "sí"},
                                   open_questions=["¿Función renal?"],
                                   concept_map="TAF -> VHB")

    assert (prompt.index("DATOS APORTADOS")
            < prompt.index("NO APORTADOS")
            < prompt.index("MAPA CONCEPTUAL")
            < prompt.index("PREGUNTA CLÍNICA"))


def test_concept_map_is_marked_non_citable():
    prompt = rag.build_user_prompt("¿Pauta?", "[1] texto", concept_map="TAF -> VHB")

    assert "NO citable" in prompt
    assert "no lo cites" in prompt


@pytest.mark.parametrize("facts", [None, {}, {"": "vacío"}])
def test_empty_patient_data_adds_no_block(facts):
    assert "DATOS APORTADOS" not in rag.build_user_prompt("¿P?", "ctx", clinical_facts=facts)
