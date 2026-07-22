"""Shared fixtures.

The point of these fixtures is to run the REAL graph — real routing, real reducers, real
interrupt/resume — with every LLM and network call replaced by a stub, so the wiring can be
tested offline and for free. Only the boundaries are faked (refine / assess / validate /
generation / retrieval); everything the pipeline actually decides stays real.
"""
import pytest

from pipeline.generation import ClinicalAnswer


CHUNK = {
    "chunk_id": "c1",
    "doc_title": "Documento de consenso de GeSIDA sobre TAR",
    "year": 2022,
    "section_number": "7.4.4",
    "heading": "7.4.4. HEPATOPATÍAS",
    "section_path": ["7. SITUACIONES ESPECIALES", "7.4.4. HEPATOPATÍAS"],
    "text": ("7. SITUACIONES ESPECIALES > 7.4.4. HEPATOPATÍAS\n\n"
             "En pacientes con coinfección por VHB se recomienda iniciar precozmente un TAR "
             "que incluya TDF o TAF y FTC o 3TC (A-I)."),
}

ANSWER_QUOTE = "iniciar precozmente un TAR que incluya TDF o TAF y FTC o 3TC"


def _answer(sufficient: bool = True) -> ClinicalAnswer:
    return ClinicalAnswer(
        sufficient_information=sufficient,
        answer="En coinfección por VHB se inicia TAR con TDF o TAF y FTC o 3TC.",
        sources_used=[{"ref": 1, "quote": ANSWER_QUOTE}] if sufficient else [],
        follow_up_questions=["¿Y si hay cirrosis?"] if sufficient else [],
    )


class StubLLM:
    """Stands in for the structured generation LLM, recording the prompts it was called with
    so a test can assert on what the pipeline actually sent."""

    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _answer()

    @property
    def last_user_prompt(self) -> str:
        return self.calls[-1][-1][1]


@pytest.fixture
def graph_env(monkeypatch):
    """Patch every boundary of the combined graph and hand the test the knobs it needs.

    Returns an object whose attributes drive the stubs: `assess_questions` (list of lists — one
    entry per assess call, so a test can script "ask once, then stop"), `valid` (what the
    validator returns) and `llm` (the recording generation stub).
    """
    from pipeline import nodes

    class Env:
        assess_questions: list[list[str]] = []
        valid = True
        llm = StubLLM()
        assess_calls = 0

    env = Env()

    def fake_refine(question):
        return {"query": f"{question} (reescrita)", "in_domain": "fuera" not in question,
                "known_facts": {}, "candidate_modifiers": ["coinfeccion_VHB"]}

    def fake_assess(question, context, **kwargs):
        i, env.assess_calls = env.assess_calls, env.assess_calls + 1
        qs = env.assess_questions[i] if i < len(env.assess_questions) else []
        return {"needs_clarification": bool(qs), "questions": qs, "branches_on": [],
                "clinically_relevant": [], "already_covered": []}

    def fake_validate(question, answer, formatted_context):
        return {"is_valid": env.valid, "error": False, "reason": "stub",
                "unsupported_claims": [] if env.valid else ["afirmación sin respaldo"]}

    monkeypatch.setattr(nodes, "refine", fake_refine)
    monkeypatch.setattr(nodes, "assess", fake_assess)
    monkeypatch.setattr(nodes, "validate", fake_validate)
    monkeypatch.setattr(nodes, "structured_llm", env.llm)
    monkeypatch.setattr(nodes, "retrieve_hybrid", lambda *a, **k: [CHUNK])
    monkeypatch.setattr(nodes, "rerank", lambda *a, **k: [CHUNK])
    return env
