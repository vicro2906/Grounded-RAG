"""Measuring the PRODUCT: what the retrieval-only evaluation cannot see.

`build_dataset` runs a retriever plus one bare generation call, so it never exercises rephrase,
assess, the validate loop or evidence — which is where the two silent failures live. An answer
the validator wrongly rejected and an answer the model declared insufficient both reach the
doctor as "no puedo responderte" over guidelines that DO cover the question, and no number in
results/ has ever counted them.

These tests drive the real graph with stubbed boundaries, so the accounting can be trusted
before a run is paid for.
"""
import evaluation as E


# --- classifying how a run ended -------------------------------------------
# Three different causes get three different fixes, so they must not collapse into "it failed".
def test_a_delivered_answer_is_the_only_answered_outcome():
    assert E._outcome({"answer": {"sufficient_information": True},
                       "validation": {"is_valid": True}}) == "answered"


def test_the_model_declaring_the_context_insufficient_is_not_an_error():
    assert E._outcome({"answer": {"sufficient_information": False},
                       "validation": {"is_valid": True}}) == "insufficient"


def test_a_rejected_answer_is_counted_apart_from_an_absent_one():
    """The dangerous one: an answer existed and the validator threw it away."""
    assert E._outcome({"answer": {"sufficient_information": True},
                       "validation": {"is_valid": False}}) == "not_validated"


def test_a_judge_outage_is_never_read_as_a_clinical_verdict():
    assert E._outcome({"answer": {"sufficient_information": True},
                       "validation": {"is_valid": False, "error": True}}) == "validation_error"


def test_a_service_outage_outranks_everything_else():
    assert E._outcome({"technical_error": "al consultar las guías",
                       "answer": {"sufficient_information": True}}) == "technical_error"


def test_a_run_that_never_produced_an_answer_is_not_silently_answered():
    assert E._outcome({}) == "technical_error"


# --- running the shipped system --------------------------------------------
CASES = [{"question": "¿Qué TAR en coinfección por VHB?", "reference": "TDF o TAF con FTC o 3TC",
          "tier": "single_hop"},
         {"question": "¿Y la dosis?", "reference": "Según ficha técnica", "tier": "simple"}]


def test_every_question_is_asked_on_its_own_thread(graph_env):
    """The graph remembers the patient across a conversation. Reusing one thread would leak the
    first question's facts into the second and measure a conversation nobody had."""
    graph_env.assess_questions = [[], []]
    E.run_product(CASES)

    assert graph_env.refine_facts == [{}, {}], "the second question inherited a patient"


def test_a_pause_is_declined_and_the_run_still_completes(graph_env):
    """Declining is how the eval keeps measuring the unaided system — and it exercises the path
    that an empty dict used to break (the offer came back forever)."""
    graph_env.assess_questions = [["¿Hay coinfección por VHB?"], []]
    _, records, _ = E.run_product(CASES[:1])

    assert [r["outcome"] for r in records] == ["answered"]


def test_only_answered_questions_reach_the_ragas_rows(graph_env):
    """Faithfulness over «no he podido responderte» is meaningless; coverage is reported
    separately and neither number substitutes for the other."""
    graph_env.assess_questions = [[], []]
    graph_env.valid = False                       # every answer gets rejected
    dataset, records, _ = E.run_product(CASES)

    assert len(records) == 2 and all(r["outcome"] == "not_validated" for r in records)
    assert dataset is None, "a run that answered nothing is a result, not a broken harness"


def test_a_rejected_answer_is_kept_so_it_can_be_re_judged(graph_env):
    """Without the text, the false-rejection rate cannot be computed at all — and that rate is
    the whole reason this path exists."""
    graph_env.assess_questions = [[]]
    graph_env.valid = False
    _, records, _ = E.run_product(CASES[:1])

    assert records[0]["rejected_answer"], "the discarded answer must survive for the judge"
    assert records[0]["attempts"] > 1, "the retry budget must have been spent before giving up"


def test_an_outage_records_what_broke_and_not_just_that_it_did(graph_env, monkeypatch):
    """Measured for real: one probe question came back `technical_error`, and re-running it
    answered fine — a transient blip. A no-answer count that cannot separate those from a
    systematic failure is a number nobody can act on, so the step and the exception ride along
    (the graph already keeps them apart: one reaches the doctor, the other only the trace)."""
    from pipeline import nodes

    def boom(*args, **kwargs):
        raise RuntimeError("Qdrant no responde")

    monkeypatch.setattr(nodes, "retrieve_hybrid", boom)
    graph_env.assess_questions = [[]]
    _, records, _ = E.run_product(CASES[:1])

    assert records[0]["outcome"] == "technical_error"
    assert records[0]["failed_step"] and "Qdrant no responde" in records[0]["technical_detail"]


def test_questions_without_a_reference_are_not_measured(graph_env):
    """Nothing can be scored against a blank reference, and counting it as a failure would
    quietly depress every rate."""
    graph_env.assess_questions = [[]]
    _, records, _ = E.run_product([{"question": "¿algo?", "reference": "  ", "tier": "simple"},
                                   CASES[0]])

    assert [r["question"] for r in records] == [CASES[0]["question"]]


# --- the false-rejection rate ----------------------------------------------
class _Judge:
    """Stands in for the strong judge: says the rejected answer matched the reference."""

    def __init__(self, agrees):
        self.agrees = agrees
        self.calls = 0

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        return E._RejectionVerdict(agrees_with_reference=self.agrees, reason="stub")


def test_a_rejection_that_matched_the_reference_is_counted_as_false(monkeypatch):
    judge = _Judge(agrees=True)
    monkeypatch.setattr(E, "chat_model", lambda *a, **k: judge)
    records = [{"outcome": "not_validated", "rejected_answer": "TDF o TAF",
                "question": "¿?", "reference": "TDF o TAF", "reason": "stub"}]

    n_false, n_rejected, listed = E.false_rejection_rate(records)
    assert (n_false, n_rejected) == (1, 1) and listed[0]["judge_reason"] == "stub"


def test_a_rejection_the_judge_upholds_is_not_a_false_one(monkeypatch):
    monkeypatch.setattr(E, "chat_model", lambda *a, **k: _Judge(agrees=False))
    records = [{"outcome": "not_validated", "rejected_answer": "ABC",
                "question": "¿?", "reference": "TDF o TAF", "reason": "stub"}]

    assert E.false_rejection_rate(records)[:2] == (0, 1)


def test_nothing_rejected_costs_no_judge_call(monkeypatch):
    judge = _Judge(agrees=True)
    monkeypatch.setattr(E, "chat_model", lambda *a, **k: judge)

    assert E.false_rejection_rate([{"outcome": "answered", "rejected_answer": ""}]) == (0, 0, [])
    assert judge.calls == 0


def test_a_judge_outage_does_not_sink_the_whole_report(monkeypatch):
    """The rate is the last thing computed after a run that costs real money; losing the report
    to one failed judge call would waste all of it."""
    class _Broken(_Judge):
        def invoke(self, messages):
            raise RuntimeError("judge caído")

    monkeypatch.setattr(E, "chat_model", lambda *a, **k: _Broken(agrees=True))
    records = [{"outcome": "not_validated", "rejected_answer": "algo",
                "question": "¿?", "reference": "otra cosa", "reason": "stub"}]

    assert E.false_rejection_rate(records)[:2] == (0, 1)
