"""The CLI loop: what the doctor actually sees in the terminal.

This is the layer the KeyError lived in, and the one no other test reaches — the graph tests
stop at the state the graph returns, and everything after that (when the answer is printed,
whether the reply is shaped the way node_refine_offer expects) is main.py's job.

The graph is replaced by a scripted fake, so these tests describe the terminal experience:
answer first, ask second, and never print the same answer twice.
"""
import main


class _Interrupt:
    """Stands in for langgraph's Interrupt: the CLI only reads `.value`."""

    def __init__(self, value):
        self.value = value


class _FakeApp:
    """Returns a scripted sequence of results, recording what it was invoked with."""

    def __init__(self, results):
        self.results = list(results)
        self.payloads = []

    def invoke(self, payload, **kwargs):
        self.payloads.append(payload)
        return self.results.pop(0)


def _fake_input(typed):
    """Like the real input(): it ECHOES the prompt to stdout before reading. That matters here
    — the clarifying questions reach the terminal only as input() prompts, so a fake that
    swallowed them would hide half of what the doctor sees."""
    def _input(prompt=""):
        print(prompt, end="")
        return next(typed)
    return _input


def _run(monkeypatch, results, answers):
    """Drive main_cli() with a scripted graph and scripted keyboard input."""
    app = _FakeApp(results)
    typed = iter(["¿Qué pauta inicio?"] + answers)
    monkeypatch.setattr(main, "build_combined_graph", lambda **kwargs: app)
    monkeypatch.setattr("builtins.input", _fake_input(typed))
    monkeypatch.setattr(main.sys, "argv", ["main.py"])
    main.main_cli()
    return app


def test_plain_answer_is_printed_once(monkeypatch, capsys):
    _run(monkeypatch, [{"output": "RESPUESTA A"}], answers=[])

    assert capsys.readouterr().out.count("RESPUESTA A") == 1


def test_the_answer_is_printed_before_the_refinement_is_offered(monkeypatch, capsys):
    """Answer-first, seen from the terminal: the question must not appear above the answer."""
    paused = {"output": "RESPUESTA A",
              "__interrupt__": (_Interrupt({"questions": ["¿Hay coinfección por VHB?"],
                                            "optional": True}),)}
    _run(monkeypatch, [paused, {"output": "RESPUESTA A"}], answers=[""])

    out = capsys.readouterr().out
    assert out.index("RESPUESTA A") < out.index("¿Hay coinfección por VHB?")


def test_declining_does_not_reprint_the_same_answer(monkeypatch, capsys):
    """Enter declines. The run ends on the answer already on screen — printing it a second time
    would read as if something new had happened."""
    paused = {"output": "RESPUESTA A",
              "__interrupt__": (_Interrupt({"questions": ["¿VHB?"], "optional": True}),)}
    app = _run(monkeypatch, [paused, {"output": "RESPUESTA A"}], answers=[""])

    assert capsys.readouterr().out.count("RESPUESTA A") == 1
    assert len(app.payloads) == 2, "the empty reply must still resume the paused run"


def test_a_refined_answer_is_printed_when_it_differs(monkeypatch, capsys):
    paused = {"output": "RESPUESTA A",
              "__interrupt__": (_Interrupt({"questions": ["¿VHB?"], "optional": True}),)}
    _run(monkeypatch, [paused, {"output": "RESPUESTA B (refinada)"}], answers=["sí"])

    out = capsys.readouterr().out
    assert "RESPUESTA A" in out and "RESPUESTA B (refinada)" in out


def test_a_single_question_resumes_with_plain_text(monkeypatch):
    """node_refine_offer keys a lone text answer by the question it answers, so the CLI must
    send the text itself — wrapping it in a dict would land it under the wrong key."""
    paused = {"output": "A", "__interrupt__": (_Interrupt({"questions": ["¿VHB?"]}),)}
    app = _run(monkeypatch, [paused, {"output": "B"}], answers=["sí, VHB positivo"])

    assert app.payloads[1].resume == "sí, VHB positivo"


def test_several_questions_resume_with_a_dict_dropping_the_blanks(monkeypatch):
    """With more than one question, free text could not be attributed to any of them, so the
    CLI pairs each answer with its question and drops the ones left empty."""
    paused = {"output": "A",
              "__interrupt__": (_Interrupt({"questions": ["¿VHB?", "¿Función renal?"]}),)}
    app = _run(monkeypatch, [paused, {"output": "B"}], answers=["sí", "   "])

    assert app.payloads[1].resume == {"¿VHB?": "sí"}


def test_the_whole_run_shares_one_thread(monkeypatch):
    """Resuming reads the paused state back by thread_id: a new id per invoke would look for a
    run that does not exist there."""
    paused = {"output": "A", "__interrupt__": (_Interrupt({"questions": ["¿VHB?"]}),)}
    app = _FakeApp([paused, {"output": "B"}])
    threads = []

    def capture(payload, **kwargs):
        threads.append(kwargs["config"]["configurable"]["thread_id"])
        return _FakeApp.invoke(app, payload, **kwargs)

    typed = iter(["¿Pauta?", "sí"])
    monkeypatch.setattr(main, "build_combined_graph", lambda **kwargs: app)
    monkeypatch.setattr(app, "invoke", capture)
    monkeypatch.setattr("builtins.input", _fake_input(typed))
    monkeypatch.setattr(main.sys, "argv", ["main.py"])
    main.main_cli()

    assert len(threads) == 2 and threads[0] == threads[1]


def test_the_mode_argument_is_honoured(monkeypatch):
    app = _FakeApp([{"output": "A"}])
    typed = iter(["¿Pauta?"])
    monkeypatch.setattr(main, "build_combined_graph", lambda **kwargs: app)
    monkeypatch.setattr("builtins.input", _fake_input(typed))
    monkeypatch.setattr(main.sys, "argv", ["main.py", "baseline"])
    main.main_cli()

    assert app.payloads == [{"question": "¿Pauta?"}]
