"""The CLI loop: what the doctor actually sees in the terminal.

This is the layer the original KeyError lived in, and the one no other test reaches — the graph
tests stop at the state the graph returns, and everything after that (when the answer is
printed, how a reply is shaped, the REPL commands, the always-visible patient line) is main.py.

The graph is replaced by a scripted fake, so these tests describe the terminal experience:
answer first, ask second, never print the same answer twice, and remember the patient across
questions until /nuevo.
"""
from types import SimpleNamespace

import main


class _Interrupt:
    """Stands in for langgraph's Interrupt: the CLI only reads `.value`."""

    def __init__(self, value):
        self.value = value


class _FakeApp:
    """A scripted graph: returns queued results and records how it was driven. Also fakes the
    checkpointer surface the REPL uses for /paciente and /nuevo."""

    def __init__(self, results):
        self.results = list(results)
        self.payloads = []
        self.state = {"patient_facts": {}}
        self.updates = []

    def invoke(self, payload, **kwargs):
        self.payloads.append(payload)
        return self.results.pop(0)

    def get_state(self, config):
        return SimpleNamespace(values=self.state)

    def update_state(self, config, values):
        self.updates.append(values)
        self.state.update(values)


def _fake_input(lines):
    """Like the real input(): it ECHOES the prompt to stdout before reading. That matters —
    the clarifying questions reach the terminal only as input() prompts, so a fake that
    swallowed them would hide half of what the doctor sees."""
    it = iter(lines)

    def _input(prompt=""):
        print(prompt, end="")
        try:
            return next(it)
        except StopIteration:
            raise EOFError    # a real stdin raises EOFError when it runs out, not StopIteration
    return _input


def _run(monkeypatch, results, lines, argv=("main.py",)):
    """Drive main_cli() with a scripted graph and a scripted stdin. `lines` is every line the
    doctor types, top-level prompts and refinement answers alike, ending in a way that exits."""
    app = _FakeApp(results)
    monkeypatch.setattr(main, "build_combined_graph", lambda **kwargs: app)
    monkeypatch.setattr("builtins.input", _fake_input(lines))
    monkeypatch.setattr(main.sys, "argv", list(argv))
    main.main_cli()
    return app


# --- answering, and printing it once ---------------------------------------
def test_plain_answer_is_printed_once(monkeypatch, capsys):
    _run(monkeypatch, [{"output": "RESPUESTA A"}], ["¿pregunta?", "/salir"])

    assert capsys.readouterr().out.count("RESPUESTA A") == 1


def test_the_answer_is_printed_before_the_refinement_is_offered(monkeypatch, capsys):
    """Answer-first, seen from the terminal: the question must not appear above the answer."""
    paused = {"output": "RESPUESTA A",
              "__interrupt__": (_Interrupt({"questions": ["¿Hay coinfección por VHB?"],
                                            "optional": True}),)}
    _run(monkeypatch, [paused, {"output": "RESPUESTA A"}], ["¿pregunta?", "", "/salir"])

    out = capsys.readouterr().out
    assert out.index("RESPUESTA A") < out.index("¿Hay coinfección por VHB?")


def test_declining_does_not_reprint_the_same_answer(monkeypatch, capsys):
    """Enter declines. The run ends on the answer already on screen — printing it a second time
    would read as if something new had happened."""
    paused = {"output": "RESPUESTA A",
              "__interrupt__": (_Interrupt({"questions": ["¿VHB?"], "optional": True}),)}
    app = _run(monkeypatch, [paused, {"output": "RESPUESTA A"}], ["¿pregunta?", "", "/salir"])

    assert capsys.readouterr().out.count("RESPUESTA A") == 1
    assert app.payloads[:2], "the empty reply must still resume the paused run"


def test_a_refined_answer_is_printed_when_it_differs(monkeypatch, capsys):
    paused = {"output": "RESPUESTA A",
              "__interrupt__": (_Interrupt({"questions": ["¿VHB?"], "optional": True}),)}
    _run(monkeypatch, [paused, {"output": "RESPUESTA B (refinada)"}],
         ["¿pregunta?", "sí", "/salir"])

    out = capsys.readouterr().out
    assert "RESPUESTA A" in out and "RESPUESTA B (refinada)" in out


# --- shaping the refinement reply ------------------------------------------
def test_a_single_question_resumes_with_plain_text(monkeypatch):
    """node_refine_offer keys a lone text answer by the question it answers, so the CLI must
    send the text itself — wrapping it in a dict would land it under the wrong key."""
    paused = {"output": "A", "__interrupt__": (_Interrupt({"questions": ["¿VHB?"]}),)}
    app = _run(monkeypatch, [paused, {"output": "B"}],
               ["¿pregunta?", "sí, VHB positivo", "/salir"])

    assert app.payloads[1].resume == "sí, VHB positivo"


def test_several_questions_resume_with_a_dict_dropping_the_blanks(monkeypatch):
    """With more than one question, free text could not be attributed to any of them, so the
    CLI pairs each answer with its question and drops the ones left empty."""
    paused = {"output": "A",
              "__interrupt__": (_Interrupt({"questions": ["¿VHB?", "¿Función renal?"]}),)}
    app = _run(monkeypatch, [paused, {"output": "B"}], ["¿pregunta?", "sí", "   ", "/salir"])

    assert app.payloads[1].resume == {"¿VHB?": "sí"}


# --- the conversation: one thread, several questions -----------------------
def test_several_questions_share_one_thread(monkeypatch):
    """Remembering the patient depends on every question landing on the same thread_id."""
    app = _FakeApp([{"output": "A"}, {"output": "B"}])
    seen = []

    def capture(payload, **kwargs):
        seen.append(kwargs["config"]["configurable"]["thread_id"])
        return _FakeApp.invoke(app, payload, **kwargs)

    monkeypatch.setattr(main, "build_combined_graph", lambda **kwargs: app)
    monkeypatch.setattr(app, "invoke", capture)
    monkeypatch.setattr("builtins.input", _fake_input(["primera", "segunda", "/salir"]))
    monkeypatch.setattr(main.sys, "argv", ["main.py"])
    main.main_cli()

    assert len(seen) == 2 and seen[0] == seen[1]


def test_exit_command_stops_the_loop_without_asking_the_graph(monkeypatch):
    app = _run(monkeypatch, [], ["/salir"])
    assert app.payloads == []


def test_a_blank_line_is_ignored(monkeypatch):
    app = _run(monkeypatch, [{"output": "A"}], ["", "   ", "¿pregunta?", "/salir"])
    assert len(app.payloads) == 1


def test_eof_ends_the_session_cleanly(monkeypatch):
    """Piped input running out (or Ctrl-Z) must end the REPL, not raise StopIteration."""
    _run(monkeypatch, [{"output": "A"}], ["¿pregunta?"])   # no /salir: input() runs dry


# --- remembering and forgetting the patient --------------------------------
def test_new_patient_clears_the_remembered_data(monkeypatch, capsys):
    app = _FakeApp([{"output": "A"}])
    app.state = {"patient_facts": {"coinfeccion_VHB": "sí"}}
    _run_with(monkeypatch, app, ["¿pregunta?", "/nuevo", "/salir"])

    assert app.updates == [{"patient_facts": {}, "prev_question": ""}]
    assert "empiezo de cero" in capsys.readouterr().out.lower()


def test_patient_command_shows_the_remembered_data(monkeypatch, capsys):
    app = _FakeApp([])
    app.state = {"patient_facts": {"coinfeccion_VHB": "sí", "CD4": "200"}}
    _run_with(monkeypatch, app, ["/paciente", "/salir"])

    out = capsys.readouterr().out
    assert "coinfeccion_VHB: sí" in out and "CD4: 200" in out


def test_patient_command_with_nothing_remembered(monkeypatch, capsys):
    app = _FakeApp([])
    _run_with(monkeypatch, app, ["/paciente", "/salir"])
    assert "Todavía no" in capsys.readouterr().out


def _run_with(monkeypatch, app, lines):
    """Variant of _run for tests that need to seed the fake app's remembered state first."""
    monkeypatch.setattr(main, "build_combined_graph", lambda **kwargs: app)
    monkeypatch.setattr("builtins.input", _fake_input(lines))
    monkeypatch.setattr(main.sys, "argv", ["main.py"])
    main.main_cli()


# --- the blocking patient-switch confirmation ------------------------------
def test_the_switch_confirmation_is_asked_before_any_answer(monkeypatch, capsys):
    """The confirm interrupt has no answer yet, so the CLI must show the warning and remembered
    data and ask — not print an (absent) answer."""
    paused = {"__interrupt__": (_Interrupt({"confirm_new_patient": True,
                                            "facts": {"embarazo": "sí"},
                                            "question": "¿y en un varón?"}),)}
    app = _run(monkeypatch, [paused, {"output": "RESPUESTA"}],
               ["¿y en un varón?", "sí", "/salir"])

    out = capsys.readouterr().out
    assert "paciente distinto" in out and "embarazo: sí" in out
    assert app.payloads[1].resume == "sí"      # the yes/no reaches the graph verbatim


# --- mode selection --------------------------------------------------------
def test_the_mode_argument_is_honoured(monkeypatch):
    app = _run(monkeypatch, [{"output": "A"}], ["¿pregunta?", "/salir"],
               argv=("main.py", "baseline"))
    assert app.payloads == [{"question": "¿pregunta?"}]
