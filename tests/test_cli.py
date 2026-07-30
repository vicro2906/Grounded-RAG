"""The CLI loop: what the doctor actually sees in the terminal.

This is the layer the original KeyError lived in, and the one no other test reaches — the graph
tests stop at the state the graph returns, and everything after that (when the answer is
printed, how a reply is shaped, the REPL commands, the patient-switch gate) is main.py.

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


CTRL_C = object()   # a scripted line the doctor "types" as Ctrl-C


def _as_chunks(result):
    """Turn a scripted final state into the (channel, chunk) sequence the CLI now consumes.

    The CLI reads the run as a stream, so the answer no longer arrives as a final state: it
    arrives as the update of the node that WROTE it, and the pause as its own chunk. Scripting
    stays in terms of the end state, which is what the narratives below are about."""
    if result.get("output"):
        yield ("updates", {"evidence": {"output": result["output"]}})
    if result.get("__interrupt__"):
        yield ("updates", {"__interrupt__": result["__interrupt__"]})


class _FakeApp:
    """A scripted graph: streams queued results and records how it was driven. Also fakes the
    checkpointer surface the REPL uses for /paciente and /nuevo."""

    def __init__(self, results):
        self.results = list(results)
        self.payloads = []
        self.state = {"patient_facts": {}}
        self.updates = []

    def stream(self, payload, **kwargs):
        self.payloads.append(payload)
        result = self.results.pop(0)
        # A dict is an end state to expand; anything else is already a chunk sequence (used to
        # script a run that fails or is cancelled partway through).
        return _as_chunks(result) if isinstance(result, dict) else iter(result)

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
            line = next(it)
        except StopIteration:
            raise EOFError    # a real stdin raises EOFError when it runs out, not StopIteration
        if line is CTRL_C:
            raise KeyboardInterrupt
        return line
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


def test_leaving_every_question_blank_declines_the_whole_offer(monkeypatch):
    """Enter on all of them is the documented way out, and it must NOT reach the graph as {}:
    LangGraph does not accept an empty dict as a resume value, so the same offer came back
    forever. (The graph side of this is pinned in test_pipeline_flow.py.)"""
    paused = {"output": "A",
              "__interrupt__": (_Interrupt({"questions": ["¿VHB?", "¿Función renal?"]}),)}
    app = _run(monkeypatch, [paused, {"output": "A"}], ["¿pregunta?", "", "  ", "/salir"])

    assert app.payloads[1].resume == ""


# --- the conversation: one thread, several questions -----------------------
def test_several_questions_share_one_thread(monkeypatch):
    """Remembering the patient depends on every question landing on the same thread_id."""
    app = _FakeApp([{"output": "A"}, {"output": "B"}])
    seen = []

    def capture(payload, **kwargs):
        seen.append(kwargs["config"]["configurable"]["thread_id"])
        return _FakeApp.stream(app, payload, **kwargs)

    monkeypatch.setattr(main, "build_combined_graph", lambda **kwargs: app)
    monkeypatch.setattr(app, "stream", capture)
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


# --- reading the run as a stream -------------------------------------------
# The CLI depends on two properties of the stream: the pause arrives as a chunk (not as a
# returned state), and a node that did nothing yields None rather than a dict.
def test_the_pause_is_picked_up_from_the_stream(monkeypatch):
    """The interrupt is just another chunk now. If it were missed, the run would end silently
    with the refinement never offered — and no error to show for it."""
    paused = {"output": "A", "__interrupt__": (_Interrupt({"questions": ["¿VHB?"]}),)}
    app = _run(monkeypatch, [paused, {"output": "B"}], ["¿pregunta?", "sí", "/salir"])

    assert len(app.payloads) == 2 and app.payloads[1].resume == "sí"


def test_a_node_that_did_nothing_does_not_break_the_loop(monkeypatch, capsys):
    """re_retrieve no-ops on the first pass and streams None. Reading `output` off it must not
    raise — the whole answer would be lost to an AttributeError."""
    run = [("updates", {"rephrase": {"output": ""}}),
           ("updates", {"re_retrieve": None}),
           ("updates", {"evidence": {"output": "RESPUESTA"}})]
    _run(monkeypatch, [run], ["¿pregunta?", "/salir"])

    assert "RESPUESTA" in capsys.readouterr().out


def test_custom_events_do_not_disturb_the_answer(monkeypatch, capsys):
    """Progress events share the stream with the updates; they must never be mistaken for
    graph state. A transient detail also belongs to the status line, which is silent here."""
    run = [("custom", {"kind": "detail", "step": "retrieval",
                       "detail": "Valorando 24 fragmentos de las guías"}),
           ("updates", {"evidence": {"output": "RESPUESTA"}})]
    _run(monkeypatch, [run], ["¿pregunta?", "/salir"])

    out = capsys.readouterr().out
    assert out.count("RESPUESTA") == 1 and "Valorando 24" not in out


# --- the sections that were read -------------------------------------------
def test_the_sections_read_are_printed_before_the_answer(monkeypatch, capsys):
    """Unlike the step labels, this one is KEPT: it is a fact about what was consulted, it
    stays true, and it is what lets the doctor recognise the ground the answer stands on."""
    run = [("custom", {"kind": "sources", "step": "retrieval",
                       "items": ["§7.4.4 · HEPATOPATÍAS (2022)"]}),
           ("updates", {"evidence": {"output": "RESPUESTA"}})]
    _run(monkeypatch, [run], ["¿pregunta?", "/salir"])

    out = capsys.readouterr().out
    assert out.index("HEPATOPATÍAS") < out.index("RESPUESTA")


def test_a_sources_event_with_nothing_in_it_prints_nothing(monkeypatch, capsys):
    """A retrieval that came back empty already ends in the insufficient-information answer;
    an empty «Guías consultadas:» heading on top of it would just be noise."""
    run = [("custom", {"kind": "sources", "step": "retrieval", "items": []}),
           ("updates", {"evidence": {"output": "RESPUESTA"}})]
    _run(monkeypatch, [run], ["¿pregunta?", "/salir"])

    assert "consultadas" not in capsys.readouterr().out.lower()


def test_only_the_first_few_sections_are_named_in_full():
    """Eight section labels do not fit on a line, and the point is recognition, not a listing —
    the sources panel gives the full account once the answer lands."""
    line = main._format_sources([f"§{i} · SECCIÓN" for i in range(1, 9)])
    assert line.count("|") == 2 and "5" in line.split("(")[-1]


# --- Ctrl-C ----------------------------------------------------------------
def _cancelled_midway():
    """A run the doctor gives up on partway through."""
    yield ("updates", {"rephrase": {"output": ""}})
    raise KeyboardInterrupt


def test_ctrl_c_during_a_query_returns_to_the_prompt(monkeypatch, capsys):
    """~20 s is long enough to change your mind. Interrupting used to end the session with a
    traceback; now it cancels the question and the session continues on the same thread."""
    app = _run(monkeypatch, [_cancelled_midway(), {"output": "RESPUESTA B"}],
               ["¿primera?", "¿segunda?", "/salir"])

    out = capsys.readouterr().out
    assert "cancelada" in out.lower() and "RESPUESTA B" in out
    assert len(app.payloads) == 2, "the next question must still reach the graph"


def test_ctrl_c_at_the_prompt_warns_before_ending_the_session(monkeypatch, capsys):
    app = _run(monkeypatch, [{"output": "A"}], [CTRL_C, "¿pregunta?", CTRL_C, CTRL_C])

    out = capsys.readouterr().out
    assert "Ctrl-C" in out
    assert len(app.payloads) == 1, "the warned Ctrl-C must not swallow the question after it"


def test_asking_something_disarms_the_pending_exit(monkeypatch):
    """Only two CONSECUTIVE Ctrl-C end the session; a question in between resets the count."""
    app = _run(monkeypatch, [{"output": "A"}], [CTRL_C, "¿pregunta?", CTRL_C, "/salir"])
    assert len(app.payloads) == 1


# --- the progress line -----------------------------------------------------
class _Tty:
    """A stdout that claims to be a terminal, to exercise the branch pytest never takes."""

    def __init__(self):
        self.written = []

    def isatty(self):
        return True

    def write(self, text):
        self.written.append(text)

    def flush(self):
        pass


def test_the_status_line_is_silent_without_a_terminal():
    """Piped runs and the pytest capture must see exactly what they saw before streaming —
    that is what keeps every assertion in this file describing the real terminal."""
    plain = SimpleNamespace(isatty=lambda: False, write=lambda text: None, flush=lambda: None)
    status = main.Status(plain)
    status.show("  Buscando…")
    status.step("generate")
    assert not status.enabled and status.width == 0


def test_the_status_line_names_the_step_that_is_running_now(monkeypatch):
    tty = _Tty()
    main.Status(tty).step("rephrase")
    assert "Buscando en las guías" in "".join(tty.written)


def test_a_shorter_label_erases_the_longer_one_before_it():
    """The line is rewritten in place, so a short label after a long one must not leave the
    tail of the previous text on screen."""
    tty = _Tty()
    status = main.Status(tty)
    status.show("una etiqueta larguísima")
    status.show("corta")
    assert status.width == len("corta")
    assert tty.written[-1] == "\rcorta" + " " * (len("una etiqueta larguísima") - len("corta"))


def test_clearing_leaves_the_line_empty_before_the_answer():
    tty = _Tty()
    status = main.Status(tty)
    status.show("  Redactando…")
    status.clear()
    assert tty.written[-1].endswith("\r") and status.width == 0


# --- mode selection --------------------------------------------------------
def test_the_mode_argument_is_honoured(monkeypatch):
    app = _run(monkeypatch, [{"output": "A"}], ["¿pregunta?", "/salir"],
               argv=("main.py", "baseline"))
    assert app.payloads == [{"question": "¿pregunta?"}]
