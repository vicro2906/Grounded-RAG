"""Entry point: compiles the LangGraph app(s) and runs the interactive CLI.

The pipeline itself lives in the `pipeline/` package. This module only wires up runtime
concerns (UTF-8 console, LangSmith tracing, model warm-up), compiles the graphs that
langgraph.json exposes to Studio, and offers a small CLI.

Choosing the retrieval mode:
    - CLI:    the default is RETRIEVAL_MODE (env var, falls back to "graph"); `python main.py
              iterative` forces one for a quick manual test.
    - Studio: the combined graph exposes a `retrieval_mode` dropdown in the run config panel,
              so all three architectures can be picked and traced live.

Persistence: the graphs registered in langgraph.json are compiled WITHOUT a checkpointer
because the LangGraph platform injects its own. The CLI is a plain embedding, so it compiles
its own instance with an in-memory checkpointer — without one, `clarify`'s `interrupt()` pauses
the run and there is no way to resume it (invoke returns `__interrupt__` and no answer).

Visible work: a clinical question takes 12-25 s across refine -> retrieval -> assess ->
generate -> validate, and the CLI used to print nothing at all until the answer was ready. It
now READS THE RUN AS A STREAM and paints a self-erasing status line from the node updates, so
the wait shows what it is doing. What is deliberately NOT streamed is the clinical text: the
validator runs after generation and can reject the answer, and retracting something a doctor
has already read is the harm the pipeline exists to prevent.

Tracing (LangSmith): enabled automatically ONLY if LANGSMITH_API_KEY is set; each run and each
OpenAI call then shows up in the LANGSMITH_PROJECT project, with the retrieval mode recorded as
a run tag and metadata so the architectures are filterable.

Usage:
    python main.py             # interactive, uses RETRIEVAL_MODE (default "graph")
    python main.py iterative   # interactive, forces the iterative mode for this run
"""
import os
import sys
import threading
import uuid

# The Windows console defaults to cp1252 and breaks accents/boxes. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from dotenv import load_dotenv
load_dotenv()

# LangSmith tracing: enable only when there is an API key (otherwise stays out of the way).
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "chatbot_vih")

import rag  # creates the OpenAI/Qdrant clients on import

# rag's functions read `rag.client` at call time, so replacing it here with the LangSmith-
# wrapped client is enough to trace retrieval and the embeddings inside the graph run.
if os.environ.get("LANGSMITH_API_KEY"):
    try:
        from langsmith.wrappers import wrap_openai
        rag.client = wrap_openai(rag.client)
    except Exception:
        pass

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from progress import read_chunk

import corpus
from pipeline import (build_graph, build_combined_graph, refinement_reply,
                      RETRIEVAL_MODE, VALID_MODES, MSG_SPECIALTY_CURRENT,
                      MSG_SPECIALTY_CHANGED, MSG_SPECIALTY_UNKNOWN,
                      MSG_REFINE_OFFER, msg_intro, MSG_CLI_HELP, MSG_NEW_PATIENT,
                      MSG_NO_PATIENT_DATA, MSG_PATIENT_HEADER, MSG_CONFIRM_NEW_PATIENT,
                      MSG_CONFIRM_NEW_PATIENT_ASK, MSG_STEP_INITIAL, MSG_STEP_RESUMED,
                      MSG_STEP_LABELS, MSG_STEP_SOURCES, MSG_STEP_SOURCES_MORE,
                      STEP_SOURCES_SHOWN, MSG_CANCELLED, MSG_EXIT_HINT)

# What the run is asked to report while it works. "updates" (one chunk per node, on completion)
# is what drives the progress line; "custom" carries the finer-grained events the nodes emit
# from inside a long step. The clinical TEXT is never streamed: `validate` runs after `generate`
# and can reject the answer, and showing a doctor something that is then retracted is exactly
# the harm this pipeline is built to avoid.
STREAM_MODES = ["updates", "custom"]

# Compiled graphs (registered in langgraph.json). The three dedicated ones are the expanded
# "teaching" views; `app` is the combined graph with the live retrieval_mode dropdown.
app_baseline  = build_graph("baseline")
app_iterative = build_graph("iterative")
app_graph     = build_graph("graph")
app = build_combined_graph()

# Preload the local models (reranker + BM25) in the background so the first real query does not
# pay their ~3.5s load. Daemon thread + warmup() swallows its own errors, so it never blocks
# import or shutdown. LightRAG stays lazy — only loaded if graph mode actually runs.
threading.Thread(target=rag.warmup, daemon=True).start()


def _confirm_new_patient(value) -> str:
    """The blocking patient-switch gate (before any answer): show the remembered data and ask
    whether to start fresh. Only an explicit «sí» clears it — Enter keeps the patient."""
    print(f"\n{MSG_CONFIRM_NEW_PATIENT}")
    print(_format_patient_facts(value.get("facts")))
    return input(f"{MSG_CONFIRM_NEW_PATIENT_ASK} ").strip()


def _collect_clarifications(value) -> dict | str:
    """Offer the refinement the graph paused on (AFTER the answer is on screen). Every question
    is optional — an empty reply declines it, and leaving them all blank declines the offer.
    `refinement_reply` shapes the answers into what the resume must carry."""
    questions = list(value.get("questions", []))
    print(f"\n{MSG_REFINE_OFFER}")
    return refinement_reply(questions, {q: input(f"  · {q} ").strip() for q in questions})


def _resume_for(interrupts):
    """One pause at a time: dispatch on its kind — the confirm gate asks yes/no BEFORE the
    answer, the refinement offers questions AFTER it."""
    value = interrupts[0].value or {}
    if value.get("confirm_new_patient"):
        return _confirm_new_patient(value)
    return _collect_clarifications(value)


def _format_patient_facts(facts: dict | None) -> str:
    """One line per remembered datum, for /paciente and for the patient-switch gate."""
    items = {k: v for k, v in (facts or {}).items() if k}
    if not items:
        return MSG_NO_PATIENT_DATA
    lines = "\n".join(f"  · {k}: {v}" if v else f"  · {k}" for k, v in items.items())
    return f"{MSG_PATIENT_HEADER}\n{lines}"


def _format_sources(items: list) -> str:
    """The sections a retrieval read, on one line: the first few in full, the rest as a count."""
    shown = items[:STEP_SOURCES_SHOWN]
    rest = len(items) - len(shown)
    line = " | ".join(shown)
    return f"{line} ({MSG_STEP_SOURCES_MORE.format(n=rest)})" if rest > 0 else line


class Status:
    """The one-line progress indicator that replaced ~20 s of silence.

    It rewrites a single line in place, so the transcript keeps only the answer. A NO-OP when
    stdout is not a tty: piped runs, LangSmith-driven runs and the pytest capture then see
    exactly the output they saw before the CLI streamed anything, which is what makes the
    existing terminal tests still describe the terminal."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self.width = 0

    def show(self, text: str) -> None:
        if not self.enabled or not text:
            return
        # Pad to whatever was there before, or the tail of a longer previous label survives.
        self.stream.write("\r" + text + " " * max(0, self.width - len(text)))
        self.stream.flush()
        self.width = len(text)

    def step(self, node: str) -> None:
        """A node finished; say what is running now (see MSG_STEP_LABELS)."""
        self.show(f"  {MSG_STEP_LABELS[node]}…" if node in MSG_STEP_LABELS else "")

    def event(self, event: dict) -> None:
        """A ProgressEvent from INSIDE a long step (see progress.py). The sections read are
        KEPT — they are a fact about what was consulted, not a passing state — while a sub-step
        detail is transient like any other label."""
        if event.get("kind") == "sources":
            items = event.get("items") or []
            if items:
                self.keep(f"  {MSG_STEP_SOURCES} {_format_sources(items)}")
        elif event.get("detail"):
            self.show(f"  {event['detail']}…")

    def keep(self, text: str) -> None:
        """Print a line that stays in the transcript, without leaving the status line wedged
        into it."""
        self.clear()
        print(text)

    def clear(self) -> None:
        """Erase the line before anything else is printed, so no status text ends up wedged
        into the answer."""
        if self.enabled and self.width:
            self.stream.write("\r" + " " * self.width + "\r")
            self.stream.flush()
        self.width = 0


def _answer_question(app_cli, question, mode, config, status):
    """Run one question to completion, handling the (optional) refinement pause. The answer is
    printed as soon as it exists — the pause happens WITH it already on screen — and only when
    it changed, so declining the offer never reprints the same text.

    Consumed as a STREAM rather than a single invoke, so the work is visible while it happens.
    The dedup is equivalent: `output` is only ever written by rephrase (to ""), out_of_domain,
    technical_error, fallback and evidence, and exactly one of those runs per stretch — so the
    last non-empty output of the stream is the output of the final state."""
    payload = {"question": question}
    shown = None
    waiting = MSG_STEP_INITIAL        # what to show until the stretch's first chunk arrives
    while True:
        interrupts = None
        status.show(f"  {waiting}…")
        try:
            for channel, chunk in app_cli.stream(payload, context={"retrieval_mode": mode},
                                                 config=config, stream_mode=STREAM_MODES):
                for kind, value in read_chunk(channel, chunk):
                    if kind == "step":
                        status.step(value)
                    elif kind == "progress":
                        status.event(value)
                    elif kind == "interrupt":
                        interrupts = value
                    elif kind == "output" and value != shown:
                        status.clear()
                        print(f"\n{value}")
                        shown = value
        except KeyboardInterrupt:
            # Abandoning a paused run is safe: the next question starts from the top on the same
            # thread, the orphaned pause is discarded and the patient data survives.
            status.clear()
            print(f"\n{MSG_CANCELLED}")
            return
        finally:
            status.clear()

        if not interrupts:
            return
        payload = Command(resume=_resume_for(interrupts))
        waiting = MSG_STEP_RESUMED


def _active_specialty(app_cli, config) -> str:
    """What the thread is actually answering from: the pinned state if a question has run,
    otherwise what the session was started with."""
    return (app_cli.get_state(config).values.get("specialty")
            or config["configurable"].get("specialty")
            or corpus.default_specialty())


def _specialty_from_argv() -> str:
    """`--specialty <id>` / `--specialty=<id>`. An explicit flag rather than a bare positional:
    the mode is already positional, and a specialty named like a retrieval mode would otherwise
    be ambiguous. An unknown value resolves to the manifest default instead of failing."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.startswith("--specialty="):
            return corpus.resolve_specialty(arg.split("=", 1)[1])
        if arg == "--specialty" and i + 1 < len(args):
            return corpus.resolve_specialty(args[i + 1])
    return corpus.default_specialty()


def main_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in VALID_MODES else RETRIEVAL_MODE
    specialty = _specialty_from_argv()
    # Own instance WITH persistence: the graphs above are for the platform (which injects its
    # own checkpointer), this one is a plain embedding and must carry its own — both to resume
    # the clarification interrupt AND to remember the patient across questions. Compiled here so
    # importing main.py for Studio never pays for it.
    app_cli = build_combined_graph(checkpointer=InMemorySaver())
    # ONE thread for the whole conversation: the checkpointer keys the accumulated patient data
    # (and any paused run) by thread_id, so every question in this session shares it.
    # `specialty` rides in `configurable` rather than in the first payload: node_rephrase pins it
    # into the state on turn 1, so it survives the session and /especialidad can change it there.
    config = {"configurable": {"thread_id": uuid.uuid4().hex, "specialty": specialty},
              "tags": [f"mode:{mode}", f"specialty:{specialty}"],
              "metadata": {"retrieval_mode": mode, "specialty": specialty}}  # LangSmith

    status = Status()
    print(msg_intro(specialty))
    quit_armed = False        # one Ctrl-C at the prompt warns, a second one ends the session
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:      # piped input exhausted / Ctrl-Z — end cleanly, not with a traceback
            break
        except KeyboardInterrupt:
            if quit_armed:
                break
            quit_armed = True
            print(f"\n{MSG_EXIT_HINT}")
            continue
        quit_armed = False
        if not text:
            continue

        low = text.lower()
        if low in ("/salir", "/exit", "/quit", "salir"):
            break
        if low in ("/ayuda", "/help"):
            print(MSG_CLI_HELP)
            continue
        if low in ("/paciente", "/datos"):
            facts = app_cli.get_state(config).values.get("patient_facts")
            print(_format_patient_facts(facts))
            continue
        if low in ("/nuevo", "/reset"):
            # Clear the session-scoped state directly on the thread — a new patient inherits
            # neither the previous one's facts NOR its conversation (so a follow-up does not
            # resolve against the old patient's question). Per-question state resets each turn.
            app_cli.update_state(config, {"patient_facts": {}, "prev_question": ""})
            print(MSG_NEW_PATIENT)
            continue
        if low.startswith("/especialidad"):
            asked = text.split(maxsplit=1)[1].strip() if " " in text else ""
            available = ", ".join(corpus.specialties())
            if not asked:
                print(MSG_SPECIALTY_CURRENT.format(
                    name=corpus.specialty(_active_specialty(app_cli, config)).display_name,
                    available=available))
            elif asked not in corpus.specialties():
                print(MSG_SPECIALTY_UNKNOWN.format(asked=asked, available=available))
            else:
                # The PATIENT survives a specialty change on purpose: the same person crosses
                # areas, and dropping their renal function because the doctor switched guidelines
                # would lose clinical data the answers depend on.
                app_cli.update_state(config, {"specialty": asked})
                config["configurable"]["specialty"] = asked
                print(MSG_SPECIALTY_CHANGED.format(name=corpus.specialty(asked).display_name))
            continue

        _answer_question(app_cli, text, mode, config, status)


if __name__ == "__main__":
    main_cli()
