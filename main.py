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

from pipeline import (build_graph, build_combined_graph, RETRIEVAL_MODE, VALID_MODES,
                      MSG_REFINE_OFFER, MSG_CLI_INTRO, MSG_CLI_HELP, MSG_NEW_PATIENT,
                      MSG_NO_PATIENT_DATA, MSG_PATIENT_HEADER)

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


def _collect_clarifications(interrupts) -> dict | str:
    """Offer the refinement the graph paused on and shape the reply the way node_refine_offer
    expects: plain text for a single question, {question: answer} for several.

    The answer is ALREADY printed by the time this runs, so every question here is optional:
    an empty reply (Enter) declines it and the run ends with the answer the doctor has."""
    questions = [q for i in interrupts for q in (i.value or {}).get("questions", [])]
    print(f"\n{MSG_REFINE_OFFER}")
    answers = {q: input(f"  · {q} ").strip() for q in questions}
    if len(answers) == 1:
        return next(iter(answers.values()))
    return {q: a for q, a in answers.items() if a}


def _format_patient_facts(facts: dict | None) -> str:
    """One line per remembered datum, for /paciente and the always-visible header."""
    items = {k: v for k, v in (facts or {}).items() if k}
    if not items:
        return MSG_NO_PATIENT_DATA
    lines = "\n".join(f"  · {k}: {v}" if v else f"  · {k}" for k, v in items.items())
    return f"{MSG_PATIENT_HEADER}\n{lines}"


def _answer_question(app_cli, question, mode, config):
    """Run one question to completion, handling the (optional) refinement pause. The answer is
    printed as soon as it exists — the pause happens WITH it already on screen — and only when
    it changed, so declining the offer never reprints the same text."""
    payload = {"question": question}
    shown = None
    while True:
        result = app_cli.invoke(payload, context={"retrieval_mode": mode}, config=config)
        output = result.get("output")
        if output and output != shown:
            print(f"\n{output}")
            shown = output
        interrupts = result.get("__interrupt__")
        if not interrupts:
            return
        payload = Command(resume=_collect_clarifications(interrupts))


def main_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in VALID_MODES else RETRIEVAL_MODE
    # Own instance WITH persistence: the graphs above are for the platform (which injects its
    # own checkpointer), this one is a plain embedding and must carry its own — both to resume
    # the clarification interrupt AND to remember the patient across questions. Compiled here so
    # importing main.py for Studio never pays for it.
    app_cli = build_combined_graph(checkpointer=InMemorySaver())
    # ONE thread for the whole conversation: the checkpointer keys the accumulated patient data
    # (and any paused run) by thread_id, so every question in this session shares it.
    config = {"configurable": {"thread_id": uuid.uuid4().hex},
              "tags": [f"mode:{mode}"], "metadata": {"retrieval_mode": mode}}  # LangSmith

    print(MSG_CLI_INTRO)
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:      # piped input exhausted / Ctrl-Z — end cleanly, not with a traceback
            break
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
            # Clear the session-scoped patient data directly on the thread — a new patient must
            # not inherit the previous one's facts. Per-question state is already reset each turn.
            app_cli.update_state(config, {"patient_facts": {}})
            print(MSG_NEW_PATIENT)
            continue

        _answer_question(app_cli, text, mode, config)


if __name__ == "__main__":
    main_cli()
