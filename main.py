"""Entry point: compiles the LangGraph app(s) and runs the interactive CLI.

The pipeline itself lives in the `pipeline/` package. This module only wires up runtime
concerns (UTF-8 console, LangSmith tracing, model warm-up), compiles the graphs that
langgraph.json exposes to Studio, and offers a small CLI.

Choosing the retrieval mode:
    - CLI:    the default is RETRIEVAL_MODE (env var, falls back to "graph"); `python main.py
              iterative` forces one for a quick manual test.
    - Studio: the combined graph exposes a `retrieval_mode` dropdown in the run config panel,
              so all three architectures can be picked and traced live.

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

from pipeline import build_graph, build_combined_graph, RETRIEVAL_MODE, VALID_MODES

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


def main_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in VALID_MODES else RETRIEVAL_MODE
    question = input("¿Cuál es tu pregunta?: ")
    result = app.invoke(
        {"question": question},
        context={"retrieval_mode": mode},                       # selects the mode
        config={"tags": [f"mode:{mode}"], "metadata": {"retrieval_mode": mode}},  # LangSmith
    )
    print(result["output"])


if __name__ == "__main__":
    main_cli()
