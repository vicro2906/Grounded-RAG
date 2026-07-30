"""Progress events: what a run is DOING, while it is still doing it.

A clinical question spends 12-25 s in refine -> retrieval -> assess -> generate -> validate.
LangGraph's `updates` stream already reports every node as it COMPLETES, which covers most of
that for free; these events exist for the two things `updates` cannot see:

  - INSIDE a single long node. The graph modes collapse their whole retrieval into one node and
    spend 5-10 s there, so between two `updates` chunks the doctor gets a frozen line.
  - WHICH GUIDELINE SECTIONS were read. That is the one piece of progress worth keeping on
    screen: it is real, it is clinically recognisable, and — unlike the answer text, which the
    validator may still reject — it can never be retracted.

They are a SIDE CHANNEL, never graph state: nothing downstream reads them, so a frontend that
ignores them loses only the view. Hence `emit` swallowing everything — a run must not fail
because nobody was watching.

Placement: this is a ROOT module rather than part of `pipeline/`. `rag.py` and `retrieval/` sit
BELOW the pipeline and cannot import from it, yet they are where the long steps actually are.
Progress reporting is observability, the same kind of dependency as `logging`, so it lives
where every layer can reach it.
"""
from typing import Literal, TypedDict

# Stable identifiers, English like every other internal name. How a step is NAMED to the doctor
# belongs to pipeline/config.py and how it is PAINTED belongs to the frontend — the same split
# evidence.py already makes between resolving an answer and rendering it.
STEP_RETRIEVAL = "retrieval"


class ProgressEvent(TypedDict, total=False):
    """One thing that just happened inside a step."""
    kind: Literal["detail", "sources"]
    step: str            # which step it came from (STEP_* above)
    detail: str          # for "detail": one line of Spanish, ready to show
    items: list[str]     # for "sources": the section labels, UNJOINED so each frontend lays
                         # them out its own way (the CLI joins them, a web view lists them)


def read_chunk(channel: str, chunk) -> list:
    """Normalize ONE stream chunk into the events a frontend has to react to:

        ("step", node)        a node finished — say what is running now
        ("progress", event)   a ProgressEvent from inside a long step
        ("state", update)     the partial state that node wrote, for a frontend that needs more
                              than the text (the web reads `answer`/`chunk_index` from here to
                              render structured sources, rather than asking the checkpointer
                              mid-run for a write that may not have landed yet)
        ("output", text)      visible text was written
        ("interrupt", tuple)  the run paused and needs a Command(resume=…)

    It takes one chunk rather than the whole stream so the SAME rules serve a sync `stream` and
    an async `astream`. And they must be the same rules, because none of the three is obvious
    and each fails silently: the pause arrives as an `__interrupt__` KEY inside an `updates`
    chunk (miss it and the refinement is never offered), a node that did nothing yields None
    (read `output` off it and the whole answer dies in an AttributeError), and the visible answer
    is the last non-empty `output` of the stretch — equivalent to the final state only because
    exactly one node writes one per run.

    Whether a repeated output should be shown again is NOT decided here: that depends on what is
    already on screen, which is the frontend's business."""
    if channel == "custom":
        return [("progress", chunk)]
    if channel != "updates":
        return []
    events = []
    for node, update in chunk.items():
        if node == "__interrupt__":
            events.append(("interrupt", update))
            continue
        events.append(("step", node))
        if not update:                       # a no-op node yields None
            continue
        events.append(("state", update))
        if update.get("output"):
            events.append(("output", update["output"]))
    return events


def emit(**event) -> None:
    """Fire-and-forget a ProgressEvent into the run's `custom` stream.

    A no-op outside a graph run: `evaluation.py` and the smoke scripts call the retrieval
    primitives directly, where `get_stream_writer()` raises — so nothing in the pipeline can
    come to depend on being observed. The import is deferred for the same reason the failure is
    swallowed: this module must stay usable from any layer, with or without langgraph present.
    """
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()(event)
    except Exception:
        pass
