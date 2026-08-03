"""Chainlit frontend: the same graph as the CLI, with the interactive moments given a UI.

    chainlit run web/app.py

Why a second frontend rather than a prettier terminal: the pipeline already had the two
moments a chat UI does better than an `input()`, and one open item that only a UI can close.

  - `confirm_patient` interrupts to ask whether this is a different patient — a yes/no, so it
    becomes two buttons instead of a typed «sí».
  - `refine_offer` interrupts to offer a refinement AFTER the answer. In the terminal declining
    it meant pressing Enter once per question; here it is one click on «Así está bien».
  - the three follow-up questions were dead text the doctor had to retype. Here they are
    buttons (the last open item of OPEN D).

What this module deliberately does NOT own: the stream's rules (`progress.read_chunk`), the
shape of a resume value (`pipeline.refinement_reply`), the citation verdicts and the wording of
an answer (`evidence.resolve_answer` / `format_answer_markdown`). All of that is shared with the
CLI on purpose — a frontend that decided any of it for itself could show a doctor something the
other frontend would refuse to.

Streaming: `astream` is used so the sync nodes run in LangGraph's executor and the event loop
stays free to paint. NOTE the known limit inherited from `retrieval/graph.py`: LightRAG is
driven through ONE module-level event loop, so two graph-mode queries running at the same time
(two browser tabs) would contend for it. Fine for a demo, not for concurrent users.
"""
import asyncio
import os
import sys
import threading
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

if os.environ.get("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "chatbot_vih")

import chainlit as cl
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import corpus
import rag
from evidence import format_answer_markdown, resolve_answer
from pipeline import (build_combined_graph, refinement_reply, RETRIEVAL_MODE, msg_intro,
                      MSG_CONFIRM_NEW_PATIENT, MSG_CONFIRM_NEW_PATIENT_ASK, MSG_NEW_PATIENT,
                      MSG_NO_PATIENT_DATA, MSG_PATIENT_HEADER,
                      MSG_STEP_INITIAL, MSG_STEP_LABELS, MSG_STEP_RESUMED, MSG_WEB_CONFIRM_NO,
                      MSG_WEB_CONFIRM_YES, MSG_WEB_NEW_PATIENT,
                      MSG_WEB_REFINE_ASK, MSG_WEB_REFINE_NO, MSG_WEB_REFINE_OFFER,
                      MSG_WEB_REFINE_YES, MSG_WEB_SHOW_PATIENT, MSG_WEB_SOURCES_STEP)
from progress import read_chunk

# ONE graph and ONE checkpointer for the process; sessions are separated by thread_id, exactly
# as the CLI separates questions within a session. Compiling is graph assembly only — no model
# is loaded here.
APP = build_combined_graph(checkpointer=InMemorySaver())
STREAM_MODES = ["updates", "custom"]

# Preload the local models so the first real query does not pay their ~3.5 s (same reason and
# same daemon-thread shape as main.py).
threading.Thread(target=rag.warmup, daemon=True).start()


# --- session plumbing ------------------------------------------------------
def _specialty() -> str:
    """The specialty this chat answers from. Chainlit's chat profile is the picker, and it is
    chosen BEFORE the first message, which matches how the value behaves everywhere else: it is
    session-scoped and must not change under the doctor mid-conversation."""
    return corpus.resolve_specialty(cl.user_session.get("chat_profile"))


def _config() -> dict:
    """This browser session's thread: one per chat, so the patient is remembered across
    questions and cleared only on an explicit new patient."""
    config = cl.user_session.get("config")
    if config is None:
        mode, specialty = RETRIEVAL_MODE, _specialty()
        config = {"configurable": {"thread_id": uuid.uuid4().hex, "specialty": specialty},
                  "tags": [f"mode:{mode}", f"specialty:{specialty}"],
                  "metadata": {"retrieval_mode": mode, "specialty": specialty}}
        cl.user_session.set("config", config)
    return config


def _patient_facts() -> dict:
    return APP.get_state(_config()).values.get("patient_facts") or {}


def _format_patient_facts(facts: dict) -> str:
    items = {k: v for k, v in (facts or {}).items() if k}
    if not items:
        return MSG_NO_PATIENT_DATA
    lines = "\n".join(f"- **{k}**: {v}" if v else f"- **{k}**" for k, v in items.items())
    return f"{MSG_PATIENT_HEADER}\n{lines}"


def _session_actions() -> list:
    return [cl.Action(name="new_patient", payload={}, label=MSG_WEB_NEW_PATIENT),
            cl.Action(name="show_patient", payload={}, label=MSG_WEB_SHOW_PATIENT)]


# --- the progress steps ----------------------------------------------------
class Steps:
    """The visible work: one collapsible step per phase, closed as the next one starts.

    A clinical question spends 12-25 s in refine -> retrieval -> assess -> generate -> validate.
    `MSG_STEP_LABELS` is keyed by the node that just FINISHED and says what runs NOW, which is
    what an `updates` chunk can tell us (see pipeline/config.py)."""

    def __init__(self):
        self.current = None
        self.label = ""

    async def switch(self, label: str) -> None:
        # Consecutive nodes can share a label (assess_context and re_retrieve both hand over to
        # generation): repeating it would show the same phase as two steps.
        if label == self.label:
            return
        await self.close()
        self.label = label
        self.current = cl.Step(name=label, type="run")
        await self.current.__aenter__()

    async def step(self, node: str) -> None:
        if node in MSG_STEP_LABELS:
            await self.switch(MSG_STEP_LABELS[node])

    async def progress(self, event: dict) -> None:
        """A ProgressEvent from inside a long step. The sections read get their OWN top-level
        step, which is why the phase step is closed first: they are a fact about what was
        consulted, worth reading without expanding anything, and the one piece of progress that
        can never have to be taken back (unlike the answer, which the validator may reject)."""
        if event.get("kind") == "sources":
            items = event.get("items") or []
            if items:
                await self.close()
                async with cl.Step(name=MSG_WEB_SOURCES_STEP, type="retrieval") as step:
                    step.output = "\n".join(f"- {item}" for item in items)
        elif event.get("detail"):
            await self.switch(event["detail"])

    async def close(self) -> None:
        if self.current is not None:
            await self.current.__aexit__(None, None, None)
            self.current, self.label = None, ""


# --- the two pauses --------------------------------------------------------
async def _confirm_new_patient(value: dict) -> str:
    """The blocking patient-switch gate, BEFORE any answer. Answering a second patient's
    question with the first one's gestation folded in is the harm the system exists to avoid,
    so this is the one pause allowed to block."""
    reply = await cl.AskActionMessage(
        content=f"{MSG_CONFIRM_NEW_PATIENT}\n{_format_patient_facts(value.get('facts'))}\n\n"
                f"{MSG_CONFIRM_NEW_PATIENT_ASK}",
        actions=[cl.Action(name="same", payload={"switch": False}, label=MSG_WEB_CONFIRM_NO),
                 cl.Action(name="new", payload={"switch": True}, label=MSG_WEB_CONFIRM_YES)],
        timeout=300,
    ).send()
    # No answer (timeout / closed tab) KEEPS the patient: losing their history silently is the
    # worse of the two mistakes, and the doctor can always start fresh explicitly.
    return "sí" if (reply or {}).get("payload", {}).get("switch") else ""


async def _offer_refinement(value: dict):
    """The optional refinement, offered with the answer ALREADY on screen. Declining is one
    click; only the doctor who wants to refine types anything."""
    questions = list(value.get("questions", []))
    listed = "\n".join(f"- {q}" for q in questions)
    reply = await cl.AskActionMessage(
        content=f"{MSG_WEB_REFINE_OFFER}\n{listed}",
        actions=[cl.Action(name="skip", payload={"refine": False}, label=MSG_WEB_REFINE_NO),
                 cl.Action(name="refine", payload={"refine": True}, label=MSG_WEB_REFINE_YES)],
        timeout=300,
    ).send()
    if not (reply or {}).get("payload", {}).get("refine"):
        return refinement_reply(questions, {})       # never {} — see refinement_reply
    typed = await cl.AskUserMessage(content=MSG_WEB_REFINE_ASK, timeout=300).send()
    text = ((typed or {}).get("output") or "").strip()
    if not text:
        return refinement_reply(questions, {})
    # One free-text message cannot be attributed to a particular question, so it is handed over
    # whole. `_fold_answers` keys it by the question when only one was asked and files it as the
    # doctor's reply otherwise, leaving the rest pending — so the refined answer keeps laying
    # out the branches of whatever went unanswered.
    return text


async def _resume_for(interrupts) -> object:
    value = getattr(interrupts[0], "value", None) or {}
    if value.get("confirm_new_patient"):
        return await _confirm_new_patient(value)
    return await _offer_refinement(value)


# --- answering -------------------------------------------------------------
async def answer_question(question: str) -> None:
    """Run one question to completion, handling either pause. Same contract as the CLI's
    `_answer_question`: the answer is shown as soon as it exists, the pause happens WITH it on
    screen, and a repeat of the same text is never posted twice."""
    config = _config()
    payload, shown = {"question": question}, None
    steps = Steps()
    waiting = MSG_STEP_INITIAL
    # Collected from the stream rather than asked of the checkpointer: both were written by
    # earlier nodes of this same stretch, so they are in hand by the time `evidence` emits text.
    answer, index, wrote = None, None, ""
    try:
        while True:
            interrupts = None
            await steps.switch(waiting)
            async for channel, chunk in APP.astream(
                    payload, context={"retrieval_mode": RETRIEVAL_MODE},
                    config=config, stream_mode=STREAM_MODES):
                for kind, value in read_chunk(channel, chunk):
                    if kind == "step":
                        wrote = value          # whichever node writes next, this is the one
                        await steps.step(value)
                    elif kind == "progress":
                        await steps.progress(value)
                    elif kind == "state":
                        answer = value.get("answer") or answer
                        index = value.get("chunk_index") or index
                    elif kind == "interrupt":
                        interrupts = value
                    elif kind == "output" and value != shown:
                        await steps.close()
                        # ONLY `evidence` produces a real answer, and only then are there
                        # sources and follow-ups. out_of_domain, fallback and technical_error
                        # write plain text and stay plain text: giving an outage a sources panel
                        # would make it read as a clinical finding, which is exactly what those
                        # messages exist to prevent.
                        certified = wrote == "evidence" and answer and index
                        await _post_answer(value, resolve_answer(answer, index)
                                           if certified else None)
                        shown = value
            if not interrupts:
                return
            payload = Command(resume=await _resume_for(interrupts))
            waiting = MSG_STEP_RESUMED
    finally:
        await steps.close()


async def _post_answer(output: str, view) -> None:
    """Post the answer (markdown + clickable follow-ups) or, with no certified view, the plain
    message the graph wrote."""
    if view is None:
        await cl.Message(content=output).send()
        return
    actions = [cl.Action(name="follow_up", payload={"question": q}, label=q)
               for q in view.follow_ups]
    await cl.Message(content=format_answer_markdown(view), actions=actions).send()


# --- Chainlit handlers -----------------------------------------------------
@cl.set_chat_profiles
async def chat_profiles():
    """One profile per specialty on disk, so the doctor picks the clinical area before asking.
    Derived from `data/specialties/`, never listed here: a second list would drift."""
    return [cl.ChatProfile(name=specialty_id,
                           markdown_description=corpus.specialty(specialty_id).display_name)
            for specialty_id in corpus.specialties()]


@cl.on_chat_start
async def on_chat_start():
    _config()
    await cl.Message(content=msg_intro(_specialty(), web=True),
                     actions=_session_actions()).send()


@cl.on_message
async def on_message(message: cl.Message):
    text = (message.content or "").strip()
    if not text:
        return
    low = text.lower()
    if low in ("/nuevo", "/reset"):
        await _new_patient()
    elif low in ("/paciente", "/datos"):
        await cl.Message(content=_format_patient_facts(_patient_facts())).send()
    else:
        await answer_question(text)


@cl.action_callback("follow_up")
async def on_follow_up(action: cl.Action):
    """A follow-up question asked with one click, and echoed as if typed so the transcript
    still reads as a conversation."""
    question = action.payload.get("question", "")
    if not question:
        return
    await cl.Message(content=question, type="user_message").send()
    await answer_question(question)


@cl.action_callback("new_patient")
async def on_new_patient(action: cl.Action):
    await _new_patient()


@cl.action_callback("show_patient")
async def on_show_patient(action: cl.Action):
    await cl.Message(content=_format_patient_facts(_patient_facts())).send()


async def _new_patient() -> None:
    """Forget the patient: their facts AND the conversation, so a follow-up does not resolve
    against the previous patient's question."""
    await asyncio.to_thread(APP.update_state, _config(),
                            {"patient_facts": {}, "prev_question": ""})
    await cl.Message(content=MSG_NEW_PATIENT, actions=_session_actions()).send()
