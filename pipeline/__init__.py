"""LangGraph pipeline for the HIV RAG chatbot.

Assembled from the shared retrieval/generation primitives in `rag.py`:
  - config    — constants, the Studio context schema, user-facing messages.
  - state     — graph state schemas (the retrieval-agnostic RAGState contract + teaching views).
  - nodes     — the combined-pipeline nodes and routing.
  - nodes_expanded — one-node-per-step retrieval, for the dedicated Studio "teaching" graphs.
  - builder   — head/tail assembly + build_graph (dedicated) / build_combined_graph.
"""
from .builder import build_graph, build_combined_graph
from .nodes import refinement_reply
from .config import (RETRIEVAL_MODE, VALID_MODES, MSG_REFINE_OFFER, msg_intro, MSG_CLI_HELP,
                     msg_out_of_domain, MSG_SPECIALTY_CURRENT, MSG_SPECIALTY_CHANGED,
                     MSG_SPECIALTY_UNKNOWN,
                     MSG_NEW_PATIENT, MSG_NO_PATIENT_DATA, MSG_PATIENT_HEADER,
                     MSG_CONFIRM_NEW_PATIENT, MSG_CONFIRM_NEW_PATIENT_ASK,
                     MSG_STEP_INITIAL, MSG_STEP_RESUMED, MSG_STEP_LABELS, MSG_STEP_SOURCES,
                     MSG_STEP_SOURCES_MORE, STEP_SOURCES_SHOWN, MSG_CANCELLED, MSG_EXIT_HINT,
                     MSG_WEB_NEW_PATIENT, MSG_WEB_SHOW_PATIENT,
                     MSG_WEB_REFINE_YES, MSG_WEB_REFINE_NO, MSG_WEB_REFINE_ASK,
                     MSG_WEB_CONFIRM_YES, MSG_WEB_CONFIRM_NO, MSG_WEB_SOURCES_STEP,
                     MSG_WEB_REFINE_OFFER)

__all__ = ["build_graph", "build_combined_graph", "refinement_reply",
           "RETRIEVAL_MODE", "VALID_MODES",
           "MSG_REFINE_OFFER", "msg_intro", "msg_out_of_domain", "MSG_CLI_HELP",
           "MSG_SPECIALTY_CURRENT", "MSG_SPECIALTY_CHANGED", "MSG_SPECIALTY_UNKNOWN",
           "MSG_NEW_PATIENT",
           "MSG_NO_PATIENT_DATA", "MSG_PATIENT_HEADER", "MSG_CONFIRM_NEW_PATIENT",
           "MSG_CONFIRM_NEW_PATIENT_ASK", "MSG_STEP_INITIAL", "MSG_STEP_RESUMED",
           "MSG_STEP_LABELS", "MSG_STEP_SOURCES", "MSG_STEP_SOURCES_MORE",
           "STEP_SOURCES_SHOWN", "MSG_CANCELLED", "MSG_EXIT_HINT",
           "MSG_WEB_NEW_PATIENT", "MSG_WEB_SHOW_PATIENT",
           "MSG_WEB_REFINE_YES", "MSG_WEB_REFINE_NO", "MSG_WEB_REFINE_ASK",
           "MSG_WEB_CONFIRM_YES", "MSG_WEB_CONFIRM_NO", "MSG_WEB_SOURCES_STEP",
           "MSG_WEB_REFINE_OFFER"]
