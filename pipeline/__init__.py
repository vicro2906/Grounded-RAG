"""LangGraph pipeline for the HIV RAG chatbot.

Assembled from the shared retrieval/generation primitives in `rag.py`:
  - config    — constants, the Studio context schema, user-facing messages.
  - state     — graph state schemas (the retrieval-agnostic RAGState contract + teaching views).
  - nodes     — the combined-pipeline nodes and routing.
  - nodes_expanded — one-node-per-step retrieval, for the dedicated Studio "teaching" graphs.
  - builder   — head/tail assembly + build_graph (dedicated) / build_combined_graph.
"""
from .builder import build_graph, build_combined_graph
from .config import (RETRIEVAL_MODE, VALID_MODES, MSG_REFINE_OFFER, MSG_CLI_INTRO, MSG_CLI_HELP,
                     MSG_NEW_PATIENT, MSG_NO_PATIENT_DATA, MSG_PATIENT_HEADER)

__all__ = ["build_graph", "build_combined_graph", "RETRIEVAL_MODE", "VALID_MODES",
           "MSG_REFINE_OFFER", "MSG_CLI_INTRO", "MSG_CLI_HELP", "MSG_NEW_PATIENT",
           "MSG_NO_PATIENT_DATA", "MSG_PATIENT_HEADER"]
