"""Pipeline constants, the Studio context schema and the user-facing messages."""
import os
from typing import Literal, TypedDict

# Validation loop: max generations (initial + retries).
MAX_ITER = 2

# Clarification loop budget. With ROUNDS=3 and PER_ROUND=1 the doctor is asked ONE
# question at a time, at most 3 times; already-answered data is never re-asked.
CLARIFY_MAX_ROUNDS = 3
CLARIFY_QUESTIONS_PER_ROUND = 1

# Retrieval strategy (Phase 4). All three feed the SAME generate -> validate -> evidence,
# so citations and the anti-hallucination validator are identical across strategies.
#   "baseline"  -> hybrid + reranker (single-shot)
#   "iterative" -> Track A: self-ask / reflect-retrieve loop for multi-hop
#   "graph"     -> Track B: LightRAG entity-relation graph (default; wins the F4 A/B on
#                  multi-hop recall at baseline latency). Needs the index built once:
#                  python -m retrieval.graph
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "graph")
VALID_MODES = ("baseline", "iterative", "graph")


class ConfigSchema(TypedDict, total=False):
    """Rendered by LangGraph Studio as a `retrieval_mode` dropdown in the run config panel,
    so the three architectures can be picked and traced live without touching the code."""
    retrieval_mode: Literal["baseline", "iterative", "graph"]


# --- User-facing messages (Spanish: shown to the doctor) ---
MSG_NOT_VALIDATED = (
    "No he podido elaborar una respuesta suficientemente fundamentada en las guías "
    "para esta consulta. Te sugiero reformular la pregunta o revisar directamente las "
    "guías; es posible que la información no esté disponible en ellas."
)
MSG_VALIDATION_ERROR = (
    "No he podido validar la respuesta por un problema técnico (no se pudo contactar "
    "con el servicio de validación). Por seguridad no la muestro sin verificar; "
    "inténtalo de nuevo en unos momentos."
)
MSG_OUT_OF_DOMAIN = (
    "Soy un asistente centrado en las guías clínicas de VIH (GeSIDA) y solo puedo "
    "ayudarte con consultas sobre el manejo clínico del VIH. ¿Tienes alguna pregunta "
    "sobre ese tema?"
)
