"""Pipeline constants, the Studio context schema and the user-facing messages."""
import os
from typing import Literal, TypedDict

from retrieval.registry import VALID_MODES  # re-exported: the catalogue of retrieval modes

# Validation loop: max generations (initial + retries).
MAX_ITER = 2

# Clarification budget. The step is a REFINEMENT offered after answering, not a gate before
# it, which flips both numbers: asking one question at a time made sense while each pause was
# the price of getting any answer at all, and it cost up to 3 sequential pauses (and 3 assess
# calls) before the doctor saw a single word. Now the answer is already on screen, so all the
# pending dimensions are offered in ONE optional pause the doctor can ignore with Enter.
CLARIFY_MAX_ROUNDS = 1          # refinement offers per question
CLARIFY_QUESTIONS_PER_ROUND = 3 # dimensions offered in that single pause (assess caps at 3)

# Retrieval strategy. Every mode feeds the SAME generate -> validate -> evidence, so
# citations and the anti-hallucination validator are identical across strategies; the modes
# themselves are declared in retrieval/registry.py (which also documents each one). The
# graph modes need their index built once (python -m retrieval.graph / .hipporag).
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "graph")


class ConfigSchema(TypedDict, total=False):
    """Rendered by LangGraph Studio as a `retrieval_mode` dropdown in the run config panel,
    so the architectures can be picked and traced live without touching the code. The values
    must be spelled out: Studio builds the dropdown from the Literal at import time, so it
    cannot be derived from VALID_MODES — keep both in sync when adding a mode."""
    retrieval_mode: Literal["baseline", "iterative", "graph", "pathrag", "hipporag"]


# --- User-facing messages (Spanish: shown to the doctor) ---
# Step labels for MSG_TECHNICAL_ERROR. They are user-facing text, so they live here with the
# rest of the Spanish rather than inline at the call site.
STEP_RETRIEVAL   = "al consultar las guías"
STEP_GENERATION  = "al redactar la respuesta"
STEP_FORMATTING  = "al preparar las fuentes"
MSG_TECHNICAL_ERROR = (
    "No he podido completar la consulta por un problema técnico ({step}). Esto NO es un "
    "resultado clínico: no significa que las guías no cubran tu pregunta ni que no haya "
    "recomendación. Vuelve a intentarlo en unos momentos."
)
# How to DECLINE differs by frontend (Enter in the terminal, a button in the web), so the offer
# itself is shared and only the hint is not — the web must not tell a doctor to press Enter.
_REFINE_OFFER = ("Puedo concretar más la respuesta para este paciente si me facilitas alguno "
                 "de estos datos")
MSG_REFINE_OFFER = f"{_REFINE_OFFER} (pulsa Enter para dejarlo así):"
MSG_WEB_REFINE_OFFER = f"{_REFINE_OFFER}:"

# --- Progress labels (visible work) ---
# A clinical question takes 12-25 s across refine -> retrieval -> assess -> generate -> validate,
# and until the run streamed there was no sign of any of it. These label the wait.
#
# KEYED BY THE NODE THAT JUST FINISHED, VALUED WITH WHAT IS RUNNING NOW. That is what the
# `updates` stream gives us — a node's chunk arrives when it COMPLETES — and the pipeline is
# linear enough after each node for "what comes next" to be the useful thing to say. Where it
# branches (validate -> evidence | refocus | fallback) the label is the common case and the next
# chunk overwrites it within the same second. Nodes with no entry are silent: `evidence` is
# followed by the answer itself, and `confirm_patient` either passes through or asks.
MSG_STEP_INITIAL = "Interpretando la consulta"
# Shown while a RESUMED stretch waits for its first chunk. Both pauses (the patient-switch gate
# and the refinement offer) hand back to a retrieval step, so one label covers them.
MSG_STEP_RESUMED = "Buscando en las guías con lo que has indicado"
MSG_STEP_LABELS = {
    "rephrase":         "Buscando en las guías",
    "retrieve":         "Ordenando los fragmentos encontrados",
    "rerank":           "Comprobando qué datos afinarían la respuesta",
    "assess_context":   "Redactando la respuesta",
    "re_retrieve":      "Redactando la respuesta",
    "generate":         "Verificando que la respuesta se apoya en las guías",
    "validate":         "Preparando las fuentes",
    "refocus_retrieve": "Rehaciendo la respuesta con la nueva evidencia",
}
# The non-baseline modes collapse retrieval into one node named after the mode; all of them hand
# over to assess_context, so they share `rerank`'s label. Derived from the catalogue so a new
# mode is covered the moment it is registered.
MSG_STEP_LABELS.update({f"{mode}_retrieve": MSG_STEP_LABELS["rerank"]
                        for mode in VALID_MODES if mode != "baseline"})

# The sections a retrieval read. Unlike the step labels this one is KEPT on screen instead of
# being overwritten: it is a fact about what was consulted, it stays true, and it is what lets
# the doctor recognise the ground the answer stands on before reading a word of it.
MSG_STEP_SOURCES = "Guías consultadas:"
MSG_STEP_SOURCES_MORE = "y {n} más"
STEP_SOURCES_SHOWN = 3

# --- CLI (conversational REPL) ---
MSG_CLI_INTRO = (
    "Asistente sobre las guías de VIH (GeSIDA). Escribe tu consulta y pulsa Enter.\n"
    "Los datos que menciones se recuerdan durante la conversación para afinar las respuestas.\n"
    "Comandos:  /nuevo (empezar con otro paciente) · /paciente (ver datos recordados) · "
    "/ayuda · /salir"
)
MSG_CLI_HELP = (
    "Comandos disponibles:\n"
    "  /nuevo     — olvida los datos del paciente actual y empieza de cero\n"
    "  /paciente  — muestra los datos que recuerdo de este paciente\n"
    "  /salir     — termina la sesión"
)
MSG_NEW_PATIENT = "De acuerdo, empiezo de cero. He olvidado los datos del paciente anterior."
MSG_CANCELLED = "Consulta cancelada. Los datos del paciente se mantienen."
MSG_EXIT_HINT = "Pulsa Ctrl-C otra vez (o escribe /salir) para terminar la sesión."
MSG_CONFIRM_NEW_PATIENT = (
    "Esta consulta parece referirse a un paciente distinto del que venía recordando.\n"
    "Datos que tengo guardados:"
)
MSG_CONFIRM_NEW_PATIENT_ASK = (
    "¿Es un paciente distinto? Escribe «sí» para empezar de cero; pulsa Enter para "
    "mantener estos datos."
)
MSG_NO_PATIENT_DATA = "Todavía no he recogido ningún dato del paciente en esta conversación."

# --- Web (Chainlit) ---
MSG_WEB_INTRO = (
    "Asistente sobre las guías de VIH (GeSIDA). Escribe tu consulta.\n\n"
    "Los datos que menciones se recuerdan durante la conversación para afinar las respuestas; "
    "usa **Paciente nuevo** para empezar de cero."
)
MSG_WEB_NEW_PATIENT = "Paciente nuevo"
MSG_WEB_SHOW_PATIENT = "Datos recordados"
MSG_WEB_REFINE_YES = "Aportar estos datos"
MSG_WEB_REFINE_NO = "Así está bien"
MSG_WEB_REFINE_ASK = (
    "Responde en un solo mensaje a las que puedas; lo que dejes sin contestar se seguirá "
    "presentando por ramas en la respuesta."
)
MSG_WEB_CONFIRM_YES = "Sí, es otro paciente"
MSG_WEB_CONFIRM_NO = "No, el mismo paciente"
MSG_WEB_SOURCES_STEP = "Guías consultadas"
MSG_PATIENT_HEADER = "Datos del paciente recordados en esta conversación:"
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
