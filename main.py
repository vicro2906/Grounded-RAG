"""
main.py — Punto de entrada: orquesta el RAG clínico (guías VIH) con LangGraph + LangSmith.

Encadena como nodos de un grafo las funciones del pipeline (rag.py) y el formateo
de evidencias (evidencias.py), para poder auditar cada paso en LangSmith.

Flujo:   question ─▶ rephrase ─▶ retrieve ─▶ rerank ─▶ generate ⇄ validate ─▶ evidence ─▶ output
         (validate reintenta generate si no es apta, hasta MAX_ITER; si se agota, salida segura)

Trazado (LangSmith):
    Se activa automáticamente SOLO si defines LANGSMITH_API_KEY en .env. Entonces
    cada invocación del grafo y cada llamada a OpenAI (chat y embeddings) aparece
    en https://smith.langchain.com dentro del proyecto LANGSMITH_PROJECT.
    Sin esa clave, el grafo funciona igual pero sin enviar trazas.

Uso:
    python main.py            # modo interactivo
"""
import os
import sys

# La consola de Windows usa cp1252 por defecto y rompe acentos/cajas. Forzamos UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from typing import TypedDict, cast

from dotenv import load_dotenv
load_dotenv()

# --- LangSmith: activar trazado solo si hay API key (si no, no estorba) ---
TRACING = bool(os.environ.get("LANGSMITH_API_KEY"))
if TRACING:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "chatbot-vih")

# --- Pipeline RAG (funciones de recuperación/generación y clientes) ---
import rag  # crea los clientes OpenAI/Qdrant al importarse
from rag import (rephrase, retrieve_hibrido, rerank, build_context, validar,
                 SYS_PROMPT, build_user_prompt, MODELO_GENERACION)
from evidencias import format_answer

# --- Trazado fino de las llamadas a OpenAI (tokens, latencia) SIN modificar rag.py ---
# Las funciones de rag leen `rag.client` en tiempo de ejecución, así que reasignar
# aquí su versión "envuelta" por LangSmith basta para que retrieve() y los embeddings
# queden trazados dentro del run del grafo.
if TRACING:
    try:
        from langsmith.wrappers import wrap_openai
        rag.client = wrap_openai(rag.client)
    except Exception:
        pass

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


# ===========================================================================
# GENERACIÓN ESTRUCTURADA (LangChain) — sustituye al response_format crudo de
# main.py. Por debajo usa el MISMO mecanismo de OpenAI (Structured Outputs,
# json_schema estricto), pero con validación Pydantic y trazado nativo en
# LangSmith. Devolvemos .model_dump() para que el resto del pipeline siga
# recibiendo el mismo dict de siempre (uso idéntico, no se toca evidencias.py).
# ===========================================================================
class SourceUsed(BaseModel):
    ref: int
    quote: str


class ClinicalAnswer(BaseModel):
    sufficient_information: bool
    answer: str
    sources_used: list[SourceUsed]
    follow_up_questions: list[str]


_structured_llm = ChatOpenAI(model=MODELO_GENERACION, temperature=0.2).with_structured_output(
    ClinicalAnswer, method="json_schema", strict=True
)

# Bucle de validación: nº máximo de generaciones (inicial + reintentos).
MAX_ITER = 2
MENSAJE_NO_VALIDADA = (
    "No he podido elaborar una respuesta suficientemente fundamentada en las guías "
    "para esta consulta. Te sugiero reformular la pregunta o revisar directamente las "
    "guías; es posible que la información no esté disponible en ellas."
)
MENSAJE_ERROR_VALIDACION = (
    "No he podido validar la respuesta por un problema técnico (no se pudo contactar "
    "con el servicio de validación). Por seguridad no la muestro sin verificar; "
    "inténtalo de nuevo en unos momentos."
)


# ===========================================================================
# ESTADO
#   InputState: lo único que hay que pasar a app.invoke() -> la pregunta.
#   RAGState:   estado interno completo que viaja por el grafo. Las claves son
#               requeridas (cada nodo las va rellenando), de modo que acceder a
#               state["..."] sea seguro a ojos del analizador de tipos. Los nodos
#               devuelven solo las claves que producen y LangGraph las fusiona.
# ===========================================================================
class InputState(TypedDict):
    question: str            # entrada del usuario


class RAGState(TypedDict):
    question: str            # entrada del usuario (original; se usa para generar)
    search_query: str        # pregunta reescrita/normalizada para el retriever (rephrase)
    candidates: list         # payloads recuperados (híbrido, ~20) antes de reordenar
    contexts: list           # payloads finales tras el reranker (top 5)
    chunk_index: dict        # {nº: chunk} para citar fuentes (build_context)
    formatted_context: str   # contexto numerado [1]/[2]… para el LLM
    answer: dict             # respuesta estructurada del LLM (generate_answer)
    intentos: int            # nº de generaciones hechas (para el bucle de validación)
    validacion: dict         # veredicto del validador {apto, motivo, ...}
    output: str              # texto final formateado con fuentes (format_answer)


# ===========================================================================
# NODOS — cada uno envuelve una de tus funciones existentes.
#   Reciben el estado completo y devuelven SOLO las claves que producen.
# ===========================================================================
def node_rephrase(state: RAGState) -> dict:
    """question -> consulta reescrita/normalizada para el retriever (no añade info)."""
    return {"search_query": rephrase(state["question"])}


def node_retrieve(state: RAGState) -> dict:
    """search_query -> ~20 candidatos por búsqueda híbrida (denso + BM25)."""
    candidates = retrieve_hibrido(state["search_query"], top_k=20, prefetch_limit=30)
    return {"candidates": candidates}


def node_rerank(state: RAGState) -> dict:
    """candidatos -> top 5 reordenados por el cross-encoder + contexto numerado."""
    contexts = rerank(state["search_query"], state["candidates"], top_k=5)
    chunk_index, formatted_context = build_context(contexts)
    return {
        "contexts": contexts,
        "chunk_index": chunk_index,
        "formatted_context": formatted_context,
    }


def node_generate(state: RAGState) -> dict:
    """contexto + pregunta -> respuesta estructurada del LLM. Si hubo un intento
    previo rechazado por el validador, inyecta su feedback para corregir."""
    user = build_user_prompt(state["question"], state["formatted_context"])
    val = state.get("validacion")
    if val and not val.get("apto", True):
        user += (
            f"\n\n    REINTENTO: tu respuesta anterior fue RECHAZADA por el validador. "
            f"Motivo: {val.get('motivo', '')}. Corrige la respuesta para que cada afirmación "
            f"esté respaldada por el contexto y aborde la pregunta. Si el contexto no respalda "
            f"la respuesta, marca informacion_suficiente=false."
        )
    mensajes = [("system", SYS_PROMPT), ("human", user)]
    answer = cast(ClinicalAnswer, _structured_llm.invoke(mensajes))  # validado por Pydantic
    return {"answer": answer.model_dump(), "intentos": state.get("intentos", 0) + 1}


def node_validate(state: RAGState) -> dict:
    """Juez de relevancia + grounding sobre la respuesta. Marca apto / no apto."""
    veredicto = validar(state["question"], state["answer"], state["formatted_context"])
    return {"validacion": veredicto}


def node_evidence(state: RAGState) -> dict:
    """respuesta + índice -> texto final con panel de fuentes y seguimiento."""
    output = format_answer(state["answer"], state["chunk_index"])
    return {"output": output}


def node_fallback(state: RAGState) -> dict:
    """No se muestra la respuesta. El mensaje depende del motivo: error técnico del
    validador vs. no se logró una respuesta apta tras los reintentos."""
    if state["validacion"].get("error", False):
        return {"output": MENSAJE_ERROR_VALIDACION}
    return {"output": MENSAJE_NO_VALIDADA}


def route_validacion(state: RAGState) -> str:
    """Tras validar:
      - error técnico del validador -> fallback (no reintentar; el juez está caído).
      - apta -> formatear (evidence).
      - no apta y quedan intentos -> regenerar con feedback.
      - no apta y agotados -> fallback (salida segura)."""
    v = state["validacion"]
    if v.get("error", False):
        return "fallback"
    if v.get("apto", False):
        return "evidence"
    if state.get("intentos", 0) >= MAX_ITER:
        return "fallback"
    return "generate"


# ===========================================================================
# GRAFO:  START ─▶ rephrase ─▶ retrieve ─▶ rerank ─▶ generate ─▶ validate ─┬─▶ evidence ─▶ END
#                                              ▲ (reintento)  │            ├─▶ generate (bucle)
#                                              └──────────────┘            └─▶ fallback ─▶ END
# ===========================================================================
def build_graph():
    builder = StateGraph(RAGState, input_schema=InputState)
    builder.add_node("rephrase", node_rephrase)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("rerank", node_rerank)
    builder.add_node("generate", node_generate)
    builder.add_node("validate", node_validate)
    builder.add_node("evidence", node_evidence)
    builder.add_node("fallback", node_fallback)

    builder.add_edge(START, "rephrase")
    builder.add_edge("rephrase", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "validate")
    # Bucle: validate -> evidence (apto) | generate (reintento) | fallback (agotado)
    builder.add_conditional_edges("validate", route_validacion,
                                  {"evidence": "evidence", "generate": "generate",
                                   "fallback": "fallback"})
    builder.add_edge("evidence", END)
    builder.add_edge("fallback", END)
    return builder.compile()


# Grafo compilado y reutilizable (lo importan tanto el CLI como la evaluación).
app = build_graph()


def main_cli():
    question = input("¿Cuál es tú pregunta?: ")
    result = app.invoke({"question": question})
    print(result["output"])


if __name__ == "__main__":
    main_cli()
