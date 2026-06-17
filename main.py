"""
main.py — Punto de entrada: orquesta el RAG clínico (guías VIH) con LangGraph + LangSmith.

Encadena como nodos de un grafo las funciones del pipeline (rag.py) y el formateo
de evidencias (evidencias.py), para poder auditar cada paso en LangSmith.

Flujo:   question ─▶ retrieve ─▶ generate ─▶ evidence ─▶ output

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

from typing import TypedDict

from dotenv import load_dotenv
load_dotenv()

# --- LangSmith: activar trazado solo si hay API key (si no, no estorba) ---
TRACING = bool(os.environ.get("LANGSMITH_API_KEY"))
if TRACING:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "chatbot-vih")

# --- Pipeline RAG (funciones de recuperación/generación y clientes) ---
import rag  # crea los clientes OpenAI/Qdrant al importarse
from rag import retrieve, build_context, SYS_PROMPT, build_user_prompt
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


_structured_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2).with_structured_output(
    ClinicalAnswer, method="json_schema", strict=True
)


# ===========================================================================
# ESTADO — lo que viaja por el grafo. Cada nodo lee unas claves y escribe otras.
#   total=False => no hace falta rellenar todas las claves de golpe; cada nodo
#   devuelve solo las que produce y LangGraph las fusiona en el estado.
# ===========================================================================
class RAGState(TypedDict, total=False):
    question: str            # entrada del usuario
    contexts: list           # payloads recuperados de Qdrant (retrieve)
    chunk_index: dict        # {nº: chunk} para citar fuentes (build_context)
    formatted_context: str   # contexto numerado [1]/[2]… para el LLM
    answer: dict             # respuesta estructurada del LLM (generate_answer)
    output: str              # texto final formateado con fuentes (format_answer)


# ===========================================================================
# NODOS — cada uno envuelve una de tus funciones existentes.
#   Reciben el estado completo y devuelven SOLO las claves que producen.
# ===========================================================================
def node_retrieve(state: RAGState) -> dict:
    """question -> contextos de Qdrant + índice de citas + contexto numerado."""
    contexts = retrieve(state["question"])
    chunk_index, formatted_context = build_context(contexts)
    return {
        "contexts": contexts,
        "chunk_index": chunk_index,
        "formatted_context": formatted_context,
    }


def node_generate(state: RAGState) -> dict:
    """contexto + pregunta -> respuesta estructurada (validada) del LLM."""
    mensajes = [
        ("system", SYS_PROMPT),
        ("human", build_user_prompt(state["question"], state["formatted_context"])),
    ]
    answer = _structured_llm.invoke(mensajes)          # objeto ClinicalAnswer validado
    return {"answer": answer.model_dump()}             # -> dict idéntico al de antes


def node_evidence(state: RAGState) -> dict:
    """respuesta + índice -> texto final con panel de fuentes y seguimiento."""
    output = format_answer(state["answer"], state["chunk_index"])
    return {"output": output}


# ===========================================================================
# GRAFO:  START ─▶ retrieve ─▶ generate ─▶ evidence ─▶ END
# ===========================================================================
def build_graph():
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("generate", node_generate)
    builder.add_node("evidence", node_evidence)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "evidence")
    builder.add_edge("evidence", END)
    return builder.compile()


# Grafo compilado y reutilizable (lo importan tanto el CLI como la evaluación).
app = build_graph()


def main_cli():
    question = input("¿Cuál es tú pregunta?: ")
    result = app.invoke({"question": question})
    print(result["output"])


if __name__ == "__main__":
    main_cli()
