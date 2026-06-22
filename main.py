"""
main.py — Entry point: orchestrates the clinical HIV RAG with LangGraph + LangSmith.

Chains the pipeline functions (rag.py) and the evidence formatting (evidence.py) as
graph nodes so every step can be audited in LangSmith.

Flow:   question -> rephrase -> retrieve -> rerank -> generate <-> validate -> evidence -> output
        - rephrase classifies the domain: if the question is unrelated to HIV, it stops
          here (direct message) and skips retrieve/rerank/generate/validate.
        - validate retries generate when not valid, up to MAX_ITER; if exhausted, safe exit.

Tracing (LangSmith):
    Enabled automatically ONLY if LANGSMITH_API_KEY is set in .env. Then each graph run
    and each OpenAI call (chat and embeddings) shows up at https://smith.langchain.com
    inside the LANGSMITH_PROJECT project. Without that key the graph works the same but
    sends no traces.

Usage:
    python main.py            # interactive mode
"""
import os
import sys

# The Windows console uses cp1252 by default and breaks accents/boxes. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from typing import TypedDict, cast

from dotenv import load_dotenv
load_dotenv()

# --- LangSmith: enable tracing only if there is an API key (otherwise stays out of the way) ---
TRACING = bool(os.environ.get("LANGSMITH_API_KEY"))
if TRACING:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "chatbot-vih")

# --- RAG pipeline (retrieval/generation functions and clients) ---
import rag  # creates the OpenAI/Qdrant clients on import
from rag import (refine, retrieve_hybrid, rerank, build_context, validate,
                 SYS_PROMPT, build_user_prompt, GENERATION_MODEL)
from agentic.iterative import iterative_search   # Track A (agentic, multi-hop)
from evidence import format_answer

# --- Fine-grained tracing of OpenAI calls (tokens, latency) WITHOUT modifying rag.py ---
# rag's functions read `rag.client` at call time, so reassigning here its LangSmith-wrapped
# version is enough for retrieve() and the embeddings to be traced inside the graph run.
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
# STRUCTURED GENERATION (LangChain) — replaces the raw response_format. Under the
# hood it uses the SAME OpenAI mechanism (Structured Outputs, strict json_schema),
# but with Pydantic validation and native LangSmith tracing. We return .model_dump()
# so the rest of the pipeline keeps receiving the same dict as before.
# ===========================================================================
class SourceUsed(BaseModel):
    ref: int
    quote: str


class ClinicalAnswer(BaseModel):
    sufficient_information: bool
    answer: str
    sources_used: list[SourceUsed]
    follow_up_questions: list[str]


_structured_llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0.2).with_structured_output(
    ClinicalAnswer, method="json_schema", strict=True
)

# Validation loop: max number of generations (initial + retries).
MAX_ITER = 2

# Retrieval strategy (Phase 4) — switch here (also works live in LangGraph Studio):
#   "baseline"  -> hybrid + reranker (single-shot, current system)
#   "iterative" -> Track A (agentic): self-ask / reflect-retrieve loop for multi-hop
#                  (single-hop questions fall back to baseline inside iterative_search)
#   "graph"     -> Track B: LightRAG entity-relation graph retrieval (needs the index
#                  built once: python -m graph.lightrag_track)
# All three feed the SAME generate -> validate -> evidence, so citations and the
# anti-hallucination validator are identical across strategies.
# F4 A/B decision: "graph" wins context_recall on the multi-hop set (0.98 vs 0.86
# iterative vs 0.84 baseline) at baseline-level latency (~11s), so it is the default.
RETRIEVAL_MODE = "graph"

# --- User-facing messages (kept in Spanish: shown to the user) ---
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


# ===========================================================================
# STATE
#   InputState: the only thing to pass to app.invoke() -> the question.
#   RAGState:   full internal state that flows through the graph. Keys are required
#               (each node fills them in) so that accessing state["..."] is safe for
#               the type checker. Nodes return only the keys they produce and LangGraph
#               merges them.
# ===========================================================================
class InputState(TypedDict):
    question: str            # user input


class RAGState(TypedDict):
    question: str            # user input (original; used for generation)
    in_domain: bool          # is the question within the HIV domain? (classified in rephrase)
    search_query: str        # rewritten/normalized query for the retriever (rephrase)
    candidates: list         # retrieved payloads (hybrid, ~20) before reranking
    contexts: list           # final payloads after the reranker (top 5)
    chunk_index: dict        # {n: chunk} for source citation (build_context)
    formatted_context: str   # numbered context [1]/[2]… for the LLM
    answer: dict             # structured answer from the LLM (generate_answer)
    attempts: int            # number of generations done (for the validation loop)
    validation: dict         # validator verdict {is_valid, reason, ...}
    output: str              # final formatted text with sources (format_answer)


# ===========================================================================
# NODES — each one wraps a pipeline function.
#   They receive the full state and return ONLY the keys they produce.
# ===========================================================================
def node_rephrase(state: RAGState) -> dict:
    """question -> rewritten/normalized query + domain classification (single LLM call).
    If the question is unrelated to HIV, the pipeline is short-circuited (see
    route_domain): retrieve, rerank, generate and validate are skipped."""
    r = refine(state["question"])
    return {"search_query": r["query"], "in_domain": r["in_domain"]}


def node_out_of_domain(state: RAGState) -> dict:
    """Question outside the HIV domain: direct message, no retrieval or generation."""
    return {"output": MSG_OUT_OF_DOMAIN}


def route_domain(state: RAGState) -> str:
    """After rephrase: out of domain -> message and end; in domain -> the configured
    retrieval strategy (baseline single-shot, Track A iterative, or Track B graph)."""
    if not state.get("in_domain", True):
        return "out_of_domain"
    return {
        "iterative": "iterative_retrieve",
        "graph": "graph_retrieve",
    }.get(RETRIEVAL_MODE, "retrieve")


def node_retrieve(state: RAGState) -> dict:
    """search_query -> ~20 candidates via hybrid search (dense + BM25)."""
    candidates = retrieve_hybrid(state["search_query"], top_k=20, prefetch_limit=30)
    return {"candidates": candidates}


def node_rerank(state: RAGState) -> dict:
    """candidates -> top 5 reordered by the cross-encoder + numbered context."""
    contexts = rerank(state["search_query"], state["candidates"], top_k=5)
    chunk_index, formatted_context = build_context(contexts)
    return {
        "contexts": contexts,
        "chunk_index": chunk_index,
        "formatted_context": formatted_context,
    }


def node_iterative_retrieve(state: RAGState) -> dict:
    """Track A: multi-hop retrieval (plan -> hop retrieve -> reflect) producing the same
    numbered-context shape the baseline rerank node yields, so generate/validate/evidence
    (and the literal-citation panel) work unchanged. Uses the ORIGINAL question; the
    iterative search does its own decomposition and normalization."""
    contexts = iterative_search(state["question"], top_k=8)
    chunk_index, formatted_context = build_context(contexts)
    return {
        "contexts": contexts,
        "chunk_index": chunk_index,
        "formatted_context": formatted_context,
    }


def node_graph_retrieve(state: RAGState) -> dict:
    """Track B: LightRAG graph retrieval. Selects source chunks via entity/relation
    traversal and maps them back to our payloads, yielding the same numbered-context shape
    as the other retrieval nodes so generate/validate/evidence are unchanged. Imported
    lazily so baseline/iterative runs do not load LightRAG."""
    from graph.lightrag_track import graph_search
    contexts = graph_search(state["question"], top_k=8)
    chunk_index, formatted_context = build_context(contexts)
    return {
        "contexts": contexts,
        "chunk_index": chunk_index,
        "formatted_context": formatted_context,
    }


def node_generate(state: RAGState) -> dict:
    """context + question -> structured answer from the LLM. If a previous attempt was
    rejected by the validator, inject its feedback to correct it."""
    user = build_user_prompt(state["question"], state["formatted_context"])
    val = state.get("validation")
    if val and not val.get("is_valid", True):
        user += (
            f"\n\n    REINTENTO: tu respuesta anterior fue RECHAZADA por el validador. "
            f"Motivo: {val.get('reason', '')}. Corrige la respuesta para que cada afirmación "
            f"esté respaldada por el contexto y aborde la pregunta. Si el contexto no respalda "
            f"la respuesta, marca informacion_suficiente=false."
        )
    messages = [("system", SYS_PROMPT), ("human", user)]
    answer = cast(ClinicalAnswer, _structured_llm.invoke(messages))  # validated by Pydantic
    return {"answer": answer.model_dump(), "attempts": state.get("attempts", 0) + 1}


def node_validate(state: RAGState) -> dict:
    """Relevance + grounding judge over the answer. Marks valid / not valid."""
    verdict = validate(state["question"], state["answer"], state["formatted_context"])
    return {"validation": verdict}


def node_evidence(state: RAGState) -> dict:
    """answer + index -> final text with the sources and follow-up panel."""
    output = format_answer(state["answer"], state["chunk_index"])
    return {"output": output}


def node_fallback(state: RAGState) -> dict:
    """The answer is not shown. The message depends on the reason: technical error of
    the validator vs. no valid answer reached after the retries."""
    if state["validation"].get("error", False):
        return {"output": MSG_VALIDATION_ERROR}
    return {"output": MSG_NOT_VALIDATED}


def route_validation(state: RAGState) -> str:
    """After validating:
      - technical error of the validator -> fallback (do not retry; the judge is down).
      - valid -> format (evidence).
      - not valid and attempts left -> regenerate with feedback.
      - not valid and exhausted -> fallback (safe exit)."""
    v = state["validation"]
    if v.get("error", False):
        return "fallback"
    if v.get("is_valid", False):
        return "evidence"
    if state.get("attempts", 0) >= MAX_ITER:
        return "fallback"
    return "generate"


# ===========================================================================
# GRAPH:  START -> rephrase -┬- out of domain -> out_of_domain -> END
#                           └- in domain -> retrieve -> rerank -> generate -> validate -┬-> evidence -> END
#                                                          ▲ (retry)  │                 ├-> generate (loop)
#                                                          └──────────┘                 └-> fallback -> END
# ===========================================================================
def build_graph():
    builder = StateGraph(RAGState, input_schema=InputState)
    builder.add_node("rephrase", node_rephrase)
    builder.add_node("out_of_domain", node_out_of_domain)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("rerank", node_rerank)
    builder.add_node("iterative_retrieve", node_iterative_retrieve)
    builder.add_node("graph_retrieve", node_graph_retrieve)
    builder.add_node("generate", node_generate)
    builder.add_node("validate", node_validate)
    builder.add_node("evidence", node_evidence)
    builder.add_node("fallback", node_fallback)

    builder.add_edge(START, "rephrase")
    # Domain guardrail: in -> retrieve; out -> direct message (cuts the pipeline)
    builder.add_conditional_edges("rephrase", route_domain,
                                  {"retrieve": "retrieve",
                                   "iterative_retrieve": "iterative_retrieve",
                                   "graph_retrieve": "graph_retrieve",
                                   "out_of_domain": "out_of_domain"})
    builder.add_edge("out_of_domain", END)
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("iterative_retrieve", "generate")
    builder.add_edge("graph_retrieve", "generate")
    builder.add_edge("generate", "validate")
    # Loop: validate -> evidence (valid) | generate (retry) | fallback (exhausted)
    builder.add_conditional_edges("validate", route_validation,
                                  {"evidence": "evidence", "generate": "generate",
                                   "fallback": "fallback"})
    builder.add_edge("evidence", END)
    builder.add_edge("fallback", END)
    return builder.compile()


# Compiled, reusable graph (imported by both the CLI and the evaluation).
app = build_graph()


def main_cli():
    question = input("¿Cuál es tú pregunta?: ")
    result = app.invoke({"question": question})
    print(result["output"])


if __name__ == "__main__":
    main_cli()
