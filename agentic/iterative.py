"""
agentic/iterative.py — Track A (Phase 4): ITERATIVE / AGENTIC retrieval for MULTI-HOP.

Classic single-shot RAG retrieves evidence for one hop and misses the rest. Here an LLM
(a) decomposes the question into normalized sub-queries (PLAN) and, after a first round,
(b) judges whether the gathered evidence is enough and, if not, emits the next sub-query
(REFLECT — self-ask). Each sub-query reuses the existing hybrid + reranker; results are
pooled (deduped by chunk_id) and finally reranked against the ORIGINAL question. Zero
re-indexing: it only changes HOW we query the store that Qdrant already holds.

Shared primitives are imported from the PARENT module `rag.py` (retrieve_rerank, rerank,
search, the rephrase/abbreviation machinery). This module only adds the loop, so the two
Phase-4 approaches (agentic here, graph in ../graph/) stay cleanly separated while sharing
the same retrieval core and the same downstream generate -> validate -> evidence.
"""
from typing import cast
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# Shared retrieval core lives in the parent module.
from rag import (retrieve_rerank, rerank, search, REPHRASE_MODEL, _ABBREV_LIST,
                 _get_reranker, _get_bm25)

MAX_HOPS = 3        # total retrieval rounds (initial sub-queries + reflect follow-ups)
PER_HOP  = 5        # chunks kept per sub-query before pooling

_PLAN_SYS = f"""Eres un PLANIFICADOR de búsqueda para un asistente sobre las guías clínicas de VIH (GeSIDA). NO respondes la pregunta: la preparas para un buscador.

Decide si la pregunta es MULTI-SALTO (requiere combinar hechos de distintas secciones o guías: p. ej. embarazo + tuberculosis + interacciones, o insuficiencia renal + cobertura de VHB) o de UN SOLO salto.

- Si es de un solo salto: is_multihop=false y sub_queries=[] .
- Si es multi-salto: is_multihop=true y descomponla en 2-3 SUB-CONSULTAS independientes, cada una centrada en UN hecho recuperable. No añadas información clínica nueva que no esté implícita en la pregunta.

NORMALIZACIÓN: en cada sub-consulta, para cualquier fármaco/término de la lista (sigla o nombre), inclúyelo SIEMPRE en AMBAS formas con el patrón «nombre completo (SIGLA)». Solo términos de la lista; no inventes siglas.

LISTA DE ABREVIATURAS (SIGLA = nombre):
{_ABBREV_LIST}

Devuelve is_multihop (bool) y sub_queries (lista de strings en español)."""


_REFLECT_SYS = """Eres un evaluador de SUFICIENCIA de evidencia para responder una pregunta clínica de VIH a partir de fragmentos de guías. NO respondes la pregunta.

Te doy la PREGUNTA original y un resumen de la EVIDENCIA recuperada hasta ahora. Decide si esa evidencia cubre TODOS los aspectos necesarios para responder por completo.

- Si es suficiente: sufficient=true y next_query="" .
- Si falta algún aspecto (un salto no cubierto): sufficient=false y next_query = UNA sub-consulta concreta, en español, que recupere lo que falta. Incluye fármacos/términos en ambas formas «nombre completo (SIGLA)» cuando aplique.

Devuelve sufficient (bool) y next_query (str)."""


class _Plan(BaseModel):
    is_multihop: bool
    sub_queries: list[str]


class _Reflect(BaseModel):
    sufficient: bool
    next_query: str


_plan_llm = None
def _get_plan_llm():
    global _plan_llm
    if _plan_llm is None:
        _plan_llm = ChatOpenAI(model=REPHRASE_MODEL, temperature=0).with_structured_output(
            _Plan, method="json_schema", strict=True
        )
    return _plan_llm


_reflect_llm = None
def _get_reflect_llm():
    global _reflect_llm
    if _reflect_llm is None:
        _reflect_llm = ChatOpenAI(model=REPHRASE_MODEL, temperature=0).with_structured_output(
            _Reflect, method="json_schema", strict=True
        )
    return _reflect_llm


def _plan(query: str) -> dict:
    """Decompose the question into normalized sub-queries (or mark it single-hop). On LLM
    failure it degrades to single-hop so the pipeline never blocks."""
    try:
        p = cast(_Plan, _get_plan_llm().invoke([("system", _PLAN_SYS), ("human", query)]))
        subs = [s.strip() for s in p.sub_queries if s and s.strip()]
        return {"is_multihop": bool(p.is_multihop and subs), "sub_queries": subs}
    except Exception:
        return {"is_multihop": False, "sub_queries": []}


def _evidence_digest(pool: dict, max_chars: int = 200) -> str:
    """Compact view of the pooled chunks for the reflect step (keeps tokens low): the
    section label + the start of each chunk body."""
    lines = []
    for p in pool.values():
        head = (p.get("heading") or "").strip()
        body = (p.get("text") or "").split("\n\n", 1)[-1].strip().replace("\n", " ")
        lines.append(f"- {head}: {body[:max_chars]}")
    return "\n".join(lines)


def _reflect(query: str, pool: dict) -> dict:
    """Judge whether the pooled evidence suffices; if not, return the next sub-query. On
    LLM failure it stops the loop (sufficient=True) to avoid spinning."""
    try:
        r = cast(_Reflect, _get_reflect_llm().invoke([
            ("system", _REFLECT_SYS),
            ("human", f"PREGUNTA:\n{query}\n\nEVIDENCIA RECUPERADA:\n{_evidence_digest(pool)}"),
        ]))
        return {"sufficient": bool(r.sufficient), "next_query": r.next_query.strip()}
    except Exception:
        return {"sufficient": True, "next_query": ""}


def _accumulate(pool: dict, payloads: list) -> None:
    """Add payloads to the pool, deduped by chunk_id (falls back to text)."""
    for p in payloads:
        key = p.get("chunk_id") or p.get("text", "")
        if key and key not in pool:
            pool[key] = p


def iterative_search(query: str, top_k: int = 8, max_hops: int = MAX_HOPS,
                     per_hop: int = PER_HOP) -> list:
    """Track A retriever for multi-hop questions: plan -> retrieve per sub-query ->
    reflect/retrieve follow-ups -> rerank the union against the ORIGINAL question.
    Single-hop questions fall back to the baseline `search` (no extra LLM cost)."""
    plan = _plan(query)
    if not plan["is_multihop"]:
        return search(query, top_k=top_k)

    pool: dict = {}
    subs = plan["sub_queries"][:max_hops]
    # The planned sub-queries are independent, so retrieve+rerank them IN PARALLEL (each does
    # an embedding + Qdrant + cross-encoder). Pre-warm the local models first so the workers
    # don't race on the lazy load. ex.map preserves input order -> dedup is deterministic.
    _get_reranker(); _get_bm25()
    with ThreadPoolExecutor(max_workers=max(1, len(subs))) as ex:
        for payloads in ex.map(lambda sq: retrieve_rerank(sq, top_k=per_hop), subs):
            _accumulate(pool, payloads)
    rounds = len(subs)

    while rounds < max_hops:
        r = _reflect(query, pool)
        if r["sufficient"] or not r["next_query"]:
            break
        _accumulate(pool, retrieve_rerank(r["next_query"], top_k=per_hop))
        rounds += 1

    # Final precision pass: rerank the pooled union against the ORIGINAL question.
    return rerank(query, list(pool.values()), top_k=top_k)
