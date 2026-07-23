"""Graph state schemas.

`RAGState` is the contract every retrieval mode honours (it fills contexts / chunk_index /
formatted_context); the tail reads only that contract, which is what makes the pipeline
retrieval-agnostic. `IterativeState` / `GraphState` add the intermediate keys the dedicated
"teaching" graphs surface in Studio.
"""
from typing import Annotated, TypedDict


class InputState(TypedDict):
    question: str            # the only field app.invoke() needs


def _add_int(a: int | None, b: int | None) -> int:
    return (a or 0) + (b or 0)


def _merge_pool(existing: list | None, update: list | None) -> list:
    """Reducer: append payloads not already present (dedup by chunk_id, fallback text). Lets
    the parallel fan-out branches and successive loop rounds accumulate into one pool key."""
    pool = list(existing or [])
    seen = {(p.get("chunk_id") or p.get("text", "")) for p in pool}
    for p in (update or []):
        key = p.get("chunk_id") or p.get("text", "")
        if key and key not in seen:
            seen.add(key)
            pool.append(p)
    return pool


class RAGState(TypedDict):
    question: str            # original user input (used for generation)
    retrieval_mode: str      # which retrieval architecture ran (set by the retrieval node)
    in_domain: bool          # is the question within the HIV domain? (rephrase)
    search_query: str        # rewritten/normalized query for the retriever (rephrase)
    candidates: list         # retrieved payloads (hybrid, ~20) before reranking
    contexts: list           # final payloads after the reranker
    chunk_index: dict        # {n: chunk} for source citation (build_context)
    formatted_context: str   # numbered context [1]/[2]… for the LLM
    # Optional part of the retrieval contract: a mode MAY expose the graph structure behind
    # its selection (PathRAG's relational paths) as text. It reaches generation as a
    # NON-CITABLE block — reasoning aid only, never a source. "" for the modes without one.
    concept_map: str
    # --- Patient data: steers generation and retrieval, never cited ------------------
    # SESSION-SCOPED, and the ONE field that deliberately survives across questions: the same
    # patient spans several questions, so `node_rephrase` ACCUMULATES this turn's facts into it
    # instead of resetting it. It is cleared only on an explicit "new patient" (the CLI's
    # /nuevo, or update_state), so the previous patient's renal function can never silently
    # steer the next patient's answer. No reducer — every writer merges by hand (rephrase folds
    # in the question's facts, `_fold_answers` the refinement's), enough as they run in order.
    patient_facts: dict           # accumulated {attr: value}; survives the question, cleared per patient
    candidate_modifiers: list     # cheap screen (refine): modifiers the question might need
    assessment: dict              # assess reasoning {branches_on, clinically_relevant, already_covered}
    pending_clarifications: list  # dimensions still unknown: unknown-block in the prompt, then
                                  # offered as the optional refinement once the answer is out
    asked_questions: list         # every clarifying question already asked (assess never repeats)
    clarify_rounds: int           # refinement rounds done THIS question
    refining: bool                # the doctor supplied a datum -> re-retrieve and answer again
    answer: dict                  # structured answer from the LLM (node_generate)
    attempts: int                 # generations done so far (validation loop)
    validation: dict              # validator verdict {is_valid, reason, ...}
    refocus_query: str            # claims the validator rejected, re-retrieved before retrying
    # A step could not run at all (service unreachable, index unavailable…). `technical_error`
    # holds the user-facing step label and short-circuits the run to a message; the exception
    # goes to `technical_detail`, which is NEVER shown but keeps the trace diagnosable —
    # swallowing an exception without recording it just turns an outage into a later mystery.
    technical_error: str
    technical_detail: str
    output: str                   # final formatted text with sources (format_answer)


class IterativeState(RAGState, total=False):
    """Extra keys for the expanded iterative graph (Track A teaching view)."""
    is_multihop: bool                      # did the planner mark the question multi-hop?
    planned: list                          # sub-questions produced by the planner
    subquery: str                          # the single sub-question handed to one fan-out branch
    next_query: str                        # follow-up sub-question from reflect ("" = done)
    pool: Annotated[list, _merge_pool]     # payloads accumulated across all branches/rounds
    hops: Annotated[int, _add_int]         # sub-queries retrieved so far (capped at MAX_HOPS)


class GraphState(RAGState, total=False):
    """Extra keys for the expanded graph (Track B teaching view)."""
    graph_hl: list              # high-level keywords (-> relations) extracted by the LLM
    graph_ll: list              # low-level keywords (-> entities) extracted by the LLM
    graph_entities: list        # entities the cosine/graph search selected (for visibility)
    graph_relationships: list   # relations the cosine/graph search selected (for visibility)
    graph_payloads: list        # source chunks gathered from the selected entities/relations
    hybrid_payloads: list       # chunks from our dense+BM25 hybrid complement
    merged: list                # the two sources merged & deduped, before final rerank
