"""Baseline retrieval architecture: one shot of rephrase -> hybrid search -> rerank.

The simplest of the three interchangeable architectures, and the building block the other
two reuse: iterative falls back to one baseline shot for single-hop questions, and graph
runs the same hybrid search as its complement branch.
"""
from rag import rephrase, retrieve_hybrid, rerank


def retrieve_rerank(query: str, top_k: int = 5, candidates: int = 20) -> list:
    """Retrieve `candidates` via hybrid search (dense + BM25 RRF) and reorder them with
    the cross-encoder, returning the best top_k."""
    cand = retrieve_hybrid(query, top_k=candidates, prefetch_limit=30)
    return rerank(query, cand, top_k=top_k)


def search(query: str, top_k: int = 5) -> list:
    """Full baseline retriever: rephrase -> hybrid -> reranker. Used directly by the
    evaluation; the graph runs the same steps as separate nodes (retrieve -> rerank).

    NOTE for the A/B: this default of 5 is smaller than the graph modes' 8, matching what the
    baseline pipeline actually ships (pipeline.nodes._retrieve_for_mode reranks to 5). So a
    run comparing modes varies context SIZE as well as the selection mechanism — pass an
    explicit top_k to control for it. The default is deliberately left alone: changing it would
    silently move the numbers in results/ that were measured at 5."""
    return retrieve_rerank(rephrase(query), top_k=top_k)
