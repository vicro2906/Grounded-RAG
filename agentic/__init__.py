"""Track A (Phase 4): agentic / iterative retrieval for multi-hop questions.

Self-contained package for the AGENTIC approach. Shared retrieval primitives (hybrid
search, reranker, clients, abbreviations) live in the PARENT module `rag.py`; this
package only adds the iterative plan -> hop-retrieve -> reflect loop on top of them.
"""
