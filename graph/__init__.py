"""Track B (Phase 4): LightRAG knowledge-graph retrieval for multi-hop questions.

Self-contained package for the GRAPH approach. Shared retrieval primitives (the reranker,
the chunk corpus, the OpenAI/Qdrant clients) live in the PARENT module `rag.py` and in the
parent `chunks/` folder; this package only adds the LightRAG entity-relation graph layer
on top and maps its selected chunks back to our payloads so citations keep working.
"""
