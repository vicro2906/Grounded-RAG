"""The interchangeable retrieval architectures (the RETRIEVAL_MODE / PIPELINE values).

Each module exposes ONE entry point honouring the same contract — question in, list of chunk
payloads out — which is what keeps the rest of the pipeline retrieval-agnostic:

  - baseline.py  -> search()            single shot: rephrase -> hybrid (dense+BM25) -> rerank
  - iterative.py -> iterative_search()  Track A: plan -> hop -> reflect (multi-hop self-ask)
  - graph.py     -> graph_search()      Track B: LightRAG entity-relation graph traversal
  - pathrag.py   -> pathrag_search()    flow-pruned relational paths over that same index
  - hipporag.py  -> hipporag_search()   HippoRAG 2: open KG + Personalized PageRank

They are declared in registry.py (the catalogue everything else derives from) and all compose
the shared primitives in rag.py, feeding the SAME generate -> validate -> evidence tail. No
eager imports of a MODE here: graph.py pulls in LightRAG, which must stay lazy so
baseline/iterative runs never pay for it.

`merge_dedup` is re-exported because combining two payload lists is not mode-specific: the
pipeline needs it whenever it composes retrievals (the validator-driven re-retrieval merges
the focused pass with the context already in hand).
"""
from ._common import merge_dedup

__all__ = ["merge_dedup"]
