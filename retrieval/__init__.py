"""The three interchangeable retrieval architectures (the RETRIEVAL_MODE / PIPELINE values).

Each module exposes ONE entry point honouring the same contract — question in, list of chunk
payloads out — which is what keeps the rest of the pipeline retrieval-agnostic:

  - baseline.py  -> search()            single shot: rephrase -> hybrid (dense+BM25) -> rerank
  - iterative.py -> iterative_search()  Track A: plan -> hop -> reflect (multi-hop self-ask)
  - graph.py     -> graph_search()      Track B: LightRAG entity-relation graph traversal

All three compose the shared primitives in rag.py and feed the SAME generate -> validate ->
evidence tail. No eager imports here: graph.py pulls in LightRAG, which must stay lazy so
baseline/iterative runs never pay for it.
"""
