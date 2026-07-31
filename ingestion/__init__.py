"""Corpus ingestion: PDF -> Markdown -> chunks -> index.

A package rather than a folder of loose scripts because the steps share vocabulary (the chunk
schema, the quality gates, the size budget) and the tests import those definitions. Run each
step with `python -m ingestion.<step>`, the same way `retrieval/` is invoked.
"""
