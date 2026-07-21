"""The catalogue of retrieval architectures.

One source of truth for "which retrieval modes exist and how do I call one". Before this,
adding a mode meant editing the same if/elif in six places (routing, re-retrieval, graph
assembly, the Studio dropdown, the evaluation A/B) and every one of them was a chance to
forget a branch. Now a mode is declared once here and the rest is data-driven.

Each mode resolves to a `search(query, top_k=..., rewritten_query=...) -> list[payload]`
function returning our original chunk payloads — the contract the pipeline tail depends on.
Imports are LAZY (a mode is loaded only when actually used) because the graph modes pull in
LightRAG, networkx and a store from disk, which no baseline run should pay for.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Mode:
    """A retrieval architecture: how to load it and how it is described to the user."""
    name: str
    description: str
    _loader: Callable[[], Callable]
    _concept_loader: Callable[[], Callable] | None = None

    def search(self) -> Callable:
        """The mode's retrieval function (imported on first use)."""
        return self._loader()

    def search_with_concept_map(self) -> Callable | None:
        """Retrieval that ALSO returns a non-citable concept map — a textual view of the graph
        structure behind the selection, which the generator may reason over but never quote.
        None for modes that produce no such map, which is why it is optional rather than part
        of every signature: the pipeline asks, and adapts."""
        return self._concept_loader() if self._concept_loader else None


def _baseline():
    from .baseline import search
    return search


def _iterative():
    from .iterative import iterative_search
    return iterative_search


def _graph():
    from .graph import graph_search
    return graph_search


def _pathrag():
    from .pathrag import pathrag_search
    return pathrag_search


def _pathrag_with_paths():
    from .pathrag import pathrag_search_with_paths
    return pathrag_search_with_paths


def _hipporag():
    from .hipporag import hipporag_search
    return hipporag_search


MODES: dict[str, Mode] = {
    m.name: m for m in (
        Mode("baseline", "hybrid (dense + BM25) + cross-encoder rerank, single shot",
             _baseline),
        Mode("iterative", "Track A: self-ask plan -> retrieve per sub-query -> reflect",
             _iterative),
        Mode("graph", "Track B: LightRAG entity-relation traversal + hybrid complement",
             _graph),
        Mode("pathrag", "PathRAG: flow-pruned relational paths over the LightRAG index",
             _pathrag, _pathrag_with_paths),
        Mode("hipporag", "HippoRAG 2: open KG + Personalized PageRank over passage nodes",
             _hipporag),
    )
}

VALID_MODES = tuple(MODES)


def get_search(name: str) -> Callable:
    """Retrieval function for `name`, or SystemExit naming the valid modes."""
    mode = MODES.get(name)
    if mode is None:
        raise SystemExit(f"Unknown retrieval mode: {name!r} (use {' | '.join(VALID_MODES)})")
    return mode.search()
