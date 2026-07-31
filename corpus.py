"""Where the corpus artifacts live, and which generation of them is active.

WHY ONE SWITCH. The corpus fans out into four artifacts that must always agree: the chunk file,
the Qdrant collection embedded from it, the LightRAG store (also read by PathRAG) and the
HippoRAG store. Rebuilding them takes about two hours, almost all of it LightRAG, so a rewrite
of the chunking cannot happen atomically — there is necessarily a window where a new chunk file
exists and the old graph stores do not know about it.

That window is dangerous in a specific, measured way. `retrieval/_common.map_to_payloads` bridges
an index's stored text back to our citable payload, falling back to a 120-character prefix when
the exact text no longer matches. Point it at a store built from DIFFERENT chunk text and the
fallback starts answering — with a sibling section. Replaying that state against the real store
resolved 4 of 4 lookups to the WRONG chunk_id: the sources panel would quote one section under a
claim taken from another, which is the single failure this system is built to prevent.

So the four locations are derived from one value instead of being written down four times. A
half-migrated state stops being representable, and the cutover is one line rather than four.

`v1` is the corpus as built by the ad-hoc PDF converters; its names are the ones already live in
Qdrant and on disk, spelled out rather than derived so that nothing has to be rebuilt or renamed
to adopt this module.

This module imports nothing from the project on purpose: `rag.py`, `retrieval/` and `ingestion/`
all need it, and `ingestion/` must stay clear of the retrieval stack.
"""
import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


@dataclass(frozen=True)
class Layout:
    """The four artifact names of one corpus generation."""
    chunks: str          # file under data/chunks/
    lightrag: str        # directory under data/ (graph mode; PathRAG reads the same one)
    hipporag: str        # directory under data/
    collection: str      # Qdrant collection holding the embeddings of `chunks`


LAYOUTS = {
    # Built by the per-PDF ad-hoc converters. Names are legacy and deliberately not derived:
    # they are what is already in Qdrant Cloud and on this machine's disk.
    "v1": Layout(chunks="chunks.jsonl",
                 lightrag="lightrag_store",
                 hipporag="hipporag_store",
                 collection="guias_vih_hibrida_ctx"),
    # Built by ingestion.extract_pdf + the redesigned chunker. The collection name drops the
    # domain: the corpus is no longer assumed to be about one specialty.
    "v2": Layout(chunks="chunks_v2.jsonl",
                 lightrag="lightrag_store_v2",
                 hipporag="hipporag_store_v2",
                 collection="clinical_guidelines_hybrid_ctx"),
}

CORPUS_VERSION = os.environ.get("CORPUS_VERSION", "v1")

if CORPUS_VERSION not in LAYOUTS:
    raise SystemExit(
        f"CORPUS_VERSION={CORPUS_VERSION!r} is not a known corpus generation. "
        f"Valid values: {', '.join(sorted(LAYOUTS))}."
    )

LAYOUT = LAYOUTS[CORPUS_VERSION]


def chunks_path() -> str:
    """The chunk file every index is built from, and the source of truth for citable payloads."""
    return os.path.join(DATA_DIR, "chunks", LAYOUT.chunks)


def lightrag_dir() -> str:
    """The LightRAG entity-graph store (graph mode builds it, PathRAG reads it)."""
    return os.path.join(DATA_DIR, LAYOUT.lightrag)


def hipporag_dir() -> str:
    return os.path.join(DATA_DIR, LAYOUT.hipporag)


def qdrant_collection() -> str:
    """The hybrid collection. `QDRANT_COLLECTION` still overrides it, which is how a run A/Bs
    against a differently-built collection of the SAME generation (e.g. contextual vs plain)."""
    return os.environ.get("QDRANT_COLLECTION") or LAYOUT.collection
