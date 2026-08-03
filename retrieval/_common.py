"""Helpers shared by the retrieval architectures.

Everything here answers one of two questions that every graph-based mode faces:

  1. "I selected some chunks — how do I get back OUR payloads?" The graph modes select chunks
     through their own index (LightRAG content, HippoRAG passage nodes, PathRAG source_id), but
     the pipeline contract requires the ORIGINAL `chunks.jsonl` payloads: only their literal
     `text` is citable and only their metadata renders the sources panel. `map_to_payloads`
     (by text) and `map_chunk_ids_to_payloads` (by id) are the two bridges back.
  2. "How do I turn my selection into the final top_k?" `house_tail` — merge with the shared
     dense+BM25 complement and rerank. Every mode ends the same way, which is what keeps the
     A/B honest: the only variable between modes is HOW chunks are selected, never how they
     are filtered afterwards.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import corpus
from abbreviations import ABBREVIATIONS
from progress import STEP_RETRIEVAL, emit
from rag import (_get_bm25, _get_reranker, rephrase, rerank, retrieve_hybrid,
                 strip_accents)

# Spanish, because it reaches the doctor verbatim as a progress line. It lives here rather than
# in pipeline/config.py for the one reason that module could not serve: `retrieval/` sits below
# the pipeline and importing from it would close a cycle.
MSG_RANKING = "Valorando {n} fragmentos de las guías"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Derived from the active corpus generation, never spelled out: the chunk file and the graph
# stores must describe the SAME text or this module's prefix fallback starts resolving lookups
# to the wrong section (see corpus.py).
CHUNKS_PATH = corpus.chunks_path()


def load_chunks() -> list[dict]:
    """The corpus every index is built from."""
    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def norm(text: str) -> str:
    """Whitespace-collapsed key for matching an index's stored text to our chunks."""
    return " ".join((text or "").split())


_chunk_lookup: dict | None = None
def get_chunk_lookup() -> dict:
    """Map normalized chunk text -> our payload (with full metadata), built once."""
    global _chunk_lookup
    if _chunk_lookup is None:
        _chunk_lookup = {norm(c["text"]): c for c in load_chunks()}
    return _chunk_lookup


_chunks_by_id: dict | None = None
def get_chunks_by_id() -> dict:
    """Map chunk_id -> our payload, built once."""
    global _chunks_by_id
    if _chunks_by_id is None:
        _chunks_by_id = {c["chunk_id"]: c for c in load_chunks()}
    return _chunks_by_id


PREFIX_CHARS = 120
_AMBIGUOUS = object()   # more than one chunk opens this way -> the prefix identifies nobody

_prefix_lookup: dict | None = None
def get_prefix_lookup() -> dict:
    """Map a chunk's opening `PREFIX_CHARS` -> its payload, EXCLUDING every prefix that more
    than one chunk shares.

    Our chunks open with their `A > B > C` breadcrumb, so siblings split out of the same
    section are identical for far longer than this window: 176 of the 517 share their prefix
    with another chunk (42 groups, the largest being an entire guideline whose title alone
    outruns the window — all 45 of its chunks collide). Keeping one payload per prefix would
    silently hand back A DIFFERENT CHUNK, and the sources panel would quote the wrong section.

    Dropping the ambiguous ones is the same call `evidence.attribute` makes: a miss costs the
    answer one chunk, a wrong hit costs it a mis-citation, so every doubt resolves to a miss."""
    global _prefix_lookup
    if _prefix_lookup is None:
        index: dict = {}
        for key, payload in get_chunk_lookup().items():
            prefix = key[:PREFIX_CHARS]
            index[prefix] = _AMBIGUOUS if prefix in index else payload
        _prefix_lookup = {k: v for k, v in index.items() if v is not _AMBIGUOUS}
    return _prefix_lookup


def map_to_payloads(chunks: list[dict]) -> list[dict]:
    """Map an index's selected chunks (by exact content) back to our payloads. Content is
    stored and returned verbatim, so an exact normalized match is expected; a prefix fallback
    covers any edge truncation, and only answers when the prefix names ONE chunk."""
    lookup = get_chunk_lookup()
    out, seen = [], set()
    for ch in chunks:
        content = ch.get("content", "")
        payload = lookup.get(norm(content))
        if payload is None:
            # Prefix fallback, built lazily (see get_prefix_lookup: an ambiguous opening is
            # dropped rather than guessed). The exact match carries every lookup while
            # chunks.jsonl and an index hold the same bytes; editing a chunk without
            # rebuilding the index is what starts routing traffic through here.
            payload = get_prefix_lookup().get(norm(content)[:PREFIX_CHARS])
        if payload is not None:
            key = payload["chunk_id"]
            if key not in seen:
                seen.add(key)
                out.append(payload)
    return out


def map_chunk_ids_to_payloads(chunk_ids) -> list[dict]:
    """Map chunk_ids to our payloads, preserving order and dropping unknown/duplicate ids.
    The id-based counterpart of `map_to_payloads`, for modes that select by id (PathRAG walks
    `source_id`, HippoRAG ranks passage nodes) instead of by text."""
    by_id = get_chunks_by_id()
    out, seen = [], set()
    for cid in chunk_ids:
        if cid in by_id and cid not in seen:
            seen.add(cid)
            out.append(by_id[cid])
    return out


def merge_dedup(*lists) -> list:
    """Concatenate payload lists, dedup by chunk_id (fallback text), preserving order
    (earlier lists win — the mode's own selection first, then the hybrid complement)."""
    out, seen = [], set()
    for lst in lists:
        for p in lst:
            key = p.get("chunk_id") or p.get("text", "")
            if key and key not in seen:
                seen.add(key)
                out.append(p)
    return out


def in_scope(payloads: list, scope: "corpus.Scope | None") -> list:
    """Drop payloads outside the scope.

    The graph modes select through their OWN index (LightRAG content, HippoRAG passage nodes,
    PathRAG source_id), which knows nothing about specialties, so their selection is filtered
    here on the way back rather than at the source. Chunks with no `specialty` are kept: they
    predate the field, and dropping them would silently empty the corpus after an upgrade."""
    if scope is None or scope.is_open:
        return payloads
    return [p for p in payloads
            if p.get("specialty", scope.specialty) == scope.specialty]


def house_tail(query: str, primary: list | Callable[[], list],
               rewritten_query: str | None = None,
               top_k: int = 8, hybrid_k: int = 10,
               scope: "corpus.Scope | None" = None) -> list:
    """The tail EVERY graph-based mode shares: merge its selection with our dense+BM25 hybrid
    complement (dedup, the mode's own chunks first) and rerank the union to top_k.

    Two reasons it is shared rather than per-mode. Comparability: the A/B measures the
    SELECTION mechanism, so the filtering after it must be identical or any score delta is
    confounded. Quality: BM25 covers the guides' heavy abbreviation/dose lexical cases that a
    pure graph walk misses, and the final cross-encoder pass is the precision guardrail that
    `validate` depends on. Pass `rewritten_query` when the caller already rephrased the question
    (the pipeline does) to skip a duplicate LLM call.

    `primary` is either the selection itself or a CALLABLE producing it. The callable form runs
    the mode's selection and the hybrid complement IN PARALLEL, which is worth it when the two
    hit different resources (a local graph walk vs Qdrant+OpenAI over the network) — that is
    the graph mode's case, and the reason it used to keep its own copy of this tail.

    `scope` restricts BOTH halves to one specialty. Since every mode ends here, that makes this
    the single place a search can be confined, which is what keeps the guarantee true for modes
    whose own index has no notion of a specialty at all."""
    _get_reranker(); _get_bm25()  # pre-warm the local models before any parallel branch

    def fetch_hybrid() -> list:
        # The complement is scoped TOO. Filtering only the mode's own selection would leak: the
        # hybrid half would keep pulling another specialty's chunks straight into the context.
        return retrieve_hybrid(rewritten_query or rephrase(query),
                               top_k=hybrid_k, prefetch_limit=30, scope=scope)

    if callable(primary):
        with ThreadPoolExecutor(max_workers=2) as pool:
            selection, hybrid = pool.submit(primary), pool.submit(fetch_hybrid)
            primary, hybrid = selection.result(), hybrid.result()
    else:
        hybrid = fetch_hybrid()

    merged = merge_dedup(in_scope(primary, scope), hybrid)
    if not merged:
        return []
    # The graph modes collapse all of this into ONE graph node, so nothing else reports from
    # here: without this the progress line sits frozen through the whole 5-10 s.
    emit(kind="detail", step=STEP_RETRIEVAL,
         detail=MSG_RANKING.format(n=len(merged)))
    return rerank(query, merged, top_k=top_k)


# --- Abbreviation normalization -------------------------------------------
# Longest first, so "TAR de rescate" is not shadowed by a shorter overlapping abbreviation.
_ABBREV_SORTED = sorted(ABBREVIATIONS.items(), key=lambda kv: -len(kv[0]))


def expand_abbrevs(text: str) -> str:
    """Rewrite guideline abbreviations into the «full name (ABBR)» form the rephrase step
    already uses, so a query term matches index entries written either way."""
    if not text:
        return text
    out = text
    for abbr, name in _ABBREV_SORTED:
        if abbr in out and name.lower() not in out.lower():
            out = out.replace(abbr, f"{name} ({abbr})")
    return out


# Longest NAME first, so «tenofovir alafenamida» contracts before «tenofovir».
_NAME_TO_ABBR = sorted(
    ((strip_accents(name.lower()), abbr.lower()) for abbr, name in ABBREVIATIONS.items()),
    key=lambda kv: -len(kv[0]),
)


def canonical_key(text: str) -> str:
    """Collapse the surface variants of ONE concept into a single key.

    The entity extractor emits the same concept many ways — «DTG», «Dolutegravir»,
    «DTG, Dolutegravir», «Dolutegravir (DTG)» are four separate nodes in the index (163 of
    our 3382 nodes are such duplicates). They fragment every graph signal: each variant holds
    a slice of the edges, so ranking by similarity returns six spellings of one concept
    instead of six concepts. Contracting names to their abbreviation and dropping case,
    accents, punctuation and repeated tokens maps all four to «dtg»."""
    key = strip_accents((text or "").lower())
    for name, abbr in _NAME_TO_ABBR:
        key = key.replace(name, abbr)
    tokens = []
    for token in re.sub(r"[^0-9a-z]+", " ", key).split():
        if token not in tokens:          # «dtg dtg» (from «DTG (Dolutegravir)») -> «dtg»
            tokens.append(token)
    return " ".join(tokens)
