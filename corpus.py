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

It also owns the two descriptions of the corpus, because they answer the same question ("what
is in here, and where"): the DOCUMENT MANIFEST (`data/corpus.toml`) and the SPECIALTY PROFILES
(`data/specialties/*.toml`). Keeping them together is what lets a specialty's prompt state which
clinical scopes it has specific guidance for without anybody maintaining that list by hand.

This module imports nothing from the project on purpose: `rag.py`, `retrieval/` and `ingestion/`
all need it, and `ingestion/` must stay clear of the retrieval stack.
"""
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MANIFEST_PATH = os.path.join(DATA_DIR, "corpus.toml")
SPECIALTIES_DIR = os.path.join(DATA_DIR, "specialties")
MARKDOWN_DIR = os.path.join(DATA_DIR, "markdown")
REFERENCE_DIR = os.path.join(DATA_DIR, "textos")


@dataclass(frozen=True)
class Layout:
    """The four artifact names of one corpus generation, plus what its chunks can be filtered by."""
    chunks: str          # file under data/chunks/
    lightrag: str        # directory under data/ (graph mode; PathRAG reads the same one)
    hipporag: str        # directory under data/
    collection: str      # Qdrant collection holding the embeddings of `chunks`
    # Whether its chunks carry `specialty`, and therefore whether a search can be confined to
    # one. Stated per generation rather than assumed: Qdrant REJECTS a filter on a field with no
    # payload index, and a generation predating the field has neither. Pretending otherwise
    # would turn every question into «no está en las guías» — a clinical statement produced by a
    # schema mismatch.
    scopable: bool = True


LAYOUTS = {
    # Built by the per-PDF ad-hoc converters. Names are legacy and deliberately not derived:
    # they are what is already in Qdrant Cloud and on this machine's disk. Its chunks have no
    # `specialty`, so it holds exactly one specialty's guidelines and cannot be narrowed.
    "v1": Layout(chunks="chunks.jsonl",
                 lightrag="lightrag_store",
                 hipporag="hipporag_store",
                 collection="guias_vih_hibrida_ctx",
                 scopable=False),
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


# ---------------------------------------------------------------------------
# The document manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Document:
    """One guideline: who wrote it, when, about what, and where its files are."""
    doc_id: str
    title: str
    specialty: str
    topics: tuple[str, ...]
    organization: str
    year: int | None
    language: str
    markdown: str          # file name under data/markdown/
    reference: str         # path under data/textos/ — the PDF's own text layer
    source_pdf: str        # file name under data/pdfs/

    @property
    def markdown_path(self) -> str:
        return os.path.join(MARKDOWN_DIR, self.markdown)

    @property
    def reference_path(self) -> str:
        return os.path.join(REFERENCE_DIR, self.reference)

    @property
    def pdf_path(self) -> str:
        return os.path.join(DATA_DIR, "pdfs", self.source_pdf)


_REQUIRED_DOC_FIELDS = ("doc_id", "title", "specialty", "topics", "organization", "year",
                        "language", "markdown", "reference", "source_pdf")


@lru_cache(maxsize=1)
def _manifest() -> dict:
    with open(MANIFEST_PATH, "rb") as fh:
        return tomllib.load(fh)


@lru_cache(maxsize=1)
def documents() -> tuple[Document, ...]:
    """Every document in the corpus, in manifest order.

    Validation is strict and happens here rather than at the point of use: a document missing a
    field would otherwise surface as a chunk whose year the generation prompt cannot see, and
    the prompt is the thing asked to arbitrate between guidelines spanning 2013 to 2025."""
    out = []
    for i, entry in enumerate(_manifest().get("document", []), 1):
        missing = [f for f in _REQUIRED_DOC_FIELDS if f not in entry]
        if missing:
            raise SystemExit(f"data/corpus.toml: document #{i} "
                             f"({entry.get('doc_id', 'no doc_id')}) is missing {missing}")
        out.append(Document(topics=tuple(entry["topics"]),
                            **{f: entry[f] for f in _REQUIRED_DOC_FIELDS if f != "topics"}))
    ids = [d.doc_id for d in out]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"data/corpus.toml: duplicate doc_id in {ids}")
    return tuple(out)


def document_for_markdown(filename: str) -> Document:
    """The manifest entry for a Markdown file, by name.

    Not finding one is an ERROR. It used to be a shrug: `lookup_meta` returned a default that
    tagged the document `topic = "vih_general"` and let it into the corpus unannounced."""
    for doc in documents():
        if doc.markdown == filename:
            return doc
    raise SystemExit(
        f"{filename} has no entry in data/corpus.toml. Add one (doc_id, title, specialty, "
        f"topics, organization, year, language, markdown, reference, source_pdf) — a document "
        f"whose year and scope the prompt cannot see must not enter the corpus.")


def topics_for(specialty_id: str) -> tuple[str, ...]:
    """The clinical scopes this specialty has DEDICATED guidance for, derived from the manifest.

    This is what SYS_PROMPT's conflict rule needs: a guideline specific to the situation beats a
    general one even when it is older (the pregnancy and TB guides are 2018, the TAR one 2022, so
    "most recent wins" alone would be clinically wrong). It used to be a hand-written list inside
    the prompt, which is to say a copy of the corpus index that nobody updated when the corpus
    changed. Deriving it means adding a document teaches the prompt about it."""
    seen: list[str] = []
    for doc in documents():
        if doc.specialty == specialty_id:
            seen.extend(t for t in doc.topics if t not in seen)
    return tuple(seen)


# ---------------------------------------------------------------------------
# Specialty profiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Modifier:
    """A patient dimension that can change a recommendation.

    Two spellings of one thing: `slug` is the closed vocabulary `refine` screens against,
    `label` is how `assess` names it in prose. They live in one entry because they were two
    independently maintained lists in two prompts, and had already drifted apart."""
    slug: str
    label: str


@dataclass(frozen=True)
class Specialty:
    id: str
    display_name: str
    grade_scheme: str
    prompts: dict          # Spanish fragments woven into the system prompts
    modifiers: tuple[Modifier, ...]
    abbreviations: dict    # ABBREVIATION -> full name

    @property
    def modifier_slugs(self) -> tuple[str, ...]:
        return tuple(m.slug for m in self.modifiers)

    @property
    def modifier_labels(self) -> tuple[str, ...]:
        return tuple(m.label for m in self.modifiers)


_REQUIRED_PROMPT_KEYS = ("subject", "domain", "out_of_domain", "fact_examples",
                         "switch_examples", "general_question_example")


@lru_cache(maxsize=None)
def specialty(specialty_id: str) -> Specialty:
    path = os.path.join(SPECIALTIES_DIR, f"{specialty_id}.toml")
    if not os.path.isfile(path):
        raise SystemExit(f"Unknown specialty {specialty_id!r}: {path} does not exist. "
                         f"Available: {', '.join(specialties())}.")
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    prompts = raw.get("prompts", {})
    missing = [k for k in _REQUIRED_PROMPT_KEYS if not prompts.get(k)]
    if missing:
        raise SystemExit(f"{path}: [prompts] is missing {missing}. Every one of them is woven "
                         f"into a system prompt; an empty value would silently blank a rule.")
    if not raw.get("abbreviations"):
        raise SystemExit(f"{path}: [abbreviations] is empty. It is what makes a sigla and its "
                         f"full name the same term to retrieval, generation and the citation check.")
    return Specialty(
        id=raw.get("id", specialty_id),
        display_name=raw.get("display_name", specialty_id),
        grade_scheme=raw.get("grade_scheme", "gesida"),
        prompts=prompts,
        modifiers=tuple(Modifier(slug=m["slug"], label=m["label"])
                        for m in raw.get("modifier", [])),
        abbreviations=dict(raw["abbreviations"]),
    )


@lru_cache(maxsize=1)
def specialties() -> tuple[str, ...]:
    """Every specialty with a profile on disk, sorted."""
    if not os.path.isdir(SPECIALTIES_DIR):
        return ()
    return tuple(sorted(f[:-5] for f in os.listdir(SPECIALTIES_DIR) if f.endswith(".toml")))


def default_specialty() -> str:
    """The specialty a run answers from when nobody picked one."""
    return os.environ.get("SPECIALTY") or _manifest().get("default_specialty") or ""


@dataclass(frozen=True)
class Scope:
    """Which slice of the corpus a search may reach.

    It lives here, not in the retrieval stack, because it is a statement about the CORPUS, and
    because `rag.py`, `retrieval/` and `evaluation.py` all need to name it. Frozen and explicit
    rather than ambient: passing it through the call chain costs five signatures, and the
    alternative (a contextvar) turns a contract into hidden state — a search that silently reads
    the wrong specialty is exactly the kind of failure this project cannot see.

    ONLY `specialty` filters. Filtering by `topic` was considered and rejected: some subjects
    exist only in the older guidelines, so a topic filter would erase them, whereas letting
    generation arbitrate between guides is visible and reversible."""
    specialty: str = ""

    @property
    def is_open(self) -> bool:
        """True when the scope restricts nothing.

        Either because no specialty was asked for (the evaluation, and any caller deliberately
        searching the whole corpus), or because the ACTIVE corpus generation does not carry the
        field to filter on. The second case is not a silent failure: a generation built before
        specialties existed holds exactly one, so searching all of it IS searching that one."""
        return not self.specialty or not LAYOUT.scopable


def resolve_specialty(specialty_id: str | None) -> str:
    """Normalize a requested specialty, falling back to the default rather than breaking.

    Same posture as the retrieval-mode resolution in the pipeline: an unknown value must not
    take the run down, but it must not silently narrow the search either — falling back to the
    default is the widest safe answer."""
    if specialty_id and specialty_id in specialties():
        return specialty_id
    return default_specialty()
