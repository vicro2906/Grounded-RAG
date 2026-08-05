#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunk_guidelines.py
===================
Splits clinical guidelines in Markdown into structured *chunks* with metadata, ready to index.

Section (raw text per heading, irregular size) → normalize adjusts the size and produces units
(well-sized text + inherited structural metadata) → build_chunks attaches the metadata that
depends on the final text → Chunk (serialized to JSONL and uploaded to Qdrant).

Strategy: STRUCTURE-AWARE (header-aware) chunking with size normalization.
  1. Parse each .md into a tree of sections by its headings, keeping the hierarchical path.
  2. Each "leaf" section (text up to the next heading) is the base unit.
  3. Normalize size against a budget that reserves room for the breadcrumb, since that is part
     of what gets embedded:
        - large sections -> split by paragraph, then by row (tables), record (wide matrices)
          or sentence, carrying a sentence-level overlap forward;
        - small sections -> merged with anything inside their parent's subtree, which is what
          keeps a graded recommendation attached to the reasoning it follows.
  4. Tag each chunk with metadata (document identity, path, content type, evidence grades).
  5. Export to JSONL (one chunk per line).

TWO TEXTS PER CHUNK, and the distinction is load-bearing:
  * `text`               — the citable literal. `evidence.py` quotes it back to the doctor.
  * `text_for_retrieval` — breadcrumb + text. This is what is embedded, BM25-indexed and fed
                           to the graph extractors, and what the size budget applies to.
They used to be one field with the breadcrumb prepended, which made 178 of 517 chunks
byte-identical for longer than the window used to map an index entry back to its payload.

Usage:
    python -m ingestion.chunk_guidelines data/markdown -o data/chunks/chunks.jsonl
    python -m ingestion.chunk_guidelines file1.md file2.md -o output.jsonl
    python -m ingestion.chunk_guidelines data/markdown --check     # gates only, writes nothing
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import corpus

# ---------------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------------

# Target sizes expressed in TOKENS (approx). Tunable to your embedding model.
# ~512-800 tokens is a good range for clinical retrieval.
TARGET_TOKENS = 600     # ideal size of a chunk
MAX_TOKENS    = 900     # above this a section is split
MIN_TOKENS    = 200     # below this we try to merge
OVERLAP_TOKENS = 80     # overlap when splitting large sections

# Document identity comes from `data/corpus.toml` (see corpus.documents). It used to be a
# DOC_REGISTRY dict right here, with a DEFAULT_META fallback that let an unregistered .md into
# the corpus SILENTLY, tagged `topic = "vih_general"` — a document whose year and scope the
# generation prompt then could not see, while that same prompt is asked to arbitrate between
# guidelines spanning 2013 to 2025. `corpus.document_for_markdown` raises instead.

# ---------------------------------------------------------------------------
# 1. TOKEN COUNTING
# ---------------------------------------------------------------------------
# `cl100k_base` is the tokenizer of the EMBEDDER (text-embedding-3-large), which is what
# actually imposes a limit (8191 tokens per input). gpt-4o uses o200k_base and does not decide
# here: a chunk is sized to be embedded, not to be generated from.
#
# There is NO character-based fallback, and that is the point. This module used to degrade to
# `len(text)/4` when tiktoken was missing, and the corpus was in fact built that way: every
# chunk in data/chunks/chunks.jsonl has n_tokens == round(n_chars/4). That estimate
# underquotes real Spanish clinical prose by 14% at the median and by 2.16x at worst, so 89 of
# the 517 chunks (17%) silently exceeded the MAX_TOKENS this module claims to enforce. A
# sizing unit that can quietly become a different unit is worse than a missing dependency,
# and tiktoken is a declared one (pyproject.toml).
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))

# ---------------------------------------------------------------------------
# 2. REGULAR EXPRESSIONS
# ---------------------------------------------------------------------------
RE_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Section number at the start of the heading: "3.2.2." or "1." -> "3.2.2"
RE_SECNUM  = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
# Evidence grades: (A-I), (A-II), (B-III), also (AII), (B-I), with/without **
RE_GRADE   = re.compile(r"\(\s*\*{0,2}\s*([ABC])\s*-?\s*(I{1,3})\s*\*{0,2}\s*\)")
RE_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
# The separator under a Markdown table's header row (`|---|---|`). Splitting a table across
# chunks means repeating the header AND this line, or the pieces stop being tables.
RE_TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
# A wide matrix ships as one record per entity-row instead of an unreadable 19-column grid, so
# "this text holds table data" has to accept both shapes. See ingestion/extract_pdf.py.
RE_RECORD_ROW = re.compile(r"^\s*[-*]\s+\S.*?:\s+\S", re.MULTILINE)

# Headings that mark a special content type (match Spanish guideline text).
RE_RECS    = re.compile(r"recomendaci(o|ó)n", re.IGNORECASE)
RE_ABREV   = re.compile(r"abreviatura|listado de abreviaturas", re.IGNORECASE)
RE_TABLA_H = re.compile(r"^(tabla|figura)\b", re.IGNORECASE)
RE_ANEXO   = re.compile(r"\banexo|algoritmo", re.IGNORECASE)
RE_METODO  = re.compile(r"metodolog(i|í)a", re.IGNORECASE)


# A breadcrumb is context for the embedding, not content, so it may never take more than this
# share of a chunk. The cap is a GUARD, not a feature: a malformed heading upstream (a paragraph
# promoted to a heading) produced breadcrumbs of 1 228 tokens, more than the whole budget, and
# every chunk beneath that heading inherited it. Fixing the extractor removes the cause; the cap
# makes the chunker unable to emit an over-budget chunk whatever its input looks like.
MAX_BREADCRUMB_TOKENS = 120


def breadcrumb_prefix(breadcrumb: List[str]) -> str:
    """The `A > B > C` context line prepended to `text_for_retrieval`, trimmed from the LEFT if
    it is oversized: the deepest headings are the specific ones, so an ancestor is what a search
    can most afford to lose."""
    if not breadcrumb:
        return ""
    parts = list(breadcrumb)
    while len(parts) > 1 and count_tokens(" > ".join(parts) + "\n\n") > MAX_BREADCRUMB_TOKENS:
        parts.pop(0)
    prefix = " > ".join(parts)
    if count_tokens(prefix + "\n\n") > MAX_BREADCRUMB_TOKENS:
        # A single heading longer than the whole allowance: keep its opening, which is where the
        # section number and the topic live.
        prefix = prefix[:MAX_BREADCRUMB_TOKENS * 4].rstrip() + "…"
    return f"{prefix}\n\n"


# ---------------------------------------------------------------------------
# 3. DATA STRUCTURES
# ---------------------------------------------------------------------------
@dataclass
class Section:
    """A 'leaf' section: a heading and the text up to the next heading."""
    level: int
    heading: str
    body: str
    breadcrumb: List[str]          # ancestor headings, including its own
    section_number: Optional[str]  # "3.2.2"  or None

    @property
    def tokens(self) -> int:
        return count_tokens(self.body)

    @property
    def overhead(self) -> int:
        """Tokens the breadcrumb will add when this section becomes `text_for_retrieval`.

        The budget applies to the string that is EMBEDDED, so it has to be reserved here rather
        than discovered afterwards. It is not a rounding error: a table caption promoted to a
        heading contributes a breadcrumb of several hundred characters to every chunk under it."""
        return count_tokens(breadcrumb_prefix(self.breadcrumb))

    @property
    def budget(self) -> int:
        """How many tokens of BODY still fit under MAX_TOKENS once the breadcrumb is added."""
        return max(MAX_TOKENS - self.overhead, MIN_TOKENS)


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    doc_id: str                    # manifest identity, stable across renames of the .md
    doc_title: Optional[str]
    specialty: str                 # what scopes a search to one medical area (Qdrant filter)
    topics: List[str]              # clinical scopes; feeds SYS_PROMPT's specific-vs-general rule
    organization: str
    year: Optional[int]
    section_path: List[str]        # heading breadcrumb
    section_number: Optional[str]
    heading: str
    heading_level: int
    content_type: str              # text|recommendations|table|abbreviations|appendix|methodology
    evidence_grades: List[str]
    chunk_index: int               # index within the document
    n_tokens: int
    n_chars: int
    text: str                      # the CITABLE literal: what evidence.py quotes back
    text_for_retrieval: str        # breadcrumb + text: what gets embedded and BM25-indexed


# ---------------------------------------------------------------------------
# 4. PARSING -> list of Section
# ---------------------------------------------------------------------------
def parse_sections(md_text: str) -> List[Section]:
    """Split the markdown into leaf sections keeping the heading path."""
    lines = md_text.splitlines()
    sections: List[Section] = []
    stack: List[tuple] = []        # [(level, heading_text), ...]
    cur_level, cur_heading, cur_breadcrumb = 0, "Preámbulo", []
    buf: List[str] = []

    def flush():
        body = "\n".join(buf).strip()
        # We also keep empty sections only if they have a real heading;
        # empty ones will later be merged or dropped during normalization.
        sections.append(Section(
            level=cur_level,
            heading=cur_heading,
            body=body,
            breadcrumb=list(cur_breadcrumb),
            section_number=_secnum(cur_heading),
        ))

    for line in lines:
        m = RE_HEADING.match(line)
        if m:
            # close the previous section
            flush()
            level = len(m.group(1))
            heading = m.group(2).strip()
            # update the ancestor stack
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            cur_level = level
            cur_heading = heading
            cur_breadcrumb = [h for (_, h) in stack]
            buf = []
        else:
            buf.append(line)
    flush()

    # drop the "Preámbulo" block if it is empty
    return [s for s in sections if not (s.heading == "Preámbulo" and not s.body)]


def _secnum(heading: str) -> Optional[str]:
    m = RE_SECNUM.match(heading)
    return m.group(1) if m else None


def _common_secnum(numbers: List[Optional[str]]) -> Optional[str]:
    """Common numeric prefix of several section numbers.
    Used to correctly label a chunk that merges subsections:
        ['4.2.1', '4.2.2']        -> '4.2'
        ['4.2.4.1', '4.2.4.2']    -> '4.2.4'
        ['4.1', '4.2']            -> '4'
        ['4.1', '4.1.1']          -> '4.1'
    Returns None if there is no common prefix (numbers from different branches).
    """
    nums = [n for n in numbers if n]
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    split = [n.split(".") for n in nums]
    common: List[str] = []
    for parts in zip(*split):
        if all(p == parts[0] for p in parts):
            common.append(parts[0])
        else:
            break
    return ".".join(common) if common else None


def _ancestor_for_secnum(breadcrumb: List[str], secnum: Optional[str]):
    """Locate in the breadcrumb the ancestor whose section number is `secnum`.
    Returns (ancestor_heading, breadcrumb_truncated_up_to_it) or (None, None)
    if not found (e.g. the ancestor was a container without its own heading).
    """
    if not secnum:
        return None, None
    for k, h in enumerate(breadcrumb):
        if _secnum(h) == secnum:
            return h, breadcrumb[:k + 1]
    return None, None


# ---------------------------------------------------------------------------
# 5. CLASSIFICATION AND METADATA EXTRACTION
# ---------------------------------------------------------------------------
def classify(sec: Section) -> str:
    """The chunk's content type, from its heading AND its body — never from the heading alone.

    A heading saying «TABLA 3» over a body with no rows used to be typed `table` anyway. That is
    how 14 chunks came to be labelled as tables while holding nothing but a caption and some
    footnotes, and `content_type` is a Qdrant filter: the label has to describe the payload, not
    the intention."""
    h = sec.heading
    if RE_ABREV.search(h):
        return "abbreviations"
    if RE_RECS.search(h) and sec.level >= 4:
        return "recommendations"
    if RE_TABLA_H.match(h):
        # The caption promises a table; only the body can confirm one.
        return "table" if has_tabular_content(sec.body) else "text"
    if RE_ANEXO.search(h):
        return "appendix"
    if RE_METODO.search(h):
        return "methodology"
    if is_mostly_table(sec.body):
        return "table"
    return "text"


def is_mostly_table(body: str) -> bool:
    """True when the body is an 'embedded' table: enough rows, and dominated by them.

    Shared with the quality gates so the classifier and its auditor cannot disagree about what
    counts as a table — a gate calibrated to a different threshold than the code it checks
    reports failures nobody can act on."""
    lines = [l for l in body.splitlines() if l.strip()]
    if not lines:
        return False
    rows = sum(1 for l in lines if RE_TABLE_ROW.match(l))
    return rows >= 3 and rows / len(lines) > 0.6


def extract_grades(text: str) -> List[str]:
    """Return normalized grades ('A-II', 'B-III'...) without duplicates, in order."""
    out, seen = [], set()
    for letter, roman in RE_GRADE.findall(text):
        g = f"{letter.upper()}-{roman.upper()}"
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def has_table(body: str) -> bool:
    return any(RE_TABLE_ROW.match(l) for l in body.splitlines())


def has_tabular_content(body: str) -> bool:
    """True if the text holds table data in EITHER supported shape.

    Owned here rather than in the quality gates because this module is what decides the shape;
    a gate calibrated to a different definition than the code it audits reports failures nobody
    can act on."""
    return has_table(body) or bool(RE_RECORD_ROW.search(body))


# ---------------------------------------------------------------------------
# 6. SIZE NORMALIZATION (merge + split)
# ---------------------------------------------------------------------------
# Sentence end: a full stop, question or exclamation mark followed by whitespace. Deliberately
# not a full sentence tokenizer — it only has to find a decent place to cut an overlap.
RE_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _tail_sentences(text: str, budget: int = OVERLAP_TOKENS) -> str:
    """The last few SENTENCES of a piece, up to `budget` tokens — the overlap carried forward.

    The overlap used to be whole PARAGRAPHS, and it almost never happened: a clinical paragraph
    is usually longer than the budget on its own, so the loop broke on the first candidate and
    carried nothing. Measured on the shipped corpus, only 12 of 53 split points had any overlap
    at all, and 0.5% of the corpus was duplicated. Sentences are small enough to actually fit."""
    sentences = [s for s in RE_SENTENCE_END.split(text) if s.strip()]
    carried, total = [], 0
    for sentence in reversed(sentences):
        cost = count_tokens(sentence)
        if total + cost > budget and carried:
            break
        carried.insert(0, sentence)
        total += cost
        if total >= budget:
            break
    return " ".join(carried).strip()


def split_table(block: str, budget: int = MAX_TOKENS) -> List[str]:
    """Split one Markdown table into pieces, REPEATING the header row in each.

    A table row without its column names is unreadable and uncitable — «| INI | BIC/FTC/TAF |»
    means nothing without «| 3er Fármaco | Pauta | Comentarios |» above it. The header travels
    with every piece, so each piece is a valid table on its own. A row is never split.

    Anything that is not a row (the caption above, the footnotes below) rides with the piece it
    is adjacent to, so a legend defining «X NR» stays next to rows that use it."""
    lines = block.splitlines()
    header = [l for l in lines[:3] if RE_TABLE_ROW.match(l)][:2]
    if not header or not any(RE_TABLE_RULE.match(l) for l in header):
        header = []                       # no recognisable header: fall back to a plain split
    header_tokens = count_tokens("\n".join(header))

    pieces, cur, cur_tokens = [], [], 0
    started = False                       # have we passed the header rows yet?
    for line in lines:
        if header and not started:
            if line in header:
                continue                  # emitted with every piece instead
            started = True
        cost = count_tokens(line)
        if cur and header_tokens + cur_tokens + cost > budget:
            pieces.append("\n".join(header + cur).strip())
            cur, cur_tokens = [], 0
        cur.append(line)
        cur_tokens += cost
    if cur:
        pieces.append("\n".join(header + cur).strip())
    return [p for p in pieces if p.strip()] or [block]


def _pack(units: List[str], budget: int, joiner: str) -> List[str]:
    """Greedily fill pieces with `units` without crossing `budget`. A single unit larger than
    the budget still goes out alone — by then it is one table row or one sentence, and cutting
    inside it would corrupt the text rather than merely make the chunk large."""
    pieces, cur, cur_tokens = [], [], 0
    for unit in units:
        cost = count_tokens(unit)
        if cur and cur_tokens + cost > budget:
            pieces.append(joiner.join(cur))
            cur, cur_tokens = [], 0
        cur.append(unit)
        cur_tokens += cost
    if cur:
        pieces.append(joiner.join(cur))
    return pieces


def _split_block(block: str, budget: int) -> List[str]:
    """Split ONE oversized block, using the largest unit that keeps it meaningful.

    Tables split by ROW (header repeated), a serialised matrix by RECORD, and prose by SENTENCE.
    Prose used to be returned whole on the argument that cutting mid-paragraph strands a
    recommendation from its condition — true, but it left 109 chunks over budget, up to 1 822
    tokens. That is worse on the same terms: the reranker only scores a 512-character window of
    a chunk, so an oversized one is judged on a fraction of itself and can lose the very
    recommendation the argument was protecting."""
    if has_table(block):
        return split_table(block, budget)
    lines = block.splitlines()
    if sum(1 for l in lines if RE_RECORD_ROW.match(l)) >= 2:
        return _pack(lines, budget, "\n")
    sentences = [s for s in RE_SENTENCE_END.split(block) if s.strip()]
    return _pack(sentences, budget, " ") if len(sentences) > 1 else [block]


def split_large(sec: Section) -> List[str]:
    """Split a large body into pieces no larger than MAX_TOKENS, with a sentence overlap.

    Paragraphs are the primary unit; a paragraph that is itself too large (a whole table, a
    matrix serialised as records) is handed to `_split_block`, which knows how to cut it without
    destroying its meaning. Nothing is emitted over budget any more: the old version passed an
    oversized paragraph straight through, which is how a single 7 506-token chunk reached an
    index whose model accepts 8 191."""
    budget = sec.budget
    pieces, cur, cur_tok = [], [], 0

    def flush() -> None:
        nonlocal cur, cur_tok
        if cur:
            pieces.append("\n\n".join(cur))
            cur, cur_tok = [], 0

    for para in re.split(r"\n\s*\n", sec.body):
        para = para.strip()
        if not para:
            continue
        ptok = count_tokens(para)
        if ptok > budget:
            flush()
            pieces.extend(_split_block(para, budget))
            continue
        if cur_tok + ptok > budget and cur:
            carried = _tail_sentences(cur[-1])
            flush()
            if carried and count_tokens(carried) + ptok <= budget:
                cur, cur_tok = [carried], count_tokens(carried)
        cur.append(para)
        cur_tok += ptok
    flush()
    return pieces or [sec.body]


def _piece_type(section_type: str, piece: str) -> str:
    """The content type of ONE piece of a split section.

    Splitting makes the section's own label a poor description of its parts. A table section
    cut by rows yields pieces that are all rows — but also one that is only the caption and the
    footnotes, which is prose. And a prose section can contain a table that ends up alone in its
    piece. Since `content_type` is a Qdrant filter, it has to describe the payload it is
    attached to, not the section it came from."""
    if section_type in ("table", "text"):
        return "table" if has_tabular_content(piece) else "text"
    return section_type


def normalize(sections: List[Section]) -> List[dict]:
    """
    Convert sections into chunk units (dicts with body+meta), applying
    merging of small sections and splitting of large ones.
    Returns dicts with: body, heading, level, breadcrumb, section_number,
    content_type.
    """
    units: List[dict] = []

    def emit(sec: Section, body: str, ctype: str,
             section_number: Optional[str] = None,
             heading: Optional[str] = None,
             breadcrumb: Optional[List[str]] = None):
        # The optional parameters allow overriding the section's identity when
        # several subsections are merged (see the merge block): the chunk is
        # tagged with the common ancestor, not with the 1st subsection.
        units.append({
            "body": body,
            "heading": sec.heading if heading is None else heading,
            "level": sec.level,
            "breadcrumb": sec.breadcrumb if breadcrumb is None else breadcrumb,
            "section_number": sec.section_number if section_number is None else section_number,
            "content_type": ctype,
        })

    i = 0
    n = len(sections)
    while i < n:
        sec = sections[i]
        ctype = classify(sec)

        # Container heading with no text of its own (e.g. "## 4." followed by
        # "### 4.1."): produces no chunk; its title lives in the breadcrumb of
        # the subsections.
        if ctype == "text" and not sec.body.strip():
            i += 1
            continue

        # Tables and abbreviation lists are never MERGED with a neighbour (they are self
        # contained), but they are no longer exempt from the size budget either. That exemption
        # is what produced a 7 506-token chunk holding the whole interaction matrix: too big for
        # the reranker's window to judge, and a quarter of a generation prompt on its own.
        # `split_large` cuts them by row and repeats the header.
        if ctype in ("table", "abbreviations"):
            for piece in split_large(sec) if sec.tokens > sec.budget else [sec.body]:
                emit(sec, piece, _piece_type(ctype, piece))
            i += 1
            continue

        # Large section -> split
        if sec.tokens > sec.budget:
            for piece in split_large(sec):
                emit(sec, piece, _piece_type(ctype, piece))
            i += 1
            continue

        # Small section -> try to merge with the following siblings under the same H2.
        # `recommendations` is included, and that is a change: keeping it out left the corpus's
        # single most valuable content — the graded recommendations — stranded in chunks of 21
        # to 63 tokens, with almost no signal for the retriever to match and no rationale around
        # them for the generator to condition on. A recommendation belongs with the reasoning
        # that precedes it. `evidence_grades` remains the reliable way to find them, and unlike
        # `content_type` it survives a merge.
        if sec.tokens < MIN_TOKENS and ctype in ("text", "recommendations"):
            merged_body = sec.body
            merged_heads = [sec.heading]
            merged_secnums = [sec.section_number]
            # Merging may absorb anything inside the seed's PARENT SUBTREE: its siblings, and
            # its own children. Both matter. This used to compare `breadcrumb[1]`, a fixed index
            # that assumed the path began with a document title it never contains — for two H3
            # sections under one H2, `breadcrumb[1]` is each section ITSELF, so same-level
            # siblings always looked like different branches and never merged. And a
            # «Recomendaciones» block is a CHILD of the section it belongs to, not a sibling, so
            # a rule about siblings alone would still have stranded it. Comparing the parent
            # path by prefix is depth-independent and is what "do not cross a section boundary"
            # actually means.
            parent = tuple(sec.breadcrumb[:-1])
            j = i + 1
            while j < n and count_tokens(merged_body) < TARGET_TOKENS:
                nxt = sections[j]
                nxt_type = classify(nxt)
                nxt_parent = tuple(nxt.breadcrumb[:-1])
                # Tables and abbreviation lists stay whole; an H2 boundary is a topic boundary.
                if (nxt_type in ("table", "abbreviations")
                        or nxt_parent[:len(parent)] != parent):
                    break
                if not nxt.body.strip():
                    j += 1
                    continue
                # The heading keeps its REAL depth. It was hardcoded to `##`, so a merged
                # subsection announced itself as a chapter and the Markdown inside the chunk
                # contradicted the hierarchy its own breadcrumb described.
                addition = f"\n\n{'#' * max(nxt.level, 2)} {nxt.heading}\n{nxt.body}"
                if count_tokens(merged_body + addition) > sec.budget:
                    break
                merged_body += addition
                merged_heads.append(nxt.heading)
                merged_secnums.append(nxt.section_number)
                j += 1
            if merged_body.strip():
                # If more than 1 subsection was actually merged, the chunk no
                # longer belongs to the 1st subsection but to their common
                # ancestor: we relabel it (e.g. 4.2.1 + 4.2.2 -> 4.2) so that
                # whoever searches by number finds the content at the right level.
                if len(merged_heads) > 1:
                    common = _common_secnum(merged_secnums)
                    if common and common != sec.section_number:
                        anc_head, anc_bc = _ancestor_for_secnum(sec.breadcrumb, common)
                        emit(sec, merged_body, ctype,
                             section_number=common,
                             heading=anc_head,      # None -> keeps sec's own
                             breadcrumb=anc_bc)      # None -> keeps sec's own
                    else:
                        emit(sec, merged_body, ctype)
                else:
                    emit(sec, merged_body, ctype)
            i = max(j, i + 1)
            continue

        # Normal-sized section
        if sec.body.strip():
            emit(sec, sec.body, ctype)
        i += 1

    return units


# ---------------------------------------------------------------------------
# 7. BUILDING CHUNKS WITH METADATA
# ---------------------------------------------------------------------------
def build_chunks(path: Path) -> List[Chunk]:
    md = path.read_text(encoding="utf-8")
    doc = corpus.document_for_markdown(path.name)
    sections = parse_sections(md)
    units = normalize(sections)

    chunks: List[Chunk] = []
    seen_text: set = set()          # to deduplicate chunks with identical text
    for idx, u in enumerate(units):
        breadcrumb = u["breadcrumb"]
        # The preamble (text before the first heading) has no path:
        # we give it the document title as context.
        if not breadcrumb:
            breadcrumb = [doc.title or "Preámbulo"]

        # THE BREADCRUMB IS NO LONGER PART OF THE CITABLE TEXT, and this is the change with the
        # widest reach in this module.
        #
        # It used to be prepended to `text`, for a good reason — the embedding sees the section
        # context — but `text` is also what `evidence.py` quotes back to the doctor and what the
        # graph indexes map through. Two costs followed. The breadcrumb ran 120 to 747 characters,
        # so 178 of 517 chunks were byte-identical for longer than the 120-character window
        # `retrieval/_common` uses to resolve a chunk back to its payload: siblings became
        # indistinguishable, and one whole guideline's 45 chunks collided with each other. And
        # one chunk was 100% breadcrumb with no body at all.
        #
        # Splitting the two fields keeps the retrieval benefit and drops the collisions:
        # `text_for_retrieval` is what gets embedded and BM25-indexed, `text` is what gets cited.
        # `contextualize.py` prepends its sentence to the former, and the uploader already
        # prefers it, so this costs nothing downstream.
        text = u["body"].strip()
        text_for_retrieval = f"{breadcrumb_prefix(breadcrumb)}{text}".strip()

        # Dedup: if two sections produce exactly the same text, we keep only
        # the first (e.g. repeated subheadings in appendix forms).
        if text in seen_text:
            continue
        seen_text.add(text)

        # Content-addressed, so editing one section does not renumber every chunk after it.
        # The id used to be sha1(file, POSITION, heading), and `stable_id` in the uploaders
        # derives each Qdrant point's UUID from it: inserting a paragraph high up therefore
        # changed the identity of everything below, and an upsert without --recreate left the
        # previous ids in the collection as orphans nobody would ever overwrite.
        cid = hashlib.sha1(
            "\x00".join([doc.doc_id, u["section_number"] or "", u["heading"], text])
            .encode("utf-8")
        ).hexdigest()[:16]

        chunks.append(Chunk(
            chunk_id=cid,
            source_file=path.name,
            doc_id=doc.doc_id,
            doc_title=doc.title,
            specialty=doc.specialty,
            topics=list(doc.topics),
            organization=doc.organization,
            year=doc.year,
            section_path=breadcrumb,
            section_number=u["section_number"],
            heading=u["heading"],
            heading_level=u["level"],
            content_type=u["content_type"],
            evidence_grades=extract_grades(u["body"]),
            chunk_index=idx,
            # Sized on what is actually EMBEDDED, not on what is cited: the budget exists
            # because of the embedder's input limit and the reranker's scoring window.
            n_tokens=count_tokens(text_for_retrieval),
            n_chars=len(text),
            text=text,
            text_for_retrieval=text_for_retrieval,
        ))
    return chunks


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------
def gather_md_files(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.glob("*.md")))
        elif p.is_file() and p.suffix.lower() == ".md":
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description="Split clinical guidelines (Markdown) into chunks with metadata.")
    ap.add_argument("inputs", nargs="+", help=".md files or a folder with .md files")
    ap.add_argument("-o", "--output", default="data/chunks/chunks.jsonl", help="Output JSONL file")
    ap.add_argument("--stats", action="store_true", help="Print statistics")
    ap.add_argument("--check", action="store_true",
                    help="Run the quality gates and exit non-zero if any fails. Writes nothing: "
                         "a corpus that does not pass must not reach an index.")
    args = ap.parse_args()

    files = gather_md_files(args.inputs)
    if not files:
        print("No .md files found", file=sys.stderr)
        sys.exit(1)

    all_chunks: List[Chunk] = []
    for f in files:
        cs = build_chunks(f)
        all_chunks.extend(cs)
        print(f"  {f.name:38s} -> {len(cs):4d} chunks", file=sys.stderr)

    if args.check:
        # Imported here so the chunker does not depend on its auditor: quality.py imports THIS
        # module for the size budget and the grade regex, and a module-level import would close
        # the cycle.
        from ingestion.quality import audit, format_report, load_sources
        gates = audit([asdict(c) for c in all_chunks], load_sources(files))
        print("\n--- QUALITY GATES ---", file=sys.stderr)
        print(format_report(gates), file=sys.stderr)
        sys.exit(0 if all(g.passed for g in gates) else 1)

    with open(args.output, "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"\n{len(all_chunks)} chunks written to {args.output}", file=sys.stderr)

    if args.stats:
        import statistics as st
        toks = [c.n_tokens for c in all_chunks]
        from collections import Counter
        ctypes = Counter(c.content_type for c in all_chunks)
        print("\n--- STATISTICS ---", file=sys.stderr)
        print(f"tokens/chunk: min={min(toks)} med={int(st.median(toks))} "
              f"mean={int(st.mean(toks))} max={max(toks)}", file=sys.stderr)
        print(f"content types: {dict(ctypes)}", file=sys.stderr)
        graded = sum(1 for c in all_chunks if c.evidence_grades)
        print(f"chunks with evidence grade: {graded}", file=sys.stderr)


if __name__ == "__main__":
    main()
