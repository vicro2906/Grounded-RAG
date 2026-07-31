#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunk_guidelines.py
===================
Splits HIV clinical guidelines (GeSIDA/SPNS) in Markdown into structured *chunks*
with metadata, ready to index in a RAG system.

Section (raw text per heading, irregular size) → normalize adjusts the size and
produces units (well-sized text + inherited structural metadata) →
build_chunks adds the metadata that depends on the final text and prepends the breadcrumb
to the body → Chunk (the final object serialized to JSONL and uploaded to Qdrant).

Strategy: STRUCTURE-AWARE (header-aware) chunking with size normalization.
  1. Parse each .md into a tree of sections by the ##, ###, ####, ##### headings,
     keeping the full hierarchical path (breadcrumb).
  2. Each "leaf" section (text up to the next heading) is the base unit.
  3. Normalize size:
        - large sections  -> split by paragraphs with overlap.
        - small sections  -> merged with siblings under the same H2 parent.
        - tables / recommendations / abbreviations -> kept intact.
  4. Tag each chunk with metadata (topic, path, content type, evidence grades,
     section numbers, etc.).
  5. Export to JSONL (one chunk per line) -> standard RAG ingestion format.

No mandatory dependencies (standard library only). If 'tiktoken' is installed it is
used to count tokens; otherwise a character-based estimate is used.

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

# ---------------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------------

# Target sizes expressed in TOKENS (approx). Tunable to your embedding model.
# ~512-800 tokens is a good range for clinical retrieval.
TARGET_TOKENS = 600     # ideal size of a chunk
MAX_TOKENS    = 900     # above this a section is split
MIN_TOKENS    = 200     # below this we try to merge
OVERLAP_TOKENS = 80     # overlap when splitting large sections

# Document registry: file-level metadata. It is more reliable to set it here than
# to try to parse it from the text. Edit it when you add docs.
DOC_REGISTRY = {
    "TAR_2022.md": {
        "doc_title": "Documento de consenso de GeSIDA/PNS sobre TAR en adultos con VIH",
        "topic": "tratamiento_antirretroviral",
        "organization": "GeSIDA/PNS",
        "year": 2022,
    },
    "VIH_TB.md": {
        "doc_title": "Tratamiento de la tuberculosis en personas con infección por VIH",
        "topic": "vih_tuberculosis",
        "organization": "GeSIDA/PNS",
        "year": 2018,
    },
    "VIH_embarazo.md": {
        "doc_title": "Infección por VIH y reproducción, embarazo, parto y profilaxis de la transmisión vertical",
        "topic": "vih_embarazo",
        "organization": "SPNS/GeSIDA/SEGO/SEIP",
        "year": 2018,
    },
    "adherencia.md": {
        "doc_title": "Recomendaciones GeSIDA/SEFH/PNS para mejorar la adherencia al tratamiento",
        "topic": "adherencia",
        "organization": "PNS/GeSIDA/SEFH",
        "year": 2020,
    },
    "medicina_preventiva.md": {
        "doc_title": "Vacunación e inmunización en personas con VIH (medicina preventiva)",
        "topic": "medicina_preventiva_vacunas",
        "organization": "GeSIDA/SEMPSPGS",
        "year": 2024,
    },
    "profilaxis.md": {
        "doc_title": "Profilaxis posexposición ocupacional y no ocupacional al VIH, VHB y VHC",
        "topic": "profilaxis_postexposicion",
        "organization": "GeSIDA/GEHEP/SEIP/SEMPSPGS",
        "year": 2025,
    },
    "ManejoclinicodelasalteracionesNC.md": {
        "doc_title": "Manejo clínico de las alteraciones neurocognitivas asociadas al VIH",
        "topic": "alteraciones_neurocognitivas",
        "organization": "GeSIDA/PNS",
        "year": 2013,
    },
}

DEFAULT_META = {
    "doc_title": None, "topic": "vih_general",
    "organization": "GeSIDA", "year": None,
}


def _norm_name(name: str) -> str:
    """Normalize a filename to look it up in the registry: lowercase and
    spaces/hyphens -> underscore. So 'medicina preventiva.md' and
    'medicina_preventiva.md' point to the same entry."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


# Registry indexed by normalized name (built once).
_NORMALIZED_REGISTRY = {_norm_name(k): v for k, v in DOC_REGISTRY.items()}


def lookup_meta(filename: str) -> dict:
    meta = _NORMALIZED_REGISTRY.get(_norm_name(filename))
    if meta is None:
        return {**DEFAULT_META, "doc_title": Path(filename).stem}
    return meta

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

# Headings that mark a special content type (match Spanish guideline text).
RE_RECS    = re.compile(r"recomendaci(o|ó)n", re.IGNORECASE)
RE_ABREV   = re.compile(r"abreviatura|listado de abreviaturas", re.IGNORECASE)
RE_TABLA_H = re.compile(r"^(tabla|figura)\b", re.IGNORECASE)
RE_ANEXO   = re.compile(r"\banexo|algoritmo", re.IGNORECASE)
RE_METODO  = re.compile(r"metodolog(i|í)a", re.IGNORECASE)


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


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    doc_title: Optional[str]
    topic: str
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
    text: str                      # chunk text, with the breadcrumb prepended


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
    h = sec.heading
    if RE_ABREV.search(h):
        return "abbreviations"
    if RE_RECS.search(h) and sec.level >= 4:
        return "recommendations"
    if RE_TABLA_H.match(h):
        return "table"
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


# ---------------------------------------------------------------------------
# 6. SIZE NORMALIZATION (merge + split)
# ---------------------------------------------------------------------------
def split_large(sec: Section) -> List[str]:
    """Split a large body by paragraphs with overlap, without breaking tables."""
    paras = re.split(r"\n\s*\n", sec.body)
    pieces, cur, cur_tok = [], [], 0
    for p in paras:
        p = p.strip()
        if not p:
            continue
        ptok = count_tokens(p)
        # a table or huge paragraph goes in its own piece
        if ptok > MAX_TOKENS:
            if cur:
                pieces.append("\n\n".join(cur)); cur, cur_tok = [], 0
            pieces.append(p)
            continue
        if cur_tok + ptok > MAX_TOKENS and cur:
            pieces.append("\n\n".join(cur))
            # overlap: carry the last paragraphs up to OVERLAP_TOKENS
            overlap, otok = [], 0
            for q in reversed(cur):
                qt = count_tokens(q)
                if otok + qt > OVERLAP_TOKENS:
                    break
                overlap.insert(0, q); otok += qt
            cur, cur_tok = list(overlap), otok
        cur.append(p); cur_tok += ptok
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces or [sec.body]


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

        # Types that are NOT merged nor split arbitrarily
        if ctype in ("table", "abbreviations"):
            emit(sec, sec.body, ctype)
            i += 1
            continue

        # Large section -> split
        if sec.tokens > MAX_TOKENS:
            for piece in split_large(sec):
                # reclassify in case the piece is a table
                sub = Section(sec.level, sec.heading, piece, sec.breadcrumb, sec.section_number)
                emit(sec, piece, "table" if (ctype == "text" and has_table(piece)) else ctype)
            i += 1
            continue

        # Small section -> try to merge with the following siblings under the same H2
        if sec.tokens < MIN_TOKENS and ctype == "text":
            merged_body = sec.body
            merged_heads = [sec.heading]
            merged_secnums = [sec.section_number]
            parent_h2 = sec.breadcrumb[1] if len(sec.breadcrumb) > 1 else (sec.breadcrumb[0] if sec.breadcrumb else "")
            j = i + 1
            while j < n and count_tokens(merged_body) < TARGET_TOKENS:
                nxt = sections[j]
                nxt_type = classify(nxt)
                nxt_parent = nxt.breadcrumb[1] if len(nxt.breadcrumb) > 1 else (nxt.breadcrumb[0] if nxt.breadcrumb else "")
                # do not merge tables/abbreviations/recos nor cross an H2 section
                if nxt_type in ("table", "abbreviations") or nxt_parent != parent_h2:
                    break
                if not nxt.body.strip():
                    j += 1
                    continue
                addition = f"\n\n## {nxt.heading}\n{nxt.body}" if nxt.body else ""
                if count_tokens(merged_body + addition) > MAX_TOKENS:
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
    meta = lookup_meta(path.name)
    sections = parse_sections(md)
    units = normalize(sections)

    chunks: List[Chunk] = []
    seen_text: set = set()          # to deduplicate chunks with identical text
    for idx, u in enumerate(units):
        breadcrumb = u["breadcrumb"]
        # The preamble (text before the first heading) has no path:
        # we give it the document title as context.
        if not breadcrumb:
            breadcrumb = [meta.get("doc_title") or "Preámbulo"]
        # Prepending the breadcrumb to the text greatly improves retrieval:
        # the embedding "sees" the section's hierarchical context.
        context_prefix = " > ".join(breadcrumb)
        text = f"{context_prefix}\n\n{u['body']}".strip() if context_prefix else u["body"]

        # Dedup: if two sections produce exactly the same text, we keep only
        # the first (e.g. repeated subheadings in appendix forms).
        norm = text.strip()
        if norm in seen_text:
            continue
        seen_text.add(norm)

        cid = hashlib.sha1(
            f"{path.name}:{idx}:{u['heading']}".encode("utf-8")
        ).hexdigest()[:16]

        chunks.append(Chunk(
            chunk_id=cid,
            source_file=path.name,
            doc_title=meta.get("doc_title"),
            topic=meta.get("topic", "vih_general"),
            organization=meta.get("organization", "GeSIDA"),
            year=meta.get("year"),
            section_path=breadcrumb,
            section_number=u["section_number"],
            heading=u["heading"],
            heading_level=u["level"],
            content_type=u["content_type"],
            evidence_grades=extract_grades(u["body"]),
            chunk_index=idx,
            n_tokens=count_tokens(text),
            n_chars=len(text),
            text=text,
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
