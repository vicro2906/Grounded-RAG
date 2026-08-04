#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PDF -> Markdown, faithfully and reproducibly.

WHY THIS EXISTS. The corpus was converted by a script written ad hoc for each PDF and then
thrown away, so the conversion could not be reviewed, repeated or fixed — and it did real
damage: the chapter holding the recommended first-line regimens arrived with ZERO table rows,
a drug lost a letter (`fuconazol` for `fluconazol`) inside dosing text, and 114 blocks were
dropped behind free-form "omitted" markers. None of it announced itself downstream.

WHAT MAKES IT SAFE. No LLM is ever constructed here, and that is the load-bearing property of
the whole system: everything the doctor is shown as a literal quote is a transcription that can
be checked against the PDF, never a model's paraphrase of one. Figures and decision algorithms
that are genuinely graphical stay OMITTED, with a marker naming the cause and the page, because
an omission the doctor can act on beats a reconstruction they cannot verify.

WHAT THE MEASUREMENTS CHANGED. Three components the design expected turned out to be
unnecessary, and each was dropped rather than written "just in case":
  * Ligature repair. pdfplumber decodes `ﬁ` correctly; the lost letters were the previous
    converter's doing. Only a guard remains, to expand any glyph that does arrive raw.
  * Evidence-grade re-anchoring. The old Markdown had 181 grades injected mid-sentence in one
    guide; pdfplumber's reading order puts them at the end of their sentence, where they belong
    (4 doubtful in 60 pages, against 181). The geometric rule the plan described was never
    needed, so it does not exist.
  * A VLM for the tables. Their text is in the PDF's text layer; they are VECTOR grids that the
    previous converter masked as pictures. `lines` recovers TABLA 3 (11x3) and the 19-column
    interaction matrix (17x20) intact.

Usage:
    python -m ingestion.extract_pdf TAR_2022            # one document from the manifest
    python -m ingestion.extract_pdf --all --report      # every document, with a quality report
    python -m ingestion.extract_pdf TAR_2022 --pages 100-110   # a slice, for inspection
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

import corpus

# pdfminer narrates every malformed object it meets; the diagnosis we care about is the triage
# step's, which is quantified and reported, not a wall of warnings.
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# pdfminer emits `(cid:NNN)` for a glyph whose font carries no ToUnicode mapping. It is the
# precise signature of a broken CMap, and it is the ONLY reliable one: the eighth PDF in
# data/pdfs/ is 81% alphabetic characters — a ratio check waves it through — while rendering
# `eficiencia` as `e(cid:332)ciencia`.
RE_CID = re.compile(r"\(cid:\d+\)")
# Above this share of characters, the text layer is not trustworthy enough to transcribe.
MAX_CID_RATE = 0.001
# Below this many characters per page on average, the document is a scan, not a native PDF.
MIN_CHARS_PER_PAGE = 200

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}

# A word this much smaller than the body font is a superscript. They are bibliographic citation
# numbers, and pdfplumber lifts them onto their own line: left inline they would read as
# «...conductas suicidas 36,37, aunque...», where a model can take `36,37` for a clinical figure.
SUPERSCRIPT_RATIO = 0.8
# A line this much larger than the body font is a heading.
HEADING_RATIO = 1.12

# Section numbering drives the heading hierarchy: `1.` -> H2, `1.2.` -> H3, `1.2.3.` -> H4.
# This is the contract ingestion.chunk_guidelines parses back out.
RE_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")

# Spanish, and staying Spanish: these match the guideline text itself, which the language policy
# keeps in Spanish. Unnumbered headings whose structural level is a convention of the genre.
UNNUMBERED_LEVELS = [
    (re.compile(r"^recomendaci(o|ó)n(es)?\s*:?\s*$", re.I), 4),
    (re.compile(r"^(tabla|figura)\s*\d+", re.I), 4),
    (re.compile(r"^bibliograf(i|í)a", re.I), 3),
    (re.compile(r"^(anexo|ap(e|é)ndice|algoritmo)", re.I), 2),
    (re.compile(r"^(listado de )?abreviaturas", re.I), 2),
    (re.compile(r"^metodolog(i|í)a", re.I), 2),
]

# Sections dropped wholesale: not clinical content, and they pollute retrieval.
RE_DROP_SECTION = re.compile(
    r"^(bibliograf(i|í)a|referencias|comit(e|é) de redacci(o|ó)n|agradecimientos|"
    r"conflicto[s]? de intere|autores|coordinador)", re.I)

RE_PAGE_NUMBER = re.compile(r"^\s*[-—]?\s*\d{1,4}\s*[-—]?\s*$")
# Dot leaders: a table-of-contents entry, never a heading. Without this the contents pages
# contribute a duplicate of EVERY heading in the document, and the outline doubles.
RE_TOC_ENTRY = re.compile(r"\.{4,}\s*\d{1,4}\s*$")
# A superscript citation that survived as its own line: only digits, commas and dashes.
RE_CITATION_LINE = re.compile(r"^[\d\s,;.\-–—]+$")

# The ONE omission marker. Spanish because the doctor reads it in the corpus; the page number is
# what makes it actionable, the cause is what lets the quality gate accept an unrecoverable
# figure while rejecting a lost table.
OMISSION = "> _[{what} omitid{o} — p. {page} — consultar PDF original]_"

# Wider than this is not rendered as a Markdown grid: a 19-column table is unreadable to the
# model, cannot be split without losing its header, and cites badly. It becomes one record per
# entity-row instead — which is also the unit a doctor asks about («¿interacción de X con Y?»).
MAX_MARKDOWN_COLUMNS = 6
# A reconstructed table must preserve at least this share of the numeric tokens in its region.
# Numbers are doses, thresholds and CD4 counts; losing them silently is the whole failure the
# old `9. TABLAS` chapter represented.
MIN_NUMERIC_KEPT = 0.95
# ... and at least this share of the region's characters, which is what catches a whole column
# or row going missing.
MIN_CONTENT_KEPT = 0.90

RE_TOKEN = re.compile(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
RE_NUMERIC = re.compile(r"\d+(?:[.,]\d+)*")
RE_NON_ALNUM = re.compile(r"[^0-9a-záéíóúüñ]+")


def _alnum(text: str) -> str:
    """Lowercased alphanumerics only — the form in which cell text and page text can be compared
    without caring where a line break fell."""
    return RE_NON_ALNUM.sub("", (text or "").lower())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class Report:
    """What the extraction did, in numbers a human can review before anything is indexed."""
    doc_id: str
    pages: int = 0
    status: str = ""
    chars: int = 0
    headings: int = 0
    tables_kept: int = 0
    tables_omitted: int = 0
    figures_omitted: int = 0
    rows_filled: int = 0            # rowspan labels propagated (see fill_group_cells)
    superscripts_dropped: int = 0
    running_heads_dropped: int = 0
    sections_dropped: int = 0
    notes: list = field(default_factory=list)

    def render(self) -> str:
        lines = [f"  {self.doc_id}  ({self.status}, {self.pages} pages, {self.chars:,} chars)",
                 f"    headings {self.headings} · tables kept {self.tables_kept} · "
                 f"tables omitted {self.tables_omitted} · figures omitted {self.figures_omitted}",
                 f"    group cells filled {self.rows_filled} · superscripts dropped "
                 f"{self.superscripts_dropped} · running heads {self.running_heads_dropped} · "
                 f"sections dropped {self.sections_dropped}"]
        lines += [f"    ! {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 0 — Triage
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Diagnosis:
    status: str          # native_ok | broken_cmap | scanned
    detail: str

    @property
    def usable(self) -> bool:
        return self.status == "native_ok"


def triage(pdf) -> Diagnosis:
    """Decide whether this PDF's text layer can be transcribed at all.

    Refusing is a RESULT, not a failure: a document that enters the corpus as mojibake answers
    questions with noise, and nothing downstream can tell that apart from a bad retrieval."""
    sample = pdf.pages[:: max(1, len(pdf.pages) // 12)][:12]
    text = "\n".join((page.extract_text() or "") for page in sample)
    if len(text) < MIN_CHARS_PER_PAGE * len(sample) / 4:
        return Diagnosis("scanned", f"only {len(text)} characters across {len(sample)} sampled "
                                    "pages; this looks like a scan, and OCR is out of scope")
    cids = RE_CID.findall(text)
    rate = len(cids) / max(len(text), 1)
    if rate > MAX_CID_RATE:
        common = Counter(cids).most_common(5)
        return Diagnosis("broken_cmap",
                         f"{len(cids)} unmapped glyphs ({rate:.2%} of the sampled text): the "
                         f"font carries no ToUnicode table, so characters decode to nothing. "
                         f"Most frequent: {', '.join(f'{c} x{n}' for c, n in common)}")
    return Diagnosis("native_ok", f"{len(text):,} characters sampled, no unmapped glyphs")


# ---------------------------------------------------------------------------
# Step 1 — Page geometry
# ---------------------------------------------------------------------------
def body_font_size(pdf, sample_pages: int = 12) -> float:
    """The document's running-text size: the most common character size. Everything else is
    measured against it, so headings and superscripts are found by CONTRAST rather than by an
    absolute threshold that would differ per guideline."""
    sizes: Counter = Counter()
    step = max(1, len(pdf.pages) // sample_pages)
    for page in pdf.pages[::step][:sample_pages]:
        for char in page.chars:
            sizes[round(char["size"], 1)] += 1
    return sizes.most_common(1)[0][0] if sizes else 10.0


def head_key(text: str) -> str:
    """A running head with its page number stripped off.

    The number changes on every page, so the raw lines never repeat and the detector misses
    them entirely — and «28 Documento de consenso para el seguimiento…» then reads as a NUMBERED
    HEADING, which put running heads into the document outline."""
    return re.sub(r"^\s*\d{1,4}\s+|\s+\d{1,4}\s*$", "", text).strip()


def find_running_heads(pdf, sample_pages: int = 20) -> set:
    """Lines repeated across most pages: the running header/footer. Detected rather than
    configured, so a new document needs no per-file rule."""
    step = max(1, len(pdf.pages) // sample_pages)
    pages = pdf.pages[::step][:sample_pages]
    counts: Counter = Counter()
    for page in pages:
        text = page.extract_text() or ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in set(lines[:2] + lines[-2:]):     # only the top and bottom of the page
            key = head_key(line)
            if len(key) > 12:
                counts[key] += 1
    threshold = max(2, len(pages) // 2)
    return {key for key, n in counts.items() if n >= threshold}


# ---------------------------------------------------------------------------
# Step 2 — Tables
# ---------------------------------------------------------------------------
# `lines` reads the ruled grid the typesetter actually drew, and it is the DEFAULT because these
# guidelines rule their tables: it recovers TABLA 3 (11x3) and the 19-column interaction matrix
# (17x20) intact.
#
# `text` — inferring columns from whitespace alignment — is NOT a fallback, and this is the
# single most damaging thing measured while building this module. On a page of ordinary prose it
# finds a table anyway, chopping paragraphs at arbitrary character positions
# («| un metaanálisis), | II (de uno o más ensay | os no aleatorizados o datos obs |»). Enabled
# as a fallback it turned 55% of the output into fake table rows AND swallowed the headings
# inside those regions, so the outline collapsed from ~165 headings to 24. A document that
# genuinely needs it declares `table_strategy = "text"` in the manifest — a decision made once,
# per PDF, by someone who looked.
TABLE_STRATEGIES = {
    "lines": {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
    "text": {"vertical_strategy": "text", "horizontal_strategy": "text"},
    "mixed": {"vertical_strategy": "lines", "horizontal_strategy": "text"},
}


def clean_cell(value: str | None) -> str:
    text = expand_ligatures(value or "")
    return re.sub(r"\s+", " ", text).strip()


def fill_group_cells(rows: list[list[str]], columns: list[int]) -> tuple[list[list[str]], int]:
    """Propagate a row-group label down the rows it spans.

    A cell spanning several rows leaves the others empty, which the previous converter read as
    "layout damaged" and threw the whole table away («primera columna vacía en 71% de las filas»).

    THE RULES ARE NARROW ON PURPOSE, because this is the one place here that can invent a
    clinical fact: a label propagated onto a row it does not own produces a false recommendation
    that the system would then quote LITERALLY, which is worse than the omission it replaces.
    So: only declared columns, only into a genuinely empty cell, only when the row has content
    of its own to be labelled, and never past a value that looks like a dose or an evidence
    grade."""
    if not columns:
        return rows, 0
    filled = 0
    carry: dict[int, str] = {}
    out = []
    for row in rows:
        new = list(row)
        has_content = any(cell for i, cell in enumerate(row) if i not in columns)
        for col in columns:
            if col >= len(new):
                continue
            value = new[col].strip()
            if value:
                # A number or a grade is a VALUE, not a group label: never carry it downwards.
                carry[col] = "" if RE_NUMERIC.search(value) or RE_GRADE_CELL.search(value) else value
            elif has_content and carry.get(col):
                new[col] = carry[col]
                filled += 1
        out.append(new)
    return out, filled


RE_GRADE_CELL = re.compile(r"\(\s*[ABC]\s*-?\s*I{1,3}\s*\)")


def table_is_faithful(rows: list[list[str]], region_text: str) -> str:
    """Empty string if the reconstruction can be trusted, otherwise the reason it cannot.

    WHAT THIS CHECKS, AND WHAT IT DOES NOT. It checks CONSERVATION — that the reconstruction did
    not silently lose the region's content. That is the failure that actually happened: thirteen
    table captions shipped with zero rows beneath them, taking the recommended first-line
    regimens and the whole interaction matrix with them.

    It does NOT check "nothing was invented", because nothing here can invent: pdfplumber
    transcribes glyphs off the page, and the one component that synthesises anything —
    `fill_group_cells` — carries its own narrow rules. An earlier draft did try, comparing token
    multisets, and it was worse than useless: a cell wrapping across lines interleaves with its
    neighbouring columns in the region's reading order, so perfectly good tables came back as
    "invented text" (7 of 10 discarded). A gate that fails on correct input teaches you to
    ignore it.

    Two measures, both robust to where the typesetter broke a line:
      * CHARACTER coverage — the region's alphanumerics must be accounted for. Forward-fill can
        only ADD, so a shortfall means a column or a row was dropped.
      * NUMERIC coverage — doses, thresholds and CD4 counts specifically, because they are the
        cells whose loss is most dangerous and least visible."""
    if len(rows) < 2:
        return "fewer than two rows"
    widths = {len(r) for r in rows}
    if len(widths) > 1:
        return f"ragged: rows have {sorted(widths)} columns"
    if not any(cell for cell in rows[0]):
        return "empty header row"

    cells = " ".join(cell for row in rows for cell in row)
    region_chars = len(_alnum(region_text))
    if region_chars and len(_alnum(cells)) / region_chars < MIN_CONTENT_KEPT:
        return (f"only {len(_alnum(cells))}/{region_chars} characters of the region survived "
                f"the reconstruction")

    region_numbers = Counter(RE_NUMERIC.findall(region_text))
    kept = Counter(RE_NUMERIC.findall(cells))
    total = sum(region_numbers.values())
    if total:
        survived = sum(min(n, kept[tok]) for tok, n in region_numbers.items())
        if survived / total < MIN_NUMERIC_KEPT:
            return f"only {survived}/{total} numeric tokens survived"
    return ""


def render_table(rows: list[list[str]], caption: str) -> str:
    """Markdown grid for a narrow table, one record per row for a wide one.

    A 19-column matrix as Markdown is unreadable to the model, impossible to split without
    losing its header, and it cites badly. As records it is the unit the question actually has
    («¿interacción de abemaciclib con bictegravir?»), it splits by row without losing anything,
    and each line is literal text `evidence.attribute` can verify. Interaction matrices are
    sparse, so only the non-empty cells are written."""
    header, body = rows[0], rows[1:]
    if len(header) <= MAX_MARKDOWN_COLUMNS:
        out = ["| " + " | ".join(header) + " |",
               "|" + "|".join("---" for _ in header) + "|"]
        out += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(out)

    lines = []
    for row in body:
        label = row[0].strip()
        pairs = [f"{header[i].strip()} «{cell.strip()}»"
                 for i, cell in enumerate(row[1:], start=1)
                 if cell.strip() and i < len(header)]
        if label and pairs:
            lines.append(f"- {label}: " + "; ".join(pairs) + ".")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3 — Text normalization
# ---------------------------------------------------------------------------
def expand_ligatures(text: str) -> str:
    """Guard, not a repair. pdfplumber decodes these correctly; a raw glyph reaching the corpus
    would be a word that no lexical search can match."""
    for glyph, letters in LIGATURES.items():
        text = text.replace(glyph, letters)
    return text


def heading_level(line: str) -> int | None:
    """The heading level of a line, or None if it is body text.

    Numbering decides it — `1.` -> H2, `1.2.` -> H3 — because that is the hierarchy the document
    itself declares, and it is the contract the chunker parses back out."""
    match = RE_NUMBERED.match(line)
    if match:
        return min(6, 1 + len(match.group(1).split(".")))
    for pattern, level in UNNUMBERED_LEVELS:
        if pattern.match(line):
            return level
    return None


def tidy(text: str) -> str:
    """Whitespace the extraction leaves behind: a space before punctuation (where a superscript
    was removed), runs of blank lines, trailing spaces."""
    text = re.sub(r" +([,.;:)])", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Step 4 — One page
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """A piece of the document, positioned so text and tables can be re-interleaved in order."""
    top: float
    kind: str            # heading | text | table | omission
    text: str
    level: int = 0


# A figure smaller than this share of the page is a logo or a rule, not content worth flagging.
MIN_FIGURE_AREA = 0.04


def _clamp(bbox, page) -> tuple:
    """Keep a detected region inside the page. pdfplumber raises on a crop that escapes it."""
    x0, top, x1, bottom = bbox
    return (max(0.0, float(x0)), max(0.0, float(top)),
            min(float(page.width), float(x1)), min(float(page.height), float(bottom)))


def text_lines(page) -> list[dict]:
    """The page's lines, each with the characters that formed it.

    pdfplumber's OWN line assembly, not a regrouping of `extract_words`. The difference is not
    cosmetic: word segmentation splits at intra-word kerning gaps, so `Neuropsichol` arrives as
    `N` + `europsichol` and `definition` as `defi` + `nition` — 231 broken tokens in one guide,
    which is precisely the damage this extractor exists to stop. It also means the Markdown and
    the reference text are joined the SAME way, so the oracle in `quality.g05` compares like
    with like instead of reporting the segmenter's habits as extraction damage."""
    return page.extract_text_lines()


def extract_page(page, body_size: float, running_heads: set, profile: dict,
                 report: Report) -> list:
    """Everything on one page, as ordered blocks: tables where they sit, text around them."""
    blocks: list = []

    # --- tables first: their regions are excluded from the running text -----
    regions: list = []
    settings = TABLE_STRATEGIES[profile.get("table_strategy", "lines")]
    for table in page.find_tables(table_settings=settings):
        rows = [[clean_cell(c) for c in row] for row in (table.extract() or [])]
        rows = [r for r in rows if any(r)]
        # One row is not a table — it is a rule, a boxed note or a stray detection. Nothing was
        # omitted, so nothing is marked: an omission marker sends the doctor to the PDF, and
        # doing that for content that was never a table is a false alarm 64 times over.
        if len(rows) < 2:
            continue
        # A rule drawn at the page edge can yield a bbox wider than the page, which pdfplumber
        # refuses to crop to. Clamping keeps one stray detection from taking down a 146-page run.
        region_text = expand_ligatures(
            page.crop(_clamp(table.bbox, page)).extract_text() or "")
        rows, filled = fill_group_cells(rows, profile.get("group_columns", [0]))
        report.rows_filled += filled
        problem = table_is_faithful(rows, region_text)
        if problem:
            report.tables_omitted += 1
            report.notes.append(f"p.{page.page_number}: table omitted ({problem})")
            blocks.append(Block(table.bbox[1], "omission",
                                OMISSION.format(what="Tabla", o="a", page=page.page_number)))
        else:
            report.tables_kept += 1
            blocks.append(Block(table.bbox[1], "table", render_table(rows, "")))
        regions.append(table.bbox)

    # --- figures: flagged, never described ---------------------------------
    page_area = max(float(page.width) * float(page.height), 1.0)
    for image in page.images:
        area = abs(image["x1"] - image["x0"]) * abs(image["bottom"] - image["top"])
        if area / page_area >= MIN_FIGURE_AREA:
            report.figures_omitted += 1
            blocks.append(Block(float(image["top"]), "omission",
                                OMISSION.format(what="Figura", o="a", page=page.page_number)))

    # --- running text, outside the table regions ---------------------------
    def outside(line) -> bool:
        return not any(x0 <= line["x0"] <= x1 and top <= line["top"] <= bottom
                       for x0, top, x1, bottom in regions)

    for line in text_lines(page):
        if not outside(line):
            continue
        chars = line["chars"]
        mean_size = sum(c["size"] for c in chars) / len(chars)
        if mean_size < body_size * SUPERSCRIPT_RATIO:
            report.superscripts_dropped += 1     # a bibliographic citation number
            continue
        text = expand_ligatures(line["text"]).strip()
        if (not text or RE_PAGE_NUMBER.match(text) or RE_CITATION_LINE.match(text)
                or RE_TOC_ENTRY.search(text)):
            continue
        if head_key(text) in running_heads:
            report.running_heads_dropped += 1
            continue
        top = line["top"]
        bold = any("bold" in c["fontname"].lower() for c in chars)
        styled = mean_size > body_size * HEADING_RATIO or bold
        level = heading_level(text)
        if level and styled:
            blocks.append(Block(top, "heading", text, level))
        elif styled:
            # Heading-styled but carrying no numbering of its own: almost always the SECOND
            # line of a heading the typesetter wrapped («2.4.3. MANEJO DE» / «INTERRUPCIONES
            # DEL TRATAMIENTO»). Marked here and merged in `assemble`, which can see what came
            # before it; left alone it becomes body text and the heading is truncated.
            blocks.append(Block(top, "heading_cont", text))
        else:
            blocks.append(Block(top, "text", text))

    return sorted(blocks, key=lambda b: b.top)


# ---------------------------------------------------------------------------
# Step 5 — The document
# ---------------------------------------------------------------------------
def assemble(blocks: list, report: Report) -> str:
    """Ordered blocks -> Markdown. Consecutive text lines rejoin into paragraphs; a dropped
    section swallows everything until the next heading of its level or above."""
    out: list = []
    paragraph: list = []
    dropping_at = None
    open_heading = False       # the last thing appended to `out` was a heading

    def flush():
        nonlocal open_heading
        if paragraph:
            out.append(" ".join(paragraph))
            paragraph.clear()
            open_heading = False

    for block in blocks:
        if block.kind == "heading_cont":
            # Heading-styled but with no numbering of its own. If a heading was just emitted it
            # is that heading's wrapped second line («2.4.3. MANEJO DE» / «INTERRUPCIONES DEL
            # TRATAMIENTO»), and appending it is what keeps the heading whole; otherwise it is a
            # bold line inside running text and belongs to the paragraph.
            if open_heading:
                out[-1] += " " + block.text
            elif dropping_at is None:
                paragraph.append(block.text)
            continue

        if block.kind == "heading":
            flush()
            if dropping_at is not None and block.level > dropping_at:
                continue                       # still inside a dropped section
            dropping_at = None
            if RE_DROP_SECTION.match(block.text):
                dropping_at = block.level
                open_heading = False
                report.sections_dropped += 1
                continue
            report.headings += 1
            out.append("#" * block.level + " " + block.text)
            open_heading = True
            continue

        open_heading = False
        if dropping_at is not None:
            continue
        if block.kind == "text":
            # A line ending mid-sentence continues the paragraph; one ending in a full stop
            # closes it. Cheap, and it keeps recommendations from fusing into a wall of text.
            paragraph.append(block.text)
            if block.text.endswith((".", ":", "?", "!")):
                flush()
        else:                                     # table or omission: their own block
            flush()
            out.append(block.text)

    flush()
    return tidy("\n\n".join(out))


def extract_document(doc, pages=None):
    """PDF -> (Markdown, reference text, Report).

    The reference text is the PDF's plain text layer, written alongside. It is not a backup: it
    is the ORACLE the quality gates check the Markdown against — every word of the Markdown must
    occur in it, which catches a dropped letter, a fused word or an injected tag without needing
    a dictionary."""
    report = Report(doc_id=doc.doc_id)
    profile = dict(doc.extraction)
    with pdfplumber.open(doc.pdf_path) as pdf:
        diagnosis = triage(pdf)
        report.status, report.pages = diagnosis.status, len(pdf.pages)
        if not diagnosis.usable:
            report.notes.append(diagnosis.detail)
            return "", "", report

        body_size = body_font_size(pdf)
        running_heads = find_running_heads(pdf)
        # Front matter (cover, panel of authors and affiliations, table of contents) is skipped
        # by PAGE, declared per document in the manifest. It resists every general rule worth
        # having: the author block is laid out as a grid, so a table detector finds "tables" in
        # it; the contents page repeats every heading in the document, so heading detection
        # finds duplicates of all of them. Where the clinical text begins is a fact about one
        # PDF, read once by a human — three lines of TOML rather than a heuristic that will be
        # wrong on the next guideline.
        first = int(profile.get("first_page", 1))
        last = int(profile.get("last_page", len(pdf.pages)))
        selected = [p for p in pdf.pages
                    if first <= p.page_number <= last
                    and (pages is None or p.page_number in pages)]

        blocks, reference = [], []
        for page in selected:
            reference.append(expand_ligatures(page.extract_text() or ""))
            # Blocks are ordered WITHIN a page; page order is document order, so each page's
            # blocks are appended as a unit rather than globally re-sorted by `top`.
            blocks.extend(extract_page(page, body_size, running_heads, profile, report))
        markdown = assemble(blocks, report)
        report.chars = len(markdown)
    return markdown, "\n".join(reference), report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_pages(spec):
    if not spec:
        return None
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return range(int(lo), int(hi) + 1)
    return range(int(spec), int(spec) + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a manifest document from PDF to Markdown.")
    ap.add_argument("doc_id", nargs="?", help="doc_id from data/corpus.toml")
    ap.add_argument("--all", action="store_true", help="every document in the manifest")
    ap.add_argument("--pages", help="page or range (1-based, inclusive), e.g. 100-110")
    ap.add_argument("--out", help="output directory (default: data/markdown/)")
    ap.add_argument("--report", action="store_true", help="print the extraction report")
    ap.add_argument("--dry-run", action="store_true", help="extract and report, write nothing")
    args = ap.parse_args()

    if args.all:
        docs = list(corpus.documents())
    elif args.doc_id:
        docs = [d for d in corpus.documents() if d.doc_id == args.doc_id]
        if not docs:
            raise SystemExit(f"{args.doc_id!r} is not in data/corpus.toml. Known: "
                             f"{', '.join(d.doc_id for d in corpus.documents())}")
    else:
        ap.error("give a doc_id or --all")

    out_dir = Path(args.out) if args.out else Path(corpus.MARKDOWN_DIR)
    failures = []
    for doc in docs:
        print(f"{doc.doc_id} …", file=sys.stderr, flush=True)
        markdown, reference, report = extract_document(doc, _parse_pages(args.pages))
        if not markdown:
            failures.append(doc.doc_id)
        elif not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{doc.doc_id}.md").write_text(markdown, encoding="utf-8")
            # The reference travels WITH the Markdown, so `--out` produces a self-contained pair
            # that can be checked before anything replaces the committed corpus.
            ref_path = (out_dir / doc.reference if args.out
                        else Path(corpus.REFERENCE_DIR) / doc.reference)
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_text(reference, encoding="utf-8")
        if args.report or not markdown:
            print(report.render(), file=sys.stderr)

    if failures:
        raise SystemExit(f"\nNot transcribed: {', '.join(failures)}. A PDF whose text layer "
                         f"cannot be decoded must not enter the corpus as noise.")


if __name__ == "__main__":
    main()
