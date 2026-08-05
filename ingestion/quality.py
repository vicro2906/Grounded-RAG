"""Quality gates over the corpus artifacts (Markdown sources and the chunks built from them).

WHY THIS EXISTS. Every defect this module checks was found by reading the corpus by hand, long
after it had been indexed, embedded and answered from. None of them announced itself: the
guideline chapter holding the recommended first-line regimens was retrieved and cited while
containing zero table rows, the reranker was fed chunks 8x over their stated budget, and a drug
name lost a letter (`fluconazol` -> `fuconazol`) so no lexical search could ever find it. A
corpus can be silently wrong in ways an end-to-end evaluation scores as merely mediocre.

So the gates are the contract, not the audit: each one states an invariant the corpus must hold,
fails the build when it does not, and carries the number measured on the day it was written so a
regression is visible as a number going up rather than as a vague sense that answers got worse.

WHAT A GATE IS ALLOWED TO BE. A gate must be deterministic, offline and cheap (the whole suite
runs in ~2 s over 517 chunks). It never calls a model — a gate judged by an LLM would have the
same failure mode as the thing it is checking.

THE ORACLE. The strongest gate here needs no dictionary and no model: `data/textos/` holds the
PDF's OWN text layer, extracted from the same pages. Every alphabetic token of the Markdown must
appear in it. That single rule catches a dropped ligature (`fuconazol` for `fluconazol`), a word
split by markup (`c_ _on` for `con`), a word fused to its neighbour (`dela`), an injected `<br>`
— and, in principle, any text an extractor invented. The manifest is what makes it possible: the
file names pair by no rule at all (`VIH_TB.md` against `textos/TB_VIH/`).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import corpus
from ingestion.chunk_guidelines import (MAX_TOKENS, RE_GRADE, RE_RECORD_ROW, RE_TABLE_ROW,
                                        count_tokens, has_tabular_content, is_mostly_table)

# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

# Window in which a chunk's opening must identify it uniquely. MIRRORS
# retrieval/_common.py:PREFIX_CHARS, which is the consumer that breaks when this is violated:
# it maps an index's stored text back to our citable payload by prefix, so two chunks sharing an
# opening make it hand back the WRONG section under the right claim. The two constants are
# asserted equal in tests/test_corpus_quality.py rather than imported, because `retrieval/`
# drags in the whole retrieval stack and ingestion must stay standalone.
PREFIX_CHARS = 120

# Typographic ligatures. The PDFs contain these as single glyphs; they must be expanded on
# extraction, never dropped. Dropping them is what produced `fuconazol` for `fluconazol`.
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}

# Words the extractor legitimately ADDS, so they cannot be in the PDF: the vocabulary of the
# omission marker itself. Deliberately tiny and closed — every other unknown token is damage.
MARKER_VOCABULARY = frozenset("figura algoritmo tabla omitida omitido p consultar pdf original".split())
_ALPHA = re.compile(r"[a-zñ]+")

# The ONE shape an omission marker may take. Spanish because the doctor reads it in the corpus.
# The page number is what makes it actionable (the doctor can open the PDF at that page) and the
# cause is what lets G4 accept an unrecoverable figure while rejecting a lost table.
OMISSION_CANONICAL = re.compile(
    r"> _\[(?P<cause>Figura|Algoritmo|Tabla) omitid[ao] — p\. (?P<page>\d+) — "
    r"consultar PDF original\]_")
# Anything that *looks* like an omission marker, so non-canonical ones are caught rather than
# ignored. The corpus built by the ad-hoc scripts holds 27 distinct spellings.
OMISSION_ANY = re.compile(r"\[[^\]]*omitid[ao][^\]]*\]")
# A table that could not be extracted is a BUG, not an accepted omission: its text is in the
# PDF's text layer and the extractor is expected to recover it.
OMISSION_ACCEPTED = {"Figura", "Algoritmo"}

# A table caption, as a heading or as a bold-only line. Both forms occur; the bold-only ones are
# why four separate tables ended up inside a single chunk.
RE_TABLE_CAPTION = re.compile(
    r"^(?:#{1,6}\s+|\*\*)\s*(?P<caption>(?:TABLA|Tabla)\s*\d+[.:]?[^\n*]*)", re.MULTILINE)

# An emphasis run closed and immediately reopened. Well-formed Markdown never contains this; the
# PDF converter produced it whenever it met a span drawn out of flow, and the damage is not
# cosmetic: it splits words across the break (`c_ _on` for `con`) and strands evidence grades
# mid-sentence.
RE_BROKEN_EMPHASIS = re.compile(r"_\s+_")
# An evidence grade whose sentence continues after it in lowercase: a full stop followed by a
# lowercase word is never a sentence end, so the marker was injected into the middle of a clause.
# Only emphasis markup and at most ONE newline may sit in between — otherwise `(A-II)** .\n\n
# ##### b. Título` (a grade correctly closing its sentence, followed by a lettered heading) reads
# as a violation, and a gate that cries wolf gets switched off.
RE_GRADE_MID_SENTENCE = re.compile(
    r"\(\s*\*{0,2}\s*[ABC]\s*-?\s*I{1,3}\s*\*{0,2}\s*\)"   # the grade
    r"[*_ \t]{0,6}\.[*_ \t]{0,4}\n?[*_ \t]{0,4}"            # a full stop, maybe wrapped in markup
    r"[a-záéíóúñü]")                                        # ... and the clause carries on
# Grade-shaped text the canonical regex does NOT capture, so a grade lost to stray markup
# (`(_ _**A-I).**_`) or an unsupported range (`(AI-AII)`) is reported instead of vanishing.
RE_GRADE_LOOSE = re.compile(
    r"\([_*\s]*[ABC][_*\s]*-?[_*\s]*I{1,3}[_*\s.]*\)"       # a grade with markup anywhere inside
    r"|\([ABC]I{1,3}\s*-\s*[ABC]I{1,3}\)")                  # a range, e.g. (AI-AII)

# Fraction of the source Markdown that must survive into the chunks.
MIN_COVERAGE = 0.995
# Paragraphs shorter than this are not worth tracking for coverage (captions, stray fragments).
COVERAGE_MIN_PARAGRAPH = 40


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Finding:
    """One violation. `subject` locates it (chunk id, file, caption) so it can be fixed."""
    subject: str
    detail: str


@dataclass(frozen=True)
class Gate:
    code: str                    # "G1"
    title: str                   # the invariant, stated positively
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings


@dataclass(frozen=True)
class Source:
    """A document as the gates see it: what was produced, and what it should have come from."""
    name: str                    # the chunks' `source_file`
    markdown: str
    reference: str | None        # the PDF's own text layer; None when the manifest names none


def _gate(code: str, title: str, findings: list[Finding]) -> Gate:
    return Gate(code=code, title=title, findings=tuple(findings))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return " ".join((text or "").split())


def body_of(chunk: dict) -> str:
    """The chunk's own content, without the `A > B > C` breadcrumb it is prefixed with.

    The chunker prepends the breadcrumb to `text` so the embedding sees the section context.
    That makes `text` a poor proxy for "does this chunk say anything", which is exactly what G2
    asks: one chunk in the corpus is 100% breadcrumb and 0% body. Note that such a chunk has no
    blank line at all — it is the breadcrumb and nothing else — so the split has to treat a
    missing separator as an EMPTY body rather than as "no breadcrumb here"."""
    text = chunk.get("text") or ""
    head, sep, rest = text.partition("\n\n")
    if " > " in head and len(head) < 300:
        return rest.strip()
    return text


def _fold(text: str) -> str:
    """Lowercased and accent-folded, so `función` and `funcion` are one token."""
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _coverage_units(md: str) -> list[str]:
    """The pieces of a source that must each survive into some chunk.

    Prose is tracked as PARAGRAPHS, but a table or a run of records is tracked ROW BY ROW. The
    chunker splits those by row — repeating the header — precisely so a 7 500-token matrix does
    not land in one chunk, so demanding that the whole block appear verbatim somewhere would
    report the fix as a loss. Rows are the right unit: it is a row going missing that loses a
    drug interaction."""
    out = []
    for block in re.split(r"\n\s*\n", md):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        lines = block.splitlines()
        if any(RE_TABLE_ROW.match(l) for l in lines) or sum(
                1 for l in lines if RE_RECORD_ROW.match(l)) >= 2:
            out.extend(l.strip() for l in lines if len(l.strip()) >= COVERAGE_MIN_PARAGRAPH)
        elif len(block) >= COVERAGE_MIN_PARAGRAPH:
            out.append(block)
    return out


# ---------------------------------------------------------------------------
# Gates over the CHUNKS
# ---------------------------------------------------------------------------
def g01_size_budget(chunks: list[dict]) -> Gate:
    findings = [
        Finding(c.get("chunk_id", "?"),
                f"{count_tokens(indexed_text(c))} tokens > {MAX_TOKENS} "
                f"({c.get('source_file')} · {c.get('heading')})")
        for c in chunks if count_tokens(indexed_text(c)) > MAX_TOKENS
    ]
    return _gate("G1", f"No chunk exceeds {MAX_TOKENS} tokens", findings)


def g02_non_empty_body(chunks: list[dict]) -> Gate:
    findings = [
        Finding(c.get("chunk_id", "?"),
                f"body is empty, only the breadcrumb ({c.get('source_file')} · {c.get('heading')})")
        for c in chunks if not body_of(c).strip()
    ]
    return _gate("G2", "Every chunk carries content of its own", findings)


def g03_unique_openings(chunks: list[dict]) -> Gate:
    groups: dict[str, list[str]] = {}
    for c in chunks:
        groups.setdefault(_norm(c.get("text", ""))[:PREFIX_CHARS], []).append(
            c.get("chunk_id", "?"))
    findings = [
        Finding(prefix[:60] + "…", f"{len(ids)} chunks share this opening: {', '.join(ids[:5])}"
                                   + ("…" if len(ids) > 5 else ""))
        for prefix, ids in groups.items() if len(ids) > 1
    ]
    return _gate("G3", f"A chunk's first {PREFIX_CHARS} characters identify it uniquely", findings)


def g10_content_type_matches_body(chunks: list[dict]) -> Gate:
    """Both directions of the label, but not symmetrically.

    A chunk typed `table` with no table in it is always a bug — that is the shape of the
    `9. TABLAS` chapter whose 13 captions carry zero rows. The converse is weaker: an
    `abbreviations` section is legitimately a table, and a `recommendations` one may end with a
    small one. Only `text` is a real mislabel, and only when the body is DOMINATED by rows,
    which is the classifier's own rule (`is_mostly_table`)."""
    findings = []
    for c in chunks:
        body, ctype = body_of(c), c.get("content_type")
        if ctype == "table" and not has_tabular_content(body):
            findings.append(Finding(c.get("chunk_id", "?"),
                                    f"typed 'table' but holds no table data ({c.get('heading')})"))
        elif ctype == "text" and is_mostly_table(body):
            findings.append(Finding(c.get("chunk_id", "?"),
                                    f"typed 'text' but the body is a table ({c.get('heading')})"))
    return _gate("G10", "A chunk typed 'table' holds table data, and a table is never typed 'text'",
                 findings)


def indexed_text(chunk: dict) -> str:
    """What actually gets embedded and BM25-indexed: the breadcrumb-prefixed form when the chunk
    has one, the citable text otherwise. The size budget is about THIS string — it exists
    because of the embedder's input limit and the reranker's scoring window, neither of which
    ever sees the citable text on its own."""
    return chunk.get("text_for_retrieval") or chunk.get("text", "")


def g11_declared_token_count(chunks: list[dict]) -> Gate:
    findings = [
        Finding(c.get("chunk_id", "?"),
                f"n_tokens={c.get('n_tokens')} but the tokenizer says "
                f"{count_tokens(indexed_text(c))}")
        for c in chunks if c.get("n_tokens") != count_tokens(indexed_text(c))
    ]
    return _gate("G11", "n_tokens is the real tokenizer count, not an estimate", findings)


# ---------------------------------------------------------------------------
# Gates over the MARKDOWN sources
# ---------------------------------------------------------------------------
def g04_omission_markers(sources: list[Source]) -> Gate:
    findings = []
    for src in sources:
        md = src.markdown
        canonical = [m.span() for m in OMISSION_CANONICAL.finditer(md)]
        for m in OMISSION_ANY.finditer(md):
            if not any(lo <= m.start() and m.end() <= hi for lo, hi in canonical):
                findings.append(Finding(src.name,
                                        f"non-canonical omission marker: {m.group(0)[:90]}"))
        for m in OMISSION_CANONICAL.finditer(md):
            if m.group("cause") not in OMISSION_ACCEPTED:
                findings.append(Finding(
                    src.name, f"a {m.group('cause').lower()} was omitted (p. {m.group('page')}); "
                              "its text is in the PDF and must be recovered, not skipped"))
    return _gate("G4", "Every omission is a figure or an algorithm, marked with cause and page",
                 findings)


def g05_faithful_to_the_pdf(sources: list[Source]) -> Gate:
    """Two checks with one purpose: no character of the source may be invented or lost.

    Raw ligature glyphs are a forward guard — they must be EXPANDED, and dropping them is what
    turned `fluconazol` into `fuconazol` in dosing text. The token oracle is the real one: any
    word in the Markdown that does not occur anywhere in the PDF's own text layer either was
    damaged in extraction or was never in the document."""
    findings = []
    for src in sources:
        for glyph, repl in LIGATURES.items():
            if glyph in src.markdown:
                findings.append(Finding(
                    src.name,
                    f"raw ligature {glyph!r} ({src.markdown.count(glyph)}x) — expand it to {repl!r}"))
        if src.reference is None:
            findings.append(Finding(src.name, "no reference text to check against; the manifest "
                                              "must name one so extraction damage is detectable"))
            continue
        known = set(_ALPHA.findall(_fold(src.reference))) | MARKER_VOCABULARY
        unknown: dict[str, int] = {}
        for token in _ALPHA.findall(_fold(src.markdown)):
            if token not in known:
                unknown[token] = unknown.get(token, 0) + 1
        for token, n in sorted(unknown.items(), key=lambda kv: -kv[1]):
            findings.append(Finding(src.name, f"{token!r} ({n}x) does not occur in the PDF text"))
    return _gate("G5", "Every word of the Markdown occurs in the PDF's own text layer", findings)


def g06_evidence_grades(sources: list[Source]) -> Gate:
    findings = []
    for src in sources:
        name, md = src.name, src.markdown
        for m in RE_GRADE_MID_SENTENCE.finditer(md):
            findings.append(Finding(name, f"grade injected mid-sentence: …{md[m.start():m.end()]}…"))
        broken = len(RE_BROKEN_EMPHASIS.findall(md))
        if broken:
            findings.append(Finding(name, f"{broken} broken emphasis runs ('_ _'), which split "
                                          "words and strand grades"))
        for m in RE_GRADE_LOOSE.finditer(md):
            if not RE_GRADE.fullmatch(m.group(0)):
                findings.append(Finding(name, f"grade not captured by the scheme: {m.group(0)!r}"))
    return _gate("G6", "Evidence grades close their sentence and match the grading scheme",
                 findings)


def g07_tables_have_data(sources: list[Source]) -> Gate:
    findings = []
    for src in sources:
        name, md = src.name, src.markdown
        captions = list(RE_TABLE_CAPTION.finditer(md))
        for i, m in enumerate(captions):
            end = captions[i + 1].start() if i + 1 < len(captions) else len(md)
            # Stop at the next heading too: a caption's data cannot live past its own section.
            nxt = re.search(r"^#{1,6}\s", md[m.end():end], re.MULTILINE)
            span = md[m.end():m.end() + nxt.start()] if nxt else md[m.end():end]
            if not has_tabular_content(span):
                findings.append(Finding(name, f"{m.group('caption').strip()[:70]} — caption with "
                                              "no table rows beneath it"))
    return _gate("G7", "Every table caption is followed by actual table data", findings)


# ---------------------------------------------------------------------------
# Gate across BOTH
# ---------------------------------------------------------------------------
def g08_coverage(chunks: list[dict], sources: list[Source]) -> Gate:
    findings = []
    for src in sources:
        name = src.name
        haystack = "\n".join(_norm(c.get("text", "")) for c in chunks
                             if c.get("source_file") == name)
        units = _coverage_units(src.markdown)
        if not units:
            continue
        missing = [u for u in units if _norm(u) not in haystack]
        covered = 1 - len(missing) / len(units)
        if covered < MIN_COVERAGE:
            findings.append(Finding(
                name, f"{covered:.1%} of the source reached the chunks (min {MIN_COVERAGE:.1%}); "
                      f"{len(missing)} lost, e.g. {_norm(missing[0])[:70]!r}"))
    return _gate("G8", f"At least {MIN_COVERAGE:.1%} of each source reaches the chunks", findings)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def g09_manifest_covers_the_corpus(sources: list[Source]) -> Gate:
    """Manifest and corpus directory must be in BIJECTION.

    One direction is already enforced where it matters (the chunker refuses an unlisted file),
    but the other is not: a manifest row whose Markdown was renamed or deleted goes unnoticed
    until a run silently answers from one document fewer."""
    findings = []
    listed = {doc.markdown for doc in corpus.documents()}
    for src in sources:
        if src.name not in listed:
            findings.append(Finding(src.name, "no entry in data/corpus.toml"))
    present = {src.name for src in sources}
    for doc in corpus.documents():
        if doc.markdown not in present:
            findings.append(Finding(doc.doc_id, f"manifest names {doc.markdown!r}, which is not "
                                                "among the sources being built"))
    return _gate("G9", "The manifest and the corpus directory describe the same documents",
                 findings)


def audit(chunks: list[dict], sources: list[Source] | None = None) -> list[Gate]:
    """Run every gate the given inputs support, in code order.

    Without `sources` only the chunk-level gates run, which is what lets the chunker check its
    own output before anything is written."""
    gates = [g01_size_budget(chunks), g02_non_empty_body(chunks), g03_unique_openings(chunks)]
    if sources:
        gates += [g04_omission_markers(sources), g05_faithful_to_the_pdf(sources),
                  g06_evidence_grades(sources), g07_tables_have_data(sources),
                  g08_coverage(chunks, sources), g09_manifest_covers_the_corpus(sources)]
    gates += [g10_content_type_matches_body(chunks), g11_declared_token_count(chunks)]
    return sorted(gates, key=lambda g: (len(g.code), g.code))


def load_sources(md_files: list[Path]) -> list[Source]:
    """Pair each Markdown with the reference text the manifest says it came from.

    A file with no manifest entry still yields a Source (with no reference) rather than raising,
    so G9 can REPORT it as a finding. A gate that crashes tells you less than one that fails."""
    out = []
    for path in md_files:
        try:
            reference_path = Path(corpus.document_for_markdown(path.name).reference_path)
            reference = (reference_path.read_text(encoding="utf-8", errors="replace")
                         if reference_path.is_file() else None)
        except SystemExit:
            reference = None
        out.append(Source(name=path.name,
                          markdown=path.read_text(encoding="utf-8"),
                          reference=reference))
    return out


def format_report(gates: list[Gate], max_findings: int = 5) -> str:
    """A report meant to be READ: the failures first, each with a handful of examples, because a
    gate reporting '178 violations' without naming one is not actionable."""
    lines = []
    for g in gates:
        mark = "PASS" if g.passed else "FAIL"
        count = "" if g.passed else f"  ({len(g.findings)} findings)"
        lines.append(f"[{mark}] {g.code}  {g.title}{count}")
        for f in g.findings[:max_findings]:
            lines.append(f"         · {f.subject}: {f.detail}")
        if len(g.findings) > max_findings:
            lines.append(f"         · … and {len(g.findings) - max_findings} more")
    failed = [g.code for g in gates if not g.passed]
    lines.append("")
    lines.append(f"{len(gates) - len(failed)}/{len(gates)} gates passed"
                 + (f"; failing: {', '.join(failed)}" if failed else ""))
    return "\n".join(lines)
