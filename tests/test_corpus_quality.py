"""The quality gates, run against the corpus that actually ships.

TWO KINDS OF TEST LIVE HERE, and the split is the point.

`test_gate` runs each gate over the COMMITTED `data/chunks/chunks.jsonl` and `data/markdown/`
— the exact bytes that get embedded, indexed and cited. Six of the ten currently fail, and they
are marked `xfail(strict=True)` with the number measured on 2026-07-31 in the reason. Strict is
what makes this useful: when the extractor and chunker rewrite fixes one, the test FAILS BECAUSE
IT PASSED, and whoever did the work has to come here and delete the entry. The debt cannot be
paid off silently, and it cannot be re-incurred silently either.

`test_gate_detects_*` feeds each gate a hand-made violation. Without these, a gate that stopped
detecting anything would show up as a row of green ticks — which is precisely the failure mode
the gates exist to prevent, turned on itself.

NOT COVERED YET, on purpose: G5 only checks that ligature glyphs are expanded rather than left
raw, so it passes today while the corpus still contains 91 words that LOST a letter
(`fuconazol` for `fluconazol`). Catching those needs the PDF's own text layer as an oracle, and
that needs the document manifest to know which reference text belongs to which Markdown. It
arrives with the manifest as G5b.
"""
import json
import re
from pathlib import Path

import pytest

import corpus
from ingestion.quality import (PREFIX_CHARS, Finding, audit, format_report, g01_size_budget,
                               g02_non_empty_body, g03_unique_openings, g04_omission_markers,
                               g05_ligatures_expanded, g06_evidence_grades, g07_tables_have_data,
                               g08_coverage, g10_content_type_matches_body,
                               g11_declared_token_count, load_sources)

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
MARKDOWN = ROOT / "data" / "markdown"

# Every gate `audit` is expected to run. Hardcoded rather than derived so that adding a gate
# without deciding whether it passes today is itself a test failure (see test_audit_runs_every_gate).
GATE_CODES = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11"]

# Gates the SHIPPED corpus fails, with what was measured on 2026-07-31. Delete an entry the
# moment the corpus stops failing it — `strict=True` will insist.
KNOWN_DEBT = {
    "G1": "89 of 517 chunks exceed the 900-token budget: the corpus was built with a "
          "characters/4 estimate instead of the tokenizer, and tables are never split",
    "G2": "1 chunk (TAR_2022 'TABLA 2') is 100% breadcrumb and 0% body",
    "G3": "42 groups of chunks share their opening 120 characters (178 chunks), because the "
          "breadcrumb is prepended to the citable text",
    "G4": "112 omission markers in 27 different spellings, none carrying a page, and most of "
          "them tables whose text is recoverable from the PDF",
    "G6": "21 findings: evidence grades injected mid-sentence plus 363 broken emphasis runs "
          "that split words ('c_ _on' for 'con')",
    "G7": "33 table captions with no rows beneath them, including TABLA 1-12 of TAR_2022 "
          "(first-line regimens and the interaction matrix)",
    "G10": "14 chunks typed 'table' that hold no table data",
    "G11": "514 of 517 chunks declare a characters/4 estimate as their token count",
}


@pytest.fixture(scope="module")
def shipped_gates() -> dict:
    """The gates over the committed corpus, keyed by code. Module-scoped: the audit reads every
    chunk and every source once (~2 s) and nothing here mutates them."""
    chunks = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    sources = load_sources(sorted(MARKDOWN.glob("*.md")))
    return {g.code: g for g in audit(chunks, sources)}


def _case(code: str):
    debt = KNOWN_DEBT.get(code)
    return pytest.param(code, marks=[pytest.mark.xfail(strict=True, reason=debt)] if debt else [])


@pytest.mark.parametrize("code", [_case(c) for c in GATE_CODES])
def test_gate(shipped_gates, code):
    gate = shipped_gates[code]
    assert gate.passed, "\n" + format_report([gate], max_findings=10)


def test_audit_runs_every_gate(shipped_gates):
    """A gate nobody runs is a comment. Adding one to `quality.audit` without listing it here
    (and deciding whether today's corpus passes it) fails the build."""
    assert sorted(shipped_gates) == sorted(GATE_CODES)


def test_prefix_window_matches_the_consumer():
    """G3's window must be the one `retrieval/_common.get_prefix_lookup` actually uses to map an
    index's text back to our citable payload. If they drift, the gate certifies uniqueness over
    a window nobody looks at, and the bridge can still hand back the wrong section."""
    from retrieval._common import PREFIX_CHARS as CONSUMER_WINDOW
    assert PREFIX_CHARS == CONSUMER_WINDOW


# ---------------------------------------------------------------------------
# Each gate against a hand-made violation: proof the detector still detects.
# ---------------------------------------------------------------------------
def _chunk(**over) -> dict:
    base = {"chunk_id": "c1", "source_file": "guia.md", "heading": "1. TEMA",
            "content_type": "text", "n_tokens": 0, "text": "1. TEMA\n\nTexto clínico."}
    base["n_tokens"] = len(base["text"]) // 4
    return {**base, **over}


def test_gate_detects_oversized_chunk():
    big = _chunk(text="1. TEMA\n\n" + "palabra " * 5000)
    assert not g01_size_budget([big]).passed
    assert g01_size_budget([_chunk()]).passed


def test_gate_detects_breadcrumb_only_chunk():
    """The real one has no blank line at all, so it must not read as 'a chunk without a
    breadcrumb whose whole text is its body'."""
    assert not g02_non_empty_body([_chunk(text="9. TABLAS > TABLA 2. Recomendaciones")]).passed
    assert g02_non_empty_body([_chunk()]).passed


def test_gate_detects_shared_opening():
    shared = "A > B > " + "x" * 200
    twins = [_chunk(chunk_id="a", text=shared + "\n\nuno"),
             _chunk(chunk_id="b", text=shared + "\n\ndos")]
    assert not g03_unique_openings(twins).passed
    assert g03_unique_openings([twins[0]]).passed


def test_gate_accepts_only_canonical_figure_omissions():
    ok = "> _[Figura omitida — p. 42 — consultar PDF original]_"
    assert g04_omission_markers({"g.md": ok}).passed
    # A lost TABLE is a bug even in canonical form: its text is in the PDF's text layer.
    lost_table = "> _[Tabla omitida — p. 42 — consultar PDF original]_"
    assert not g04_omission_markers({"g.md": lost_table}).passed
    # The 27 free-form spellings the ad-hoc converters produced carry no page and no fixed cause.
    assert not g04_omission_markers({"g.md": "> _[Tabla omitida — consultar PDF original]_"}).passed


def test_gate_detects_raw_ligature():
    assert not g05_ligatures_expanded({"g.md": "eﬁcacia del TAR"}).passed
    assert g05_ligatures_expanded({"g.md": "eficacia del TAR"}).passed


def test_gate_detects_grade_defects():
    mid = "se debe evitar la interrupción **(A-II).** de una pauta eficaz frente a VHB"
    assert not g06_evidence_grades({"g.md": mid}).passed
    assert not g06_evidence_grades({"g.md": "la CVP c_ _on una técnica"}).passed
    assert not g06_evidence_grades({"g.md": "El cambio a RAL es adecuado (_ _**A-I).**_"}).passed
    assert g06_evidence_grades({"g.md": "Se recomienda iniciar TAR precozmente (A-I)."}).passed


def test_gate_accepts_a_grade_that_closes_its_sentence_before_a_lettered_heading():
    """`(A-II).` followed by `##### b. Título` is correct: the lowercase letter belongs to the
    heading, not to a continuing clause. The first draft of this gate flagged it."""
    assert g06_evidence_grades({"g.md": "adherencia **(A-II)** .\n\n##### b. Papel del farmacéutico"}).passed


def test_gate_detects_caption_without_table():
    empty = "#### TABLA 3. Combinaciones de TAR de inicio\n\n> _[Imagen omitida]_\n\nNotas: †"
    assert not g07_tables_have_data({"g.md": empty}).passed
    rows = "#### TABLA 3. Combinaciones\n\n|3er fármaco|Pauta|\n|---|---|\n|INI|BIC/FTC/TAF|"
    assert g07_tables_have_data({"g.md": rows}).passed


def test_gate_accepts_a_wide_matrix_serialised_as_records():
    """TABLA 9 is 19 columns wide; it ships as one record per drug row, not as a Markdown grid.
    G7 and G10 must recognise that shape or the fix would read as a failure."""
    records = ("#### TABLA 9. Asociaciones contraindicadas\n\n"
               "- abemaciclib: con DTG «X hFCO1»; con BIC «X NR».\n"
               "- abiraterona: con EFV «X NR».")
    assert g07_tables_have_data({"g.md": records}).passed
    assert g10_content_type_matches_body([_chunk(content_type="table", text=records)]).passed


def test_gate_detects_lost_paragraph():
    md = "# Guía\n\n" + "Un párrafo clínico suficientemente largo como para contar en cobertura.\n"
    assert not g08_coverage([_chunk(text="1. TEMA\n\notra cosa")], {"guia.md": md}).passed


def test_gate_detects_mislabelled_table():
    table_rows = "1. TEMA\n\n|a|b|\n|---|---|\n|1|2|\n|3|4|"
    assert not g10_content_type_matches_body([_chunk(content_type="table", text="1. T\n\nprosa")]).passed
    assert not g10_content_type_matches_body([_chunk(content_type="text", text=table_rows)]).passed
    # An abbreviations list is legitimately tabular; only 'text' is a real mislabel.
    assert g10_content_type_matches_body([_chunk(content_type="abbreviations", text=table_rows)]).passed


def test_gate_detects_estimated_token_count():
    assert not g11_declared_token_count([_chunk(n_tokens=3)]).passed


def test_report_names_the_offenders():
    """A gate reporting '178 violations' without naming one cannot be acted on."""
    report = format_report([g01_size_budget([_chunk(chunk_id="culpable",
                                                    text="x\n\n" + "palabra " * 5000)])])
    assert "FAIL" in report and "culpable" in report and "G1" in report


# ---------------------------------------------------------------------------
# The corpus GENERATION switch: content is only half of "the corpus is coherent";
# the other half is that all four artifacts describe the same text.
# ---------------------------------------------------------------------------
def test_v1_points_at_the_artifacts_that_are_already_live():
    """The v1 names are legacy and must keep matching what is in Qdrant Cloud and on disk. If
    they drift, adopting corpus.py silently orphans a 1.5-hour index build and points retrieval
    at a store that does not exist."""
    v1 = corpus.LAYOUTS["v1"]
    assert (v1.chunks, v1.lightrag, v1.hipporag, v1.collection) == (
        "chunks.jsonl", "lightrag_store", "hipporag_store", "guias_vih_hibrida_ctx")


def test_no_generation_shares_an_artifact_with_another():
    """The whole point of the switch: chunks v2 sitting next to a v1 graph store is the state
    that wakes the prefix fallback and mis-cites sections. If two generations share even one
    artifact name, that state becomes representable again."""
    for field in ("chunks", "lightrag", "hipporag", "collection"):
        names = [getattr(layout, field) for layout in corpus.LAYOUTS.values()]
        assert len(names) == len(set(names)), f"{field} is shared across corpus generations"


def test_unknown_generation_fails_loudly(monkeypatch):
    """A typo in CORPUS_VERSION must not quietly fall back to v1 and answer from the old index."""
    import importlib
    monkeypatch.setenv("CORPUS_VERSION", "v99")
    try:
        with pytest.raises(SystemExit, match="v1"):
            importlib.reload(corpus)
    finally:
        monkeypatch.delenv("CORPUS_VERSION", raising=False)
        importlib.reload(corpus)


def test_consumers_derive_their_locations_from_the_switch():
    from retrieval import _common
    import rag
    assert _common.CHUNKS_PATH == corpus.chunks_path()
    assert rag.COLLECTION_HYBRID == corpus.qdrant_collection()


def test_no_module_hardcodes_an_artifact_location():
    """The architectural guard, in the shape of tests/test_llm_client.py: the drift is invisible
    otherwise. A hardcoded `data/lightrag_store` keeps working perfectly on v1 and only breaks on
    the day someone switches generation — by then, quietly, against the wrong store."""
    literals = re.compile(r"\"(chunks\.jsonl|lightrag_store|hipporag_store|guias_vih_hibrida[a-z_]*)\"")
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part.startswith(".") for part in rel.parts) or rel.parts[0] == "tests":
            continue
        if path.name == "corpus.py":            # the one module allowed to name them
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if literals.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, ("derive these from corpus.py so every artifact moves together:\n"
                           + "\n".join(offenders))
