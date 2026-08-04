"""The extractor's decisions, tested as pure functions — no PDF, no network.

Everything here is a rule that decides whether a piece of a guideline reaches the doctor intact,
gets dropped, or (worst) arrives subtly wrong. They are separable from pdfplumber on purpose: a
test that needs a 146-page PDF to run is a test nobody runs while changing the rule it covers.

The one contract worth naming: `extract_pdf.OMISSION` is what the extractor WRITES and
`quality.OMISSION_CANONICAL` is what the gate ACCEPTS. They live in different modules and would
drift silently — the gate would start rejecting correct output, and the obvious "fix" would be
to loosen the gate.
"""
import re

import pytest

from ingestion import extract_pdf as ex
from ingestion.quality import OMISSION_ACCEPTED, OMISSION_CANONICAL


# --- the cross-module contract ---------------------------------------------
@pytest.mark.parametrize("what,ending", [("Figura", "a"), ("Algoritmo", "o"), ("Tabla", "a")])
def test_the_marker_the_extractor_writes_is_the_one_the_gate_accepts(what, ending):
    marker = ex.OMISSION.format(what=what, o=ending, page=42)
    match = OMISSION_CANONICAL.search(marker)
    assert match, f"the gate does not recognise {marker!r}"
    assert match.group("cause") == what and match.group("page") == "42"


def test_a_lost_table_is_reported_rather_than_accepted():
    """The extractor emits a Tabla marker when a reconstruction fails its conservation checks.
    That marker must NOT be an accepted omission: a table's text is in the PDF, so a lost one is
    a bug to fix, not a limitation to record."""
    assert "Tabla" not in OMISSION_ACCEPTED
    assert {"Figura", "Algoritmo"} <= OMISSION_ACCEPTED


# --- triage ----------------------------------------------------------------
class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePDF:
    def __init__(self, text, pages=12):
        self.pages = [_FakePage(text) for _ in range(pages)]


def test_triage_accepts_a_healthy_text_layer():
    assert ex.triage(_FakePDF("Se recomienda iniciar TAR precozmente (A-I). " * 20)).usable


def test_triage_rejects_a_broken_cmap():
    """The real case: the eighth PDF is 81% alphabetic characters — a ratio check waves it
    through — while rendering `eficiencia` as `e(cid:332)ciencia`. The unmapped-glyph marker is
    the only reliable signal, and the diagnosis names the offending glyphs so the decision to
    fix or drop the document can be made on evidence."""
    text = "la e(cid:332)ciencia del tratamiento " * 30
    diagnosis = ex.triage(_FakePDF(text))
    assert not diagnosis.usable and diagnosis.status == "broken_cmap"
    assert "cid:332" in diagnosis.detail


def test_triage_rejects_a_scan():
    diagnosis = ex.triage(_FakePDF(""))
    assert not diagnosis.usable and diagnosis.status == "scanned"
    assert "OCR" in diagnosis.detail


# --- ligatures -------------------------------------------------------------
def test_ligatures_are_expanded_not_dropped():
    """Dropping them is what produced `fuconazol` for `fluconazol` in dosing text — a word no
    lexical search can match."""
    assert ex.expand_ligatures("eﬁcacia y ﬂuconazol") == "eficacia y fluconazol"


# --- heading hierarchy -----------------------------------------------------
@pytest.mark.parametrize("line,level", [
    ("1. INTRODUCCIÓN", 2),
    ("3.2. EVALUACIÓN", 3),
    ("3.2.1. RESISTENCIAS", 4),
    ("Recomendaciones", 4),
    ("TABLA 3. Combinaciones de TAR", 4),
    ("Anexo I", 2),
])
def test_the_numbering_decides_the_level(line, level):
    """`1.` -> H2, `1.2.` -> H3: the hierarchy the document itself declares, and the contract
    ingestion.chunk_guidelines parses back out of the Markdown."""
    assert ex.heading_level(line) == level


def test_running_text_is_not_a_heading():
    assert ex.heading_level("En España hay cinco ITINN comercializados.") is None


# --- rowspan / forward fill ------------------------------------------------
def test_a_group_label_is_propagated_down_the_rows_it_spans():
    """The pattern the previous converter read as «primera columna vacía en 71% de las filas
    (layout dañado)» and used to discard the whole table."""
    rows = [["INI", "BIC/FTC/TAF"], ["", "DTG/ABC/3TC"], ["", "DTG/3TC"]]
    filled, n = ex.fill_group_cells(rows, [0])
    assert [r[0] for r in filled] == ["INI", "INI", "INI"] and n == 2


def test_a_dose_is_never_carried_into_the_next_row():
    """This is the one place here that can invent a clinical fact, and a propagated value would
    be quoted LITERALLY — worse than the omission it replaces. A number is a VALUE, not a group
    label, so it never carries."""
    rows = [["600 mg", "EFV"], ["", "NVP"]]
    filled, n = ex.fill_group_cells(rows, [0])
    assert filled[1][0] == "" and n == 0


def test_an_evidence_grade_is_never_carried_either():
    rows = [["(A-I)", "recomendación"], ["", "otra"]]
    filled, n = ex.fill_group_cells(rows, [0])
    assert filled[1][0] == "" and n == 0


def test_a_row_with_nothing_of_its_own_gets_no_label():
    """An empty row is spacing, not a row the label owns."""
    rows = [["INI", "BIC/FTC/TAF"], ["", ""]]
    filled, n = ex.fill_group_cells(rows, [0])
    assert filled[1][0] == "" and n == 0


def test_nothing_is_filled_when_no_column_is_declared():
    rows = [["INI", "BIC"], ["", "DTG"]]
    assert ex.fill_group_cells(rows, [])[1] == 0


# --- the conservation gate -------------------------------------------------
GOOD = [["3er Fármaco", "Pauta"], ["INI", "BIC/FTC/TAF"], ["ITINN", "DOR+FTC/TAF"]]
GOOD_REGION = "3er Fármaco Pauta INI BIC/FTC/TAF ITINN DOR+FTC/TAF"


def test_a_faithful_table_passes():
    assert ex.table_is_faithful(GOOD, GOOD_REGION) == ""


def test_a_table_that_lost_a_column_is_rejected():
    """The failure that actually happened: thirteen captions shipped with no rows, taking the
    first-line regimens and the interaction matrix with them."""
    lost = [["3er Fármaco"], ["INI"], ["ITINN"]]
    assert "characters of the region survived" in ex.table_is_faithful(lost, GOOD_REGION)


def test_a_table_that_lost_only_its_doses_is_still_rejected():
    """The numeric check exists because losing a dose is both the most dangerous omission and
    the least visible one: the table still looks complete. Here the reconstruction keeps 98% of
    the region's characters — well past the content gate — and drops the one number."""
    prose = ("Comentario clínico suficientemente largo como para que el recuento de caracteres "
             "no sea lo que rechace esta tabla, sino la pérdida de la cifra")
    rows = [["Fármaco", "Comentario"], ["RAL", prose]]
    region = f"Fármaco Comentario RAL {prose} 400 mg"

    verdict = ex.table_is_faithful(rows, region)
    assert "numeric tokens survived" in verdict, verdict


def test_a_ragged_or_headerless_table_is_rejected():
    assert "ragged" in ex.table_is_faithful([["a", "b"], ["c"]], "a b c")
    assert "empty header" in ex.table_is_faithful([["", ""], ["c", "d"]], "c d")
    assert "fewer than two rows" in ex.table_is_faithful([["a", "b"]], "a b")


def test_a_wrapped_cell_does_not_read_as_invented_text():
    """A cell wrapping across lines interleaves with its neighbouring columns in the region's
    reading order. An earlier draft compared token multisets and discarded 7 of 10 real tables
    over exactly this; a gate that fails on correct input teaches you to ignore it."""
    rows = [["Fármaco", "Comentario"],
            ["DTG/3TC", "No recomendado con CD4 menor de 200/μL"]]
    region = "Fármaco Comentario\nDTG/3TC No recomendado\ncon CD4 menor\nde 200/μL"
    assert ex.table_is_faithful(rows, region) == ""


# --- rendering -------------------------------------------------------------
def test_a_narrow_table_is_rendered_as_a_grid():
    out = ex.render_table(GOOD, "")
    assert out.splitlines()[0] == "| 3er Fármaco | Pauta |"
    assert out.splitlines()[1] == "|---|---|"
    assert "| INI | BIC/FTC/TAF |" in out


def test_a_wide_matrix_becomes_one_record_per_row():
    """19 columns as Markdown is unreadable to the model, impossible to split without losing the
    header, and it cites badly. As records it is the unit the question actually has
    («¿interacción de abemaciclib con bictegravir?»), and each line is literal, verifiable text."""
    header = ["", "DTG", "BIC", "EFV", "ATV/r", "DRV/r", "TAF", "ABC"]
    row = ["abemaciclib", "", "", "X NR", "X hFCO1", "", "", ""]
    out = ex.render_table([header, row], "")
    assert out == "- abemaciclib: EFV «X NR»; ATV/r «X hFCO1»."
    assert "DTG" not in out, "empty cells must not be written: the matrices are sparse"


def test_a_record_row_without_a_label_is_dropped():
    header = ["", "DTG", "BIC", "EFV", "ATV/r", "DRV/r", "TAF", "ABC"]
    assert ex.render_table([header, ["", "", "", "X", "", "", "", ""]], "") == ""


# --- assembly --------------------------------------------------------------
def _blocks(*specs):
    return [ex.Block(i, kind, text, level)
            for i, (kind, text, level) in enumerate(specs)]


def test_lines_rejoin_into_paragraphs_until_a_sentence_closes():
    out = ex.assemble(_blocks(("text", "En pacientes con coinfección por VHB", 0),
                              ("text", "se recomienda iniciar TAR precozmente.", 0),
                              ("text", "Otro párrafo.", 0)), ex.Report("d"))
    assert out.startswith("En pacientes con coinfección por VHB se recomienda iniciar "
                          "TAR precozmente.\n\nOtro párrafo.")


def test_a_dropped_section_swallows_its_subsections_but_not_its_successor():
    """Bibliography is not clinical content and pollutes retrieval; the next real section must
    survive it."""
    report = ex.Report("d")
    out = ex.assemble(_blocks(("heading", "Bibliografía", 3),
                              ("text", "1. Workowski K. et al. 2021;", 0),
                              ("heading", "4.1. Sub", 4),
                              ("text", "más referencias", 0),
                              ("heading", "5. TRATAMIENTO", 2),
                              ("text", "contenido clínico.", 0)), report)
    assert "Workowski" not in out and "más referencias" not in out
    assert "## 5. TRATAMIENTO" in out and "contenido clínico." in out
    assert report.sections_dropped == 1


def test_a_table_is_its_own_block():
    out = ex.assemble(_blocks(("text", "Texto previo.", 0),
                              ("table", "| a | b |", 0),
                              ("text", "Texto posterior.", 0)), ex.Report("d"))
    assert out == "Texto previo.\n\n| a | b |\n\nTexto posterior.\n"


def test_the_space_a_removed_superscript_leaves_is_closed_up():
    """pdfplumber lifts citation superscripts onto their own line, leaving «suicidas , aunque».
    Left inline they would read as clinical figures, so they are dropped — and the gap closed."""
    assert ex.tidy("conductas suicidas , aunque") == "conductas suicidas, aunque\n"
