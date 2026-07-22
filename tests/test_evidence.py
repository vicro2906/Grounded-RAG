"""Citation integrity — the last barrier between a fabricated quote and the doctor.

The cases below are the ones that decide whether the sources panel can be trusted: what gets
certified as literal, what gets silently rewritten, and which evidence grades ride along.
"""
import pytest

from evidence import attribute, format_answer, section_label, split_items

ANSWER_QUOTE = "iniciar precozmente un TAR que incluya TDF o TAF y FTC o 3TC"


CHUNK = {
    "chunk_id": "c1",
    "doc_title": "Documento de consenso de GeSIDA sobre TAR",
    "year": 2022,
    "section_number": "7.4.4",
    "heading": "7.4.4. HEPATOPATÍAS",
    "text": ("7. SITUACIONES ESPECIALES > 7.4.4. HEPATOPATÍAS\n\n"
             "En pacientes con coinfección por VHB se recomienda iniciar precozmente un TAR "
             "que incluya TDF o TAF y FTC o 3TC (A-I). No debe suspenderse el tratamiento "
             "frente al VHB sin vigilancia estrecha (B-II)."),
}


# --- what must be certified ------------------------------------------------
def test_genuine_quote_is_exact_with_its_own_grade():
    status, shown, grades = attribute(
        "iniciar precozmente un TAR que incluya TDF o TAF y FTC o 3TC", CHUNK)
    assert status == "exact"
    assert grades == ["A-I"], "the grade must be scoped to the quoted clause, not the item"


def test_wording_drift_is_absorbed_by_fuzzy():
    """Punctuation/accents differ but the clinical content is identical -> still citable."""
    status, shown, _ = attribute(
        "En pacientes con coinfeccion por VHB, se recomienda iniciar precozmente un TAR que "
        "incluya TDF o TAF y FTC o 3TC", CHUNK)
    assert status == "fuzzy"
    assert "TDF o TAF" in shown


# --- what must be rejected -------------------------------------------------
def test_filler_quote_is_not_certified():
    """Regression: "se recomienda" matched literally in most chunks, so it was certified as an
    exact citation and inherited BOTH evidence grades of the item."""
    status, _, grades = attribute("se recomienda", CHUNK)
    assert status == "miss"
    assert grades == []


def test_quote_naming_a_different_drug_is_rejected():
    """Regression: this scored as fuzzy and the panel then displayed the REAL sentence
    (TDF o TAF), visually backing an answer that had said ABC."""
    status, shown, _ = attribute(
        "se recomienda iniciar precozmente un TAR que incluya ABC", CHUNK)
    assert status == "miss"
    assert shown == ""


def test_drug_divergence_is_caught_across_spellings():
    """The model writes full names, the guide uses abbreviations (and vice versa): the check
    must see «abacavir» and «TDF» as different drugs, not as untyped words."""
    status, _, _ = attribute(
        "se recomienda iniciar precozmente un TAR que incluya abacavir", CHUNK)
    assert status == "miss"


def test_same_drug_in_the_other_spelling_is_still_citable():
    chunk = {"text": "Se recomienda una pauta con tenofovir alafenamida y emtricitabina (A-I)."}
    assert attribute("Se recomienda una pauta con TAF y FTC", chunk)[0] == "fuzzy"


def test_quote_altering_a_figure_is_rejected():
    chunk = {"text": "Se recomienda profilaxis si el recuento de CD4 es menor de 200 células/µL (A-I)."}
    assert attribute("Se recomienda profilaxis si el recuento de CD4 es menor de 500 células/µL",
                     chunk)[0] == "miss"


def test_invented_sentence_is_rejected():
    assert attribute("se recomienda administrar azitromicina semanal de por vida", CHUNK)[0] == "miss"


# --- panel rendering -------------------------------------------------------
def test_rejected_quote_degrades_to_section_without_inventing_a_citation():
    answer = {"sufficient_information": True, "answer": "Se inicia TAR con ABC.",
              "sources_used": [{"ref": 1, "quote": "un TAR que incluya ABC"}],
              "follow_up_questions": []}
    out = format_answer(answer, {1: CHUNK})
    assert "sección consultada" in out
    assert "TDF o TAF" not in out, "a rejected quote must not pull the real sentence into view"


def test_a_reference_the_model_invented_is_ignored_not_crashed():
    """The model is asked for the numbers it used, and it can return [7] when only 5 chunks were
    given. That must drop silently: an IndexError here would take down an answer that is
    otherwise fine, at the very last step."""
    answer = {"sufficient_information": True, "answer": "Texto.",
              "sources_used": [{"ref": 7, "quote": ANSWER_QUOTE}], "follow_up_questions": []}
    out = format_answer(answer, {1: CHUNK})
    assert "FUENTES" not in out
    assert "Texto." in out, "the answer itself must survive a bad reference"


def test_two_quotes_from_the_same_section_are_one_source():
    """Grouping is by section, not by quote: the doctor should see one reference with two
    supporting sentences, not the same section listed twice."""
    answer = {"sufficient_information": True, "answer": "Texto.",
              "sources_used": [{"ref": 1, "quote": ANSWER_QUOTE},
                               {"ref": 2, "quote": "No debe suspenderse el tratamiento frente "
                                                   "al VHB sin vigilancia estrecha"}],
              "follow_up_questions": []}
    out = format_answer(answer, {1: CHUNK, 2: CHUNK})
    assert "FUENTES (1)" in out
    assert out.count("§7.4.4") == 1


def test_the_disclaimer_rides_with_every_visible_answer():
    answer = {"sufficient_information": True, "answer": "Texto.",
              "sources_used": [{"ref": 1, "quote": ANSWER_QUOTE}], "follow_up_questions": []}
    assert "no sustituye" in format_answer(answer, {1: CHUNK})


def test_insufficient_information_shows_no_sources_or_followups():
    answer = {"sufficient_information": False, "answer": "No disponible.",
              "sources_used": [], "follow_up_questions": []}
    out = format_answer(answer, {1: CHUNK})
    assert "INFORMACIÓN INSUFICIENTE" in out
    assert "FUENTES" not in out


# --- helpers ---------------------------------------------------------------
def test_split_items_drops_the_breadcrumb_and_separates_recommendations():
    chunk = {"text": "7. SITUACIONES > 7.1. INICIO\n\n"
                     "1. Primera recomendación (A-I).\n2. Segunda recomendación (B-II)."}
    items = split_items(chunk)
    assert [it["grades"] for it in items] == [["A-I"], ["B-II"]]


def test_section_label_uses_the_chunk_own_number():
    assert section_label(CHUNK).startswith("§7.4.4")
