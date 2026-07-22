"""Retrieval helpers whose failure modes are silent.

`canonical_key` and `expand_abbrevs` exist because the guides write the same concept many
ways; when they stop collapsing a variant nothing crashes — the graph modes just quietly rank
six spellings of one concept instead of six concepts.
"""
from retrieval._common import canonical_key, expand_abbrevs, merge_dedup


# --- canonical_key: one key per concept ------------------------------------
def test_surface_variants_of_one_drug_collapse():
    keys = {canonical_key(v) for v in
            ("DTG", "Dolutegravir", "dolutegravir", "DTG, Dolutegravir",
             "Dolutegravir (DTG)", "DTG=Dolutegravir")}
    assert len(keys) == 1, f"expected one canonical key, got {keys}"


def test_accents_and_punctuation_do_not_split_a_concept():
    assert canonical_key("Coinfección VHB") == canonical_key("coinfeccion, vhb")


def test_distinct_concepts_keep_distinct_keys():
    assert canonical_key("TDF") != canonical_key("TAF")


def test_longest_name_contracts_first():
    """«tenofovir alafenamida» must contract to TAF, not be eaten by the shorter «tenofovir»."""
    assert canonical_key("tenofovir alafenamida") == canonical_key("TAF")


def test_canonical_key_of_empty_input_is_empty():
    assert canonical_key("") == "" and canonical_key(None) == ""


# --- expand_abbrevs: query written both ways -------------------------------
def test_abbreviation_is_expanded_to_both_forms():
    out = expand_abbrevs("Pauta con DTG")
    assert "DTG" in out and "dolutegravir" in out.lower()


def test_already_expanded_text_is_left_alone():
    text = "dolutegravir (DTG)"
    assert expand_abbrevs(text) == text


# --- merge_dedup: the mode's own selection wins ----------------------------
def test_merge_dedup_keeps_first_occurrence_and_order():
    a = [{"chunk_id": "1", "text": "a"}, {"chunk_id": "2", "text": "b"}]
    b = [{"chunk_id": "2", "text": "b"}, {"chunk_id": "3", "text": "c"}]
    assert [p["chunk_id"] for p in merge_dedup(a, b)] == ["1", "2", "3"]


def test_merge_dedup_falls_back_to_text_without_chunk_id():
    a = [{"text": "misma frase"}]
    b = [{"text": "misma frase"}, {"text": "otra"}]
    assert len(merge_dedup(a, b)) == 2
