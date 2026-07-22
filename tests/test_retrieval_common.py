"""Retrieval helpers whose failure modes are silent.

`canonical_key` and `expand_abbrevs` exist because the guides write the same concept many
ways; when they stop collapsing a variant nothing crashes — the graph modes just quietly rank
six spellings of one concept instead of six concepts.
"""
import pytest

from retrieval._common import (canonical_key, expand_abbrevs, load_chunks,
                               map_chunk_ids_to_payloads, map_to_payloads, merge_dedup)
from retrieval.registry import MODES, VALID_MODES, get_search


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


# --- the bridge back to citable payloads -----------------------------------
# The graph modes select chunks inside their own index, but only OUR payloads carry the literal
# text and the metadata the sources panel renders. If this mapping silently drops a chunk, the
# evidence for a claim disappears without any error — the answer just gets thinner.
def test_the_corpus_is_loadable_and_has_ids():
    chunks = load_chunks()
    assert len(chunks) > 500
    assert all(c.get("chunk_id") and c.get("text") for c in chunks)


def test_selected_content_maps_back_to_our_payload():
    chunk = load_chunks()[0]
    mapped = map_to_payloads([{"content": chunk["text"]}])
    assert [p["chunk_id"] for p in mapped] == [chunk["chunk_id"]]


def test_whitespace_differences_do_not_break_the_mapping():
    chunk = load_chunks()[0]
    mangled = chunk["text"].replace("\n\n", "\n \n").replace(" ", "  ")
    assert map_to_payloads([{"content": mangled}])[0]["chunk_id"] == chunk["chunk_id"]


def test_unknown_content_is_dropped_rather_than_guessed():
    assert map_to_payloads([{"content": "texto que no está en el corpus"}]) == []


def test_the_same_chunk_selected_twice_is_returned_once():
    chunk = load_chunks()[0]
    assert len(map_to_payloads([{"content": chunk["text"]}, {"content": chunk["text"]}])) == 1


def test_ids_map_in_the_order_they_were_ranked():
    """PathRAG and HippoRAG select by id, and their ORDER is the ranking they computed."""
    ids = [c["chunk_id"] for c in load_chunks()[:3]]
    assert [p["chunk_id"] for p in map_chunk_ids_to_payloads(reversed(ids))] == ids[::-1]


def test_unknown_or_repeated_ids_are_skipped():
    real = load_chunks()[0]["chunk_id"]
    mapped = map_chunk_ids_to_payloads([real, "no-existe", real])
    assert [p["chunk_id"] for p in mapped] == [real]


# --- the mode catalogue ----------------------------------------------------
# Everything else (routing, the Studio dropdown, re_retrieve, the eval A/B) is derived from
# this, so an inconsistency here spreads everywhere at once.
def test_valid_modes_matches_the_catalogue():
    assert VALID_MODES == tuple(MODES)
    assert set(VALID_MODES) >= {"baseline", "iterative", "graph", "pathrag", "hipporag"}


def test_every_mode_describes_itself():
    for name, mode in MODES.items():
        assert mode.name == name and mode.description


def test_an_unknown_mode_fails_loudly_naming_the_valid_ones():
    """A typo in PIPELINE must stop the eval before it spends money, not silently run baseline."""
    with pytest.raises(SystemExit) as err:
        get_search("inventado")
    assert "baseline" in str(err.value)


def test_only_pathrag_offers_a_concept_map():
    """The concept map is an OPTIONAL part of the contract: the pipeline asks and adapts, which
    is what keeps the tail from knowing anything mode-specific."""
    assert MODES["pathrag"].search_with_concept_map() is not None
    assert MODES["baseline"].search_with_concept_map() is None
    assert MODES["graph"].search_with_concept_map() is None
