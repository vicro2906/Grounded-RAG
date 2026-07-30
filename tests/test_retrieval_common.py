"""Retrieval helpers whose failure modes are silent.

`canonical_key` and `expand_abbrevs` exist because the guides write the same concept many
ways; when they stop collapsing a variant nothing crashes — the graph modes just quietly rank
six spellings of one concept instead of six concepts.
"""
import threading

import pytest

from retrieval import _common
from retrieval._common import (canonical_key, expand_abbrevs, house_tail, load_chunks,
                               map_chunk_ids_to_payloads, map_to_payloads, merge_dedup)
from retrieval.registry import MODES, VALID_MODES, get_search
from rag import score_window


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


# --- the slice the cross-encoder is shown -----------------------------------
# The reranker used to score each chunk's PREFIX. With 89% of chunks longer than the window and
# guideline sections that open with context before recommending anything, it was often judging a
# preamble — measured: the section holding the VHB recommendation scored -0.900 on its prefix
# (rank 10 of 25, cut by top_k=8) and +0.880 in full. Nothing failed; the answer just went
# missing and the doctor was told the guides did not cover it.
def test_a_short_chunk_is_passed_through_whole():
    assert score_window("texto corto", "cualquier consulta", chars=512) == "texto corto"


def test_the_window_follows_the_query_instead_of_the_prefix():
    text = ("Introducción sobre epidemiología y prevalencia. " * 12
            + "Se recomienda iniciar TAR con TDF o TAF y FTC o 3TC. "
            + "Anexo sin relación. " * 12)
    window = score_window(text, "¿pauta con TDF o TAF y FTC?", chars=200)
    assert "TDF o TAF" in window and len(window) == 200


def test_accents_do_not_hide_the_relevant_slice():
    """The guides and the questions spell the same word both ways; folding is what keeps the
    window from being chosen by a spelling accident."""
    text = "Relleno. " * 40 + "Manejo de la coinfección por hepatitis B. " + "Relleno. " * 40
    assert "coinfección" in score_window(text, "coinfeccion hepatitis", chars=120)


def test_a_three_letter_abbreviation_still_steers_the_window():
    """The regression this cost: filtering query terms by LENGTH silently dropped TDF, TAF, FTC,
    3TC, DTG, VHB and TAR — the most discriminative tokens the guides have — and left the window
    to be picked by «paciente»."""
    text = "Relleno clinico general. " * 30 + "Pauta preferente con DTG. " + "Anexo. " * 30
    assert "DTG" in score_window(text, "¿pauta con DTG?", chars=120)


def test_function_words_do_not_get_a_vote():
    """They appear in every window, so counting them makes the score uniform and the choice
    collapses back to the prefix."""
    text = "Para el paciente que se encuentra con los datos. " * 8 + "Dosis de 300 mg al dia."
    assert "300 mg" in score_window(text, "¿que dosis para el paciente con 300 mg?", chars=100)


def test_a_query_with_nothing_to_match_falls_back_to_the_prefix():
    text = "".join(f"parrafo {i}. " for i in range(200))
    assert score_window(text, "¿y?", chars=100) == text[:100]


def test_the_budget_is_never_exceeded():
    """The budget IS the latency: 512 chars take ~0.55 s for 25 chunks, 2048 take ~9.4 s
    because the batch pads to the longest item."""
    text = "palabra " * 500
    assert len(score_window(text, "palabra concreta buscada", chars=300)) == 300


# --- house_tail: the tail four modes share ---------------------------------
# It decides the final context of pathrag, hipporag and graph at once, so a change here moves
# every one of their A/B numbers together — and until now nothing pinned its behaviour.
@pytest.fixture
def tail_env(monkeypatch):
    """Replace the tail's four boundaries (both local models, the hybrid and the reranker) so
    the merge/rerank logic can be exercised offline."""

    class Env:
        hybrid = [{"chunk_id": "h1", "text": "hybrid"}]
        rephrased: list = []               # queries the tail had to rephrase itself
        hybrid_queries: list = []
        rerank_calls: list = []

    env = Env()

    def fake_retrieve_hybrid(query, **kwargs):
        env.hybrid_queries.append(query)
        return list(env.hybrid)

    def fake_rerank(query, payloads, top_k=5):
        env.rerank_calls.append((query, list(payloads), top_k))
        return list(payloads)[:top_k]

    monkeypatch.setattr(_common, "_get_reranker", lambda: None)
    monkeypatch.setattr(_common, "_get_bm25", lambda: None)
    monkeypatch.setattr(_common, "rephrase", lambda q: env.rephrased.append(q) or f"{q} (r)")
    monkeypatch.setattr(_common, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(_common, "rerank", fake_rerank)
    return env


def test_the_modes_own_selection_is_merged_ahead_of_the_complement(tail_env):
    primary = [{"chunk_id": "g1", "text": "graph"}]
    out = house_tail("pregunta", primary, "reescrita", top_k=8)
    assert [p["chunk_id"] for p in out] == ["g1", "h1"]
    _, candidates, top_k = tail_env.rerank_calls[0]
    assert [p["chunk_id"] for p in candidates] == ["g1", "h1"] and top_k == 8


def test_the_reranker_scores_against_the_original_question(tail_env):
    """The rewritten query steers RETRIEVAL; the cross-encoder must judge relevance against
    what the doctor actually asked."""
    house_tail("pregunta", [], "reescrita")
    assert tail_env.hybrid_queries == ["reescrita"]
    assert tail_env.rerank_calls[0][0] == "pregunta"


def test_an_already_rewritten_query_is_not_rephrased_again(tail_env):
    house_tail("pregunta", [], "reescrita")
    assert tail_env.rephrased == []


def test_without_a_rewritten_query_the_tail_rephrases_once(tail_env):
    house_tail("pregunta", [])
    assert tail_env.rephrased == ["pregunta"] and tail_env.hybrid_queries == ["pregunta (r)"]


def test_nothing_retrieved_returns_empty_without_paying_for_a_rerank(tail_env):
    tail_env.hybrid = []
    assert house_tail("pregunta", [], "reescrita") == []
    assert tail_env.rerank_calls == []


def test_a_callable_selection_yields_the_same_result_as_a_list(tail_env):
    primary = [{"chunk_id": "g1", "text": "graph"}]
    eager = house_tail("pregunta", primary, "reescrita")
    lazy = house_tail("pregunta", lambda: primary, "reescrita")
    assert [p["chunk_id"] for p in eager] == [p["chunk_id"] for p in lazy]


def test_a_callable_selection_runs_alongside_the_hybrid(tail_env, monkeypatch):
    """The whole point of the callable form: the graph mode's traversal and the hybrid hit
    different resources and must overlap. Both sides meet at a barrier — if the tail ran them
    one after the other, the first would wait out the timeout and break it."""
    gate = threading.Barrier(2, timeout=5)

    def blocking_hybrid(query, **kwargs):
        gate.wait()
        return [{"chunk_id": "h1", "text": "hybrid"}]

    monkeypatch.setattr(_common, "retrieve_hybrid", blocking_hybrid)

    def selection():
        gate.wait()
        return [{"chunk_id": "g1", "text": "graph"}]

    out = house_tail("pregunta", selection, "reescrita")
    assert [p["chunk_id"] for p in out] == ["g1", "h1"]


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
