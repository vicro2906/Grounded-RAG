"""The chunker's decisions, on synthetic Markdown.

The quality gates in `test_corpus_quality.py` measure the corpus that ships; these tests pin the
RULES that produce it, so a regression is attributable to a rule rather than to a number moving.
Each one corresponds to a defect measured in the shipped v1 corpus — the size budget applying to
the wrong string, tables emitted whole, recommendations stranded alone, an overlap that never
happened, and ids that changed whenever anything above them was edited.
"""
import hashlib

import pytest

from ingestion.chunk_guidelines import (MAX_BREADCRUMB_TOKENS, MAX_TOKENS, MIN_TOKENS, Section,
                                        _tail_sentences, breadcrumb_prefix, build_chunks,
                                        classify, count_tokens, normalize, parse_sections,
                                        split_table)


def _sections(md: str):
    return parse_sections(md)


def _units(md: str):
    return normalize(_sections(md))


# ---------------------------------------------------------------------------
# The size budget applies to what is EMBEDDED
# ---------------------------------------------------------------------------
def test_budget_reserves_room_for_the_breadcrumb():
    """A chunk's cost is breadcrumb + body, because that is the string handed to the embedder
    and to the reranker. Sizing the body alone is how chunks landed over budget."""
    shallow = Section(2, "1. TEMA", "cuerpo", ["1. TEMA"], "1")
    deep = Section(5, "1.1.1.1 SUB", "cuerpo",
                   ["1. UN TÍTULO DE CAPÍTULO CONSIDERABLEMENTE LARGO", "1.1 SUBSECCIÓN",
                    "1.1.1 SUB-SUBSECCIÓN", "1.1.1.1 SUB"], "1.1.1.1")
    assert deep.overhead > shallow.overhead
    assert deep.budget < shallow.budget
    assert shallow.budget + shallow.overhead <= MAX_TOKENS


def test_breadcrumb_can_never_dominate_a_chunk():
    """The guard against a malformed heading upstream: a paragraph promoted to a heading gave
    breadcrumbs of 1 228 tokens, more than the whole budget, inherited by every chunk beneath."""
    huge = ["Ancestro " + "palabra " * 400, "Sección concreta"]
    prefix = breadcrumb_prefix(huge)
    assert count_tokens(prefix) <= MAX_BREADCRUMB_TOKENS
    # Trimmed from the LEFT: the deepest heading is the specific one and must survive.
    assert "Sección concreta" in prefix


def test_no_unit_exceeds_the_budget():
    md = "## 1. TEMA\n\n" + "\n\n".join(f"Párrafo clínico número {i}. " * 40 for i in range(12))
    for unit in _units(md):
        assert count_tokens(unit["body"]) <= MAX_TOKENS


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def test_a_split_table_repeats_its_header():
    """A row without its column names is unreadable and uncitable: «| INI | BIC/FTC/TAF |» says
    nothing without «| 3er Fármaco | Pauta |» above it."""
    rows = "\n".join(f"| fármaco{i} | dosis{i} | comentario largo número {i} |" for i in range(200))
    table = "| 3er Fármaco | Pauta | Comentarios |\n|---|---|---|\n" + rows
    pieces = split_table(table, budget=300)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith("| 3er Fármaco | Pauta | Comentarios |")
        assert "|---|---|---|" in piece
        assert count_tokens(piece) <= 300 or piece.count("\n") <= 2


def test_a_table_row_is_never_split_across_chunks():
    rows = [f"| fármaco{i} | una dosis bastante larga para forzar el corte {i} |" for i in range(60)]
    pieces = split_table("| A | B |\n|---|---|\n" + "\n".join(rows), budget=120)
    emitted = [l for p in pieces for l in p.splitlines() if l not in ("| A | B |", "|---|---|")]
    assert emitted == rows


def test_a_table_is_no_longer_exempt_from_the_budget():
    """Tables used to be emitted whole whatever their size, which is how a single 7 506-token
    chunk holding the interaction matrix reached the index."""
    md = ("## 9. TABLAS\n\n### TABLA 9. Interacciones\n\n| Fármaco | DTG | BIC |\n|---|---|---|\n"
          + "\n".join(f"| fármaco número {i} | X NR | X hFCO1 |" for i in range(400)))
    units = _units(md)
    assert len(units) > 1
    assert all(count_tokens(u["body"]) <= MAX_TOKENS for u in units)


def test_a_wide_matrix_splits_by_record():
    md = ("## 9. TABLAS\n\n### TABLA 9. Interacciones\n\n"
          + "\n".join(f"- fármaco{i}: con DTG «X NR»; con BIC «X hFCO1»; con EFV «X NR»."
                      for i in range(400)))
    units = _units(md)
    assert len(units) > 1
    assert all(count_tokens(u["body"]) <= MAX_TOKENS for u in units)
    # Every record survives intact somewhere.
    joined = "\n".join(u["body"] for u in units)
    assert joined.count("- fármaco399:") == 1


# ---------------------------------------------------------------------------
# content_type describes the payload, not the intention
# ---------------------------------------------------------------------------
def test_a_caption_without_a_table_is_not_typed_table():
    """14 chunks were typed `table` while holding only a caption and footnotes. content_type is
    a Qdrant filter: it has to describe what is there."""
    empty = Section(4, "TABLA 3. Combinaciones de TAR de inicio",
                    "> _[Figura omitida — p. 42 — consultar PDF original]_\n\nNotas: †", [], None)
    assert classify(empty) == "text"
    real = Section(4, "TABLA 3. Combinaciones", "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
                   [], None)
    assert classify(real) == "table"


def test_a_piece_of_a_split_table_is_typed_by_its_own_content():
    """Cutting a table yields pieces that are rows — and one that is caption plus footnotes,
    which is prose."""
    md = ("## 9. TABLAS\n\n### TABLA 1. Algo\n\n| A | B |\n|---|---|\n"
          + "\n".join(f"| fila número {i} | valor {i} |" for i in range(400))
          + "\n\n" + "Nota al pie explicando los símbolos. " * 200)
    types = {u["content_type"] for u in _units(md)}
    assert types == {"table", "text"}


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def test_a_small_recommendation_merges_with_its_section():
    """Keeping `recommendations` out of the merge left the corpus's most valuable content in
    chunks of 21 to 63 tokens: a graded recommendation with no rationale around it."""
    md = ("## 2. TEMA\n\n### 2.1. Subsección\n\nUn párrafo breve de contexto clínico.\n\n"
          "#### Recomendaciones\n\n- Se recomienda hacer algo concreto **(A-III)**.\n")
    units = _units(md)
    assert len(units) == 1
    assert "Recomendaciones" in units[0]["body"]
    assert "párrafo breve" in units[0]["body"]


def test_a_merged_subsection_keeps_its_real_heading_depth():
    """The merged heading was hardcoded to `##`, so a sub-subsection announced itself as a
    chapter and the Markdown inside the chunk contradicted its own breadcrumb."""
    md = ("## 2. TEMA\n\n### 2.1. Uno\n\nTexto corto.\n\n### 2.2. Dos\n\nMás texto corto.\n")
    body = _units(md)[0]["body"]
    assert "### 2.2. Dos" in body and "\n## 2.2. Dos" not in body


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------
def test_the_overlap_is_not_empty_when_a_section_is_split():
    """Paragraph-level overlap almost never fired: a clinical paragraph is usually longer than
    the budget on its own, so only 12 of 53 split points in the shipped corpus carried any."""
    paragraphs = [f"Frase inicial del párrafo {i}. Frase intermedia. Frase final del párrafo {i}."
                  for i in range(40)]
    md = "## 1. TEMA\n\n" + "\n\n".join(paragraphs)
    units = _units(md)
    assert len(units) > 1
    # The overlap carries the tail of the LAST PARAGRAPH of the previous piece.
    tail = _tail_sentences(units[0]["body"].split("\n\n")[-1])
    assert tail and units[1]["body"].startswith(tail)


def test_the_overlap_carries_whole_sentences():
    text = "Primera frase completa. Segunda frase completa. Tercera frase completa."
    assert _tail_sentences(text, budget=8).endswith("Tercera frase completa.")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def test_editing_a_section_does_not_renumber_the_chunks_after_it(tmp_path, monkeypatch):
    """The id used to be sha1(file, POSITION, heading), and the uploaders derive each Qdrant
    point's UUID from it: inserting a paragraph high up changed the identity of everything
    below, leaving the previous points in the collection as orphans nobody would overwrite."""
    import corpus
    doc = corpus.documents()[0]
    monkeypatch.setattr(corpus, "document_for_markdown", lambda name: doc)

    # Both sections are well over MIN_TOKENS so neither is merged into the other: the test is
    # about identity under edits, not about merging.
    tail = "\n\n### 1.2. Segunda\n\n" + "Contenido estable de la segunda sección. " * 60
    first = "Texto original de la primera sección. " * 60
    before = tmp_path / "a.md"
    before.write_text(f"## 1. TEMA\n\n### 1.1. Primera\n\n{first}" + tail, encoding="utf-8")
    after = tmp_path / "b.md"
    after.write_text(f"## 1. TEMA\n\n### 1.1. Primera\n\n{first} Una frase añadida." + tail,
                     encoding="utf-8")

    ids_before = {c.heading: c.chunk_id for c in build_chunks(before)}
    ids_after = {c.heading: c.chunk_id for c in build_chunks(after)}
    assert ids_before["1.2. Segunda"] == ids_after["1.2. Segunda"]
    assert ids_before["1.1. Primera"] != ids_after["1.1. Primera"]


def test_the_citable_text_carries_no_breadcrumb(tmp_path, monkeypatch):
    """The split that fixes 178 chunks sharing their opening: `text` is what gets quoted,
    `text_for_retrieval` is what gets embedded."""
    import corpus
    doc = corpus.documents()[0]
    monkeypatch.setattr(corpus, "document_for_markdown", lambda name: doc)
    path = tmp_path / "a.md"
    path.write_text("## 1. CAPÍTULO\n\n### 1.1. Sección\n\nEl cuerpo clínico de la sección.",
                    encoding="utf-8")
    chunk = build_chunks(path)[0]
    assert chunk.text == "El cuerpo clínico de la sección."
    assert chunk.text_for_retrieval.startswith("1. CAPÍTULO > 1.1. Sección")
    assert chunk.text_for_retrieval.endswith(chunk.text)
    assert chunk.n_tokens == count_tokens(chunk.text_for_retrieval)


def test_an_unlisted_document_is_refused(tmp_path):
    """It used to enter the corpus silently tagged `topic = "vih_general"`."""
    path = tmp_path / "guia_desconocida.md"
    path.write_text("## 1. TEMA\n\nTexto.", encoding="utf-8")
    with pytest.raises(SystemExit, match="corpus.toml"):
        build_chunks(path)
