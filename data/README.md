# Data / corpus

Source material of the chatbot: the 7 HIV clinical guidelines from **GeSIDA**.

- **`pdfs/`** — the original PDFs. **Not versioned in this repository** (they are in
  `.gitignore`): they are the work of GeSIDA, so they are kept locally only and not
  redistributed. Downloaded from GeSIDA if needed.
- **`markdown/`** — the PDFs converted to Markdown; this is the **actual corpus** that the
  chunking (`../ingestion/`) runs on.
- **`textos/`** — plain-text extractions of the originals (provenance / backup).
- **`chunks/`** — the chunked corpus (`chunks.jsonl`, `chunks_contextual.jsonl`), produced by
  `../ingestion/` and consumed by retrieval and the LightRAG index build.
- **`lightrag_store/`** — the generated LightRAG graph index (not versioned; rebuilt with
  `python -m retrieval.graph`).

## How the Markdown was generated

Each `.md` was obtained from the corresponding PDF with **extraction code generated with Claude
Code and adapted to each PDF separately**. There is no single generic script: each guide has
layout peculiarities (tables, section numbering, footnotes, columns) that are not
extrapolable to the others, so the extraction was tuned document by document.

The conversion relies on the PDF→Markdown **transcription** library `pymupdf4llm`, which
**copies the literal text, it does not generate it** (no vision models "interpreting" the
structure). This is a deliberate decision: the project's #1 goal is not to hallucinate, and
that starts with the corpus being **faithful to the original**, without a model "rewriting"
the guidelines' content in the ingestion step.

The **prompt** used to drive this conversion is in [`prompt.txt`](prompt.txt). It sets
fidelity as an absolute rule (no paraphrasing/summarizing/reconstructing; omit, with a
`> _[... omitido — consultar PDF original]_` marker, anything that cannot be extracted
reliably) and defines the process: inspect the PDF to find its peculiarities → build a
`pymupdf4llm` script adapted to them → validate headings and samples → iterate until it is
faithful.

> The rights and authorship of the content belong to GeSIDA; here it is only transcribed for
> use by the prototype.
