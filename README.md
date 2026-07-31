# Grounded-RAG — answers anchored in structured documents

A **RAG** architecture that answers questions over a corpus of **structured documents**
(clinical guidelines, regulations, technical documentation…) **always citing the literal
evidence**. Built on LangGraph, with a focus on **not hallucinating**, connecting concepts to
navigate the corpus, and a polished conversational UX.

The design is **domain-agnostic**: here it is demonstrated on the **HIV clinical guidelines
from GeSIDA** (in Spanish, for medical use), but the same architecture reproduces over any
corpus of structured documents of the same style (see [Reproducing with another
corpus](#reproducing-with-another-corpus)).

> **On authorship.** Personal project. The **design and the architecture decisions are
> mine**; I used **Claude Code (Anthropic)** as an **agentic programming** assistant to
> implement it. The repository also serves as an example of how far that agentic workflow
> goes when the ideas and judgment are supplied by the person. Detailed breakdown in
> [Authorship and workflow](#authorship-and-workflow).
>
> **Status: prototype** for demonstration (no real users yet).

---

## Design priorities (in order)

1. **Reduce hallucinations.** Every answer is anchored in **verifiable literal citations**
   from the guidelines; synthetic text (graph descriptions, chunk context, doctor-provided
   data) may *steer* retrieval or generation, but is **never cited**.
2. **Connect abstract concepts** to navigate the guidelines (knowledge graph).
3. **Conversational-assistant UX** (interactive clarification, citations, follow-up).

**Compliance (GDPR).** The goal is for every model/service to be private or local and, where
possible, in an EU region (the doctor's data may include health information, GDPR Art. 9).
Each LLM call is encapsulated so it can be migrated to Azure OpenAI without friction; the
reranker and BM25 run locally; Qdrant and LangSmith in an EU region.

---

## Architecture (LangGraph)

```
question ─▶ rephrase ─┬─ out of domain ─▶ out_of_domain ─▶ END
                      └─ in domain ─▶ [RETRIEVAL] ─▶ assess_context ⇄ clarify (interrupt)
                             ├─ baseline : hybrid (dense + BM25) + rerank
                             ├─ iterative: plan → sub-queries → reflect
                             ├─ graph    : entity-relation graph (LightRAG) + hybrid  ◀ default
                             ├─ pathrag  : flow-pruned relational paths over that graph
                             └─ hipporag : open KG + Personalized PageRank (HippoRAG 2)
                                              re_retrieve (1×) ─▶ generate ⇄ validate ─▶ evidence
```

- **Shared head and tail, interchangeable retrieval.** The five retrieval modes honour the
  same state contract; generation, validation and citation are identical. This makes the
  pipeline **retriever-agnostic** and cheap to experiment with: a mode is one module plus one
  line in `retrieval/registry.py`, from which routing, the Studio dropdown and the evaluation
  A/B are all derived.
- **Hybrid retrieval** dense (`text-embedding-3-large`) + sparse **BM25**, RRF fusion, and a
  local multilingual **reranker** (cross-encoder, GPU-optional).
- **Contextual Retrieval.** Each chunk is enriched with an LLM-generated context sentence
  (what it is, which section/population it applies to) that is **embedded and indexed** to
  improve dense and lexical matching, **keeping the literal text** for the citation.
- **Clarification gate (slot-filling).** Before answering, the system detects which patient
  datum would change the answer (pregnancy, renal function, HBV coinfection…) and **pauses**
  to ask the doctor; the datum re-triggers retrieval but is not cited.
- **Validation + evidence.** A judge checks relevance and *grounding*; the final answer is
  accompanied by a sources panel with literal citations (fuzzy match) and a clinical
  disclaimer.

Exhaustive detail of decisions and findings in [`CLAUDE.md`](CLAUDE.md).

---

## Stack and models

- **Orchestration:** LangGraph (+ LangGraph Studio for debugging) · traces in **LangSmith (EU)**.
- **Generation:** `gpt-4o`. **Rephrase / validation / contextualization:** `gpt-4o-mini`.
- **Embeddings:** `text-embedding-3-large` (3072d). **Reranker:** `jina-reranker-v2-base-multilingual`.
- **Vector store:** Qdrant Cloud (eu-west), hybrid collection with Contextual Retrieval.
- **Graph:** LightRAG (entities + relations, file storage, rebuildable).

---

## Corpus provenance

The `.md` files in `data/markdown/` were generated from the original GeSIDA PDFs (not
redistributed in this repository; see [`data/README.md`](data/README.md)) with **extraction
code generated with Claude Code and adapted to each PDF separately**: each guide has layout
peculiarities (tables, numbering, footnotes) that are not extrapolable to the others, so there
was no single generic script but one tuned to each document. The conversion relies on
PDF→Markdown **transcription** libraries (`pymupdf4llm`) that **copy the text, they do not
generate it** — the corpus is faithful to the original and does not introduce hallucinations
already at the source, consistent with the project's priority #1.

The **prompt** that guided the whole process is versioned in
[`data/prompt.txt`](data/prompt.txt): it enforces fidelity over completeness (forbids
paraphrasing, summarizing or reconstructing, and requires marking and omitting anything that
cannot be extracted with a guarantee), and implements the flow PDF inspection → adapted script
→ validation → iteration.

---

## Reproducing with another corpus

The architecture has nothing intrinsically "HIV" about it: it is a pattern of **grounded RAG
over structured documents**. To instantiate it on another corpus (another clinical specialty,
regulations, technical manuals…) you change the **domain layer**, not the engine:

**What gets replaced (domain-specific):**
- The **corpus**: the documents in `data/` and their transcription (the `data/prompt.txt`
  prompt is reusable for any structured PDF) + re-running the chunking in `ingestion/`.
- The **abbreviation glossary** (`abbreviations.py`) with the new domain's.
- The **domain guardrail** (the `in_domain` classification of the *rephrase* node) that
  defines which questions are "on topic".
- The **prompts** that name the domain (`SYS_PROMPT`, contextualization, *assess*) and the
  Qdrant collection names.

**What is reused as-is (the engine):** the LangGraph graph, the hybrid + graph + reranker
retrieval, the Contextual Retrieval, the clarification gate, and the validation + literal
citation. That is, **all the engineering value is portable**; what is specific is the domain
configuration.

> Note: today those domain pieces live in the code (not parameterized in a config file).
> Extracting them into configuration is natural future work, but the conceptual
> engine/domain separation already exists.

---

## How to run

Requirements: Python 3.12+, dependencies managed with [`uv`](https://docs.astral.sh/uv/), and
a `.env` (see [`.env.example`](.env.example)) with `OPENAI_API_KEY`, `QDRANT_URL`,
`QDRANT_API_KEY` and, optionally, the LangSmith keys.

```bash
# interactive CLI
python main.py

# LangGraph Studio (visual step-by-step debugging of the graph)
langgraph dev --no-reload

# Force a retrieval mode
python main.py iterative        # baseline | iterative | graph
```

Corpus indexing (once):

```bash
python -m ingestion.contextualize                                   # enriches the chunks
python -m ingestion.upload_to_qdrant_hybrid data/chunks/chunks_contextual.jsonl --collection guias_vih_hibrida_ctx
python -m retrieval.graph                                           # builds the graph
```

---

## Evaluation

A single **`EVAL_SET`** (151 questions) labelled by type
(**simple / single_hop / multihop / adversarial**) measures quality **by question type** with
**RAGAS** (faithfulness + context precision + context recall). The A/B between retrievers only
changes retrieval (generation is shared), and the collection is selectable via env to compare
index versions:

```bash
PIPELINE=graph python evaluation.py
QDRANT_COLLECTION=guias_vih_hibrida PIPELINE=graph python evaluation.py   # A/B without context
```

> The set's references were drafted by the model from the guidelines and **are pending
> clinical review**; the numbers are indicative.

---

## Repository structure

| Path | What it is |
|------|--------|
| `main.py` | Entry point: compiles the graphs, wires runtime concerns and the CLI. |
| `pipeline/` | The LangGraph app: state, nodes, graph assembly and structured generation. |
| `rag.py` | Retrieval/generation primitives: hybrid, rerank, refine, validate, prompts. |
| `evidence.py` | Answer formatting and sources panel with literal citations. |
| `evaluation.py` | RAGAS evaluation (`EVAL_SET`, per tier). |
| `retrieval/` | The three interchangeable retrieval architectures, one module per mode: `baseline.py`, `iterative.py` (Track A) and `graph.py` (Track B, LightRAG). |
| `ingestion/` | Corpus→index scripts: chunking, contextualization and upload to Qdrant. |
| `data/` | Corpus and derived data: `markdown/` (the 7 GeSIDA guides, real source), `textos/`, `chunks/` (the chunked corpus, `chunks.jsonl`) and `lightrag_store/` (generated graph index, not versioned). The original PDFs (`pdfs/`) are not versioned (see `data/README.md`). |
| `results/` | RAGAS evaluation CSVs (one file per pipeline). |
| `docs/` | Design documents and architecture diagrams. |
| `CLAUDE.md` | Living document of context, decisions and findings. |

---

## Authorship and workflow

This project was built with an **agentic programming** workflow: I supply the **ideas,
judgment and decisions**, and **Claude Code (Anthropic)** carries out the implementation under
that direction. The important distinction is not "hand-written vs. generated code", but **who
decides what gets done and why**: that is mine from start to finish.

**What I did (design and judgment):**
- **Conception and goal** of the system: an evidence-grounded medical RAG for the GeSIDA
  guidelines, with the explicit priority #1 of **not hallucinating**.
- **Architecture and design decisions:** the LangGraph pipeline with a shared head/tail and
  interchangeable retrieval; the clarification gate (*slot-filling*) with clinical knowledge
  as the primary path; the enriched re-retrieval; the decision to **discard the graph's
  synthetic text for generation** and cite only literal text. I resolved every trade-off
  (cost, latency, quality), and the "why" of each is documented in [`CLAUDE.md`](CLAUDE.md).
- **Stack choice** (LangGraph, Qdrant EU, LightRAG, local reranker) and the **constraints**
  that condition it (GDPR, EU region, encapsulating the LLM calls to migrate to Azure).
- **Evaluation methodology:** defining the question *tiers*, which RAGAS metrics are reliable
  and which are artifacts, and reading the A/Bs to decide (e.g. graph as the default).
- **The corpus and its handling:** the requirement of absolute fidelity in the PDF→Markdown
  transcription and the prompt that governs it ([`data/prompt.txt`](data/prompt.txt)).
- **Continuous direction:** the phased roadmap, the review of every change, and the course
  corrections when the implementation did not fit the judgment.

**What I delegated to Claude Code (implementation under direction):**
- **Writing the code** of the nodes, retrieval primitives, integrations (Qdrant, LangSmith,
  LightRAG) and utilities, from my design decisions.
- **The PDF→Markdown extraction scripts**, adapted to each guide's peculiarities.
- **Debugging and iteration:** environment setup (including the reranker's GPU acceleration),
  error resolution, refactors and repetitive work.
- **Drafting documentation artifacts** (including much of this README and of
  [`CLAUDE.md`](CLAUDE.md)) following my outline and corrections.

In short: **the engineering of decisions is mine; the mechanical execution is assisted.** The
value I want to show here is twofold — the system design and the ability to direct an agentic
development workflow with judgment.

---

## License

The **code** is published under the **MIT** license (see [`LICENSE`](LICENSE)): use, copy or
modify it freely while keeping the copyright notice. The license **does NOT cover the contents
of `data/`**: the GeSIDA guidelines are the work and property of their authors and are included
solely for the prototype's demonstration.

## Disclaimer

Clinical decision-support tool in prototype stage. It **does not replace medical judgment**
nor does it constitute clinical advice.
