# CLAUDE.md — HIV medical chatbot (RAG over GeSIDA guidelines)

Project context to resume it in any new conversation. Keep this file up to date when
important decisions, status or findings change.

> **Language note.** This document (and the codebase comments/docs) are in English so
> non-Spanish-speaking colleagues can read them. What stays in Spanish is only: the
> doctor-facing UI, the LLM prompts, the guideline/chunk content, and the values of
> `abbreviations.py` (see the Conventions section). Keep speaking to the user (Victor) in
> Spanish in chat.

## What it is

RAG chatbot for **doctors** that answers questions about the **HIV clinical guidelines from
GeSIDA** (7 PDFs, in Spanish). Current nature: **prototype for demonstration** (no real users
yet). Priorities, in order: (1) **reduce hallucinations**, (2) **connect abstract concepts**
to navigate the guidelines, (3) **Claude-style UX**.

Compliance: under GDPR, every model/service must be private or local and, where possible, in
an EU region. The doctor's data may include health information (GDPR Art. 9).

## Current architecture (LangGraph)

Entry point: `main.py` (compiles the graph `app`). The graph itself lives in the `pipeline/`
package (`config`, `state`, `nodes`, `nodes_expanded`, `builder`, `generation`); `main.py` only
wires runtime concerns (UTF-8, LangSmith, warm-up) and the CLI. Graph nodes:

```
question ─▶ rephrase ─┬─ out of domain ─▶ out_of_domain ─▶ END
                      └─ in domain ─▶ [RETRIEVAL_MODE] ─▶ assess_context ⇄ clarify (interrupt)
                             ├─ baseline : retrieve (hybrid) ─▶ rerank (20→5)        │ (on exit)
                             ├─ iterative: iterative_retrieve (plan→hop→reflect, →8)   ▼
                             └─ graph    : graph_retrieve (LightRAG + own hybrid, →8) ◀ DEFAULT
                                              re_retrieve (1×) ─▶ generate ⇄ validate ─▶ evidence ─▶ output
                                                              (not valid+exhausted ─▶ fallback)
```

The head (rephrase + domain guardrail) and the tail (assess_context→generate→validate→evidence)
are IDENTICAL across the three modes; only the retrieval node changes, chosen by
`RETRIEVAL_MODE` in `pipeline/config.py` (default `"graph"` after the Phase-4 A/B).

- **rephrase** (`rag.refine`, gpt-4o-mini): a single call that (a) classifies whether the
  question belongs to the HIV domain (`in_domain`); (b) rewrites the query WITHOUT adding
  info, normalizing terms in BOTH forms «full name (ABBR)» with `abbreviations.py`; and (c)
  **screens** the patient data already present (`known_facts`→`clinical_facts`) and the
  clinical modifiers the question might need (`candidate_modifiers`) — the cheap half of the
  clarification step. If out of domain → `out_of_domain` (direct message, short-circuits the
  pipeline). Generation uses the ORIGINAL question, not the rewritten one.
- **assess_context + clarify** (`rag.assess`, **gpt-4o** `ASSESS_MODEL`; nodes in `pipeline/nodes.py`):
  **interactive clarification gate** (Phase 5, slot-filling) between retrieval and `generate`.
  It reasons over structured fields (order = CoT) via TWO paths, with **CLINICAL KNOWLEDGE as
  the PRIMARY one**: **(1) `clinically_relevant`** (primary): ALL modifiers that clinically
  change the answer to this kind of question, INDEPENDENT of the context (exhaustive within
  what is relevant) and **(2) `branches_on`** (complement): extra dimensions the retrieved
  context conditions on. **Why knowledge first (decided with the user):** the retriever's
  failure (silent recall miss → generic answer with a confident air) is both more likely AND
  more harmful than the question-model erring (visible, corrected by the doctor, bounded by
  `asked_questions` and the cap, and NEVER corrupts the answer, which is re-retrieved and
  validated). Only the ANSWER is anchored in evidence, not the QUESTIONS. It subtracts
  `already_covered` (what `clinical_facts` already pins in any unit + what was already asked)
  and emits `questions` ordered by **clinical impact** (not by path), one per pending
  dimension. It reasoned inconsistently with gpt-4o-mini (skipped the CoT) → gpt-4o is used
  (`ASSESS_MODEL`); it costs more (runs up to 3× per question) but a better model extracts
  more relevant detail. The reasoning is exposed in the state (`assessment`) for the trace. It
  is safe because the datum is NOT cited: it only steers the branch AND re-triggers retrieval
  (`re_retrieve`), so the conditional evidence is retrieved before generating and `validate`
  still requires grounding. If there are questions → `clarify` **pauses** the graph with
  `interrupt()` and asks the doctor; on resume, `clarify` folds the answer into
  `clinical_facts` (manual merge) and adds 1 to `clarify_rounds`. Two caps: `CLARIFY_MAX_ROUNDS`
  (=3, number of pauses) and `CLARIFY_QUESTIONS_PER_ROUND` (=1, questions per pause) → by
  default it asks ONE thing at a time, at most 3 times. `assess` orders the pending ones by
  **clinical impact** and the first `max_questions` are taken.
  **`clinical_facts`/`clarify_rounds` carry NO reducer**: they are RESET per question in
  `node_rephrase` (Studio persists the thread state; with accumulating reducers the previous
  patient's data and the spent round budget leaked into the next question and `assess_context`
  was skipped). Within ONE question the merge/increment is done by `clarify` by hand (reading
  the state), enough because they are written sequentially.
- **re_retrieve** (node in `pipeline/nodes.py`): runs **ONCE, on EXIT from the clarification loop**
  (right before `generate`), not on every round. **Re-retrieves with ALL the `clinical_facts`
  injected into the query** (dispatches on `retrieval_mode`, reuses the collapsed
  baseline/iterative/graph functions) and OVERWRITES `contexts` → so the doctor's datum PULLS
  the conditional passages (the HBV branch, the first-trimester one…) so `generate` CITES
  them, not just steers generation. No-op if there is no data (e.g. a question that does not
  clarify) → leaves the initial context intact. **Total: 1 initial retrieve + 1 final
  re_retrieve** (previously it was 1 + N per round). The `assess_context ⇄ clarify` loop runs
  entirely on the **initial context**: since `assess` is knowledge-primary, it does not need
  re-retrieval between rounds (the intermediate re_retrieves were redundant). Tested: with
  "HBV coinfection" the HBV-specific chunks enter the top-5. The `clinical_facts` ALSO enter
  `generate` as a NON-citable **"DATOS APORTADOS POR EL MÉDICO"** block (they select the
  guide's branch; the literal citations still come from the chunks). Requires a checkpointer:
  provided by `langgraph dev`/Studio (no own one is compiled in the graphs).
- **Retrieval — 3 selectable modes (`RETRIEVAL_MODE`), all ending in the same generate:**
  - **baseline** = `node_retrieve` (`rag.retrieve_hybrid`: dense text-embedding-3-large 3072d +
    sparse BM25 `Qdrant/bm25`, RRF fusion, 20 candidates) → `node_rerank` (`rag.rerank`,
    local cross-encoder, 20→5).
  - **iterative** (Track A) = `node_iterative_retrieve` → `agentic.iterative.iterative_search`
    (plan→hop→reflect, top 8). Uses the ORIGINAL question (its own plan); ignores the rewritten one.
  - **graph** (Track B, DEFAULT) = `node_graph_retrieve` → `graph.lightrag_track.graph_search`
    (entity+relation graph `mode="hybrid"` + `retrieve_hybrid` dense+BM25 complement →
    dedup fusion → rerank → top 8).
- **rerank** (`rag.rerank`): local cross-encoder `jinaai/jina-reranker-v2-base-multilingual`
  (fastembed/ONNX, multilingual, GDPR-ok). Used by the three modes to refine to the final top.
- **generate** (`pipeline.generation.structured_llm`, gpt-4o): structured output with
  `ChatOpenAI.with_structured_output` (Pydantic `ClinicalAnswer`, strict json_schema).
  Returns dict: sufficient_information, answer, sources_used[{ref,quote}], follow_up_questions.
- **validate** (`rag.validate`, gpt-4o-mini): relevance + semantic grounding judge.
  Loop with `generate` (injects feedback on retry), `MAX_ITER=2`. valid → evidence;
  not valid and exhausted → `fallback`; technical error of the judge → `fallback` (no "failing open").
- **evidence** (`evidence.format_answer`): formats the answer + sources panel with
  literal citations (fuzzy match) + follow-up questions + clinical disclaimer. Text without ANSI.

`rag.search()` = rephrase → hybrid → rerank (used by the evaluation).

### Graph retrieval (LightRAG): what is used and what is NOT

Indexing (once, `graph._build_index`): for each chunk, gpt-4o-mini extracts **entities** and
**relations** with an **LLM-generated description**; they are stored in `lightrag_store/`:
`kv_store_full_entities/relations.json` (description + source chunks `source_id`),
`vdb_entities/relationships/chunks.json` (embeddings for matching), `kv_store_text_chunks.json`
(literal text) and `graph_chunk_entity_relation.graphml` (topology).

Query (`graph_search`, `hybrid` mode): (1) LightRAG extracts high/low-level keywords from the
question; (2) low-level → vdb_entities → entities; high-level → vdb_relationships →
relations; (3) it walks the graph (neighbours) and, via `source_id`, gathers the **source
chunks** (up to `chunk_top_k=20`), mapped to `chunks.jsonl` with `_map_to_payloads` (exact
match + prefix fallback; 20/20 in a test). (4) **Complement (replaces LightRAG's internal
dense search):** the chunks from OUR `retrieve_hybrid` (dense + BM25 RRF) over the rewritten
query are added — BM25 helps with abbreviations/doses. (5) It is **merged** (dedup by
`chunk_id`, graph first) and the **reranker** refines to top 8.

**Design decision (key):** the entity/relation descriptions DO influence the **selection**
(their embedding is what matches the question), but are **discarded for generation**: only the
**literal chunks** are passed to gpt-4o. Reason = priority #1 (do not hallucinate): the
descriptions are synthetic LLM text, not citable and with possible extraction errors; our
`evidence.py` requires a verifiable literal citation. **Open idea (A/B-able):** pass the
*relation* descriptions as a "concept map, NOT citable" block to help multi-hop reasoning,
keeping strict grounding + validate.

## Key files

- `main.py` — entry point: compiles the graphs, wires runtime concerns (UTF-8, LangSmith,
  warm-up) and the CLI. Thin; the graph lives in `pipeline/`.
- **`pipeline/`** — the LangGraph app, assembled from `rag.py`'s primitives: `config.py`
  (constants, Studio context schema, `MSG_*`), `state.py` (state schemas + reducers),
  `nodes.py` (combined-pipeline nodes + routing), `nodes_expanded.py` (one-node-per-step
  retrieval for the dedicated Studio graphs), `builder.py` (head/tail assembly +
  `build_graph`/`build_combined_graph`), `generation.py` (structured `ClinicalAnswer` LLM).
- `rag.py` — retrieval/generation primitives: clients, embeddings, retrieve/retrieve_hybrid,
  rerank, refine, search, validate, assess, generate_answer (raw version), SYS_PROMPT,
  build_user_prompt, model constants.
- `evidence.py` — answer and sources formatting.
- `evaluation.py` — RAGAS evaluation. **A SINGLE set `EVAL_SET` (151 questions)** with a
  `tier` field per question (**simple / single_hop / multihop / adversarial**) → measures
  performance BY QUESTION TYPE. It is built by folding the old pools (golden + multihop, now
  internal components `_PREV_SINGLE`/`_PREV_MULTI`) with a new tiered block (`_TIERED_NEW`);
  there is NO dataset selector any more. A/B with `PIPELINE` (baseline/iterative/graph, via
  env): only retrieval changes, generation is shared. **It ALWAYS runs the full RAGAS suite**
  (faithfulness+precision+recall; answer_relevancy omitted on purpose as a Spanish artifact);
  the lean modes `RETRIEVAL_ONLY`/`RECALL_ONLY` were removed. `RunConfig(timeout=600,
  max_workers=8)` anti-timeout. Prints global means **and per tier**, measures latency and
  dumps `results/ragas_results_<PIPELINE>.csv`. The references were drafted by the model
  from the guidelines → **pending clinical review**.
- `abbreviations.py` — ABBREVIATION→name dictionary from the guides (values in Spanish).
- **`agentic/`** — Track A (Phase 4). `iterative.py`: `iterative_search` (plan/hop/reflect).
  Imports the shared primitives from `rag.py` (parent folder).
- **`graph/`** — Track B (Phase 4). `lightrag_track.py`: graph build + `graph_search`
  (imports `rerank` from `rag.py`, corpus from `chunks/`, store in `lightrag_store/`).
- `chunks/` — `chunk_guidelines.py` (structural chunking), `upload_to_qdrant.py` (dense),
  `upload_to_qdrant_hybrid.py` (dense+BM25), `chunks.jsonl` (517 chunks).
- `data/markdown/` — the 7 guides in Markdown (corpus source). `data/pdfs/`, `data/textos/` — originals.
  The `.md` files were transcribed from the PDFs with **code generated by Claude Code adapted to each PDF**
  (non-extrapolable peculiarities), using `pymupdf4llm` (transcribes, **does not invent**). The prompt
  that guided the conversion (absolute fidelity, inspection→script→validation→iteration) is in
  `data/prompt.txt`. See `data/README.md`.
- `docs/` — design documents and architecture diagrams. `results/` — RAGAS evaluation CSVs.
- `langgraph.json` — LangGraph Studio config (exposes `main.py:app`).

## Models and services

- Generation: **gpt-4o** (`GENERATION_MODEL`). Rephrase/validation: **gpt-4o-mini**
  (`REPHRASE_MODEL` / `VALIDATION_MODEL`).
- Embeddings: `text-embedding-3-large` (3072d). Reranker: jina-reranker-v2-base-multilingual.
- Qdrant Cloud (**eu-west** region). Collections: `guias_vih` (dense only, backup),
  `guias_vih_hibrida` (dense + sparse BM25, no context) and **`guias_vih_hibrida_ctx`**
  (dense + BM25 with Contextual Retrieval, **the default active one**). The active one is set
  by `COLLECTION_HYBRID` in `rag.py` (default `guias_vih_hibrida_ctx`), overridable with
  `QDRANT_COLLECTION` to A/B against the non-contextual one.
- LangSmith in the **EU** region: `.env` with `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`,
  project **`chatbot_vih`** (underscore). Tracing auto-enables if `LANGSMITH_API_KEY` is set.

## How to run

- App (interactive CLI): `.venv\Scripts\python.exe main.py`
- LangGraph Studio: `.venv\Scripts\langgraph.exe dev` → opens Studio (EU). See steps in the
  **Trace View** tab (not Turn View).
- **Choosing the retrieval strategy (Phase 4):** `main.py` compiles FOUR graphs (via
  `pipeline/builder.py`) and `langgraph.json` registers them so Studio shows a **graph selector**:
  - **Dedicated graphs** `app_baseline` / `app_iterative` / `app_graph` (build_graph(mode)):
    each one contains ONLY its architecture, with retrieval **EXPANDED into its real nodes**
    (teaching view) instead of hiding it behind a single `retrieve`:
      - baseline: `retrieve → rerank`
      - iterative: `iter_generate_subquestions → (iter_single | FAN-OUT Send×N
        ▶ iter_retrieve_one → iter_reflect ⇄ Send×1) → iter_rerank`. iter_generate_subquestions
        ONLY generates the sub-questions; each sub-question is retrieved in its OWN run of
        `iter_retrieve_one` (`Send`/fan-out pattern → one visible call per sub-question in the
        trace); they converge on iter_reflect, which decides whether evidence is missing and,
        if so, does another `Send` (another round) or exits to iter_rerank. `pool`/`hops`
        accumulate via reducers (`_merge_pool` dedup, `_add_int`). Single-hop → iter_single
        (one-shot baseline).
      - graph: TWO PARALLEL branches from rephrase converging on merge:
        `rephrase ─┬ graph_keywords (LLM: high-level keywords→relations, low-level→entities)
        → graph_select (cosine entities/relations + traversal + chunks) ─┬ graph_merge
        → graph_rerank` and `└ graph_hybrid (dense+BM25) ─┘`. graph_select passes the
        already-extracted keywords to `aquery_data` (LightRAG skips its internal LLM) and
        exposes entities/relations/chunks in the state. `graph_merge` uses **`defer=True`** to
        run ONCE after both branches (of different length; without defer the fan-in would fire
        twice).
    Each node reuses the SAME primitives as `iterative_search`/`graph_search` (no behaviour
    change) and the intermediate states are visible in Studio (`IterativeState`/`GraphState`
    extend `RAGState`). Selection: Studio's graph dropdown.
  - **Combined graph** `app` (build_combined_graph): the three routes in one graph; it exposes
    a `context_schema` (`ConfigSchema`) with the **`retrieval_mode`** field as a **dropdown**
    in the run config panel (live choice). Also used by the CLI.
  - **Shared head/tail** are factored into `_add_common` (in `pipeline/builder.py`); the
    retrieval section into `_add_retrieval_collapsed/_expanded(mode)`. This is what makes the
    pipeline RETRIEVAL-AGNOSTIC: each retrieval
    node honours the SAME state contract (fills `contexts`/`chunk_index`/`formatted_context`)
    and the tail only reads that contract, never anything mode-specific.
  - **Via env / CLI:** `RETRIEVAL_MODE` (env var, default `"graph"`) is the combined graph's
    default mode; or `python main.py iterative` to force one in a run.
  - **Mode resolution (combined), in `_resolve_mode`:** context (Studio) → `configurable`
    → `RETRIEVAL_MODE`. Unknown values fall back to the default without breaking.
  - **Traces:** the used mode is recorded in the state (`retrieval_mode`) and, from the CLI,
    as a tag `mode:<x>` and metadata → filterable in LangSmith. LightRAG's internal keyword
    LLM call IS traced now (wrapped with `traceable` in `_make_rag(trace_llm=True)`, only at
    query time, not at index build).
- **Build the LightRAG graph (once):** `.venv\Scripts\python.exe -m graph.lightrag_track`
  (entity extraction over the 517 chunks; resumable, uses the LLM cache).
- RAGAS evaluation / A/B: set `PIPELINE` and `DATASET` in `evaluation.py` and run
  `.venv\Scripts\python.exe evaluation.py`.
- **Dependencies: ALWAYS `uv add` (never pip).** `uv` is not on PATH:
  `& "$env:USERPROFILE\.local\bin\uv.exe" add <package>`.

## Phased roadmap

Agreed order: measure → orchestrate → cheap retrieval → refine+validate → graph (if needed) → UX.

- **Phase 0 — Compass: DONE.** RAGAS golden set with references (drafted by the model, not a
  doctor → caveat). Dense baseline (gpt-4o-mini judge): faithfulness 0.81,
  answer_relevancy 0.58 (MISLEADING, artifact of the metric in Spanish), context_precision
  0.97, context_recall 0.94 (somewhat inflated). Detail in `results/ragas_results.csv`.
- **Phase 1 — LangGraph + LangSmith: DONE.**
- **Phase 2 — Cheap retrieval: DONE.** Hybrid (2a) + reranker (2b).
- **Phase 3 — Refine: DONE.** rephrasing + abbreviation normalization; **domain guardrail**
  (the `in_domain` classification of the rephrase node cuts off out-of-topic questions in ~2 s,
  saving retrieve/rerank/generate/validate); and the **Validate block** (node `validate` after
  `generate`, gpt-4o-mini judge of relevance + semantic grounding; loop with retry injecting
  feedback, `MAX_ITER=2`; if exhausted → `fallback` with a safe message; technical error of
  the judge → `fallback`, no "failing open"). The validator does NOT re-check literal
  citations (that is already done by `evidence`). The question-type classification (vector vs
  graph) was deferred to Phase 4.
- **Phase 4 — Multi-hop: DECIDED (LightRAG graph by default).** Two paths for multi-hop
  questions, compared with an A/B over `MULTIHOP_SET` (16 questions, in `evaluation.py`):
  - **Track A — agentic/iterative** (`agentic/iterative.py`): self-ask loop
    plan → retrieve per sub-query → reflect (MAX_HOPS=3), reuses hybrid+rerank, zero
    reindexing. DONE and tested.
  - **Track B — LightRAG graph** (`graph/lightrag_track.py`): entity-relation graph in file
    storage (`lightrag_store/`), retrieves chunks by traversal and maps them to our payloads
    (preserves citations). Index BUILT (517 chunks, ~1.5 h).
  - Decision by the A/B (multi-hop quality first; then speed/cost/updating). Both paths are
    selectable in the graph via `RETRIEVAL_MODE` and in the eval via `PIPELINE`, and end in the
    SAME generate→validate→evidence.
  - **A/B RESULT (16 multi-hop, context_recall, gpt-4o-mini judge): graph 0.979 >>
    iterative 0.863 > baseline 0.844.** Latency: graph ~11 s ≈ baseline, iterative ~24 s.
    **DECISION: graph (LightRAG) by default** (`RETRIEVAL_MODE="graph"`). Caveats: only clean
    context_recall was measured (context_precision and faithfulness dropped due to judge
    timeouts with many workers → NaN; pending measuring the winner's precision at low
    concurrency); n=16, model-drafted references. The graph's high recall might come with
    slightly less precision (retrieves broader), mitigated by reranker→top8 + validate.
- **Phase 5 — Claude-style UX:** Streamlit/web, streaming, citations, multi-turn memory.
  - **IN PROGRESS — clarification questions (slot-filling): DONE (validated in Studio).**
    `assess_context`/`clarify` gate between retrieval and generate (see Architecture). **Hybrid**
    scheme: screening in `refine` (`known_facts`/`candidate_modifiers`) + evidence-grounded
    confirmation in `rag.assess` (structured reasoning `branches_on`/`already_covered`/`questions`,
    gpt-4o-mini). Pauses with `interrupt()`, folds answers into `clinical_facts` (non-citable,
    steers generation), cap `CLARIFY_MAX_ROUNDS=1`. Tested end-to-end (baseline + MemorySaver)
    and unit cases of `assess` (conditional / datum already known in any unit / non-conditional).
    Final validation: LangGraph Studio (`langgraph dev`, persistence provided by the platform).
  - **Enriched re-retrieval: DONE** (node `re_retrieve`, increment 1). After clarifying, the
    `clinical_facts` are injected into the query and re-trigger retrieval → the doctor's datum
    pulls the conditional passages before generating (tested: HBV changes the top-5).
    PENDING: streaming, web, multi-turn conversation memory, and (increment 2) an "implicit
    knowledge modifiers" path in `assess` (flagged and always followed by re-retrieval +
    validate) as a safety net against retrieval failures.

## Pending / next steps (as of 2026-06-29)

0. **`EVAL_SET` (151 questions, 4 tiers) DONE** and prepared in `evaluation.py` (see bullet).
   Replaces the Phase-4 A/B instrument (the multihop pool saturated graph recall at 0.979 and
   had no simple questions). PENDING: launch it (full RAGAS, ~$15 with the mini judge for the 3
   pipelines, ~$15 more with the gpt-4o judge for the final number) and clinical review of the
   references. Before launching, probe with a subset (~10) to measure real cost in the dashboard.
1. **Contextual Retrieval (enrich chunks with context) — DONE (index built).**
   `chunks/contextualize.py`: per chunk, gpt-4o-mini generates ONE dense context sentence
   (entities/abbreviations/recommendation grade; situating it in its guide via
   title+section_path+neighbour window `--window`, resumable, high `max_retries` because of the
   200k TPM cap) → `chunks_contextual.jsonl` (517) with `context` and `text_for_retrieval`
   (= context + text). The uploader embeds/BM25-indexes `text_for_retrieval` BUT the payload
   keeps the literal `text` (citable). Uploaded to a NEW collection
   **`guias_vih_hibrida_ctx`** (the original `guias_vih_hibrida` intact). **Collection
   selectable** via `QDRANT_COLLECTION` (env, read by `rag.py`) or `--collection` (uploader) →
   boosts the THREE retrievers at once. Real cost ~$0.23 (contextualizing) + ~$0.03
   (re-embedding). PENDING: **A/B `EVAL_SET` against `guias_vih_hibrida` vs
   `guias_vih_hibrida_ctx`** to confirm the improvement before making it the default collection.
2. **Decide the diversion layer (simple→baseline router) ONLY with data:** run `EVAL_SET`
   with pure pipelines and read whether the graph degrades on the `simple`/`single_hop` tiers.
   If there is no gap → the router adds no quality (only latency). Do NOT add before measuring
   (it dirties the test: the simple tier would then measure baseline, not the chosen pipeline).
3. **(Idea) Relation descriptions as a non-citable "concept map"** in the graph's generation
   prompt, to help multi-hop reasoning without breaking grounding. Prototype behind a flag and
   A/B against the current version (faithfulness + recall).
4. **(Idea) HippoRAG 2 as a replacement for LightRAG:** better evidence on multi-hop, fewer
   tokens, and —key— does NOT degrade simple questions (unlike LightRAG/GraphRAG). Prior spike:
   verify backends swappable to Azure/EU (GDPR) and the license. A/B after EVAL_SET.
5. **Phase 5 — UX.** Interactive clarification + enriched re-retrieval DONE (validate in
   Studio); continue with streaming, web (Streamlit/Chainlit), multi-turn memory and concept
   navigation. Increment 2 pending: "implicit knowledge modifiers" path in `assess` (flagged
   and always followed by re-retrieval + validate).

Evaluation artifacts versioned in `results/`: `ragas_results.csv` (Phase-0 baseline) and
`ragas_results_{baseline,iterative,graph}_retrieval.csv` (Phase-4 A/B, context_recall). The
new A/B over `EVAL_SET` dumps `results/ragas_results_<PIPELINE>.csv` (full RAGAS).

## Important findings (do not lose)

- **Abbreviations (largely resolved):** the corpus mostly uses abbreviations (DTG 144 vs
  "dolutegravir" 23, BIC 54 vs 8...) but also full names. That is why the rephrase includes
  BOTH forms and the glossary is also in SYS_PROMPT (they are part of the guides, not external
  knowledge). The problem had 3 layers: retrieval (rephrase), generator comprehension
  (glossary) and model capability.
- **Generation model:** gpt-4o-mini was the quality bottleneck — it failed or answered BADLY
  in cases with nuances/abbreviations even when it had the evidence. **gpt-4o solves it.** That
  is why generation is on gpt-4o.
- **The reranker improves precision/ordering but did NOT raise recall@5** on the test set
  (6/8 same as the hybrid). Useful but not essential.

## Optimizations / known issues

- **RERANKER LATENCY (RESOLVED, Phase 4):** the rerank took **~128 s** (95% of the total
  time) because the cross-encoder pads the batch to the length of the longest chunk on CPU.
  **Fix applied in `rag.rerank`: only `p["text"][:512]` is scored (constant
  `RERANK_SCORE_CHARS`) and the full payloads are returned.** A multi-hop query dropped from
  ~145 s to ~23 s (cold). Critical for Track A, which reranks several times. Machine: 12 cores.
  `RERANK_SCORE_CHARS` is now env-configurable (default 512).
- **THE RERANKER IS STILL THE RETRIEVAL BOTTLENECK (measured):** `retrieve_hybrid`
  (embed+Qdrant+BM25) ~0.44 s vs `rerank` of 20 docs **~3.8 s @512** (scales ~linearly:
  ~1.8 s @256, ~1.0 s @128). Called 1× (baseline/graph) up to 4× (iterative: 3 sub-questions
  + final). Lowering to 256 gives ~2× but **changes the top-8** (up to ~3/8 on some queries) →
  the default is not lowered (priority #1: do not hallucinate). **Parallelization yields little
  in iterative (~1.1×)** because 3 rerankers on CPU saturate the cores; it does help in graph
  (traversal ∥ hybrid = different resources).
- **Optimizations applied (this session):** (1) `iterative_search` retrieves the planned
  sub-questions IN PARALLEL (`ThreadPoolExecutor`); (2) `graph_search` runs traversal ∥ hybrid
  in parallel; (3) `rag.warmup()` preloads reranker+BM25 and `main.py` launches it on a daemon
  thread at import (kills the ~3.5 s of the 1st query); (4) thread-safe locks on the lazy model
  loads. Studio's dedicated graphs already parallelized (Send / branches).
- **RERANKER ON GPU (DONE, the biggest latency improvement).** Measured: `rerank` 20 docs went
  from **~3.8 s (CPU) to ~0.45 s (GPU GTX 1650)** → retrieval: baseline 7.6→5.5 s, **iterative
  15.8→4.9 s (~3.2×, it had 4 reranks)**, graph 11.5→7.7 s. Now the remaining time is the LLM
  calls (rephrase/plan) and `graph_select` (LightRAG), not the reranker. Setup (Windows):
  - Recent NVIDIA driver (610.62, supports CUDA 13.3). **NOTE:** the update left the
    `nvlddmkm` service disabled (`Start=4`) and driver files half-done → a clean driver
    reinstall fixed it.
  - **`onnxruntime-gpu==1.22.0`** (CUDA **12** build; the 1.27 from PyPI is CUDA **13** and does
    not match) + wheels `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`,
    `nvidia-cufft-cu12`, `nvidia-curand-cu12` (NOT the full CUDA Toolkit). The onnxruntime-gpu
    version must match the CUDA major of the wheels (see which `cublas64_XX`/`cufft64_XX`
    `onnxruntime_providers_cuda.dll` imports).
  - **Key trick (Windows):** the wheels' DLLs (`site-packages/nvidia/*/bin`) are NOT on the
    search path, so `_init_cuda_dlls()` in `rag.py` adds them AND **pre-loads** them with
    `ctypes.WinDLL` before importing fastembed (without this the CUDA provider fails and falls
    back to CPU silently). It runs only if `RERANK_DEVICE` is `cuda`/`auto`.
  - Enable: `RERANK_DEVICE=cuda` in `.env` (gitignored; specific to this machine). Removing it
    or `cpu` goes back to CPU. `onnxruntime-gpu` is NOT in `pyproject` (it would break
    machines without a GPU).
- **The reranker cost is IRREDUCIBLE on CPU without losing quality (measured).** Tested:
  (a) lowering chars 512→256 changes the top‑8 (up to 3/8); (b) reranking fewer candidates
  (top15 vs top20) changes the top‑5 (24/30 match; some query 2/5); (c) a single rerank pass
  in iterative changes the top‑8 (3‑6/8). The cross-encoder reorders strongly (a chunk at
  position 18 of the hybrid enters its top‑5), so it needs the 20 candidates @512. NOT
  applied: they would degrade the evidence (priority #1). The only real way is the GPU.
- **GPU available: GTX 1650 4 GB, driver 511.09 (CUDA max 11.6).** For `onnxruntime-gpu` you
  must UPDATE the driver (≥522 for CUDA 11.8, or ≥528 for CUDA 12) + CUDA toolkit + cuDNN,
  then `RERANK_DEVICE=cuda`. System setup (admin), not immediate.
- Local models (BM25, reranker) with lazy load + `warmup()`: without warm-up, the 1st query
  pays ~3.5 s of loading. Studio keeps the models loaded between queries.
- generate (gpt-4o) ~5 s: consider streaming to improve perceived latency (Phase 5).
- **OpenAI LIMIT: gpt-4o at 30,000 TPM (low).** Generation CANNOT be parallelized: running
  several pipelines at once in the eval → `429 RateLimitError` and crash. Run the pipelines
  SEQUENTIALLY (the generation in `build_dataset` is already sequential → it does not crash).
- **RAGAS barely parallelizes** (≈serial even if you raise `max_workers`) and
  `context_precision` is the heavy/fragile metric (1 judge call per chunk → with many workers
  it gives TimeoutError → NaN). `context_recall` is the light and robust one. Stable config:
  `RunConfig(timeout=600, max_workers=8, max_retries=10)`. For a cheap A/B use
  `RETRIEVAL_ONLY=1` (no gpt-4o generation) or `RECALL_ONLY=1` (recall only) in `evaluation.py`.
- **LightRAG graph build:** bottleneck = `max_parallel_insert` (default 2 → raised to 8 in
  `graph.lightrag_track`; also `llm_model_max_async=16`). Even so ~1.5 h for the ~517
  extractions. Resumable and with an LLM cache (re-running is cheap).
- **Machine suspension:** sleeping the laptop KILLS long runs (the connection drops). For long
  jobs disable suspension (`powercfg /change standby-timeout-ac/dc 0`) and restore it
  afterwards. (It happened: an overnight run died when the machine slept.)
- LangSmith + LangGraph Studio **verified OK** this session (Studio starts and traces; the EU
  endpoint responds 204 when sending metadata).

## Conventions

- **Software-architecture fundamentals (ALWAYS, non-negotiable):** every change must uphold
  sound software-architecture principles — this is a portfolio repository and the code quality
  is itself on display. Concretely: **separation of concerns / single responsibility** (each
  module and function does one thing; e.g. `rag.py` = primitives, `pipeline/` = the graph,
  `evidence.py` = formatting), **no leaky abstractions** (retrieval honours the `RAGState`
  contract; the tail never reads anything mode-specific), **DRY** (share primitives, do not
  duplicate logic across tracks), small and readable functions, meaningful names, and comments
  that explain a non-obvious WHY rather than restating the code. When adding or modifying code,
  keep the design clean: extend the right module, factor shared logic, and prefer the simplest
  structure that preserves the behaviour — never trade architecture away for a quick patch.
- **Language policy (updated 2026-07-02):** the code and documentation are in **English** so
  non-Spanish-speaking colleagues can read them. Concretely:
  - **English:** all identifiers, comments and docstrings; the project documents (`README.md`,
    this `CLAUDE.md`, `data/README.md`, `data/prompt.txt`); and developer-facing CLI/tooling
    output (argparse help, `print`/stderr messages, developer `SystemExit`/error messages in
    the scripts).
  - **Spanish (do NOT translate — it would affect functionality or is user/domain-facing):**
    everything shown to the doctor (the `MSG_*` messages in `pipeline/config.py`, the evidence panel
    labels in `evidence.py`, the CLI `input()` prompt); the LLM prompts (`SYS_PROMPT`,
    `build_user_prompt`, and the system prompts of refine/assess/validate/plan/reflect and of
    `contextualize.py`); the guideline/chunk content (`data/markdown/`, `data/textos/`,
    `chunks/*.jsonl`); the values of `abbreviations.py` and the `doc_title`/`topic` metadata in
    `chunk_guidelines.py`'s `DOC_REGISTRY`; regexes that match Spanish guideline text; and any
    Spanish literal that appears verbatim in generated data (e.g. the `"Preámbulo"` breadcrumb
    fallback, the `> _[... omitido — consultar PDF original]_` omission marker). In
    `data/prompt.txt` the prose is English but the Spanish-specific rules (connector-word list,
    TAR/VIH/TB normalization, omission marker) stay verbatim.
  - The user (Victor) is addressed in **Spanish** in chat, even though the files are in English.
  - When adding new code/docs, follow this policy from the start (do not leave new Spanish
    comments/docstrings).
- Every LLM call is encapsulated so it can be switched to Azure OpenAI (private GPT-4) without
  friction the day compliance requires it.
- Direct commits to `main` (single-dev flow). Messages in Spanish. **Do NOT add the
  `Co-Authored-By: Claude…` trailer** to commits: the repo is public (portfolio) and the
  attribution to Claude Code goes in the README and the description, not as a git co-author
  (that trailer made "claude" appear in GitHub's Contributors list). In `.gitignore`:
  `.env`, `*.log`, `.langgraph_api/`, `lightrag_store/` and `data/pdfs/` (the graph is rebuilt
  with `python -m graph.lightrag_track`).
