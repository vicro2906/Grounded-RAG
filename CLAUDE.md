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
an EU region. The doctor's data may include health information (GDPR Art. 9). **State of that
goal, honestly:** Qdrant and LangSmith are EU; OpenAI is not yet, and while this is a prototype
with no real users that is accepted. The code no longer stands in the way — the endpoint is one
`.env` variable (`OPENAI_BASE_URL`) wired through a single factory — but the switch also needs
a new EU-region OpenAI project, which cannot be done from here. See OPEN H in the backlog.

## Current architecture (LangGraph)

Entry point: `main.py` (compiles the graph `app`). The graph itself lives in the `pipeline/`
package (`config`, `state`, `nodes`, `nodes_expanded`, `builder`, `generation`); `main.py` only
wires runtime concerns (UTF-8, LangSmith, warm-up) and the CLI. Graph nodes:

```
question ─▶ rephrase ─┬─ out of domain ─▶ out_of_domain ─▶ END
                      └─ in domain ─▶ [RETRIEVAL_MODE] ─▶ assess_context
                             ├─ baseline : retrieve (hybrid) ─▶ rerank (20→5)          │
                             ├─ iterative: iterative_retrieve (plan→hop→reflect, →8)   │
                             ├─ graph    : graph_retrieve (LightRAG + own hybrid, →8) ◀ DEFAULT
                             ├─ pathrag  : pathrag_retrieve (flow-pruned paths, →8)   │
                             └─ hipporag : hipporag_retrieve (triples→PPR, →8)        ▼
                                              re_retrieve ─▶ generate ─▶ validate ─▶ evidence
                                                   ▲            ▲            │            │
                                                   │            │            │            ▼
                                                   │  refocus_retrieve ◀─────┘   refine_offer
                                                   │  (chases the rejected      (interrupt, OPTIONAL)
                                                   │   claims; not valid,        ├─ declined ─▶ output
                                                   └───┴─ retries left)          └─ answered ─▶ ↺
                                                              (not valid+exhausted ─▶ fallback)
```

The head (rephrase + domain guardrail) and the tail (assess_context→generate→validate→evidence)
are IDENTICAL across the five modes; only the retrieval node changes, chosen by
`RETRIEVAL_MODE` in `pipeline/config.py` (default `"graph"` after the Phase-4 A/B; the new
modes are pending their own A/B). The modes are declared ONCE in `retrieval/registry.py`
(name → lazily-imported search function) and everything else — routing, the Studio dropdown,
`re_retrieve`, the eval's `PIPELINE` — is derived from that catalogue, so adding a mode is a
module plus one registry line.

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
  still requires grounding.
  **ANSWER-FIRST (redesigned 2026-07-22 — this is the important part).** `assess` no longer
  GATES the answer. It used to `interrupt()` before `generate`, so a doctor asking a normal
  clinical question faced **up to 3 sequential pauses (and 3 gpt-4o assess calls) before
  reading a single word** — at the point of care that is worse than a slightly generic answer.
  Now its output is used TWICE and blocks nothing:
    1. the pending dimensions go into the generation prompt as an explicit **"DATOS DEL PACIENTE
       NO APORTADOS"** block (`rag._format_open_questions`), which is what makes not asking
       SAFE: told the datum is unknown, the model lays out the branches («si …; en cambio,
       si …») instead of quietly picking one and stating it as THE recommendation;
    2. the same list is then offered by **`refine_offer`**, AFTER `evidence` has written the
       answer, as an OPTIONAL pause: ignoring it (empty reply) ends the run with the answer
       already on screen; answering loops back through `re_retrieve` → `generate` so the datum
       pulls the conditional passages and the refined answer CITES them.
  Caps flipped accordingly: `CLARIFY_MAX_ROUNDS`=1 (one refinement offer) and
  `CLARIFY_QUESTIONS_PER_ROUND`=3 (all pending dimensions in that single pause) — one question
  at a time only made sense while each pause was the price of getting any answer at all.
  Questions left unanswered STAY in `pending_clarifications`, so the refined answer keeps
  presenting their branches (they are simply not offered again — the budget is spent).
  **Session vs per-question state (multi-turn, 2026-07-23).** `patient_facts` is **SESSION-scoped**:
  the same patient spans several questions, so `node_rephrase` ACCUMULATES the facts screened
  from each question into it instead of resetting it, and `_fold_answers` merges the
  refinement's on top. It is cleared ONLY on an explicit new patient (the CLI's `/nuevo`, which
  runs `update_state({patient_facts: {}})`), so the previous patient's renal function can never
  silently steer the next patient's answer. **Everything else is per-question and IS reset in
  `node_rephrase`** — the clarification budget (`clarify_rounds`/`asked_questions`), the
  validation loop (`attempts`/`validation`), the error/refocus flags. None carries a reducer;
  each writer merges by hand (sequential writes). The split is what makes a follow-up remember
  the patient while the round budget still resets so the next question can raise its own
  refinement.
- **re_retrieve** (node in `pipeline/nodes.py`): sits on the path to `generate`, and is a
  **no-op on the first pass** — it only does work when a refinement ADDED facts
  (`clarify_rounds` > 0). Facts that came in the question itself were already in the initial
  retrieval query, so re-retrieving with them would be a redundant full pass (latency fix).
  When the doctor does supply a datum it **re-retrieves with ALL the `patient_facts` injected
  into the query** (dispatches on `retrieval_mode` through `_retrieve_for_mode`) and OVERWRITES
  `contexts` → so the datum PULLS the conditional passages (the HBV branch, the
  first-trimester one…) and `generate` CITES them, not just steers generation. **Carried-over
  facts are handled by `_with_facts`, not re_retrieve:** the INITIAL retrieval of every turn
  already enriches its query with the accumulated `patient_facts`, so a datum from an earlier
  turn steers this turn's retrieval from the start; re_retrieve is only for facts a refinement
  adds MID-turn (after the initial retrieval already ran). **Total: 1 initial retrieve, plus 1
  more only if the refinement was taken up.** Tested: with
  "HBV coinfection" the HBV-specific chunks enter the top-5. The `clinical_facts` ALSO enter
  `generate` as a NON-citable **"DATOS APORTADOS POR EL MÉDICO"** block (they select the
  guide's branch; the literal citations still come from the chunks). **Requires a checkpointer**
  — `interrupt()` persists the paused run and needs somewhere to persist it. `build_graph` /
  `build_combined_graph` take one as an optional argument: the graphs registered in
  `langgraph.json` are compiled WITHOUT one (the platform injects its own), and every plain
  embedding must supply it. The CLI compiles its own instance with `InMemorySaver`
  (**fixed 2026-07-22**: it did not, so `invoke` returned `__interrupt__` with no `output` and
  `main.py` died with `KeyError: 'output'` on any question that triggered clarification —
  i.e. most clinical ones). Covered by `tests/test_pipeline_flow.py`.
- **Retrieval — 5 selectable modes (`RETRIEVAL_MODE`), all ending in the same generate.**
  Every mode is a function `f(query, top_k=…, rewritten_query=…) -> list[chunk payload]`
  (the `rewritten_query` kwarg is how the pipeline hands over its already-rephrased query so
  no mode rephrases twice). The four graph/multi-hop modes end in the SAME
  `_common.house_tail` (merge with the dense+BM25 complement → rerank → top 8), so an A/B
  measures the SELECTION mechanism and nothing else:
  - **baseline** = `node_retrieve` (`rag.retrieve_hybrid`: dense text-embedding-3-large 3072d +
    sparse BM25 `Qdrant/bm25`, RRF fusion, 20 candidates) → `node_rerank` (`rag.rerank`,
    local cross-encoder, 20→5).
  - **iterative** (Track A) = `retrieval.iterative.iterative_search`
    (plan→hop→reflect, top 8). Plans over the ORIGINAL question; the single-hop fallback reuses
    the caller's rewritten query instead of re-rephrasing.
  - **graph** (Track B, DEFAULT) = `retrieval.graph.graph_search`
    (entity+relation graph `mode="hybrid"` + `retrieve_hybrid` dense+BM25 complement →
    dedup fusion → rerank → top 8).
  - **pathrag** = `retrieval.pathrag.pathrag_search` (arXiv 2502.14902). Reads the EXISTING
    LightRAG store (no new index): LLM keywords → cosine vs entity embeddings → top-40
    distinct CONCEPTS → flow-based pruning (resource 1.0 decayed by α=0.7/degree, early stop
    θ=0.01, ≤4 hops, best path per node pair) → top-15 paths → their nodes'/edges' `source_id`
    → ≤20 chunks → house_tail. It ALSO returns the paths as a **non-citable concept map**
    (see below).
  - **hipporag** = `retrieval.hipporag.hipporag_search` (arXiv 2502.14802, HippoRAG 2).
    Native reimplementation (the `hipporag` PyPI package pins openai==1.91/tiktoken==0.7/vllm
    and does not install on Windows). Query→triple cosine (top-5) → **recognition memory**
    (gpt-4o-mini drops the merely-similar triples) → reset vector = phrase nodes of the
    surviving triples (∝ score) + ALL passage nodes (∝ query-passage cosine × 0.05) →
    `networkx.pagerank` → top-20 passages → house_tail. No triple survives → dense fallback.
- **rerank** (`rag.rerank`): local cross-encoder `jinaai/jina-reranker-v2-base-multilingual`
  (fastembed/ONNX, multilingual, GDPR-ok). Used by the three modes to refine to the final top.
- **concept_map (non-citable, optional part of the retrieval contract):** a mode MAY expose
  the graph structure behind its selection as text (`RAGState.concept_map`; today only
  `pathrag`, via `pathrag_search_with_paths`). It reaches `generate` as a block labelled
  «MAPA CONCEPTUAL … NO citable» placed right before the question (paths ordered by ASCENDING
  reliability, so the most reliable sits closest to the question — the paper's answer to
  "lost in the middle"). It may guide multi-hop reasoning but is NEVER a source: `validate`
  still requires every claim to be grounded in the numbered chunks, and `evidence` still
  fuzzy-matches literal quotes. This is the Phase-4 "open idea" from the graph section,
  implemented — pending its own A/B (faithfulness + recall with and without it).
- **generate** (`pipeline.generation.structured_llm`, gpt-4o): structured output with
  `ChatOpenAI.with_structured_output` (Pydantic `ClinicalAnswer`, strict json_schema).
  Returns dict: sufficient_information, answer, sources_used[{ref,quote}], follow_up_questions.
- **validate** (`rag.validate`, **gpt-4o** since 2026-07-22): relevance + semantic grounding
  judge. Loop with `generate` (injects feedback on retry), `MAX_ITER=2`. valid → evidence;
  not valid and exhausted → `fallback`; technical error of the judge → `fallback` (no "failing
  open"). **Why the strong model:** this is the anti-hallucination guarantee (priority #1) and
  BOTH of its failure modes are invisible — a wrong «valid» ships a hallucination, a wrong
  «invalid» silently swallows a correct answer and shows `MSG_NOT_VALIDATED`. It ran on
  gpt-4o-mini while `assess` (deciding which QUESTION to ask — visible, reversible, bounded)
  ran on gpt-4o: the reversible decision had the better model and the irreversible one the
  worse. Costs more per question, but `assess` now runs once instead of up to 3×, so the net
  change is roughly flat.
- **refocus_retrieve** (node in `pipeline/nodes.py`, added 2026-07-22): the retry path of the
  validation loop. **The rejection used to loop straight back to `generate` with the SAME
  context**, which cannot fix the usual cause of a grounding failure (the retriever missed the
  passage) — so the retry just reworded an unsupported claim until the budget ran out and the
  doctor got `MSG_NOT_VALIDATED`. Now the validator's `unsupported_claims` — previously thrown
  away as prose feedback — become the retrieval query: it re-runs the SAME mode over
  «question + rejected claims», **merges** the result with the context already in hand (so the
  claims that WERE grounded keep their support) and reranks back to the same size. `generate`
  is then told the context may have changed. No claims to chase (a relevance-only rejection) →
  no-op, no retrieval paid for. This is the pipeline's ONLY path to recover from a bad
  selection; `refocus_query` records what it chased, for the trace.
- **technical failures (added 2026-07-23):** every node that reaches out to a service —
  retrieval (all modes), `re_retrieve`, `refocus_retrieve`, `generate`, `evidence` — is wrapped
  by `nodes.guarded(step)` and leaves through `route_on_error`, so an outage ends in the
  `technical_error` node with `MSG_TECHNICAL_ERROR` naming the step. **Two reasons, and the
  second is the clinical one:** an unhandled exception used to propagate out of `invoke` and
  show the doctor a Python traceback; and a technical failure must NEVER be dressed up as a
  clinical result — «no está en las guías» is a statement about the guidelines that could
  change a decision, so the message says explicitly that it is not one. The step label reaches
  the doctor (`technical_error`), the exception does not (`technical_detail`, kept for the
  trace — swallowing it silently would turn an outage into a later mystery). `validate` keeps
  its OWN message (`MSG_VALIDATION_ERROR`): there an answer exists and simply could not be
  verified. **Trap pinned by a test:** `guarded` re-raises `GraphBubbleUp`, because LangGraph
  signals control flow with exceptions and `interrupt()` raises one — a bare `except Exception`
  would turn the refinement pause into a fake outage. It is not load-bearing today (the node
  that pauses is unguarded) and is kept deliberately, so `test_the_guard_lets_the_pause_through`
  tests it directly rather than through the graph.
- **evidence** (`evidence.format_answer`): formats the answer + sources panel with
  literal citations (fuzzy match) + follow-up questions + clinical disclaimer. Text without ANSI.
  It is the LAST barrier before the doctor, so `attribute()` resolves every doubt to `miss`
  (section shown, no quote) rather than certifying. **Hardened 2026-07-22** after two holes
  found by inspection: (a) no minimum quote length — «se recomienda» (13 chars) matched
  literally in almost any chunk, was certified as an exact citation and INHERITED the item's
  evidence grades; now a quote must clear `MIN_QUOTE_CHARS` (40) or name something clinical
  (`_is_substantive`); (b) a fuzzy match SUBSTITUTES the guideline sentence for the model's
  quote, so a quote saying «ABC» was displayed backed by a real sentence saying «TDF o TAF» —
  now `_clinical_tokens` (drugs in EITHER spelling + figures) must not diverge, or it is a
  `miss`. Also fixed the grade over-attribution (a partial quote came back tagged with every
  grade in the item). All of it pinned in `tests/test_evidence.py`.

`retrieval.baseline.search()` = rephrase → hybrid → rerank (used by the evaluation).

### Graph retrieval (LightRAG): what is used and what is NOT

Indexing (once, `retrieval.graph._build_index`): for each chunk, gpt-4o-mini extracts **entities** and
**relations** with an **LLM-generated description**; they are stored in `data/lightrag_store/`:
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
- `rag.py` — retrieval/generation primitives ONLY (no architecture lives here): clients and the
  `chat_model`/`embeddings_model` factories (the single endpoint binding, `OPENAI_BASE_URL`),
  embeddings, retrieve/retrieve_hybrid, rerank, refine, validate, assess, generate_answer
  (raw version), SYS_PROMPT, build_user_prompt, model constants.
- `evidence.py` — answer and sources formatting + citation integrity.
- **`tests/`** — pytest suite, 122 tests, **no API calls and no network**
  (`.venv\Scripts\python.exe -m pytest`, ~8 s; most of that is importing `main` in the CLI
  tests, which warms the reranker). The principle: the LLM is replaced ONLY where the
  randomness enters (refine / assess / validate / generation / retrieval, in `conftest.py`), so
  everything the pipeline DECIDES stays real and every branch can be visited on demand —
  including the ones a manual run almost never reaches (judge rejects, judge errors, budget
  exhausted, second question on the same thread).
  - `test_pipeline_flow.py` — the REAL graph end to end: routing, the interrupt/resume contract
    the CLI depends on, answer-before-offer, the refinement loop and its cap, the
    validator-driven re-retrieval, and both state leaks pinned as regressions.
  - `test_pipeline_routing.py` — the small decisions in isolation, including the one that must
    never fail open (a judge that errored routes to the safety message), plus fact folding and
    the `re_retrieve` no-op that keeps the first pass cheap.
  - `test_evidence.py` — citation integrity: what gets certified, what gets rejected, and that
    a rejected quote never drags the real guideline sentence into view.
  - `test_rag_contracts.py` — the graceful-degradation contracts around each LLM call (refine
    fails → the question still goes through; validate fails → error, never "valid") and the
    non-citable prompt blocks, including their order.
  - `test_cli.py` — the terminal experience: answer-before-offer, declining never reprints, the
    REPL commands (`/nuevo`/`/paciente`/`/salir`), one thread across questions, EOF ends
    cleanly. Drives `main_cli` with a scripted graph and a fake `input`.
  - `test_retrieval_common.py` — concept collapsing, the mapping back to citable payloads, and
    the mode catalogue.
  - `test_llm_client.py` — an ARCHITECTURAL guard: it greps the tree and fails if any module
    builds its own OpenAI client instead of going through the factory.
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
- **`retrieval/`** — the interchangeable architectures, one module per `RETRIEVAL_MODE` value,
  all honouring the same contract (question in → chunk payloads out) and composing `rag.py`'s
  primitives: `registry.py` (**the catalogue**: mode name → lazily-imported search function;
  the single source of truth the pipeline/eval derive from), `_common.py` (**shared helpers**:
  chunk↔payload mapping by text or id, `merge_dedup`, `house_tail`, `expand_abbrevs` and
  `canonical_key`), `baseline.py` (`search`: rephrase → hybrid → rerank; the building block
  the others reuse), `iterative.py` (Track A, plan/hop/reflect), `graph.py` (Track B, LightRAG
  index build + traversal; store in `data/lightrag_store/`), `pathrag.py` (flow-pruned paths
  over that same store — no index of its own) and `hipporag.py` (HippoRAG 2; own store in
  `data/hipporag_store/`, built with `python -m retrieval.hipporag`).
- `ingestion/` — corpus→index scripts (run once per corpus change): `chunk_guidelines.py`
  (structural chunking), `contextualize.py` (Contextual Retrieval), `upload_to_qdrant.py`
  (dense) and `upload_to_qdrant_hybrid.py` (dense+BM25).
- `data/chunks/` — the chunked corpus: `chunks.jsonl` (517 chunks) and
  `chunks_contextual.jsonl` (with the Contextual-Retrieval sentence). `data/lightrag_store/` —
  the generated LightRAG graph index (gitignored, rebuilt with `python -m retrieval.graph`).
- `data/markdown/` — the 7 guides in Markdown (corpus source). `data/pdfs/`, `data/textos/` — originals.
  The `.md` files were transcribed from the PDFs with **code generated by Claude Code adapted to each PDF**
  (non-extrapolable peculiarities), using `pymupdf4llm` (transcribes, **does not invent**). The prompt
  that guided the conversion (absolute fidelity, inspection→script→validation→iteration) is in
  `data/prompt.txt`. See `data/README.md`.
- `docs/` — design documents and architecture diagrams. `results/` — RAGAS evaluation CSVs.
- `langgraph.json` — LangGraph Studio config (exposes `main.py:app`).

## Models and services

- Generation, **validation** and clarification-assessment: **gpt-4o** (`GENERATION_MODEL` /
  `VALIDATION_MODEL` / `ASSESS_MODEL`) — everything that can put an unsupported claim in front
  of a doctor. Rephrase: **gpt-4o-mini** (`REPHRASE_MODEL`), the only mechanical step left.
- Embeddings: `text-embedding-3-large` (3072d). Reranker: jina-reranker-v2-base-multilingual.
- Qdrant Cloud (**eu-west** region). Collections: `guias_vih` (dense only, backup),
  `guias_vih_hibrida` (dense + sparse BM25, no context) and **`guias_vih_hibrida_ctx`**
  (dense + BM25 with Contextual Retrieval, **the default active one**). The active one is set
  by `COLLECTION_HYBRID` in `rag.py` (default `guias_vih_hibrida_ctx`), overridable with
  `QDRANT_COLLECTION` to A/B against the non-contextual one.
- LangSmith in the **EU** region: `.env` with `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`,
  project **`chatbot_vih`** (underscore). Tracing auto-enables if `LANGSMITH_API_KEY` is set.

## How to run

- App (**conversational CLI**): `.venv\Scripts\python.exe main.py`. It compiles its OWN graph
  instance with an `InMemorySaver` (the platform is not there to inject one) and runs a REPL on
  ONE thread, so the patient is remembered across questions. The answer prints as soon as it
  exists; if there are still-unknown patient data the run then pauses to OFFER a refinement, and
  `Command(resume=…)` carries the reply — Enter declines and ends with the answer already shown
  (it only reprints when the text changed). Commands: **`/nuevo`** (forget the patient and start
  fresh — `update_state({patient_facts: {}})`), **`/paciente`** (show the remembered data),
  `/ayuda`, `/salir`.
- Tests: `.venv\Scripts\python.exe -m pytest` (122 tests, ~9 s, no API calls). Run them before
  committing. They freeze the MECHANICS, not the medicine — whether an answer is clinically
  right is what `evaluation.py` and a clinician are for.
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
    Only the modes in `builder.EXPANDED_MODES` have a dedicated graph (a mode earns one after
    winning its A/B); `build_graph` on any other mode raises a ValueError pointing at the
    combined graph. **pathrag/hipporag are collapsed-only for now.**
  - **Combined graph** `app` (build_combined_graph): ALL five routes in one graph; it exposes
    a `context_schema` (`ConfigSchema`) with the **`retrieval_mode`** field as a **dropdown**
    in the run config panel (live choice). Also used by the CLI. Nodes and routing targets are
    generated from `VALID_MODES`, so registering a mode makes it selectable automatically.
  - **Shared head/tail** are factored into `_add_common` (in `pipeline/builder.py`); the
    retrieval section into `_add_retrieval_collapsed/_expanded(mode)`. This is what makes the
    pipeline RETRIEVAL-AGNOSTIC: each retrieval
    node honours the SAME state contract (fills `contexts`/`chunk_index`/`formatted_context`,
    optionally `concept_map`) and the tail only reads that contract, never anything
    mode-specific. The collapsed nodes are BUILT by a factory (`nodes._make_retrieval_node`)
    from the registry, so there is no hand-written node per mode.
  - **Via env / CLI:** `RETRIEVAL_MODE` (env var, default `"graph"`) is the combined graph's
    default mode; or `python main.py iterative` to force one in a run.
  - **Mode resolution (combined), in `_resolve_mode`:** context (Studio) → `configurable`
    → `RETRIEVAL_MODE`. Unknown values fall back to the default without breaking.
  - **Traces:** the used mode is recorded in the state (`retrieval_mode`) and, from the CLI,
    as a tag `mode:<x>` and metadata → filterable in LangSmith. LightRAG's internal keyword
    LLM call IS traced now (wrapped with `traceable` in `_make_rag(trace_llm=True)`, only at
    query time, not at index build).
- **Build the graph indexes (once each, both resumable and gitignored):**
  - `.venv\Scripts\python.exe -m retrieval.graph` — LightRAG entity graph over the 517 chunks
    (~1.5 h, uses the LLM cache). **Also required by pathrag**, which reads the same store.
  - `.venv\Scripts\python.exe -m retrieval.hipporag` — HippoRAG 2 store (~10 min, ~$0.25).
    Resumes at the `openie.jsonl` line level; add `--smoke` instead to inspect a demo query
    (triples retrieved, what recognition memory kept, top passages) without Qdrant.
  - `.venv\Scripts\python.exe -m retrieval.pathrag` — no index to build; prints store stats and
    walks a demo query (keywords → nodes → paths → chunks → concept map).
  - `.venv\Scripts\python.exe -m retrieval.pathrag --describe-es` — regenerates the concept
    map's descriptions in Spanish into `descriptions_es.jsonl` (~5 min, ~$0.30, resumable per
    chunk). Needed only for `pathrag`; without it the map falls back to LightRAG's English
    descriptions. See finding #3 below.
- RAGAS evaluation / A/B: `PIPELINE=<mode> .venv\Scripts\python.exe evaluation.py`. **Probe
  first with `EVAL_SAMPLE=3`** (stratified, 3 per tier = 12 questions, cents) to check wiring
  and latency before spending on the full 151.
- **Dependencies: ALWAYS `uv add` (never pip).** `uv` is not on PATH:
  `& "$env:USERPROFILE\.local\bin\uv.exe" add <package>`.
  **WARNING (bitten 2026-07-21):** any `uv add` re-resolves the lock and reinstalls the CPU
  `onnxruntime` (a fastembed dependency), which SHADOWS the manually installed
  `onnxruntime-gpu` and silently drops the reranker back to CPU. After any `uv add`, check
  `ort.get_available_providers()` contains `CUDAExecutionProvider`; if not, restore with
  `uv pip install --reinstall "onnxruntime-gpu==1.22.0"` (do NOT `uv pip uninstall
  onnxruntime` first — it deletes files the GPU build shares and breaks the import).

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
  - **Track A — iterative** (`retrieval/iterative.py`): self-ask loop
    plan → retrieve per sub-query → reflect (MAX_HOPS=3), reuses hybrid+rerank, zero
    reindexing. DONE and tested.
  - **Track B — LightRAG graph** (`retrieval/graph.py`): entity-relation graph in file
    storage (`data/lightrag_store/`), retrieves chunks by traversal and maps them to our payloads
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
  - **Clarification questions (slot-filling): DONE, and REDESIGNED to answer-first
    (2026-07-22).** **Hybrid** scheme: screening in `refine`
    (`known_facts`/`candidate_modifiers`) + reasoning in `rag.assess` (structured
    `clinically_relevant`/`branches_on`/`already_covered`/`questions`, gpt-4o). The pause is now
    a REFINEMENT offered after the answer (`refine_offer`), not a gate before it, and what the
    doctor did not provide reaches generation as an explicit UNKNOWN block so the answer
    presents the branches — see Architecture. Covered end-to-end offline in
    `tests/test_pipeline_flow.py`. **Still to validate live in Studio** (`langgraph dev`), and
    with a clinician: whether a branch-laden answer reads better than the old interrogation is
    a judgement no test makes.
  - **Enriched re-retrieval: DONE** (node `re_retrieve`, increment 1). After clarifying, the
    `clinical_facts` are injected into the query and re-trigger retrieval → the doctor's datum
    pulls the conditional passages before generating (tested: HBV changes the top-5).
    PENDING: streaming, web, multi-turn conversation memory, and (increment 2) an "implicit
    knowledge modifiers" path in `assess` (flagged and always followed by re-retrieval +
    validate) as a safety net against retrieval failures.

## Pending / next steps (as of 2026-07-22)

**Critical review of 2026-07-22 — backlog in priority order.** A full read of the system
surfaced the items below. The first block is DONE (same session); the rest is open work,
ordered by value/cost. Everything here is deliberate design debt, not discovery: read it as
the plan, not as a bug list to rediscover.

- **DONE — CLI resumable** (`interrupt()` had no checkpointer → `KeyError: 'output'`),
  **per-question state reset** (`attempts`/`validation` leaked into the next question through a
  persistent thread), **citation integrity hardened** (filler quotes certified as literal +
  fuzzy silently swapping the clinical content), **pytest suite** (47 tests, offline), and
  **the self-correcting retry** (`refocus_retrieve`: a grounding rejection now re-retrieves the
  rejected claims instead of regenerating over the same context — see Architecture).
  **PENDING for it:** measure whether it actually rescues answers (rejection → valid rate) and
  what it adds in latency on the bad path. It is reasoned, not measured.
- **DONE (2026-07-22) — risk allocation and the clarification flow.** `validate` moved to
  gpt-4o (it is the invisible failure); `assess` now runs ONCE per question instead of up to 3×,
  so the net cost is roughly flat. The clarification became ANSWER-FIRST (see Architecture).
  **PENDING for both:** measure the false-rejection rate that motivated the model swap, and
  whether the conditional "branch" answers actually read better to a clinician than the old
  interrogation did. Still open from the original finding: `assess` runs even when `refine`
  returned an empty `candidate_modifiers`, a screen already computed for free.
- **OPEN D — multi-turn memory: patient facts DONE (2026-07-23), contextual rewriting +
  auto-detect PENDING.** The fix was SCOPING, not resetting: `patient_facts` is now
  session-scoped (accumulates across questions, feeds both retrieval via `_with_facts` and
  generation), visible with `/paciente` and cleared with `/nuevo`; the CLI is a REPL on one
  thread. **Still pending (staged):** (2) rewriting a follow-up against the conversation so
  «¿y en embarazo?» is understood as a continuation (needs history threaded into `refine`), and
  (3) auto-detecting a probable patient switch (contradictory facts) and asking to confirm
  before answering. The 3 follow-up questions the system emits are still not clickable.
- **OPEN E — the evaluation does not measure the shipped system.** `build_dataset` calls
  retriever + `generate_answer` directly, skipping `refine`, `assess`/`clarify`, `re_retrieve`,
  the `validate` loop and `evidence`. So no number covers the riskiest component: the FALSE
  REJECTION rate of `validate` (a correct answer silently replaced by `MSG_NOT_VALIDATED`).
  There is also no ABSTENTION metric, though the `adversarial` tier exists to measure it.
- **OPEN F — the A/B is confounded and more expensive than it needs to be.** `house_tail` mixes
  every graph mode's selection with the same dense+BM25 complement, so the four modes share a
  good part of their final context and the measured deltas are compressed; an ablation WITHOUT
  the complement is what isolates the selection mechanism. And since only retrieval varies,
  restoring a retrieval-only path (recall/precision, no gpt-4o generation) would run all five
  modes over the 151 questions for a fraction of the current cost.
- **OPEN G — rich metadata unused.** `topic` / `year` / `section_number` / `evidence_grades` are
  in every payload and filter nothing in Qdrant. With guidelines from 2020 and 2022 coexisting,
  `SYS_PROMPT` rule 5 ("state both versions") is unhelpful when one is simply older.
- **OPEN H — GDPR: the code is ready, the account is not (decided 2026-07-22).** It is a
  PROTOTYPE with no real users, so running on OpenAI US is accepted **for now**; the move to
  Europe is wanted in the near future. What was done: `OPENAI_BASE_URL` (in `rag.py`, mirrored
  by the standalone ingestion scripts) is now the single endpoint knob, and `rag.chat_model()` /
  `rag.embeddings_model()` are the only places a client is built — pinned by
  `tests/test_llm_client.py`, which fails if any module constructs its own (the drift is
  otherwise invisible: a stray `ChatOpenAI(...)` works perfectly until residency matters).
  LightRAG builds its own client, so `retrieval/graph.py` hands it the endpoint explicitly.
  **What is left, and it is NOT code:** OpenAI configures residency **per project and only at
  creation** — an existing project cannot be migrated. So it needs a NEW project with region
  Europe (eligibility is limited to certain account types: check whether the region selector
  appears when creating it), its own API key, and then `OPENAI_BASE_URL=https://eu.api.openai.com/v1`
  in `.env`. Those projects run with zero data retention. Qdrant and LangSmith are already EU.
  Still pending regardless of region: **masking `clinical_facts` in the LangSmith traces**
  (Art. 9 health data currently travels in the prompt and into the trace).
- **OPEN I — minor.** `rapidfuzz` is a declared dependency and is used nowhere (`evidence.py`
  matches with `difflib`); either use it there or drop it.

0. **THE FULL A/B IS THE BLOCKER.** Five modes are implemented and wired
   (baseline / iterative / graph / pathrag / hipporag). A **stratified probe
   (`EVAL_SAMPLE=3` = 12 questions, gpt-4o-mini judge, 0 NaN) ran on 2026-07-22** over the
   three graph modes (~$1.05):

   | | graph | pathrag | hipporag |
   |---|---|---|---|
   | faithfulness | **0.887** | 0.877 | 0.765 |
   | context_precision | **0.946** | 0.920 | 0.930 |
   | context_recall | 0.910 | **0.979** | 0.972 |
   | latency s/query | 12.0 | **11.0** | 13.0 |

   context_recall per tier — the two new modes beat graph exactly where graph is weakest:
   `simple` 0.833 → **1.000** (both) and `single_hop` 0.889 → **1.000** (pathrag); hipporag
   takes `multihop` (1.000 vs 0.917). That is the property they were adopted for: more recall
   WITHOUT degrading easy questions. Against it, hipporag's faithfulness drops on `single_hop`
   (0.583). **With n=3 per tier none of this decides anything** — one bad answer moves a whole
   cell; it says where to look in the full run, not which mode wins.
   PENDING: the full `EVAL_SET` (151 questions) per pipeline — ~$4-5 each with the mini judge,
   so ~$22 for the five; the account had ~$4.45 left after the probe. And the clinical review
   of the references, still the deepest caveat on every number here.
1. **Contextual Retrieval (enrich chunks with context) — DONE (index built).**
   `ingestion/contextualize.py`: per chunk, gpt-4o-mini generates ONE dense context sentence
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
3. **Non-citable "concept map" — IMPLEMENTED (pathrag), pending its A/B.** The paths reach
   generation as a non-citable block (see the concept_map bullet in Architecture). PENDING:
   A/B it on/off (faithfulness + recall).
   **Finding (fixed 2026-07-21): LightRAG's descriptions were in ENGLISH** — its extraction
   prompt is English, so it described a Spanish corpus in English (84% of nodes, 70% of
   edges), and that text was going into a prompt answered in Spanish over Spanish guidelines.
   Fixed by regenerating ONLY the descriptions from their source chunk
   (`--describe-es` → `descriptions_es.jsonl`, 9493/9537 items, ~$0.30), which also makes them
   clinically specific instead of a translation. Re-extracting the whole graph in Spanish
   (LightRAG accepts `{language}`) was rejected: ~1.5 h AND it would invalidate the index the
   graph mode was measured on. **Still open:** the map's usefulness is unproven, and the
   noise upstream is real — for the HBV question the keyword extractor invented «VIH-2»,
   «mutaciones de resistencia» and «tratamiento optimizado», none of which are in the
   question, and they drag in irrelevant nodes. Tighten `_KEYWORDS_SYS` only if the A/B says
   the map hurts.
4. **HippoRAG 2 — IMPLEMENTED (`retrieval/hipporag.py`), pending its A/B.** Native
   reimplementation, NOT the PyPI package (`hipporag` pins openai==1.91 / tiktoken==0.7 /
   vllm==0.6.6.post1: incompatible with our stack and vllm does not install on Windows).
   Index: **517 passages, 7792 phrases, 5609 triples, 8309 nodes / 16481 edges**, built in
   ~10 min for ~$0.25 (vs LightRAG's ~1.5 h) with ZERO extraction failures. Manual smoke on
   the HBV question: the top triple is literally the answer («…se debe iniciar precozmente un
   TAR que incluya TDF o TAF y FTC o 3TC»), recognition memory kept 2/5, and the top passage
   is `TAR_2022 §7.4.4 HEPATOPATÍAS (VHC, VHB, CIRROSIS)`. On the simple question (AZT
   intraparto) it returns the right chunk at rank 1, like baseline — the "does not degrade
   simple QA" property, though n=3 proves nothing yet.
4 bis. **PathRAG — IMPLEMENTED (`retrieval/pathrag.py`), pending its A/B.** Implemented from
   the paper (the BUPT-GAMMA repo is a LightRAG derivative of unclear provenance and was not
   used). NO new index: it reads `data/lightrag_store/` directly (graphml + `vdb_entities`
   base64 float32 matrix + `kv_store_text_chunks`), all format knowledge isolated in
   `_PathStore` — the ONE point coupled to the lightrag-hku layout, validated on load.
   **Two findings while testing, both fixed:**
   (a) The LightRAG graph holds up to SEVEN spellings of the same concept («TAR»,
   «Tratamiento Antirretroviral», «TAR=Tratamiento Antirretroviral»…; 163 of 3382 nodes are
   such duplicates). Without collapsing them the top-40 node budget filled with TAR variants
   and the paths were trivial → `_common.canonical_key` (accent/case/punctuation folding +
   name→abbreviation contraction + repeated-token drop) collapses them. Same helper is used
   for HippoRAG's phrase nodes.
   (b) **Ranking paths by the paper's flow value alone ordered them almost INVERSELY to
   clinical usefulness.** Flow is split across a node's neighbours, so for a one-edge path
   `S(P) = 1 + α/degree`: two leaf nodes keep nearly all of it (degree 1 → 1.70) while the
   concepts that matter are penalized for being well connected (VHB, degree 23 → 1.03; TAR,
   degree 210 → 1.00). On the HBV question «Pre-TAR Era → Post-TAR Era» outranked
   «TAF → VHB» — and since the map is ordered worst-to-best, the noise landed in the
   highest-attention slot next to the question. The paper does not hit this because it assumes
   every retrieved node is already relevant, leaving flow to judge only connection strength.
   **Fix: rank by `flow × relevance`**, relevance being the endpoints' cosine to the question,
   which `retrieve_nodes` already computed and was throwing away. After it, FTC→TAF and
   TAF→VHB rank top and Pre-TAR Era drops to 5th. The relevance is **min-max rescaled over the
   retrieved nodes** (`_rescale`): raw cosines sit in a narrow 0.60-0.68 band against a
   1.00-1.70 flow range, so unscaled they barely reordered anything. The scores are NOT
   printed in the prompt (a bare «1.70» invites being read as clinical confidence);
   `path_scores` exposes them and `_trace_paths` records them in LangSmith.
   **Caveat: all three PathRAG corrections (canonical_key, flow×relevance, rescaling) were
   justified by manual inspection of ONE question, not measured.** They are reasoned, not
   validated — the A/B is what decides whether PathRAG earns its place at all.
5. **Phase 5 — UX.** Interactive clarification + enriched re-retrieval DONE (validate in
   Studio); continue with streaming, web (Streamlit/Chainlit), multi-turn memory and concept
   navigation. Increment 2 pending: "implicit knowledge modifiers" path in `assess` (flagged
   and always followed by re-retrieval + validate).

Evaluation artifacts versioned in `results/`: `ragas_results.csv` (Phase-0 baseline) and
`ragas_results_{baseline,iterative,graph}_retrieval.csv` (Phase-4 A/B, context_recall). The
new A/B over `EVAL_SET` dumps `results/ragas_results_<PIPELINE>.csv` (full RAGAS, now with a
`tier` column so a run can be re-sliced by question type afterwards).
`ragas_results_{graph,pathrag,hipporag}.csv` are the **12-question probe** of 2026-07-22, NOT
the full A/B — read them as such (n=3 per tier).

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
- **Optimizations applied (2026-07-02, redundant LLM/retrieval calls on the critical path):**
  (1) the iterative single-hop fallback re-rephrased a question the graph had already
  rephrased (a duplicate gpt-4o-mini call, ~1-2 s, on EVERY single-hop iterative run) →
  `iterative_search` now takes an optional `search_query` and the nodes pass the state's one;
  (2) `re_retrieve` re-ran a FULL retrieval pass (in graph mode including another LightRAG
  keyword LLM call, ~1-8 s) whenever `clinical_facts` was non-empty — but facts seeded by
  `refine` from the question itself are already in the initial retrieval query, so it now
  no-ops unless a clarification round actually ADDED facts (`clarify_rounds > 0`). Remaining
  candidates NOT applied (complexity vs. readability): speculative single-hop prefetch during
  `_plan`, speculative generate during assess, LightRAG storage warm-up at import (unsafe with
  Studio's reloader and breaks the "LightRAG stays lazy" decision).
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
  `retrieval.graph`; also `llm_model_max_async=16`). Even so ~1.5 h for the ~517
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
  module and function does one thing; e.g. `rag.py` = primitives, `retrieval/` = the three
  architectures, `pipeline/` = the graph, `evidence.py` = formatting), **no leaky abstractions** (retrieval honours the `RAGState`
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
    `data/chunks/*.jsonl`); the values of `abbreviations.py` and the `doc_title`/`topic` metadata in
    `chunk_guidelines.py`'s `DOC_REGISTRY`; regexes that match Spanish guideline text; and any
    Spanish literal that appears verbatim in generated data (e.g. the `"Preámbulo"` breadcrumb
    fallback, the `> _[... omitido — consultar PDF original]_` omission marker). In
    `data/prompt.txt` the prose is English but the Spanish-specific rules (connector-word list,
    TAR/VIH/TB normalization, omission marker) stay verbatim.
  - The user (Victor) is addressed in **Spanish** in chat, even though the files are in English.
  - When adding new code/docs, follow this policy from the start (do not leave new Spanish
    comments/docstrings).
- **Every LLM client is built in ONE place: `rag.chat_model()` / `rag.embeddings_model()`**,
  which bind `OPENAI_BASE_URL`. Never instantiate `ChatOpenAI`/`OpenAI`/`OpenAIEmbeddings` in a
  module — `tests/test_llm_client.py` fails the build if you do. This is what makes changing
  region (or provider, the day compliance requires Azure) a `.env` line instead of a hunt
  through seven modules. The standalone ingestion scripts read the same variable directly, on
  purpose: they must not import the retrieval stack.
- Direct commits to `main` (single-dev flow). Messages in Spanish. **Do NOT add the
  `Co-Authored-By: Claude…` trailer** to commits: the repo is public (portfolio) and the
  attribution to Claude Code goes in the README and the description, not as a git co-author
  (that trailer made "claude" appear in GitHub's Contributors list). In `.gitignore`:
  `.env`, `*.log`, `.langgraph_api/`, `data/lightrag_store/` and `data/pdfs/` (the graph is
  rebuilt with `python -m retrieval.graph`).
