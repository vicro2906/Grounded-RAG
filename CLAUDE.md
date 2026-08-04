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
  info, normalizing terms in BOTH forms «full name (ABBR)» with `abbreviations.py`; (c)
  **screens** the patient data already present (`known_facts`→`patient_facts`) and the
  clinical modifiers the question might need (`candidate_modifiers`) — the cheap half of the
  clarification step; (d) **resolves an elliptical follow-up** against the PREVIOUS question
  (`prev_question`, session-scoped): «¿y en embarazo?» is rewritten into a standalone query
  carrying the earlier topic (which must come literally from one of the two questions — no new
  clinical info); and (e) **flags a probable patient switch** (`possible_new_patient`) when the
  question contradicts the accumulated `patient_facts`. If out of domain → `out_of_domain`
  (direct message). Then the **`confirm_patient` gate**: a no-op unless (e) fired AND there is
  remembered data to contradict, in which case it `interrupt()`s to confirm before answering —
  answering a second patient's question with the first's gestation/renal function folded in is
  the harm the system exists to avoid, so this is the ONE pause allowed to block (answer-first
  holds for the clarification because a generic answer is safe; a cross-patient one is not). On
  «sí» it drops the previous patient's facts, keeping only this question's own (`turn_facts`);
  anything else keeps the patient. Generation uses the ORIGINAL question, not the rewritten one.
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
    dedup fusion → rerank → top 8). It hands its traversal to `house_tail` as a CALLABLE, which
    is how it keeps running traversal ∥ hybrid in parallel without keeping a second copy of the
    tail (it had one until 2026-07-29 — same logic, two places to drift).
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

### Visible work: how a run reports itself (2026-07-29)

A clinical question spends **~35 s** in refine → retrieval → assess → generate → validate
(measured over 20 questions in graph mode: mean 34.4 s, median 34.6, max 50.8 — the "12-25 s"
figure that circulated came from the retrieval-only eval path, which pays neither assess nor
validate), and the CLI used to print **nothing at all** until the answer was ready. It now reads the run as a
STREAM (`app.stream(stream_mode=["updates", "custom"])`) and paints a self-erasing status line.

- **`updates` is the free half.** LangGraph emits one chunk per node AS IT COMPLETES, covering
  every step including the three the `guarded` decorator does not wrap (`rephrase`,
  `assess_context`, `validate` — ~10 s of the wait). `MSG_STEP_LABELS` in `pipeline/config.py`
  maps a node to what runs NEXT, because a chunk means "that one is done"; the pipeline is
  linear enough after each node for that to be the useful thing to say. Nothing in the graph
  changed for this.
- **`custom` (`progress.py`) is for what `updates` cannot see:** inside a single long node. The
  graph modes collapse retrieval into ONE node and spend 5-10 s there, so `house_tail` emits a
  `detail` mid-way, and every retrieval path emits `sources` — the guideline sections it read,
  labelled with `evidence.section_label` so they cannot disagree with the sources panel later.
  The sections line is **KEPT** on screen instead of being overwritten: it is a fact about what
  was consulted, and it is the transparency moment (the doctor recognises the ground the answer
  stands on before reading a word).
- **The clinical TEXT is deliberately NOT streamed, and this is a decision, not a TODO.**
  `validate` runs AFTER `generate` and can reject the answer, so streaming tokens means a doctor
  can read something that is then retracted — the same category of harm as the cross-patient
  answer that `confirm_patient` blocks for, and the reason answer-first was allowed everywhere
  else (a generic answer is safe; an unsupported one is not). There are three lesser costs too:
  `evidence.format_answer` wraps the text and DROPS uncertifiable citations, so the draft would
  be printed twice and differently; the draft arrives without the sources panel that is the
  product's whole differentiator; and on a validator rejection (`MAX_ITER=2`) two contradictory
  drafts would stream. **The technical path is known if this is ever reconsidered:**
  `stream_mode="messages"` yields the generation deltas without touching `generation.py` or
  `node_generate` (LangGraph's message handler flips `_should_stream`, and the final chunk
  carries `parsed`, so `.invoke()` still returns a validated `ClinicalAnswer`); `answer` is
  already field 2 of the schema so text starts after a one-token boolean; it would need
  `disable_streaming=True` by default in `rag.chat_model` so only generation takes that path,
  and a `metadata["langgraph_node"] == "generate"` filter or the validator's JSON paints itself
  as the answer. Parsing partials off `structured_llm.stream()` is NOT viable — the parser is a
  non-generator `RunnableLambda` and emits a single final object.

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
match + prefix fallback, the latter only when the opening names ONE chunk — see the finding
below; 20/20 in a test). (4) **Complement (replaces LightRAG's internal
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
  warm-up) and the CLI. Thin; the graph lives in `pipeline/`. It READS THE RUN AS A STREAM
  (`Status` + `_answer_question`) — see "visible work" below.
- `progress.py` — the progress side channel (root module ON PURPOSE: `rag.py` and `retrieval/`
  sit below the pipeline and cannot import from it, yet they are where the long steps are;
  this is observability, like `logging`). `emit()` is fire-and-forget and a NO-OP outside a
  graph run, so `evaluation.py` and the smoke scripts, which call the primitives directly,
  cannot break on it. `read_chunk()` normalizes ONE stream chunk into the events a frontend
  reacts to — shared by the CLI and the web so neither invents its own reading of the stream.
- **`web/app.py`** — the Chainlit frontend (`chainlit run web/app.py`). Thin by design: it owns
  no rule the CLI does not also follow (the stream's rules are `progress.read_chunk`, the resume
  encoding is `pipeline.refinement_reply`, the citation verdicts and the wording are
  `evidence.resolve_answer` / `format_answer_markdown`). What it adds is the UI for the moments a
  chat surface does better than an `input()`: the patient-switch gate is two buttons, declining
  the refinement is ONE click instead of one Enter per question, and the three follow-up
  questions are `cl.Action` buttons — the last open item of OPEN D. Progress arrives as
  collapsible `cl.Step`s and the sections read get their own kept step. Uses `astream` so the
  sync nodes run in LangGraph's executor. `.chainlit/config.toml` + `chainlit.md` are ours and
  committed; `.chainlit/translations/` is regenerated boilerplate and gitignored.
- **`pipeline/`** — the LangGraph app, assembled from `rag.py`'s primitives: `config.py`
  (constants, Studio context schema, `MSG_*`), `state.py` (state schemas + reducers),
  `nodes.py` (combined-pipeline nodes + routing), `nodes_expanded.py` (one-node-per-step
  retrieval for the dedicated Studio graphs), `builder.py` (head/tail assembly +
  `build_graph`/`build_combined_graph`), `generation.py` (structured `ClinicalAnswer` LLM).
- `rag.py` — retrieval/generation primitives ONLY (no architecture lives here): clients and the
  `chat_model`/`embeddings_model` factories (the single endpoint binding, `OPENAI_BASE_URL`),
  embeddings, retrieve/retrieve_hybrid, rerank, refine, validate, assess, generate_answer
  (raw version), SYS_PROMPT, build_user_prompt, model constants.
- `evidence.py` — citation integrity, in two halves: **`resolve_answer(answer, chunk_index) ->
  AnswerView`** (the answer AS DATA: text, sources grouped by section with their certified
  citations and grades, follow-ups) and **`format_answer`**, which RENDERS that view as the
  72-column terminal panels. Every integrity verdict lives in the resolver, so no frontend can
  bypass it. `AnswerView` is deliberately NOT stored in the graph state — `answer` and
  `chunk_index` already are, both plain data, so a web frontend calls `resolve_answer` itself
  and no dataclass ever has to survive a checkpointer's serializer.
- **`tests/`** — pytest suite, 242 tests + 8 xfail, **no API calls and no network**
  (`.venv\Scripts\python.exe -m pytest`, ~18 s; most of that is importing `main` in the CLI
  tests, which warms the reranker). The principle: the LLM is replaced ONLY where the
  randomness enters (refine / assess / validate / generation / retrieval, in `conftest.py`), so
  everything the pipeline DECIDES stays real and every branch can be visited on demand —
  including the ones a manual run almost never reaches (judge rejects, judge errors, budget
  exhausted, second question on the same thread).
  - **`test_corpus_quality.py`** — the quality gates over the corpus THAT SHIPS, plus the corpus
    generation switch. Two kinds of test, and the split is the point. The gates run against the
    committed `chunks.jsonl` and `data/markdown/`, and the eight that currently FAIL are marked
    `xfail(strict=True)` carrying the number measured on 2026-07-31 — strict, so when the
    extractor/chunker rewrite fixes one the test **fails because it passed** and the debt entry
    has to be deleted by hand. The debt can be neither paid nor re-incurred silently. Alongside
    them, every gate is fed a hand-made violation, because a gate that stopped detecting anything
    would otherwise show up as a row of green ticks.
  - `test_pipeline_flow.py` — the REAL graph end to end: routing, the interrupt/resume contract
    the CLI depends on, answer-before-offer, the refinement loop and its cap, the
    validator-driven re-retrieval, and both state leaks pinned as regressions. **Also the
    STREAM contract** the CLI now depends on (the pause arriving as a `__interrupt__` chunk, a
    no-op node yielding an empty update, the `sources` event landing before the answer, and
    every `MSG_STEP_LABELS` key naming a node that exists) — without it that contract sat
    between a real graph driven by `invoke` and a fake graph in `test_cli.py`, asserted by
    nobody.
  - `test_pipeline_routing.py` — the small decisions in isolation, including the one that must
    never fail open (a judge that errored routes to the safety message), plus fact folding,
    the `re_retrieve` no-op that keeps the first pass cheap, and the progress side channel
    (emitting outside a run is harmless; the sections are labelled like the sources panel).
  - `test_evidence.py` — citation integrity: what gets certified, what gets rejected, and that
    a rejected quote never drags the real guideline sentence into view — asserted BOTH on the
    rendered panel and on the `AnswerView`, so a frontend reading the data cannot reach a
    different verdict from the terminal. The panel tests were left untouched through the
    resolver/renderer split: that they still pass is the proof the output did not move.
  - `test_rag_contracts.py` — the graceful-degradation contracts around each LLM call (refine
    fails → the question still goes through; validate fails → error, never "valid") and the
    non-citable prompt blocks, including their order.
  - `test_cli.py` — the terminal experience: answer-before-offer, declining never reprints, the
    REPL commands (`/nuevo`/`/paciente`/`/salir`), one thread across questions, EOF ends
    cleanly, Ctrl-C cancels a query without ending the session, and the status line stays
    SILENT without a tty (which is what keeps every other assertion here describing the real
    terminal). Drives `main_cli` with a scripted graph and a fake `input`.
  - `test_retrieval_common.py` — concept collapsing, the mapping back to citable payloads, the
    mode catalogue, and `house_tail` (the tail four modes share: merge order, reranking against
    the ORIGINAL question, and the callable form running the selection alongside the hybrid).
  - `test_llm_client.py` — an ARCHITECTURAL guard: it greps the tree and fails if any module
    builds its own OpenAI client instead of going through the factory.
  - `test_evaluation.py` — the product-measurement path: how each run is CLASSIFIED (three
    causes of "no answer", each needing a different fix), that every question gets its own
    thread (one thread would leak the previous patient and measure a conversation nobody had),
    that a run answering NOTHING is reported rather than aborted, that an outage records what
    broke, and that a failed judge does not sink a report the run already paid for.
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
- **`data/corpus.toml`** — the DOCUMENT MANIFEST (replaces the old `DOC_REGISTRY`): per document
  `doc_id, title, specialty, topics, organization, year, language, markdown, reference,
  source_pdf`. An unlisted `.md` is now an ERROR — it used to enter the corpus silently tagged
  `topic="vih_general"`, i.e. a document whose year and scope the generation prompt could not
  see while that same prompt arbitrates between guides spanning 2013-2025. It is also the ONLY
  thing that knows which reference text belongs to which Markdown (the names pair by no rule:
  `VIH_TB.md` ↔ `textos/TB_VIH/`), which is what makes the G5 oracle possible.
- **`data/specialties/<id>.toml`** — the SPECIALTY PROFILE: display name, the in-domain
  description for the guardrail, the clinical modifiers (`slug` for the screener + `label` for
  the clarifier, ONE entry so the two prompts cannot drift — they already had), the grade scheme
  and the abbreviations. **`specific_scopes` is NOT here: it is derived from the manifest**, so
  adding a document teaches SYS_PROMPT's conflict rule about it.
- **`corpus.py`** — **the corpus GENERATION switch** and the loader for both files above,
  and a root module for the same reason
  `progress.py` is one (`rag.py`, `retrieval/` and `ingestion/` all need it, and `ingestion/`
  must not import the retrieval stack). The corpus fans out into FOUR artifacts that must always
  describe the same text — `chunks.jsonl`, the Qdrant collection, the LightRAG store (PathRAG
  reads it too) and the HippoRAG store — and rebuilding them takes ~2 h, so a chunking rewrite
  cannot be atomic. `CORPUS_VERSION` (env, default `v1`) picks a `Layout` and every location is
  derived from it, which makes the half-migrated state UNREPRESENTABLE: chunks v2 next to a v1
  graph store is exactly what wakes `map_to_payloads`' prefix fallback and mis-cites sections
  (measured: 4 of 4 lookups resolved to the wrong chunk). `v1`'s names are spelled out because
  they are already live in Qdrant and on disk; the cutover to `v2` is one line, and reversible.
  Pinned by `tests/test_corpus_quality.py`, including a source-level guard that fails if any
  module names an artifact instead of deriving it — which immediately caught
  `upload_to_qdrant_hybrid.py` defaulting to `guias_vih_hibrida` while the app queried
  `guias_vih_hibrida_ctx`, i.e. uploading to a collection nobody searched.
  It also owns **`Scope`** (which slice of the corpus a search may reach) and the per-generation
  flag **`scopable`**. **v1 is NOT scopable, and that is correct, not a workaround:** Qdrant
  REJECTS a filter on a field with no payload index, the live collection predates `specialty`,
  and a plain equality filter over an absent field would match nothing — turning every question
  into «no está en las guías», a clinical claim produced by a schema mismatch. A corpus built
  before specialties existed holds exactly one, so searching all of it IS searching that one.
- `ingestion/` — corpus→index steps, a PACKAGE (run them with `python -m ingestion.<step>`, like
  `retrieval/`): `chunk_guidelines.py` (structural chunking), **`quality.py`** (the quality
  gates), `contextualize.py` (Contextual Retrieval), `upload_to_qdrant.py` (dense) and
  `upload_to_qdrant_hybrid.py` (dense+BM25).
- **`ingestion/quality.py`** — the corpus quality GATES, and the reason they are code rather than
  a checklist: every defect they check was found by reading the corpus by hand long after it had
  been indexed and answered from, and none announced itself. `python -m ingestion.chunk_guidelines
  data/markdown --check` runs them and **writes nothing** on failure. Each gate is deterministic,
  offline and calls no model (a gate judged by an LLM shares the failure mode of what it checks).
  `count_tokens` is now STRICT — no characters/4 fallback — because the shipped corpus was built
  with that fallback and 89 of its 517 chunks silently exceed the budget the chunker claims to
  enforce.
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
  `/ayuda`, `/salir`. **Ctrl-C** cancels the query in flight and returns to the prompt (safe:
  the next question restarts from the top on the same thread, the orphaned pause is discarded
  and `patient_facts` survives); a second one at an idle prompt ends the session.
- Tests: `.venv\Scripts\python.exe -m pytest` (242 tests + 8 xfail, ~18 s, no API calls). Run them before
  committing. They freeze the MECHANICS, not the medicine — whether an answer is clinically
  right is what `evaluation.py` and a clinician are for.
- **Web (Chainlit): `.venv\Scripts\chainlit.exe run web/app.py`** → http://localhost:8000.
  **Launch it FROM THE REPO ROOT:** the config it must pick up is `.chainlit/config.toml` there
  (Spanish UI, app name); launched from elsewhere Chainlit silently regenerates a DEFAULT one
  next to the app, which is why `web/.chainlit/` is gitignored. Same
  graph, one `thread_id` per browser session. The patient-switch gate and the refinement offer
  are dialogs, the follow-up questions are buttons, and progress arrives as collapsible steps.
  `--headless` skips opening a browser; `--port N` moves it.
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
- **Corpus (run per corpus change, in this order; `python -m ingestion.<step>`):**
  `chunk_guidelines data/markdown -o <chunks>` → `contextualize` → `upload_to_qdrant_hybrid`.
  **Check before spending anything:** `python -m ingestion.chunk_guidelines data/markdown --check`
  runs the quality gates, writes nothing and exits non-zero if the corpus is not fit to index.
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
- **Measuring the SHIPPED system: `EVAL_TARGET=product PIPELINE=<mode> … evaluation.py`.** Runs
  the compiled graph instead of retriever+generate, and reports what only it can see: the
  no-answer rate by cause, the false-rejection rate of `validate` and the validator retry (see
  OPEN E). Dumps `results/ragas_results_<mode>_product.csv` (the answered rows, as usual) plus
  `results/product_runs_<mode>.csv` (one row per question, with its outcome). Slower than the
  retrieval path — it pays rephrase + assess + validate per question — so probe it the same way.
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
- **Phase 5 — Claude-style UX:** web (Chainlit), progress streaming, citations, multi-turn memory.
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
  - **Visible work (progress streaming): DONE (2026-07-29).** The CLI reads the run as a stream
    and shows the step in flight plus the guideline sections it read; the clinical text stays
    unstreamed on purpose. See the "Visible work" section in Architecture for the decision and
    the technical path if it is ever reconsidered.
    PENDING: the **web frontend (Chainlit)** — the two `interrupt()`s become a yes/no dialog and
    optional fields, the follow-up questions become clickable (the last open item of OPEN D),
    and the sources become collapsible elements; the structured `AnswerView` that web needs
    (`evidence.py` currently returns a 72-column terminal string, not data); and (increment 2)
    an "implicit knowledge modifiers" path in `assess` (flagged and always followed by
    re-retrieval + validate) as a safety net against retrieval failures.

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
- **OPEN D — multi-turn memory: DONE (2026-07-23).** The fix was SCOPING, not resetting.
  (1) `patient_facts` is session-scoped (accumulates across questions, feeds both retrieval via
  `_with_facts` and generation), visible with `/paciente` and cleared with `/nuevo`; the CLI is
  a REPL on one thread. (2) `refine` resolves an elliptical follow-up against `prev_question`
  (previous turn's question, session-scoped, cleared by `/nuevo`), so «¿y en embarazo?»
  retrieves as a standalone query — "previous question only" (the cheap 90%, not an N-window).
  (3) `refine` also flags `possible_new_patient` when the question CONTRADICTS the accumulated
  facts, and the **`confirm_patient` gate** pauses to confirm BEFORE answering (the deliberate
  exception to answer-first — a cross-patient recommendation is the harm; a generic answer is
  not). Only an explicit «sí» clears the data; Enter keeps the patient. **Still pending:** the 3
  follow-up questions the system emits are not clickable.
- **OPEN E — measuring the shipped system: the PATH exists now (2026-07-30), the full numbers
  do not.** `build_dataset` still calls retriever + `generate_answer` directly, skipping
  `refine`, `assess`, `re_retrieve`, the `validate` loop and `evidence` — that path stays,
  because isolating retrieval is what an A/B between modes needs. Alongside it,
  **`EVAL_TARGET=product`** runs the COMPILED GRAPH (`run_product`), one FRESH thread per
  question (reusing one would leak the previous patient's facts and measure a conversation
  nobody had) and declining every pause. What it reports that nothing did before:
  - **the no-answer rate** — how often a question the guidelines DO cover ends with no clinical
    content, split by cause because each needs a different fix: `insufficient` (the model said
    the context does not answer it), `not_validated` (an answer existed and the judge threw it
    away), `validation_error`, `technical_error`. This is the failure the VHB question hit.
  - **the false-rejection rate** — of the answers `validate` discarded, how many actually agreed
    with the reference. Judged against the REFERENCE, not by re-running the same judge, which
    would only measure its own flakiness. It is the pipeline's most dangerous failure and the
    only one with no visible symptom.
  - **the validator retry**, measured at last: how many regenerations ended answered, and how
    many of them went back for evidence (`refocus_retrieve`). Reasoned about since 2026-07-22,
    never measured.
  RAGAS scores only the ANSWERED rows and the coverage is reported apart, on purpose: a high
  faithfulness over a third of the questions is not a good system. A run that answers NOTHING is
  reported, not aborted — that is the loudest possible result, and the records are what say so.
  **Correction to a long-standing assumption: the `adversarial` tier does NOT measure
  abstention.** Its 30 questions ARE answerable from the guidelines; they are negation,
  over-generalization and distractor traps («¿está indicada la PEP siempre que hay un
  pinchazo?» → «no siempre, cuando…»). The right answer is the nuance, not «no está en las
  guías», so an abstention metric over that tier would have scored the correct behaviour as
  failure.
  **First numbers (stratified probe, 5 per tier = 20 questions, graph mode, 2026-07-30):**
  18/20 answered (90%); 1 `insufficient` and 1 `technical_error`, BOTH in the `adversarial`
  tier — every other tier answered 1.0. Re-running the technical one answered fine in 25 s, so
  it was a transient service blip, not a systematic failure; that is exactly why the record
  carries `failed_step`/`technical_detail` now. 0 rejections, so the false-rejection rate has
  no data yet — it needs the full run.
  **Latency is the surprise: mean 34.4 s, median 34.6 s, max 50.8 s** — the "12-25 s" quoted
  elsewhere in this file measured retrieval + one generation, not the shipped path, which also
  pays rephrase, assess (gpt-4o) and validate (gpt-4o). Per tier the ordering is not the
  expected one either: `simple` is the SLOWEST (44.3 s mean) and `single_hop` the fastest
  (24.5 s). Unexplained; worth a look before optimizing anything.
  **Pending:** the full 151-question run, which must happen AFTER the reranker window fix —
  every number moved. Use `EVAL_RAGAS=0` for it: the product metrics need no judge, and that is
  what makes running the shipped system over the whole set affordable.
- **OPEN F — the A/B is confounded and more expensive than it needs to be.** `house_tail` mixes
  every graph mode's selection with the same dense+BM25 complement, so the four modes share a
  good part of their final context and the measured deltas are compressed; an ablation WITHOUT
  the complement is what isolates the selection mechanism. And since only retrieval varies,
  restoring a retrieval-only path (recall/precision, no gpt-4o generation) would run all five
  modes over the 151 questions for a fraction of the current cost.
- **OPEN G — rich metadata: the PROMPT half is DONE (2026-07-30), the FILTERING half is not.**
  The concrete defect was worse than "unused metadata": `build_context` emitted `[1] {text}` and
  nothing else, so **the guide and the year never reached the model at all**, while `SYS_PROMPT`
  rule 5 asked it to arbitrate between contradictory fragments — across SEVEN guides spanning
  2013 to 2025 — whose dates it could not see. Rule 8 even forbade mentioning years. The
  metadata did reach the doctor, but only afterwards, in the sources panel.
  Fixed: each fragment is now headed by «Guía (año)», in the SAME shape `evidence.py` renders,
  and only in the prompt string (the payload is untouched, so `evidence.attribute` still matches
  quotes against the literal text and a quote of the header degrades to a `miss`). Rule 5 now
  RESOLVES the conflict instead of shrugging: a guide **specific** to the situation prevails over
  a general one even if older (the pregnancy and TB guides are 2018, TAR is 2022 — "most recent
  wins" alone would have been clinically wrong), and at equal scope the most recent wins and the
  older one is named as superseded. Rule 8 gained the matching exception; rule 9 declares the
  header non-citable. Pinned in `tests/test_rag_contracts.py`.
  **Still open:** nothing filters or boosts by `year`/`topic` in Qdrant, though the payload
  indexes exist. Deliberately not done — some topics only exist in the older guides and a filter
  would erase them; disambiguating at generation is visible and reversible. Revisit if the
  measurement (OPEN E) shows the model mis-picking versions.
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

0 bis. **CORPUS REWRITE — IN PROGRESS (started 2026-07-31). Read this before running any A/B.**
   An audit of the corpus found the damage is mostly UPSTREAM of everything measured so far, in
   the PDF→Markdown conversion, which does not exist as versioned code (each of the 7 PDFs was
   converted by an ad-hoc script, guided by `data/prompt.txt` and then discarded). Measured:
   **114 omission markers** in 27 spellings; TAR_2022's `9. TABLAS` chapter (20% of the document)
   holds **0 table rows across 13 tables**, including TABLA 3 (first-line regimens) and TABLA 9
   (the interaction matrix); **91 lost fi/fl ligatures**, among them `fuconazol` for
   `fluconazol` in dosing text, which no lexical search can find; **212 evidence grades injected
   mid-sentence**, splitting 34 words and, in one case in the NC guide, inverting the clinical
   meaning; **1443 `<br>`** standing in for table structure. And `chunks.jsonl` is stale against
   its own chunker (517 vs 559; only 198 ids match) and was built with a characters/4 estimate,
   so **89 chunks (17%) exceed the token budget** the chunker claims to enforce.
   **The finding that made this tractable: none of it needs a VLM.** The lost content is IN THE
   PDF's TEXT LAYER — verified in `data/textos/TAR_2022/TAR_2022.txt`, which holds TABLA 3 and
   TABLA 9 in full. They are VECTOR grids, not raster images; `ignore_graphics=True` masked them
   as pictures, and VIH_embarazo's tables were rejected by a heuristic reading a ROWSPAN
   ("primera columna vacía en 71% de las filas") as a broken layout.
   **Decided with the user (2026-07-31):** re-extract all 7 now (the A/B numbers were already
   invalidated by the reranker-window fix, so this is the cheapest moment); **`pdfplumber`
   (MIT)**, not PyMuPDF (AGPL-3.0, and this repo is MIT and public); the ~35 graphical figures
   and decision algorithms stay OMITTED with a normalized marker, because **no LLM may ever touch
   the extraction path** — that is what keeps everything citable a verifiable transcription.
   **Status:** Phase 0 DONE (quality gates + the `corpus.py` generation switch) and **Phase 1
   DONE** (manifest, specialty profiles, per-specialty prompts, `Scope` filtering end to end,
   CLI `--specialty` / `/especialidad` and a Chainlit chat profile per specialty — see Key
   files). Next: the extractor, then the chunker. The migration sequence, the per-table safety
   gates and the three sites that assume the breadcrumb lives inside `text` are in the approved
   plan. **Nothing has been re-indexed yet: `CORPUS_VERSION` is still `v1` and the app answers
   exactly as before.**
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
5. **Phase 5 — UX.** Interactive clarification, enriched re-retrieval and **progress streaming**
   DONE; continue with the **web frontend (Chainlit, decided 2026-07-29)** and concept
   navigation. Increment 2 pending: "implicit knowledge modifiers" path in `assess` (flagged
   and always followed by re-retrieval + validate).

Evaluation artifacts versioned in `results/`: `ragas_results.csv` (Phase-0 baseline) and
`ragas_results_{baseline,iterative,graph}_retrieval.csv` (Phase-4 A/B, context_recall). The
new A/B over `EVAL_SET` dumps `results/ragas_results_<PIPELINE>.csv` (full RAGAS, now with a
`tier` column so a run can be re-sliced by question type afterwards).
`ragas_results_{graph,pathrag,hipporag}.csv` are the **12-question probe** of 2026-07-22, NOT
the full A/B — read them as such (n=3 per tier).

## Important findings (do not lose)

- **`Command(resume={})` DOES NOT RESUME an `interrupt()` (measured 2026-07-29).** Empirical
  rule on langgraph 1.2.5, tested value by value against the real graph: `""`, `0`, `False` and
  any NON-empty dict all resume correctly; **an empty dict is the one value LangGraph does not
  recognise as an answer**, so the interrupt is raised again — with the state never written,
  which means the round budget is never spent and the pause repeats indefinitely. `None` raises.
  This shipped as a live bug: `CLARIFY_QUESTIONS_PER_ROUND` is 3, the CLI encoded the answers as
  `{question: answer}` dropping the blanks, and pressing Enter on all three (the documented way
  to decline) produced `{}` — so **declining a refinement was impossible**, the same three
  questions came back forever. Fixed in `pipeline.nodes.refinement_reply`, which is now the ONE
  place any frontend shapes a resume value (the CLI and the web must not each invent an
  encoding), and pinned against the REAL graph in `tests/test_pipeline_flow.py` — the fake graph
  in `test_cli.py` resumes on anything, which is exactly why it never caught this.

- **THE BRIDGE BACK TO CITABLE PAYLOADS COULD RETURN THE WRONG CHUNK (found and fixed
  2026-07-30).** `_common.map_to_payloads` resolves an index's selected chunks by exact
  normalized text, with a fallback keyed on the first 120 characters — built as a dict
  comprehension, so the LAST chunk per key won. But our chunks open with their `A > B > C`
  breadcrumb, and **176 of the 517 share that opening with a sibling** (42 groups; the largest
  is a whole guideline whose TITLE alone outruns 120 chars, so **all 45 of its chunks collide**).
  A fallback lookup for any of them returned a DIFFERENT chunk — the sources panel would quote
  the wrong section under the right claim, which is exactly the harm `evidence.py` spends its
  fuzzy-matching to prevent. **It was dormant, not harmless:** chunks.jsonl and the LightRAG
  store hold byte-identical text today, so the exact match takes every lookup and the fallback
  never fires. Editing a chunk without rebuilding the store wakes it — demonstrated by patching
  4 chunk texts and replaying the lookup against the store's old text: **4 of 4 resolved to the
  wrong `chunk_id`**. **Fix: an ambiguous opening is dropped from the fallback index**
  (`get_prefix_lookup`), so it answers only when the prefix names one chunk. Same reasoning as
  `evidence.attribute` resolving every doubt to `miss` — a miss costs the answer one chunk, a
  wrong hit costs it a mis-citation. Pinned in `tests/test_retrieval_common.py` (the old code
  returns the sibling on that test; the new one returns nothing).
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
- **THE RERANKER WAS BEING SHOWN THE WRONG 512 CHARACTERS (found and fixed 2026-07-29).** The
  latency fix below scored `p["text"][:512]`, i.e. the chunk's PREFIX. But **89% of our chunks
  are longer than that** (median 1616 chars, p90 3456, max 25 329) and a guideline section opens
  with context before it recommends anything — so the cross-encoder was routinely judging a
  preamble. Measured on «¿Qué pauta de TAR en coinfección por VHB?»: `TAR_2022 §7.4.4
  HEPATOPATÍAS`, the section that literally holds the answer («iniciar precozmente un TAR que
  incluya TDF o TAF y FTC o 3TC (A-I)», at char 1802 of 2508), reached the candidate pool through
  BOTH branches (graph traversal AND dense+BM25 — both retrievers agreed) and then scored
  **-0.900** on its prefix, ranking **10th of 25**, below four TUBERCULOSIS chunks. `top_k=8` cut
  it, and the doctor was told the guidelines did not cover it. The same chunk scores **+0.880**
  in full. **Widening the window is not the fix:** 2048 chars found it but took **9.4 s** instead
  of 0.55 s (the batch pads to the longest item, so cost is superlinear), and 1024 did not find
  it at all. **The fix is choosing WHERE the window goes** — `rag.score_window` picks the slice
  with the most distinct query terms, same 512-char budget: the answer comes back at **rank 1**
  in 467 ms (vs 556 ms for the prefix). Across 5 queries latency is unchanged and the top-8
  shifts by 0-3 chunks. **So the reranker was not the weak component — it was blindfolded.**
  Two consequences: (a) **every pending A/B number will move**, so measurement (OPEN E) has to
  run after this, not before; (b) the sub-bug this exposed — filtering query terms by LENGTH
  (`len > 3`) silently discarded TDF, TAF, FTC, 3TC, DTG, BIC, VHB and TAR, the most
  discriminative tokens the guides have — is why `score_window` uses a Spanish stopword list
  instead of a length floor. Pinned in `tests/test_retrieval_common.py`.
- **ABSTAINING ON A LOW RERANKER SCORE: MEASURED AND REJECTED (2026-07-30).** The idea was a
  safety net for priority #1: if the best chunk scores low, the guides probably do not cover the
  question, so say so instead of generating. It does not survive contact with the data.
  Measured over 40 questions (10 per tier) that the guides DEMONSTRABLY answer — every EVAL_SET
  question has a reference — the top-1 score spans **-0.828 to +2.060**, median +0.520. So even
  the most permissive threshold tried, -0.5, would silence **10% of answerable questions**; 0.0
  silences 25%; +0.2 silences 38%. Per tier the floor is low everywhere (`single_hop` -0.828,
  `adversarial` -0.664, `multihop` -0.511) — this is not one outlier.
  Worse, the benefit cannot be measured at all: **the eval set has no negative class**. Every
  question is covered, so there is no way to know how many genuinely uncovered ones a threshold
  would catch. A known cost against an unmeasurable benefit is not a trade, and shipping the
  threshold would have converted a safety net into a generator of false «no está en las guías» —
  the same invisible failure the whole system is built to avoid.
  Third argument, from the reranker finding above: the score is not stable enough to carry a
  decision. The SAME chunk scored -0.900 and +0.880 depending only on which 512-char window it
  was shown. A signal that swings 1.8 points on a windowing detail cannot gate an answer.
  **What would unblock it:** a labelled set of IN-DOMAIN questions this corpus does not cover
  (out-of-domain ones are already stopped by the `refine` guardrail). That is a dataset task
  needing clinical input, not a code task. Until it exists, abstention stays where it already
  works on CONTENT rather than on a scalar: the generator's `sufficient_information` (which
  correctly declined the VIH-2/efavirenz trap in the probe) and `validate`.
  `rag.rerank` was deliberately NOT changed to expose its scores: there is nothing to consume
  them.
- **A displayed «literal citation» can be ungrammatical because THE CORPUS is (found
  2026-07-29).** In `TAR_2022.md` some evidence-grade markers landed MID-SENTENCE during the
  PDF→Markdown transcription: `…se debe evitar la interrupción_ _**(A-II).** de una pauta eficaz
  frente a VHB_`. `evidence.py` quotes the stored text faithfully, so the defect is upstream, in
  `data/markdown/`. Scope not yet quantified; do NOT "fix" it in the citation logic.

## Optimizations / known issues

- **RERANKER LATENCY (RESOLVED, Phase 4):** the rerank took **~128 s** (95% of the total
  time) because the cross-encoder pads the batch to the length of the longest chunk on CPU.
  **Fix applied in `rag.rerank`: only 512 chars of each chunk are scored (constant
  `RERANK_SCORE_CHARS`) and the full payloads are returned.** Which 512 matters — it was the
  prefix until 2026-07-29 and that silently lost answers; see the reranker finding above. A multi-hop query dropped from
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
  `RunConfig(timeout=600, max_workers=8, max_retries=10)`. There is no cheap retrieval-only
  path any more (`RETRIEVAL_ONLY`/`RECALL_ONLY` were removed); every run pays full gpt-4o
  generation, which is what makes the 5-mode A/B expensive — see OPEN F.
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
