# CLAUDE.md — Chatbot médico VIH (RAG sobre guías GeSIDA)

Contexto del proyecto para retomarlo en cualquier conversación nueva. Mantener
este archivo al día cuando cambien decisiones, estado o hallazgos importantes.

## Qué es

Chatbot RAG para **médicos** que responde preguntas sobre las **guías clínicas de
VIH de GeSIDA** (7 PDFs, en español). Naturaleza actual: **prototipo para demostrar**
(aún sin usuarios reales). Prioridades, por orden: (1) **reducir alucinaciones**,
(2) **conectar conceptos abstractos** para navegar las guías, (3) **UX tipo Claude**.

Cumplimiento: por RGPD, todo modelo/servicio debe ser privado o local y, a ser posible,
en región UE. Los datos del médico pueden incluir información de salud (Art. 9 RGPD).

## Arquitectura actual (LangGraph)

Punto de entrada: `main.py` (grafo compilado `app`). Nodos del grafo:

```
question ─▶ rephrase ─┬─ fuera de dominio ─▶ out_of_domain ─▶ END
                      └─ en dominio ─▶ [RETRIEVAL_MODE] ─▶ generate ⇄ validate ─▶ evidence ─▶ output
                             ├─ baseline : retrieve (híbrido) ─▶ rerank (20→5)        (no válido+agotado
                             ├─ iterative: iterative_retrieve (plan→hop→reflect, →8)    ─▶ fallback)
                             └─ graph    : graph_retrieve (LightRAG + híbrido propio, →8)  ◀ DEFECTO
```

La cabecera (rephrase + guardia de dominio) y la cola (generate→validate→evidence) son
IDÉNTICAS en los tres modos; solo cambia el nodo de recuperación, elegido por
`RETRIEVAL_MODE` en `main.py` (por defecto `"graph"` tras el A/B de F4).

- **rephrase** (`rag.refine`, gpt-4o-mini): una llamada que (a) clasifica si la pregunta
  es del dominio VIH (`in_domain`) y (b) reescribe la consulta SIN añadir info, normalizando
  términos en AMBAS formas «nombre completo (SIGLA)» con `abbreviations.py`. Si está fuera
  de dominio → `out_of_domain` (mensaje directo, corta el pipeline). La generación usa la
  pregunta ORIGINAL, no la reescrita.
- **Recuperación — 3 modos seleccionables (`RETRIEVAL_MODE`), todos terminan en el mismo generate:**
  - **baseline** = `node_retrieve` (`rag.retrieve_hybrid`: densa text-embedding-3-large 3072d +
    sparse BM25 `Qdrant/bm25`, fusión RRF, 20 candidatos) → `node_rerank` (`rag.rerank`,
    cross-encoder local, 20→5).
  - **iterative** (Track A) = `node_iterative_retrieve` → `agentic.iterative.iterative_search`
    (plan→hop→reflect, top 8). Usa la pregunta ORIGINAL (su propio plan); ignora la reescrita.
  - **graph** (Track B, POR DEFECTO) = `node_graph_retrieve` → `graph.lightrag_track.graph_search`
    (grafo entidad+relación `mode="hybrid"` + complemento `retrieve_hybrid` densa+BM25 →
    fusión dedup → rerank → top 8).
- **rerank** (`rag.rerank`): cross-encoder local `jinaai/jina-reranker-v2-base-multilingual`
  (fastembed/ONNX, multilingüe, RGPD-ok). Lo usan los tres modos para afinar al top final.
- **generate** (`main._structured_llm`, gpt-4o): salida estructurada con
  `ChatOpenAI.with_structured_output` (Pydantic `ClinicalAnswer`, json_schema estricto).
  Devuelve dict: sufficient_information, answer, sources_used[{ref,quote}], follow_up_questions.
- **validate** (`rag.validate`, gpt-4o-mini): juez de relevancia + grounding semántico.
  Bucle con `generate` (inyecta feedback al reintentar), `MAX_ITER=2`. válido → evidence;
  no válido y agotado → `fallback`; error técnico del juez → `fallback` (no se "falla abierto").
- **evidence** (`evidence.format_answer`): formatea respuesta + panel de fuentes con
  citas literales (fuzzy match) + preguntas de seguimiento + aviso clínico. Texto sin ANSI.

`rag.search()` = rephrase → híbrido → rerank (lo usa la evaluación).

### Recuperación por grafo (LightRAG): qué se usa y qué NO

Indexado (una vez, `graph._build_index`): por cada chunk, gpt-4o-mini extrae **entidades** y
**relaciones** con **descripción generada por el LLM**; se guardan en `lightrag_store/`:
`kv_store_full_entities/relations.json` (descripción + chunks de origen `source_id`),
`vdb_entities/relationships/chunks.json` (embeddings para casar), `kv_store_text_chunks.json`
(texto literal) y `graph_chunk_entity_relation.graphml` (topología).

Consulta (`graph_search`, modo `hybrid`): (1) LightRAG saca keywords high/low-level de la
pregunta; (2) low-level → vdb_entities → entidades; high-level → vdb_relationships →
relaciones; (3) recorre el grafo (vecinos) y, vía `source_id`, junta los **chunks de origen**
(hasta `chunk_top_k=20`), mapeados a `chunks.jsonl` con `_map_to_payloads` (match exacto +
fallback por prefijo; en prueba 20/20). (4) **Complemento (sustituye a la densa interna de
LightRAG):** se añaden los chunks de NUESTRA `retrieve_hybrid` (densa + BM25 RRF) sobre la
consulta reescrita — el BM25 ayuda con siglas/dosis. (5) Se **fusiona** (dedup por
`chunk_id`, grafo primero) y el **reranker** afina a top 8.

**Decisión de diseño (clave):** las descripciones de entidades/relaciones SÍ influyen en
la **selección** (su embedding es lo que casa con la pregunta), pero se **descartan para la
generación**: a gpt-4o solo le pasamos los **chunks literales**. Motivo = prioridad nº1
(no alucinar): las descripciones son texto sintético del LLM, no citable y con posibles
errores de extracción; nuestro `evidence.py` exige cita literal verificable. **Idea abierta
(A/B-able):** pasar las descripciones de *relaciones* como bloque "mapa de conceptos, NO
citable" para ayudar al razonamiento multi-hop, manteniendo el grounding estricto + validate.

## Ficheros clave

- `main.py` — grafo LangGraph + LangSmith + generación estructurada (punto de entrada).
- `rag.py` — pipeline: clientes, embeddings, retrieve/retrieve_hybrid, rerank, refine,
  search, validate, generate_answer (versión cruda), SYS_PROMPT, build_user_prompt, constantes de modelo.
- `evidence.py` — formateo de respuesta y fuentes.
- `evaluation.py` — evaluación RAGAS. Dos sets (`GOLDEN_SET` 47 single-hop;
  `MULTIHOP_SET` 16 multi-hop) y A/B: selector `PIPELINE` (baseline/iterative/graph, vía
  env) + `DATASET`. Modos: completo (faithfulness+precision+recall) o lean por env
  `RETRIEVAL_ONLY=1` (sin generación gpt-4o; precision+recall) / `RECALL_ONLY=1` (solo
  recall). `RunConfig(timeout=600, max_workers=8)` anti-timeout. Mide latencia y vuelca
  `resultados_ragas_<PIPELINE>[_retrieval].csv`.
- `abbreviations.py` — diccionario SIGLA→nombre de las guías (valores en español).
- **`agentic/`** — Track A (F4). `iterative.py`: `iterative_search` (plan/hop/reflect).
  Importa los primitivos compartidos de `rag.py` (carpeta padre).
- **`graph/`** — Track B (F4). `lightrag_track.py`: build del grafo + `graph_search`
  (importa `rerank` de `rag.py`, corpus de `chunks/`, store en `lightrag_store/`).
- `chunks/` — `chunk_guias.py` (chunking estructural), `subir_a_qdrant.py` (denso),
  `subir_a_qdrant_hibrido.py` (denso+BM25), `chunks.jsonl` (517 chunks).
- `markdown/` — las 7 guías en Markdown (fuente del corpus). `pdfs/`, `textos/` — originales.
- `langgraph.json` — config de LangGraph Studio (expone `main.py:app`).

## Modelos y servicios

- Generación: **gpt-4o** (`GENERATION_MODEL`). Rephrase/validación: **gpt-4o-mini**
  (`REPHRASE_MODEL` / `VALIDATION_MODEL`).
- Embeddings: `text-embedding-3-large` (3072d). Reranker: jina-reranker-v2-base-multilingual.
- Qdrant Cloud (región **eu-west**). Colecciones: `guias_vih` (solo denso, respaldo) y
  **`guias_vih_hibrida`** (denso + sparse BM25, la activa).
- LangSmith en región **UE**: `.env` con `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`,
  proyecto **`chatbot_vih`** (guion bajo). Trazado se autoactiva si hay `LANGSMITH_API_KEY`.

## Cómo ejecutar

- App (CLI interactivo): `.venv\Scripts\python.exe main.py`
- LangGraph Studio: `.venv\Scripts\langgraph.exe dev` → abre Studio (UE). Ver pasos en
  pestaña **Trace View** (no Turn View).
- **Elegir la estrategia de recuperación (F4):** `main.py` compila CUATRO grafos y
  `langgraph.json` los registra para que Studio muestre un **selector de grafo**:
  - **Grafos dedicados** `app_baseline` / `app_iterative` / `app_graph` (build_graph(mode)):
    cada uno contiene SOLO su arquitectura, con la recuperación **EXPANDIDA en sus nodos
    reales** (vista didáctica) en vez de esconderla tras un único `retrieve`:
      - baseline: `retrieve → rerank`
      - iterative: `iter_plan → (iter_single | iter_hop ⇄ iter_reflect) → iter_rerank`
        (loop self-ask, tope `MAX_HOPS`; single-hop cae a `iter_single` = one-shot baseline)
      - graph: `graph_traverse → graph_hybrid → graph_merge → graph_rerank`
    Cada nodo reusa las MISMAS primitivas que `iterative_search`/`graph_search` (sin cambio
    de comportamiento) y los estados intermedios se ven en Studio (`IterativeState`/
    `GraphState` extienden `RAGState`). Selección: desplegable de grafos de Studio.
  - **Grafo combinado** `app` (build_combined_graph): las tres rutas en un grafo; expone un
    `context_schema` (`ConfigSchema`) con el campo **`retrieval_mode`** como **desplegable**
    en el panel de config del run (elección en vivo). Lo usa también el CLI.
  - **Head/tail compartidos** se factorizan en `_add_common`; la sección de retrieval en
    `_add_retrieval(mode)`. Esto es lo que hace el pipeline AGNÓSTICO al retrieval: cada nodo
    de retrieval cumple el MISMO contrato de estado (rellena `contexts`/`chunk_index`/
    `formatted_context`) y el tail solo lee ese contrato, nunca nada específico del modo.
  - **Por env / CLI:** `RETRIEVAL_MODE` (env var, por defecto `"graph"`) es el modo por
    defecto del combinado; o `python main.py iterative` para forzar uno en un run.
  - **Resolución de modo (combinado), en `_resolve_mode`:** context (Studio) → `configurable`
    → `RETRIEVAL_MODE`. Valores desconocidos caen al default sin romper.
  - **Trazas:** el modo usado se registra en el estado (`retrieval_mode`) y, desde el CLI,
    como tag `mode:<x>` y metadata → filtrables en LangSmith. La llamada LLM interna de
    keywords de LightRAG SÍ se traza ahora (envuelta con `traceable` en
    `_make_rag(trace_llm=True)`, solo en consulta, no en el build del índice).
- **Construir el grafo LightRAG (una vez):** `.venv\Scripts\python.exe -m graph.lightrag_track`
  (extracción de entidades sobre los 517 chunks; reanudable, usa caché de LLM).
- Evaluación RAGAS / A/B: ajustar `PIPELINE` y `DATASET` en `evaluation.py` y correr
  `.venv\Scripts\python.exe evaluation.py`.
- **Dependencias: SIEMPRE `uv add` (nunca pip).** `uv` no está en PATH:
  `& "$env:USERPROFILE\.local\bin\uv.exe" add <paquete>`.

## Roadmap por fases

Orden acordado: medir → orquestar → retrieval barato → refine+validate → grafo (si hace falta) → UX.

- **F0 — Brújula: HECHA.** Golden set RAGAS con referencias (las redactó el modelo, no un
  médico → caveat). Baseline denso (juez gpt-4o-mini): faithfulness 0.81,
  answer_relevancy 0.58 (ENGAÑOSO, artefacto de la métrica en español), context_precision
  0.97, context_recall 0.94 (algo inflado). Detalle en `resultados_ragas.csv`.
- **F1 — LangGraph + LangSmith: HECHA.**
- **F2 — Retrieval barato: HECHA.** Híbrido (2a) + reranker (2b).
- **F3 — Refine: HECHA.** rephrasing + normalización de abreviaturas; **guardrail de
  dominio** (la clasificación `in_domain` del nodo rephrase corta las preguntas fuera de
  tema en ~2 s, ahorrando retrieve/rerank/generate/validate); y el **bloque Validate**
  (nodo `validate` tras `generate`, juez gpt-4o-mini de relevancia + grounding semántico;
  bucle con reintento inyectando feedback, `MAX_ITER=2`; si se agota → `fallback` con
  mensaje seguro; error técnico del juez → `fallback`, sin "fallar abierto"). El validador
  NO re-chequea citas literales (eso ya lo hace `evidence`). La clasificación de tipo de
  pregunta (vector vs grafo) se difiere a F4.
- **F4 — Multi-hop: DECIDIDA (grafo LightRAG por defecto).** Dos vías para preguntas
  multi-salto, comparadas con un A/B sobre `MULTIHOP_SET` (16 preguntas, en `evaluation.py`):
  - **Track A — agéntico/iterativo** (`agentic/iterative.py`): bucle self-ask
    plan → recuperar por sub-consulta → reflect (MAX_HOPS=3), reusa hybrid+rerank, cero
    reindexado. HECHO y probado.
  - **Track B — grafo LightRAG** (`graph/lightrag_track.py`): grafo entidad-relación en
    almacenamiento de ficheros (`lightrag_store/`), recupera chunks por traversía y los
    mapea a nuestros payloads (conserva citas). Índice CONSTRUIDO (517 chunks, ~1.5 h).
  - Decisión por el A/B (calidad multi-hop primero; luego velocidad/coste/actualización).
    Ambas vías son seleccionables en el grafo vía `RETRIEVAL_MODE` y en la eval vía
    `PIPELINE`, y terminan en el MISMO generate→validate→evidence.
  - **RESULTADO A/B (16 multi-hop, context_recall, juez gpt-4o-mini): graph 0.979 >>
    iterative 0.863 > baseline 0.844.** Latencia: graph ~11 s ≈ baseline, iterative ~24 s.
    **DECISIÓN: graph (LightRAG) por defecto** (`RETRIEVAL_MODE="graph"`). Caveats: solo se
    midió context_recall limpio (context_precision y faithfulness se cayeron por timeouts
    del juez con muchos workers → NaN; pendiente medir precision del ganador a baja
    concurrencia); n=16, referencias del modelo. El recall alto del grafo podría venir con
    algo menos de precisión (recupera más amplio), mitigado por reranker→top8 + validate.
- **F5 — UX tipo Claude:** Streamlit/web, streaming, citaciones, memoria multi-turno.

## Pendiente / próximos pasos (a fecha 2026-06-22)

1. **Medir `context_precision` (y faithfulness) del grafo** a baja concurrencia
   (`RETRIEVAL_ONLY=1`, `max_workers≈4-8`, `timeout=600`) para cerrar el cuadro del A/B:
   el recall del grafo es altísimo (0.979) y conviene confirmar que la precisión no cae
   demasiado (recupera amplio). Reusar baseline (recall 0.844) ya medido.
2. **(Idea) Descripciones de relaciones como "mapa de conceptos" no citable** en el prompt
   de generación del grafo, para ayudar al razonamiento multi-hop sin romper el grounding.
   Prototipar tras un flag y A/B contra la versión actual (faithfulness + recall).
3. **F5 — UX.** Tras cerrar (1).

Artefactos de evaluación versionados: `resultados_ragas.csv` (baseline F0) y
`resultados_ragas_{baseline,iterative,graph}_retrieval.csv` (A/B F4, context_recall).

## Hallazgos importantes (no perder)

- **Abreviaturas (resuelto en gran parte):** el corpus usa sobre todo siglas (DTG 144 vs
  "dolutegravir" 23, BIC 54 vs 8...) pero también nombres completos. Por eso el rephrase
  incluye AMBAS formas y el glosario está también en SYS_PROMPT (son parte de las guías,
  no conocimiento externo). El problema era de 3 capas: recuperación (rephrase), comprensión
  del generador (glosario) y capacidad del modelo.
- **Modelo de generación:** gpt-4o-mini era el cuello de botella de calidad — fallaba o
  respondía MAL en casos con matices/abreviaturas aun teniendo la evidencia. **gpt-4o lo
  resuelve.** Por eso la generación está en gpt-4o.
- **El reranker mejora precisión/orden pero NO subió el recall@5** en el set de prueba
  (6/8 igual que el híbrido). Útil pero no imprescindible.

## Optimizaciones / problemas conocidos

- **LATENCIA DEL RERANKER (RESUELTO, F4):** el rerank tardaba **~128 s** (95% del tiempo
  total) porque el cross-encoder rellena (padding) el lote a la longitud del chunk más
  largo en CPU. **Fix aplicado en `rag.rerank`: se puntúa solo `p["text"][:512]`
  (constante `RERANK_SCORE_CHARS`) y se devuelven los payloads completos.** Una consulta
  multi-hop bajó de ~145 s a ~23 s (en frío). Crítico para Track A, que rerankea varias
  veces. Equipo: 12 cores, sin GPU (con GPU NVIDIA, onnxruntime-gpu aceleraría 10-50×).
- Modelos locales (BM25, reranker) con carga perezosa: la 1ª consulta del proceso paga
  ~3.5 s de carga del reranker. Studio lo mantiene cargado entre consultas.
- generate (gpt-4o) ~5 s: considerar streaming para mejorar latencia percibida (F5).
- **LÍMITE OpenAI: gpt-4o a 30.000 TPM (bajo).** La generación NO se puede paralelizar:
  correr varios pipelines a la vez en la eval → `429 RateLimitError` y crash. Correr los
  pipelines en SECUENCIAL (la generación de `build_dataset` ya es secuencial → no peta).
- **RAGAS apenas paraleliza** (≈serial aunque subas `max_workers`) y `context_precision` es
  la métrica pesada/frágil (1 llamada de juez por chunk → con muchos workers da TimeoutError
  → NaN). `context_recall` es la ligera y robusta. Config estable: `RunConfig(timeout=600,
  max_workers=8, max_retries=10)`. Para A/B barato usar `RETRIEVAL_ONLY=1` (sin generación
  gpt-4o) o `RECALL_ONLY=1` (solo recall) en `evaluation.py`.
- **Build del grafo LightRAG:** cuello de botella = `max_parallel_insert` (default 2 → subido
  a 8 en `graph.lightrag_track`; también `llm_model_max_async=16`). Aun así ~1.5 h por las
  ~517 extracciones. Reanudable y con caché de LLM (re-correr es barato).
- **Suspensión del equipo:** dormir el portátil MATA las corridas largas (cae la conexión).
  Para jobs largos desactivar suspensión (`powercfg /change standby-timeout-ac/dc 0`) y
  restaurarla luego. (Pasó: una corrida nocturna murió al dormirse la máquina.)
- LangSmith + LangGraph Studio **verificados OK** esta sesión (Studio arranca y traza; el
  endpoint UE responde 204 al enviar metadata).

## Convenciones

- **Idioma del código: inglés** (identificadores, comentarios, docstrings). Se mantiene en
  español lo que se muestra al usuario (mensajes, etiquetas del panel), los prompts a los
  LLM, y el contenido/manejo de las guías y chunks (incluido `chunks/` y los valores de
  `abbreviations.py`). Al usuario se le habla en español.
- Toda llamada LLM encapsulada para poder cambiar a Azure OpenAI (GPT-4 privado) sin
  fricción el día que se requiera por cumplimiento.
- Commits directos a `main` (flujo de un solo dev). Mensajes en español. En `.gitignore`:
  `.env`, `*.log`, `.langgraph_api/` y `lightrag_store/` (el grafo se reconstruye con
  `python -m graph.lightrag_track`).
