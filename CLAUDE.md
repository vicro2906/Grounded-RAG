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
                      └─ en dominio ─▶ [RETRIEVAL_MODE] ─▶ assess_context ⇄ clarify (interrupt)
                             ├─ baseline : retrieve (híbrido) ─▶ rerank (20→5)        │ (al salir)
                             ├─ iterative: iterative_retrieve (plan→hop→reflect, →8)   ▼
                             └─ graph    : graph_retrieve (LightRAG + híbrido propio, →8) ◀ DEFECTO
                                              re_retrieve (1×) ─▶ generate ⇄ validate ─▶ evidence ─▶ output
                                                              (no válido+agotado ─▶ fallback)
```

La cabecera (rephrase + guardia de dominio) y la cola (assess_context→generate→validate→evidence)
son IDÉNTICAS en los tres modos; solo cambia el nodo de recuperación, elegido por
`RETRIEVAL_MODE` en `main.py` (por defecto `"graph"` tras el A/B de F4).

- **rephrase** (`rag.refine`, gpt-4o-mini): una llamada que (a) clasifica si la pregunta
  es del dominio VIH (`in_domain`); (b) reescribe la consulta SIN añadir info, normalizando
  términos en AMBAS formas «nombre completo (SIGLA)» con `abbreviations.py`; y (c) **criba**
  los datos del paciente ya presentes (`known_facts`→`clinical_facts`) y los modificadores
  clínicos que la pregunta podría necesitar (`candidate_modifiers`) — la mitad barata del paso
  de clarificación. Si está fuera de dominio → `out_of_domain` (mensaje directo, corta el
  pipeline). La generación usa la pregunta ORIGINAL, no la reescrita.
- **assess_context + clarify** (`rag.assess`, **gpt-4o** `ASSESS_MODEL`; nodos en `main.py`): **puerta de
  clarificación interactiva** (F5, slot-filling) entre la recuperación y `generate`. Razona en
  campos estructurados (orden = CoT) por DOS vías, con el **CONOCIMIENTO CLÍNICO como PRINCIPAL**:
  **(1) `clinically_relevant`** (principal): TODOS los modificadores que clínicamente cambian la
  respuesta a este tipo de pregunta, INDEPENDIENTE del contexto (exhaustiva dentro de lo
  pertinente) y **(2) `branches_on`** (complemento): dimensiones extra que el contexto recuperado
  condiciona. **Por qué conocimiento primero (decidido con el usuario):** el fallo del retriever
  (recall miss silencioso → respuesta genérica con aire de segura) es más probable Y más grave
  que un error del modelo de preguntas (visible, lo corrige el médico, acotado por `asked_questions`
  y el tope, y NUNCA corrompe la respuesta, que se re-recupera y valida). Solo la RESPUESTA se
  ancla en evidencia, no las PREGUNTAS. Resta `already_covered` (lo que `clinical_facts` ya fija en
  cualquier unidad + lo ya preguntado) y emite `questions` ordenadas por **impacto clínico** (no por
  vía), una por dimensión pendiente. Razonó inconsistente con gpt-4o-mini (se saltaba el CoT) → se
  usa gpt-4o (`ASSESS_MODEL`); cuesta más (corre hasta 3× por pregunta) pero un mejor modelo extrae
  más detalle relevante. El razonamiento se expone en el estado (`assessment`) para la traza. Es
  seguro porque el dato NO se cita: solo dirige la rama Y re-dispara la recuperación
  (`re_retrieve`), de modo que la evidencia condicional se recupera antes de generar y `validate`
  sigue exigiendo grounding. Si hay preguntas → `clarify` **pausa**
  el grafo con `interrupt()` y pregunta al médico; al reanudar, `clarify` funde la respuesta en
  `clinical_facts` (merge a mano) y suma 1 a `clarify_rounds`. Dos cotas: `CLARIFY_MAX_ROUNDS`
  (=3, nº de pausas) y `CLARIFY_QUESTIONS_PER_ROUND` (=1, preguntas por pausa) → por defecto se
  pregunta UNA cosa cada vez, como mucho 3 veces. `assess` ordena las pendientes por **impacto
  clínico** y se toman las `max_questions` primeras.
  **`clinical_facts`/`clarify_rounds` NO llevan reducer**: se REINICIAN por pregunta en
  `node_rephrase` (Studio persiste el estado del thread; con reducers acumulativos los datos del
  paciente anterior y el presupuesto de rondas gastado se filtraban a la siguiente pregunta y
  `assess_context` se saltaba). Dentro de UNA pregunta el merge/incremento lo hace `clarify` a
  mano (lectura del estado), suficiente porque se escriben en secuencia.
- **re_retrieve** (nodo en `main.py`): corre **UNA sola vez, al SALIR del bucle de clarificación**
  (justo antes de `generate`), no en cada ronda. **Re-recupera con TODOS los `clinical_facts`
  inyectados en la consulta** (despacha según `retrieval_mode`, reusa las funciones colapsadas
  baseline/iterative/graph) y SOBRESCRIBE `contexts` → así el dato del médico TIRA de los pasajes
  condicionales (rama de VHB, de primer trimestre…) para que `generate` los CITE, no solo dirige
  la generación. No-op si no hay datos (p. ej. pregunta que no clarifica) → deja el contexto
  inicial intacto. **Total: 1 retrieve inicial + 1 re_retrieve final** (antes era 1 + N por ronda).
  El bucle `assess_context ⇄ clarify` corre todo sobre el **contexto inicial**: como `assess` es
  primario en conocimiento, no necesita re-recuperar entre rondas (los re_retrieve intermedios
  eran redundantes). Probado: con "coinfección VHB" entran al top-5 los chunks específicos de VHB.
  Los `clinical_facts` entran ADEMÁS en `generate` como bloque
  **"DATOS APORTADOS POR EL MÉDICO" NO citable** (seleccionan la rama de la guía; las citas
  literales siguen saliendo de los chunks). Requiere checkpointer: lo provee `langgraph dev`/Studio
  (no se compila uno propio en los grafos).
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
- `evaluation.py` — evaluación RAGAS. **UN ÚNICO set `EVAL_SET` (151 preguntas)** con campo
  `tier` por pregunta (**simple / single_hop / multihop / adversarial**) → mide performance
  POR TIPO de pregunta. Se construye fundiendo los antiguos pools (golden + multihop, ahora
  componentes internos `_PREV_SINGLE`/`_PREV_MULTI`) con un bloque tiered nuevo
  (`_TIERED_NEW`); ya NO hay selector de dataset. A/B con `PIPELINE`
  (baseline/iterative/graph, vía env): solo cambia el retrieval, la generación es compartida.
  **SIEMPRE corre el suite RAGAS completo** (faithfulness+precision+recall; answer_relevancy
  omitida a propósito por ser artefacto en español); se eliminaron los modos lean
  `RETRIEVAL_ONLY`/`RECALL_ONLY`. `RunConfig(timeout=600, max_workers=8)` anti-timeout.
  Imprime medias globales **y por tier**, mide latencia y vuelca `resultados/resultados_ragas_<PIPELINE>.csv`.
  Las referencias las redactó el modelo desde las guías → **pendiente revisión clínica**.
- `abbreviations.py` — diccionario SIGLA→nombre de las guías (valores en español).
- **`agentic/`** — Track A (F4). `iterative.py`: `iterative_search` (plan/hop/reflect).
  Importa los primitivos compartidos de `rag.py` (carpeta padre).
- **`graph/`** — Track B (F4). `lightrag_track.py`: build del grafo + `graph_search`
  (importa `rerank` de `rag.py`, corpus de `chunks/`, store en `lightrag_store/`).
- `chunks/` — `chunk_guias.py` (chunking estructural), `subir_a_qdrant.py` (denso),
  `subir_a_qdrant_hibrido.py` (denso+BM25), `chunks.jsonl` (517 chunks).
- `data/markdown/` — las 7 guías en Markdown (fuente del corpus). `data/pdfs/`, `data/textos/` — originales.
- `docs/` — documentos de diseño y diagramas de arquitectura. `resultados/` — CSV de evaluación RAGAS.
- `langgraph.json` — config de LangGraph Studio (expone `main.py:app`).

## Modelos y servicios

- Generación: **gpt-4o** (`GENERATION_MODEL`). Rephrase/validación: **gpt-4o-mini**
  (`REPHRASE_MODEL` / `VALIDATION_MODEL`).
- Embeddings: `text-embedding-3-large` (3072d). Reranker: jina-reranker-v2-base-multilingual.
- Qdrant Cloud (región **eu-west**). Colecciones: `guias_vih` (solo denso, respaldo),
  `guias_vih_hibrida` (denso + sparse BM25, sin contexto) y **`guias_vih_hibrida_ctx`**
  (denso + BM25 con Contextual Retrieval, **la activa por defecto**). La activa la fija
  `COLLECTION_HYBRID` en `rag.py` (default `guias_vih_hibrida_ctx`), sobreescribible con
  `QDRANT_COLLECTION` para A/B contra la no contextual.
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
      - iterative: `iter_generate_subquestions → (iter_single | FAN-OUT Send×N
        ▶ iter_retrieve_one → iter_reflect ⇄ Send×1) → iter_rerank`. iter_generate_subquestions
        SOLO genera las subpreguntas; cada subpregunta se recupera en su PROPIA ejecución de
        `iter_retrieve_one` (patrón `Send`/fan-out → una llamada visible por subpregunta en la
        traza); convergen en iter_reflect, que decide si falta evidencia y, si sí, hace otro
        `Send` (otra ronda) o sale a iter_rerank. `pool`/`hops` se acumulan con reducers
        (`_merge_pool` dedup, `_add_int`). Single-hop → iter_single (one-shot baseline).
      - graph: DOS ramas PARALELAS desde rephrase que convergen en merge:
        `rephrase ─┬ graph_keywords (LLM: keywords high-level→relaciones, low-level→entidades)
        → graph_select (cosine entidades/relaciones + traversía + chunks) ─┬ graph_merge
        → graph_rerank` y `└ graph_hybrid (denso+BM25) ─┘`. graph_select pasa las keywords ya
        extraídas a `aquery_data` (LightRAG se salta su LLM interno) y expone entidades/
        relaciones/chunks en el estado. `graph_merge` usa **`defer=True`** para correr UNA vez
        tras ambas ramas (de distinta longitud; sin defer el fan-in se dispararía dos veces).
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
  0.97, context_recall 0.94 (algo inflado). Detalle en `resultados/resultados_ragas.csv`.
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
  - **EN CURSO — preguntas de clarificación (slot-filling): HECHO (validación en Studio).**
    Puerta `assess_context`/`clarify` entre retrieval y generate (ver Arquitectura). Esquema
    **híbrido**: cribado en `refine` (`known_facts`/`candidate_modifiers`) + confirmación
    anclada en la evidencia en `rag.assess` (razonamiento estructurado
    `branches_on`/`already_covered`/`questions`, gpt-4o-mini). Pausa con `interrupt()`, funde
    respuestas en `clinical_facts` (no citable, dirige la generación), cota `CLARIFY_MAX_ROUNDS=1`.
    Probado end-to-end (baseline + MemorySaver) y casos unitarios de `assess` (condicional /
    dato ya conocido en cualquier unidad / no condicional). Validación final: LangGraph Studio
    (`langgraph dev`, persistencia la pone la plataforma).
  - **Re-recuperación enriquecida: HECHO** (nodo `re_retrieve`, incremento 1). Tras clarificar,
    los `clinical_facts` se inyectan en la consulta y re-disparan la recuperación → el dato del
    médico tira de los pasajes condicionales antes de generar (probado: VHB cambia el top-5).
    PENDIENTE: streaming, web, memoria multi-turno de conversación, y (incremento 2) vía de
    "modificadores por conocimiento implícito" en `assess` (marcada y siempre seguida de
    re-retrieval + validate) como red de seguridad ante fallos de recuperación.

## Pendiente / próximos pasos (a fecha 2026-06-29)

0. **`EVAL_SET` (151 preguntas, 4 tiers) HECHO** y preparado en `evaluation.py` (ver bullet).
   Sustituye al instrumento del A/B de F4 (el multihop saturaba recall del grafo a 0.979 y
   no tenía preguntas simples). PENDIENTE: lanzarlo (full RAGAS, ~$15 con juez mini por las 3
   pipelines, ~$15 más con juez gpt-4o para el número final) y revisión clínica de referencias.
   Antes de lanzar, sonda con un subconjunto (~10) para medir coste real en el dashboard.
1. **Contextual Retrieval (enriquecer chunks con contexto) — HECHO (índice construido).**
   `chunks/contextualize.py`: por chunk, gpt-4o-mini genera UNA frase densa de contexto
   (entidades/siglas/grado de recomendación; situándolo en su guía vía título+section_path+
   ventana de vecinos `--window`, resumible, `max_retries` alto por el tope de 200k TPM) →
   `chunks_contextual.jsonl` (517) con `context` y `text_for_retrieval` (= contexto + texto).
   El uploader embebe/BM25-indexa `text_for_retrieval` PERO el payload conserva `text` literal
   (citable). Subido a colección NUEVA **`guias_vih_hibrida_ctx`** (la original
   `guias_vih_hibrida` intacta). **Colección elegible** vía `QDRANT_COLLECTION` (env, lo lee
   `rag.py`) o `--collection` (uploader) → repunta los TRES retrievers a la vez. Coste real
   ~$0.23 (contextualizar) + ~$0.03 (re-embeber). PENDIENTE: **A/B `EVAL_SET` contra
   `guias_vih_hibrida` vs `guias_vih_hibrida_ctx`** para confirmar la mejora antes de hacerla
   la colección por defecto.
2. **Decidir capa de desviación (router simple→baseline) SOLO con datos:** correr `EVAL_SET`
   con pipelines puras y leer si el grafo se degrada en los tiers `simple`/`single_hop`. Si no
   hay gap → el router no aporta calidad (solo latencia). NO añadir antes de medir (ensucia el
   test: el tier simple pasaría a medir baseline, no la pipeline elegida).
3. **(Idea) Descripciones de relaciones como "mapa de conceptos" no citable** en el prompt
   de generación del grafo, para ayudar al razonamiento multi-hop sin romper el grounding.
   Prototipar tras un flag y A/B contra la versión actual (faithfulness + recall).
4. **(Idea) HippoRAG 2 como reemplazo de LightRAG:** mejor evidencia en multi-hop, menos
   tokens, y —clave— NO degrada las preguntas simples (a diferencia de LightRAG/GraphRAG).
   Spike previo: verificar backends swappables a Azure/EU (RGPD) y licencia. A/B tras EVAL_SET.
5. **F5 — UX.** Clarificación interactiva + re-recuperación enriquecida HECHAS (validar en
   Studio); seguir con streaming, web (Streamlit/Chainlit), memoria multi-turno y navegación
   por conceptos. Incremento 2 pendiente: vía de "modificadores por conocimiento implícito" en
   `assess` (marcada y siempre seguida de re-retrieval + validate).

Artefactos de evaluación versionados en `resultados/`: `resultados_ragas.csv` (baseline F0) y
`resultados_ragas_{baseline,iterative,graph}_retrieval.csv` (A/B F4, context_recall). El A/B
nuevo sobre `EVAL_SET` vuelca `resultados/resultados_ragas_<PIPELINE>.csv` (full RAGAS).

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
  veces. Equipo: 12 cores. `RERANK_SCORE_CHARS` ahora es env-configurable (default 512).
- **EL RERANKER SIGUE SIENDO EL CUELLO DE LA RECUPERACIÓN (medido):** `retrieve_hybrid`
  (embed+Qdrant+BM25) ~0.44 s vs `rerank` de 20 docs **~3.8 s @512** (escala ~lineal:
  ~1.8 s @256, ~1.0 s @128). Se llama 1× (baseline/graph) hasta 4× (iterative: 3 subpreguntas
  + final). Bajar a 256 da ~2× pero **cambia el top-8** (hasta ~3/8 en algunas consultas) →
  no se baja el default (prioridad nº1: no alucinar). **La paralelización rinde poco en
  iterative (~1.1×)** porque 3 rerankers en CPU saturan los núcleos; sí ayuda en graph
  (traversía ∥ híbrido = recursos distintos).
- **Optimizaciones aplicadas (esta sesión):** (1) `iterative_search` recupera las
  subpreguntas planificadas EN PARALELO (`ThreadPoolExecutor`); (2) `graph_search` corre
  traversía ∥ híbrido en paralelo; (3) `rag.warmup()` precarga reranker+BM25 y `main.py` lo
  lanza en un hilo daemon al importar (mata el ~3.5 s de la 1ª consulta); (4) locks
  thread-safe en las cargas perezosas de modelos. Los grafos dedicados de Studio ya
  paralelizaban (Send / ramas).
- **RERANKER EN GPU (HECHO, la mayor mejora de latencia).** Medido: `rerank` 20 docs pasó de
  **~3.8 s (CPU) a ~0.45 s (GPU GTX 1650)** → retrieval: baseline 7.6→5.5 s, **iterative
  15.8→4.9 s (~3.2×, tenía 4 reranks)**, graph 11.5→7.7 s. Ahora el tiempo restante son las
  llamadas LLM (rephrase/plan) y `graph_select` (LightRAG), no el reranker. Setup (Windows):
  - Driver NVIDIA reciente (610.62, soporta CUDA 13.3). **OJO:** la actualización dejó el
    servicio `nvlddmkm` deshabilitado (`Start=4`) y archivos del driver a medias →
    reinstalación limpia del driver lo arregló.
  - **`onnxruntime-gpu==1.22.0`** (build CUDA **12**; el 1.27 de PyPI es CUDA **13** y no casa)
    + wheels `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`,
    `nvidia-cufft-cu12`, `nvidia-curand-cu12` (NO el CUDA Toolkit completo). Versión de
    onnxruntime-gpu debe casar con la CUDA major de las wheels (ver qué `cublas64_XX`/`cufft64_XX`
    importa `onnxruntime_providers_cuda.dll`).
  - **Truco clave (Windows):** las DLLs de las wheels (`site-packages/nvidia/*/bin`) NO están
    en el search path, así que `_init_cuda_dlls()` en `rag.py` las añade Y las **pre-carga**
    con `ctypes.WinDLL` antes de importar fastembed (sin esto el provider CUDA falla y cae a
    CPU en silencio). Se ejecuta solo si `RERANK_DEVICE` es `cuda`/`auto`.
  - Activar: `RERANK_DEVICE=cuda` en `.env` (gitignored; específico de esta máquina). Quitarlo
    o `cpu` vuelve a CPU. `onnxruntime-gpu` NO va en `pyproject` (rompería máquinas sin GPU).
- **El coste del reranker es IRREDUCIBLE en CPU sin perder calidad (medido).** Probado:
  (a) bajar chars 512→256 cambia top‑8 (hasta 3/8); (b) reranquear menos candidatos (top15
  vs top20) cambia top‑5 (24/30 coinciden; alguna consulta 2/5); (c) una sola pasada de
  rerank en iterative cambia top‑8 (3‑6/8). El cross-encoder reordena fuerte (un chunk en
  el puesto 18 del híbrido entra a su top‑5), así que necesita los 20 candidatos @512. NO
  aplicadas: degradarían la evidencia (prioridad nº1). La única vía real es la GPU.
- **GPU disponible: GTX 1650 4 GB, driver 511.09 (CUDA máx 11.6).** Para `onnxruntime-gpu`
  hay que ACTUALIZAR driver (≥522 para CUDA 11.8, o ≥528 para CUDA 12) + CUDA toolkit +
  cuDNN, luego `RERANK_DEVICE=cuda`. Setup de sistema (admin), no inmediato.
- Modelos locales (BM25, reranker) con carga perezosa + `warmup()`: sin warm-up, la 1ª
  consulta paga ~3.5 s de carga. Studio mantiene los modelos cargados entre consultas.
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
