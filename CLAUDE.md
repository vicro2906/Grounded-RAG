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
                      └─ en dominio ─▶ retrieve (híbrido) ─▶ rerank ─▶ generate ⇄ validate ─▶ evidence ─▶ output
                                                                            (no válido+agotado ─▶ fallback)
```

- **rephrase** (`rag.refine`, gpt-4o-mini): una llamada que (a) clasifica si la pregunta
  es del dominio VIH (`in_domain`) y (b) reescribe la consulta SIN añadir info, normalizando
  términos en AMBAS formas «nombre completo (SIGLA)» con `abbreviations.py`. Si está fuera
  de dominio → `out_of_domain` (mensaje directo, corta el pipeline). La generación usa la
  pregunta ORIGINAL, no la reescrita.
- **retrieve** (`rag.retrieve_hybrid`): búsqueda híbrida densa (OpenAI
  text-embedding-3-large, 3072d) + sparse BM25 (fastembed `Qdrant/bm25`, IDF en Qdrant),
  fusión RRF. Trae 20 candidatos.
- **rerank** (`rag.rerank`): cross-encoder local `jinaai/jina-reranker-v2-base-multilingual`
  (fastembed/ONNX, multilingüe, RGPD-ok). Reordena 20 → top 5.
- **generate** (`main._structured_llm`, gpt-4o): salida estructurada con
  `ChatOpenAI.with_structured_output` (Pydantic `ClinicalAnswer`, json_schema estricto).
  Devuelve dict: sufficient_information, answer, sources_used[{ref,quote}], follow_up_questions.
- **validate** (`rag.validate`, gpt-4o-mini): juez de relevancia + grounding semántico.
  Bucle con `generate` (inyecta feedback al reintentar), `MAX_ITER=2`. válido → evidence;
  no válido y agotado → `fallback`; error técnico del juez → `fallback` (no se "falla abierto").
- **evidence** (`evidence.format_answer`): formatea respuesta + panel de fuentes con
  citas literales (fuzzy match) + preguntas de seguimiento + aviso clínico. Texto sin ANSI.

`rag.search()` = rephrase → híbrido → rerank (lo usa la evaluación).

## Ficheros clave

- `main.py` — grafo LangGraph + LangSmith + generación estructurada (punto de entrada).
- `rag.py` — pipeline: clientes, embeddings, retrieve/retrieve_hybrid, rerank, refine,
  search, validate, generate_answer (versión cruda), SYS_PROMPT, build_user_prompt, constantes de modelo.
- `evidence.py` — formateo de respuesta y fuentes.
- `evaluation.py` — evaluación RAGAS (golden set de 47 preguntas; `RETRIEVER=search`).
- `abbreviations.py` — diccionario SIGLA→nombre de las guías (valores en español).
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
- Evaluación RAGAS: `.venv\Scripts\python.exe evaluation.py` (usa el pipeline completo).
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
- **F4 — Grafo (LightRAG): CONDICIONAL.** Solo si la eval demuestra que hace falta para
  preguntas abstractas. NO es prioritario (el retrieval ya es fuerte).
- **F5 — UX tipo Claude:** Streamlit/web, streaming, citaciones, memoria multi-turno.

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

## Optimizaciones / problemas conocidos (PENDIENTE)

- **LATENCIA DEL RERANKER (crítico, sin aplicar):** el rerank tardaba **~128 s** (95% del
  tiempo total). Causa: el cross-encoder rellena (padding) el lote a la longitud del chunk
  más largo en CPU. **Fix medido: truncar a ~512 caracteres SOLO el texto que se puntúa
  (no el que se devuelve/genera) → ~5 s** (con 20 candidatos). Es decir, en `rag.rerank`
  usar `p["text"][:512]` para puntuar. Reduce el turno de ~140 s a ~12 s. Equipo: 12 cores,
  sin GPU confirmada (si hubiera GPU NVIDIA, onnxruntime-gpu lo aceleraría 10-50×).
- Modelos locales (BM25, reranker) con carga perezosa: la 1ª consulta del proceso paga
  ~3.5 s de carga del reranker. Studio lo mantiene cargado entre consultas.
- generate (gpt-4o) ~5 s: considerar streaming para mejorar latencia percibida (F5).

## Convenciones

- **Idioma del código: inglés** (identificadores, comentarios, docstrings). Se mantiene en
  español lo que se muestra al usuario (mensajes, etiquetas del panel), los prompts a los
  LLM, y el contenido/manejo de las guías y chunks (incluido `chunks/` y los valores de
  `abbreviations.py`). Al usuario se le habla en español.
- Toda llamada LLM encapsulada para poder cambiar a Azure OpenAI (GPT-4 privado) sin
  fricción el día que se requiera por cumplimiento.
- Commits directos a `main` (flujo de un solo dev). Mensajes en español. `.env` y `*.log`
  y `.langgraph_api/` en .gitignore.
