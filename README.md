# grounded-rag — respuestas ancladas en documentos estructurados

Arquitectura **RAG** que responde preguntas sobre un corpus de **documentos estructurados**
(guías clínicas, normativa, documentación técnica…) **citando siempre la evidencia literal**.
Construida sobre LangGraph, con foco en **no alucinar**, conectar conceptos para navegar el
corpus y una UX conversacional cuidada.

El diseño es **agnóstico al dominio**: aquí se demuestra sobre las **guías clínicas de VIH de
GeSIDA** (en español, para uso médico), pero la misma arquitectura se reproduce sobre cualquier
corpus de documentos estructurados del mismo estilo (ver [Reproducir con otro corpus](#reproducir-con-otro-corpus)).

> **Sobre la autoría.** Proyecto personal. El **diseño y las decisiones de arquitectura son
> míos**; usé **Claude Code (Anthropic)** como asistente de **programación agéntica** para
> implementarlo. El repositorio sirve también como ejemplo de hasta dónde llega ese flujo de
> trabajo agéntico cuando las ideas y el criterio los pone la persona.
>
> **Estado: prototipo** para demostración (sin usuarios reales todavía).
>
> _Nombre `grounded-rag` propuesto; cámbialo si prefieres otro (VeriRAG, CiteRAG, anchored-rag…)._

---

## Prioridades de diseño (por orden)

1. **Reducir alucinaciones.** Toda respuesta se ancla en **citas literales verificables** de
   las guías; el texto sintético (descripciones del grafo, contexto de chunks, datos del
   médico) puede *dirigir* la recuperación o la generación, pero **nunca se cita**.
2. **Conectar conceptos abstractos** para navegar las guías (grafo de conocimiento).
3. **UX tipo asistente conversacional** (clarificación interactiva, citaciones, seguimiento).

**Cumplimiento (RGPD).** El objetivo es que todo modelo/servicio sea privado o local y, a ser
posible, en región UE (los datos del médico pueden incluir información de salud, Art. 9 RGPD).
Cada llamada a un LLM está encapsulada para poder migrar a Azure OpenAI sin fricción; el
reranker y BM25 corren en local; Qdrant y LangSmith en región UE.

---

## Arquitectura (LangGraph)

```
question ─▶ rephrase ─┬─ fuera de dominio ─▶ out_of_domain ─▶ END
                      └─ en dominio ─▶ [RETRIEVAL] ─▶ assess_context ⇄ clarify (interrupt)
                             ├─ baseline : híbrido (denso + BM25) + rerank
                             ├─ iterative: plan → sub-consultas → reflect
                             └─ graph    : grafo entidad-relación (LightRAG) + híbrido  ◀ defecto
                                              re_retrieve (1×) ─▶ generate ⇄ validate ─▶ evidence
```

- **Cabecera y cola compartidas, recuperación intercambiable.** Los tres modos de
  recuperación cumplen el mismo contrato de estado; la generación, validación y citación son
  idénticas. Esto hace el pipeline **agnóstico al retriever** y barato de experimentar.
- **Recuperación híbrida** densa (`text-embedding-3-large`) + sparse **BM25**, fusión RRF, y
  **reranker** local multilingüe (cross-encoder, opción GPU).
- **Contextual Retrieval.** Cada chunk se enriquece con una frase de contexto generada por un
  LLM (qué es, a qué sección/población aplica) que se **embebe e indexa** para mejorar el
  matching denso y léxico, **conservando el texto literal** para la cita.
- **Puerta de clarificación (slot-filling).** Antes de responder, el sistema detecta qué dato
  del paciente cambiaría la respuesta (embarazo, función renal, coinfección VHB…) y **pausa**
  para preguntar al médico; el dato re-dispara la recuperación pero no se cita.
- **Validación + evidencia.** Un juez comprueba relevancia y *grounding*; la respuesta final
  se acompaña de un panel de fuentes con citas literales (fuzzy match) y aviso clínico.

Detalle exhaustivo de decisiones y hallazgos en [`CLAUDE.md`](CLAUDE.md).

---

## Stack y modelos

- **Orquestación:** LangGraph (+ LangGraph Studio para depurar) · trazas en **LangSmith (UE)**.
- **Generación:** `gpt-4o`. **Rephrase / validación / contextualización:** `gpt-4o-mini`.
- **Embeddings:** `text-embedding-3-large` (3072d). **Reranker:** `jina-reranker-v2-base-multilingual`.
- **Vector store:** Qdrant Cloud (eu-west), colección híbrida con Contextual Retrieval.
- **Grafo:** LightRAG (entidades + relaciones, almacenamiento en ficheros, reconstruible).

---

## Procedencia del corpus

Los `.md` de `data/markdown/` se generaron a partir de los PDFs originales (`data/pdfs/`) con
**código de extracción generado con Claude Code y adaptado a cada PDF por separado**: cada guía
tiene particularidades de maquetación (tablas, numeración, notas al pie) que no son
extrapolables a las demás, así que no hubo un único script genérico sino uno ajustado a cada
documento. La conversión se apoya en librerías de **transcripción** PDF→Markdown
(`pymupdf4llm`) que **copian el texto, no lo generan** — el corpus es fiel al original y no
introduce alucinaciones ya en la fuente, coherente con la prioridad nº 1 del proyecto.

El **prompt** que guió todo el proceso está versionado en [`data/prompt.txt`](data/prompt.txt):
impone fidelidad por encima de completitud (prohíbe parafrasear, resumir o reconstruir, y obliga
a omitir con marca lo que no se extraiga con garantía), e implementa el flujo inspección del
PDF → script adaptado → validación → iteración.

---

## Reproducir con otro corpus

La arquitectura no tiene nada intrínsecamente "de VIH": es un patrón de **grounded RAG sobre
documentos estructurados**. Para instanciarla sobre otro corpus (otra especialidad clínica,
normativa, manuales técnicos…) se cambia la **capa de dominio**, no el motor:

**Lo que se sustituye (específico del dominio):**
- El **corpus**: los documentos en `data/` y su transcripción (el prompt de `data/prompt.txt`
  es reutilizable para cualquier PDF estructurado) + re-ejecutar el chunking de `chunks/`.
- El **glosario de siglas** (`abbreviations.py`) por el del nuevo dominio.
- La **guardia de dominio** (clasificación `in_domain` del nodo *rephrase*) que define qué
  preguntas están "dentro de tema".
- Los **prompts** que nombran el dominio (`SYS_PROMPT`, contextualización, *assess*) y los
  nombres de colección en Qdrant.

**Lo que se reutiliza tal cual (el motor):** el grafo LangGraph, la recuperación híbrida +
grafo + reranker, el Contextual Retrieval, la puerta de clarificación, y la
validación + citación literal. Es decir, **todo el valor de ingeniería es portable**; lo
específico es la configuración del dominio.

> Nota: hoy esas piezas de dominio están en el código (no parametrizadas en un fichero de
> config). Extraerlas a configuración es trabajo futuro natural, pero la separación
> conceptual motor/dominio ya existe.

---

## Cómo ejecutar

Requisitos: Python 3.12+, dependencias gestionadas con [`uv`](https://docs.astral.sh/uv/), y un
`.env` (ver [`.env.example`](.env.example)) con `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`
y, opcionalmente, las claves de LangSmith.

```bash
# CLI interactivo
python main.py

# LangGraph Studio (depuración visual del grafo, paso a paso)
langgraph dev --no-reload

# Forzar un modo de recuperación
python main.py iterative        # baseline | iterative | graph
```

Indexado del corpus (una vez):

```bash
python chunks/contextualize.py                                      # enriquece los chunks
python chunks/subir_a_qdrant_hibrido.py chunks/chunks_contextual.jsonl --collection guias_vih_hibrida_ctx
python -m graph.lightrag_track                                      # construye el grafo
```

---

## Evaluación

Un único conjunto **`EVAL_SET`** (151 preguntas) etiquetadas por tipo
(**simple / single_hop / multihop / adversarial**) mide la calidad **por tipo de pregunta** con
**RAGAS** (faithfulness + context precision + context recall). El A/B entre retrievers solo
cambia la recuperación (la generación es compartida), y la colección es elegible por env para
comparar versiones del índice:

```bash
PIPELINE=graph python evaluation.py
QDRANT_COLLECTION=guias_vih_hibrida PIPELINE=graph python evaluation.py   # A/B sin contexto
```

> Las referencias del set las redactó el modelo a partir de las guías y **están pendientes de
> revisión clínica**; los números son indicativos.

---

## Estructura del repositorio

| Ruta | Qué es |
|------|--------|
| `main.py` | Grafo LangGraph + generación estructurada (punto de entrada). |
| `rag.py` | Pipeline de recuperación: híbrido, rerank, refine, validate, prompts. |
| `evidence.py` | Formateo de respuesta y panel de fuentes con citas literales. |
| `evaluation.py` | Evaluación RAGAS (`EVAL_SET`, por tier). |
| `agentic/` · `graph/` | Recuperación iterativa (Track A) y por grafo LightRAG (Track B). |
| `chunks/` | Chunking, contextualización y subida a Qdrant (incluye `chunks.jsonl`). |
| `data/` | Las 7 guías GeSIDA: `markdown/` (corpus), `pdfs/` y `textos/` (originales). |
| `resultados/` | CSV de evaluación RAGAS (un fichero por pipeline). |
| `docs/` | Documentos de diseño y diagramas de arquitectura. |
| `CLAUDE.md` | Documento vivo de contexto, decisiones y hallazgos. |

---

## Licencia

El **código** se publica bajo licencia **MIT** (ver [`LICENSE`](LICENSE)): úsalo, cópialo o
modifícalo libremente conservando el aviso de copyright. La licencia **NO cubre el contenido de
`data/`**: las guías de GeSIDA son obra y propiedad de sus autores y se incluyen únicamente para
la demostración del prototipo.

## Aviso

Herramienta de apoyo a la decisión clínica en fase de prototipo. **No sustituye el juicio
médico** ni constituye consejo clínico.
