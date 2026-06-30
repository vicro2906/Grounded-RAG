# Chatbot médico VIH — RAG sobre las guías GeSIDA

Asistente conversacional (RAG) para **profesionales médicos** que responde preguntas
sobre las **guías clínicas de VIH de GeSIDA** citando siempre la evidencia literal.
Construido sobre LangGraph, con foco en **no alucinar**, conectar conceptos para navegar
las guías y una UX cuidada.

> **Sobre la autoría.** Es un proyecto personal. El **diseño, las decisiones de arquitectura
> y el razonamiento clínico/producto son míos**; usé **Claude Code (Anthropic)** como
> asistente de **programación agéntica** para implementarlo. El repositorio sirve también
> como ejemplo de hasta dónde llega ese flujo de trabajo agéntico cuando las ideas y el
> criterio los pone la persona.
>
> **Estado: prototipo** para demostración (sin usuarios reales todavía).

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
| `chunks/` | Chunking, contextualización y subida a Qdrant. |
| `markdown/` · `pdfs/` | Las 7 guías GeSIDA (fuente del corpus). |
| `CLAUDE.md` | Documento vivo de contexto, decisiones y hallazgos. |

---

## Aviso

Herramienta de apoyo a la decisión clínica en fase de prototipo. **No sustituye el juicio
médico** ni constituye consejo clínico. El contenido procede de las guías GeSIDA, cuya autoría
y derechos pertenecen a sus autores.
