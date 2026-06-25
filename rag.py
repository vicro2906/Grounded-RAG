import os
import sys
import json
from openai import OpenAI

# The Windows console uses cp1252 by default and breaks when printing accents,
# 'µ' or the '═'/'─' boxes. Force UTF-8 on stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass
from qdrant_client import QdrantClient
from qdrant_client import models

from dotenv import load_dotenv
load_dotenv()

QDRANT_URL     = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key = OPENAI_API_KEY)
# Qdrant Cloud exposes the REST API on BOTH 6333 (the client's default) and 443. We default
# to 443 because restrictive networks (corporate/campus WiFi, some VPNs) often block the
# non-standard 6333 outbound while 443 is universally open — symptom is
# `ResponseHandlingException(ConnectTimeout('timed out'))` on every query. Override with
# QDRANT_PORT if needed (e.g. 6333 for a self-hosted instance).
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "443"))
qdrant = QdrantClient(
    url = QDRANT_URL,
    port = QDRANT_PORT,
    api_key = QDRANT_API_KEY,
    timeout = 30,
)

# --- GPU enablement for the cross-encoder (must run BEFORE onnxruntime/fastembed import) ---
def _init_cuda_dlls() -> None:
    """Make onnxruntime-gpu find the pip-installed CUDA/cuDNN DLLs on Windows. The nvidia-*
    wheels drop their DLLs under site-packages/nvidia/*/bin, which is NOT on the default DLL
    search path, so the CUDA provider silently fails to load and falls back to CPU. We add
    those dirs AND pre-load the DLLs into the process (so dependency resolution matches them
    by already-loaded module name) — that combination is what reliably initializes the CUDA
    session. No-op off Windows or if the nvidia wheels / GPU are not present.
    Requires onnxruntime-gpu matching the CUDA major of the nvidia-*-cuXX wheels (here CUDA 12:
    onnxruntime-gpu 1.22.x + nvidia-*-cu12). Enabled only when RERANK_DEVICE is cuda/auto."""
    if sys.platform != "win32":
        return
    import glob, ctypes
    try:
        base = os.path.dirname(__import__("nvidia").__file__)
    except Exception:
        return  # nvidia CUDA wheels not installed -> stay on CPU
    bins = glob.glob(os.path.join(base, "*", "bin"))
    for d in bins:
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
    dlls = [f for d in bins for f in glob.glob(os.path.join(d, "*.dll"))]
    for _ in range(3):  # a few passes so inter-DLL deps resolve regardless of load order
        for f in dlls:
            try:
                ctypes.WinDLL(f)
            except Exception:
                pass


if os.environ.get("RERANK_DEVICE", "cpu").lower() in ("cuda", "auto"):
    _init_cuda_dlls()

# --- Hybrid search (dense semantic + sparse lexical BM25) ---
from fastembed import SparseTextEmbedding

COLLECTION_DENSE  = "guias_vih"            # original collection (dense only)
COLLECTION_HYBRID = "guias_vih_hibrida"    # dense + sparse BM25 (Phase 2)

# Centralized LLM models: the final clinical answer uses a strong model (quality is
# critical); rephrasing is a simple task and runs on a cheap one.
GENERATION_MODEL = "gpt-4o"
REPHRASE_MODEL   = "gpt-4o-mini"
VALIDATION_MODEL = "gpt-4o-mini"   # grounding/relevance judge (safety net)

# A lock guards the lazy model loads: retrieval now runs sub-queries / branches in parallel
# (iterative fan-out, graph traverse∥hybrid), so two threads could hit a cold model at once.
# Double-checked locking keeps the fast path lock-free once the model is loaded.
import threading
_model_lock = threading.Lock()

_bm25 = None
def _get_bm25() -> SparseTextEmbedding:
    """Lazy-load the BM25 model (instantiated/downloaded only once; thread-safe)."""
    global _bm25
    if _bm25 is None:
        with _model_lock:
            if _bm25 is None:
                _bm25 = SparseTextEmbedding("Qdrant/bm25")
    return _bm25

def get_embedding(text: str):
    """Transform the query into an embedding to compare against the vector database."""
    response = client.embeddings.create(model = "text-embedding-3-large", input = text)
    return response.data[0].embedding

def retrieve(query: str, top_k: int = 5):
    """Retrieve the context most similar to the question (dense only)."""
    query_vector = get_embedding(query)
    response = qdrant.query_points(collection_name = COLLECTION_DENSE,
                             query = models.NearestQuery(nearest= query_vector),
                             limit= top_k,
                             with_payload = True)

    return [r.payload for r in response.points]


def retrieve_hybrid(query: str, top_k: int = 5, prefetch_limit: int = 20):
    """Hybrid search: combine semantic (dense vector) and lexical (sparse BM25)
    retrieval, fusing them with RRF in Qdrant. Pulls prefetch_limit candidates from
    each branch and returns the payloads of the top_k fused results."""
    dense_vec = get_embedding(query)
    sparse = next(iter(_get_bm25().query_embed(query)))
    response = qdrant.query_points(
        collection_name=COLLECTION_HYBRID,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=prefetch_limit),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
                using="bm25",
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [r.payload for r in response.points]


# --- Reranker (local, multilingual cross-encoder) ---
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
# Device for the cross-encoder. Default "cpu" = current behaviour. Set RERANK_DEVICE=cuda to
# run it on an NVIDIA GPU (needs `onnxruntime-gpu` + CUDA/cuDNN installed; see CLAUDE.md). It
# only helps THIS local model — embeddings are OpenAI (remote) and generation is gpt-4o. If
# CUDA is requested but unavailable, we fall back to CPU instead of crashing.
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "cpu").lower()  # "cpu" | "cuda" | "auto"
_reranker = None
def _get_reranker():
    """Lazy-load the cross-encoder (downloaded/instantiated only once; thread-safe)."""
    global _reranker
    if _reranker is None:
        with _model_lock:
            if _reranker is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                try:
                    if RERANK_DEVICE == "cuda":
                        _reranker = TextCrossEncoder(RERANKER_MODEL, cuda=True)
                    elif RERANK_DEVICE == "auto":
                        _reranker = TextCrossEncoder(RERANKER_MODEL)  # fastembed picks device
                    else:
                        _reranker = TextCrossEncoder(RERANKER_MODEL, cuda=False)
                except Exception:
                    # CUDA asked for but no GPU provider available -> safe CPU fallback.
                    _reranker = TextCrossEncoder(RERANKER_MODEL, cuda=False)
    return _reranker


def warmup() -> None:
    """Preload the local models (reranker + BM25) so the FIRST query does not pay their
    ~3.5s load. Safe to call repeatedly and from a background thread (lazy + locked)."""
    try:
        _get_reranker()
        _get_bm25()
    except Exception:
        pass


# The cross-encoder pads every item in a batch to the LONGEST one on CPU, so a single
# long chunk makes the whole rerank crawl (~128 s observed). We only need a strong
# relevance SIGNAL, not the full text, so we score a truncated prefix while still
# RETURNING the full payloads. Measured effect: ~128 s -> ~5 s.
# This is the dominant cost of retrieval on CPU and scales ~linearly with the length:
# 20 docs take ~3.8 s @512, ~1.8 s @256, ~1.0 s @128. Tunable via RERANK_SCORE_CHARS, but
# the DEFAULT stays 512 because lowering it shifts the top-8 (measured: 256 changes up to
# ~3/8 chunks on some queries) and quality/no-hallucination is priority #1. The real fix for
# rerank latency is the GPU (see RERANK_DEVICE) — it cuts each call ~10-50x.
RERANK_SCORE_CHARS = int(os.environ.get("RERANK_SCORE_CHARS", "512"))

def rerank(query: str, payloads: list, top_k: int = 5) -> list:
    """Reorder the chunks with a cross-encoder that looks at the question and the
    text TOGETHER (far more precise than the retriever's cosine) and return the top_k.
    Unlike the bi-encoder, it usually understands abbreviations (e.g. DTG=dolutegravir).
    Scores only the first RERANK_SCORE_CHARS of each chunk (latency fix); returns the
    full payloads untouched."""
    if not payloads:
        return []
    docs = [p["text"][:RERANK_SCORE_CHARS] for p in payloads]
    scores = list(_get_reranker().rerank(query, docs))
    ordered = sorted(zip(scores, payloads), key=lambda x: x[0], reverse=True)
    return [p for _, p in ordered[:top_k]]


def retrieve_rerank(query: str, top_k: int = 5, candidates: int = 20) -> list:
    """Phase 2 retrieval pipeline: retrieve 'candidates' via hybrid search and
    reorder them with the cross-encoder, returning the best top_k."""
    cand = retrieve_hybrid(query, top_k=candidates, prefetch_limit=30)
    return rerank(query, cand, top_k=top_k)


# ---------------------------------------------------------------------------
# REPHRASING — query preprocessing to improve retrieval.
#   Rewrites the question WITHOUT adding new content and normalizes terms: since the
#   guidelines use both abbreviations and full names, it includes BOTH forms
#   (e.g. "bictegravir (BIC)") to match the retrievable text.
# ---------------------------------------------------------------------------
from typing import cast
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from abbreviations import ABBREVIATIONS

_ABBREV_LIST = "\n".join(f"{abbr} = {name}" for abbr, name in ABBREVIATIONS.items())

_REPHRASE_SYS = f"""Eres un componente de PREPROCESADO de consultas para un asistente sobre las guías clínicas de VIH (GeSIDA). Haces DOS tareas y NO respondes la pregunta.

A) CLASIFICAR si la pregunta pertenece al dominio (campo in_domain):
   - in_domain = true si trata sobre VIH o su manejo clínico (tratamiento antirretroviral, fármacos, coinfecciones, profilaxis, embarazo, vacunas, efectos adversos, resistencias, etc.) y podría responderse con guías clínicas de VIH.
   - in_domain = false SOLO si es CLARAMENTE ajena (cultura general, política, charla, u otros temas médicos sin relación con el VIH). Ante la duda, marca true: es mejor dejarla pasar y que el resto del sistema decida.

B) REESCRIBIR la consulta para el buscador (campo rewritten_query; solo importa si in_domain=true — si es false, puedes devolver la pregunta tal cual):
   1. No añadas información clínica nueva (fármacos, dosis, cifras, criterios) que no esté en la pregunta original, y no cambies su significado ni intención.
   2. Aclara la redacción si es ambigua o coloquial, manteniéndola concisa y en español, como una sola consulta.
   3. NORMALIZACIÓN DE TÉRMINOS: para CUALQUIER fármaco o término que aparezca en la lista de abreviaturas de abajo (escrito como sigla o como nombre completo), inclúyelo SIEMPRE en AMBAS formas, con el patrón «nombre completo (SIGLA)». Ejemplos: "bictegravir" -> "bictegravir (BIC)"; "TAF" -> "tenofovir alafenamida (TAF)". Solo los términos de la lista; no inventes siglas.

C) CRIBADO de datos del paciente (cribado barato, NO respondas; solo importa si in_domain=true):
   1. known_facts: extrae los datos clínicos del PACIENTE que YA aparecen explícitos en la pregunta, como pares «atributo: valor». Ejemplos: "embarazo: sí", "semana_gestacion: 12", "aclaramiento_renal: 25 ml/min", "coinfeccion_VHB: sí", "CD4: 200", "carga_viral: indetectable", "pauta_actual: BIC/FTC/TAF". Solo lo dicho explícitamente; NO inventes ni infieras valores.
   2. candidate_modifiers: lista de modificadores clínicos que esta pregunta PODRÍA necesitar para una recomendación precisa y que NO están ya resueltos en known_facts. Elige de esta lista cerrada cuando apliquen: "gestacion", "funcion_renal", "funcion_hepatica", "coinfeccion_VHB", "coinfeccion_VHC", "cd4", "carga_viral", "resistencias", "pauta_actual", "interacciones", "edad_pediatrica", "lactancia". No es una decisión final (eso lo confirma otro componente con la evidencia recuperada); aquí solo señalas candidatos plausibles. Si la pregunta es general/definitoria y no depende de datos del paciente, devuelve lista vacía.

LISTA DE ABREVIATURAS (SIGLA = nombre):
{_ABBREV_LIST}

Devuelve in_domain (bool), rewritten_query (str), known_facts (lista de strings "atributo: valor") y candidate_modifiers (lista de strings)."""


class _Refined(BaseModel):
    in_domain: bool
    rewritten_query: str
    known_facts: list[str]
    candidate_modifiers: list[str]


_refine_llm = None
def _get_refine_llm():
    """Lazy-load the refine LLM (rephrase + domain classification)."""
    global _refine_llm
    if _refine_llm is None:
        _refine_llm = ChatOpenAI(model=REPHRASE_MODEL, temperature=0).with_structured_output(
            _Refined, method="json_schema", strict=True
        )
    return _refine_llm


def _facts_to_dict(facts: list[str]) -> dict:
    """Parse the LLM's "atributo: valor" strings into a {attribute: value} dict (the shape
    the clinical profile uses). Lines without a colon are kept under their own key."""
    out: dict = {}
    for f in facts or []:
        if not isinstance(f, str) or not f.strip():
            continue
        key, sep, val = f.partition(":")
        key = key.strip()
        if key:
            out[key] = val.strip() if sep else ""
    return out


def refine(query: str) -> dict:
    """Single LLM call that (a) rewrites the query for the retriever without adding
    info and normalizing terms, (b) classifies whether it belongs to the HIV domain, and
    (c) screens the patient data already present in the question (known_facts) and the
    clinical modifiers the question might still need (candidate_modifiers) — the cheap half
    of the hybrid clarification step. Returns
    {"query": str, "in_domain": bool, "known_facts": dict, "candidate_modifiers": list}.
    On LLM failure it does not block: returns the original query, in_domain=True, no facts."""
    try:
        out = cast(_Refined,
                   _get_refine_llm().invoke([("system", _REPHRASE_SYS), ("human", query)]))
        return {"query": out.rewritten_query.strip() or query, "in_domain": out.in_domain,
                "known_facts": _facts_to_dict(out.known_facts),
                "candidate_modifiers": list(out.candidate_modifiers or [])}
    except Exception:
        return {"query": query, "in_domain": True, "known_facts": {}, "candidate_modifiers": []}


def rephrase(query: str) -> str:
    """Rewritten query only (compatibility helper; used by the evaluation)."""
    return refine(query)["query"]


def search(query: str, top_k: int = 5) -> list:
    """Full retrieval pipeline (Phase 2 + 3): rephrase -> hybrid -> reranker."""
    return retrieve_rerank(rephrase(query), top_k=top_k)


# Track A (iterative/agentic, multi-hop) lives in ../agentic/iterative.py and Track B
# (LightRAG graph) in ../graph/lightrag_track.py. Both import the shared primitives above
# (retrieve_hybrid, rerank, retrieve_rerank, search) from this parent module.


def build_context(context: list):
    """Build a formatted, numbered text out of the retrieved chunks."""
    final_context = ""
    chunk_index = {}
    for i in range(len(context)):
        chunk = context[i]
        chunk_index[i+1] = chunk
        final_context += f"[{i+1}] {chunk['text']}\n\n"
    return chunk_index, final_context


# Reusable system prompt (kept in Spanish: drives Spanish output over Spanish guides).
SYS_PROMPT = """
    Eres un asistente clínico especializado en el manejo del VIH. Respondes preguntas médicas utilizando EXCLUSIVAMENTE la información de los fragmentos de guías clínicas que te proporciona el sistema RAG.

    REGLAS CLÍNICAS:
    1. Usa únicamente el contexto proporcionado. No uses conocimiento externo ni supongas información.
    2. No inventes recomendaciones, dosis, tratamientos ni criterios clínicos.
    3. Si la respuesta no está en el contexto, marca "informacion_suficiente": false y usa como respuesta: "La información no está disponible en las guías proporcionadas."
    4. Si el contexto es parcial o insuficiente, indícalo explícitamente dentro de la propia respuesta.
    5. Si hay conflicto entre fragmentos, menciona ambas versiones sin resolverlo por tu cuenta.
    6. Lenguaje clínico, preciso y estructurado.
    6 bis. Si el usuario incluye un bloque "DATOS APORTADOS POR EL MÉDICO", úsalos ÚNICAMENTE para seleccionar entre las recomendaciones del contexto la que aplica a ese paciente (p. ej. la rama de gestación, de insuficiencia renal o de coinfección). Esos datos NO son una fuente: no los cites, no los incluyas en "fragmentos_usados" ni en "cita_textual", y toda afirmación clínica debe seguir respaldada por los fragmentos del contexto. Si el contexto no cubre el escenario indicado por esos datos, dilo explícitamente y marca "informacion_suficiente" según corresponda.

    REDACCIÓN DE LA RESPUESTA:
    7. Redacta una respuesta completa, cohesionada y bien estructurada en prosa, con los párrafos que requiera la pregunta. No la trocees artificialmente; desarrolla la idea con naturalidad clínica, integrando la justificación dentro de la propia explicación.
    8. No incluyas en el texto de la respuesta los títulos, secciones, años ni números de fragmento: esos datos los añade el sistema automáticamente como fuentes al final. Escribe la respuesta como prosa limpia, sin marcadores tipo [1] ni "Fuente del contexto".

    REGLAS DE CITACIÓN (críticas):
    9. Cada fragmento del contexto viene numerado: [1], [2], etc.
    10. En "fragmentos_usados" incluye ÚNICAMENTE los fragmentos que realmente sustentan tu respuesta. Si solo usaste 2 de los 5, devuelve solo esos 2. No incluyas fragmentos irrelevantes ni "por si acaso".
    11. Para cada fragmento usado, copia en "cita_textual" la frase EXACTA Y LITERAL del fragmento que respalda tu afirmación, carácter por carácter, sin reescribirla, resumirla ni corregirla. Debe poder encontrarse tal cual dentro del texto del fragmento.

    PREGUNTAS DE SEGUIMIENTO:
    12. Solo cuando "informacion_suficiente" sea true, genera EXACTAMENTE 3 preguntas de seguimiento ("preguntas_seguimiento") que un clínico podría plantear de forma natural justo después de esta consulta.
    13. Cada pregunta debe: (a) ser específica y clínicamente útil; (b) abordar un aspecto NO resuelto ya en tu respuesta (profundizar en un matiz, un escenario clínico contiguo, monitorización, interacciones, manejo alternativo, etc.); (c) poder responderse previsiblemente con guías clínicas de VIH (GeSIDA/SPNS). NO formules preguntas de cultura general ni que dependan de datos del paciente concreto que no se han aportado.
    14. Redáctalas breves, autocontenidas, en español y terminadas en "?". No las numeres ni les añadas prefijos.
    15. Cuando "informacion_suficiente" sea false, devuelve "preguntas_seguimiento" como una lista vacía []. Las preguntas de seguimiento deben relacionarse siempre con la respuesta dada; si no hay respuesta, no se plantean.

    FORMATO DE SALIDA:
    Devuelve EXCLUSIVAMENTE un objeto JSON válido, sin texto antes ni después y sin envolverlo en bloques de código:
    {
    "informacion_suficiente": true,
    "respuesta": "respuesta clínica completa, cohesionada y estructurada en prosa, con los párrafos que haga falta",
    "fragmentos_usados": [
        {"ref": 1, "cita_textual": "frase literal copiada del fragmento [1]"},
        {"ref": 3, "cita_textual": "frase literal copiada del fragmento [3]"}
    ],
    "preguntas_seguimiento": ["¿…?", "¿…?", "¿…?"]
    }
    Si "informacion_suficiente" es false, tanto "fragmentos_usados" como "preguntas_seguimiento" deben ser listas vacías [].

    ABREVIATURAS DE LAS GUÍAS (forman parte de las guías; NO son conocimiento externo):
    Trata cada sigla y su nombre completo como EL MISMO término. Si el contexto usa la sigla (p.ej. «BIC») y la pregunta el nombre completo («bictegravir»), o al revés, son el mismo fármaco/término y debes responder usando ese fragmento. Lista (SIGLA = nombre):
    """ + _ABBREV_LIST


def _format_clinical_facts(clinical_facts: dict | None) -> str:
    """Render the doctor-provided patient data as a NON-CITABLE block for the prompt. Returns
    "" when there are no facts, so the prompt is unchanged for questions that did not clarify."""
    facts = {k: v for k, v in (clinical_facts or {}).items() if k}
    if not facts:
        return ""
    lines = "\n".join(f"    - {k}: {v}" if v else f"    - {k}" for k, v in facts.items())
    return (
        "\n    DATOS APORTADOS POR EL MÉDICO (NO citable; úsalos SOLO para seleccionar la "
        "recomendación aplicable, nunca como fuente ni en cita_textual):\n"
        f"{lines}\n"
    )


def build_user_prompt(query: str, context: str, clinical_facts: dict | None = None) -> str:
    """User prompt with the numbered context, the clinical question and (optionally) the
    patient data the doctor supplied through the clarification step."""
    return f"""
    CONTEXTO (fragmentos de guías clínicas sobre VIH,numerados):

    {context}
{_format_clinical_facts(clinical_facts)}
    PREGUNTA CLÍNICA:
    {query}

    Responde siguiendo las reglas del sistema y devuelve únicamente el objeto JSON especificado.
    """


def generate_answer(query: str, context: str):
    """Call the LLM to get the answer conditioned on the retrieved data."""

    sys_prompt = SYS_PROMPT
    prompt = build_user_prompt(query, context)
    ANSWER_SCHEMA = {
        "name": "clinical_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sufficient_information": {"type": "boolean"},
                "answer": {"type": "string"},
                "sources_used": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref": {"type": "integer"},
                            "quote": {"type": "string"},
                        },
                        "required": ["ref", "quote"],
                        "additionalProperties": False,
                    },
                },
                "follow_up_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["sufficient_information", "answer", "sources_used", "follow_up_questions"],
            "additionalProperties": False,
        },
    }
    response = client.chat.completions.create(
        model= GENERATION_MODEL,
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}],
        temperature = 0.2,
        response_format = {"type": "json_schema","json_schema": ANSWER_SCHEMA}  # type: ignore[arg-type]
        )

    # return the structured answer dict parsed from the JSON
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("The model returned no content (content=None)")
    else:
        return json.loads(content)


# ---------------------------------------------------------------------------
# VALIDATION — LLM judge that checks RELEVANCE (the answer addresses the question)
# and FAITHFULNESS/grounding (every claim is supported by the context, even if not a
# literal quote). It is the anti-hallucination safety net. Literal citation integrity
# is NOT validated here: evidence.format_answer already handles it.
# ---------------------------------------------------------------------------
_VALIDATE_SYS = f"""Eres un VALIDADOR de respuestas clínicas sobre VIH. Recibes una PREGUNTA, un CONTEXTO (fragmentos de guías) y una RESPUESTA. Decide si la respuesta es APTA según DOS criterios:

1. RELEVANCIA: la respuesta aborda realmente lo que pregunta el usuario.
2. FIDELIDAD (grounding): TODA afirmación clínica de la respuesta está respaldada por el CONTEXTO, aunque no sea una cita literal (basta con apoyo semántico claro). Marca NO apta si hay afirmaciones inventadas, no respaldadas por el contexto o que lo contradigan.

Juzga ÚNICAMENTE con el contexto proporcionado, sin usar conocimiento externo. Las abreviaturas de las guías cuentan como respaldo válido (p.ej. si el contexto dice «BIC» y la respuesta «bictegravir», es el mismo fármaco). Lista (SIGLA = nombre):
{_ABBREV_LIST}

Devuelve: is_valid (bool), reason (explicación breve) y unsupported_claims (lista de frases de la respuesta sin apoyo en el contexto; vacía si es apta)."""


class _Validation(BaseModel):
    is_valid: bool
    reason: str
    unsupported_claims: list[str]


_validate_llm = None
def _get_validate_llm():
    """Lazy-load the judge LLM (structured output)."""
    global _validate_llm
    if _validate_llm is None:
        _validate_llm = ChatOpenAI(model=VALIDATION_MODEL, temperature=0).with_structured_output(
            _Validation, method="json_schema", strict=True
        )
    return _validate_llm


def validate(question: str, answer: dict, formatted_context: str) -> dict:
    """Validate relevance + grounding of the answer against the context. Returns
    {is_valid, error, reason, unsupported_claims}:
      - error=True  -> validation could NOT run (technical failure of the judge); the
                       graph will not show the answer and will warn (no 'fail open').
      - sufficient_information=False -> nothing to validate (valid).
    """
    if not answer.get("sufficient_information", False):
        return {"is_valid": True, "error": False,
                "reason": "Declared insufficient information; nothing to validate.",
                "unsupported_claims": []}
    try:
        v = cast(_Validation, _get_validate_llm().invoke([
            ("system", _VALIDATE_SYS),
            ("human", f"PREGUNTA:\n{question}\n\nCONTEXTO:\n{formatted_context}\n\n"
                      f"RESPUESTA A VALIDAR:\n{answer.get('answer', '')}"),
        ]))
        return {"is_valid": v.is_valid, "error": False, "reason": v.reason,
                "unsupported_claims": v.unsupported_claims}
    except Exception:
        return {"is_valid": False, "error": True,
                "reason": "Could not reach the validation service (technical error).",
                "unsupported_claims": []}


# ---------------------------------------------------------------------------
# CLARIFICATION ASSESSMENT — decides whether to ask the doctor for missing patient data
# before answering. It reasons from TWO sources, in priority order:
#   (1) EVIDENCE-GROUNDED (branches_on): dimensions the RETRIEVED guidance actually branches
#       on. The most defensible: we ask only what the guides themselves condition on.
#   (2) IMPLICIT CLINICAL KNOWLEDGE (clinically_relevant): well-established modifiers that
#       typically change the answer to THIS question but are NOT visible in the current
#       context — a SAFETY NET for retrieval misses (the conditional passage may simply not
#       have been retrieved). This is sound because the doctor's answer is never used as a
#       cited source: it only steers which branch applies AND re-triggers retrieval
#       (node_re_retrieve), so the conditional evidence gets pulled before generation, and
#       validate still guards grounding. Extra first-hand data can only help.
# already_covered subtracts what the known facts already pin (forces the model to not re-ask).
# The cheap screen (candidate_modifiers, known_facts) is done earlier in refine().
# ---------------------------------------------------------------------------
_ASSESS_SYS = f"""Eres un componente que decide si, ANTES de responder, conviene pedir al médico algún dato del paciente. Recibes una PREGUNTA, el CONTEXTO recuperado (fragmentos de guías de VIH), los DATOS YA CONOCIDOS del paciente y unos MODIFICADORES CANDIDATOS detectados en un cribado previo. NO respondas la pregunta.

Razona en CUATRO pasos, rellenando los campos EN ORDEN:
1. branches_on (vía EVIDENCIA): dimensiones clínicas del paciente para las que el CONTEXTO recuperado da recomendaciones DISTINTAS (p. ej. "gestación", "función renal", "coinfección VHB", "CD4", "carga viral", "resistencias", "pauta actual"). Si el contexto no condiciona por ningún dato del paciente, déjala vacía. Los MODIFICADORES CANDIDATOS son solo pistas; aquí manda lo que el contexto realmente condiciona.
2. clinically_relevant (vía CONOCIMIENTO CLÍNICO, red de seguridad): dimensiones que, por conocimiento clínico ESTABLECIDO del VIH, suelen cambiar la respuesta a ESTA pregunta pero que NO aparecen condicionadas en el contexto actual (puede que el fragmento no se haya recuperado). Sé CONSERVADOR: solo modificadores de impacto reconocido y pertinentes a la pregunta (gestación/lactancia, función renal, función hepática, coinfección VHB/VHC, interacciones farmacológicas, CD4/carga viral, resistencias, edad pediátrica). NO incluyas dimensiones irrelevantes para la pregunta ni para preguntas generales/definitorias.
3. already_covered: de las dimensiones de branches_on Y clinically_relevant, las que YA ESTÁN RESUELTAS por cualquiera de estas dos fuentes: (a) los DATOS YA CONOCIDOS las determinan en CUALQUIER forma o unidad equivalente (si se conoce "semana_gestacion" o "trimestre" → gestación cubierta; "aclaramiento_renal" → función renal cubierta; valor de CD4 → CD4 cubierto); o (b) la dimensión YA FUE PREGUNTADA (aparece en PREGUNTAS YA FORMULADAS), aunque el dato siga incompleto. Sé generoso: cubierta si cualquier dato conocido permite deducirla o si ya se preguntó.
4. questions: UNA pregunta por cada dimensión pendiente = (branches_on ∪ clinically_relevant) − already_covered. PRIORIZA primero las de branches_on (más defendibles). Concretas, en español, terminadas en "?", pidiendo UN dato cada una. Máximo 3. NUNCA repitas, ni reformules, una pregunta que ya esté en PREGUNTAS YA FORMULADAS. Si no queda ninguna pendiente, lista vacía.

needs_clarification = true SOLO si questions no está vacía.

Las abreviaturas de las guías cuentan como el mismo término (lista SIGLA = nombre):
{_ABBREV_LIST}

Devuelve branches_on, clinically_relevant, already_covered, questions (listas) y needs_clarification (bool)."""


class _Assessment(BaseModel):
    branches_on: list[str]         # dims the retrieved context branches on (evidence-grounded)
    clinically_relevant: list[str] # dims clinical knowledge flags but the context misses (net)
    already_covered: list[str]     # dims already pinned by known facts (forces subtraction)
    questions: list[str]           # one per pending dim ((branches_on ∪ relevant) − covered)
    needs_clarification: bool


_assess_llm = None
def _get_assess_llm():
    """Lazy-load the clarification-assessment LLM (structured output)."""
    global _assess_llm
    if _assess_llm is None:
        _assess_llm = ChatOpenAI(model=VALIDATION_MODEL, temperature=0).with_structured_output(
            _Assessment, method="json_schema", strict=True
        )
    return _assess_llm


def assess(question: str, formatted_context: str, known_facts: dict | None = None,
           candidate_modifiers: list | None = None, asked_questions: list | None = None,
           max_questions: int = 1) -> dict:
    """Decide whether to ask the doctor for missing patient data before answering. Returns
    {"needs_clarification": bool, "questions": [...], "branches_on": [...],
    "clinically_relevant": [...], "already_covered": [...]} — the reasoning fields are kept
    for transparency (visible in the trace: which dimensions came from the evidence vs from
    clinical knowledge). `asked_questions` are the questions already posed in previous rounds;
    assess must NOT repeat them (the across-rounds anti-duplicate guard). `max_questions` caps
    how many questions to ask THIS round (the LLM orders them by priority — evidence first — so
    we keep the top ones); with max_questions=1 the doctor is asked one at a time. On LLM
    failure it does NOT block the pipeline: returns needs_clarification=False."""
    facts = {k: v for k, v in (known_facts or {}).items() if k}
    facts_txt = ("\n".join(f"- {k}: {v}" if v else f"- {k}" for k, v in facts.items())
                 or "(ninguno aportado)")
    cands_txt = ", ".join(candidate_modifiers or []) or "(ninguno)"
    asked_txt = "\n".join(f"- {q}" for q in (asked_questions or [])) or "(ninguna)"
    try:
        a = cast(_Assessment, _get_assess_llm().invoke([
            ("system", _ASSESS_SYS),
            ("human", f"PREGUNTA:\n{question}\n\nCONTEXTO:\n{formatted_context}\n\n"
                      f"DATOS YA CONOCIDOS:\n{facts_txt}\n\n"
                      f"MODIFICADORES CANDIDATOS:\n{cands_txt}\n\n"
                      f"PREGUNTAS YA FORMULADAS:\n{asked_txt}"),
        ]))
        qs = [q.strip() for q in (a.questions or [])
              if isinstance(q, str) and q.strip()][:max(1, max_questions)]
        needs = bool(a.needs_clarification and qs)
        return {"needs_clarification": needs, "questions": qs if needs else [],
                "branches_on": list(a.branches_on or []),
                "clinically_relevant": list(a.clinically_relevant or []),
                "already_covered": list(a.already_covered or [])}
    except Exception:
        return {"needs_clarification": False, "questions": [],
                "branches_on": [], "clinically_relevant": [], "already_covered": []}
