import os
import sys
import json
from openai import OpenAI

# La consola de Windows usa cp1252 por defecto y rompe al imprimir acentos,
# 'µ' o las cajas '═'/'─' de las fuentes. Forzamos UTF-8 en la salida.
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
qdrant = QdrantClient(
    url = QDRANT_URL,
    api_key = QDRANT_API_KEY
)

# --- Búsqueda híbrida (denso semántico + sparse léxico BM25) ---
from fastembed import SparseTextEmbedding

COLLECTION_DENSA   = "guias_vih"            # colección original (solo denso)
COLLECTION_HIBRIDA = "guias_vih_hibrida"    # denso + sparse BM25 (Fase 2)

# Modelos LLM (centralizados): la respuesta clínica final usa un modelo potente
# (la calidad es crítica); el rephrasing es una tarea simple y va con uno barato.
MODELO_GENERACION = "gpt-4o"
MODELO_REPHRASE   = "gpt-4o-mini"
MODELO_VALIDACION = "gpt-4o-mini"   # juez de grounding/relevancia (red de seguridad)

_bm25 = None
def _get_bm25() -> SparseTextEmbedding:
    """Carga perezosa del modelo BM25 (se instancia/descarga una sola vez)."""
    global _bm25
    if _bm25 is None:
        _bm25 = SparseTextEmbedding("Qdrant/bm25")
    return _bm25

def get_embedding(text: str):
    """
    Transforms the query into an embedding for latter comparison with the vector database
    """
    response = client.embeddings.create(model = "text-embedding-3-large", input = text)
    return response.data[0].embedding

def retrieve(query: str,top_k: int = 5):
    """retrieves the context identified similar to the question (solo denso)"""
    query_vector = get_embedding(query)
    response = qdrant.query_points(collection_name = COLLECTION_DENSA,
                             query = models.NearestQuery(nearest= query_vector),
                             limit= top_k,
                             with_payload = True)

    return [r.payload for r in response.points]


def retrieve_hibrido(query: str, top_k: int = 5, prefetch_limit: int = 20):
    """Búsqueda híbrida: combina recuperación semántica (vector denso) y léxica
    (sparse BM25) y las fusiona con RRF en Qdrant. De cada rama trae
    prefetch_limit candidatos y devuelve los payloads de los top_k fusionados."""
    dense_vec = get_embedding(query)
    sparse = next(iter(_get_bm25().query_embed(query)))
    response = qdrant.query_points(
        collection_name=COLLECTION_HIBRIDA,
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


# --- Reranker (cross-encoder local, multilingüe) ---
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
_reranker = None
def _get_reranker():
    """Carga perezosa del cross-encoder (se descarga/instancia una sola vez)."""
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, payloads: list, top_k: int = 5) -> list:
    """Reordena los chunks con un cross-encoder que mira la pregunta y el texto
    JUNTOS (mucho más preciso que el coseno del retriever) y devuelve los top_k.
    A diferencia del bi-encoder, suele entender abreviaturas (p.ej. DTG=dolutegravir)."""
    if not payloads:
        return []
    docs = [p["text"] for p in payloads]
    scores = list(_get_reranker().rerank(query, docs))
    ordenados = sorted(zip(scores, payloads), key=lambda x: x[0], reverse=True)
    return [p for _, p in ordenados[:top_k]]


def retrieve_rerank(query: str, top_k: int = 5, candidates: int = 20) -> list:
    """Pipeline de recuperación de Fase 2: recupera 'candidates' por búsqueda
    híbrida y los reordena con el cross-encoder, devolviendo los top_k mejores."""
    cand = retrieve_hibrido(query, top_k=candidates, prefetch_limit=30)
    return rerank(query, cand, top_k=top_k)


# ---------------------------------------------------------------------------
# REPHRASING — preprocesado de la consulta para mejorar la recuperación.
#   Reescribe la pregunta SIN añadir contenido nuevo y normaliza los términos:
#   como las guías usan tanto siglas como nombres completos, incluye AMBAS formas
#   (p.ej. "bictegravir (BIC)") para casar con el texto recuperable.
# ---------------------------------------------------------------------------
from typing import cast
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from abreviaturas import ABREVIATURAS

_LISTA_ABREV = "\n".join(f"{sigla} = {nombre}" for sigla, nombre in ABREVIATURAS.items())

_REPHRASE_SYS = f"""Eres un componente de PREPROCESADO de consultas para un buscador (retriever) sobre guías clínicas de VIH (GeSIDA). Reescribe la pregunta del usuario para que el buscador la entienda mejor. NO la respondas.

REGLAS:
1. No añadas información clínica nueva (fármacos, dosis, cifras, criterios) que no esté en la pregunta original, y no cambies su significado ni intención.
2. Aclara la redacción si es ambigua o coloquial, manteniéndola concisa y en español, como una sola consulta.
3. NORMALIZACIÓN DE TÉRMINOS (lo más importante): para CUALQUIER fármaco o término que aparezca en la lista de abreviaturas de abajo (esté escrito en la pregunta como sigla o como nombre completo), escríbelo incluyendo SIEMPRE AMBAS formas, con el patrón «nombre completo (SIGLA)». Ejemplos: "bictegravir" -> "bictegravir (BIC)"; "TAF" -> "tenofovir alafenamida (TAF)"; "terapia dual" no se toca si no está en la lista. Solo aplica a los términos de la lista; no inventes siglas.

LISTA DE ABREVIATURAS (SIGLA = nombre):
{_LISTA_ABREV}

Devuelve únicamente la consulta reescrita."""


class _ConsultaReescrita(BaseModel):
    query_reescrita: str


_rephrase_llm = None
def _get_rephrase_llm():
    """Carga perezosa del LLM de rephrasing (salida estructurada)."""
    global _rephrase_llm
    if _rephrase_llm is None:
        _rephrase_llm = ChatOpenAI(model=MODELO_REPHRASE, temperature=0).with_structured_output(
            _ConsultaReescrita, method="json_schema", strict=True
        )
    return _rephrase_llm


def rephrase(query: str) -> str:
    """Reescribe la consulta para el retriever (sin añadir info) e incluye ambas
    formas —nombre completo + sigla— de los términos conocidos. Devuelve la
    consulta reescrita; ante cualquier fallo, devuelve la original."""
    try:
        out = cast(_ConsultaReescrita,
                   _get_rephrase_llm().invoke([("system", _REPHRASE_SYS), ("human", query)]))
        return out.query_reescrita.strip() or query
    except Exception:
        return query


def buscar(query: str, top_k: int = 5) -> list:
    """Pipeline de recuperación completo (Fase 2 + 3): rephrase -> híbrido -> reranker."""
    return retrieve_rerank(rephrase(query), top_k=top_k)

def build_context(context:list): 
    """Builds a formatted text out of the chunks retrieved"""
    final_context = ""
    chunk_index = {}
    for i in range(len(context)):
        chunk = context[i]
        chunk_index[i+1] = chunk
        final_context += f"[{i+1}] {chunk['text']}\n\n"
    return chunk_index,final_context


# Prompt de sistema reutilizable (lo usan tanto main.py como graph.py).
SYS_PROMPT = """
    Eres un asistente clínico especializado en el manejo del VIH. Respondes preguntas médicas utilizando EXCLUSIVAMENTE la información de los fragmentos de guías clínicas que te proporciona el sistema RAG.

    REGLAS CLÍNICAS:
    1. Usa únicamente el contexto proporcionado. No uses conocimiento externo ni supongas información.
    2. No inventes recomendaciones, dosis, tratamientos ni criterios clínicos.
    3. Si la respuesta no está en el contexto, marca "informacion_suficiente": false y usa como respuesta: "La información no está disponible en las guías proporcionadas."
    4. Si el contexto es parcial o insuficiente, indícalo explícitamente dentro de la propia respuesta.
    5. Si hay conflicto entre fragmentos, menciona ambas versiones sin resolverlo por tu cuenta.
    6. Lenguaje clínico, preciso y estructurado.

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
    """ + _LISTA_ABREV


def build_user_prompt(query: str, context: str) -> str:
    """Prompt de usuario con el contexto numerado y la pregunta clínica."""
    return f"""
    CONTEXTO (fragmentos de guías clínicas sobre VIH,numerados):

    {context}

    PREGUNTA CLÍNICA:
    {query}

    Responde siguiendo las reglas del sistema y devuelve únicamente el objeto JSON especificado.
    """


def generate_answer(query:str,context: str):
    """Makes a call to the llm in order to get the answer conditioned on the retrieved data"""

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
        model= MODELO_GENERACION,
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}],
        temperature = 0.2,
        response_format = {"type": "json_schema","json_schema": ANSWER_SCHEMA}  # type: ignore[arg-type]
        )
    
    #devuelve el diccionario de la respuesta estructurada como un json
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("El modelo no devolvió contenido (content=None)")
    else:
        return json.loads(content)


# ---------------------------------------------------------------------------
# VALIDACIÓN — juez LLM que comprueba RELEVANCIA (la respuesta aborda la pregunta)
# y FIDELIDAD/grounding (toda afirmación está respaldada por el contexto, aunque no
# sea cita literal). Es la red de seguridad anti-alucinaciones. La integridad de las
# citas literales NO se valida aquí: ya la maneja evidencias.format_answer.
# ---------------------------------------------------------------------------
_VALIDATE_SYS = f"""Eres un VALIDADOR de respuestas clínicas sobre VIH. Recibes una PREGUNTA, un CONTEXTO (fragmentos de guías) y una RESPUESTA. Decide si la respuesta es APTA según DOS criterios:

1. RELEVANCIA: la respuesta aborda realmente lo que pregunta el usuario.
2. FIDELIDAD (grounding): TODA afirmación clínica de la respuesta está respaldada por el CONTEXTO, aunque no sea una cita literal (basta con apoyo semántico claro). Marca NO apta si hay afirmaciones inventadas, no respaldadas por el contexto o que lo contradigan.

Juzga ÚNICAMENTE con el contexto proporcionado, sin usar conocimiento externo. Las abreviaturas de las guías cuentan como respaldo válido (p.ej. si el contexto dice «BIC» y la respuesta «bictegravir», es el mismo fármaco). Lista (SIGLA = nombre):
{_LISTA_ABREV}

Devuelve: apto (bool), motivo (explicación breve) y afirmaciones_no_respaldadas (lista de frases de la respuesta sin apoyo en el contexto; vacía si es apta)."""


class _Validacion(BaseModel):
    apto: bool
    motivo: str
    afirmaciones_no_respaldadas: list[str]


_validate_llm = None
def _get_validate_llm():
    """Carga perezosa del LLM-juez (salida estructurada)."""
    global _validate_llm
    if _validate_llm is None:
        _validate_llm = ChatOpenAI(model=MODELO_VALIDACION, temperature=0).with_structured_output(
            _Validacion, method="json_schema", strict=True
        )
    return _validate_llm


def validar(question: str, answer: dict, formatted_context: str) -> dict:
    """Valida relevancia + grounding de la respuesta contra el contexto. Si la
    respuesta declara información insuficiente, no hay nada que validar (apta).
    Ante un fallo del juez, no bloquea (apta) pero lo deja constar en el motivo."""
    if not answer.get("sufficient_information", False):
        return {"apto": True, "motivo": "Información insuficiente declarada; nada que validar.",
                "afirmaciones_no_respaldadas": []}
    try:
        v = cast(_Validacion, _get_validate_llm().invoke([
            ("system", _VALIDATE_SYS),
            ("human", f"PREGUNTA:\n{question}\n\nCONTEXTO:\n{formatted_context}\n\n"
                      f"RESPUESTA A VALIDAR:\n{answer.get('answer', '')}"),
        ]))
        return {"apto": v.apto, "motivo": v.motivo,
                "afirmaciones_no_respaldadas": v.afirmaciones_no_respaldadas}
    except Exception:
        return {"apto": True, "motivo": "Validador no disponible; no se pudo verificar.",
                "afirmaciones_no_respaldadas": []}
