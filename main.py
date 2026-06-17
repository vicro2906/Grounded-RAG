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
from evidencias import format_answer

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

def get_embedding(text: str):
    """
    Transforms the query into an embedding for latter comparison with the vector database
    """
    response = client.embeddings.create(model = "text-embedding-3-large", input = text)
    return response.data[0].embedding

def retrieve(query: str,top_k: int = 5):
    """retrieves the context identified similar to the question"""
    query_vector = get_embedding(query)
    response = qdrant.query_points(collection_name = "guias_vih",
                             query = models.NearestQuery(nearest= query_vector),
                             limit= top_k,
                             with_payload = True)
    
    return [r.payload for r in response.points]

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
    """


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
        model= "gpt-4o-mini",
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


def main():
    question = input("¿Cuál es tú pregunta?: ")
    context: list = retrieve(question)
    chunk_index,formatted_context = build_context(context)
    answer = generate_answer(question, formatted_context)
    print(format_answer(answer,chunk_index))



if __name__ == "__main__":
    main()
