"""
Evaluación del RAG clínico (guías VIH) con RAGAS — script único.

Tú rellenas abajo el GOLDEN_SET (pregunta + referencia escrita por ti) y el
script hace todo lo demás: pasa cada pregunta por tu RAG (main.py) y calcula
las métricas de RAGAS.

Flujo:
    1. Rellena las "referencia" del GOLDEN_SET (las preguntas ya están puestas).
    2. python evaluacion_rag.py
    3. Mira resultados_ragas.csv (detalle pregunta a pregunta).

Requisitos:
    pip install ragas langchain-openai
    Variables de entorno cargadas por main.py (OPENAI_API_KEY, QDRANT_*).
"""

import sys
sys.dont_write_bytecode = True  # evita crear la carpeta __pycache__/

import warnings
import pandas as pd

# Silencia DeprecationWarning internos de ragas (rutas "modernas" que son APIs
# distintas, incompatibles con el evaluate() clásico que usamos aquí).
warnings.filterwarnings(
    "ignore",
    message=r".*is deprecated and will be removed.*",
    category=DeprecationWarning,
)

# --- Tu pipeline existente, sin tocar ---
from main import retrieve, build_context, generate_answer

# --- RAGAS ---
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ===========================================================================
# CONFIG DEL JUEZ
#   JUEZ_FUERTE = False -> gpt-4o-mini (barato, para iterar)
#   JUEZ_FUERTE = True  -> gpt-4o      (para el número que vayas a reportar)
# ===========================================================================
JUEZ_FUERTE = False
MODELO_BARATO = "gpt-4o-mini"
MODELO_FUERTE = "gpt-4o"
MODELO_JUEZ = MODELO_FUERTE if JUEZ_FUERTE else MODELO_BARATO


# ===========================================================================
# GOLDEN SET  ->  RELLENA TÚ EL CAMPO "referencia" DE CADA PREGUNTA
#   - "pregunta":  ya puestas (puedes editarlas/añadir/quitar)
#   - "referencia": la respuesta correcta según TUS guías, escrita por ti
#   Las preguntas con "referencia" vacía se omiten automáticamente.
# ===========================================================================
GOLDEN_SET = [
    {"pregunta": "En un paciente con VIH con carga viral indetectable en DTG/3TC, ¿qué factores obligarían a no mantener una terapia dual según las recomendaciones actuales?", "referencia": ""},
    {"pregunta": "¿Qué pruebas deben realizarse antes de iniciar abacavir y qué ocurriría si el resultado no está disponible en un paciente con infección aguda?", "referencia": ""},
    {"pregunta": "Si un paciente con VIH tiene fracaso virológico con viremias bajas persistentes, ¿cuál es el algoritmo recomendado antes de cambiar TAR?", "referencia": ""},
    {"pregunta": "¿Por qué la adherencia subóptima puede generar resistencia incluso cuando la carga viral se mantiene relativamente baja?", "referencia": ""},
    {"pregunta": "¿En qué situaciones clínicas se recomienda no reducir a regímenes de menos de tres fármacos aunque el paciente esté suprimido?", "referencia": ""},
    {"pregunta": "¿Qué problemas farmacológicos surgen al tratar tuberculosis con rifampicina en un paciente que toma inhibidores de integrasa?", "referencia": ""},
    {"pregunta": "¿Cuál es el momento óptimo para iniciar TAR en un paciente con VIH que presenta tuberculosis activa y CD4 muy bajos?", "referencia": ""},
    {"pregunta": "¿Por qué algunos antirretrovirales deben evitarse en pacientes con insuficiencia renal significativa?", "referencia": ""},
    {"pregunta": "¿Qué implicaciones tiene la infección por VIH-2 en la elección del tratamiento inicial?", "referencia": ""},
    {"pregunta": "En un paciente con infección aguda por VIH, ¿por qué iniciar TAR inmediatamente puede tener beneficios inmunológicos y epidemiológicos?", "referencia": ""},
    {"pregunta": "¿En qué situaciones no se recomienda profilaxis postexposición (PEP) aunque haya contacto con fluidos potencialmente infecciosos?", "referencia": ""},
    {"pregunta": "¿Qué factores determinan el riesgo de transmisión de VIH tras una exposición ocupacional?", "referencia": ""},
    {"pregunta": "¿Por qué la profilaxis postexposición debe iniciarse idealmente antes de 72 horas y qué ocurre si se inicia después?", "referencia": ""},
    {"pregunta": "¿Qué vacunas están contraindicadas o deben evaluarse con precaución en personas con VIH con CD4 bajos?", "referencia": ""},
    {"pregunta": "¿Qué factores inmunológicos pueden disminuir la inmunogenicidad de las vacunas en pacientes con VIH?", "referencia": ""},
    {"pregunta": "Si un paciente tiene carga viral indetectable pero adherencia irregular, ¿puede seguir transmitiendo VIH?", "referencia": ""},
    {"pregunta": "¿Puede un paciente con CD4 normales tener igualmente deterioro neurocognitivo asociado al VIH?", "referencia": ""},
    {"pregunta": "Si la adherencia es alta pero encontramos un fracaso virológico, ¿qué causas no relacionadas con adherencia deben investigarse?", "referencia": ""},
    {"pregunta": "¿Por qué la introducción del TAR redujo la incidencia de demencia asociada al VIH, pero no eliminó los trastornos neurocognitivos?", "referencia": ""},
    {"pregunta": "Si un paciente con VIH tiene carga viral indetectable en plasma, pero presenta replicación viral detectable en LCR, ¿puede desarrollar deterioro neurocognitivo relacionado con el VIH?", "referencia": ""},
    {"pregunta": "Si una persona con VIH mantiene carga viral indetectable durante años, ¿puede seguir teniendo inflamación crónica sistémica que aumente su riesgo cardiovascular?", "referencia": ""},
    {"pregunta": "¿Puede un paciente con CD4 elevados y carga viral indetectable desarrollar trastornos neurocognitivos asociados al VIH (HAND), y qué mecanismos lo explicarían?", "referencia": ""},
    {"pregunta": "Si un paciente presenta fracaso virológico con buena adherencia documentada, ¿qué papel puede tener la resistencia viral preexistente o transmitida?", "referencia": ""},
    {"pregunta": "Si un paciente tiene síndrome metabólico mientras toma TAR, ¿cómo diferenciar si la causa es la infección por VIH, los efectos del tratamiento o los factores clásicos de riesgo cardiovascular?", "referencia": ""},
    {"pregunta": "Si un paciente tiene carga viral indetectable pero con adherencia intermitente, ¿qué riesgo existe de rebote viral, desarrollo de resistencia y pérdida futura de supresión virológica?", "referencia": ""},
    {"pregunta": "Si un paciente con VIH tiene factores clásicos de riesgo cardiovascular controlados, ¿por qué sigue teniendo mayor riesgo de enfermedad cardiovascular que la población general?", "referencia": ""},
    {"pregunta": "¿Puede un paciente con VIH con carga viral indetectable tener reservorios virales activos en tejidos, y qué implicaciones tiene esto para la curación del VIH?", "referencia": ""},
    {"pregunta": "Si el TAR ha reducido drásticamente las complicaciones neurológicas graves del VIH, ¿por qué siguen observándose formas leves o moderadas de trastornos neurocognitivos en un porcentaje significativo de pacientes?", "referencia": ""},
    {"pregunta": "Si un paciente presenta fracaso virológico con viremia baja persistente (<200 copias/mL), ¿es obligatorio cambiar el tratamiento o primero deben evaluarse otros factores?", "referencia": ""},
    {"pregunta": "En un paciente con historia de fracaso previo con ITINN, ¿es seguro cambiar a CAB+RPV inyectable si actualmente está suprimido?", "referencia": ""},
    {"pregunta": "Si un paciente con VIH tiene insuficiencia renal avanzada, ¿qué antirretrovirales deberían evitarse o ajustarse de dosis?", "referencia": ""},
    {"pregunta": "En un paciente con carga viral indetectable y múltiples comorbilidades, ¿qué factores deben considerarse antes de simplificar el tratamiento?", "referencia": ""},
    {"pregunta": "Si un paciente tiene adherencia irregular, ¿por qué algunos regímenes con inhibidores de integrasa tienen mayor barrera genética que otros?", "referencia": ""},
    {"pregunta": "En un paciente con fracaso virológico y múltiples mutaciones de resistencia, ¿cuál es el principio fundamental para diseñar el nuevo régimen?", "referencia": ""},
    {"pregunta": "Si un paciente con VIH tiene hepatitis B crónica, ¿qué implicaciones tiene esto para elegir el TAR?", "referencia": ""},
    {"pregunta": "¿Por qué en pacientes con VIH se recomienda evitar la monoterapia antirretroviral, incluso con fármacos potentes?", "referencia": ""},
    {"pregunta": "Si un paciente suprimido cambia de TAR por toxicidad, ¿qué parámetros deben monitorizarse tras el cambio para confirmar eficacia?", "referencia": ""},
    {"pregunta": "Si un paciente tiene interacciones farmacológicas complejas por polifarmacia, ¿qué clases de antirretrovirales suelen ser más fáciles de manejar?", "referencia": ""},
    {"pregunta": "Si un paciente presenta rebote viral tras años de supresión, ¿cuáles son las tres causas principales que deben investigarse antes de cambiar TAR?", "referencia": ""},
]


def construir_dataset(golden_set: list[dict]) -> EvaluationDataset:
    """Pasa cada pregunta por TU RAG y arma el dataset que consume RAGAS."""
    filas = []
    omitidas = []
    for i, caso in enumerate(golden_set, 1):
        pregunta = caso["pregunta"].strip()
        referencia = caso.get("referencia", "").strip()

        if not referencia:
            omitidas.append(i)
            continue

        # --- tu pipeline, idéntico a main() ---
        payloads = retrieve(pregunta)                       # list[dict]
        _, contexto_formateado = build_context(payloads)
        salida = generate_answer(pregunta, contexto_formateado)

        filas.append({
            "user_input": pregunta,
            "retrieved_contexts": [p["text"] for p in payloads if p],
            "response": salida["answer"],
            "reference": referencia,
        })
        print(f"[{i}/{len(golden_set)}] procesada")

    if omitidas:
        print(f"\nOmitidas por no tener referencia rellena: {omitidas}")
    if not filas:
        raise SystemExit("No hay ninguna pregunta con referencia. Rellena el GOLDEN_SET.")

    print(f"\n{len(filas)} preguntas listas para evaluar.\n")
    return EvaluationDataset.from_list(filas)


def main():
    dataset = construir_dataset(GOLDEN_SET)

    print(f"Evaluando con juez: {MODELO_JUEZ}\n")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=MODELO_JUEZ))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-large")
    )

    metricas = [
        Faithfulness(),                          # ¿la respuesta se ciñe al contexto? (no usa referencia)
        ResponseRelevancy(),                     # ¿responde a la pregunta? (no usa referencia)
        LLMContextPrecisionWithReference(),      # retriever: ¿lo relevante está priorizado? (usa referencia)
        LLMContextRecall(),                      # retriever: ¿se recuperó lo necesario? (usa referencia)
    ]

    resultado = evaluate(
        dataset=dataset,
        metrics=metricas,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    assert isinstance(resultado, EvaluationResult)

    print("\n=== Scores agregados ===")
    print(resultado)

    df = resultado.to_pandas()
    df.to_csv("resultados_ragas.csv", index=False)
    print("\nDetalle guardado en resultados_ragas.csv")

    pd.set_option("display.max_colwidth", 60)
    print("\n=== Medias por métrica ===")
    print(df.select_dtypes("number").mean())


if __name__ == "__main__":
    main()
