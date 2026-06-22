"""
Evaluation of the clinical HIV RAG with RAGAS — single script.

You fill in the GOLDEN_SET below (question + reference written by you) and the script
does the rest: it runs each question through the RAG and computes the RAGAS metrics.

Flow:
    1. Fill in the "reference" of each GOLDEN_SET entry (the questions are already set).
    2. python evaluation.py
    3. Check resultados_ragas.csv (per-question detail).

Requirements:
    ragas, langchain-openai. Env vars loaded from .env (OPENAI_API_KEY, QDRANT_*).
"""

import os
import sys
sys.dont_write_bytecode = True  # avoid creating the __pycache__/ folder

import time
import statistics
import warnings
import pandas as pd

# Silence ragas' internal DeprecationWarning (the "modern" paths are different APIs,
# incompatible with the classic evaluate() used here).
warnings.filterwarnings(
    "ignore",
    message=r".*is deprecated and will be removed.*",
    category=DeprecationWarning,
)

# --- Existing pipeline ---
from rag import retrieve, retrieve_hybrid, retrieve_rerank, search, build_context, generate_answer

# ===========================================================================
# A/B CONFIG (Phase 4) — pick WHICH dataset and WHICH retrieval pipeline to run.
# Running the SAME questions through different retrievers (generation is shared, so
# ONLY retrieval varies) is what makes the multi-hop comparison fair.
#
#   PIPELINE (retrieval strategy):
#     "baseline"  -> search: rephrase + hybrid + reranker (Phases 2-3, current system)
#     "iterative" -> Track A: self-ask / reflect-retrieve loop (rag.iterative_search)
#     "graph"     -> Track B: LightRAG graph retrieval (graph.lightrag_retrieve)
#
#   DATASET is chosen below, after MULTIHOP_SET is defined:
#     GOLDEN_SET    -> 47 single-hop questions (F0 baseline / regression net)
#     MULTIHOP_SET  -> multi-hop questions (Phase 4 target)
# ===========================================================================
PIPELINE = os.environ.get("PIPELINE", "baseline")   # override per A/B run without editing

# RETRIEVAL_ONLY: skip the gpt-4o generation step and score ONLY the retrieval metrics
# (context_recall + context_precision). This is the lean multi-hop A/B: it measures exactly
# what differs between strategies (which chunks each one finds) while avoiding the gpt-4o
# 30k-TPM bottleneck entirely (no answer is generated). Enable with RETRIEVAL_ONLY=1.
# RECALL_ONLY: even leaner — only context_recall (the single most decision-relevant metric
# for multi-hop: "was everything needed retrieved?"). It is the light, robust one;
# context_precision is heavy (one judge call per chunk) and times out under load. RECALL_ONLY
# implies RETRIEVAL_ONLY. Enable with RECALL_ONLY=1.
RECALL_ONLY = os.environ.get("RECALL_ONLY") == "1"
RETRIEVAL_ONLY = RECALL_ONLY or os.environ.get("RETRIEVAL_ONLY") == "1"


def get_pipeline(name: str):
    """Return the retrieval function for the chosen pipeline. Track A/B retrievers are
    imported lazily so the baseline keeps working before they exist (and so 'graph' does
    not require LightRAG to be installed unless you actually run it)."""
    if name == "baseline":
        return search
    if name == "iterative":
        from agentic.iterative import iterative_search   # Track A package
        return iterative_search
    if name == "graph":
        from graph.lightrag_track import graph_search     # Track B package
        return graph_search
    raise SystemExit(f"Unknown PIPELINE: {name!r} (use baseline | iterative | graph)")

# --- RAGAS ---
from ragas import EvaluationDataset, evaluate
from ragas.run_config import RunConfig
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
# JUDGE CONFIG
#   STRONG_JUDGE = False -> gpt-4o-mini (cheap, for iterating)
#   STRONG_JUDGE = True  -> gpt-4o      (for the number you will report)
# ===========================================================================
STRONG_JUDGE = False
CHEAP_MODEL = "gpt-4o-mini"
STRONG_MODEL = "gpt-4o"
JUDGE_MODEL = STRONG_MODEL if STRONG_JUDGE else CHEAP_MODEL


# ===========================================================================
# GOLDEN SET  ->  FILL IN THE "reference" FIELD OF EACH QUESTION
#   - "question":  already set (you can edit/add/remove)
#   - "reference": the correct answer according to the guidelines, written by you
#   Questions with an empty "reference" are skipped automatically.
# ===========================================================================
GOLDEN_SET = [
    {"question": "En un paciente con VIH con carga viral indetectable en DTG/3TC, ¿qué factores obligarían a no mantener una terapia dual según las recomendaciones actuales?", "reference": "La terapia dual DTG/3TC no debe mantenerse si existe coinfección por el virus de la hepatitis B (las pautas duales no cubren el VHB), si hay resistencia conocida o sospechada a lamivudina (mutación M184V/I) o a los inhibidores de integrasa, si no se dispone de estudio de resistencias previo, si hay antecedente de fracaso virológico o si la adherencia no está garantizada. Igualmente se abandonaría ante viremia detectable/fracaso virológico o aparición de resistencias."},
    {"question": "¿Qué pruebas deben realizarse antes de iniciar abacavir y qué ocurriría si el resultado no está disponible en un paciente con infección aguda?", "reference": "Antes de iniciar abacavir es obligatorio determinar el alelo HLA-B*5701 (A-I), ya que los portadores tienen un riesgo de hasta el 50% de reacción de hipersensibilidad; si el HLA-B*5701 es positivo no debe prescribirse abacavir. Si en una infección aguda se opta por inicio rápido y aún no se dispone del resultado de HLA-B*5701 (ni del estudio de resistencias), no deben usarse regímenes con abacavir ni con ITINN; se recomienda iniciar con TDF o TAF/FTC."},
    {"question": "Si un paciente con VIH tiene fracaso virológico con viremias bajas persistentes, ¿cuál es el algoritmo recomendado antes de cambiar TAR?", "reference": "Antes de cambiar el TAR debe confirmarse el fracaso con una segunda determinación de carga viral, evaluar y reforzar la adherencia, descartar interacciones farmacológicas y problemas de absorción, revisar la potencia y barrera genética del régimen y realizar un estudio de resistencias (genotipo) si la viremia lo permite. Solo entonces se decide el cambio, guiado por el resultado de resistencias."},
    {"question": "¿Por qué la adherencia subóptima puede generar resistencia incluso cuando la carga viral se mantiene relativamente baja?", "reference": "Una adherencia intermedia mantiene concentraciones subterapéuticas del fármaco que permiten cierta replicación viral bajo presión selectiva; ese entorno favorece la selección de mutaciones de resistencia aunque no haya un rebote franco de la carga viral, especialmente con fármacos de baja barrera genética (lamivudina/emtricitabina con M184V, o los ITINN)."},
    {"question": "¿En qué situaciones clínicas se recomienda no reducir a regímenes de menos de tres fármacos aunque el paciente esté suprimido?", "reference": "No se recomienda reducir a menos de tres fármacos en presencia de coinfección por VHB (que requiere dos fármacos activos frente al VHB), cuando existe resistencia previa o archivada que comprometa los componentes de la pauta reducida, ante antecedentes de fracasos virológicos múltiples, cuando no puede garantizarse una buena adherencia o cuando no se cumplen los criterios de los ensayos (supresión estable y mantenida, sin resistencia a los fármacos del régimen reducido)."},
    {"question": "¿Qué problemas farmacológicos surgen al tratar tuberculosis con rifampicina en un paciente que toma inhibidores de integrasa?", "reference": "La rifampicina es un inductor enzimático potente (CYP3A4 y glucuronidación/UGT1A1) que reduce las concentraciones plasmáticas de los inhibidores de integrasa. Con rifampicina no pueden utilizarse EVG/c, bictegravir ni cabotegravir; el dolutegravir requiere doblar la dosis a 50 mg/12 h (en ausencia de resistencia a INI) hasta 2 semanas después de finalizar la rifampicina, y el raltegravir debe ajustarse. El tercer fármaco de elección con rifampicina es efavirenz."},
    {"question": "¿Cuál es el momento óptimo para iniciar TAR en un paciente con VIH que presenta tuberculosis activa y CD4 muy bajos?", "reference": "Se recomienda iniciar el TAR de forma precoz una vez iniciado el tratamiento antituberculoso. En pacientes con CD4 muy bajos (<50 cél/µL) debe comenzarse dentro de las primeras 2 semanas; con CD4 más altos puede diferirse hasta unas 8 semanas. La excepción es la meningitis tuberculosa, en la que el inicio se retrasa por el alto riesgo de síndrome inflamatorio de reconstitución inmune grave."},
    {"question": "¿Por qué algunos antirretrovirales deben evitarse en pacientes con insuficiencia renal significativa?", "reference": "Algunos antirretrovirales son nefrotóxicos o se eliminan por vía renal y se acumulan al disminuir el filtrado glomerular. El tenofovir disoproxilo (TDF) puede causar tubulopatía y deterioro renal y debe evitarse; los ITIAN de eliminación renal (3TC, FTC y el propio TDF) requieren ajuste de dosis según el aclaramiento. Además, potenciadores como cobicistat elevan la creatinina sérica (inhiben su secreción tubular) sin reducir el filtrado real, lo que dificulta la monitorización. Fármacos como DTG, ABC e ITINN no necesitan ajuste renal."},
    {"question": "¿Qué implicaciones tiene la infección por VIH-2 en la elección del tratamiento inicial?", "reference": "El VIH-2 es intrínsecamente resistente a los ITINN y a la enfuvirtida, y responde peor a algunos inhibidores de la proteasa, por lo que el tratamiento inicial debe basarse en 2 ITIAN combinados con un inhibidor de integrasa (o un IP/p activo), evitando los ITINN. La monitorización es más compleja por la falta de cargas virales estandarizadas."},
    {"question": "En un paciente con infección aguda por VIH, ¿por qué iniciar TAR inmediatamente puede tener beneficios inmunológicos y epidemiológicos?", "reference": "El inicio inmediato en la infección aguda preserva la función inmune, limita el tamaño del reservorio viral, reduce la activación e inflamación inmune y mejora la recuperación de los CD4 (beneficio inmunológico). Epidemiológicamente, la fase aguda cursa con carga viral muy elevada y máxima infectividad, por lo que suprimir la replicación precozmente reduce de forma importante el riesgo de transmisión."},
    {"question": "¿En qué situaciones no se recomienda profilaxis postexposición (PEP) aunque haya contacto con fluidos potencialmente infecciosos?", "reference": "No se recomienda PEP cuando la exposición no supone riesgo apreciable: fuente VIH negativa o con carga viral indetectable confirmada, contacto con fluidos no infecciosos (saliva, orina, sudor o lágrimas sin sangre visible), exposición sobre piel intacta, o cuando han transcurrido más de 72 horas desde la exposición. También puede no indicarse en exposiciones de muy bajo riesgo tras valorar el caso."},
    {"question": "¿Qué factores determinan el riesgo de transmisión de VIH tras una exposición ocupacional?", "reference": "El riesgo depende del tipo de exposición (percutánea profunda > superficial > mucosa o piel no intacta), del volumen de inóculo (aguja hueca de gran calibre, dispositivo visiblemente manchado de sangre, inserción en vaso sanguíneo), de la profundidad de la lesión y de la carga viral de la fuente (máximo en infección aguda o fracaso virológico, mínimo si está indetectable)."},
    {"question": "¿Por qué la profilaxis postexposición debe iniciarse idealmente antes de 72 horas y qué ocurre si se inicia después?", "reference": "La PEP actúa impidiendo que el virus establezca la infección antes de su diseminación sistémica, por lo que debe iniciarse cuanto antes, idealmente en las primeras horas y siempre dentro de las 72 horas. Pasado ese plazo la eficacia es muy baja y, en general, no se recomienda iniciarla."},
    {"question": "¿Qué vacunas están contraindicadas o deben evaluarse con precaución en personas con VIH con CD4 bajos?", "reference": "Las vacunas de virus vivos atenuados (triple vírica/sarampión-rubéola-parotiditis, varicela y fiebre amarilla) están contraindicadas en caso de inmunodepresión grave (CD4 <200 cél/µL o <15%) y deben posponerse hasta la recuperación inmunológica. Las vacunas inactivadas son seguras, aunque su respuesta puede estar disminuida con CD4 bajos."},
    {"question": "¿Qué factores inmunológicos pueden disminuir la inmunogenicidad de las vacunas en pacientes con VIH?", "reference": "Reducen la respuesta vacunal un recuento de CD4 bajo (nadir y actual), la carga viral detectable o la replicación activa, el grado de activación e inflamación inmune crónica, la edad avanzada y las comorbilidades. La inmunogenicidad mejora con un TAR efectivo y la recuperación inmune, por lo que conviene vacunar con CD4 altos y carga viral suprimida."},
    {"question": "Si un paciente tiene carga viral indetectable pero adherencia irregular, ¿puede seguir transmitiendo VIH?", "reference": "El principio U=U (indetectable = intransmisible) solo es válido con una supresión virológica mantenida y estable. Con adherencia irregular no puede garantizarse la indetectabilidad continua, por lo que existe riesgo de episodios de viremia (blips o rebote) y, por tanto, de transmisión; no se cumple la condición de intransmisibilidad."},
    {"question": "¿Puede un paciente con CD4 normales tener igualmente deterioro neurocognitivo asociado al VIH?", "reference": "Sí. Los trastornos neurocognitivos asociados al VIH (HAND) pueden presentarse pese a tener CD4 normales y carga viral suprimida, debido a neuroinflamación persistente, daño acumulado por un nadir de CD4 bajo, replicación en el sistema nervioso central y comorbilidades. El recuento actual de CD4 no excluye el diagnóstico."},
    {"question": "Si la adherencia es alta pero encontramos un fracaso virológico, ¿qué causas no relacionadas con adherencia deben investigarse?", "reference": "Deben investigarse interacciones farmacológicas que reduzcan las concentraciones del TAR (inductores enzimáticos, antiácidos o cationes con los INI), problemas de absorción, una posología o requisitos con alimentos incorrectos, la presencia de resistencia transmitida o preexistente/archivada y una potencia o barrera genética insuficiente del régimen."},
    {"question": "¿Por qué la introducción del TAR redujo la incidencia de demencia asociada al VIH, pero no eliminó los trastornos neurocognitivos?", "reference": "El TAR controla la replicación sistémica y redujo drásticamente la forma más grave (la demencia asociada al VIH), pero persisten las formas leves y moderadas por neuroinflamación crónica, penetración variable de los fármacos en el sistema nervioso central, daño neuronal previo (nadir bajo de CD4) y factores comórbidos; el VIH establece efectos en el SNC que el TAR no revierte por completo."},
    {"question": "Si un paciente con VIH tiene carga viral indetectable en plasma, pero presenta replicación viral detectable en LCR, ¿puede desarrollar deterioro neurocognitivo relacionado con el VIH?", "reference": "Sí. El llamado escape viral en LCR refleja una replicación compartimentada en el sistema nervioso central, por penetración insuficiente de los fármacos o resistencia local, que puede producir daño neuronal y deterioro neurocognitivo a pesar de mantener la carga viral plasmática indetectable."},
    {"question": "Si una persona con VIH mantiene carga viral indetectable durante años, ¿puede seguir teniendo inflamación crónica sistémica que aumente su riesgo cardiovascular?", "reference": "Sí. Pese a la supresión virológica persiste una activación inmune e inflamación crónica residual (por translocación microbiana, coinfecciones como CMV y el reservorio viral persistente) que, junto con efectos metabólicos de algunos antirretrovirales y los factores de riesgo clásicos, mantiene un riesgo cardiovascular aumentado."},
    {"question": "¿Puede un paciente con CD4 elevados y carga viral indetectable desarrollar trastornos neurocognitivos asociados al VIH (HAND), y qué mecanismos lo explicarían?", "reference": "Sí. Los mecanismos incluyen el establecimiento temprano del reservorio viral en el sistema nervioso central, la neuroinflamación y activación microglial persistentes, la penetración variable del TAR en el SNC, el daño acumulado por un nadir de CD4 bajo y las comorbilidades. El buen control periférico (CD4 altos y carga viral indetectable) no garantiza el control en el compartimento del SNC."},
    {"question": "Si un paciente presenta fracaso virológico con buena adherencia documentada, ¿qué papel puede tener la resistencia viral preexistente o transmitida?", "reference": "La resistencia transmitida (adquirida en la primoinfección) o la archivada de exposiciones previas a antirretrovirales puede comprometer la actividad de fármacos del régimen aunque la adherencia sea correcta, especialmente con fármacos de baja barrera como los ITINN. Por ello se recomienda realizar estudio de resistencias basal y de nuevo en el momento del fracaso."},
    {"question": "Si un paciente tiene síndrome metabólico mientras toma TAR, ¿cómo diferenciar si la causa es la infección por VIH, los efectos del tratamiento o los factores clásicos de riesgo cardiovascular?", "reference": "Hay que valorar las tres contribuciones: la propia infección por VIH (inflamación y activación inmune crónicas), los efectos del TAR (algunos IP, ciertos inhibidores de integrasa y TAF se asocian a aumento de peso y dislipemia) y los factores clásicos (dieta, sedentarismo, genética, edad). La diferenciación se apoya en la relación temporal con el inicio o cambio del TAR, el perfil metabólico conocido de cada fármaco y la evaluación del riesgo individual."},
    {"question": "Si un paciente tiene carga viral indetectable pero con adherencia intermitente, ¿qué riesgo existe de rebote viral, desarrollo de resistencia y pérdida futura de supresión virológica?", "reference": "La adherencia intermitente conlleva riesgo de rebote virológico, de selección de mutaciones de resistencia (sobre todo con fármacos de baja barrera genética) y de pérdida progresiva de opciones terapéuticas y de la supresión a largo plazo. La adherencia subóptima es la principal causa de fracaso virológico."},
    {"question": "Si un paciente con VIH tiene factores clásicos de riesgo cardiovascular controlados, ¿por qué sigue teniendo mayor riesgo de enfermedad cardiovascular que la población general?", "reference": "Por la inflamación e inmunoactivación crónicas asociadas al VIH (presentes incluso con carga viral suprimida), la disfunción endotelial, los efectos metabólicos de algunos antirretrovirales, la mayor prevalencia de tabaquismo y el daño vascular acumulado. Estos factores hacen que el riesgo real supere al estimado solo con los factores clásicos."},
    {"question": "¿Puede un paciente con VIH con carga viral indetectable tener reservorios virales activos en tejidos, y qué implicaciones tiene esto para la curación del VIH?", "reference": "Sí. El VIH persiste integrado en células latentes (linfocitos T CD4 de memoria) y en santuarios tisulares (tejido linfoide, sistema nervioso central, tracto digestivo). El TAR no elimina ese reservorio, lo que explica el rebote viral al suspender el tratamiento y constituye la principal barrera para la curación del VIH."},
    {"question": "Si el TAR ha reducido drásticamente las complicaciones neurológicas graves del VIH, ¿por qué siguen observándose formas leves o moderadas de trastornos neurocognitivos en un porcentaje significativo de pacientes?", "reference": "Porque persisten mecanismos que el TAR no controla por completo: neuroinflamación crónica, penetración variable de los fármacos en el sistema nervioso central, reservorio viral en el SNC, daño neuronal previo (nadir de CD4 bajo) y comorbilidades (edad, factores vasculares, consumo de tóxicos). Estos factores mantienen formas leves y moderadas pese a la desaparición de las complicaciones graves."},
    {"question": "Si un paciente presenta fracaso virológico con viremia baja persistente (<200 copias/mL), ¿es obligatorio cambiar el tratamiento o primero deben evaluarse otros factores?", "reference": "No es obligatorio cambiar de inmediato. Primero hay que confirmar y repetir la determinación, evaluar la adherencia, descartar interacciones farmacológicas y blips, y valorar un estudio de resistencias; viremias persistentes <200 copias/mL a menudo no se consideran fracaso virológico establecido. La decisión de cambiar se toma según la evolución y el resultado del estudio de resistencias."},
    {"question": "En un paciente con historia de fracaso previo con ITINN, ¿es seguro cambiar a CAB+RPV inyectable si actualmente está suprimido?", "reference": "No. La pauta inyectable de cabotegravir + rilpivirina está contraindicada si existe evidencia actual o previa de resistencia o de fracaso virológico previo a los ITINN o a los inhibidores de integrasa, aunque el paciente esté virológicamente suprimido en el momento actual, por el elevado riesgo de fracaso y de desarrollo de resistencias."},
    {"question": "Si un paciente con VIH tiene insuficiencia renal avanzada, ¿qué antirretrovirales deberían evitarse o ajustarse de dosis?", "reference": "Debe evitarse el tenofovir disoproxilo (TDF) por su nefrotoxicidad, y ajustar la dosis de los ITIAN de eliminación renal (3TC, FTC y TDF) según el filtrado glomerular; el TAF se utiliza con precaución según la función renal. Dolutegravir, abacavir y los ITINN no requieren ajuste renal y facilitan el manejo. Hay que recordar que cobicistat eleva la creatinina sin reducir el filtrado real."},
    {"question": "En un paciente con carga viral indetectable y múltiples comorbilidades, ¿qué factores deben considerarse antes de simplificar el tratamiento?", "reference": "Antes de simplificar hay que valorar la historia de resistencias y de fracasos previos, la coinfección por VHB, las interacciones farmacológicas derivadas de la polifarmacia, la función renal y hepática, la adherencia esperable y que se cumplan los criterios de supresión estable. Se elige la pauta de menor toxicidad y barrera genética adecuada al perfil del paciente."},
    {"question": "Si un paciente tiene adherencia irregular, ¿por qué algunos regímenes con inhibidores de integrasa tienen mayor barrera genética que otros?", "reference": "Los inhibidores de integrasa de segunda generación (dolutegravir, bictegravir) tienen mayor barrera genética que los de primera generación (raltegravir, elvitegravir), pues requieren la acumulación de varias mutaciones para perder eficacia. Por eso se prefieren cuando la adherencia es irregular o se busca robustez frente al desarrollo de resistencias."},
    {"question": "En un paciente con fracaso virológico y múltiples mutaciones de resistencia, ¿cuál es el principio fundamental para diseñar el nuevo régimen?", "reference": "El principio fundamental es construir un régimen con al menos dos (preferiblemente tres) fármacos plenamente activos, seleccionados según el estudio de resistencias histórico y actual, recurriendo si es necesario a nuevas clases o mecanismos de acción. Nunca debe añadirse un único fármaco activo a un régimen que está fracasando."},
    {"question": "Si un paciente con VIH tiene hepatitis B crónica, ¿qué implicaciones tiene esto para elegir el TAR?", "reference": "El régimen debe incluir dos fármacos activos frente al VHB, habitualmente tenofovir (TAF o TDF) junto con 3TC o FTC. No deben usarse pautas sin cobertura frente al VHB, y no debe suspenderse el tenofovir/componente anti-VHB sin una alternativa activa por el riesgo de reactivación y hepatitis grave; esto contraindica pautas duales como DTG/3TC."},
    {"question": "¿Por qué en pacientes con VIH se recomienda evitar la monoterapia antirretroviral, incluso con fármacos potentes?", "reference": "Porque la monoterapia, incluso con fármacos potentes, no mantiene una supresión duradera y selecciona resistencias, ya que ejerce una presión farmacológica insuficiente sobre la elevada tasa de replicación y mutación del VIH. El principio terapéutico es combinar siempre varios fármacos activos (TAR combinado)."},
    {"question": "Si un paciente suprimido cambia de TAR por toxicidad, ¿qué parámetros deben monitorizarse tras el cambio para confirmar eficacia?", "reference": "Tras el cambio debe confirmarse el mantenimiento de la supresión virológica mediante la carga viral (por ejemplo, en torno a las 4 semanas y después de forma periódica), vigilar la resolución de la toxicidad que motivó el cambio, evaluar la tolerancia y la adherencia al nuevo régimen y vigilar posibles nuevas interacciones o efectos adversos."},
    {"question": "Si un paciente tiene interacciones farmacológicas complejas por polifarmacia, ¿qué clases de antirretrovirales suelen ser más fáciles de manejar?", "reference": "Los inhibidores de integrasa sin potenciador (dolutegravir, bictegravir, raltegravir) presentan menos interacciones que las pautas potenciadas con ritonavir o cobicistat (inhibidores enzimáticos potentes) o que los ITINN (inductores/inhibidores enzimáticos), por lo que son la opción más fácil de manejar en pacientes polimedicados. Debe recordarse la interacción de los INI con cationes y antiácidos."},
    {"question": "Si un paciente presenta rebote viral tras años de supresión, ¿cuáles son las tres causas principales que deben investigarse antes de cambiar TAR?", "reference": "Las tres causas principales son: (1) adherencia subóptima o interrupciones del tratamiento, que es la más frecuente; (2) interacciones farmacológicas o problemas de absorción que reducen las concentraciones del TAR; y (3) resistencia viral, preexistente/archivada o de nueva aparición. Debe confirmarse el rebote y realizarse un estudio de resistencias antes de cambiar el tratamiento."},

    # --- Questions with SPECIFIC TERMS (drugs, doses, abbreviations) to stress the
    #     lexical/hybrid (BM25) search. Added in Phase 2.
    {"question": "¿Qué inhibidores de la integrasa no pueden administrarse junto con rifampicina?", "reference": "Con rifampicina no pueden administrarse elvitegravir/cobicistat, bictegravir (BIC) ni cabotegravir (CAB), ni raltegravir en su pauta de 1200 mg/24h; tampoco los ITINN distintos de efavirenz (RPV, ETR, DOR) ni los inhibidores de la proteasa. Entre los inhibidores de integrasa, las excepciones utilizables son dolutegravir (con ajuste de dosis) y raltegravir 800 mg/12h."},
    {"question": "¿A qué dosis debe administrarse dolutegravir cuando se combina con rifampicina?", "reference": "Con rifampicina, dolutegravir debe administrarse a 50 mg cada 12 horas (en pacientes sin resistencia a inhibidores de integrasa), manteniendo esa dosis hasta 2 semanas después de finalizar la rifampicina."},
    {"question": "¿Está recomendado el uso de tenofovir alafenamida (TAF) junto con rifampicina?", "reference": "No. No está recomendado el uso de tenofovir alafenamida (TAF) con rifampicina porque la rifampicina reduce sus concentraciones plasmáticas; como ITIAN con rifampicina se prefieren TDF, ABC, 3TC o FTC."},
    {"question": "¿Qué resultado de la prueba HLA-B*5701 contraindica el abacavir?", "reference": "Un resultado positivo de HLA-B*5701 contraindica el abacavir: no debe prescribirse ABC si la prueba es positiva, por el alto riesgo de reacción de hipersensibilidad."},
    {"question": "¿Cuál es el tercer fármaco de elección junto a rifampicina en la coinfección tuberculosis-VIH?", "reference": "El tercer fármaco de elección junto a rifampicina es efavirenz (EFV) a dosis estándar (A-I); como alternativas se recomiendan raltegravir 800 mg/12h o dolutegravir 50 mg/12h."},
    {"question": "¿Cuál es la pauta de TAR de inicio preferente en un paciente con VIH y tuberculosis en tratamiento con rifampicina?", "reference": "La pauta preferente es tenofovir DF/emtricitabina (o abacavir/lamivudina) a dosis habituales más efavirenz a dosis de 600 mg/día (A-I)."},
    {"question": "¿En qué pacientes es adecuado el cambio a la terapia dual dolutegravir más lamivudina (DTG + 3TC)?", "reference": "El cambio a dolutegravir + lamivudina (DTG + 3TC) es una opción adecuada en pacientes con replicación viral suprimida que quieran simplificar o evitar efectos adversos, sin resistencia conocida o sospechada a lamivudina ni a inhibidores de integrasa (y sin coinfección por VHB)."},
    {"question": "¿A qué dosis se administra raltegravir cuando se combina con rifampicina?", "reference": "Con rifampicina, raltegravir se administra a dosis de 800 mg cada 12 horas."},
]


# ===========================================================================
# MULTI-HOP SET (Phase 4)  ->  the target of the A/B comparison.
#   These questions are deliberately MULTI-HOP: answering each one correctly requires
#   chaining facts that live in DIFFERENT guides or sections (e.g. mixing the pregnancy
#   guide with the TB guide and the drug-interaction part of the TAR consensus). Classic
#   single-shot RAG tends to retrieve evidence for only one hop and miss the rest.
#
#   CAVEAT (same as GOLDEN_SET / Phase 0): the references were drafted by the model from
#   the guidelines, NOT reviewed by a clinician. The absolute scores are indicative; what
#   matters is the comparison BETWEEN pipelines on the SAME set.
#
#   Extra fields per entry (ignored by RAGAS, used to slice results):
#     - guides: source guides whose content must be combined.
#     - hops:   number of distinct reasoning steps the question requires.
#   Guide labels = source_file in chunks.jsonl: TAR_2022, VIH_TB, VIH_embarazo,
#   adherencia, medicina_preventiva, profilaxis, neurocognitivo.
# ===========================================================================
MULTIHOP_SET = [
    {"question": "En una gestante con VIH que además inicia tratamiento para tuberculosis con rifampicina, ¿qué régimen antirretroviral es preferible y qué interacción hay que evitar?",
     "reference": "Hay que combinar dos exigencias: un TAR seguro en el embarazo y compatible con rifampicina. La rifampicina es un inductor potente que reduce las concentraciones de la mayoría de antirretrovirales: no pueden usarse bictegravir, elvitegravir/cobicistat ni cabotegravir, y los IP potenciados con ritonavir/cobicistat tampoco. La opción preferida es 2 ITIAN (tenofovir/emtricitabina o abacavir/lamivudina) más un fármaco compatible: dolutegravir con la dosis ajustada a 50 mg/12 h (manteniéndola hasta 2 semanas tras finalizar la rifampicina) o efavirenz a dosis estándar, ambos con experiencia de uso en gestación. Debe vigilarse la carga viral por el riesgo de infradosificación y la transmisión vertical.",
     "guides": ["VIH_embarazo", "VIH_TB", "TAR_2022"], "hops": 3},
    {"question": "Un paciente suprimido con tenofovir/emtricitabina + un tercer fármaco y coinfección por VHB quiere simplificar a dolutegravir + lamivudina. ¿Es adecuado?",
     "reference": "No. La coinfección por VHB exige mantener dos fármacos activos frente al VHB (habitualmente tenofovir —TAF o TDF— junto con 3TC o FTC). La pauta dual DTG/3TC no cubre adecuadamente el VHB y, al retirar el tenofovir, existe riesgo de reactivación de la hepatitis B con posible hepatitis grave. Por tanto, la coinfección por VHB contraindica esta simplificación; debe mantenerse un régimen con cobertura anti-VHB.",
     "guides": ["TAR_2022"], "hops": 2},
    {"question": "En un paciente con insuficiencia renal avanzada que además tiene hepatitis B crónica, ¿cómo se resuelve el conflicto para cubrir el VHB sin dañar el riñón?",
     "reference": "Hay un conflicto: el VHB exige tenofovir (activo frente al VHB), pero el tenofovir disoproxilo (TDF) es nefrotóxico y debe evitarse en insuficiencia renal. La solución es usar tenofovir alafenamida (TAF), que mantiene actividad anti-VHB con mucha menor toxicidad renal y se puede emplear en grados moderados de insuficiencia (con precaución según el filtrado), en lugar de TDF. Deben ajustarse al filtrado glomerular los ITIAN de eliminación renal y recordar que cobicistat eleva la creatinina sin reducir el filtrado real. No debe retirarse la cobertura anti-VHB sin una alternativa activa.",
     "guides": ["TAR_2022"], "hops": 2},
    {"question": "Tras una exposición ocupacional de riesgo en una profesional sanitaria embarazada, ¿está indicada la profilaxis postexposición y cómo influye el embarazo en la pauta?",
     "reference": "El embarazo no contraindica la PEP: si la exposición es de riesgo y la fuente es VIH positiva (o de serología desconocida con riesgo), debe iniciarse cuanto antes, idealmente en las primeras horas y siempre dentro de las 72 horas. Se eligen antirretrovirales con experiencia de seguridad en gestación, evitando los desaconsejados en embarazo, y manteniendo la pauta 4 semanas con seguimiento serológico. Debe valorarse el riesgo-beneficio e informar a la paciente, pero el embarazo en sí no es motivo para no administrar la PEP.",
     "guides": ["profilaxis", "VIH_embarazo"], "hops": 2},
    {"question": "En un paciente con tuberculosis y CD4 muy bajos, ¿cuándo iniciar el TAR y cómo evitar a la vez el síndrome de reconstitución inmune y las interacciones con la rifampicina?",
     "reference": "Con CD4 muy bajos (<50 cél/µL) el TAR debe iniciarse precozmente, dentro de las primeras 2 semanas del tratamiento antituberculoso, para reducir la mortalidad; con CD4 más altos puede diferirse hasta unas 8 semanas. La excepción es la meningitis tuberculosa, donde se retrasa el inicio por el alto riesgo de SIRI grave. Para evitar las interacciones con la rifampicina (inductor potente), se eligen fármacos compatibles: efavirenz a dosis estándar o dolutegravir 50 mg/12 h, evitando bictegravir, elvitegravir/cobicistat, cabotegravir y los IP potenciados.",
     "guides": ["VIH_TB", "TAR_2022"], "hops": 3},
    {"question": "En un paciente que inicia TAR con CD4 muy bajos y va a viajar a una zona con fiebre amarilla, ¿puede vacunarse y cuándo?",
     "reference": "La vacuna de la fiebre amarilla es de virus vivos atenuados y está contraindicada con inmunodepresión grave (CD4 <200 cél/µL o <15%). Por tanto, en un paciente con CD4 muy bajos no debe administrarse de inicio: hay que esperar a la recuperación inmune con el TAR (CD4 ≥200 de forma estable) y entonces valorar la vacunación antes del viaje. Mientras tanto, las vacunas inactivadas sí son seguras, aunque su respuesta puede estar disminuida con CD4 bajos.",
     "guides": ["medicina_preventiva", "TAR_2022"], "hops": 2},
    {"question": "En un paciente con deterioro neurocognitivo y escape virológico en LCR pese a carga viral plasmática indetectable, ¿qué hay que tener en cuenta al elegir el TAR?",
     "reference": "El escape viral en LCR refleja replicación compartimentada en el SNC por penetración insuficiente de los fármacos o resistencia local, y puede causar daño neurocognitivo pese a la supresión plasmática. Al elegir el TAR conviene tener en cuenta la penetración de los fármacos en el SNC y orientar el régimen, idealmente guiado por un estudio de resistencias en LCR, hacia fármacos con buena penetración en el sistema nervioso central. Debe confirmarse y monitorizarse la carga viral en LCR.",
     "guides": ["neurocognitivo", "TAR_2022"], "hops": 2},
    {"question": "Un paciente con adherencia irregular y antecedente de fracaso con un ITINN quiere pasarse a la pauta inyectable de cabotegravir + rilpivirina. ¿Es buena idea?",
     "reference": "No. La pauta inyectable de cabotegravir + rilpivirina está contraindicada si hay evidencia actual o previa de resistencia o fracaso a los ITINN o a los inhibidores de integrasa, como es el caso (fracaso previo a un ITINN), aunque ahora esté suprimido, por el alto riesgo de fracaso y de resistencias. Además, en adherencia irregular se prefieren regímenes con alta barrera genética (dolutegravir, bictegravir); el inyectable de acción prolongada no resuelve por sí solo el problema de adherencia y exige cumplimiento estricto de las visitas.",
     "guides": ["adherencia", "TAR_2022"], "hops": 2},
    {"question": "En una gestante con VIH y adherencia irregular, ¿qué riesgos añade la mala adherencia y qué hay que reforzar para prevenir la transmisión vertical?",
     "reference": "La adherencia irregular impide alcanzar y mantener la supresión virológica, lo que aumenta el riesgo de transmisión vertical (que es mínimo con carga viral indetectable) y el de selección de resistencias. Hay que reforzar intensamente la adherencia, monitorizar estrechamente la carga viral —en especial cerca del parto—, y si la viremia no está controlada al final de la gestación valorar medidas adicionales de profilaxis de la transmisión vertical (incluida la pauta intraparto y la profilaxis al recién nacido) y la vía del parto según la carga viral.",
     "guides": ["VIH_embarazo", "adherencia"], "hops": 2},
    {"question": "En un paciente con fracaso virológico y múltiples resistencias que además desarrolla tuberculosis, ¿cómo se diseña el rescate teniendo en cuenta la rifampicina?",
     "reference": "El principio del rescate es construir un régimen con al menos dos (preferiblemente tres) fármacos plenamente activos según el estudio de resistencias histórico y actual, sin añadir nunca un solo fármaco activo a un régimen que fracasa. La tuberculosis añade la restricción de la rifampicina: hay que evitar los antirretrovirales cuyas concentraciones reduce (bictegravir, elvitegravir/cobicistat, cabotegravir, IP potenciados) y, si se necesitan, valorar sustituir la rifampicina por rifabutina para poder usar IP potenciados, o ajustar dolutegravir a 50 mg/12 h. La selección de fármacos activos manda, y se adapta la pauta antituberculosa para hacerla compatible.",
     "guides": ["TAR_2022", "VIH_TB"], "hops": 3},
    {"question": "En un paciente polimedicado con riesgo cardiovascular que quiere simplificar el TAR, ¿qué clase de antirretrovirales facilita el manejo y qué considerar por el perfil metabólico?",
     "reference": "Los inhibidores de integrasa sin potenciador (dolutegravir, bictegravir, raltegravir) tienen menos interacciones que las pautas potenciadas con ritonavir/cobicistat o que los ITINN, por lo que son la opción más fácil en pacientes polimedicados. Por el perfil metabólico hay que tener en cuenta que algunos inhibidores de integrasa y el TAF se asocian a aumento de peso y dislipemia, lo que importa en un paciente con riesgo cardiovascular; conviene elegir la pauta de menor impacto metabólico y adecuada barrera genética, valorando también el riesgo cardiovascular residual asociado a la propia infección. Antes de simplificar deben descartarse resistencias previas, coinfección por VHB y garantizar la adherencia.",
     "guides": ["TAR_2022"], "hops": 2},
    {"question": "¿Por qué un paciente con buen control inmunovirológico (CD4 altos, carga viral indetectable) puede aun así presentar deterioro neurocognitivo, y qué relación tiene esto con la adherencia?",
     "reference": "El buen control periférico no garantiza el control en el SNC: pueden persistir trastornos neurocognitivos (HAND) por establecimiento temprano del reservorio en el SNC, neuroinflamación crónica, penetración variable del TAR, daño acumulado por un nadir de CD4 bajo y comorbilidades. Esto se relaciona con la adherencia en doble sentido: el deterioro cognitivo dificulta la adherencia al tratamiento (olvidos, errores de dosis), y a su vez una adherencia subóptima favorece la replicación residual y el daño; por ello conviene detectar el deterioro cognitivo y reforzar y simplificar la adherencia en estos pacientes.",
     "guides": ["neurocognitivo", "adherencia"], "hops": 2},
    {"question": "En una gestante que llega al tercer trimestre con carga viral detectable, ¿qué medidas de profilaxis de la transmisión vertical deben adoptarse?",
     "reference": "Con carga viral detectable cerca del parto el riesgo de transmisión vertical aumenta y deben reforzarse las medidas: optimizar y reforzar el TAR materno y la adherencia, repetir la carga viral, administrar zidovudina intravenosa intraparto, valorar la cesárea electiva si la carga viral permanece elevada (en torno a >1000 copias/mL) y aplicar profilaxis antirretroviral al recién nacido (combinada si el riesgo es alto). El objetivo es alcanzar la máxima reducción posible de la viremia antes del parto.",
     "guides": ["VIH_embarazo"], "hops": 2},
    {"question": "Tras una exposición sexual de riesgo, ¿cuándo NO está indicada la PEP y qué seguimiento se hace si sí se indica?",
     "reference": "No se indica PEP cuando no hay riesgo apreciable: fuente VIH negativa o con carga viral indetectable confirmada, contacto con fluidos no infecciosos, exposición sobre piel intacta, o cuando han transcurrido más de 72 horas. Si está indicada, se inicia cuanto antes (idealmente en las primeras horas, siempre <72 h) con una pauta de tres fármacos durante 4 semanas y se realiza seguimiento serológico del VIH (basal y a las 4-6 semanas y 3 meses, con esquemas según la prueba), además de cribado de otras ITS y de VHB/VHC y valoración de adherencia y tolerancia.",
     "guides": ["profilaxis"], "hops": 2},
    {"question": "En un paciente con VIH-2 que además requiere un régimen donde se usarían ITINN, ¿qué limitación hay y cómo afecta a la elección del tratamiento?",
     "reference": "El VIH-2 es intrínsecamente resistente a los ITINN (y a la enfuvirtida) y responde peor a algunos inhibidores de la proteasa, por lo que cualquier estrategia basada en ITINN es inválida frente al VIH-2. El tratamiento debe basarse en 2 ITIAN combinados con un inhibidor de integrasa (o un IP/p activo), evitando los ITINN. La monitorización es más compleja por la falta de cargas virales estandarizadas para el VIH-2.",
     "guides": ["TAR_2022"], "hops": 2},
    {"question": "En un paciente con tuberculosis que necesita un inhibidor de la proteasa potenciado en su TAR, ¿cómo se compatibiliza con el tratamiento antituberculoso?",
     "reference": "La rifampicina no puede combinarse con inhibidores de la proteasa potenciados con ritonavir o cobicistat porque reduce drásticamente sus concentraciones. Para compatibilizarlos se sustituye la rifampicina por rifabutina (ajustando su dosis), que tiene mucha menor inducción enzimática y permite mantener el IP potenciado. Alternativamente, si se conserva la rifampicina, debe rediseñarse el TAR hacia fármacos compatibles (efavirenz a dosis estándar o dolutegravir 50 mg/12 h) en lugar del IP potenciado.",
     "guides": ["VIH_TB", "TAR_2022"], "hops": 2},
]


# Dataset under evaluation: GOLDEN_SET (single-hop regression) or MULTIHOP_SET (Phase 4).
DATASET = MULTIHOP_SET


def build_dataset(dataset: list[dict], retriever) -> tuple[EvaluationDataset, list[float]]:
    """Run each question through `retriever` + the shared generator and assemble the
    dataset RAGAS consumes. Only retrieval varies between pipelines (generation is fixed)
    so the comparison is fair. Also returns the per-question latency (retrieval +
    generation) for the velocity axis of the A/B."""
    rows = []
    omitted = []
    latencies = []
    for i, case in enumerate(dataset, 1):
        question = case["question"].strip()
        reference = case.get("reference", "").strip()

        if not reference:
            omitted.append(i)
            continue

        # --- pipeline: swappable retrieval + (optional) shared generation ---
        t0 = time.perf_counter()
        payloads = retriever(question)                      # list[dict]
        # In RETRIEVAL_ONLY we skip generation: the kept metrics (recall/precision) only
        # need the retrieved contexts + reference, not a generated answer.
        answer_text = ""
        if not RETRIEVAL_ONLY:
            _, formatted_context = build_context(payloads)
            answer_text = generate_answer(question, formatted_context)["answer"]
        dt = time.perf_counter() - t0
        latencies.append(dt)

        rows.append({
            "user_input": question,
            "retrieved_contexts": [p["text"] for p in payloads if p],
            "response": answer_text,
            "reference": reference,
        })
        print(f"[{i}/{len(dataset)}] processed ({dt:.1f}s)")

    if omitted:
        print(f"\nSkipped (no reference filled in): {omitted}")
    if not rows:
        raise SystemExit("No question has a reference. Fill in the dataset.")

    print(f"\n{len(rows)} questions ready to evaluate.\n")
    return EvaluationDataset.from_list(rows), latencies


def main():
    retriever = get_pipeline(PIPELINE)
    mode = "retrieval-only" if RETRIEVAL_ONLY else "full"
    print(f"Pipeline: {PIPELINE} | mode: {mode} | dataset: {len(DATASET)} preguntas\n")
    dataset, latencies = build_dataset(DATASET, retriever)

    print(f"Evaluating with judge: {JUDGE_MODEL}\n")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-large")
    )

    # answer_relevancy omitted on purpose: it is misleading on Spanish answers (Phase 0
    # artifact ~0.42) and adds cost without signal. The retrieval metrics (precision/recall)
    # are the axes a multi-hop A/B turns on; faithfulness (grounding) is only added in the
    # full run, since it needs the generated answer.
    if RECALL_ONLY:
        metrics = [LLMContextRecall()]
    elif RETRIEVAL_ONLY:
        metrics = [LLMContextPrecisionWithReference(), LLMContextRecall()]
    else:
        metrics = [Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()]

    # Default RAGAS concurrency (16 workers, 180s timeout) overwhelms the judge -> NaN.
    # 8 workers + a long timeout + retries is the stable point (16 still timed out on the
    # heavy precision metric). Generation (gpt-4o) is sequential anyway in build_dataset.
    run_config = RunConfig(timeout=600, max_workers=8, max_retries=10)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )
    assert isinstance(result, EvaluationResult)

    print("\n=== Aggregate scores ===")
    print(result)

    df = result.to_pandas()
    suffix = f"{PIPELINE}_retrieval" if RETRIEVAL_ONLY else PIPELINE
    out_csv = f"resultados_ragas_{suffix}.csv"     # per-pipeline/mode file so runs don't clash
    df.to_csv(out_csv, index=False)
    print(f"\nDetail saved to {out_csv}")

    pd.set_option("display.max_colwidth", 60)
    print("\n=== Means per metric ===")
    print(df.select_dtypes("number").mean())

    # Velocity axis of the A/B: latency per query (retrieval + generation).
    if latencies:
        print("\n=== Latency (s/query) ===")
        print(f"mean={statistics.mean(latencies):.1f}  median={statistics.median(latencies):.1f}  "
              f"max={max(latencies):.1f}  total={sum(latencies):.0f}")


if __name__ == "__main__":
    main()
