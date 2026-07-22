"""RAGAS evaluation of the clinical HIV RAG.

ONE evaluation set (EVAL_SET); every question carries a "tier" (simple / single_hop /
multihop / adversarial) so the full RAGAS suite can be sliced by question type. To run the
A/B, pick a retriever with PIPELINE; only retrieval varies (generation is shared), which is
what makes the comparison fair.

    PIPELINE=graph python evaluation.py     # any mode in retrieval/registry.py
    -> results/ragas_results_<pipeline>.csv (per-question detail) + per-tier means.
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
from rag import build_context, generate_answer, chat_model, embeddings_model, COLLECTION_HYBRID
from retrieval.registry import VALID_MODES, get_search

# ===========================================================================
# A/B CONFIG — which retrieval pipeline runs over the single EVAL_SET (generation is shared,
# so only retrieval varies). The available modes and what each one does are declared in
# retrieval/registry.py. Which Qdrant collection retrieval hits is set by QDRANT_COLLECTION
# (rag.py reads it; DEFAULT the Contextual-Retrieval build), so you can A/B collections too.
# The run always uses the FULL RAGAS suite: gpt-4o generation (sequential, ~30k-TPM bound) +
# a judge call per chunk for precision, so it is the expensive one — budget for it. Probe
# with EVAL_SAMPLE=3 (stratified, cents) before committing to a full run.
# ===========================================================================
PIPELINE = os.environ.get("PIPELINE", "baseline")
if PIPELINE not in VALID_MODES:   # fail before building the dataset, not after
    raise SystemExit(f"Unknown PIPELINE: {PIPELINE!r} (use {' | '.join(VALID_MODES)})")


def get_pipeline(name: str):
    """Retrieval function for the chosen pipeline, from the mode catalogue. The registry
    imports each mode lazily, so evaluating 'baseline' never loads LightRAG or a graph store.
    Every mode is called the same way — retriever(question) — so the A/B compares the
    SELECTION mechanism with generation and metrics held fixed."""
    return get_search(name)

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


# JUDGE — gpt-4o-mini (cheap, for iterating) vs gpt-4o (the number you report).
STRONG_JUDGE = False
CHEAP_MODEL = "gpt-4o-mini"
STRONG_MODEL = "gpt-4o"
JUDGE_MODEL = STRONG_MODEL if STRONG_JUDGE else CHEAP_MODEL


# ===========================================================================
# QUESTION POOLS — each item has "question", "reference" (correct answer per the guides;
# drafted by the model, pending clinical review) and tier metadata. Empty references are
# skipped. Folded into EVAL_SET below.
# ===========================================================================
_PREV_SINGLE = [
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
_PREV_MULTI = [
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


# ===========================================================================
# PURPOSE-BUILT TIERED QUESTIONS  ->  the discriminative core of EVAL_SET.
#
#   WHY THESE EXIST: the former multi-hop pool already SATURATED graph recall (0.979),
#   so it could no longer tell two good retrievers apart (no headroom) and had no simple
#   questions to detect the "complex retriever degrades simple QA" tradeoff. These add
#   genuinely simple/atomic questions, deliberately harder multi-hop ones, and a tier of
#   adversarial/conditional questions framed so that surface similarity pulls the WRONG
#   passage — i.e. questions where even a strong retriever can fail.
#
#   SIZING: the A/B is a PAIRED comparison (same questions through every pipeline; only
#   retrieval varies), the high-power design. To detect a ~0.07 absolute difference in
#   mean context_recall with per-question difference SD ~0.20 at alpha=0.05/power=0.80 you
#   need n ~= (1.96+0.84)^2 * 0.20^2 / 0.07^2 ~= 64; a Wilcoxon (RAGAS scores are
#   non-normal, clustered near 0/1) needs ~15% more ~= 74. EVAL_SET (151) clears that with
#   margin on the AGGREGATE and gives ~30-48 per tier, enough to detect per-tier effects of
#   ~0.10 absolute — the size of the reported "simple-QA degradation" (5-10 F1). Report the
#   aggregate as the decision metric; read the tiers as direction.
#
#   FIELDS per question: tier, hops (int), guides (source guides), stress (what it
#   stresses) — all ignored by RAGAS, used only to slice results. Guide labels = source_file
#   stem in chunks.jsonl. NB the real files are "medicina preventiva.md"
#   (label medicina_preventiva) and "ManejoclinicodelasalteracionesNC.md" (neurocognitivo).
#
#   CAVEAT: references were drafted by the model FROM the guidelines, NOT reviewed by a
#   clinician. Absolute scores are indicative; the signal is the comparison BETWEEN
#   pipelines on the SAME set. PENDING clinician review.
# ===========================================================================
_TIERED_NEW = [
    # ---------------------------------------------------------------- TIER: simple (26)
    {"question": "¿Qué fármaco antirretroviral se administra por vía intravenosa durante el parto cuando la carga viral materna no está controlada?",
     "reference": "Zidovudina (AZT) intravenosa intraparto, indicada cuando la carga viral materna no está suprimida cerca del parto, como medida de profilaxis de la transmisión vertical.",
     "tier": "simple", "hops": 1, "guides": ["VIH_embarazo"], "stress": "lexical"},
    {"question": "¿Por debajo de qué cifra de CD4 están contraindicadas las vacunas de virus vivos atenuados en personas con VIH?",
     "reference": "Por debajo de 200 cél/µL (o <15%): con inmunodepresión grave las vacunas de virus vivos atenuados (triple vírica, varicela, fiebre amarilla) están contraindicadas y deben posponerse hasta la recuperación inmune.",
     "tier": "simple", "hops": 1, "guides": ["medicina_preventiva"], "stress": "numeric"},
    {"question": "¿Cuál es el plazo máximo recomendado para iniciar la profilaxis postexposición (PEP) tras una exposición de riesgo?",
     "reference": "72 horas; pasado ese plazo la eficacia es muy baja y en general no se recomienda iniciarla. Idealmente debe comenzarse en las primeras horas.",
     "tier": "simple", "hops": 1, "guides": ["profilaxis"], "stress": "numeric"},
    {"question": "¿Cuántas semanas dura una pauta completa de profilaxis postexposición al VIH?",
     "reference": "Cuatro semanas (28 días) de tratamiento antirretroviral, habitualmente con una pauta de tres fármacos.",
     "tier": "simple", "hops": 1, "guides": ["profilaxis"], "stress": "numeric"},
    {"question": "Según las guías actuales, ¿a partir de qué recuento de CD4 se recomienda iniciar el TAR en una persona con infección por VIH?",
     "reference": "Se recomienda iniciar el TAR en TODAS las personas con infección por VIH con independencia del recuento de CD4 (tratamiento universal), idealmente de forma precoz tras el diagnóstico.",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "conceptual"},
    {"question": "En la coinfección VIH-tuberculosis, ¿con qué fármaco antituberculoso se producen las interacciones más relevantes con el TAR?",
     "reference": "Con la rifampicina, un inductor enzimático potente (CYP3A4 y glucuronidación/UGT1A1) que reduce las concentraciones de numerosos antirretrovirales.",
     "tier": "simple", "hops": 1, "guides": ["VIH_TB"], "stress": "lexical"},
    {"question": "¿Qué fármaco antituberculoso puede sustituir a la rifampicina para poder mantener un inhibidor de la proteasa potenciado?",
     "reference": "La rifabutina, que tiene mucha menor capacidad de inducción enzimática y permite mantener inhibidores de la proteasa potenciados, ajustando su dosis.",
     "tier": "simple", "hops": 1, "guides": ["VIH_TB"], "stress": "lexical"},
    {"question": "En la coinfección VIH-VHB, ¿qué dos componentes con actividad frente al VHB suele incluir el régimen?",
     "reference": "Tenofovir (TAF o TDF) junto con emtricitabina o lamivudina (FTC o 3TC); el régimen debe incluir dos fármacos activos frente al VHB.",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "lexical"},
    {"question": "¿Qué mutación de resistencia se asocia clásicamente a la pérdida de actividad de lamivudina y emtricitabina?",
     "reference": "La mutación M184V/I, que confiere resistencia a lamivudina (3TC) y emtricitabina (FTC).",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "lexical"},
    {"question": "¿Qué efecto tiene cobicistat sobre la creatinina sérica y cómo se interpreta?",
     "reference": "Cobicistat eleva la creatinina sérica porque inhibe su secreción tubular, SIN reducir el filtrado glomerular real; es un aumento no progresivo que no refleja deterioro renal verdadero.",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "conceptual"},
    {"question": "¿Qué inhibidores de la integrasa se consideran de segunda generación y de alta barrera genética?",
     "reference": "Dolutegravir y bictegravir; requieren la acumulación de varias mutaciones para perder eficacia, frente a la baja barrera de raltegravir y elvitegravir.",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "lexical"},
    {"question": "¿Qué tipo de vacunas son seguras en personas con VIH con independencia del recuento de CD4?",
     "reference": "Las vacunas inactivadas (no vivas) son seguras independientemente del recuento de CD4, aunque su inmunogenicidad puede estar reducida con CD4 bajos.",
     "tier": "simple", "hops": 1, "guides": ["medicina_preventiva"], "stress": "conceptual"},
    {"question": "¿Qué fármaco antirretroviral análogo de nucleótido se asocia a tubulopatía y deterioro de la función renal?",
     "reference": "El tenofovir disoproxilo fumarato (TDF), que puede causar tubulopatía proximal y deterioro renal; en riesgo o insuficiencia renal se prefiere tenofovir alafenamida (TAF) o evitarlo.",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "lexical"},
    {"question": "¿Cómo se denomina el fenómeno de replicación del VIH detectable en el líquido cefalorraquídeo con carga viral plasmática indetectable?",
     "reference": "Escape viral en LCR (escape virológico en el sistema nervioso central): replicación compartimentada en el SNC pese a la supresión plasmática, por penetración insuficiente de fármacos o resistencia local.",
     "tier": "simple", "hops": 1, "guides": ["neurocognitivo"], "stress": "lexical"},
    {"question": "¿Qué concepto mide la capacidad de penetración de los antirretrovirales en el sistema nervioso central?",
     "reference": "El índice CPE (CSF Penetration-Effectiveness), que estima la penetración-efectividad de cada antirretroviral en el líquido cefalorraquídeo/SNC.",
     "tier": "simple", "hops": 1, "guides": ["neurocognitivo"], "stress": "lexical"},
    {"question": "¿Frente a qué clase de antirretrovirales es intrínsecamente resistente el VIH-2?",
     "reference": "El VIH-2 es intrínsecamente resistente a los ITINN (inhibidores no nucleósidos de la transcriptasa inversa) y a la enfuvirtida, y responde peor a algunos inhibidores de la proteasa.",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "conceptual"},
    {"question": "¿Cuál es la principal causa de fracaso virológico en pacientes en tratamiento antirretroviral?",
     "reference": "La adherencia subóptima al tratamiento es la principal causa de fracaso virológico.",
     "tier": "simple", "hops": 1, "guides": ["adherencia"], "stress": "conceptual"},
    {"question": "¿Qué significa el principio U=U en el contexto del VIH?",
     "reference": "U=U (indetectable = intransmisible): una persona con VIH que mantiene una carga viral indetectable de forma estable y mantenida con el TAR no transmite el VIH por vía sexual.",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "conceptual"},
    {"question": "Cuando la carga viral materna permanece elevada cerca del término, ¿qué vía del parto se recomienda valorar?",
     "reference": "La cesárea electiva (programada), que se valora cuando la carga viral materna permanece elevada (en torno a >1000 copias/mL) cerca del parto para reducir la transmisión vertical.",
     "tier": "simple", "hops": 1, "guides": ["VIH_embarazo"], "stress": "numeric"},
    {"question": "¿Qué prueba de laboratorio debe realizarse de forma basal antes de iniciar el TAR para detectar resistencias transmitidas?",
     "reference": "Un estudio de resistencias genotípico basal, para detectar resistencias transmitidas que condicionen la elección del primer régimen.",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "conceptual"},
    {"question": "En un paciente con VIH y tuberculosis sin TAR previo, ¿qué tratamiento se inicia primero?",
     "reference": "Se inicia primero el tratamiento antituberculoso y, a continuación, el TAR de forma precoz (el momento exacto depende del recuento de CD4), no de forma simultánea.",
     "tier": "simple", "hops": 1, "guides": ["VIH_TB"], "stress": "conceptual"},
    {"question": "¿Qué profilaxis se administra al recién nacido de madre con VIH?",
     "reference": "Profilaxis antirretroviral al recién nacido expuesto (habitualmente zidovudina; pauta combinada de varios fármacos si el riesgo de transmisión es alto), iniciada precozmente tras el nacimiento.",
     "tier": "simple", "hops": 1, "guides": ["VIH_embarazo"], "stress": "lexical"},
    {"question": "¿Cómo se define la supresión virológica en una persona en TAR?",
     "reference": "Como una carga viral plasmática por debajo del límite de detección del ensayo (habitualmente <50 copias/mL), mantenida en el tiempo.",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "numeric"},
    {"question": "¿Qué tipo de pruebas permiten diagnosticar la infección aguda por VIH antes de la seroconversión completa?",
     "reference": "Las pruebas que detectan el virus o el antígeno p24 directamente: la carga viral (ARN-VIH) y los inmunoanálisis de cuarta generación (antígeno p24 + anticuerpos), que se positivizan antes que los anticuerpos aislados.",
     "tier": "simple", "hops": 1, "guides": ["profilaxis"], "stress": "conceptual"},
    {"question": "¿Cuál es el régimen de elección del tercer fármaco con rifampicina cuando NO se usan inhibidores de la integrasa?",
     "reference": "Efavirenz a dosis estándar (600 mg/día), que mantiene concentraciones adecuadas con rifampicina y es el tercer fármaco de elección no INI en la coinfección TB-VIH.",
     "tier": "simple", "hops": 1, "guides": ["VIH_TB"], "stress": "lexical"},
    {"question": "¿Qué reacción adversa grave se asocia al abacavir en portadores del alelo HLA-B*5701?",
     "reference": "Una reacción de hipersensibilidad potencialmente grave; por eso el abacavir está contraindicado si el HLA-B*5701 es positivo.",
     "tier": "simple", "hops": 1, "guides": ["TAR_2022"], "stress": "lexical"},

    # ---------------------------------------------------------------- TIER: multihop (32)
    {"question": "Gestante en primer trimestre, con coinfección por VHB, que además debe iniciar tratamiento de tuberculosis con rifampicina: ¿qué régimen antirretroviral concilia las tres restricciones?",
     "reference": "Hay que combinar tres exigencias. (1) VHB: el régimen debe incluir dos fármacos activos frente al VHB → tenofovir (en gestación se usa TDF, con experiencia de seguridad) + emtricitabina/lamivudina; no debe retirarse la cobertura anti-VHB. (2) Rifampicina (inductor potente): no pueden usarse bictegravir, elvitegravir/cobicistat, cabotegravir ni IP potenciados; las opciones compatibles son efavirenz a dosis estándar o dolutegravir 50 mg/12 h (hasta 2 semanas tras finalizar la rifampicina). (3) Embarazo: elegir fármacos con experiencia de seguridad gestacional. Una pauta razonable es TDF/FTC + dolutegravir (50 mg/12 h mientras dure la rifampicina) o + efavirenz, vigilando estrechamente la carga viral por el riesgo de infradosificación y de transmisión vertical.",
     "tier": "multihop", "hops": 4, "guides": ["VIH_embarazo", "TAR_2022", "VIH_TB"], "stress": "conflicting-constraints"},
    {"question": "Paciente con meningitis tuberculosa y CD4 <50 cél/µL: ¿cuándo iniciar el TAR y qué fármacos elegir teniendo en cuenta la rifampicina?",
     "reference": "Aunque con CD4 <50 la norma es iniciar el TAR de forma muy precoz (primeras 2 semanas), la meningitis tuberculosa es la EXCEPCIÓN: el inicio del TAR se retrasa por el alto riesgo de síndrome inflamatorio de reconstitución inmune (SIRI) grave a nivel del SNC. En cuanto a los fármacos, hay que evitar los incompatibles con rifampicina (bictegravir, elvitegravir/cobicistat, cabotegravir, IP potenciados) y elegir efavirenz a dosis estándar o dolutegravir 50 mg/12 h. El conflicto se resuelve dando prioridad a evitar el SIRI meníngeo (diferir) sin perder la compatibilidad con la rifampicina.",
     "tier": "multihop", "hops": 3, "guides": ["VIH_TB", "TAR_2022"], "stress": "exception"},
    {"question": "Paciente con insuficiencia renal avanzada y hepatitis B crónica que además inicia rifampicina por tuberculosis: ¿cómo se cubre el VHB sin dañar el riñón y manteniendo compatibilidad con la rifampicina?",
     "reference": "Triple conflicto. El VHB exige tenofovir, pero el TDF es nefrotóxico → se prefiere tenofovir alafenamida (TAF) por su menor toxicidad renal (con precaución según el filtrado) junto con FTC/3TC para cubrir el VHB. Por la insuficiencia renal hay que ajustar los ITIAN de eliminación renal al filtrado glomerular. Por la rifampicina, el tercer fármaco debe ser compatible (efavirenz o dolutegravir 50 mg/12 h), evitando los inductibles. No debe retirarse la cobertura anti-VHB sin alternativa activa por riesgo de reactivación.",
     "tier": "multihop", "hops": 4, "guides": ["TAR_2022", "VIH_TB"], "stress": "conflicting-constraints"},
    {"question": "Paciente polimedicado con alto riesgo cardiovascular, deterioro neurocognitivo y adherencia irregular que quiere simplificar el TAR: ¿qué clase de antirretroviral concilia la baja interacción, la alta barrera genética y la penetración en el SNC?",
     "reference": "Los inhibidores de la integrasa sin potenciador (dolutegravir, bictegravir) son la mejor opción porque (1) tienen pocas interacciones, frente a las pautas potenciadas con ritonavir/cobicistat o los ITINN, lo que importa en un polimedicado; (2) son de alta barrera genética, clave con adherencia irregular; y (3) dolutegravir tiene buena penetración en el SNC, relevante por el deterioro neurocognitivo. Hay que tener en cuenta el perfil metabólico (algunos INI y el TAF se asocian a aumento de peso/dislipemia) por el riesgo cardiovascular, descartar coinfección por VHB y resistencias antes de simplificar, y recordar la interacción de los INI con cationes/antiácidos.",
     "tier": "multihop", "hops": 4, "guides": ["TAR_2022", "neurocognitivo", "adherencia"], "stress": "conflicting-constraints"},
    {"question": "Profesional sanitaria embarazada sufre una exposición percutánea profunda con una fuente VIH positiva en fracaso virológico (carga viral elevada): ¿está indicada la PEP, cómo influye la carga viral de la fuente en el riesgo y cómo afecta el embarazo a la elección?",
     "reference": "La PEP está indicada y el embarazo NO la contraindica. El riesgo es alto porque concurren una exposición percutánea profunda y una fuente con carga viral elevada (el fracaso virológico maximiza la infectividad). Debe iniciarse cuanto antes (idealmente en las primeras horas, siempre <72 h) y mantenerse 4 semanas. En la gestante se eligen antirretrovirales con experiencia de seguridad en embarazo, evitando los desaconsejados, con seguimiento serológico. Se informa del riesgo-beneficio, pero el embarazo no es motivo para no administrarla.",
     "tier": "multihop", "hops": 3, "guides": ["profilaxis", "VIH_embarazo"], "stress": "conflicting-constraints"},
    {"question": "Paciente con infección por VIH-2 que desarrolla tuberculosis y necesita un tercer fármaco compatible con rifampicina: ¿por qué no sirve la opción habitual y qué alternativas quedan?",
     "reference": "La opción habitual con rifampicina sería efavirenz, pero el VIH-2 es intrínsecamente resistente a los ITINN, así que efavirenz NO es válido aquí. Las alternativas son: usar un inhibidor de la integrasa compatible con rifampicina ajustando la dosis (dolutegravir 50 mg/12 h) sobre una base de 2 ITIAN, o bien sustituir la rifampicina por rifabutina para poder emplear un inhibidor de la proteasa potenciado activo. La elección debe basarse en fármacos con actividad demostrada frente al VIH-2 (2 ITIAN + INI o IP/p), nunca en ITINN.",
     "tier": "multihop", "hops": 4, "guides": ["TAR_2022", "VIH_TB"], "stress": "trap-conflict"},
    {"question": "Paciente con fracaso previo a raltegravir (resistencia a inhibidores de integrasa de primera generación) que ahora desarrolla tuberculosis: ¿cómo se diseña el TAR teniendo en cuenta a la vez la resistencia y la rifampicina?",
     "reference": "Manda la resistencia: hay que construir un régimen con al menos dos fármacos plenamente activos según el genotipo histórico y actual, sin añadir un único fármaco activo. La resistencia a INI de primera generación puede comprometer también dolutegravir (que con rifampicina necesitaría 50 mg/12 h y, además, dosis ajustadas frente a resistencia a INI), por lo que a menudo la mejor estrategia es sustituir la rifampicina por rifabutina para poder usar un IP potenciado activo (darunavir/ritonavir) más otros fármacos activos. Se prioriza la actividad frente al virus y se adapta la pauta antituberculosa para hacerla compatible.",
     "tier": "multihop", "hops": 4, "guides": ["TAR_2022", "VIH_TB"], "stress": "conflicting-constraints"},
    {"question": "Gestante con adherencia irregular y deterioro neurocognitivo leve: ¿cómo se relacionan ambos problemas y qué hay que reforzar para prevenir la transmisión vertical?",
     "reference": "Existe un círculo vicioso: el deterioro neurocognitivo dificulta la adherencia (olvidos, errores de dosis) y la adherencia irregular impide mantener la supresión virológica, lo que aumenta el riesgo de transmisión vertical (mínimo solo con carga viral indetectable) y de selección de resistencias. Hay que reforzar intensamente la adherencia (simplificación de pauta, apoyos, supervisión), monitorizar estrechamente la carga viral —sobre todo cerca del parto— y, si la viremia no se controla, aplicar medidas de profilaxis de la transmisión vertical (AZT IV intraparto, valorar cesárea, profilaxis al recién nacido).",
     "tier": "multihop", "hops": 3, "guides": ["neurocognitivo", "adherencia", "VIH_embarazo"], "stress": "mechanistic"},
    {"question": "Paciente recién diagnosticado con CD4 muy bajos y tuberculosis activa que tiene previsto viajar a una zona endémica de fiebre amarilla: ¿puede vacunarse y cómo se priorizan las intervenciones?",
     "reference": "Prioridad clínica: tratar la tuberculosis e iniciar el TAR (con CD4 muy bajos, de forma precoz, salvo meningitis TB), eligiendo fármacos compatibles con rifampicina. La vacuna de la fiebre amarilla es de virus vivos atenuados y está contraindicada con inmunodepresión grave (CD4 <200), por lo que NO debe administrarse ahora; se difiere hasta lograr la recuperación inmune (CD4 ≥200 estable) con el TAR, y entonces se valora antes del viaje. Mientras tanto, las vacunas inactivadas indicadas sí pueden administrarse.",
     "tier": "multihop", "hops": 3, "guides": ["medicina_preventiva", "VIH_TB", "TAR_2022"], "stress": "prioritization"},
    {"question": "Paciente suprimido con coinfección por VHB y antecedente de fracaso a un ITINN que solicita pasar a la pauta inyectable de cabotegravir + rilpivirina: ¿qué dos motivos lo desaconsejan?",
     "reference": "Hay una doble contraindicación. (1) La pauta CAB+RPV no cubre el VHB, y como contiene rilpivirina (un ITINN) y cabotegravir (un INI), sustituiría a un régimen con tenofovir, dejando el VHB sin cobertura → riesgo de reactivación. (2) La pauta inyectable está contraindicada si hay evidencia actual o previa de resistencia/fracaso a ITINN o INI, como es el caso (fracaso previo a un ITINN), aun estando hoy suprimido, por el alto riesgo de fracaso y resistencias. Cualquiera de los dos motivos por separado ya la desaconseja.",
     "tier": "multihop", "hops": 2, "guides": ["TAR_2022", "adherencia"], "stress": "double-contraindication"},
    {"question": "Paciente con deterioro neurocognitivo progresivo y escape viral en LCR que además presenta resistencias en el genotipo plasmático: ¿qué dos criterios deben guiar la elección del régimen de rescate?",
     "reference": "Dos criterios simultáneos: (1) la ACTIVIDAD frente al virus, seleccionando fármacos plenamente activos según el estudio de resistencias (idealmente también un genotipo en LCR si es posible), con al menos dos fármacos activos; y (2) la PENETRACIÓN en el SNC, orientando el régimen hacia fármacos con buena penetración en el sistema nervioso central para controlar la replicación compartimentada responsable del deterioro. Debe confirmarse y monitorizarse la carga viral en LCR.",
     "tier": "multihop", "hops": 3, "guides": ["neurocognitivo", "TAR_2022"], "stress": "conflicting-constraints"},
    {"question": "Gestante en el tercer trimestre con carga viral detectable por mala adherencia: ¿qué cadena de medidas de profilaxis de la transmisión vertical deben adoptarse?",
     "reference": "Con carga viral detectable cerca del parto aumenta el riesgo de transmisión vertical y deben encadenarse medidas: reforzar el TAR materno y la adherencia y repetir la carga viral; administrar zidovudina intravenosa intraparto; valorar la cesárea electiva si la viremia permanece elevada (en torno a >1000 copias/mL); y aplicar profilaxis antirretroviral al recién nacido (combinada si el riesgo es alto). El objetivo es reducir al máximo la viremia antes del parto.",
     "tier": "multihop", "hops": 3, "guides": ["VIH_embarazo", "adherencia"], "stress": "sequencing"},
    {"question": "Paciente con riesgo cardiovascular elevado que quiere simplificar a terapia dual y tiene además hepatitis B crónica: ¿qué impide la simplificación y qué hay que valorar del perfil metabólico?",
     "reference": "La coinfección por VHB IMPIDE la simplificación a terapia dual del tipo DTG/3TC, porque esa pauta no cubre el VHB y al retirar el tenofovir hay riesgo de reactivación; debe mantenerse un régimen con dos fármacos activos frente al VHB (tenofovir + FTC/3TC). En cuanto al perfil metabólico, hay que tener presente que algunos inhibidores de integrasa y el TAF se asocian a aumento de peso y dislipemia, relevante por el riesgo cardiovascular, eligiendo la opción de menor impacto metabólico dentro de lo que permita la cobertura del VHB.",
     "tier": "multihop", "hops": 2, "guides": ["TAR_2022"], "stress": "trap-conflict"},
    {"question": "Paciente con CD4 altos y carga viral plasmática indetectable que consulta por deterioro cognitivo: ¿por qué puede tener HAND pese al buen control y qué hay que descartar respecto al SNC y a la adherencia?",
     "reference": "El buen control periférico no garantiza el control en el SNC: el HAND puede persistir por el establecimiento temprano del reservorio en el SNC, neuroinflamación crónica, penetración variable del TAR, daño acumulado por un nadir de CD4 bajo y comorbilidades. Hay que descartar un escape viral en LCR (replicación compartimentada con plasma indetectable) y evaluar la adherencia, ya que el propio deterioro cognitivo la empeora y la adherencia subóptima favorece la replicación residual; conviene reforzar y simplificar el tratamiento.",
     "tier": "multihop", "hops": 3, "guides": ["neurocognitivo", "adherencia"], "stress": "mechanistic"},
    {"question": "Sanitario con exposición percutánea con aguja hueca visiblemente manchada de sangre de una fuente de serología desconocida: ¿cómo se estratifica el riesgo y qué decisión y seguimiento de PEP corresponden?",
     "reference": "El riesgo se estratifica por el tipo de exposición (percutánea profunda con aguja hueca de gran calibre y dispositivo visiblemente ensangrentado = riesgo alto) y por la probabilidad de que la fuente sea VIH positiva (desconocida → se valora el contexto epidemiológico). Ante una exposición de riesgo se recomienda iniciar la PEP cuanto antes (<72 h), pauta de tres fármacos durante 4 semanas, intentando determinar la serología de la fuente para suspenderla si resulta negativa. El seguimiento incluye serología VIH basal y a las semanas siguientes, cribado de VHB/VHC y otras ITS, y control de adherencia y tolerancia.",
     "tier": "multihop", "hops": 3, "guides": ["profilaxis"], "stress": "stratification"},
    {"question": "Paciente con fracaso virológico, buena adherencia documentada y genotipo con múltiples mutaciones, que además inicia rifampicina por tuberculosis: enuncia el principio del rescate y cómo lo modifica la rifampicina.",
     "reference": "Principio del rescate: construir un régimen con al menos dos (preferiblemente tres) fármacos plenamente activos según el genotipo histórico y actual, sin añadir nunca un único fármaco activo a un régimen que fracasa. La rifampicina añade la restricción de evitar los antirretrovirales cuyas concentraciones reduce (bictegravir, elvitegravir/cobicistat, cabotegravir, IP potenciados); si el rescate requiere un IP potenciado, se sustituye la rifampicina por rifabutina; si se usa dolutegravir se ajusta a 50 mg/12 h. La selección de fármacos activos manda y la pauta antituberculosa se adapta para ser compatible.",
     "tier": "multihop", "hops": 3, "guides": ["TAR_2022", "VIH_TB"], "stress": "conflicting-constraints"},
    {"question": "Mujer con VIH en edad fértil, suprimida con un régimen basado en efavirenz, que planifica un embarazo: ¿qué aspectos del TAR y del seguimiento deben revisarse antes de la concepción?",
     "reference": "Hay que revisar la idoneidad del régimen para la gestación (eligiendo fármacos con experiencia de seguridad en embarazo), confirmar la supresión virológica estable y la adherencia, comprobar la ausencia de interacciones y de coinfecciones que condicionen la pauta (VHB), actualizar el cribado y las vacunas indicadas antes del embarazo, y planificar una monitorización estrecha de la carga viral durante la gestación —especialmente cerca del parto— para minimizar la transmisión vertical. La decisión de mantener o cambiar el régimen se individualiza según el perfil de seguridad y la supresión.",
     "tier": "multihop", "hops": 3, "guides": ["VIH_embarazo", "TAR_2022", "medicina_preventiva"], "stress": "planning"},
    {"question": "Paciente con tuberculosis en tratamiento con rifampicina cuyo TAR óptimo, por resistencias, requeriría un inhibidor de la proteasa potenciado: ¿cómo se compatibilizan ambos tratamientos?",
     "reference": "La rifampicina no puede combinarse con IP potenciados con ritonavir/cobicistat porque reduce drásticamente sus concentraciones. Para compatibilizarlos se sustituye la rifampicina por rifabutina (ajustando su dosis), que tiene mucha menor inducción y permite mantener el IP potenciado necesario por las resistencias. Si no es posible cambiar a rifabutina, habría que rediseñar el TAR hacia fármacos compatibles con rifampicina, lo que puede no ser viable si las resistencias obligan al IP potenciado.",
     "tier": "multihop", "hops": 2, "guides": ["VIH_TB", "TAR_2022"], "stress": "conflicting-constraints"},
    {"question": "Paciente con carga viral indetectable mantenida durante años pero con adherencia que ha pasado a ser intermitente en los últimos meses: ¿qué tres riesgos encadenados aparecen y qué papel tiene la barrera genética del régimen?",
     "reference": "Riesgos encadenados: (1) rebote virológico por concentraciones subterapéuticas; (2) selección de mutaciones de resistencia bajo presión selectiva, sobre todo con fármacos de baja barrera (3TC/FTC con M184V, ITINN); y (3) pérdida progresiva de opciones terapéuticas y de la supresión a largo plazo. La barrera genética del régimen modula el riesgo: con inhibidores de integrasa de segunda generación (dolutegravir, bictegravir) se necesitan varias mutaciones para perder eficacia, por lo que protegen frente a la resistencia mejor que las pautas de baja barrera cuando la adherencia es irregular.",
     "tier": "multihop", "hops": 3, "guides": ["adherencia", "TAR_2022"], "stress": "mechanistic"},
    {"question": "Paciente con VIH e insuficiencia renal avanzada y deterioro neurocognitivo: ¿cómo se concilia la elección de un régimen renal-seguro con la necesidad de buena penetración en el SNC?",
     "reference": "Por la insuficiencia renal hay que evitar el TDF (nefrotóxico) y ajustar los ITIAN de eliminación renal al filtrado; abacavir, dolutegravir y los ITINN no requieren ajuste renal y facilitan el manejo. Por el deterioro neurocognitivo conviene priorizar fármacos con buena penetración en el SNC. Dolutegravir concilia bien ambas exigencias (sin ajuste renal y con buena penetración en el SNC) sobre una base de ITIAN ajustada/segura (p. ej. abacavir si HLA-B*5701 negativo, o TAF con precaución), evitando el cobicistat por su elevación de la creatinina que dificulta la monitorización.",
     "tier": "multihop", "hops": 3, "guides": ["TAR_2022", "neurocognitivo"], "stress": "conflicting-constraints"},
    {"question": "Tras una exposición sexual de riesgo en una persona que podría además estar en periodo de infección aguda por otra exposición previa, ¿qué hay que tener en cuenta antes de iniciar la PEP?",
     "reference": "Antes de iniciar la PEP debe descartarse una infección por VIH ya establecida con una prueba basal (incluyendo pruebas que detecten infección aguda: antígeno p24/carga viral, no solo anticuerpos), porque dar una pauta de PEP de dos/tres fármacos a alguien ya infectado equivaldría a una monoterapia/biterapia encubierta con riesgo de resistencias. Si la prueba basal es negativa o no concluyente y la exposición es de riesgo, se inicia la PEP sin demora (<72 h) y se completa el estudio; el seguimiento serológico posterior confirmará o descartará la infección.",
     "tier": "multihop", "hops": 3, "guides": ["profilaxis", "TAR_2022"], "stress": "trap-conflict"},
    {"question": "Paciente con hepatitis B crónica al que se le va a retirar el tenofovir por toxicidad renal: ¿qué riesgo específico aparece y cómo debe manejarse el cambio?",
     "reference": "Retirar el tenofovir (componente activo frente al VHB) sin una alternativa activa frente al VHB puede provocar una reactivación de la hepatitis B con riesgo de hepatitis grave (flare). El cambio debe garantizar que se mantiene la cobertura anti-VHB: sustituir TDF por tenofovir alafenamida (TAF), que es activo frente al VHB y mucho menos nefrotóxico, en lugar de eliminar el tenofovir; y monitorizar la función hepática y la carga viral del VHB tras el cambio.",
     "tier": "multihop", "hops": 2, "guides": ["TAR_2022"], "stress": "trap-conflict"},
    {"question": "Paciente con VIH que va a recibir quimioterapia (inmunosupresión añadida) y consulta por vacunación: ¿cómo influyen el recuento de CD4 y el momento respecto a la inmunosupresión?",
     "reference": "La respuesta vacunal depende del estado inmune: conviene vacunar con CD4 altos y carga viral suprimida, e idealmente antes de iniciar la inmunosupresión adicional, porque tanto los CD4 bajos como la quimioterapia reducen la inmunogenicidad. Las vacunas de virus vivos atenuados están contraindicadas con inmunodepresión grave (CD4 <200 o por la propia quimioterapia); las inactivadas son seguras aunque con respuesta posiblemente disminuida. Se planifica el calendario para maximizar la respuesta antes de la mayor inmunosupresión.",
     "tier": "multihop", "hops": 3, "guides": ["medicina_preventiva", "TAR_2022"], "stress": "timing"},
    {"question": "Paciente con tuberculosis y VIH que, a las pocas semanas de iniciar el TAR, presenta empeoramiento clínico y reaparición de fiebre y adenopatías: ¿qué entidad hay que considerar y cómo se relaciona con el momento de inicio del TAR?",
     "reference": "Hay que considerar un síndrome inflamatorio de reconstitución inmune (SIRI/IRIS) asociado a la tuberculosis: un empeoramiento paradójico por la recuperación inmune tras iniciar el TAR, más frecuente cuanto más bajos son los CD4 y más precoz el inicio del TAR. Se relaciona directamente con el momento de inicio: por eso, aunque con CD4 <50 se inicia precozmente, en la meningitis tuberculosa se difiere para evitar un SIRI del SNC grave. El manejo es continuar el tratamiento antituberculoso y el TAR y, según gravedad, añadir corticoides.",
     "tier": "multihop", "hops": 3, "guides": ["VIH_TB", "TAR_2022"], "stress": "mechanistic"},
    {"question": "Paciente con riesgo cardiovascular y carga viral indetectable que pregunta por qué su riesgo sigue siendo alto pese a tener los factores clásicos controlados, y cómo influye el propio TAR: ¿qué se le explica?",
     "reference": "Aunque controle los factores clásicos, persiste un riesgo cardiovascular aumentado por la inflamación e inmunoactivación crónicas asociadas al VIH (presentes incluso con carga viral suprimida), la disfunción endotelial y el daño vascular acumulado. Además, algunos antirretrovirales contribuyen: ciertos IP y algunos inhibidores de integrasa y el TAF se asocian a cambios metabólicos (aumento de peso, dislipemia). Se actúa optimizando los factores modificables y, si procede, eligiendo una pauta de menor impacto metabólico, sin renunciar a la eficacia virológica.",
     "tier": "multihop", "hops": 3, "guides": ["TAR_2022", "neurocognitivo"], "stress": "mechanistic"},
    {"question": "Gestante con VIH y tuberculosis diagnosticada en el segundo trimestre: ¿cómo se ordenan el inicio del tratamiento antituberculoso, el ajuste del TAR por la rifampicina y la vigilancia de la transmisión vertical?",
     "reference": "Se inicia el tratamiento antituberculoso sin demora y se ajusta el TAR para que sea compatible con la rifampicina (efavirenz a dosis estándar o dolutegravir 50 mg/12 h; evitar bictegravir, elvitegravir/cobicistat, cabotegravir e IP potenciados), usando fármacos con experiencia de seguridad en gestación. Por el embarazo y el riesgo de infradosificación que añade la rifampicina, se intensifica la monitorización de la carga viral, especialmente cerca del parto, para asegurar la supresión y minimizar la transmisión vertical, aplicando las medidas intraparto y neonatales si la viremia no se controla.",
     "tier": "multihop", "hops": 4, "guides": ["VIH_embarazo", "VIH_TB", "TAR_2022"], "stress": "sequencing"},
    {"question": "Paciente que tras años suprimido presenta rebote virológico confirmado y refiere adherencia correcta: ¿qué causas no adherenciales deben investigarse y en qué orden actuar antes de cambiar el TAR?",
     "reference": "Primero confirmar el rebote con una segunda carga viral y reevaluar críticamente la adherencia (incluida la real, no solo la referida). Si la adherencia es realmente correcta, investigar interacciones farmacológicas que reduzcan las concentraciones (inductores, antiácidos/cationes con los INI) y problemas de absorción o requisitos con alimentos, y considerar resistencia transmitida/archivada o de nueva aparición. Debe realizarse un estudio de resistencias antes de cambiar, y el cambio se guía por su resultado, construyendo un régimen con fármacos plenamente activos.",
     "tier": "multihop", "hops": 3, "guides": ["adherencia", "TAR_2022"], "stress": "sequencing"},
    {"question": "Paciente con deterioro neurocognitivo y polifarmacia por comorbilidades: ¿por qué la elección del TAR afecta tanto al riesgo de interacciones como a la cognición, y qué clase se prefiere?",
     "reference": "El TAR influye en ambos frentes: las pautas potenciadas con ritonavir/cobicistat y los ITINN tienen muchas interacciones (problemáticas en polifarmacia), mientras que los inhibidores de integrasa sin potenciador (dolutegravir, bictegravir, raltegravir) las minimizan; y la penetración del fármaco en el SNC condiciona el control de la replicación cerebral relevante para la cognición. Se prefiere un inhibidor de integrasa sin potenciador con buena penetración en el SNC (dolutegravir), vigilando las interacciones con cationes/antiácidos y revisando la medicación concomitante.",
     "tier": "multihop", "hops": 2, "guides": ["neurocognitivo", "TAR_2022"], "stress": "mechanistic"},
    {"question": "Persona sin VIH con exposiciones sexuales de riesgo repetidas que acude por una nueva exposición: ¿cómo se diferencia la PEP de la indicación de PrEP y qué cabe plantear?",
     "reference": "La PEP es una intervención reactiva y puntual tras una exposición concreta: pauta de tres fármacos durante 4 semanas iniciada en <72 h. Las exposiciones de riesgo REPETIDAS, en cambio, son indicación de valorar la profilaxis preexposición (PrEP) como estrategia preventiva continuada. Ante una nueva exposición se indica la PEP si procede y, en el seguimiento, se plantea la transición a PrEP para las exposiciones futuras, además del cribado de ITS y la educación preventiva.",
     "tier": "multihop", "hops": 2, "guides": ["profilaxis"], "stress": "discrimination"},
    {"question": "Paciente con VIH-2 e insuficiencia renal: ¿cómo se combinan las restricciones del VIH-2 y de la función renal en la elección del régimen?",
     "reference": "Por el VIH-2 hay que evitar los ITINN (inactivos) y basar el tratamiento en 2 ITIAN + un inhibidor de la integrasa (o un IP/p activo). Por la insuficiencia renal hay que evitar el TDF y ajustar los ITIAN de eliminación renal al filtrado, prefiriendo opciones sin ajuste renal o el TAF con precaución; dolutegravir no requiere ajuste renal y es válido frente al VIH-2, por lo que una base de ITIAN renal-segura + dolutegravir concilia ambas restricciones. La monitorización del VIH-2 es más compleja por la falta de cargas virales estandarizadas.",
     "tier": "multihop", "hops": 3, "guides": ["TAR_2022"], "stress": "conflicting-constraints"},
    {"question": "Paciente con tuberculosis y CD4 de 300 cél/µL sin meningitis: ¿en qué plazo se inicia el TAR y por qué difiere del de un paciente con CD4 muy bajos?",
     "reference": "Con CD4 más altos (p. ej. 300) y sin afectación meníngea, el inicio del TAR puede diferirse hasta unas 8 semanas tras comenzar el tratamiento antituberculoso, porque el beneficio de mortalidad del inicio muy precoz se concentra en los CD4 muy bajos (<50, donde se inicia en las 2 primeras semanas) y diferir reduce el riesgo de SIRI y de toxicidades/interacciones solapadas. La meningitis tuberculosa es la excepción en que se difiere aún más por el riesgo de SIRI grave del SNC.",
     "tier": "multihop", "hops": 2, "guides": ["VIH_TB", "TAR_2022"], "stress": "threshold-reasoning"},
    {"question": "Paciente con coinfección por VHB que interrumpe el TAR por su cuenta: ¿qué riesgo hepático específico añade la coinfección frente a un paciente sin VHB?",
     "reference": "Frente a un paciente sin VHB, la interrupción añade el riesgo de reactivación de la hepatitis B con posible hepatitis aguda grave (flare hepático) al retirarse bruscamente los fármacos activos frente al VHB (tenofovir, 3TC/FTC). Por eso en la coinfección por VHB no debe suspenderse el componente anti-VHB sin una alternativa activa, y las interrupciones son especialmente peligrosas; además del habitual riesgo de rebote virológico del VIH y de selección de resistencias.",
     "tier": "multihop", "hops": 2, "guides": ["TAR_2022"], "stress": "comparative"},

    # ---------------------------------------------------------------- TIER: adversarial (20)
    {"question": "¿Puede usarse la pauta dual dolutegravir + lamivudina en cualquier paciente con carga viral indetectable que quiera simplificar?",
     "reference": "No, no en cualquiera. DTG/3TC NO debe usarse si hay coinfección por VHB (no lo cubre), resistencia conocida o sospechada a lamivudina (M184V/I) o a inhibidores de integrasa, ausencia de estudio de resistencias previo, antecedente de fracaso virológico, o adherencia no garantizada. Solo es adecuada en pacientes suprimidos de forma estable que cumplan esos criterios de seguridad.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022"], "stress": "negation"},
    {"question": "¿Está indicada la profilaxis postexposición siempre que se produzca un pinchazo accidental con una aguja?",
     "reference": "No siempre. No se indica PEP cuando no hay riesgo apreciable: fuente VIH negativa o con carga viral indetectable confirmada, contacto con fluidos no infecciosos, exposición sobre piel intacta, aguja sólida sin sangre visible de bajo riesgo, o si han transcurrido más de 72 horas. La indicación depende del tipo de exposición, del inóculo y del estado serológico/carga viral de la fuente.",
     "tier": "adversarial", "hops": 2, "guides": ["profilaxis"], "stress": "negation"},
    {"question": "Para evitar el síndrome de reconstitución inmune, ¿conviene retrasar el inicio del TAR en todos los pacientes con tuberculosis?",
     "reference": "No. En general el TAR se inicia precozmente: con CD4 muy bajos (<50) dentro de las 2 primeras semanas, porque retrasarlo aumenta la mortalidad. Solo se difiere de forma marcada en la meningitis tuberculosa, por el alto riesgo de SIRI grave del SNC. Retrasar el TAR en todos los pacientes con TB sería un error: el equilibrio entre riesgo de SIRI y de mortalidad depende del recuento de CD4 y de la localización de la TB.",
     "tier": "adversarial", "hops": 2, "guides": ["VIH_TB", "TAR_2022"], "stress": "overgeneralization-trap"},
    {"question": "Si un paciente está indetectable, ¿puede prescindir del preservativo con su pareja sin más condiciones?",
     "reference": "Solo bajo la condición de U=U: la intransmisibilidad sexual requiere una supresión virológica mantenida y estable (carga viral indetectable confirmada y sostenida con buena adherencia). Con adherencia irregular o supresión no confirmada no se cumple la condición y existe riesgo de viremia y transmisión. Además, el preservativo sigue protegiendo frente a otras ITS y embarazos no deseados, lo que puede aconsejarlo igualmente.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022"], "stress": "conditional"},
    {"question": "Un paciente con VIH-2 va a iniciar TAR y, como tiene tuberculosis, se plantea efavirenz por su buena compatibilidad con la rifampicina. ¿Es correcto?",
     "reference": "No es correcto. Aunque el efavirenz es compatible con la rifampicina, el VIH-2 es intrínsecamente resistente a los ITINN, por lo que el efavirenz NO es activo frente al VIH-2 y no debe usarse. La buena compatibilidad farmacocinética con la rifampicina no compensa la falta de actividad antiviral. Hay que elegir 2 ITIAN + un inhibidor de integrasa compatible con rifampicina (dolutegravir 50 mg/12 h) o sustituir la rifampicina por rifabutina para usar un IP/p activo.",
     "tier": "adversarial", "hops": 3, "guides": ["TAR_2022", "VIH_TB"], "stress": "distractor-trap"},
    {"question": "En un paciente que simplifica el TAR y tiene hepatitis B crónica, ¿se puede retirar el tenofovir si el resto del régimen es potente?",
     "reference": "No. En la coinfección por VHB no debe retirarse el tenofovir (ni el componente activo frente al VHB) sin una alternativa activa, aunque el resto del régimen sea potente frente al VIH, por el riesgo de reactivación de la hepatitis B y hepatitis grave. La potencia frente al VIH no sustituye la cobertura del VHB: hay que mantener dos fármacos activos frente al VHB.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022"], "stress": "distractor-trap"},
    {"question": "Ante un fracaso virológico con resistencias, ¿es razonable añadir un solo fármaco nuevo plenamente activo al régimen que está fracasando?",
     "reference": "No. Nunca debe añadirse un único fármaco activo a un régimen que fracasa, porque equivale funcionalmente a una monoterapia sobre un virus replicante y selecciona resistencia al nuevo fármaco. El principio es construir un régimen con al menos dos (preferiblemente tres) fármacos plenamente activos según el estudio de resistencias.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022"], "stress": "negation"},
    {"question": "¿Se puede administrar la vacuna triple vírica (sarampión-rubéola-parotiditis) a una persona con VIH?",
     "reference": "Depende del estado inmune: la triple vírica es de virus vivos atenuados, por lo que está contraindicada con inmunodepresión grave (CD4 <200 cél/µL o <15%). Si el paciente tiene CD4 por encima de ese umbral y no está gravemente inmunodeprimido, puede administrarse cuando esté indicada. No es ni un sí ni un no absolutos: el recuento de CD4 decide.",
     "tier": "adversarial", "hops": 2, "guides": ["medicina_preventiva"], "stress": "conditional"},
    {"question": "Como el dolutegravir tiene alta barrera genética, ¿es seguro pasar a la pauta inyectable de cabotegravir + rilpivirina en un paciente con fracaso previo a inhibidores de integrasa?",
     "reference": "No. El antecedente de resistencia o fracaso a inhibidores de integrasa (o a ITINN) contraindica la pauta inyectable cabotegravir + rilpivirina, por el alto riesgo de fracaso y de resistencias, aunque el paciente esté hoy suprimido. La alta barrera del dolutegravir oral no se traslada a esta combinación inyectable de cabotegravir (INI) + rilpivirina (ITINN) en presencia de resistencia previa a esas clases.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022", "adherencia"], "stress": "distractor-trap"},
    {"question": "Si la carga viral plasmática es indetectable, ¿se puede descartar el deterioro neurocognitivo asociado al VIH?",
     "reference": "No. El HAND puede presentarse pese a carga viral plasmática indetectable y CD4 normales, por neuroinflamación crónica, daño acumulado (nadir bajo de CD4), reservorio y posible escape viral en LCR (replicación en el SNC con plasma indetectable). El control periférico no excluye el deterioro neurocognitivo.",
     "tier": "adversarial", "hops": 2, "guides": ["neurocognitivo"], "stress": "negation"},
    {"question": "Como cobicistat eleva la creatinina, ¿debe interpretarse siempre como deterioro renal y motivar la retirada del fármaco?",
     "reference": "No. Cobicistat eleva la creatinina sérica porque inhibe su secreción tubular, sin reducir el filtrado glomerular real; es un aumento esperado, leve y no progresivo que NO indica deterioro renal verdadero y no justifica por sí mismo retirar el fármaco. Hay que distinguirlo de una caída real del filtrado, valorando otros parámetros de función renal.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022"], "stress": "distractor-trap"},
    {"question": "Una viremia persistente de bajo nivel (<200 copias/mL) en un paciente por lo demás estable, ¿obliga a cambiar inmediatamente el TAR?",
     "reference": "No obliga a un cambio inmediato. Viremias persistentes de bajo nivel <200 copias/mL a menudo no se consideran fracaso virológico establecido. Primero hay que confirmar/repetir la determinación, evaluar la adherencia, descartar interacciones y blips, y valorar un estudio de resistencias; la decisión de cambiar se toma según la evolución y el resultado, no de forma automática.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022"], "stress": "conditional"},
    {"question": "Si una gestante con VIH tiene carga viral indetectable mantenida, ¿es obligatoria la cesárea para prevenir la transmisión vertical?",
     "reference": "No. Con carga viral indetectable mantenida cerca del parto el riesgo de transmisión vertical es muy bajo y no está indicada la cesárea por el VIH; puede plantearse el parto vaginal. La cesárea electiva se reserva para cuando la carga viral permanece elevada (en torno a >1000 copias/mL) o no está controlada. La vía del parto se decide según la carga viral, no de forma sistemática.",
     "tier": "adversarial", "hops": 2, "guides": ["VIH_embarazo"], "stress": "negation"},
    {"question": "Como el efavirenz se usa con rifampicina, ¿es la mejor opción de tercer fármaco para cualquier paciente con tuberculosis?",
     "reference": "No para cualquiera. El efavirenz es el tercer fármaco de elección con rifampicina en muchos casos, pero no sirve si hay VIH-2 o resistencia a ITINN (inactivos), ni se prefiere si hay antecedentes psiquiátricos relevantes u otras contraindicaciones; en gestación y otros contextos puede preferirse dolutegravir (50 mg/12 h). La elección se individualiza; la compatibilidad con rifampicina es solo uno de los criterios.",
     "tier": "adversarial", "hops": 2, "guides": ["VIH_TB", "TAR_2022"], "stress": "overgeneralization-trap"},
    {"question": "Dado que el tenofovir alafenamida (TAF) es menos nefrotóxico que el TDF, ¿puede usarse TAF junto con rifampicina sin problema?",
     "reference": "No sin problema: aunque el TAF es mejor a nivel renal, NO se recomienda con rifampicina porque la rifampicina reduce sus concentraciones plasmáticas. Como ITIAN con rifampicina se prefieren TDF, abacavir, 3TC o FTC. La ventaja renal del TAF no resuelve la interacción farmacocinética con la rifampicina.",
     "tier": "adversarial", "hops": 2, "guides": ["VIH_TB", "TAR_2022"], "stress": "distractor-trap"},
    {"question": "Si un paciente refiere tomar toda la medicación, ¿se puede descartar la adherencia como causa de un fracaso virológico?",
     "reference": "No se puede descartar solo con lo que refiere el paciente. La adherencia autodeclarada sobreestima la real, y la adherencia subóptima es la primera causa de fracaso; debe evaluarse con métodos objetivos (registros de dispensación, recuento, niveles si procede) además de la entrevista. Solo tras confirmar una adherencia realmente correcta se investigan otras causas (interacciones, absorción, resistencias).",
     "tier": "adversarial", "hops": 2, "guides": ["adherencia", "TAR_2022"], "stress": "distractor-trap"},
    {"question": "¿Basta con que hayan pasado menos de 72 horas para que la profilaxis postexposición esté indicada?",
     "reference": "No basta. El plazo <72 h es necesario pero no suficiente: además debe haber un riesgo apreciable de transmisión (tipo de exposición, inóculo, estado VIH/carga viral de la fuente). Si la fuente es VIH negativa o indetectable, el contacto es con fluidos no infecciosos o sobre piel intacta, no se indica PEP aunque se esté dentro del plazo.",
     "tier": "adversarial", "hops": 2, "guides": ["profilaxis"], "stress": "conditional"},
    {"question": "En un paciente suprimido con buena adherencia, ¿la monoterapia con un inhibidor de integrasa de alta barrera es una opción de simplificación válida?",
     "reference": "No. Se recomienda evitar la monoterapia antirretroviral incluso con fármacos potentes y de alta barrera, porque no mantiene una supresión duradera y selecciona resistencias. Las estrategias de simplificación validadas son pautas combinadas (p. ej. terapia dual como DTG/3TC o DTG/RPV en candidatos seleccionados), no la monoterapia.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022"], "stress": "negation"},
    {"question": "Como las vacunas inactivadas son seguras en personas con VIH, ¿da igual el recuento de CD4 al vacunar?",
     "reference": "Seguras sí, pero no da igual el momento: con CD4 bajos o carga viral detectable la inmunogenicidad (la respuesta protectora) está disminuida. Por eso conviene vacunar con CD4 altos y carga viral suprimida para lograr una mejor respuesta, aunque la administración en sí sea segura con CD4 bajos. Seguridad e inmunogenicidad son cuestiones distintas.",
     "tier": "adversarial", "hops": 2, "guides": ["medicina_preventiva"], "stress": "conditional"},
    {"question": "Si el TAR redujo drásticamente la demencia asociada al VIH, ¿significa que un buen control virológico elimina el riesgo de trastornos neurocognitivos?",
     "reference": "No. El TAR eliminó en gran medida la forma más grave (la demencia asociada al VIH), pero persisten formas leves y moderadas de HAND por neuroinflamación crónica, penetración variable de los fármacos en el SNC, reservorio en el SNC, daño previo (nadir bajo de CD4) y comorbilidades. Un buen control virológico reduce pero no elimina el riesgo neurocognitivo.",
     "tier": "adversarial", "hops": 2, "guides": ["neurocognitivo"], "stress": "negation"},
    {"question": "En España, ¿puede una madre con VIH y carga viral indetectable dar el pecho con seguridad si lo desea?",
     "reference": "En entornos con acceso garantizado a lactancia artificial segura, como España, se recomienda evitar la lactancia materna y alimentar con fórmula, porque la leche materna puede transmitir el VIH y ese riesgo no es nulo ni siquiera con carga viral indetectable. La indetectabilidad reduce el riesgo pero la recomendación de evitar la lactancia se mantiene; si la madre insiste, se maneja como una situación de riesgo con acompañamiento, no como una opción equivalente.",
     "tier": "adversarial", "hops": 2, "guides": ["VIH_embarazo"], "stress": "distractor-trap"},
    {"question": "Como el tenofovir alafenamida (TAF) es más seguro a nivel renal y óseo que el TDF, ¿debe sustituirse el TDF por TAF en todos los pacientes?",
     "reference": "No en todos. El TAF es preferible cuando hay riesgo o daño renal u óseo, pero la elección se individualiza: el TDF sigue siendo válido en muchos pacientes, se asocia a un perfil lipídico más favorable y menor ganancia de peso que el TAF, y no se recomienda con rifampicina (donde se prefiere TDF). La ventaja renal/ósea del TAF no justifica una sustitución universal; depende del perfil del paciente y de las interacciones.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022"], "stress": "overgeneralization-trap"},
    {"question": "Si un paciente lleva años con carga viral indetectable, ¿puede dejar de monitorizarse la carga viral?",
     "reference": "No. Aunque la frecuencia de monitorización puede espaciarse en pacientes estables y adherentes, la carga viral debe seguir controlándose periódicamente para detectar precozmente un eventual fracaso virológico (por pérdida de adherencia, interacciones o resistencias). Suspender la monitorización por completo no es adecuado.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022", "adherencia"], "stress": "negation"},
    {"question": "¿La profilaxis preexposición (PrEP) protege también frente a otras infecciones de transmisión sexual además del VIH?",
     "reference": "No. La PrEP previene la adquisición del VIH, pero no protege frente a otras ITS (sífilis, gonorrea, clamidia, VHB/VHC, etc.). Por ello se mantiene la recomendación de preservativo y de cribado periódico de ITS en las personas en PrEP; confiar solo en la PrEP frente a todas las ITS es un error.",
     "tier": "adversarial", "hops": 2, "guides": ["profilaxis"], "stress": "distractor-trap"},
    {"question": "Un resultado HLA-B*5701 negativo, ¿garantiza que el paciente no tendrá ninguna reacción adversa al abacavir?",
     "reference": "No. Un HLA-B*5701 negativo reduce drásticamente el riesgo de la reacción de hipersensibilidad mediada por ese alelo y permite prescribir abacavir, pero no garantiza la ausencia de cualquier otro efecto adverso (gastrointestinales, exantema no relacionado, posible debate sobre riesgo cardiovascular). Negativo no equivale a riesgo cero de toda reacción.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022"], "stress": "negation"},
    {"question": "Si un paciente tolera bien su pauta potenciada con ritonavir, ¿da igual mantenerla cuando se le añaden varios fármacos nuevos por otras enfermedades?",
     "reference": "No da igual. El ritonavir (y el cobicistat) es un inhibidor enzimático potente con numerosas interacciones farmacológicas; al añadir nuevos fármacos aumenta el riesgo de interacciones clínicamente relevantes aunque el paciente tolere bien el antirretroviral en sí. En polifarmacia conviene revisar las interacciones y a menudo se prefieren inhibidores de integrasa sin potenciador. La buena tolerancia no equivale a ausencia de interacciones.",
     "tier": "adversarial", "hops": 2, "guides": ["TAR_2022"], "stress": "distractor-trap"},
    {"question": "Para una exposición de bajo riesgo, ¿puede administrarse la PEP con un solo antirretroviral para simplificar?",
     "reference": "No. La pauta recomendada de PEP es de tres fármacos antirretrovirales; no debe usarse mono ni biterapia, ya que una pauta insuficiente puede no prevenir la infección y, si hubiera infección establecida, seleccionar resistencias. Lo que se decide según el riesgo es SI se indica la PEP, no reducir el número de fármacos de la pauta.",
     "tier": "adversarial", "hops": 2, "guides": ["profilaxis"], "stress": "distractor-trap"},
    {"question": "¿Cualquier inhibidor de la integrasa puede combinarse con rifampicina si se ajusta la dosis?",
     "reference": "No. Solo dolutegravir (a 50 mg/12 h) y raltegravir (a 800 mg/12 h) pueden usarse con rifampicina ajustando la dosis. Bictegravir, elvitegravir/cobicistat y cabotegravir NO pueden combinarse con rifampicina porque sus concentraciones caen de forma inaceptable y no se corrigen con ajuste de dosis. El ajuste no es una solución universal para toda la clase.",
     "tier": "adversarial", "hops": 2, "guides": ["VIH_TB", "TAR_2022"], "stress": "overgeneralization-trap"},
    {"question": "Dado que las vacunas de virus vivos están contraindicadas con CD4 <200, ¿una vez superado ese umbral son todas igualmente seguras?",
     "reference": "No automáticamente. Superar los 200 cél/µL permite considerar varias vacunas vivas (p. ej. triple vírica, varicela) cuando están indicadas, pero la decisión sigue siendo individualizada y algunas, como la fiebre amarilla, requieren una valoración específica del riesgo-beneficio y del grado de recuperación inmune. El umbral de 200 es necesario pero no convierte a todas las vacunas vivas en igualmente seguras de forma indiscriminada.",
     "tier": "adversarial", "hops": 2, "guides": ["medicina_preventiva"], "stress": "conditional"},
    {"question": "Si un paciente con tuberculosis mejora clínicamente a las pocas semanas, ¿puede acortarse la duración del tratamiento antituberculoso?",
     "reference": "No. La mejoría clínica precoz no permite acortar la duración del tratamiento antituberculoso, que debe completarse durante el tiempo establecido para erradicar el bacilo y evitar recaídas y resistencias. Acortarlo por la mejoría sintomática es un error; la duración viene determinada por la pauta y la respuesta microbiológica, no por la sensación de mejoría.",
     "tier": "adversarial", "hops": 2, "guides": ["VIH_TB"], "stress": "distractor-trap"},
]


def _tag(items, tier):
    """Stamp a default tier/hops onto questions from the previously-separate pools so the
    whole evaluation is ONE tiered set. Existing fields win (e.g. multihop keeps its hops)."""
    return [{**it, "tier": it.get("tier", tier), "hops": it.get("hops", 1)} for it in items]


# ===========================================================================
# EVAL_SET — the single evaluation set (151 questions). The pools are folded in and every
# question gets a "tier" so performance can be sliced by question type:
#   simple — atomic factual/lexical | single_hop — single-area reasoning |
#   multihop — cross-guide reasoning | adversarial — negation / "it depends" / distractor.
# ===========================================================================
EVAL_SET = (
    _tag(_PREV_SINGLE[:39], "single_hop")   # former golden: clinical-reasoning block
    + _tag(_PREV_SINGLE[39:], "simple")     # former golden: specific-terms / lexical block
    + _tag(_PREV_MULTI, "multihop")         # former multi-hop set (keeps its own hops/guides)
    + _TIERED_NEW                           # purpose-built simple/multihop/adversarial tiers
)
# EVAL_SAMPLE=N runs a STRATIFIED subset of N per tier (cheap probe); 0/unset = the full set.
_SAMPLE = int(os.environ.get("EVAL_SAMPLE", "0"))
if _SAMPLE > 0:
    from collections import defaultdict
    _by_tier: dict[str, list] = defaultdict(list)
    for _c in EVAL_SET:
        _by_tier[_c.get("tier")].append(_c)
    DATASET = [c for tier in _by_tier for c in _by_tier[tier][:_SAMPLE]]
else:
    DATASET = EVAL_SET


def build_dataset(dataset: list[dict], retriever) -> tuple[EvaluationDataset, list[float]]:
    """Run each question through `retriever` + the shared generator and assemble the dataset
    RAGAS consumes. Also returns per-question latency (retrieval + generation) for the A/B."""
    rows = []
    omitted = []
    latencies = []
    for i, case in enumerate(dataset, 1):
        question = case["question"].strip()
        reference = case.get("reference", "").strip()

        if not reference:
            omitted.append(i)
            continue

        # Swappable retrieval + shared generation.
        t0 = time.perf_counter()
        payloads = retriever(question)                      # list[dict]
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
    print(f"Pipeline: {PIPELINE} | mode: full RAGAS | dataset: {len(DATASET)} preguntas "
          f"| coleccion Qdrant: {COLLECTION_HYBRID}\n")
    dataset, latencies = build_dataset(DATASET, retriever)

    print(f"Evaluating with judge: {JUDGE_MODEL}\n")
    evaluator_llm = LangchainLLMWrapper(chat_model(JUDGE_MODEL))
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings_model())

    # answer_relevancy is omitted on purpose — misleading on Spanish answers (Phase 0 artifact).
    metrics = [Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()]

    # RAGAS's default 16 workers overwhelm the judge (-> NaN on the heavy precision metric);
    # 8 workers + long timeout + retries is the stable point. Generation is sequential anyway.
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
    # Carry the tier into the dump: without it the CSV cannot be re-sliced by question type
    # afterwards, which is the whole point of this set (and comparing runs is what the A/B is).
    tier_by_q = {c["question"].strip(): c.get("tier") for c in DATASET if c.get("tier")}
    if tier_by_q and "user_input" in df.columns:
        df["tier"] = df["user_input"].map(tier_by_q)

    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", f"ragas_results_{PIPELINE}.csv")  # per-pipeline file
    df.to_csv(out_csv, index=False)
    print(f"\nDetail saved to {out_csv}")

    pd.set_option("display.max_colwidth", 60)
    print("\n=== Means per metric ===")
    print(df.select_dtypes("number").mean())

    # Per-tier slice: does the pipeline degrade on some question types? — the whole point.
    if "tier" in df.columns:
        num_cols = df.select_dtypes("number").columns
        print("\n=== Means per tier (n per tier) ===")
        print(df.groupby("tier")[num_cols].mean())
        print(df["tier"].value_counts().rename("n"))

    # Velocity axis of the A/B: latency per query (retrieval + generation).
    if latencies:
        print("\n=== Latency (s/query) ===")
        print(f"mean={statistics.mean(latencies):.1f}  median={statistics.median(latencies):.1f}  "
              f"max={max(latencies):.1f}  total={sum(latencies):.0f}")


if __name__ == "__main__":
    main()
