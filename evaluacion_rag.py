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
from rag import retrieve, retrieve_hibrido, build_context, generate_answer

# Retriever a evaluar. Cambia a `retrieve` para reproducir el baseline denso (F0).
RETRIEVER = retrieve_hibrido

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
    {"pregunta": "En un paciente con VIH con carga viral indetectable en DTG/3TC, ¿qué factores obligarían a no mantener una terapia dual según las recomendaciones actuales?", "referencia": "La terapia dual DTG/3TC no debe mantenerse si existe coinfección por el virus de la hepatitis B (las pautas duales no cubren el VHB), si hay resistencia conocida o sospechada a lamivudina (mutación M184V/I) o a los inhibidores de integrasa, si no se dispone de estudio de resistencias previo, si hay antecedente de fracaso virológico o si la adherencia no está garantizada. Igualmente se abandonaría ante viremia detectable/fracaso virológico o aparición de resistencias."},
    {"pregunta": "¿Qué pruebas deben realizarse antes de iniciar abacavir y qué ocurriría si el resultado no está disponible en un paciente con infección aguda?", "referencia": "Antes de iniciar abacavir es obligatorio determinar el alelo HLA-B*5701 (A-I), ya que los portadores tienen un riesgo de hasta el 50% de reacción de hipersensibilidad; si el HLA-B*5701 es positivo no debe prescribirse abacavir. Si en una infección aguda se opta por inicio rápido y aún no se dispone del resultado de HLA-B*5701 (ni del estudio de resistencias), no deben usarse regímenes con abacavir ni con ITINN; se recomienda iniciar con TDF o TAF/FTC."},
    {"pregunta": "Si un paciente con VIH tiene fracaso virológico con viremias bajas persistentes, ¿cuál es el algoritmo recomendado antes de cambiar TAR?", "referencia": "Antes de cambiar el TAR debe confirmarse el fracaso con una segunda determinación de carga viral, evaluar y reforzar la adherencia, descartar interacciones farmacológicas y problemas de absorción, revisar la potencia y barrera genética del régimen y realizar un estudio de resistencias (genotipo) si la viremia lo permite. Solo entonces se decide el cambio, guiado por el resultado de resistencias."},
    {"pregunta": "¿Por qué la adherencia subóptima puede generar resistencia incluso cuando la carga viral se mantiene relativamente baja?", "referencia": "Una adherencia intermedia mantiene concentraciones subterapéuticas del fármaco que permiten cierta replicación viral bajo presión selectiva; ese entorno favorece la selección de mutaciones de resistencia aunque no haya un rebote franco de la carga viral, especialmente con fármacos de baja barrera genética (lamivudina/emtricitabina con M184V, o los ITINN)."},
    {"pregunta": "¿En qué situaciones clínicas se recomienda no reducir a regímenes de menos de tres fármacos aunque el paciente esté suprimido?", "referencia": "No se recomienda reducir a menos de tres fármacos en presencia de coinfección por VHB (que requiere dos fármacos activos frente al VHB), cuando existe resistencia previa o archivada que comprometa los componentes de la pauta reducida, ante antecedentes de fracasos virológicos múltiples, cuando no puede garantizarse una buena adherencia o cuando no se cumplen los criterios de los ensayos (supresión estable y mantenida, sin resistencia a los fármacos del régimen reducido)."},
    {"pregunta": "¿Qué problemas farmacológicos surgen al tratar tuberculosis con rifampicina en un paciente que toma inhibidores de integrasa?", "referencia": "La rifampicina es un inductor enzimático potente (CYP3A4 y glucuronidación/UGT1A1) que reduce las concentraciones plasmáticas de los inhibidores de integrasa. Con rifampicina no pueden utilizarse EVG/c, bictegravir ni cabotegravir; el dolutegravir requiere doblar la dosis a 50 mg/12 h (en ausencia de resistencia a INI) hasta 2 semanas después de finalizar la rifampicina, y el raltegravir debe ajustarse. El tercer fármaco de elección con rifampicina es efavirenz."},
    {"pregunta": "¿Cuál es el momento óptimo para iniciar TAR en un paciente con VIH que presenta tuberculosis activa y CD4 muy bajos?", "referencia": "Se recomienda iniciar el TAR de forma precoz una vez iniciado el tratamiento antituberculoso. En pacientes con CD4 muy bajos (<50 cél/µL) debe comenzarse dentro de las primeras 2 semanas; con CD4 más altos puede diferirse hasta unas 8 semanas. La excepción es la meningitis tuberculosa, en la que el inicio se retrasa por el alto riesgo de síndrome inflamatorio de reconstitución inmune grave."},
    {"pregunta": "¿Por qué algunos antirretrovirales deben evitarse en pacientes con insuficiencia renal significativa?", "referencia": "Algunos antirretrovirales son nefrotóxicos o se eliminan por vía renal y se acumulan al disminuir el filtrado glomerular. El tenofovir disoproxilo (TDF) puede causar tubulopatía y deterioro renal y debe evitarse; los ITIAN de eliminación renal (3TC, FTC y el propio TDF) requieren ajuste de dosis según el aclaramiento. Además, potenciadores como cobicistat elevan la creatinina sérica (inhiben su secreción tubular) sin reducir el filtrado real, lo que dificulta la monitorización. Fármacos como DTG, ABC e ITINN no necesitan ajuste renal."},
    {"pregunta": "¿Qué implicaciones tiene la infección por VIH-2 en la elección del tratamiento inicial?", "referencia": "El VIH-2 es intrínsecamente resistente a los ITINN y a la enfuvirtida, y responde peor a algunos inhibidores de la proteasa, por lo que el tratamiento inicial debe basarse en 2 ITIAN combinados con un inhibidor de integrasa (o un IP/p activo), evitando los ITINN. La monitorización es más compleja por la falta de cargas virales estandarizadas."},
    {"pregunta": "En un paciente con infección aguda por VIH, ¿por qué iniciar TAR inmediatamente puede tener beneficios inmunológicos y epidemiológicos?", "referencia": "El inicio inmediato en la infección aguda preserva la función inmune, limita el tamaño del reservorio viral, reduce la activación e inflamación inmune y mejora la recuperación de los CD4 (beneficio inmunológico). Epidemiológicamente, la fase aguda cursa con carga viral muy elevada y máxima infectividad, por lo que suprimir la replicación precozmente reduce de forma importante el riesgo de transmisión."},
    {"pregunta": "¿En qué situaciones no se recomienda profilaxis postexposición (PEP) aunque haya contacto con fluidos potencialmente infecciosos?", "referencia": "No se recomienda PEP cuando la exposición no supone riesgo apreciable: fuente VIH negativa o con carga viral indetectable confirmada, contacto con fluidos no infecciosos (saliva, orina, sudor o lágrimas sin sangre visible), exposición sobre piel intacta, o cuando han transcurrido más de 72 horas desde la exposición. También puede no indicarse en exposiciones de muy bajo riesgo tras valorar el caso."},
    {"pregunta": "¿Qué factores determinan el riesgo de transmisión de VIH tras una exposición ocupacional?", "referencia": "El riesgo depende del tipo de exposición (percutánea profunda > superficial > mucosa o piel no intacta), del volumen de inóculo (aguja hueca de gran calibre, dispositivo visiblemente manchado de sangre, inserción en vaso sanguíneo), de la profundidad de la lesión y de la carga viral de la fuente (máximo en infección aguda o fracaso virológico, mínimo si está indetectable)."},
    {"pregunta": "¿Por qué la profilaxis postexposición debe iniciarse idealmente antes de 72 horas y qué ocurre si se inicia después?", "referencia": "La PEP actúa impidiendo que el virus establezca la infección antes de su diseminación sistémica, por lo que debe iniciarse cuanto antes, idealmente en las primeras horas y siempre dentro de las 72 horas. Pasado ese plazo la eficacia es muy baja y, en general, no se recomienda iniciarla."},
    {"pregunta": "¿Qué vacunas están contraindicadas o deben evaluarse con precaución en personas con VIH con CD4 bajos?", "referencia": "Las vacunas de virus vivos atenuados (triple vírica/sarampión-rubéola-parotiditis, varicela y fiebre amarilla) están contraindicadas en caso de inmunodepresión grave (CD4 <200 cél/µL o <15%) y deben posponerse hasta la recuperación inmunológica. Las vacunas inactivadas son seguras, aunque su respuesta puede estar disminuida con CD4 bajos."},
    {"pregunta": "¿Qué factores inmunológicos pueden disminuir la inmunogenicidad de las vacunas en pacientes con VIH?", "referencia": "Reducen la respuesta vacunal un recuento de CD4 bajo (nadir y actual), la carga viral detectable o la replicación activa, el grado de activación e inflamación inmune crónica, la edad avanzada y las comorbilidades. La inmunogenicidad mejora con un TAR efectivo y la recuperación inmune, por lo que conviene vacunar con CD4 altos y carga viral suprimida."},
    {"pregunta": "Si un paciente tiene carga viral indetectable pero adherencia irregular, ¿puede seguir transmitiendo VIH?", "referencia": "El principio U=U (indetectable = intransmisible) solo es válido con una supresión virológica mantenida y estable. Con adherencia irregular no puede garantizarse la indetectabilidad continua, por lo que existe riesgo de episodios de viremia (blips o rebote) y, por tanto, de transmisión; no se cumple la condición de intransmisibilidad."},
    {"pregunta": "¿Puede un paciente con CD4 normales tener igualmente deterioro neurocognitivo asociado al VIH?", "referencia": "Sí. Los trastornos neurocognitivos asociados al VIH (HAND) pueden presentarse pese a tener CD4 normales y carga viral suprimida, debido a neuroinflamación persistente, daño acumulado por un nadir de CD4 bajo, replicación en el sistema nervioso central y comorbilidades. El recuento actual de CD4 no excluye el diagnóstico."},
    {"pregunta": "Si la adherencia es alta pero encontramos un fracaso virológico, ¿qué causas no relacionadas con adherencia deben investigarse?", "referencia": "Deben investigarse interacciones farmacológicas que reduzcan las concentraciones del TAR (inductores enzimáticos, antiácidos o cationes con los INI), problemas de absorción, una posología o requisitos con alimentos incorrectos, la presencia de resistencia transmitida o preexistente/archivada y una potencia o barrera genética insuficiente del régimen."},
    {"pregunta": "¿Por qué la introducción del TAR redujo la incidencia de demencia asociada al VIH, pero no eliminó los trastornos neurocognitivos?", "referencia": "El TAR controla la replicación sistémica y redujo drásticamente la forma más grave (la demencia asociada al VIH), pero persisten las formas leves y moderadas por neuroinflamación crónica, penetración variable de los fármacos en el sistema nervioso central, daño neuronal previo (nadir bajo de CD4) y factores comórbidos; el VIH establece efectos en el SNC que el TAR no revierte por completo."},
    {"pregunta": "Si un paciente con VIH tiene carga viral indetectable en plasma, pero presenta replicación viral detectable en LCR, ¿puede desarrollar deterioro neurocognitivo relacionado con el VIH?", "referencia": "Sí. El llamado escape viral en LCR refleja una replicación compartimentada en el sistema nervioso central, por penetración insuficiente de los fármacos o resistencia local, que puede producir daño neuronal y deterioro neurocognitivo a pesar de mantener la carga viral plasmática indetectable."},
    {"pregunta": "Si una persona con VIH mantiene carga viral indetectable durante años, ¿puede seguir teniendo inflamación crónica sistémica que aumente su riesgo cardiovascular?", "referencia": "Sí. Pese a la supresión virológica persiste una activación inmune e inflamación crónica residual (por translocación microbiana, coinfecciones como CMV y el reservorio viral persistente) que, junto con efectos metabólicos de algunos antirretrovirales y los factores de riesgo clásicos, mantiene un riesgo cardiovascular aumentado."},
    {"pregunta": "¿Puede un paciente con CD4 elevados y carga viral indetectable desarrollar trastornos neurocognitivos asociados al VIH (HAND), y qué mecanismos lo explicarían?", "referencia": "Sí. Los mecanismos incluyen el establecimiento temprano del reservorio viral en el sistema nervioso central, la neuroinflamación y activación microglial persistentes, la penetración variable del TAR en el SNC, el daño acumulado por un nadir de CD4 bajo y las comorbilidades. El buen control periférico (CD4 altos y carga viral indetectable) no garantiza el control en el compartimento del SNC."},
    {"pregunta": "Si un paciente presenta fracaso virológico con buena adherencia documentada, ¿qué papel puede tener la resistencia viral preexistente o transmitida?", "referencia": "La resistencia transmitida (adquirida en la primoinfección) o la archivada de exposiciones previas a antirretrovirales puede comprometer la actividad de fármacos del régimen aunque la adherencia sea correcta, especialmente con fármacos de baja barrera como los ITINN. Por ello se recomienda realizar estudio de resistencias basal y de nuevo en el momento del fracaso."},
    {"pregunta": "Si un paciente tiene síndrome metabólico mientras toma TAR, ¿cómo diferenciar si la causa es la infección por VIH, los efectos del tratamiento o los factores clásicos de riesgo cardiovascular?", "referencia": "Hay que valorar las tres contribuciones: la propia infección por VIH (inflamación y activación inmune crónicas), los efectos del TAR (algunos IP, ciertos inhibidores de integrasa y TAF se asocian a aumento de peso y dislipemia) y los factores clásicos (dieta, sedentarismo, genética, edad). La diferenciación se apoya en la relación temporal con el inicio o cambio del TAR, el perfil metabólico conocido de cada fármaco y la evaluación del riesgo individual."},
    {"pregunta": "Si un paciente tiene carga viral indetectable pero con adherencia intermitente, ¿qué riesgo existe de rebote viral, desarrollo de resistencia y pérdida futura de supresión virológica?", "referencia": "La adherencia intermitente conlleva riesgo de rebote virológico, de selección de mutaciones de resistencia (sobre todo con fármacos de baja barrera genética) y de pérdida progresiva de opciones terapéuticas y de la supresión a largo plazo. La adherencia subóptima es la principal causa de fracaso virológico."},
    {"pregunta": "Si un paciente con VIH tiene factores clásicos de riesgo cardiovascular controlados, ¿por qué sigue teniendo mayor riesgo de enfermedad cardiovascular que la población general?", "referencia": "Por la inflamación e inmunoactivación crónicas asociadas al VIH (presentes incluso con carga viral suprimida), la disfunción endotelial, los efectos metabólicos de algunos antirretrovirales, la mayor prevalencia de tabaquismo y el daño vascular acumulado. Estos factores hacen que el riesgo real supere al estimado solo con los factores clásicos."},
    {"pregunta": "¿Puede un paciente con VIH con carga viral indetectable tener reservorios virales activos en tejidos, y qué implicaciones tiene esto para la curación del VIH?", "referencia": "Sí. El VIH persiste integrado en células latentes (linfocitos T CD4 de memoria) y en santuarios tisulares (tejido linfoide, sistema nervioso central, tracto digestivo). El TAR no elimina ese reservorio, lo que explica el rebote viral al suspender el tratamiento y constituye la principal barrera para la curación del VIH."},
    {"pregunta": "Si el TAR ha reducido drásticamente las complicaciones neurológicas graves del VIH, ¿por qué siguen observándose formas leves o moderadas de trastornos neurocognitivos en un porcentaje significativo de pacientes?", "referencia": "Porque persisten mecanismos que el TAR no controla por completo: neuroinflamación crónica, penetración variable de los fármacos en el sistema nervioso central, reservorio viral en el SNC, daño neuronal previo (nadir de CD4 bajo) y comorbilidades (edad, factores vasculares, consumo de tóxicos). Estos factores mantienen formas leves y moderadas pese a la desaparición de las complicaciones graves."},
    {"pregunta": "Si un paciente presenta fracaso virológico con viremia baja persistente (<200 copias/mL), ¿es obligatorio cambiar el tratamiento o primero deben evaluarse otros factores?", "referencia": "No es obligatorio cambiar de inmediato. Primero hay que confirmar y repetir la determinación, evaluar la adherencia, descartar interacciones farmacológicas y blips, y valorar un estudio de resistencias; viremias persistentes <200 copias/mL a menudo no se consideran fracaso virológico establecido. La decisión de cambiar se toma según la evolución y el resultado del estudio de resistencias."},
    {"pregunta": "En un paciente con historia de fracaso previo con ITINN, ¿es seguro cambiar a CAB+RPV inyectable si actualmente está suprimido?", "referencia": "No. La pauta inyectable de cabotegravir + rilpivirina está contraindicada si existe evidencia actual o previa de resistencia o de fracaso virológico previo a los ITINN o a los inhibidores de integrasa, aunque el paciente esté virológicamente suprimido en el momento actual, por el elevado riesgo de fracaso y de desarrollo de resistencias."},
    {"pregunta": "Si un paciente con VIH tiene insuficiencia renal avanzada, ¿qué antirretrovirales deberían evitarse o ajustarse de dosis?", "referencia": "Debe evitarse el tenofovir disoproxilo (TDF) por su nefrotoxicidad, y ajustar la dosis de los ITIAN de eliminación renal (3TC, FTC y TDF) según el filtrado glomerular; el TAF se utiliza con precaución según la función renal. Dolutegravir, abacavir y los ITINN no requieren ajuste renal y facilitan el manejo. Hay que recordar que cobicistat eleva la creatinina sin reducir el filtrado real."},
    {"pregunta": "En un paciente con carga viral indetectable y múltiples comorbilidades, ¿qué factores deben considerarse antes de simplificar el tratamiento?", "referencia": "Antes de simplificar hay que valorar la historia de resistencias y de fracasos previos, la coinfección por VHB, las interacciones farmacológicas derivadas de la polifarmacia, la función renal y hepática, la adherencia esperable y que se cumplan los criterios de supresión estable. Se elige la pauta de menor toxicidad y barrera genética adecuada al perfil del paciente."},
    {"pregunta": "Si un paciente tiene adherencia irregular, ¿por qué algunos regímenes con inhibidores de integrasa tienen mayor barrera genética que otros?", "referencia": "Los inhibidores de integrasa de segunda generación (dolutegravir, bictegravir) tienen mayor barrera genética que los de primera generación (raltegravir, elvitegravir), pues requieren la acumulación de varias mutaciones para perder eficacia. Por eso se prefieren cuando la adherencia es irregular o se busca robustez frente al desarrollo de resistencias."},
    {"pregunta": "En un paciente con fracaso virológico y múltiples mutaciones de resistencia, ¿cuál es el principio fundamental para diseñar el nuevo régimen?", "referencia": "El principio fundamental es construir un régimen con al menos dos (preferiblemente tres) fármacos plenamente activos, seleccionados según el estudio de resistencias histórico y actual, recurriendo si es necesario a nuevas clases o mecanismos de acción. Nunca debe añadirse un único fármaco activo a un régimen que está fracasando."},
    {"pregunta": "Si un paciente con VIH tiene hepatitis B crónica, ¿qué implicaciones tiene esto para elegir el TAR?", "referencia": "El régimen debe incluir dos fármacos activos frente al VHB, habitualmente tenofovir (TAF o TDF) junto con 3TC o FTC. No deben usarse pautas sin cobertura frente al VHB, y no debe suspenderse el tenofovir/componente anti-VHB sin una alternativa activa por el riesgo de reactivación y hepatitis grave; esto contraindica pautas duales como DTG/3TC."},
    {"pregunta": "¿Por qué en pacientes con VIH se recomienda evitar la monoterapia antirretroviral, incluso con fármacos potentes?", "referencia": "Porque la monoterapia, incluso con fármacos potentes, no mantiene una supresión duradera y selecciona resistencias, ya que ejerce una presión farmacológica insuficiente sobre la elevada tasa de replicación y mutación del VIH. El principio terapéutico es combinar siempre varios fármacos activos (TAR combinado)."},
    {"pregunta": "Si un paciente suprimido cambia de TAR por toxicidad, ¿qué parámetros deben monitorizarse tras el cambio para confirmar eficacia?", "referencia": "Tras el cambio debe confirmarse el mantenimiento de la supresión virológica mediante la carga viral (por ejemplo, en torno a las 4 semanas y después de forma periódica), vigilar la resolución de la toxicidad que motivó el cambio, evaluar la tolerancia y la adherencia al nuevo régimen y vigilar posibles nuevas interacciones o efectos adversos."},
    {"pregunta": "Si un paciente tiene interacciones farmacológicas complejas por polifarmacia, ¿qué clases de antirretrovirales suelen ser más fáciles de manejar?", "referencia": "Los inhibidores de integrasa sin potenciador (dolutegravir, bictegravir, raltegravir) presentan menos interacciones que las pautas potenciadas con ritonavir o cobicistat (inhibidores enzimáticos potentes) o que los ITINN (inductores/inhibidores enzimáticos), por lo que son la opción más fácil de manejar en pacientes polimedicados. Debe recordarse la interacción de los INI con cationes y antiácidos."},
    {"pregunta": "Si un paciente presenta rebote viral tras años de supresión, ¿cuáles son las tres causas principales que deben investigarse antes de cambiar TAR?", "referencia": "Las tres causas principales son: (1) adherencia subóptima o interrupciones del tratamiento, que es la más frecuente; (2) interacciones farmacológicas o problemas de absorción que reducen las concentraciones del TAR; y (3) resistencia viral, preexistente/archivada o de nueva aparición. Debe confirmarse el rebote y realizarse un estudio de resistencias antes de cambiar el tratamiento."},

    # --- Preguntas con TÉRMINOS ESPECÍFICOS (fármacos, dosis, siglas) para
    #     estresar la búsqueda léxica/híbrida (BM25). Añadidas en Fase 2.
    {"pregunta": "¿Qué inhibidores de la integrasa no pueden administrarse junto con rifampicina?", "referencia": "Con rifampicina no pueden administrarse elvitegravir/cobicistat, bictegravir (BIC) ni cabotegravir (CAB), ni raltegravir en su pauta de 1200 mg/24h; tampoco los ITINN distintos de efavirenz (RPV, ETR, DOR) ni los inhibidores de la proteasa. Entre los inhibidores de integrasa, las excepciones utilizables son dolutegravir (con ajuste de dosis) y raltegravir 800 mg/12h."},
    {"pregunta": "¿A qué dosis debe administrarse dolutegravir cuando se combina con rifampicina?", "referencia": "Con rifampicina, dolutegravir debe administrarse a 50 mg cada 12 horas (en pacientes sin resistencia a inhibidores de integrasa), manteniendo esa dosis hasta 2 semanas después de finalizar la rifampicina."},
    {"pregunta": "¿Está recomendado el uso de tenofovir alafenamida (TAF) junto con rifampicina?", "referencia": "No. No está recomendado el uso de tenofovir alafenamida (TAF) con rifampicina porque la rifampicina reduce sus concentraciones plasmáticas; como ITIAN con rifampicina se prefieren TDF, ABC, 3TC o FTC."},
    {"pregunta": "¿Qué resultado de la prueba HLA-B*5701 contraindica el abacavir?", "referencia": "Un resultado positivo de HLA-B*5701 contraindica el abacavir: no debe prescribirse ABC si la prueba es positiva, por el alto riesgo de reacción de hipersensibilidad."},
    {"pregunta": "¿Cuál es el tercer fármaco de elección junto a rifampicina en la coinfección tuberculosis-VIH?", "referencia": "El tercer fármaco de elección junto a rifampicina es efavirenz (EFV) a dosis estándar (A-I); como alternativas se recomiendan raltegravir 800 mg/12h o dolutegravir 50 mg/12h."},
    {"pregunta": "¿Cuál es la pauta de TAR de inicio preferente en un paciente con VIH y tuberculosis en tratamiento con rifampicina?", "referencia": "La pauta preferente es tenofovir DF/emtricitabina (o abacavir/lamivudina) a dosis habituales más efavirenz a dosis de 600 mg/día (A-I)."},
    {"pregunta": "¿En qué pacientes es adecuado el cambio a la terapia dual dolutegravir más lamivudina (DTG + 3TC)?", "referencia": "El cambio a dolutegravir + lamivudina (DTG + 3TC) es una opción adecuada en pacientes con replicación viral suprimida que quieran simplificar o evitar efectos adversos, sin resistencia conocida o sospechada a lamivudina ni a inhibidores de integrasa (y sin coinfección por VHB)."},
    {"pregunta": "¿A qué dosis se administra raltegravir cuando se combina con rifampicina?", "referencia": "Con rifampicina, raltegravir se administra a dosis de 800 mg cada 12 horas."},
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
        payloads = RETRIEVER(pregunta)                      # list[dict]
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
