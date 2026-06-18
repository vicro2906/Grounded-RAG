"""
abreviaturas.py — Diccionario SIGLA -> nombre completo de las guías GeSIDA de VIH.

Lo usa el nodo de rephrasing (rag.rephrase) para normalizar los términos de la
consulta: como las guías usan TANTO siglas como nombres completos (p.ej. "DTG"
aparece mucho más que "dolutegravir", pero ambos existen), la consulta se reescribe
incluyendo AMBAS formas para casar con el texto recuperable.
"""

ABREVIATURAS = {
    # --- Fármacos antirretrovirales ---
    "3TC": "lamivudina",
    "ABC": "abacavir",
    "ATV": "atazanavir",
    "BIC": "bictegravir",
    "CAB": "cabotegravir",
    "COBI": "cobicistat",
    "DRV": "darunavir",
    "DTG": "dolutegravir",
    "DOR": "doravirina",
    "EFV": "efavirenz",
    "ETR": "etravirina",
    "EVG": "elvitegravir",
    "EVG/c": "elvitegravir potenciado con cobicistat",
    "FTC": "emtricitabina",
    "LPV": "lopinavir",
    "MVC": "maraviroc",
    "NVP": "nevirapina",
    "RAL": "raltegravir",
    "RPV": "rilpivirina",
    "RTV": "ritonavir",
    "TAF": "tenofovir alafenamida",
    "TDF": "tenofovir disoproxil fumarato",
    "TDx": "tenofovir disoproxil",
    "TFV": "tenofovir",
    "XTC": "lamivudina o emtricitabina",

    # --- Clases de fármacos ---
    "FAR": "fármacos antirretrovirales",
    "INI": "inhibidor de la integrasa",
    "IP": "inhibidor de la proteasa",
    "IP/p": "inhibidor de la proteasa potenciado",
    "ITIAN": "inhibidor de la transcriptasa inversa análogo de nucleósido/nucleótido",
    "ITINN": "inhibidor de la transcriptasa inversa no nucleósido",

    # --- Conceptos clínicos ---
    "AP": "acción prolongada",
    "BID": "dos veces al día",
    "QD": "una vez al día",
    "CVP": "carga viral plasmática",
    "FGe": "filtrado glomerular estimado",
    "FV": "fracaso virológico",
    "MR": "mutaciones de resistencia",
    "TAMs": "mutaciones asociadas a resistencia a análogos de la timidina",
    "PDDI": "potenciales interacciones farmacológicas",
    "RHS": "reacción de hipersensibilidad",
    "SIRI": "síndrome inflamatorio de reconstitución inmune",
    "SNC": "sistema nervioso central",
    "TAR": "tratamiento antirretroviral",
    "TO": "tratamiento optimizado",
    "TB": "tuberculosis",
    "ITS": "infecciones de transmisión sexual",
    "NPJ": "neumonía por Pneumocystis jirovecii",
    "PrEP": "profilaxis preexposición",

    # --- Virus ---
    "VIH-1": "virus de la inmunodeficiencia humana tipo 1",
    "VIH-2": "virus de la inmunodeficiencia humana tipo 2",
    "VHA": "virus de la hepatitis A",
    "VHB": "virus de la hepatitis B",
    "VHC": "virus de la hepatitis C",
}
