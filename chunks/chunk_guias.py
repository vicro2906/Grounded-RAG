#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunk_guias.py
==============
Trocea guías clínicas de VIH (GeSIDA/SPNS) en formato Markdown en *chunks*
estructurados con metadatos, listos para indexar en un sistema RAG.

Section (texto crudo por encabezados, tamaño irregular) → normalize ajusta el tamaño y 
produce units (texto del tamaño bueno + metadatos estructurales heredados) → 
build_chunks le añade los metadatos que dependen del texto final y le antepone el breadcrumb 
al cuerpo → Chunk (el objeto definitivo que se serializa a JSONL y se sube a Qdrant).

Estrategia: troceo CONSCIENTE DE LA ESTRUCTURA (header-aware) con
normalización de tamaño.
  1. Parsea cada .md en un árbol de secciones según los encabezados ##, ###,
     ####, #####  conservando la ruta jerárquica completa (breadcrumb).
  2. Cada sección "hoja" (texto hasta el siguiente encabezado) es la unidad base.
  3. Normaliza tamaño:
        - secciones grandes  -> se parten por párrafos con solape.
        - secciones pequeñas -> se fusionan con hermanas del mismo padre H2.
        - tablas / recomendaciones / abreviaturas -> se mantienen intactas.
  4. Etiqueta cada chunk con metadatos (tema, ruta, tipo de contenido, grados
     de evidencia, números de sección, etc.).
  5. Exporta a JSONL (un chunk por línea) -> formato estándar de ingesta RAG.

Sin dependencias obligatorias (solo librería estándar). Si 'tiktoken' está
instalado se usa para contar tokens; si no, se usa una estimación por caracteres.

Uso:
    python chunk_guias.py /ruta/a/los/md  -o salida.jsonl
    python chunk_guias.py archivo1.md archivo2.md -o salida.jsonl
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# 0. CONFIGURACIÓN
# ---------------------------------------------------------------------------

# Tamaños objetivo expresados en TOKENS (aprox). Ajustables a tu modelo de
# embeddings. ~512-800 tokens es un buen rango para recuperación clínica.
TARGET_TOKENS = 600     # tamaño ideal de un chunk
MAX_TOKENS    = 900     # por encima de esto se parte una sección
MIN_TOKENS    = 200     # por debajo de esto se intenta fusionar
OVERLAP_TOKENS = 80     # solape al partir secciones grandes

# Registro de documentos: metadatos a nivel de fichero. Es más fiable
# asignarlos aquí que intentar parsearlos del texto. Edítalo si añades docs.
DOC_REGISTRY = {
    "TAR_2022.md": {
        "doc_title": "Documento de consenso de GeSIDA/PNS sobre TAR en adultos con VIH",
        "topic": "tratamiento_antirretroviral",
        "organization": "GeSIDA/PNS",
        "year": 2022,
    },
    "VIH_TB.md": {
        "doc_title": "Tratamiento de la tuberculosis en personas con infección por VIH",
        "topic": "vih_tuberculosis",
        "organization": "GeSIDA/PNS",
        "year": 2018,
    },
    "VIH_embarazo.md": {
        "doc_title": "Infección por VIH y reproducción, embarazo, parto y profilaxis de la transmisión vertical",
        "topic": "vih_embarazo",
        "organization": "SPNS/GeSIDA/SEGO/SEIP",
        "year": 2018,
    },
    "adherencia.md": {
        "doc_title": "Recomendaciones GeSIDA/SEFH/PNS para mejorar la adherencia al tratamiento",
        "topic": "adherencia",
        "organization": "PNS/GeSIDA/SEFH",
        "year": 2020,
    },
    "medicina_preventiva.md": {
        "doc_title": "Vacunación e inmunización en personas con VIH (medicina preventiva)",
        "topic": "medicina_preventiva_vacunas",
        "organization": "GeSIDA/SEMPSPGS",
        "year": 2024,
    },
    "profilaxis.md": {
        "doc_title": "Profilaxis posexposición ocupacional y no ocupacional al VIH, VHB y VHC",
        "topic": "profilaxis_postexposicion",
        "organization": "GeSIDA/GEHEP/SEIP/SEMPSPGS",
        "year": 2025,
    },
    "ManejoclinicodelasalteracionesNC.md": {
        "doc_title": "Manejo clínico de las alteraciones neurocognitivas asociadas al VIH",
        "topic": "alteraciones_neurocognitivas",
        "organization": "GeSIDA/PNS",
        "year": 2013,
    },
}

DEFAULT_META = {
    "doc_title": None, "topic": "vih_general",
    "organization": "GeSIDA", "year": None,
}


def _norm_name(name: str) -> str:
    """Normaliza un nombre de archivo para buscarlo en el registro:
    minúsculas y espacios/guiones -> guion bajo. Así 'medicina preventiva.md'
    y 'medicina_preventiva.md' apuntan a la misma entrada."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


# Registro indexado por nombre normalizado (se construye una sola vez).
_NORMALIZED_REGISTRY = {_norm_name(k): v for k, v in DOC_REGISTRY.items()}


def lookup_meta(filename: str) -> dict:
    meta = _NORMALIZED_REGISTRY.get(_norm_name(filename))
    if meta is None:
        return {**DEFAULT_META, "doc_title": Path(filename).stem}
    return meta

# ---------------------------------------------------------------------------
# 1. CONTEO DE TOKENS
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:
    # Estimación: ~4 caracteres por token (razonable para español).
    def count_tokens(text: str) -> int:
        return max(1, round(len(text) / 4))

# ---------------------------------------------------------------------------
# 2. EXPRESIONES REGULARES
# ---------------------------------------------------------------------------
RE_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Número de sección al inicio del encabezado: "3.2.2." o "1." -> "3.2.2"
RE_SECNUM  = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
# Grados de evidencia: (A-I), (A-II), (B-III), también (AII), (B-I), con/sin **
RE_GRADE   = re.compile(r"\(\s*\*{0,2}\s*([ABC])\s*-?\s*(I{1,3})\s*\*{0,2}\s*\)")
RE_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

# Encabezados que marcan un tipo de contenido especial
RE_RECS    = re.compile(r"recomendaci(o|ó)n", re.IGNORECASE)
RE_ABREV   = re.compile(r"abreviatura|listado de abreviaturas", re.IGNORECASE)
RE_TABLA_H = re.compile(r"^(tabla|figura)\b", re.IGNORECASE)
RE_ANEXO   = re.compile(r"\banexo|algoritmo", re.IGNORECASE)
RE_METODO  = re.compile(r"metodolog(i|í)a", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 3. ESTRUCTURAS DE DATOS
# ---------------------------------------------------------------------------
@dataclass
class Section:
    """Una sección 'hoja': un encabezado y el texto hasta el siguiente encabezado."""
    level: int
    heading: str
    body: str
    breadcrumb: List[str]          # encabezados ancestros incluido el propio
    section_number: Optional[str]  # "3.2.2"  o None

    @property
    def tokens(self) -> int:
        return count_tokens(self.body)


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    doc_title: Optional[str]
    topic: str
    organization: str
    year: Optional[int]
    section_path: List[str]        # breadcrumb de encabezados
    section_number: Optional[str]
    heading: str
    heading_level: int
    content_type: str              # text|recommendations|table|abbreviations|appendix|methodology
    evidence_grades: List[str]
    chunk_index: int               # índice dentro del documento
    n_tokens: int
    n_chars: int
    text: str                      # texto del chunk, con breadcrumb antepuesto


# ---------------------------------------------------------------------------
# 4. PARSEO -> lista de Section
# ---------------------------------------------------------------------------
def parse_sections(md_text: str) -> List[Section]:
    """Divide el markdown en secciones hoja conservando la ruta de encabezados."""
    lines = md_text.splitlines()
    sections: List[Section] = []
    stack: List[tuple] = []        # [(level, heading_text), ...]
    cur_level, cur_heading, cur_breadcrumb = 0, "Preámbulo", []
    buf: List[str] = []

    def flush():
        body = "\n".join(buf).strip()
        # Guardamos también secciones vacías solo si tienen encabezado real;
        # las vacías se fusionarán luego o se descartan en normalización.
        sections.append(Section(
            level=cur_level,
            heading=cur_heading,
            body=body,
            breadcrumb=list(cur_breadcrumb),
            section_number=_secnum(cur_heading),
        ))

    for line in lines:
        m = RE_HEADING.match(line)
        if m:
            # cerrar la sección anterior
            flush()
            level = len(m.group(1))
            heading = m.group(2).strip()
            # actualizar la pila de ancestros
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            cur_level = level
            cur_heading = heading
            cur_breadcrumb = [h for (_, h) in stack]
            buf = []
        else:
            buf.append(line)
    flush()

    # descartar el bloque "Preámbulo" si está vacío
    return [s for s in sections if not (s.heading == "Preámbulo" and not s.body)]


def _secnum(heading: str) -> Optional[str]:
    m = RE_SECNUM.match(heading)
    return m.group(1) if m else None


def _common_secnum(numbers: List[Optional[str]]) -> Optional[str]:
    """Prefijo numérico común de varios números de sección.
    Sirve para etiquetar correctamente un chunk que fusiona subsecciones:
        ['4.2.1', '4.2.2']        -> '4.2'
        ['4.2.4.1', '4.2.4.2']    -> '4.2.4'
        ['4.1', '4.2']            -> '4'
        ['4.1', '4.1.1']          -> '4.1'
    Si no hay prefijo común (números de ramas distintas) devuelve None.
    """
    nums = [n for n in numbers if n]
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    split = [n.split(".") for n in nums]
    common: List[str] = []
    for parts in zip(*split):
        if all(p == parts[0] for p in parts):
            common.append(parts[0])
        else:
            break
    return ".".join(common) if common else None


def _ancestor_for_secnum(breadcrumb: List[str], secnum: Optional[str]):
    """Localiza en el breadcrumb el ancestro cuyo número de sección es `secnum`.
    Devuelve (heading_del_ancestro, breadcrumb_truncado_hasta_él) o (None, None)
    si no se encuentra (p.ej. el ancestro era un contenedor sin encabezado propio).
    """
    if not secnum:
        return None, None
    for k, h in enumerate(breadcrumb):
        if _secnum(h) == secnum:
            return h, breadcrumb[:k + 1]
    return None, None


# ---------------------------------------------------------------------------
# 5. CLASIFICACIÓN Y EXTRACCIÓN DE METADATOS
# ---------------------------------------------------------------------------
def classify(sec: Section) -> str:
    h = sec.heading
    if RE_ABREV.search(h):
        return "abbreviations"
    if RE_RECS.search(h) and sec.level >= 4:
        return "recommendations"
    if RE_TABLA_H.match(h):
        return "table"
    if RE_ANEXO.search(h):
        return "appendix"
    if RE_METODO.search(h):
        return "methodology"
    # tabla "incrustada": cuerpo mayoritariamente filas de tabla
    body_lines = [l for l in sec.body.splitlines() if l.strip()]
    if body_lines:
        table_lines = sum(1 for l in body_lines if RE_TABLE_ROW.match(l))
        if table_lines >= 3 and table_lines / len(body_lines) > 0.6:
            return "table"
    return "text"


def extract_grades(text: str) -> List[str]:
    """Devuelve grados normalizados ('A-II', 'B-III'...) sin duplicados, en orden."""
    out, seen = [], set()
    for letter, roman in RE_GRADE.findall(text):
        g = f"{letter.upper()}-{roman.upper()}"
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def has_table(body: str) -> bool:
    return any(RE_TABLE_ROW.match(l) for l in body.splitlines())


# ---------------------------------------------------------------------------
# 6. NORMALIZACIÓN DE TAMAÑO (merge + split)
# ---------------------------------------------------------------------------
def split_large(sec: Section) -> List[str]:
    """Parte un cuerpo grande por párrafos con solape, sin romper tablas."""
    paras = re.split(r"\n\s*\n", sec.body)
    pieces, cur, cur_tok = [], [], 0
    for p in paras:
        p = p.strip()
        if not p:
            continue
        ptok = count_tokens(p)
        # una tabla o párrafo enorme va en su propia pieza
        if ptok > MAX_TOKENS:
            if cur:
                pieces.append("\n\n".join(cur)); cur, cur_tok = [], 0
            pieces.append(p)
            continue
        if cur_tok + ptok > MAX_TOKENS and cur:
            pieces.append("\n\n".join(cur))
            # solape: arrastrar los últimos párrafos hasta OVERLAP_TOKENS
            overlap, otok = [], 0
            for q in reversed(cur):
                qt = count_tokens(q)
                if otok + qt > OVERLAP_TOKENS:
                    break
                overlap.insert(0, q); otok += qt
            cur, cur_tok = list(overlap), otok
        cur.append(p); cur_tok += ptok
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces or [sec.body]


def normalize(sections: List[Section]) -> List[dict]:
    """
    Convierte secciones en unidades de chunk (dicts con body+meta), aplicando
    fusión de secciones pequeñas y partición de las grandes.
    Devuelve dicts con: body, heading, level, breadcrumb, section_number,
    content_type.
    """
    units: List[dict] = []

    def emit(sec: Section, body: str, ctype: str,
             section_number: Optional[str] = None,
             heading: Optional[str] = None,
             breadcrumb: Optional[List[str]] = None):
        # Los parámetros opcionales permiten sobrescribir la identidad de la
        # sección cuando se fusionan varias subsecciones (ver bloque de merge):
        # el chunk se etiqueta con el ancestro común y no con la 1ª subsección.
        units.append({
            "body": body,
            "heading": sec.heading if heading is None else heading,
            "level": sec.level,
            "breadcrumb": sec.breadcrumb if breadcrumb is None else breadcrumb,
            "section_number": sec.section_number if section_number is None else section_number,
            "content_type": ctype,
        })

    i = 0
    n = len(sections)
    while i < n:
        sec = sections[i]
        ctype = classify(sec)

        # Encabezado contenedor sin texto propio (p.ej. "## 4." seguido de
        # "### 4.1."): no genera chunk; su título vive en el breadcrumb de
        # las subsecciones.
        if ctype == "text" and not sec.body.strip():
            i += 1
            continue

        # Tipos que NO se fusionan ni se parten arbitrariamente
        if ctype in ("table", "abbreviations"):
            emit(sec, sec.body, ctype)
            i += 1
            continue

        # Sección grande -> partir
        if sec.tokens > MAX_TOKENS:
            for piece in split_large(sec):
                # reclasificar por si la pieza es una tabla
                sub = Section(sec.level, sec.heading, piece, sec.breadcrumb, sec.section_number)
                emit(sec, piece, "table" if (ctype == "text" and has_table(piece)) else ctype)
            i += 1
            continue

        # Sección pequeña -> intentar fusionar con hermanas siguientes del mismo H2
        if sec.tokens < MIN_TOKENS and ctype == "text":
            merged_body = sec.body
            merged_heads = [sec.heading]
            merged_secnums = [sec.section_number]
            parent_h2 = sec.breadcrumb[1] if len(sec.breadcrumb) > 1 else (sec.breadcrumb[0] if sec.breadcrumb else "")
            j = i + 1
            while j < n and count_tokens(merged_body) < TARGET_TOKENS:
                nxt = sections[j]
                nxt_type = classify(nxt)
                nxt_parent = nxt.breadcrumb[1] if len(nxt.breadcrumb) > 1 else (nxt.breadcrumb[0] if nxt.breadcrumb else "")
                # no fusionar tablas/abreviaturas/recos ni cruzar de sección H2
                if nxt_type in ("table", "abbreviations") or nxt_parent != parent_h2:
                    break
                if not nxt.body.strip():
                    j += 1
                    continue
                addition = f"\n\n## {nxt.heading}\n{nxt.body}" if nxt.body else ""
                if count_tokens(merged_body + addition) > MAX_TOKENS:
                    break
                merged_body += addition
                merged_heads.append(nxt.heading)
                merged_secnums.append(nxt.section_number)
                j += 1
            if merged_body.strip():
                # Si realmente se han fusionado >1 subsección, el chunk ya no
                # pertenece a la 1ª subsección sino a su ancestro común: lo
                # reetiquetamos (p.ej. 4.2.1 + 4.2.2 -> 4.2) para que quien
                # busque por número encuentre el contenido en el nivel correcto.
                if len(merged_heads) > 1:
                    common = _common_secnum(merged_secnums)
                    if common and common != sec.section_number:
                        anc_head, anc_bc = _ancestor_for_secnum(sec.breadcrumb, common)
                        emit(sec, merged_body, ctype,
                             section_number=common,
                             heading=anc_head,      # None -> conserva el de sec
                             breadcrumb=anc_bc)      # None -> conserva el de sec
                    else:
                        emit(sec, merged_body, ctype)
                else:
                    emit(sec, merged_body, ctype)
            i = max(j, i + 1)
            continue

        # Sección de tamaño normal
        if sec.body.strip():
            emit(sec, sec.body, ctype)
        i += 1

    return units


# ---------------------------------------------------------------------------
# 7. CONSTRUCCIÓN DE CHUNKS CON METADATOS
# ---------------------------------------------------------------------------
def build_chunks(path: Path) -> List[Chunk]:
    md = path.read_text(encoding="utf-8")
    meta = lookup_meta(path.name)
    sections = parse_sections(md)
    units = normalize(sections)

    chunks: List[Chunk] = []
    seen_text: set = set()          # para deduplicar chunks de texto idéntico
    for idx, u in enumerate(units):
        breadcrumb = u["breadcrumb"]
        # El preámbulo (texto antes del primer encabezado) no tiene ruta:
        # le damos como contexto el título del documento.
        if not breadcrumb:
            breadcrumb = [meta.get("doc_title") or "Preámbulo"]
        # Anteponer el breadcrumb al texto mejora mucho la recuperación:
        # el embedding "ve" el contexto jerárquico de la sección.
        context_prefix = " > ".join(breadcrumb)
        text = f"{context_prefix}\n\n{u['body']}".strip() if context_prefix else u["body"]

        # Dedup: si dos secciones generan exactamente el mismo texto, conservamos
        # solo la primera (p.ej. subencabezados repetidos en formularios de anexo).
        norm = text.strip()
        if norm in seen_text:
            continue
        seen_text.add(norm)

        cid = hashlib.sha1(
            f"{path.name}:{idx}:{u['heading']}".encode("utf-8")
        ).hexdigest()[:16]

        chunks.append(Chunk(
            chunk_id=cid,
            source_file=path.name,
            doc_title=meta.get("doc_title"),
            topic=meta.get("topic", "vih_general"),
            organization=meta.get("organization", "GeSIDA"),
            year=meta.get("year"),
            section_path=breadcrumb,
            section_number=u["section_number"],
            heading=u["heading"],
            heading_level=u["level"],
            content_type=u["content_type"],
            evidence_grades=extract_grades(u["body"]),
            chunk_index=idx,
            n_tokens=count_tokens(text),
            n_chars=len(text),
            text=text,
        ))
    return chunks


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------
def gather_md_files(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.glob("*.md")))
        elif p.is_file() and p.suffix.lower() == ".md":
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description="Trocea guías VIH (Markdown) en chunks con metadatos.")
    ap.add_argument("inputs", nargs="+", help="Archivos .md o carpeta con .md")
    ap.add_argument("-o", "--output", default="chunks.jsonl", help="Archivo JSONL de salida")
    ap.add_argument("--stats", action="store_true", help="Imprime estadísticas")
    args = ap.parse_args()

    files = gather_md_files(args.inputs)
    if not files:
        print("No se encontraron archivos .md", file=sys.stderr)
        sys.exit(1)

    all_chunks: List[Chunk] = []
    for f in files:
        cs = build_chunks(f)
        all_chunks.extend(cs)
        print(f"  {f.name:38s} -> {len(cs):4d} chunks", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"\n{len(all_chunks)} chunks escritos en {args.output}", file=sys.stderr)

    if args.stats:
        import statistics as st
        toks = [c.n_tokens for c in all_chunks]
        from collections import Counter
        ctypes = Counter(c.content_type for c in all_chunks)
        print("\n--- ESTADÍSTICAS ---", file=sys.stderr)
        print(f"tokens/chunk: min={min(toks)} med={int(st.median(toks))} "
              f"media={int(st.mean(toks))} max={max(toks)}", file=sys.stderr)
        print(f"tipos de contenido: {dict(ctypes)}", file=sys.stderr)
        graded = sum(1 for c in all_chunks if c.evidence_grades)
        print(f"chunks con grado de evidencia: {graded}", file=sys.stderr)


if __name__ == "__main__":
    main()
