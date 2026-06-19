"""
evidencias.py — formateo de respuesta + panel de FUENTES para el RAG de VIH.
"""
import re
import textwrap
import difflib

WIDTH = 72

# La respuesta formateada es DATO que consumen varios frontends (LangGraph Studio,
# una futura web, la API), no solo el terminal. Por eso NO se incrustan códigos de
# color ANSI: ensuciarían cualquier consumidor que no sea una terminal. El color es
# una cuestión de presentación al imprimir, no del contenido.
class C:
    RESET = BOLD = DIM = BLUE = GREEN = GRAY = YELLOW = ""

# ───────────────────────────────────────────────────────── normalización
_EMPH = re.compile(r"[_*`]+")
_WS   = re.compile(r"\s+")
_GRADE_INLINE = re.compile(r"\(\s*[ABC]\s*-?\s*I{1,3}\s*\)")   # (A-I) (AII) (A- I)…
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

def _norm(s: str) -> str:
    """Minúsculas; quita énfasis markdown, grados en línea y puntuación; colapsa
    espacios. Se quita puntuación para que diferencias de coma/espacio/errata no
    bloqueen una coincidencia real."""
    s = _EMPH.sub("", s or "")
    s = _GRADE_INLINE.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip().lower()

# ───────────────────────────────────────────────────────── grados
_GRADE = re.compile(r"\(\s*([ABC])\s*-?\s*(I{1,3})\s*\)")

def _grades_in(text: str) -> list:
    out = []
    for letter, roman in _GRADE.findall(text):
        g = f"{letter}-{roman}"
        if g not in out:
            out.append(g)
    return out

# ─────────────────────────────────────────── partir un bloque en recomendaciones
_ITEM_SPLIT = re.compile(r"(?:^|\n)\s*(?:\d+[.\-]|•|-)\s+", re.MULTILINE)

def _strip_breadcrumb(text: str) -> str:
    """Quita la miga de pan inicial 'A > B > C' con la que empieza cada chunk."""
    parts = text.split("\n\n", 1)
    if parts and " > " in parts[0] and len(parts[0]) < 300:
        return parts[1] if len(parts) > 1 else ""
    return text

def split_items(chunk: dict) -> list:
    """[{'text': frase, 'grades': [...]}], una por recomendación. Los chunks que
    no son lista colapsan a un único item."""
    body = _strip_breadcrumb(chunk.get("text", ""))
    pieces = [p.strip() for p in _ITEM_SPLIT.split(body) if p.strip()]
    if len(pieces) <= 1:
        body = body.strip()
        return [{"text": body, "grades": _grades_in(body)}] if body else []
    return [{"text": p, "grades": _grades_in(p)} for p in pieces]

# ─────────────────────────────────────────── grados acotados a la cita
def _grades_for_quote(item_text: str, quote: str) -> list:
    """Dentro del item, conserva solo los grados cuya cláusula está realmente
    cubierta por la cita (cláusula = tramo que termina en un grado en línea).
    Si nada encaja, devuelve todos los grados del item."""
    nq = _norm(quote)
    segments, last = [], 0
    for m in _GRADE_INLINE.finditer(item_text):
        seg = item_text[last:m.start()]
        g = _grades_in(m.group(0))
        segments.append((seg, g[0] if g else None))
        last = m.end()
    kept = []
    for seg, g in segments:
        if g is None:
            continue
        ns = _norm(seg)
        if not ns:
            continue
        sm = difflib.SequenceMatcher(None, ns, nq)
        blk = sm.find_longest_match(0, len(ns), 0, len(nq))
        if blk.size / max(len(ns), 1) >= 0.5 and g not in kept:
            kept.append(g)
    return kept or _grades_in(item_text)

# ─────────────────────────────────────────── cita -> recomendación
def attribute(quote: str, chunk: dict):
    """
    Devuelve (status, frase_a_mostrar, grados):
      'exact' – la cita aparece literal (tras normalizar) dentro de un item
      'fuzzy' – la cita encaja por contención (>= .72); se muestra la frase de
                la guía en vez de la del modelo
      'miss'  – ningún item encaja; el panel pone una nota discreta
    """
    items = split_items(chunk)
    if not items:
        return "miss", "", []
    nq = _norm(quote)
    if not nq:
        return "miss", "", []

    for it in items:
        if nq in _norm(it["text"]):
            return "exact", quote.strip(), _grades_for_quote(it["text"], quote)

    best, best_score = None, 0.0
    for it in items:
        nt = _norm(it["text"])
        sm = difflib.SequenceMatcher(None, nq, nt)
        block = sm.find_longest_match(0, len(nq), 0, len(nt))
        containment = block.size / max(len(nq), 1)
        score = max(sm.ratio(), containment)
        if score > best_score:
            best, best_score = it, score
    if best is not None and best_score >= 0.72:
        return "fuzzy", best["text"], _grades_for_quote(best["text"], quote)

    return "miss", "", []

# ─────────────────────────────────────────── etiqueta de sección
_GENERIC = {"recomendaciones", "recomendaciones:", "recommendations"}
_NUM_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.*)$")

def _breadcrumb_parts(chunk: dict) -> list:
    """Devuelve la ruta de secciones. Prefiere section_path del payload; si no
    está, la reconstruye desde la miga de pan 'A > B > C' con la que empieza el
    texto del chunk (presente en el 100% de los bloques)."""
    path = chunk.get("section_path") or []
    if path:
        return [re.sub(r"\s+", " ", p).strip() for p in path]
    head = (chunk.get("text", "").split("\n\n", 1)[0]).strip()
    if " > " in head and len(head) < 300:
        return [re.sub(r"\s+", " ", p).strip() for p in head.split(" > ")]
    return []

def _split_num(part: str):
    """'3.1. FACTORES...' -> ('3.1', 'FACTORES...'); sin número -> (None, part)."""
    m = _NUM_PREFIX.match(part)
    return (m.group(1), m.group(2).strip()) if m else (None, part.strip())

def section_label(chunk: dict) -> str:
    heading = (chunk.get("heading") or "").strip()
    clean = re.sub(r"^[\d.\s]+", "", heading).strip()
    sec = (chunk.get("section_number") or "").strip()

    # 1) el chunk ya tiene su propio número -> está localizado
    if sec:
        return f"§{sec} · {clean}" if clean else f"§{sec}"

    # 2) heading genérico/sin número (p.ej. RECOMENDACIONES): hace falta el padre
    path = _breadcrumb_parts(chunk)
    if not path:
        return clean

    # hoja = última parte de la ruta; los ancestros la contextualizan
    leaf = clean or _split_num(path[-1])[1]
    ancestors = path[:-1] if _norm(path[-1]) == _norm(heading) or not clean else path

    # ancestro NUMERADO más cercano -> es el que de verdad desambigua 3.1 vs 4.2
    for part in reversed(ancestors):
        num, title = _split_num(part)
        if num:
            return f"§{num} {title} › {leaf}"

    # sin ancestro numerado: encadenar lo que haya
    chain = " › ".join(_split_num(p)[1] for p in ancestors)
    return f"{chain} › {leaf}" if chain else leaf

# ─────────────────────────────────────────── preguntas de seguimiento
def _followup_lines(answer) -> list:
    """Bloque 'PREGUNTAS DE SEGUIMIENTO' a partir de answer['follow_up_questions'].
    Devuelve [] si no hay preguntas, para no pintar el panel vacío."""
    qs = answer.get("follow_up_questions") or []
    qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
    if not qs:
        return []
    out = [
        f"{C.GRAY}{'─'*WIDTH}{C.RESET}",
        f"{C.BOLD}  PREGUNTAS DE SEGUIMIENTO{C.RESET} {C.DIM}({len(qs)}){C.RESET}",
        f"{C.GRAY}{'─'*WIDTH}{C.RESET}\n",
    ]
    for i, q in enumerate(qs, 1):
        wrapped = textwrap.fill(q, width=WIDTH-6,
                                initial_indent=f"  {i}. ", subsequent_indent="     ")
        out.append(f"{C.BLUE}{wrapped}{C.RESET}")
    out.append("")
    return out


# ─────────────────────────────────────────── aviso clínico
_DISCLAIMER_TXT = ("Herramienta de apoyo clínico; no sustituye en ningún caso el "
                   "juicio del profesional sanitario.")

def _disclaimer() -> str:
    """Aviso discreto al final. Solo se añade cuando hay una respuesta visible:
    si la información es insuficiente, el sistema no da contenido clínico, así que
    no aplica."""
    wrapped = textwrap.fill(_DISCLAIMER_TXT, width=WIDTH - 2,
                            initial_indent="  ", subsequent_indent="  ")
    return f"\n{C.GRAY}{'─'*WIDTH}{C.RESET}\n{C.DIM}{wrapped}{C.RESET}\n"


# ─────────────────────────────────────────── formateo final
def format_answer(answer, index):
    if not answer.get("sufficient_information", False):
        # Sin respuesta no se plantean preguntas de seguimiento: estas deben
        # relacionarse siempre con la respuesta dada.
        return (
            f"\n{C.BOLD}{C.BLUE}{'═'*WIDTH}{C.RESET}\n"
            f"{C.BOLD}{C.BLUE}  INFORMACIÓN INSUFICIENTE{C.RESET}\n"
            f"{C.BOLD}{C.BLUE}{'═'*WIDTH}{C.RESET}\n\n"
            f"  {answer.get('answer','La información no está disponible en las guías proporcionadas.')}\n"
        )

    lines = []
    lines.append(f"\n{C.BOLD}{C.BLUE}{'═'*WIDTH}{C.RESET}")
    lines.append(f"{C.BOLD}{C.BLUE}  RESPUESTA{C.RESET}")
    lines.append(f"{C.BOLD}{C.BLUE}{'═'*WIDTH}{C.RESET}\n")
    for para in answer["answer"].split("\n"):
        if para.strip():
            lines.append(textwrap.fill(para.strip(), width=WIDTH-2,
                                       initial_indent="  ", subsequent_indent="  "))
            lines.append("")

    # agrupar por identidad de sección (doc + section_number/heading)
    groups, order = {}, []
    for f in answer["sources_used"]:
        chunk = index.get(f["ref"])
        if chunk is None:
            continue
        key = (chunk.get("doc_title",""),
               chunk.get("section_number") or chunk.get("heading",""))
        if key not in groups:
            order.append(key)
            groups[key] = {"chunk": chunk, "evidence": [], "considered": False}
        status, sentence, grades = attribute(f.get("quote",""), chunk)
        if status == "miss":
            groups[key]["considered"] = True
        else:
            groups[key]["evidence"].append(
                {"status": status, "sentence": sentence, "grades": grades})

    if not order:
        return "\n".join(lines + _followup_lines(answer)) + _disclaimer()

    lines.append(f"{C.GRAY}{'─'*WIDTH}{C.RESET}")
    lines.append(f"{C.BOLD}  FUENTES{C.RESET} {C.DIM}({len(order)}){C.RESET}")
    lines.append(f"{C.GRAY}{'─'*WIDTH}{C.RESET}\n")

    for n, key in enumerate(order, start=1):
        g = groups[key]; chunk = g["chunk"]
        title = textwrap.fill(f"{chunk.get('doc_title','')} ({chunk.get('year','')})",
                              width=WIDTH-6, subsequent_indent="      ")
        lines.append(f"  {C.BOLD}{C.GREEN}[{n}]{C.RESET} {C.BOLD}{title}{C.RESET}")
        if chunk.get("organization"):
            lines.append(f"      {C.DIM}{chunk['organization']}{C.RESET}")
        sec = section_label(chunk)
        if sec:
            lines.append(f"      {sec}")

        for ev in g["evidence"]:
            grade_tag = f"  {C.DIM}[{', '.join(ev['grades'])}]{C.RESET}" if ev["grades"] else ""
            if ev["status"] == "exact":
                prefix, close, colour, tail = "      « ", " »", C.BLUE, ""
            else:
                prefix, close, colour = "      ‹ ", " ›", C.YELLOW
                tail = f" {C.DIM}(según la guía){C.RESET}"
            cite = textwrap.fill(ev["sentence"], width=WIDTH-8,
                                 initial_indent=prefix, subsequent_indent="        ")
            lines.append(f"{colour}{cite}{close}{C.RESET}{grade_tag}{tail}")

        if g["considered"] and not g["evidence"]:
            lines.append(f"      {C.DIM}(sección consultada · sin cita literal localizable){C.RESET}")
        lines.append("")

    lines += _followup_lines(answer)
    return "\n".join(lines) + _disclaimer()
