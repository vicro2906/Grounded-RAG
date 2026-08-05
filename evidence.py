"""
evidence.py — answer resolution (citation integrity) + SOURCES panel for the HIV RAG.

Two responsibilities, deliberately split:

  1. `resolve_answer` turns the model's raw answer plus the numbered chunks into an `AnswerView`
     — the answer AS DATA. Every integrity decision lives here (`attribute`), so no frontend can
     bypass it or reach a different verdict about what may be shown as a citation.
  2. `format_answer` RENDERS that view as the 72-column terminal panels.

The split exists because the module claimed to emit data and did not: it returned one string of
boxes and fixed-width wrapping, which a web view cannot turn back into collapsible sources or
clickable follow-ups. The terminal output is unchanged, byte for byte.
"""
import re
import textwrap
import difflib
from dataclasses import dataclass

from abbreviations import ABBREVIATIONS

WIDTH = 72

# --- Citation-integrity thresholds ---------------------------------------
# This module is the LAST barrier between a fabricated quote and the doctor, so the two knobs
# that decide whether a quote is certified are named and justified here rather than inlined.
#
# A quote must IDENTIFY a recommendation, not echo the guides' boilerplate: "se recomienda"
# appears verbatim in most chunks, so without a floor it would be certified as a literal
# citation AND inherit the item's evidence grades while supporting nothing. A quote clears the
# floor by being long enough to be specific, or by naming something clinical (drug or figure).
MIN_QUOTE_CHARS = 40
# Fuzzy acceptance: enough to absorb punctuation/wording drift, never enough to accept a
# different statement (which is additionally guarded by the clinical-token check below).
FUZZY_THRESHOLD = 0.72

# The formatted answer is DATA consumed by several frontends (Studio, a future web app, the
# API), not just the terminal — so no ANSI codes are embedded. Color is a print-time concern.
class C:
    RESET = BOLD = DIM = BLUE = GREEN = GRAY = YELLOW = ""

# Panel labels, in Spanish because the doctor reads them. Shared by every rendering of an
# AnswerView so the terminal and the web cannot drift into calling the same thing two names.
LBL_ANSWER       = "RESPUESTA"
LBL_INSUFFICIENT = "INFORMACIÓN INSUFICIENTE"
LBL_SOURCES      = "FUENTES"
LBL_FOLLOW_UPS   = "PREGUNTAS DE SEGUIMIENTO"
LBL_PER_GUIDE    = "(según la guía)"
LBL_CONSULTED    = "(sección consultada · sin cita literal localizable)"

# ───────────────────────────────────────────────────────── normalization
_EMPH = re.compile(r"[_*`]+")
_WS   = re.compile(r"\s+")
_GRADE_INLINE = re.compile(r"\(\s*[ABC]\s*-?\s*I{1,3}\s*\)")   # (A-I) (AII) (A- I)…
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

def _norm(s: str) -> str:
    """Lowercase; strip markdown emphasis, inline grades and punctuation; collapse
    whitespace. Punctuation is removed so comma/space/typo differences do not block a
    real match."""
    s = _EMPH.sub("", s or "")
    s = _GRADE_INLINE.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip().lower()

# ───────────────────────────────────────────────────────── grades
_GRADE = re.compile(r"\(\s*([ABC])\s*-?\s*(I{1,3})\s*\)")

def _grades_in(text: str) -> list:
    out = []
    for letter, roman in _GRADE.findall(text):
        g = f"{letter}-{roman}"
        if g not in out:
            out.append(g)
    return out

# ─────────────────────────────────────────── split a block into recommendations
_ITEM_SPLIT = re.compile(r"(?:^|\n)\s*(?:\d+[.\-]|•|-)\s+", re.MULTILINE)

def citable_text(chunk: dict) -> str:
    """The chunk's own words — what may be quoted back to the doctor.

    From the v2 corpus this is simply `text`: the breadcrumb lives in `text_for_retrieval`,
    which is embedded but never cited. A v1 chunk has no such field and carries its
    `A > B > C` path inside `text`, so it still needs stripping. The branch keys on the SCHEMA
    rather than sniffing for « > », which would eventually mangle a v2 chunk whose first
    paragraph happens to contain one; it disappears when v1 does."""
    text = chunk.get("text", "")
    if chunk.get("text_for_retrieval"):
        return text
    head, sep, rest = text.partition("\n\n")
    return rest if sep and " > " in head and len(head) < 300 else text

def split_items(chunk: dict) -> list:
    """[{'text': sentence, 'grades': [...]}], one per recommendation. Chunks that are
    not a list collapse to a single item."""
    body = citable_text(chunk)
    pieces = [p.strip() for p in _ITEM_SPLIT.split(body) if p.strip()]
    if len(pieces) <= 1:
        body = body.strip()
        return [{"text": body, "grades": _grades_in(body)}] if body else []
    return [{"text": p, "grades": _grades_in(p)} for p in pieces]

# ─────────────────────────────────── clinical tokens (drugs and figures)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# One pattern per drug/term, matching EITHER spelling — the guides (and the model) mix «TAF»
# and «tenofovir alafenamida» freely, so keying on the abbreviation alone would read a quote
# written in full names as committing to nothing. Built over _norm'd text on both sides, so
# «EVG/c» and «VIH-1» survive the punctuation stripping; word-bounded so «3TC» does not fire
# inside a longer token.
_ABBREV_RE = {
    abbr: re.compile(rf"\b({re.escape(_norm(abbr))}|{re.escape(_norm(name))})\b")
    for abbr, name in ABBREVIATIONS.items() if _norm(abbr)
}


def _clinical_tokens(text: str) -> set:
    """The drug abbreviations and figures a piece of text commits to.

    These are the tokens whose divergence changes the clinical meaning: fuzzy matching may
    absorb a comma or a rewording, but «TDF» vs «ABC» or «200» vs «500» is a different
    recommendation, not the same one written differently."""
    normalized = _norm(text)
    tokens = set(_NUMBER.findall(normalized))
    tokens |= {abbr for abbr, rx in _ABBREV_RE.items() if rx.search(normalized)}
    return tokens


def _is_substantive(quote: str) -> bool:
    """Does the quote carry enough content to identify a recommendation? See MIN_QUOTE_CHARS."""
    return len(_norm(quote)) >= MIN_QUOTE_CHARS or bool(_clinical_tokens(quote))


# ─────────────────────────────────────────── grades scoped to the quote
def _grades_for_quote(item_text: str, quote: str) -> list:
    """Within the item, keep only the grades whose clause is actually covered by the
    quote (clause = span ending in an inline grade). If nothing matches, return all the
    grades of the item."""
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
    if kept:
        return kept
    # No clause matched the quote. Falling back to EVERY grade in the item over-credits a
    # quote that only covers a slice of it (a partial quote of a two-recommendation item would
    # come back tagged [A-I, B-II]), so the fallback only applies when the quote covers the
    # item substantially.
    covered = len(_norm(quote)) / max(len(_norm(item_text)), 1)
    return _grades_in(item_text) if covered >= 0.6 else []

# ─────────────────────────────────────────── quote -> recommendation
def attribute(quote: str, chunk: dict):
    """
    Return (status, sentence_to_show, grades):
      'exact' – the quote appears literally (after normalizing) inside an item
      'fuzzy' – the quote matches by containment (>= FUZZY_THRESHOLD) AND commits to the same
                drugs/figures; the guideline sentence is shown instead of the model's
      'miss'  – nothing certifiable; the panel shows the section with a discreet note

    'miss' is the SAFE outcome, so every doubt resolves to it: the doctor then sees the section
    that was consulted without a quote, instead of a citation the guides do not back. Two
    rejections matter beyond "no match found":
      - a quote too thin to identify a recommendation (see MIN_QUOTE_CHARS), which would
        otherwise certify as literal and inherit the item's evidence grades;
      - a fuzzy match that names DIFFERENT drugs or figures than the sentence it matched.
        That is the dangerous case: fuzzy replaces the model's quote with the guideline
        sentence, so an answer that said «ABC» would be displayed backed by a real sentence
        saying «TDF o TAF». Fuzzy may fix wording, never swap the clinical content.
    """
    items = split_items(chunk)
    if not items:
        return "miss", "", []
    nq = _norm(quote)
    if not nq or not _is_substantive(quote):
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
    if best is not None and best_score >= FUZZY_THRESHOLD:
        if _clinical_tokens(quote) - _clinical_tokens(best["text"]):
            return "miss", "", []      # names drugs/figures the matched sentence does not
        return "fuzzy", best["text"], _grades_for_quote(best["text"], quote)

    return "miss", "", []

# ─────────────────────────────────────────── section label
_GENERIC = {"recomendaciones", "recomendaciones:", "recommendations"}
_NUM_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.*)$")

def _breadcrumb_parts(chunk: dict) -> list:
    """The section path, from `section_path` — the payload field that carries it.

    It used to fall back to parsing the breadcrumb out of the chunk text. That fallback is gone
    with the text it parsed: `section_path` is present on every chunk of every generation, so
    the fallback only ever answered when the real field was missing, which never happened."""
    return [re.sub(r"\s+", " ", p).strip() for p in (chunk.get("section_path") or [])]

def _split_num(part: str):
    """'3.1. FACTORES...' -> ('3.1', 'FACTORES...'); without number -> (None, part)."""
    m = _NUM_PREFIX.match(part)
    return (m.group(1), m.group(2).strip()) if m else (None, part.strip())

def section_label(chunk: dict) -> str:
    heading = (chunk.get("heading") or "").strip()
    clean = re.sub(r"^[\d.\s]+", "", heading).strip()
    sec = (chunk.get("section_number") or "").strip()

    # 1) the chunk already has its own number -> it is located
    if sec:
        return f"§{sec} · {clean}" if clean else f"§{sec}"

    # 2) generic/unnumbered heading (e.g. RECOMENDACIONES): the parent is needed
    path = _breadcrumb_parts(chunk)
    if not path:
        return clean

    # leaf = last part of the path; the ancestors give it context
    leaf = clean or _split_num(path[-1])[1]
    ancestors = path[:-1] if _norm(path[-1]) == _norm(heading) or not clean else path

    # nearest NUMBERED ancestor -> the one that truly disambiguates 3.1 vs 4.2
    for part in reversed(ancestors):
        num, title = _split_num(part)
        if num:
            return f"§{num} {title} › {leaf}"

    # no numbered ancestor: chain whatever is available
    chain = " › ".join(_split_num(p)[1] for p in ancestors)
    return f"{chain} › {leaf}" if chain else leaf

# ─────────────────────────────────────────── follow-up questions
def _followups(answer) -> list:
    """The follow-up questions, cleaned. A frontend that can make them CLICKABLE needs the list
    itself, not a rendered block — which is why this is separate from the panel below."""
    return [q.strip() for q in (answer.get("follow_up_questions") or [])
            if isinstance(q, str) and q.strip()]


def _followup_lines(qs: list) -> list:
    """'PREGUNTAS DE SEGUIMIENTO' block. Returns [] if there are no questions, so the empty
    panel is not drawn."""
    if not qs:
        return []
    out = [
        f"{C.GRAY}{'─'*WIDTH}{C.RESET}",
        f"{C.BOLD}  {LBL_FOLLOW_UPS}{C.RESET} {C.DIM}({len(qs)}){C.RESET}",
        f"{C.GRAY}{'─'*WIDTH}{C.RESET}\n",
    ]
    for i, q in enumerate(qs, 1):
        wrapped = textwrap.fill(q, width=WIDTH-6,
                                initial_indent=f"  {i}. ", subsequent_indent="     ")
        out.append(f"{C.BLUE}{wrapped}{C.RESET}")
    out.append("")
    return out


# ─────────────────────────────────────────── clinical disclaimer
_DISCLAIMER_TXT = ("Herramienta de apoyo clínico; no sustituye en ningún caso el "
                   "juicio del profesional sanitario.")

def _disclaimer() -> str:
    """Discreet footer. Added only when there is a visible answer: if the information
    is insufficient the system gives no clinical content, so it does not apply."""
    wrapped = textwrap.fill(_DISCLAIMER_TXT, width=WIDTH - 2,
                            initial_indent="  ", subsequent_indent="  ")
    return f"\n{C.GRAY}{'─'*WIDTH}{C.RESET}\n{C.DIM}{wrapped}{C.RESET}\n"


# ─────────────────────────────────────────── the answer as DATA
INSUFFICIENT_TEXT = "La información no está disponible en las guías proporcionadas."


@dataclass(frozen=True)
class Citation:
    """One quote that survived the integrity check. `sentence` is the model's own words when
    `status` is 'exact' and the GUIDELINE's sentence when it is 'fuzzy' — which is why the
    renderer marks the two differently."""
    status: str          # "exact" | "fuzzy"
    sentence: str
    grades: list


@dataclass(frozen=True)
class Source:
    """One guideline section behind the answer, with whatever could be certified from it."""
    title: str
    year: object
    organization: str
    section: str
    citations: list
    consulted_only: bool   # used, but nothing literal could be located in it


@dataclass(frozen=True)
class AnswerView:
    """The answer as data: what a frontend needs, and nothing about how to paint it."""
    sufficient: bool
    text: str
    sources: list
    follow_ups: list


def resolve_answer(answer, index) -> AnswerView:
    """Model answer + numbered chunks -> AnswerView, applying the citation integrity rules.

    Sources are grouped by SECTION identity (document + section number/heading), because several
    chunks of one section supporting the same claim are one source to a reader, not three. A
    quote that `attribute` rejects does not disappear: its section still shows as consulted,
    marked as having no locatable literal citation — saying less is the safe failure here."""
    if not answer.get("sufficient_information", False):
        # With no answer there are no follow-up questions: they must always relate to a
        # given answer.
        return AnswerView(False, answer.get("answer", INSUFFICIENT_TEXT), [], [])

    groups, order = {}, []
    for f in answer["sources_used"]:
        chunk = index.get(f["ref"])
        if chunk is None:
            continue
        key = (chunk.get("doc_title", ""),
               chunk.get("section_number") or chunk.get("heading", ""))
        if key not in groups:
            order.append(key)
            groups[key] = {"chunk": chunk, "citations": [], "considered": False}
        status, sentence, grades = attribute(f.get("quote", ""), chunk)
        if status == "miss":
            groups[key]["considered"] = True
        else:
            groups[key]["citations"].append(Citation(status, sentence, grades))

    sources = []
    for key in order:
        g = groups[key]
        chunk = g["chunk"]
        sources.append(Source(
            title=chunk.get("doc_title", ""), year=chunk.get("year", ""),
            organization=chunk.get("organization", ""), section=section_label(chunk),
            citations=g["citations"],
            consulted_only=bool(g["considered"] and not g["citations"])))

    return AnswerView(True, answer["answer"], sources, _followups(answer))


# ─────────────────────────────────────────── terminal rendering
def format_answer(answer, index):
    """The AnswerView rendered as the terminal panels. One of several possible renderings of the
    same view — a web frontend consumes `resolve_answer` directly instead."""
    view = resolve_answer(answer, index)
    if not view.sufficient:
        return (
            f"\n{C.BOLD}{C.BLUE}{'═'*WIDTH}{C.RESET}\n"
            f"{C.BOLD}{C.BLUE}  {LBL_INSUFFICIENT}{C.RESET}\n"
            f"{C.BOLD}{C.BLUE}{'═'*WIDTH}{C.RESET}\n\n"
            f"  {view.text}\n"
        )

    lines = []
    lines.append(f"\n{C.BOLD}{C.BLUE}{'═'*WIDTH}{C.RESET}")
    lines.append(f"{C.BOLD}{C.BLUE}  {LBL_ANSWER}{C.RESET}")
    lines.append(f"{C.BOLD}{C.BLUE}{'═'*WIDTH}{C.RESET}\n")
    for para in view.text.split("\n"):
        if para.strip():
            lines.append(textwrap.fill(para.strip(), width=WIDTH-2,
                                       initial_indent="  ", subsequent_indent="  "))
            lines.append("")

    if not view.sources:
        return "\n".join(lines + _followup_lines(view.follow_ups)) + _disclaimer()

    lines.append(f"{C.GRAY}{'─'*WIDTH}{C.RESET}")
    lines.append(f"{C.BOLD}  {LBL_SOURCES}{C.RESET} {C.DIM}({len(view.sources)}){C.RESET}")
    lines.append(f"{C.GRAY}{'─'*WIDTH}{C.RESET}\n")

    for n, src in enumerate(view.sources, start=1):
        title = textwrap.fill(f"{src.title} ({src.year})",
                              width=WIDTH-6, subsequent_indent="      ")
        lines.append(f"  {C.BOLD}{C.GREEN}[{n}]{C.RESET} {C.BOLD}{title}{C.RESET}")
        if src.organization:
            lines.append(f"      {C.DIM}{src.organization}{C.RESET}")
        if src.section:
            lines.append(f"      {src.section}")

        for cite in src.citations:
            grade_tag = f"  {C.DIM}[{', '.join(cite.grades)}]{C.RESET}" if cite.grades else ""
            if cite.status == "exact":
                prefix, close, colour, tail = "      « ", " »", C.BLUE, ""
            else:
                prefix, close, colour = "      ‹ ", " ›", C.YELLOW
                tail = f" {C.DIM}{LBL_PER_GUIDE}{C.RESET}"
            text = textwrap.fill(cite.sentence, width=WIDTH-8,
                                 initial_indent=prefix, subsequent_indent="        ")
            lines.append(f"{colour}{text}{close}{C.RESET}{grade_tag}{tail}")

        if src.consulted_only:
            lines.append(f"      {C.DIM}{LBL_CONSULTED}{C.RESET}")
        lines.append("")

    lines += _followup_lines(view.follow_ups)
    return "\n".join(lines) + _disclaimer()


# ─────────────────────────────────────────── markdown rendering (web)
def format_answer_markdown(view: AnswerView) -> str:
    """The same AnswerView as Markdown, for a frontend that lays out its own width.

    Deliberately WITHOUT the follow-up questions: a web view turns them into buttons, and the
    terminal's numbered list would be dead text beside them. Everything else is the same
    account with the same labels, so what a doctor is told cannot depend on the frontend —
    including the disclaimer, which rides with every visible answer."""
    if not view.sufficient:
        return f"**{LBL_INSUFFICIENT}**\n\n{view.text}\n\n---\n_{_DISCLAIMER_TXT}_"

    out = [view.text]
    if view.sources:
        out.append(f"---\n\n**{LBL_SOURCES} ({len(view.sources)})**")
        for n, src in enumerate(view.sources, start=1):
            head = f"**[{n}] {src.title} ({src.year})**"
            if src.organization:
                head += f" · {src.organization}"
            block = [head]
            if src.section:
                block.append(f"{src.section}")
            for cite in src.citations:
                grades = f" `{', '.join(cite.grades)}`" if cite.grades else ""
                tail = "" if cite.status == "exact" else f" _{LBL_PER_GUIDE}_"
                block.append(f"> {cite.sentence}{grades}{tail}")
            if src.consulted_only:
                block.append(f"_{LBL_CONSULTED}_")
            out.append("  \n".join(block))
    out.append(f"---\n_{_DISCLAIMER_TXT}_")
    return "\n\n".join(out)
