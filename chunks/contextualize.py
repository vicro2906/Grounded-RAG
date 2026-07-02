#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contextualize.py
================
Contextual Retrieval (Anthropic-style) for the GeSIDA guideline chunks.

PROBLEM IT SOLVES: when the guides are chunked, each chunk loses the context it came from.
A fragment that says "it is given at 50 mg/12 h" mentions NEITHER the drug NOR that it is the
rifampicin regimen; a fragment about "first trimester" may not say "pregnancy". The search
engine (dense + BM25) then fails to match that chunk to the question even if it is the right one.

WHAT IT DOES: for each chunk, a cheap LLM (gpt-4o-mini) writes 1-2 Spanish sentences that
SITUATE the fragment within its guide (what it is about, which section/population it applies to),
using as context the guide title, the section_path and a WINDOW of neighbouring chunks.
That context is PREPENDED to the text ONLY for retrieval (`text_for_retrieval`), while the
literal `text` is kept intact for the CITATION. That is: the synthetic text steers the
matching, but is NEVER cited (respects priority #1: do not hallucinate / verifiable citation).

OUTPUT: chunks_contextual.jsonl = the same chunks + two new fields:
  - "context":            the generated sentence(s).
  - "text_for_retrieval": f"{context}\n\n{text}"  (what the uploader embeds and BM25-indexes).
Then: python chunks/upload_to_qdrant_hybrid.py chunks/chunks_contextual.jsonl --recreate
(the uploader already uses text_for_retrieval if present, and stores the literal `text` in the payload).

Resumable: if --out already exists, the already-done chunk_ids are skipped (written incrementally).

Usage:
    python chunks/contextualize.py --dry-run                 # estimates cost, no LLM call
    python chunks/contextualize.py --limit 10                # PROBE: only 10 (measures real cost)
    python chunks/contextualize.py                           # all (resumable)
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

DEFAULT_IN  = "chunks/chunks.jsonl"
DEFAULT_OUT = "chunks/chunks_contextual.jsonl"
DEFAULT_MODEL = "gpt-4o-mini"     # cheap; encapsulated here so we can migrate to Azure if needed
DEFAULT_WINDOW = 2                # number of neighbour chunks (before and after) passed as context

# gpt-4o-mini prices (verify; may change). For the --dry-run estimate.
PRICE_IN_PER_1M  = 0.15
PRICE_OUT_PER_1M = 0.60
EST_OUT_TOKENS = 80              # the 1-2 context sentences

SYS_PROMPT = (
    "Eres un asistente que prepara fragmentos de guías clínicas de VIH (GeSIDA) para un "
    "buscador. Dado un fragmento y su contexto dentro de la guía, escribe UNA frase en "
    "español, DENSA en términos clínicos, que describa de qué trata el fragmento para "
    "mejorar su recuperación. Nombra EXPLÍCITAMENTE las entidades clave que aparecen o a las "
    "que se refiere: fármacos y sus siglas, condición o situación clínica, población y el "
    "tema concreto; y conserva el grado de recomendación (p. ej. A-I) si lo hay. "
    "REGLAS ESTRICTAS: empieza DIRECTAMENTE por el contenido clínico; NO uses 'El/Este "
    "fragmento', 'El/Este texto', 'El documento' ni ninguna fórmula que mencione 'fragmento', "
    "'texto' o 'documento'; UNA sola frase (máximo ~40 palabras); NO añadas información que no "
    "esté en el fragmento o su contexto; NO inventes datos, dosis ni recomendaciones; NO "
    "incluyas relleno genérico (p. ej. 'dirigido a profesionales sanitarios'); responde solo "
    "con la frase, sin comillas ni preámbulos."
)


def load_chunks(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def group_ordered(chunks):
    """Group by guide and sort by chunk_index, so each chunk can be given its neighbours."""
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c.get("source_file", "?"), []).append(c)
    for doc in by_doc.values():
        doc.sort(key=lambda c: c.get("chunk_index", 0))
    return by_doc


def build_user_prompt(chunk, neighbors_text):
    section = " > ".join(chunk.get("section_path") or []) or "(sin sección)"
    heading = chunk.get("heading", "") or ""
    return (
        f"GUÍA: {chunk.get('doc_title', '')}\n"
        f"TEMA: {chunk.get('topic', '')}\n"
        f"SECCIÓN: {section} {heading}\n\n"
        f"FRAGMENTOS VECINOS (solo para contexto, NO los resumas):\n{neighbors_text}\n\n"
        f"FRAGMENTO A SITUAR:\n{chunk['text']}\n\n"
        f"Escribe el contexto (1-2 frases):"
    )


def neighbors_for(doc_chunks, i, window):
    """Text of the `window` chunks before and after chunk i within its guide."""
    lo, hi = max(0, i - window), min(len(doc_chunks), i + window + 1)
    parts = [doc_chunks[j]["text"] for j in range(lo, hi) if j != i]
    return "\n---\n".join(parts) or "(sin vecinos)"


def estimate_cost(by_doc, window):
    """Token/cost estimate WITHOUT calling the LLM (uses each chunk's n_tokens)."""
    sys_tok = len(SYS_PROMPT) // 4
    in_tok = 0
    n = 0
    for doc in by_doc.values():
        for i, c in enumerate(doc):
            lo, hi = max(0, i - window), min(len(doc), i + window + 1)
            neigh = sum(doc[j].get("n_tokens", 0) for j in range(lo, hi) if j != i)
            in_tok += sys_tok + c.get("n_tokens", 0) + neigh + 120  # 120 = approx scaffolding
            n += 1
    out_tok = n * EST_OUT_TOKENS
    cost = in_tok / 1e6 * PRICE_IN_PER_1M + out_tok / 1e6 * PRICE_OUT_PER_1M
    return n, in_tok, out_tok, cost


def contextualize_one(client, model, chunk, neighbors_text):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": build_user_prompt(chunk, neighbors_text)},
        ],
        temperature=0,
        max_tokens=160,
    )
    return resp.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN, help="input chunks.jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output jsonl (resumable)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="neighbours before/after")
    ap.add_argument("--limit", type=int, default=0, help="only N chunks (real-cost PROBE)")
    ap.add_argument("--source", default="", help="only chunks whose source_file contains this "
                    "text (e.g. VIH_TB) — to probe hard cases; neighbours are taken from the whole doc")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="concurrency; low (2) so as not to saturate the TPM cap (200k on mini)")
    ap.add_argument("--dry-run", action="store_true", help="estimate cost; no LLM call")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.inp))
    by_doc = group_ordered(chunks)

    n, in_tok, out_tok, cost = estimate_cost(by_doc, args.window)
    print(f"{n} chunks | ~{in_tok:,} input tok + ~{out_tok:,} output tok | "
          f"estimated cost (mini) ~ ${cost:.3f}")
    if args.dry_run:
        print("dry-run: the LLM is not called.")
        return

    if not OPENAI_API_KEY:
        print("Missing OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["chunk_id"] for l in out_path.open(encoding="utf-8")}
        print(f"Resuming: {len(done)} chunks already done in {out_path.name}, skipped.")

    # Work list: (chunk, neighbors_text), skipping the already-done ones and honouring --limit.
    work = []
    for doc in by_doc.values():
        for i, c in enumerate(doc):
            if c["chunk_id"] in done:
                continue
            if args.source and args.source not in c.get("source_file", ""):
                continue   # neighbors still come from the full doc, only the TARGET is filtered
            work.append((c, neighbors_for(doc, i, args.window)))
    if args.limit:
        work = work[:args.limit]
    print(f"To process: {len(work)} chunks (model {args.model}, window {args.window}).")

    # High max_retries: the SDK retries 429s (TPM) with exponential backoff honouring
    # Retry-After, so the job self-regulates to the tokens/min cap instead of skipping chunks.
    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=10)
    write_lock = threading.Lock()
    f_out = out_path.open("a", encoding="utf-8")   # append: incremental + resumable

    def task(item):
        chunk, neigh = item
        ctx = contextualize_one(client, args.model, chunk, neigh)
        enriched = dict(chunk)
        enriched["context"] = ctx
        enriched["text_for_retrieval"] = f"{ctx}\n\n{chunk['text']}"
        return enriched

    n_ok = 0
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(task, it): it for it in work}
            for fut in as_completed(futures):
                chunk, _ = futures[fut]
                try:
                    enriched = fut.result()
                except Exception as e:
                    print(f"  failure on {chunk['chunk_id']}: {e}", file=sys.stderr)
                    continue
                with write_lock:
                    f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    f_out.flush()
                    n_ok += 1
                    if n_ok % 25 == 0:
                        print(f"  {n_ok}/{len(work)} contextualized")
    finally:
        f_out.close()

    print(f"Done: {n_ok} chunks contextualized -> {out_path}")
    print("Next (upload to a NEW collection, does not overwrite the current one):")
    print(f"  python chunks/upload_to_qdrant_hybrid.py {args.out} "
          "--collection guias_vih_hibrida_ctx")


if __name__ == "__main__":
    main()
