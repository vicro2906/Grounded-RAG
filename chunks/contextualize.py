#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contextualize.py
================
Contextual Retrieval (estilo Anthropic) para los chunks de las guías GeSIDA.

PROBLEMA QUE RESUELVE: al trocear las guías, cada chunk pierde el contexto del que vino.
Un fragmento que dice "se administra a 50 mg/12 h" no menciona NI el fármaco NI que es la
pauta con rifampicina; un fragmento sobre "primer trimestre" puede no decir "embarazo".
El buscador (denso + BM25) entonces no casa ese chunk con la pregunta aunque sea el correcto.

QUÉ HACE: por cada chunk, un LLM barato (gpt-4o-mini) escribe 1-2 frases en español que
SITÚAN el fragmento dentro de su guía (de qué trata, a qué sección/población aplica),
usando como contexto el título de la guía, el section_path y una VENTANA de chunks vecinos.
Ese contexto se ANTEPONE al texto SOLO para la recuperación (`text_for_retrieval`), mientras
que el `text` literal se conserva intacto para la CITA. Es decir: el texto sintético dirige
el matching, pero NUNCA se cita (respeta la prioridad nº1: no alucinar / cita verificable).

SALIDA: chunks_contextual.jsonl = los mismos chunks + dos campos nuevos:
  - "context":            la(s) frase(s) generada(s).
  - "text_for_retrieval": f"{context}\n\n{text}"  (lo que el uploader embebe e indexa en BM25).
Luego: python chunks/subir_a_qdrant_hibrido.py chunks/chunks_contextual.jsonl --recreate
(el uploader ya usa text_for_retrieval si existe, y guarda `text` literal en el payload).

Resumible: si --out ya existe, se saltan los chunk_id ya hechos (se escribe incremental).

Uso:
    python chunks/contextualize.py --dry-run                 # estima coste, no llama al LLM
    python chunks/contextualize.py --limit 10                # SONDA: solo 10 (mide coste real)
    python chunks/contextualize.py                           # todos (resumible)
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
DEFAULT_MODEL = "gpt-4o-mini"     # barato; encapsulado aquí para poder migrar a Azure si hace falta
DEFAULT_WINDOW = 2                # nº de chunks vecinos (antes y después) que se pasan como contexto

# Precios gpt-4o-mini (verificar; pueden cambiar). Para la estimación del --dry-run.
PRICE_IN_PER_1M  = 0.15
PRICE_OUT_PER_1M = 0.60
EST_OUT_TOKENS = 80              # las 1-2 frases de contexto

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
    """Agrupa por guía y ordena por chunk_index, para poder dar a cada chunk sus vecinos."""
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
    """Texto de los `window` chunks antes y después del chunk i dentro de su guía."""
    lo, hi = max(0, i - window), min(len(doc_chunks), i + window + 1)
    parts = [doc_chunks[j]["text"] for j in range(lo, hi) if j != i]
    return "\n---\n".join(parts) or "(sin vecinos)"


def estimate_cost(by_doc, window):
    """Estimación de tokens/coste SIN llamar al LLM (usa n_tokens de cada chunk)."""
    sys_tok = len(SYS_PROMPT) // 4
    in_tok = 0
    n = 0
    for doc in by_doc.values():
        for i, c in enumerate(doc):
            lo, hi = max(0, i - window), min(len(doc), i + window + 1)
            neigh = sum(doc[j].get("n_tokens", 0) for j in range(lo, hi) if j != i)
            in_tok += sys_tok + c.get("n_tokens", 0) + neigh + 120  # 120 = scaffolding aprox
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
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN, help="chunks.jsonl de entrada")
    ap.add_argument("--out", default=DEFAULT_OUT, help="jsonl de salida (resumible)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="vecinos antes/después")
    ap.add_argument("--limit", type=int, default=0, help="solo N chunks (SONDA de coste real)")
    ap.add_argument("--source", default="", help="solo chunks cuyo source_file contenga este "
                    "texto (p.ej. VIH_TB) — para sondear casos difíciles; vecinos se toman del doc completo")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="concurrencia; bajo (2) para no saturar el tope de TPM (200k en mini)")
    ap.add_argument("--dry-run", action="store_true", help="estima coste; no llama al LLM")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.inp))
    by_doc = group_ordered(chunks)

    n, in_tok, out_tok, cost = estimate_cost(by_doc, args.window)
    print(f"{n} chunks | ~{in_tok:,} tok entrada + ~{out_tok:,} tok salida | "
          f"coste estimado (mini) ~ ${cost:.3f}")
    if args.dry_run:
        print("dry-run: no se llama al LLM.")
        return

    if not OPENAI_API_KEY:
        print("Falta OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["chunk_id"] for l in out_path.open(encoding="utf-8")}
        print(f"Reanudando: {len(done)} chunks ya hechos en {out_path.name}, se saltan.")

    # Lista de trabajo: (chunk, neighbors_text), saltando los ya hechos y respetando --limit.
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
    print(f"A procesar: {len(work)} chunks (modelo {args.model}, ventana {args.window}).")

    # max_retries alto: el SDK reintenta los 429 (TPM) con backoff exponencial respetando
    # Retry-After, así el job se autorregula al tope de tokens/min en vez de saltarse chunks.
    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=10)
    write_lock = threading.Lock()
    f_out = out_path.open("a", encoding="utf-8")   # append: incremental + resumible

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
                    print(f"  fallo en {chunk['chunk_id']}: {e}", file=sys.stderr)
                    continue
                with write_lock:
                    f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    f_out.flush()
                    n_ok += 1
                    if n_ok % 25 == 0:
                        print(f"  {n_ok}/{len(work)} contextualizados")
    finally:
        f_out.close()

    print(f"Listo: {n_ok} chunks contextualizados -> {out_path}")
    print("Siguiente (sube a una coleccion NUEVA, no sobrescribe la actual):")
    print(f"  python chunks/subir_a_qdrant_hibrido.py {args.out} "
          "--collection guias_vih_hibrida_ctx")


if __name__ == "__main__":
    main()
