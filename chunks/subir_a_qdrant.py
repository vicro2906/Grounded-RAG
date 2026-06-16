#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subir_a_qdrant.py
=================
Lee chunks.jsonl (de chunk_guias.py), calcula embeddings con el modelo más
avanzado de OpenAI (text-embedding-3-large, 3072 dim) y los sube a una
colección de Qdrant Cloud. Los metadatos van como 'payload' para poder filtrar.

Requisitos:
    pip install qdrant-client openai

Credenciales por variable de entorno (no las pongas en el código):
    export OPENAI_API_KEY="sk-..."
    export QDRANT_URL="https://xxxxx.cloud.qdrant.io:6333"
    export QDRANT_API_KEY="..."          # API key del cluster de Qdrant Cloud

Uso:
    python subir_a_qdrant.py chunks.jsonl
    python subir_a_qdrant.py chunks.jsonl --recreate     # borra y recrea la colección
    python subir_a_qdrant.py chunks.jsonl --dry-run      # no sube; solo valida y estima coste
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client import models

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

QDRANT_URL     = os.environ.get("QDRANT_URL")        # https://....cloud.qdrant.io:6333
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

COLLECTION = "guias_vih"

# Modelo de embeddings más avanzado de OpenAI.
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM   = 3072          # dimensión nativa de text-embedding-3-large
# Matryoshka: podrías reducir a 1024 o 256 pasando dimensions=... a la API,
# ahorrando almacenamiento a costa de algo de calidad. Por defecto, máxima
# calidad (3072). Si lo cambias, ajusta también la búsqueda.
TRUNCATE_DIM = None         # p.ej. 1024 para reducir; None = nativo (3072)

# OpenAI admite muchos inputs por llamada; 96 es un lote prudente.
BATCH_SIZE = 96
PRICE_PER_1M_TOKENS = 0.13  # USD orientativo de text-embedding-3-large

# Índices de payload para filtrar en las búsquedas.
KEYWORD_FIELDS = ["topic", "content_type", "source_file", "organization",
                  "evidence_grades"]
INTEGER_FIELDS = ["year", "heading_level"]


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------
def load_chunks(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def stable_id(chunk_id: str) -> str:
    """UUID determinista a partir del chunk_id -> upserts idempotentes."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def vector_dim() -> int:
    return TRUNCATE_DIM or EMBED_DIM


def embed_batch(client: OpenAI, texts):
    """Llama a la API de embeddings con reintentos simples."""
    kwargs = {"model": EMBED_MODEL, "input": texts}
    if TRUNCATE_DIM:
        kwargs["dimensions"] = TRUNCATE_DIM
    for intento in range(5):
        try:
            resp = client.embeddings.create(**kwargs)
            return [d.embedding for d in resp.data]
        except Exception as e:                      # rate limit / red
            wait = 2 ** intento
            print(f"    aviso: fallo de API ({e}); reintento en {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Fallaron los reintentos de embeddings")


# ---------------------------------------------------------------------------
# COLECCIÓN
# ---------------------------------------------------------------------------
def ensure_collection(client: QdrantClient, recreate: bool):
    exists = client.collection_exists(COLLECTION)
    if exists and recreate:
        client.delete_collection(COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=vector_dim(),
                distance=models.Distance.COSINE,
            ),
        )
        for f in KEYWORD_FIELDS:
            client.create_payload_index(
                COLLECTION, field_name=f,
                field_schema=models.PayloadSchemaType.KEYWORD)
        for f in INTEGER_FIELDS:
            client.create_payload_index(
                COLLECTION, field_name=f,
                field_schema=models.PayloadSchemaType.INTEGER)
        print(f"Coleccion '{COLLECTION}' creada (dim={vector_dim()}, coseno).")
    else:
        print(f"Coleccion '{COLLECTION}' ya existe; se hara upsert.")


# ---------------------------------------------------------------------------
# INGESTA
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Archivo chunks.jsonl")
    ap.add_argument("--recreate", action="store_true", help="Borra la coleccion antes de subir")
    ap.add_argument("--dry-run", action="store_true", help="No sube; valida y estima coste")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.jsonl))
    tot_tokens = sum(c.get("n_tokens", 0) for c in chunks)
    coste = tot_tokens / 1_000_000 * PRICE_PER_1M_TOKENS
    print(f"{len(chunks)} chunks | ~{tot_tokens:,} tokens | "
          f"coste estimado embeddings ~ ${coste:.3f}")

    if args.dry_run:
        print("dry-run: no se sube nada.")
        return

    # Validacion de credenciales
    missing = [n for n, v in [("OPENAI_API_KEY", OPENAI_API_KEY),
                              ("QDRANT_URL", QDRANT_URL),
                              ("QDRANT_API_KEY", QDRANT_API_KEY)] if not v]
    if missing:
        print(f"Faltan variables de entorno: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    ensure_collection(qdrant, args.recreate)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vectors = embed_batch(openai_client, [c["text"] for c in batch])
        points = [
            models.PointStruct(
                id=stable_id(c["chunk_id"]),
                vector=vec,
                payload=c,
            )
            for c, vec in zip(batch, vectors)
        ]
        qdrant.upsert(collection_name=COLLECTION, points=points)
        print(f"  subidos {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print("Listo. Ingesta completada en Qdrant Cloud.")


# ---------------------------------------------------------------------------
# EJEMPLO DE BUSQUEDA (con filtro de metadatos)
# ---------------------------------------------------------------------------
def ejemplo_busqueda():
    """Busqueda semantica restringida a recomendaciones de grado A."""
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    consulta = "cuando iniciar el tratamiento antirretroviral en un paciente con tuberculosis?"
    qvec = embed_batch(openai_client, [consulta])[0]

    res = qdrant.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=5,
        query_filter=models.Filter(must=[
            models.FieldCondition(key="content_type",
                                  match=models.MatchValue(value="recommendations")),
            models.FieldCondition(key="evidence_grades",
                                  match=models.MatchAny(any=["A-I", "A-II", "A-III"])),
        ]),
    ).points
    for p in res:
        print(round(p.score, 3), p.payload["doc_title"], "->", p.payload["heading"])


if __name__ == "__main__":
    main()
