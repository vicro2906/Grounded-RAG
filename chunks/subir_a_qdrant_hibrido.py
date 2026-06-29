#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subir_a_qdrant_hibrido.py
=========================
Igual que subir_a_qdrant.py, pero crea una colección NUEVA preparada para
búsqueda híbrida (densa + sparse) sin tocar la colección original 'guias_vih'.

Cada punto lleva DOS vectores:
  - "dense": embedding semántico de OpenAI (text-embedding-3-large, 3072 dim).
  - "bm25":  vector sparse léxico (BM25, calculado en LOCAL con fastembed). El
             IDF lo calcula Qdrant en su lado (Modifier.IDF).

La búsqueda híbrida se hace en query-time con la Query API de Qdrant
(prefetch denso + prefetch sparse, fusionados con RRF). Ver rag.py.

Uso:
    python chunks/subir_a_qdrant_hibrido.py chunks/chunks.jsonl
    python chunks/subir_a_qdrant_hibrido.py chunks/chunks.jsonl --recreate
    python chunks/subir_a_qdrant_hibrido.py chunks/chunks.jsonl --dry-run
    # Contextual Retrieval a una coleccion NUEVA (no toca la actual; permite A/B y volver atras):
    python chunks/subir_a_qdrant_hibrido.py chunks/chunks_contextual.jsonl --collection guias_vih_hibrida_ctx
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
from fastembed import SparseTextEmbedding

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

QDRANT_URL     = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

COLLECTION = "guias_vih_hibrida"     # colección NUEVA; la original queda intacta

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM   = 3072
TRUNCATE_DIM = None                  # None = 3072 nativo

BM25_MODEL = "Qdrant/bm25"           # sparse léxico, corre en local

BATCH_SIZE = 96
PRICE_PER_1M_TOKENS = 0.13

KEYWORD_FIELDS = ["topic", "content_type", "source_file", "organization",
                  "evidence_grades"]
INTEGER_FIELDS = ["year", "heading_level"]


def load_chunks(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def stable_id(chunk_id: str) -> str:
    """UUID determinista a partir del chunk_id -> upserts idempotentes."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def vector_dim() -> int:
    return TRUNCATE_DIM or EMBED_DIM


def embed_batch(client: OpenAI, texts):
    """Embeddings densos de OpenAI con reintentos simples."""
    kwargs = {"model": EMBED_MODEL, "input": texts}
    if TRUNCATE_DIM:
        kwargs["dimensions"] = TRUNCATE_DIM
    for intento in range(5):
        try:
            resp = client.embeddings.create(**kwargs)
            return [d.embedding for d in resp.data]
        except Exception as e:
            wait = 2 ** intento
            print(f"    aviso: fallo de API ({e}); reintento en {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Fallaron los reintentos de embeddings")


def ensure_collection(client: QdrantClient, recreate: bool, collection: str):
    exists = client.collection_exists(collection)
    if exists and recreate:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            # Vectores con NOMBRE: denso + sparse en cada punto.
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_dim(),
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                # Qdrant calcula el IDF de BM25 en su lado a partir del corpus.
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )
        for f in KEYWORD_FIELDS:
            client.create_payload_index(
                collection, field_name=f,
                field_schema=models.PayloadSchemaType.KEYWORD)
        for f in INTEGER_FIELDS:
            client.create_payload_index(
                collection, field_name=f,
                field_schema=models.PayloadSchemaType.INTEGER)
        print(f"Coleccion '{collection}' creada (dense {vector_dim()}d coseno + sparse bm25 IDF).")
    else:
        print(f"Coleccion '{collection}' ya existe; se hara upsert.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Archivo chunks.jsonl")
    ap.add_argument("--collection", default=COLLECTION,
                    help=f"Nombre de la coleccion destino (por defecto '{COLLECTION}'). "
                         "Usa otro nombre (p.ej. guias_vih_hibrida_ctx) para NO sobrescribir "
                         "la actual y poder hacer A/B o volver atras.")
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

    missing = [n for n, v in [("OPENAI_API_KEY", OPENAI_API_KEY),
                              ("QDRANT_URL", QDRANT_URL),
                              ("QDRANT_API_KEY", QDRANT_API_KEY)] if not v]
    if missing:
        print(f"Faltan variables de entorno: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    bm25 = SparseTextEmbedding(BM25_MODEL)        # descarga el modelo la 1ª vez
    ensure_collection(qdrant, args.recreate, args.collection)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        # Contextual Retrieval: si el chunk trae "text_for_retrieval" (contexto + texto,
        # generado por chunks/contextualize.py), se EMBEBE e INDEXA en BM25 esa versión
        # enriquecida para mejorar el matching; si no, cae al texto crudo (compatibilidad).
        # El payload conserva el chunk entero, así que p["text"] sigue siendo el LITERAL
        # citable (evidence.py cita "text", no "text_for_retrieval").
        textos = [c.get("text_for_retrieval", c["text"]) for c in batch]
        dense_vecs = embed_batch(openai_client, textos)
        sparse_vecs = list(bm25.embed(textos))    # BM25 local (documentos)
        points = [
            models.PointStruct(
                id=stable_id(c["chunk_id"]),
                vector={
                    "dense": dense,
                    "bm25": models.SparseVector(
                        indices=sp.indices.tolist(),
                        values=sp.values.tolist(),
                    ),
                },
                payload=c,
            )
            for c, dense, sp in zip(batch, dense_vecs, sparse_vecs)
        ]
        qdrant.upsert(collection_name=args.collection, points=points)
        print(f"  subidos {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"Listo. Ingesta hibrida completada en Qdrant Cloud (coleccion '{args.collection}').")


if __name__ == "__main__":
    main()
