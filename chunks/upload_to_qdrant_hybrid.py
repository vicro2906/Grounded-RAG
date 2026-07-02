#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_qdrant_hybrid.py
==========================
Same as upload_to_qdrant.py, but creates a NEW collection prepared for
hybrid search (dense + sparse) without touching the original 'guias_vih' collection.

Each point carries TWO vectors:
  - "dense": OpenAI semantic embedding (text-embedding-3-large, 3072 dim).
  - "bm25":  sparse lexical vector (BM25, computed LOCALLY with fastembed). The
             IDF is computed by Qdrant on its side (Modifier.IDF).

The hybrid search is done at query time with Qdrant's Query API
(dense prefetch + sparse prefetch, fused with RRF). See rag.py.

Usage:
    python chunks/upload_to_qdrant_hybrid.py chunks/chunks.jsonl
    python chunks/upload_to_qdrant_hybrid.py chunks/chunks.jsonl --recreate
    python chunks/upload_to_qdrant_hybrid.py chunks/chunks.jsonl --dry-run
    # Contextual Retrieval to a NEW collection (does not touch the current one; allows A/B and rollback):
    python chunks/upload_to_qdrant_hybrid.py chunks/chunks_contextual.jsonl --collection guias_vih_hibrida_ctx
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

COLLECTION = "guias_vih_hibrida"     # NEW collection; the original stays intact

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM   = 3072
TRUNCATE_DIM = None                  # None = native 3072

BM25_MODEL = "Qdrant/bm25"           # sparse lexical, runs locally

BATCH_SIZE = 96
PRICE_PER_1M_TOKENS = 0.13

KEYWORD_FIELDS = ["topic", "content_type", "source_file", "organization",
                  "evidence_grades"]
INTEGER_FIELDS = ["year", "heading_level"]


def load_chunks(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def stable_id(chunk_id: str) -> str:
    """Deterministic UUID from the chunk_id -> idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def vector_dim() -> int:
    return TRUNCATE_DIM or EMBED_DIM


def embed_batch(client: OpenAI, texts):
    """OpenAI dense embeddings with simple retries."""
    kwargs = {"model": EMBED_MODEL, "input": texts}
    if TRUNCATE_DIM:
        kwargs["dimensions"] = TRUNCATE_DIM
    for attempt in range(5):
        try:
            resp = client.embeddings.create(**kwargs)
            return [d.embedding for d in resp.data]
        except Exception as e:
            wait = 2 ** attempt
            print(f"    warning: API failure ({e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Embedding retries failed")


def ensure_collection(client: QdrantClient, recreate: bool, collection: str):
    exists = client.collection_exists(collection)
    if exists and recreate:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            # NAMED vectors: dense + sparse on each point.
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_dim(),
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                # Qdrant computes the BM25 IDF on its side from the corpus.
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
        print(f"Collection '{collection}' created (dense {vector_dim()}d cosine + sparse bm25 IDF).")
    else:
        print(f"Collection '{collection}' already exists; will upsert.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="chunks.jsonl file")
    ap.add_argument("--collection", default=COLLECTION,
                    help=f"Target collection name (default '{COLLECTION}'). "
                         "Use another name (e.g. guias_vih_hibrida_ctx) to NOT overwrite "
                         "the current one and be able to A/B or roll back.")
    ap.add_argument("--recreate", action="store_true", help="Delete the collection before uploading")
    ap.add_argument("--dry-run", action="store_true", help="No upload; validate and estimate cost")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.jsonl))
    tot_tokens = sum(c.get("n_tokens", 0) for c in chunks)
    cost = tot_tokens / 1_000_000 * PRICE_PER_1M_TOKENS
    print(f"{len(chunks)} chunks | ~{tot_tokens:,} tokens | "
          f"estimated embedding cost ~ ${cost:.3f}")

    if args.dry_run:
        print("dry-run: nothing is uploaded.")
        return

    missing = [n for n, v in [("OPENAI_API_KEY", OPENAI_API_KEY),
                              ("QDRANT_URL", QDRANT_URL),
                              ("QDRANT_API_KEY", QDRANT_API_KEY)] if not v]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    bm25 = SparseTextEmbedding(BM25_MODEL)        # downloads the model the 1st time
    ensure_collection(qdrant, args.recreate, args.collection)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        # Contextual Retrieval: if the chunk carries "text_for_retrieval" (context + text,
        # produced by chunks/contextualize.py), that enriched version is EMBEDDED and
        # BM25-INDEXED to improve matching; otherwise it falls back to the raw text (compat).
        # The payload keeps the whole chunk, so p["text"] is still the citable LITERAL
        # (evidence.py cites "text", not "text_for_retrieval").
        texts = [c.get("text_for_retrieval", c["text"]) for c in batch]
        dense_vecs = embed_batch(openai_client, texts)
        sparse_vecs = list(bm25.embed(texts))    # local BM25 (documents)
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
        print(f"  uploaded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"Done. Hybrid ingestion completed in Qdrant Cloud (collection '{args.collection}').")


if __name__ == "__main__":
    main()
