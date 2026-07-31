#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_qdrant.py
===================
Reads chunks.jsonl (from chunk_guidelines.py), computes embeddings with OpenAI's most
advanced model (text-embedding-3-large, 3072 dim) and uploads them to a
Qdrant Cloud collection. Metadata goes as the 'payload' so it can be filtered on.

Requirements:
    pip install qdrant-client openai

Credentials via environment variables (do not put them in the code):
    export OPENAI_API_KEY="sk-..."
    export QDRANT_URL="https://xxxxx.cloud.qdrant.io:6333"
    export QDRANT_API_KEY="..."          # API key of the Qdrant Cloud cluster

Usage:
    python -m ingestion.upload_to_qdrant data/chunks/chunks.jsonl
    python -m ingestion.upload_to_qdrant data/chunks/chunks.jsonl --recreate  # deletes and recreates the collection
    python -m ingestion.upload_to_qdrant data/chunks/chunks.jsonl --dry-run   # no upload; only validate and estimate cost
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
# CONFIGURATION
# ---------------------------------------------------------------------------

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

QDRANT_URL     = os.environ.get("QDRANT_URL")        # https://....cloud.qdrant.io:6333
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Same knob as the app (rag.OPENAI_BASE_URL): ONE .env value repoints every OpenAI call
# in the project, ingestion included. Read rather than imported so this script stays
# standalone (it must not drag in the retrieval stack).
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or None

COLLECTION = "guias_vih"

# OpenAI's most advanced embedding model.
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM   = 3072          # native dimension of text-embedding-3-large
# Matryoshka: you could reduce to 1024 or 256 by passing dimensions=... to the API,
# saving storage at the cost of some quality. By default, maximum quality (3072).
# If you change it, adjust the search accordingly.
TRUNCATE_DIM = None         # e.g. 1024 to reduce; None = native (3072)

# OpenAI accepts many inputs per call; 96 is a prudent batch.
BATCH_SIZE = 96
PRICE_PER_1M_TOKENS = 0.13  # indicative USD for text-embedding-3-large

# Payload indexes to filter on in the searches.
KEYWORD_FIELDS = ["topic", "content_type", "source_file", "organization",
                  "evidence_grades"]
INTEGER_FIELDS = ["year", "heading_level"]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
def load_chunks(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def stable_id(chunk_id: str) -> str:
    """Deterministic UUID from the chunk_id -> idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def vector_dim() -> int:
    return TRUNCATE_DIM or EMBED_DIM


def embed_batch(client: OpenAI, texts):
    """Call the embeddings API with simple retries."""
    kwargs = {"model": EMBED_MODEL, "input": texts}
    if TRUNCATE_DIM:
        kwargs["dimensions"] = TRUNCATE_DIM
    for attempt in range(5):
        try:
            resp = client.embeddings.create(**kwargs)
            return [d.embedding for d in resp.data]
        except Exception as e:                      # rate limit / network
            wait = 2 ** attempt
            print(f"    warning: API failure ({e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Embedding retries failed")


# ---------------------------------------------------------------------------
# COLLECTION
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
        print(f"Collection '{COLLECTION}' created (dim={vector_dim()}, cosine).")
    else:
        print(f"Collection '{COLLECTION}' already exists; will upsert.")


# ---------------------------------------------------------------------------
# INGESTION
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="chunks.jsonl file")
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

    # Credential validation
    missing = [n for n, v in [("OPENAI_API_KEY", OPENAI_API_KEY),
                              ("QDRANT_URL", QDRANT_URL),
                              ("QDRANT_API_KEY", QDRANT_API_KEY)] if not v]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
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
        print(f"  uploaded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print("Done. Ingestion completed in Qdrant Cloud.")


# ---------------------------------------------------------------------------
# SEARCH EXAMPLE (with metadata filter)
# ---------------------------------------------------------------------------
def example_search():
    """Semantic search restricted to grade-A recommendations."""
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    query = "cuando iniciar el tratamiento antirretroviral en un paciente con tuberculosis?"
    qvec = embed_batch(openai_client, [query])[0]

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
