#!/usr/bin/env python
"""build the local chroma index from the merchant kb json."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import chromadb
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "chromadb is not installed. Install retrieval dependencies before building the index."
    ) from exc

from transaction_classifier.config import settings
from transaction_classifier.knowledge.merchant_kb import (
    MerchantKnowledgeBase,
    build_retrieval_document,
)
from transaction_classifier.knowledge.retrieval import MerchantDenseEmbedder


def _batched(seq: list, batch_size: int):
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=settings.knowledge_base_path)
    parser.add_argument("--chroma-dir", type=Path, default=settings.knowledge_chroma_dir)
    parser.add_argument(
        "--collection-name",
        type=str,
        default=settings.knowledge_collection_name,
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default=settings.knowledge_dense_model_name,
    )
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--upsert-batch-size", type=int, default=256)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Knowledge base file not found: {args.input}")

    kb = MerchantKnowledgeBase(
        dense_embedder=MerchantDenseEmbedder(args.dense_model),
        reranker=None,
    )
    kb.load(args.input)
    if not kb.entries:
        raise SystemExit(f"No KB entries found in {args.input}")

    embedder = MerchantDenseEmbedder(args.dense_model)
    if not embedder.is_available:
        raise SystemExit(f"Dense embedder {args.dense_model} is unavailable.")

    documents = [build_retrieval_document(entry) for entry in kb.entries]
    ids = [entry.entry_id for entry in kb.entries]
    metadatas = [
        {
            "canonical_name": entry.canonical_name,
            "mapped_category": entry.mapped_category or "",
            "source": entry.source,
        }
        for entry in kb.entries
    ]

    print(
        f"Encoding {len(documents)} merchant documents with {args.dense_model}...",
        flush=True,
    )
    embeddings = embedder.encode(documents, batch_size=args.embed_batch_size)

    if args.reset and args.chroma_dir.exists():
        shutil.rmtree(args.chroma_dir)
    args.chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(args.chroma_dir))
    try:
        client.delete_collection(args.collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=args.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for batch_ids, batch_documents, batch_embeddings, batch_metadatas in zip(
        _batched(ids, args.upsert_batch_size),
        _batched(documents, args.upsert_batch_size),
        _batched(embeddings, args.upsert_batch_size),
        _batched(metadatas, args.upsert_batch_size),
    ):
        collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings.tolist(),
            metadatas=batch_metadatas,
        )

    print(
        f"Built Chroma collection '{args.collection_name}' with {collection.count()} documents at "
        f"{args.chroma_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
