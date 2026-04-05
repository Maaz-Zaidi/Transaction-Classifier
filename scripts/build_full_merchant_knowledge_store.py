#!/usr/bin/env python
"""build the full merchant store from the canada foursquare dump."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pyarrow.fs as pafs
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

try:
    import chromadb
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "chromadb is not installed. Install retrieval dependencies before building the full store."
    ) from exc

from transaction_classifier.config import settings
from transaction_classifier.data.preprocess import clean_transaction
from transaction_classifier.knowledge.curated_merchants import CURATED_MERCHANTS
from transaction_classifier.knowledge.foursquare import build_metadata_text, map_foursquare_labels
from transaction_classifier.knowledge.merchant_kb import KnowledgeEntry, build_retrieval_document
from transaction_classifier.knowledge.retrieval import MerchantDenseEmbedder


@dataclass(slots=True)
class PendingEntry:
    entry_id: str
    canonical_name: str
    display_name: str | None
    mapped_category: str | None
    mapping_confidence: float
    metadata_text: str
    source: str
    raw_category_labels: list[str]
    locality: str | None
    region: str | None
    country: str | None
    document: str
    aliases: list[tuple[str, int]]


def _stable_entry_id(source: str, canonical_name: str) -> str:
    digest = hashlib.sha1(f"{source}:{canonical_name}".encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def _canonical_aliases(normalized: str, display_name: str | None) -> list[tuple[str, int]]:
    seen: set[str] = set()
    aliases: list[tuple[str, int]] = []

    def add(value: str | None, priority: int) -> None:
        alias = clean_transaction(str(value or ""))
        if len(alias) < 3 or alias in seen:
            return
        seen.add(alias)
        aliases.append((alias, priority))

    add(normalized, 100)
    add(display_name, 90)

    tokens = normalized.split()
    if len(tokens) > 1:
        last = tokens[-1]
        if last.isdigit() or len(last) <= 2:
            add(" ".join(tokens[:-1]), 75)

    return aliases


def _entry_from_curated(raw: dict) -> PendingEntry | None:
    canonical_name = clean_transaction(raw.get("canonical_name", ""))
    if len(canonical_name) < 3:
        return None

    aliases: list[tuple[str, int]] = []
    seen: set[str] = set()
    for priority, group in ((100, [canonical_name]), (95, raw.get("aliases", [])), (80, raw.get("stripped_aliases", []))):
        for value in group:
            alias = clean_transaction(str(value or ""))
            if len(alias) < 3 or alias in seen:
                continue
            seen.add(alias)
            aliases.append((alias, priority))

    entry = KnowledgeEntry(
        entry_id=_stable_entry_id(str(raw.get("source", "curated_public")), canonical_name),
        canonical_name=canonical_name,
        aliases=[alias for alias, _ in aliases],
        mapped_category=raw.get("mapped_category"),
        mapping_confidence=float(raw.get("mapping_confidence", 0.0) or 0.0),
        metadata_text=str(raw.get("metadata_text", "") or "").strip(),
        source=str(raw.get("source", "curated_public") or "curated_public"),
        stripped_aliases=[
            clean_transaction(str(alias))
            for alias in raw.get("stripped_aliases", [])
            if clean_transaction(str(alias))
        ],
        raw_aliases=[str(alias).strip() for alias in raw.get("raw_aliases", []) if str(alias).strip()],
        raw_category_labels=[
            str(label).strip()
            for label in raw.get("raw_category_labels", [])
            if str(label).strip()
        ],
        locality=str(raw.get("locality")).strip()
        if raw.get("locality") is not None and str(raw.get("locality")).strip()
        else None,
        region=str(raw.get("region")).strip()
        if raw.get("region") is not None and str(raw.get("region")).strip()
        else None,
        country=str(raw.get("country")).strip()
        if raw.get("country") is not None and str(raw.get("country")).strip()
        else None,
        display_name=str(raw.get("display_name")).strip()
        if raw.get("display_name") is not None and str(raw.get("display_name")).strip()
        else None,
    )

    return PendingEntry(
        entry_id=entry.entry_id,
        canonical_name=entry.canonical_name,
        display_name=entry.display_name,
        mapped_category=entry.mapped_category,
        mapping_confidence=entry.mapping_confidence,
        metadata_text=entry.metadata_text,
        source=entry.source,
        raw_category_labels=entry.raw_category_labels,
        locality=entry.locality,
        region=entry.region,
        country=entry.country,
        document=build_retrieval_document(entry),
        aliases=aliases,
    )


def _entry_from_foursquare(
    raw_name: str,
    locality: str | None,
    region: str | None,
    category_labels: list[str],
) -> PendingEntry | None:
    canonical_name = clean_transaction(raw_name)
    if len(canonical_name) < 3:
        return None

    display_name = str(raw_name or "").strip() or None
    mapped_category, confidence = map_foursquare_labels(category_labels)
    metadata_text = build_metadata_text(category_labels, locality, region, "CA")
    aliases = _canonical_aliases(canonical_name, display_name)
    if not aliases:
        return None

    entry = KnowledgeEntry(
        entry_id=_stable_entry_id("foursquare", canonical_name),
        canonical_name=canonical_name,
        aliases=[alias for alias, _ in aliases],
        mapped_category=mapped_category,
        mapping_confidence=float(confidence),
        metadata_text=metadata_text,
        source="foursquare",
        raw_category_labels=category_labels[:8],
        locality=locality,
        region=region,
        country="CA",
        display_name=display_name,
    )

    return PendingEntry(
        entry_id=entry.entry_id,
        canonical_name=entry.canonical_name,
        display_name=entry.display_name,
        mapped_category=entry.mapped_category,
        mapping_confidence=entry.mapping_confidence,
        metadata_text=entry.metadata_text,
        source=entry.source,
        raw_category_labels=entry.raw_category_labels,
        locality=entry.locality,
        region=entry.region,
        country=entry.country,
        document=build_retrieval_document(entry),
        aliases=aliases,
    )


def _init_store(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;

        CREATE TABLE IF NOT EXISTS entries (
            entry_id TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL UNIQUE,
            display_name TEXT,
            mapped_category TEXT,
            mapping_confidence REAL NOT NULL,
            metadata_text TEXT NOT NULL,
            source TEXT NOT NULL,
            raw_category_labels TEXT NOT NULL,
            locality TEXT,
            region TEXT,
            country TEXT,
            document TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS aliases (
            alias TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (alias, entry_id)
        );

        CREATE INDEX IF NOT EXISTS aliases_alias_idx
        ON aliases(alias, priority DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS merchant_fts
        USING fts5(entry_id UNINDEXED, document, tokenize = 'unicode61');
        """
    )
    conn.commit()


def _candidate_parquet_paths(hf_fs: HfFileSystem, arrow_fs: pafs.FileSystem) -> list[str]:
    release_roots = sorted(hf_fs.ls("datasets/foursquare/fsq-os-places/release", detail=False))
    if not release_roots:
        raise RuntimeError("No Foursquare releases found on Hugging Face.")

    latest_release = release_roots[-1]
    parquet_paths = sorted(hf_fs.ls(f"{latest_release}/places/parquet", detail=False))
    if not parquet_paths:
        raise RuntimeError(f"No place parquet files found under {latest_release}.")

    candidate_paths: list[str] = []
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path, filesystem=arrow_fs)
        should_scan = False
        for row_group_index in range(parquet_file.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                if column.path_in_schema != "country":
                    continue
                stats = column.statistics
                if stats is None or stats.min is None or stats.max is None:
                    should_scan = True
                    break
                if stats.min <= "CA" <= stats.max:
                    should_scan = True
                    break
            if should_scan:
                break
        if should_scan:
            candidate_paths.append(parquet_path)
    return candidate_paths


def _flush_pending(
    pending: list[PendingEntry],
    *,
    conn: sqlite3.Connection,
    collection,
    embedder: MerchantDenseEmbedder | None,
    embed_batch_size: int,
) -> int:
    if not pending:
        return 0

    entry_rows = [
        (
            item.entry_id,
            item.canonical_name,
            item.display_name,
            item.mapped_category,
            item.mapping_confidence,
            item.metadata_text,
            item.source,
            json.dumps(item.raw_category_labels, ensure_ascii=True),
            item.locality,
            item.region,
            item.country,
            item.document,
        )
        for item in pending
    ]
    alias_rows = [
        (alias, item.entry_id, priority)
        for item in pending
        for alias, priority in item.aliases
    ]
    fts_rows = [(item.entry_id, item.document) for item in pending]

    conn.executemany(
        """
        INSERT OR IGNORE INTO entries (
            entry_id,
            canonical_name,
            display_name,
            mapped_category,
            mapping_confidence,
            metadata_text,
            source,
            raw_category_labels,
            locality,
            region,
            country,
            document
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        entry_rows,
    )
    conn.executemany(
        """
        INSERT INTO aliases(alias, entry_id, priority)
        VALUES (?, ?, ?)
        ON CONFLICT(alias, entry_id) DO UPDATE SET priority = MAX(priority, excluded.priority)
        """,
        alias_rows,
    )
    conn.executemany(
        "INSERT INTO merchant_fts(entry_id, document) VALUES (?, ?)",
        fts_rows,
    )
    conn.commit()

    if collection is not None and embedder is not None:
        documents = [item.document for item in pending]
        embeddings = embedder.encode(documents, batch_size=embed_batch_size)
        collection.upsert(
            ids=[item.entry_id for item in pending],
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "canonical_name": item.canonical_name,
                    "mapped_category": item.mapped_category or "",
                    "source": item.source,
                }
                for item in pending
            ],
        )

    inserted = len(pending)
    pending.clear()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-db", type=Path, default=settings.knowledge_store_path)
    parser.add_argument("--chroma-dir", type=Path, default=settings.knowledge_chroma_dir)
    parser.add_argument("--collection-name", type=str, default=settings.knowledge_collection_name)
    parser.add_argument("--dense-model", type=str, default=settings.knowledge_dense_model_name)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--flush-size", type=int, default=1024)
    parser.add_argument("--max-foursquare-rows", type=int)
    parser.add_argument("--skip-curated", action="store_true")
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset and args.output_db.exists():
        args.output_db.unlink()
    if args.reset and args.chroma_dir.exists():
        shutil.rmtree(args.chroma_dir)

    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    args.chroma_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.output_db)
    _init_store(conn)

    collection = None
    embedder = None
    if not args.skip_chroma:
        embedder = MerchantDenseEmbedder(args.dense_model)
        client = chromadb.PersistentClient(path=str(args.chroma_dir))
        try:
            client.delete_collection(args.collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=args.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    seen_canonicals: set[str] = set()
    pending: list[PendingEntry] = []
    inserted_entries = 0
    duplicate_rows = 0

    if not args.skip_curated:
        for raw in CURATED_MERCHANTS:
            entry = _entry_from_curated(raw)
            if entry is None or entry.canonical_name in seen_canonicals:
                continue
            seen_canonicals.add(entry.canonical_name)
            pending.append(entry)
        inserted_entries += _flush_pending(
            pending,
            conn=conn,
            collection=collection,
            embedder=embedder,
            embed_batch_size=args.embed_batch_size,
        )
        print(f"Inserted curated seed entries: {inserted_entries}", flush=True)

    hf_fs = HfFileSystem(
        token=(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None)
    )
    arrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(hf_fs))
    parquet_paths = _candidate_parquet_paths(hf_fs, arrow_fs)
    print(
        f"Streaming full Canada merchant corpus from {len(parquet_paths)} parquet files.",
        flush=True,
    )

    scanned_rows = 0
    for index, parquet_path in enumerate(parquet_paths, start=1):
        if args.max_foursquare_rows is not None and scanned_rows >= args.max_foursquare_rows:
            break

        table = pq.read_table(
            parquet_path,
            filesystem=arrow_fs,
            columns=["name", "country", "locality", "region", "fsq_category_labels"],
            filters=[("country", "=", "CA")],
        )
        if table.num_rows == 0:
            continue

        if args.max_foursquare_rows is not None:
            remaining = args.max_foursquare_rows - scanned_rows
            if remaining <= 0:
                break
            if table.num_rows > remaining:
                table = table.slice(0, remaining)

        scanned_rows += table.num_rows
        names = table.column("name").to_pylist()
        localities = table.column("locality").to_pylist()
        regions = table.column("region").to_pylist()
        labels = table.column("fsq_category_labels").to_pylist()

        for raw_name, locality_value, region_value, fsq_labels in zip(
            names, localities, regions, labels
        ):
            locality = str(locality_value or "").strip() or None
            region = str(region_value or "").strip() or None
            category_labels = [str(label).strip() for label in fsq_labels or [] if str(label).strip()]
            entry = _entry_from_foursquare(str(raw_name or ""), locality, region, category_labels)
            if entry is None:
                continue
            if entry.canonical_name in seen_canonicals:
                duplicate_rows += 1
                continue

            seen_canonicals.add(entry.canonical_name)
            pending.append(entry)
            if len(pending) >= args.flush_size:
                inserted_entries += _flush_pending(
                    pending,
                    conn=conn,
                    collection=collection,
                    embedder=embedder,
                    embed_batch_size=args.embed_batch_size,
                )

        if index % 5 == 0 or index == len(parquet_paths):
            print(
                f"Scanned {scanned_rows} Canada rows across {index}/{len(parquet_paths)} parquet files; "
                f"unique entries={len(seen_canonicals)} duplicates_skipped={duplicate_rows}",
                flush=True,
            )

    inserted_entries += _flush_pending(
        pending,
        conn=conn,
        collection=collection,
        embedder=embedder,
        embed_batch_size=args.embed_batch_size,
    )
    conn.close()

    print(
        f"Built full merchant store with {inserted_entries} entries at {args.output_db}",
        flush=True,
    )
    if collection is not None:
        print(
            f"Built Chroma collection '{args.collection_name}' with {collection.count()} documents at "
            f"{args.chroma_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
