#!/usr/bin/env python
"""build a merchant kb from public place data.

it uses foursquare canada data when available.
if that is missing, it falls back to the curated merchant list in the repo.

candidate names come from unlabeled transaction strings, not test labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from transaction_classifier.config import settings
from transaction_classifier.data.preprocess import clean_transaction
from transaction_classifier.knowledge.curated_merchants import CURATED_MERCHANTS
from transaction_classifier.knowledge.foursquare import (
    build_metadata_text,
    map_foursquare_labels,
)


def _load_candidate_names(path: Path, column: str | None) -> set[str]:
    df = pd.read_csv(path)
    if column is None:
        for candidate in ("cleaned", "description", "raw_example"):
            if candidate in df.columns:
                column = candidate
                break
    if column is None or column not in df.columns:
        raise ValueError(f"Could not find a usable text column in {path}")

    names = {
        clean_transaction(str(value))
        for value in df[column].fillna("")
        if clean_transaction(str(value))
    }
    return {name for name in names if len(name) >= 3}


def _alias_variants(*values: str) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = clean_transaction(str(value or ""))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        aliases.append(normalized)

        tokens = normalized.split()
        if len(tokens) > 1:
            shortened = " ".join(tokens[:-1]).strip()
            if len(shortened) >= 3 and shortened not in seen:
                seen.add(shortened)
                aliases.append(shortened)
    return aliases


def _merge_entries(*entry_groups: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for group in entry_groups:
        for raw in group:
            canonical_name = clean_transaction(raw.get("canonical_name", ""))
            aliases = [
                clean_transaction(alias)
                for alias in raw.get("aliases", [])
                if clean_transaction(alias)
            ]
            stripped_aliases = [
                clean_transaction(alias)
                for alias in raw.get("stripped_aliases", [])
                if clean_transaction(alias)
            ]
            raw_aliases = [str(alias).strip() for alias in raw.get("raw_aliases", []) if str(alias).strip()]
            if canonical_name and canonical_name not in aliases:
                aliases.insert(0, canonical_name)
            if not aliases:
                continue

            key = canonical_name or aliases[0]
            current = merged.get(key)
            if current is None:
                merged[key] = {
                    "canonical_name": key,
                    "display_name": str(raw.get("display_name", "") or "").strip() or None,
                    "aliases": aliases,
                    "stripped_aliases": stripped_aliases,
                    "raw_aliases": raw_aliases,
                    "mapped_category": raw.get("mapped_category"),
                    "mapping_confidence": float(raw.get("mapping_confidence", 0.0) or 0.0),
                    "metadata_text": str(raw.get("metadata_text", "") or "").strip(),
                    "source": raw.get("source", "external"),
                    "raw_category_labels": list(raw.get("raw_category_labels", []) or []),
                    "locality": raw.get("locality"),
                    "region": raw.get("region"),
                    "country": raw.get("country"),
                }
                continue

            current["aliases"] = sorted(set(current["aliases"]) | set(aliases))
            current["stripped_aliases"] = sorted(
                set(current.get("stripped_aliases", [])) | set(stripped_aliases)
            )
            current["raw_aliases"] = sorted(set(current.get("raw_aliases", [])) | set(raw_aliases))
            if raw.get("display_name") and not current.get("display_name"):
                current["display_name"] = str(raw["display_name"]).strip()
            if (
                float(raw.get("mapping_confidence", 0.0) or 0.0)
                > float(current.get("mapping_confidence", 0.0) or 0.0)
            ):
                current["mapped_category"] = raw.get("mapped_category")
                current["mapping_confidence"] = float(raw.get("mapping_confidence", 0.0) or 0.0)
            if raw.get("metadata_text"):
                if current.get("metadata_text"):
                    current["metadata_text"] = (
                        f"{current['metadata_text']}. {str(raw['metadata_text']).strip()}"
                    )
                else:
                    current["metadata_text"] = str(raw["metadata_text"]).strip()
            current["raw_category_labels"] = list(
                dict.fromkeys(list(current.get("raw_category_labels", [])) + list(raw.get("raw_category_labels", []) or []))
            )
            for field in ("locality", "region", "country"):
                if raw.get(field) and not current.get(field):
                    current[field] = raw.get(field)

    return list(merged.values())


def _build_foursquare_entries(
    candidate_names: set[str] | None,
    *,
    max_rows: int | None = None,
) -> list[dict]:
    hf_fs = HfFileSystem(
        token=(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None)
    )
    arrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(hf_fs))

    release_roots = sorted(
        hf_fs.ls("datasets/foursquare/fsq-os-places/release", detail=False)
    )
    if not release_roots:
        raise RuntimeError("No Foursquare releases found on Hugging Face.")
    latest_release = release_roots[-1]
    parquet_paths = sorted(
        hf_fs.ls(f"{latest_release}/places/parquet", detail=False)
    )
    if not parquet_paths:
        raise RuntimeError(f"No place parquet files found under {latest_release}.")

    candidate_parquet_paths: list[str] = []
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
            candidate_parquet_paths.append(parquet_path)

    print(
        f"Selected {len(candidate_parquet_paths)} of {len(parquet_paths)} parquet files that may contain Canada rows.",
        flush=True,
    )

    aggregates: dict[str, dict[str, Counter[str] | Counter[tuple[str, float]]]] = defaultdict(
        lambda: {
            "display_names": Counter(),
            "category_labels": Counter(),
            "localities": Counter(),
            "regions": Counter(),
            "mapped_categories": Counter(),
        }
    )

    scanned = 0
    matched = 0

    for index, parquet_path in enumerate(candidate_parquet_paths, start=1):
        if max_rows is not None and scanned >= max_rows:
            break

        table = pq.read_table(
            parquet_path,
            filesystem=arrow_fs,
            columns=["name", "country", "locality", "region", "fsq_category_labels"],
            filters=[("country", "=", "CA")],
        )
        if table.num_rows == 0:
            continue

        if max_rows is not None:
            remaining = max_rows - scanned
            if remaining <= 0:
                break
            if table.num_rows > remaining:
                table = table.slice(0, remaining)

        scanned += table.num_rows

        names = table.column("name").to_pylist()
        localities = table.column("locality").to_pylist()
        regions = table.column("region").to_pylist()
        label_lists = table.column("fsq_category_labels").to_pylist()

        for raw_name, locality_value, region_value, fsq_labels in zip(
            names, localities, regions, label_lists
        ):
            normalized = clean_transaction(str(raw_name or ""))
            if candidate_names is not None and normalized not in candidate_names:
                continue

            matched += 1
            bucket = aggregates[normalized]
            display_name = str(raw_name or "").strip()
            if display_name:
                bucket["display_names"][display_name] += 1

            labels = [str(label).strip() for label in fsq_labels or [] if label]
            for label in labels:
                bucket["category_labels"][label] += 1

            locality = str(locality_value or "").strip()
            if locality:
                bucket["localities"][locality] += 1

            region = str(region_value or "").strip()
            if region:
                bucket["regions"][region] += 1

            mapped_category, confidence = map_foursquare_labels(labels)
            if mapped_category is not None and confidence > 0:
                bucket["mapped_categories"][mapped_category] += confidence

        if index % 10 == 0 or index == len(candidate_parquet_paths):
            print(
                f"Scanned {scanned} Canada rows across {index}/{len(candidate_parquet_paths)} parquet files; "
                f"matched {matched} candidate merchants.",
                flush=True,
            )

    entries: list[dict] = []
    for normalized, bucket in aggregates.items():
        labels = [label for label, _ in bucket["category_labels"].most_common(3)]
        locality = bucket["localities"].most_common(1)[0][0] if bucket["localities"] else None
        region = bucket["regions"].most_common(1)[0][0] if bucket["regions"] else None
        mapped_category = None
        mapping_confidence = 0.0
        if bucket["mapped_categories"]:
            mapped_category, score = bucket["mapped_categories"].most_common(1)[0]
            total_score = float(sum(bucket["mapped_categories"].values()))
            mapping_confidence = score / total_score if total_score else 0.0

        entries.append(
            {
                "canonical_name": normalized,
                "display_name": bucket["display_names"].most_common(1)[0][0]
                if bucket["display_names"]
                else None,
                "aliases": _alias_variants(
                    normalized,
                    *[name for name, _ in bucket["display_names"].most_common(5)],
                ),
                "stripped_aliases": _alias_variants(
                    *[
                        " ".join(clean_transaction(name).split()[:-1])
                        for name, _ in bucket["display_names"].most_common(5)
                    ]
                ),
                "raw_aliases": [name for name, _ in bucket["display_names"].most_common(5)],
                "mapped_category": mapped_category,
                "mapping_confidence": mapping_confidence,
                "metadata_text": build_metadata_text(labels, locality, region, "CA"),
                "source": "foursquare",
                "raw_category_labels": labels,
                "locality": locality,
                "region": region,
                "country": "CA",
            }
        )

    print(
        f"Foursquare scan matched {matched} candidate merchants from {scanned} scanned rows.",
        flush=True,
    )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("data/real/full_descriptions.csv"),
    )
    parser.add_argument("--candidate-column")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.knowledge_base_path,
    )
    parser.add_argument("--full-canada", action="store_true")
    parser.add_argument("--skip-foursquare", action="store_true")
    parser.add_argument("--allow-curated-only", action="store_true")
    parser.add_argument("--max-foursquare-rows", type=int)
    args = parser.parse_args()

    candidate_names = None
    if not args.full_canada:
        candidate_names = _load_candidate_names(args.candidate_csv, args.candidate_column)
        print(f"Loaded {len(candidate_names)} unlabeled candidate merchant names.", flush=True)
    else:
        print("Building full Canada merchant document set (no candidate prefilter).", flush=True)

    foursquare_entries: list[dict] = []
    if not args.skip_foursquare:
        try:
            foursquare_entries = _build_foursquare_entries(
                candidate_names,
                max_rows=args.max_foursquare_rows,
            )
        except Exception as exc:
            if not args.allow_curated_only:
                raise
            print(f"WARNING: Foursquare build skipped: {exc}", flush=True)

    merged_entries = _merge_entries(CURATED_MERCHANTS, foursquare_entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged_entries, f, indent=2, sort_keys=True)

    print(f"Wrote {len(merged_entries)} KB entries to {args.output}", flush=True)


if __name__ == "__main__":
    main()
