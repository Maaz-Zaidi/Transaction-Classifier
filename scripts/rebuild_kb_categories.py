#!/usr/bin/env python
"""Rebuild mapped_category and mapping_confidence for all KB entries using the updated mapper.

This script updates the SQLite knowledge store in-place. It reads raw_category_labels
from each entry, runs the new map_foursquare_labels() function, and writes the results
back. Entries without labels are left unchanged.

Usage:
    python scripts/rebuild_kb_categories.py
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transaction_classifier.config import settings
from transaction_classifier.knowledge.foursquare import map_foursquare_labels


def main():
    db_path = settings.knowledge_store_path
    if not db_path.exists():
        print(f"ERROR: SQLite store not found at {db_path}")
        sys.exit(1)

    print(f"Opening {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    has_labels = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE raw_category_labels IS NOT NULL AND raw_category_labels != '[]'"
    ).fetchone()[0]
    print(f"Total entries: {total:,}")
    print(f"Entries with labels: {has_labels:,}")

    # snapshot before
    before_has_cat = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE mapped_category IS NOT NULL"
    ).fetchone()[0]
    print(f"Before: {before_has_cat:,} entries have a mapped category")

    batch_size = 10000
    updated = 0
    gained_category = 0
    changed_category = 0
    lost_category = 0
    offset = 0

    while True:
        rows = conn.execute(
            """
            SELECT entry_id, raw_category_labels, mapped_category, mapping_confidence
            FROM entries
            WHERE raw_category_labels IS NOT NULL AND raw_category_labels != '[]'
            ORDER BY entry_id
            LIMIT ? OFFSET ?
            """,
            (batch_size, offset),
        ).fetchall()

        if not rows:
            break

        updates = []
        for row in rows:
            labels = json.loads(row["raw_category_labels"])
            if not labels:
                continue

            new_cat, new_conf = map_foursquare_labels(labels)
            old_cat = row["mapped_category"]
            old_conf = row["mapping_confidence"] or 0.0

            if new_cat != old_cat or abs(new_conf - old_conf) > 0.001:
                updates.append((new_cat, new_conf, row["entry_id"]))
                if old_cat is None and new_cat is not None:
                    gained_category += 1
                elif old_cat is not None and new_cat is None:
                    lost_category += 1
                elif old_cat != new_cat:
                    changed_category += 1

        if updates:
            conn.executemany(
                "UPDATE entries SET mapped_category = ?, mapping_confidence = ? WHERE entry_id = ?",
                updates,
            )
            conn.commit()
            updated += len(updates)

        offset += batch_size
        if offset % 100000 == 0:
            print(f"  processed {offset:,} / {has_labels:,} ...")

    # snapshot after
    after_has_cat = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE mapped_category IS NOT NULL"
    ).fetchone()[0]

    print(f"\nDone. Updated {updated:,} entries.")
    print(f"  Gained category (None -> something): {gained_category:,}")
    print(f"  Changed category (X -> Y):           {changed_category:,}")
    print(f"  Lost category (something -> None):   {lost_category:,}")
    print(f"After: {after_has_cat:,} entries have a mapped category (was {before_has_cat:,})")

    conn.close()


if __name__ == "__main__":
    main()
