"""tests for the merchant knowledge base."""

import json
import sqlite3

from transaction_classifier.knowledge.merchant_kb import (
    MerchantKnowledgeBase,
    build_enriched_transaction_text,
)


class DummyDenseEmbedder:
    is_available = False


class DummyReranker:
    is_available = True

    def __init__(self, scores):
        self._scores = list(scores)

    def score(self, query, documents):
        return self._scores[: len(documents)]


def _write_sqlite_kb(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE entries (
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
        CREATE TABLE aliases (
            alias TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (alias, entry_id)
        );
        CREATE VIRTUAL TABLE merchant_fts
        USING fts5(entry_id UNINDEXED, document, tokenize = 'unicode61');
        """
    )
    conn.execute(
        """
        INSERT INTO entries (
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
        (
            "merchant:goodlife",
            "GOODLIFE FITNESS",
            "GoodLife Fitness",
            "Entertainment & Recreation",
            0.91,
            "place types: Gym / Fitness Center. location: Ottawa, ON, CA",
            "foursquare",
            json.dumps(["Gym / Fitness Center"]),
            "Ottawa",
            "ON",
            "CA",
            "display name: GoodLife Fitness. canonical name: GOODLIFE FITNESS. "
            "external metadata: gym and fitness club. place types: Gym / Fitness Center. "
            "location: Ottawa, ON, CA",
        ),
    )
    conn.execute(
        "INSERT INTO aliases(alias, entry_id, priority) VALUES (?, ?, ?)",
        ("GOODLIFE FITNESS", "merchant:goodlife", 100),
    )
    conn.execute(
        "INSERT INTO merchant_fts(entry_id, document) VALUES (?, ?)",
        (
            "merchant:goodlife",
            "display name: GoodLife Fitness. canonical name: GOODLIFE FITNESS. "
            "external metadata: gym and fitness club. place types: Gym / Fitness Center. "
            "location: Ottawa, ON, CA",
        ),
    )
    conn.commit()
    conn.close()


def test_search_exact_alias(tmp_path):
    kb_path = tmp_path / "merchant_kb.json"
    with open(kb_path, "w") as f:
        json.dump(
            [
                {
                    "canonical_name": "REAL FRUIT BUBBLE TEA",
                    "aliases": ["REAL FRUIT BUBB"],
                    "mapped_category": "Food & Dining",
                    "mapping_confidence": 0.98,
                    "metadata_text": "bubble tea shop and cafe",
                    "source": "curated_public",
                }
            ],
            f,
        )

    kb = MerchantKnowledgeBase(
        kb_path,
        dense_embedder=DummyDenseEmbedder(),
        reranker=DummyReranker([0.99]),
    )
    match = kb.search("REAL FRUIT BUBB")

    assert match is not None
    assert match.entry.mapped_category == "Food & Dining"
    assert match.similarity == 1.0


def test_build_enriched_text_includes_metadata(tmp_path):
    kb_path = tmp_path / "merchant_kb.json"
    with open(kb_path, "w") as f:
        json.dump(
            [
                {
                    "canonical_name": "UNIVERSITY OF OTTAWA",
                    "aliases": ["HTSP- UNIV OTTAWA"],
                    "mapped_category": None,
                    "mapping_confidence": 0.0,
                    "metadata_text": "public university with tuition and student fees",
                    "source": "curated_public",
                }
            ],
            f,
        )

    kb = MerchantKnowledgeBase(
        kb_path,
        dense_embedder=DummyDenseEmbedder(),
        reranker=DummyReranker([0.99]),
    )
    match = kb.search("HTSP- UNIV OTTAWA")
    enriched = build_enriched_transaction_text("HTSP- UNIV OTTAWA", match)

    assert "external metadata" in enriched
    assert "tuition" in enriched


def test_search_uses_lexical_plus_rerank_without_chroma(tmp_path):
    kb_path = tmp_path / "merchant_kb.json"
    with open(kb_path, "w") as f:
        json.dump(
            [
                {
                    "canonical_name": "GOODLIFE FITNESS",
                    "display_name": "GoodLife Fitness",
                    "aliases": ["GOODLIFE FITNESS"],
                    "mapped_category": "Healthcare & Medical",
                    "mapping_confidence": 0.91,
                    "metadata_text": "gym and fitness club",
                    "source": "foursquare",
                    "raw_category_labels": ["Gym / Fitness Center"],
                }
            ],
            f,
        )

    kb = MerchantKnowledgeBase(
        kb_path,
        dense_embedder=DummyDenseEmbedder(),
        reranker=DummyReranker([0.93]),
    )
    match = kb.search("GOODLIFE CLUBS", min_similarity=0.58)

    assert match is not None
    assert match.entry.canonical_name == "GOODLIFE FITNESS"
    assert match.strategy == "rerank"
    assert match.similarity == 0.93


def test_search_sqlite_store_uses_fts_and_rerank(tmp_path):
    kb_path = tmp_path / "merchant_kb.sqlite3"
    _write_sqlite_kb(kb_path)

    kb = MerchantKnowledgeBase(
        kb_path,
        dense_embedder=DummyDenseEmbedder(),
        reranker=DummyReranker([0.91]),
    )
    match = kb.search("GOODLIFE CLUBS", min_similarity=0.58)

    assert match is not None
    assert match.entry.canonical_name == "GOODLIFE FITNESS"
    assert match.strategy == "rerank"
    assert match.similarity == 0.91
