"""Tests for the external merchant knowledge base."""

import json

from transaction_classifier.knowledge.merchant_kb import (
    MerchantKnowledgeBase,
    build_enriched_transaction_text,
)


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

    kb = MerchantKnowledgeBase(kb_path)
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

    kb = MerchantKnowledgeBase(kb_path)
    match = kb.search("HTSP- UNIV OTTAWA")
    enriched = build_enriched_transaction_text("HTSP- UNIV OTTAWA", match)

    assert "external metadata" in enriched
    assert "tuition" in enriched
