"""External merchant knowledge base and metadata enrichment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transaction_classifier.data.preprocess import clean_transaction


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    canonical_name: str
    aliases: list[str]
    mapped_category: str | None
    mapping_confidence: float
    metadata_text: str
    source: str


@dataclass(frozen=True, slots=True)
class KnowledgeMatch:
    entry: KnowledgeEntry
    matched_alias: str
    similarity: float


def normalize_merchant_name(text: str) -> str:
    """Normalize merchant-like text into the format used by the classifier."""
    return clean_transaction(text or "")


def build_enriched_transaction_text(cleaned_text: str, match: KnowledgeMatch) -> str:
    """Create a compact metadata-enriched text for the classifier."""
    parts = [cleaned_text]
    if match.entry.canonical_name and match.entry.canonical_name != cleaned_text:
        parts.append(f"merchant identity: {match.entry.canonical_name}")
    if match.entry.metadata_text:
        parts.append(f"external metadata: {match.entry.metadata_text}")
    return ". ".join(parts)


class MerchantKnowledgeBase:
    """Merchant KB backed by curated/external metadata."""

    def __init__(self, path: Path | None = None):
        self._entries: list[KnowledgeEntry] = []
        self._alias_to_entry: dict[str, KnowledgeEntry] = {}
        self._aliases: list[str] = []
        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._loaded = False

        if path is not None:
            self.load(path)

    def load(self, path: Path) -> None:
        with open(path) as f:
            raw_entries = json.load(f)

        entries: list[KnowledgeEntry] = []
        alias_to_entry: dict[str, KnowledgeEntry] = {}

        for raw in raw_entries:
            canonical_name = normalize_merchant_name(raw.get("canonical_name", ""))
            aliases = [
                normalize_merchant_name(alias)
                for alias in raw.get("aliases", [])
                if normalize_merchant_name(alias)
            ]
            if canonical_name and canonical_name not in aliases:
                aliases.insert(0, canonical_name)
            if not aliases:
                continue

            entry = KnowledgeEntry(
                canonical_name=canonical_name or aliases[0],
                aliases=aliases,
                mapped_category=raw.get("mapped_category"),
                mapping_confidence=float(raw.get("mapping_confidence", 0.0) or 0.0),
                metadata_text=str(raw.get("metadata_text", "") or "").strip(),
                source=str(raw.get("source", "external") or "external"),
            )
            entries.append(entry)

            for alias in aliases:
                alias_to_entry[alias] = entry

        self._entries = entries
        self._alias_to_entry = alias_to_entry
        self._aliases = list(alias_to_entry.keys())

        if self._aliases:
            self._tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=50000,
            )
            self._tfidf_matrix = self._tfidf.fit_transform(self._aliases)

        self._loaded = True

    def search(self, cleaned_text: str, min_similarity: float = 0.58) -> KnowledgeMatch | None:
        if not self._loaded:
            return None

        query = normalize_merchant_name(cleaned_text)
        if len(query) < 3:
            return None

        if query in self._alias_to_entry:
            return KnowledgeMatch(
                entry=self._alias_to_entry[query],
                matched_alias=query,
                similarity=1.0,
            )

        if self._tfidf is None or self._tfidf_matrix is None or not self._aliases:
            return None

        query_vec = self._tfidf.transform([query])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < min_similarity:
            return None

        matched_alias = self._aliases[best_idx]
        return KnowledgeMatch(
            entry=self._alias_to_entry[matched_alias],
            matched_alias=matched_alias,
            similarity=best_score,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def size(self) -> int:
        return len(self._entries)

