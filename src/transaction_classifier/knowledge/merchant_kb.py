"""merchant knowledge base with exact, lexical, dense, and rerank lookup."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transaction_classifier.config import settings
from transaction_classifier.data.preprocess import clean_transaction
from transaction_classifier.knowledge.retrieval import MerchantDenseEmbedder, MerchantReranker

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency at import time
    chromadb = None


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    entry_id: str
    canonical_name: str
    aliases: list[str]
    mapped_category: str | None
    mapping_confidence: float
    metadata_text: str
    source: str
    stripped_aliases: list[str] = field(default_factory=list)
    raw_aliases: list[str] = field(default_factory=list)
    raw_category_labels: list[str] = field(default_factory=list)
    locality: str | None = None
    region: str | None = None
    country: str | None = None
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeMatch:
    entry: KnowledgeEntry
    matched_alias: str
    similarity: float
    strategy: str = "exact"
    dense_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


def normalize_merchant_name(text: str) -> str:
    """normalize merchant text the same way the classifier does."""
    return clean_transaction(text or "")


def build_retrieval_document(entry: KnowledgeEntry) -> str:
    """build the text blob used for retrieval and reranking."""
    parts: list[str] = []

    if entry.display_name:
        parts.append(f"display name: {entry.display_name}")
    if entry.canonical_name:
        parts.append(f"canonical name: {entry.canonical_name}")
    if entry.aliases:
        parts.append("aliases: " + " ; ".join(entry.aliases[:12]))
    if entry.stripped_aliases:
        parts.append("alternate forms: " + " ; ".join(entry.stripped_aliases[:12]))
    if entry.metadata_text:
        parts.append("external metadata: " + entry.metadata_text)
    if entry.raw_category_labels:
        parts.append("place types: " + " ; ".join(entry.raw_category_labels[:6]))

    location_bits = [bit for bit in (entry.locality, entry.region, entry.country) if bit]
    if location_bits:
        parts.append("location: " + ", ".join(location_bits))

    return ". ".join(part for part in parts if part)


def build_enriched_transaction_text(cleaned_text: str, match: KnowledgeMatch) -> str:
    """build a short text with kb metadata for the classifier."""
    parts = [cleaned_text]
    display_name = match.entry.display_name or match.entry.canonical_name
    if display_name and display_name != cleaned_text:
        parts.append(f"merchant identity: {display_name}")
    if match.entry.metadata_text:
        parts.append(f"external metadata: {match.entry.metadata_text}")
    if match.entry.raw_category_labels:
        parts.append("place types: " + "; ".join(match.entry.raw_category_labels[:3]))
    return ". ".join(parts)


class MerchantKnowledgeBase:
    """merchant kb with exact, lexical, dense, and rerank search."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        chroma_dir: Path | None = None,
        collection_name: str | None = None,
        dense_embedder: MerchantDenseEmbedder | None = None,
        reranker: MerchantReranker | None = None,
    ):
        self._entries: list[KnowledgeEntry] = []
        self._entry_by_id: dict[str, KnowledgeEntry] = {}
        self._alias_to_entry_ids: dict[str, list[str]] = {}
        self._documents: dict[str, str] = {}
        self._doc_ids: list[str] = []
        self._lexical: TfidfVectorizer | None = None
        self._lexical_matrix = None
        self._chroma_dir = Path(chroma_dir or settings.knowledge_chroma_dir)
        self._collection_name = collection_name or settings.knowledge_collection_name
        self._chroma_client = None
        self._collection = None
        self._dense_embedder = dense_embedder or MerchantDenseEmbedder(
            settings.knowledge_dense_model_name
        )
        self._reranker = reranker or MerchantReranker(settings.knowledge_reranker_model_name)
        self._loaded = False
        self._sqlite_conn: sqlite3.Connection | None = None
        self._sqlite_path: Path | None = None
        self._sqlite_fts_ready = False
        self._entry_count = 0

        if path is not None:
            self.load(path)

    def load(self, path: Path) -> None:
        resolved = Path(path)
        if resolved.suffix in {".sqlite3", ".sqlite", ".db"}:
            self._load_sqlite(resolved)
            return
        self._load_json(resolved)

    def _reset_state(self) -> None:
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
        self._entries = []
        self._entry_by_id = {}
        self._alias_to_entry_ids = {}
        self._documents = {}
        self._doc_ids = []
        self._lexical = None
        self._lexical_matrix = None
        self._sqlite_conn = None
        self._sqlite_path = None
        self._sqlite_fts_ready = False
        self._entry_count = 0
        self._chroma_client = None
        self._collection = None
        self._loaded = False

    def _load_json(self, path: Path) -> None:
        self._reset_state()
        with open(path) as f:
            raw_entries = json.load(f)

        entries: list[KnowledgeEntry] = []
        entry_by_id: dict[str, KnowledgeEntry] = {}
        alias_to_entry_ids: dict[str, list[str]] = {}
        documents: dict[str, str] = {}

        for index, raw in enumerate(raw_entries):
            canonical_name = normalize_merchant_name(raw.get("canonical_name", ""))
            aliases = self._normalize_aliases(raw.get("aliases", []))
            stripped_aliases = self._normalize_aliases(raw.get("stripped_aliases", []))
            raw_aliases = [
                str(alias).strip() for alias in raw.get("raw_aliases", []) if str(alias).strip()
            ]

            if canonical_name and canonical_name not in aliases:
                aliases.insert(0, canonical_name)
            if not aliases:
                continue

            entry_id = str(raw.get("entry_id") or raw.get("id") or f"merchant:{index}")
            entry = KnowledgeEntry(
                entry_id=entry_id,
                canonical_name=canonical_name or aliases[0],
                aliases=aliases,
                mapped_category=raw.get("mapped_category"),
                mapping_confidence=float(raw.get("mapping_confidence", 0.0) or 0.0),
                metadata_text=str(raw.get("metadata_text", "") or "").strip(),
                source=str(raw.get("source", "external") or "external"),
                stripped_aliases=stripped_aliases,
                raw_aliases=raw_aliases,
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
            entries.append(entry)
            entry_by_id[entry_id] = entry
            documents[entry_id] = build_retrieval_document(entry)

            for alias in aliases + stripped_aliases:
                alias_to_entry_ids.setdefault(alias, [])
                if entry_id not in alias_to_entry_ids[alias]:
                    alias_to_entry_ids[alias].append(entry_id)

        self._entries = entries
        self._entry_by_id = entry_by_id
        self._alias_to_entry_ids = alias_to_entry_ids
        self._documents = documents
        self._doc_ids = list(documents.keys())
        self._entry_count = len(entries)

        if self._doc_ids:
            lexical_texts = [documents[entry_id] for entry_id in self._doc_ids]
            self._lexical = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                lowercase=True,
                min_df=1,
            )
            self._lexical_matrix = self._lexical.fit_transform(lexical_texts)

        self._load_chroma_collection()
        self._loaded = True

    def _load_sqlite(self, path: Path) -> None:
        self._reset_state()
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        self._sqlite_conn = conn
        self._sqlite_path = path
        self._entry_count = int(conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
        self._sqlite_fts_ready = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'merchant_fts'"
            ).fetchone()
        )

        self._load_chroma_collection()
        self._loaded = True

    def _load_chroma_collection(self) -> None:
        self._chroma_client = None
        self._collection = None

        if chromadb is None or not self._chroma_dir.exists():
            return

        try:
            self._chroma_client = chromadb.PersistentClient(path=str(self._chroma_dir))
            self._collection = self._chroma_client.get_collection(self._collection_name)
        except Exception:
            self._chroma_client = None
            self._collection = None

    @staticmethod
    def _normalize_aliases(raw_aliases: list[str]) -> list[str]:
        aliases: list[str] = []
        for alias in raw_aliases or []:
            normalized = normalize_merchant_name(str(alias))
            if normalized and normalized not in aliases:
                aliases.append(normalized)
        return aliases

    @staticmethod
    def _fts_query(query: str) -> str | None:
        tokens = [token.lower() for token in re.findall(r"[A-Z0-9]+", query) if len(token) >= 2]
        if not tokens:
            return None
        return " OR ".join(f'"{token}"' for token in tokens[:8])

    @staticmethod
    def _empty_bucket() -> dict[str, float | int | None]:
        return {
            "dense_score": None,
            "lexical_score": None,
            "dense_rank": None,
            "lexical_rank": None,
            "fused_score": 0.0,
        }

    def _lookup_exact_candidate_ids(self, alias: str) -> list[str]:
        if self._sqlite_conn is not None:
            rows = self._sqlite_conn.execute(
                "SELECT entry_id FROM aliases WHERE alias = ? ORDER BY priority DESC, entry_id LIMIT 8",
                (alias,),
            ).fetchall()
            return [str(row["entry_id"]) for row in rows]
        return list(self._alias_to_entry_ids.get(alias, []))

    def _get_entry(self, entry_id: str) -> KnowledgeEntry:
        cached = self._entry_by_id.get(entry_id)
        if cached is not None:
            return cached

        if self._sqlite_conn is None:
            raise KeyError(entry_id)

        row = self._sqlite_conn.execute(
            """
            SELECT
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
                country
            FROM entries
            WHERE entry_id = ?
            """,
            (entry_id,),
        ).fetchone()
        if row is None:
            raise KeyError(entry_id)

        alias_rows = self._sqlite_conn.execute(
            "SELECT alias FROM aliases WHERE entry_id = ? ORDER BY priority DESC, alias LIMIT 16",
            (entry_id,),
        ).fetchall()
        aliases = [str(alias_row["alias"]) for alias_row in alias_rows]
        raw_category_labels = json.loads(row["raw_category_labels"] or "[]")

        entry = KnowledgeEntry(
            entry_id=str(row["entry_id"]),
            canonical_name=str(row["canonical_name"]),
            aliases=aliases or [str(row["canonical_name"])],
            mapped_category=row["mapped_category"],
            mapping_confidence=float(row["mapping_confidence"] or 0.0),
            metadata_text=str(row["metadata_text"] or "").strip(),
            source=str(row["source"] or "external"),
            raw_category_labels=[str(label) for label in raw_category_labels],
            locality=str(row["locality"]).strip() if row["locality"] else None,
            region=str(row["region"]).strip() if row["region"] else None,
            country=str(row["country"]).strip() if row["country"] else None,
            display_name=str(row["display_name"]).strip() if row["display_name"] else None,
        )
        self._entry_by_id[entry_id] = entry
        return entry

    def _get_document(self, entry_id: str) -> str:
        cached = self._documents.get(entry_id)
        if cached is not None:
            return cached

        if self._sqlite_conn is None:
            return ""

        row = self._sqlite_conn.execute(
            "SELECT document FROM entries WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        document = str(row["document"]) if row is not None and row["document"] else ""
        self._documents[entry_id] = document
        return document

    def _lexical_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._sqlite_conn is not None:
            if not self._sqlite_fts_ready:
                return []
            fts_query = self._fts_query(query)
            if fts_query is None:
                return []
            try:
                rows = self._sqlite_conn.execute(
                    """
                    SELECT entry_id, bm25(merchant_fts) AS rank
                    FROM merchant_fts
                    WHERE merchant_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []

            results: list[tuple[str, float]] = []
            for rank, row in enumerate(rows):
                results.append((str(row["entry_id"]), 1.0 / (rank + 1)))
            return results

        if self._lexical is None or self._lexical_matrix is None or not self._doc_ids:
            return []

        query_vec = self._lexical.transform([query])
        scores = cosine_similarity(query_vec, self._lexical_matrix).flatten()
        if scores.size == 0:
            return []

        ranked = np.argsort(scores)[::-1]
        results: list[tuple[str, float]] = []
        for idx in ranked[:top_k]:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            results.append((self._doc_ids[int(idx)], score))
        return results

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._collection is None or not self._dense_embedder.is_available:
            return []

        try:
            query_embedding = self._dense_embedder.encode([query], batch_size=1)[0].tolist()
            payload = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["distances"],
            )
        except Exception:
            return []

        ids = payload.get("ids", [[]])[0]
        distances = payload.get("distances", [[]])[0]
        results: list[tuple[str, float]] = []
        for entry_id, distance in zip(ids, distances):
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            results.append((str(entry_id), similarity))
        return results

    def _fuse_candidates(
        self,
        lexical_results: list[tuple[str, float]],
        dense_results: list[tuple[str, float]],
        *,
        seed_entry_ids: list[str] | None = None,
    ) -> dict[str, dict[str, float | int | None]]:
        fused: dict[str, dict[str, float | int | None]] = {}
        rrf_k = settings.knowledge_rrf_k

        if seed_entry_ids:
            for rank, entry_id in enumerate(seed_entry_ids):
                bucket = fused.setdefault(entry_id, self._empty_bucket())
                bucket["lexical_score"] = 1.0
                bucket["dense_score"] = 1.0
                bucket["lexical_rank"] = rank
                bucket["dense_rank"] = rank
                bucket["fused_score"] = float(bucket["fused_score"]) + (4.0 / (rank + 1))

        for rank, (entry_id, score) in enumerate(lexical_results):
            bucket = fused.setdefault(entry_id, self._empty_bucket())
            bucket["lexical_score"] = score
            bucket["lexical_rank"] = rank
            bucket["fused_score"] = float(bucket["fused_score"]) + (
                settings.knowledge_lexical_weight / (rrf_k + rank + 1)
            )

        for rank, (entry_id, score) in enumerate(dense_results):
            bucket = fused.setdefault(entry_id, self._empty_bucket())
            bucket["dense_score"] = score
            bucket["dense_rank"] = rank
            bucket["fused_score"] = float(bucket["fused_score"]) + (
                settings.knowledge_dense_weight / (rrf_k + rank + 1)
            )

        return fused

    def _rerank(
        self,
        query: str,
        candidate_ids: list[str],
    ) -> tuple[str | None, list[float]]:
        if not candidate_ids:
            return None, []

        if not self._reranker.is_available:
            return None, []

        try:
            documents = [self._get_document(entry_id) for entry_id in candidate_ids]
            scores = self._reranker.score(query, documents)
        except Exception:
            return None, []

        if not scores:
            return None, []

        best_index = int(np.argmax(np.asarray(scores)))
        return candidate_ids[best_index], scores

    def search(self, cleaned_text: str, min_similarity: float = 0.58) -> KnowledgeMatch | None:
        if not self._loaded:
            return None

        query = normalize_merchant_name(cleaned_text)
        if len(query) < 3:
            return None

        exact_entry_ids = self._lookup_exact_candidate_ids(query)
        if len(exact_entry_ids) == 1:
            entry = self._get_entry(exact_entry_ids[0])
            return KnowledgeMatch(
                entry=entry,
                matched_alias=query,
                similarity=1.0,
                strategy="exact",
                dense_score=1.0,
                lexical_score=1.0,
                fused_score=1.0,
                rerank_score=1.0,
            )

        lexical_results = self._lexical_search(query, settings.knowledge_lexical_candidates)
        dense_results = self._dense_search(query, settings.knowledge_dense_candidates)
        candidate_map = self._fuse_candidates(
            lexical_results,
            dense_results,
            seed_entry_ids=exact_entry_ids,
        )

        if not candidate_map:
            return None

        ranked_candidates = sorted(
            candidate_map.items(),
            key=lambda item: float(item[1]["fused_score"]),
            reverse=True,
        )
        rerank_ids = [
            entry_id for entry_id, _ in ranked_candidates[: settings.knowledge_rerank_candidates]
        ]

        best_entry_id, rerank_scores = self._rerank(query, rerank_ids)
        if best_entry_id is not None:
            best_score = float(rerank_scores[rerank_ids.index(best_entry_id)])
            if best_score >= min_similarity:
                bucket = candidate_map[best_entry_id]
                entry = self._get_entry(best_entry_id)
                return KnowledgeMatch(
                    entry=entry,
                    matched_alias=query if best_entry_id in exact_entry_ids else entry.canonical_name,
                    similarity=best_score,
                    strategy="rerank",
                    dense_score=float(bucket["dense_score"])
                    if bucket["dense_score"] is not None
                    else None,
                    lexical_score=float(bucket["lexical_score"])
                    if bucket["lexical_score"] is not None
                    else None,
                    fused_score=float(bucket["fused_score"]),
                    rerank_score=best_score,
                )

        best_entry_id, bucket = ranked_candidates[0]
        best_score = max(
            float(bucket["dense_score"]) if bucket["dense_score"] is not None else 0.0,
            float(bucket["lexical_score"]) if bucket["lexical_score"] is not None else 0.0,
        )
        if best_score < min_similarity:
            return None

        entry = self._get_entry(best_entry_id)
        return KnowledgeMatch(
            entry=entry,
            matched_alias=query if best_entry_id in exact_entry_ids else entry.canonical_name,
            similarity=best_score,
            strategy="dense_lexical",
            dense_score=float(bucket["dense_score"]) if bucket["dense_score"] is not None else None,
            lexical_score=float(bucket["lexical_score"])
            if bucket["lexical_score"] is not None
            else None,
            fused_score=float(bucket["fused_score"]),
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def size(self) -> int:
        return self._entry_count if self._sqlite_conn is not None else len(self._entries)

    @property
    def entries(self) -> list[KnowledgeEntry]:
        return list(self._entries)

    @property
    def chroma_ready(self) -> bool:
        return self._collection is not None
