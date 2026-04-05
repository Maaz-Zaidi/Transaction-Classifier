"""helpers for merchant retrieval and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


@dataclass(slots=True)
class DenseRetrievalResult:
    document_ids: list[str]
    scores: list[float]


class MerchantDenseEmbedder:
    """load the dense retriever the first time we need it."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self._lock = Lock()
        self._available = True

    @property
    def is_available(self) -> bool:
        return self._available

    def _ensure_loaded(self) -> bool:
        if not self._available:
            return False

        if self._model is not None and self._tokenizer is not None:
            return True

        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return True

            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except Exception:
                self._available = False
                return False

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()

                target_device = self.device
                if target_device is None:
                    target_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.device = target_device
                self._model.to(self.device)
                return True
            except Exception:
                self._available = False
                self._tokenizer = None
                self._model = None
                return False

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if not self._ensure_loaded():
            raise RuntimeError(f"Dense embedder {self.model_name} is unavailable.")

        import torch

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
                vectors.append(normalized.cpu().numpy().astype(np.float32))

        return np.concatenate(vectors, axis=0)


class MerchantReranker:
    """load the reranker the first time we need it."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self._lock = Lock()
        self._available = True

    @property
    def is_available(self) -> bool:
        return self._available

    def _ensure_loaded(self) -> bool:
        if not self._available:
            return False

        if self._model is not None and self._tokenizer is not None:
            return True

        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return True

            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except Exception:
                self._available = False
                return False

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.eval()

                target_device = self.device
                if target_device is None:
                    target_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.device = target_device
                self._model.to(self.device)
                return True
            except Exception:
                self._available = False
                self._tokenizer = None
                self._model = None
                return False

    def score(self, query: str, documents: list[str], batch_size: int = 16) -> list[float]:
        if not documents:
            return []

        if not self._ensure_loaded():
            raise RuntimeError(f"Reranker {self.model_name} is unavailable.")

        import torch

        scores: list[float] = []
        pairs = [[query, document] for document in documents]

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            encoded = self._tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                logits = self._model(**encoded, return_dict=True).logits.view(-1).float()
                probs = _sigmoid(logits.cpu().numpy())
                scores.extend(float(value) for value in probs.tolist())

        return scores
