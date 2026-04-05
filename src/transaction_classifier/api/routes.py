"""fastapi routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from transaction_classifier.api.schemas import (
    ClassificationOutput,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
)
from transaction_classifier.categories import ALL_LABELS
from transaction_classifier.models.ensemble import Ensemble

router = APIRouter()

# app.py fills these in at startup
_ensemble: Ensemble | None = None
_model_version: str = "unknown"


def set_ensemble(ensemble: Ensemble, version: str = "0.1.0") -> None:
    global _ensemble, _model_version
    _ensemble = ensemble
    _model_version = version


def _get_ensemble() -> Ensemble:
    if _ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    return _ensemble


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    ensemble = _get_ensemble()
    descriptions = [t.description for t in request.transactions]

    start = time.perf_counter()
    results = ensemble.classify_batch(descriptions)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return ClassifyResponse(
        results=[
            ClassificationOutput(
                description=r.transaction,
                category=r.category,
                confidence=round(r.confidence, 4),
                source=r.source,
                flagged_for_review=r.flagged_for_review,
            )
            for r in results
        ],
        processing_time_ms=round(elapsed_ms, 2),
        model_version=_model_version,
    )


@router.get("/categories")
def categories():
    return {"categories": ALL_LABELS}


@router.get("/health", response_model=HealthResponse)
def health():
    ensemble = _get_ensemble()
    zeroshot_loaded = bool(
        ensemble.zeroshot_model is not None and ensemble.zeroshot_model.is_loaded
    )
    knowledge_base_loaded = bool(
        ensemble.knowledge_base is not None and ensemble.knowledge_base.is_loaded
    )
    knowledge_retrieval_ready = bool(
        ensemble.knowledge_base is not None and ensemble.knowledge_base.chroma_ready
    )
    sgd_loaded = bool(ensemble.sgd_model is not None and ensemble.sgd_model.is_fitted)
    return HealthResponse(
        status="ok",
        model_version=_model_version,
        sgd_loaded=sgd_loaded,
        bert_loaded=zeroshot_loaded,
        knowledge_base_loaded=knowledge_base_loaded,
        knowledge_retrieval_ready=knowledge_retrieval_ready,
        zeroshot_loaded=zeroshot_loaded,
        primary_model=ensemble.primary_source,
        rules_count=len(ensemble.rules_engine._rules),
    )
