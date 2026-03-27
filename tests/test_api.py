"""Tests for the FastAPI endpoints."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transaction_classifier.api.routes import router, set_ensemble
from transaction_classifier.models.ensemble import Ensemble
from transaction_classifier.models.sgd_model import SGDModel
from transaction_classifier.rules.engine import RulesEngine


def _make_ensemble() -> Ensemble:
    """Create a minimal trained ensemble for testing."""
    rules_engine = RulesEngine()
    sgd = SGDModel()
    texts = [
        "TIM HORTONS", "STARBUCKS", "PIZZA",
        "UBER", "GAS", "TAXI",
        "AMAZON", "STORE", "MALL",
        "NETFLIX", "CINEMA", "GAME",
        "PHARMACY", "DENTAL", "DOCTOR",
        "ROGERS", "HYDRO", "BELL",
        "MORTGAGE", "TRANSFER", "LOAN",
        "PAYROLL", "SALARY", "DEPOSIT",
        "CRA", "TAX", "GOVERNMENT",
        "CHARITY", "DONATION", "RED CROSS",
    ]
    labels = (
        ["Food & Dining"] * 3
        + ["Transportation"] * 3
        + ["Shopping & Retail"] * 3
        + ["Entertainment & Recreation"] * 3
        + ["Healthcare & Medical"] * 3
        + ["Utilities & Services"] * 3
        + ["Financial Services"] * 3
        + ["Income"] * 3
        + ["Government & Legal"] * 3
        + ["Charity & Donations"] * 3
    )
    sgd.train(texts, labels)
    return Ensemble(rules_engine=rules_engine, sgd_model=sgd)


@pytest.fixture
def client():
    """Create test client with a minimal trained ensemble (no lifespan)."""
    app = FastAPI()
    app.include_router(router)

    ensemble = _make_ensemble()
    set_ensemble(ensemble)

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["sgd_loaded"] is True


def test_categories(client):
    resp = client.get("/categories")
    assert resp.status_code == 200
    cats = resp.json()["categories"]
    assert len(cats) == 10
    assert "Food & Dining" in cats


def test_classify_batch(client):
    resp = client.post("/classify", json={
        "transactions": [
            {"description": "TIM HORTONS"},
            {"description": "UBER TRIP"},
            {"description": "SOME RANDOM STORE"},
        ]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    assert data["processing_time_ms"] >= 0
    assert data["model_version"]

    for r in data["results"]:
        assert "description" in r
        assert "category" in r
        assert "confidence" in r
        assert "source" in r
        assert "flagged_for_review" in r


def test_classify_empty_batch(client):
    resp = client.post("/classify", json={"transactions": []})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 0


def test_classify_known_merchant_uses_rules(client):
    resp = client.post("/classify", json={
        "transactions": [{"description": "TIM HORTONS"}]
    })
    result = resp.json()["results"][0]
    assert result["source"] == "rules"
    assert result["category"] == "Food & Dining"
