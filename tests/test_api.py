"""Tests for the FastAPI endpoints."""

import pytest

from transaction_classifier.api.routes import categories, classify, health, set_ensemble
from transaction_classifier.api.schemas import ClassifyRequest, TransactionInput
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
def api_ready():
    """Seed the module-level ensemble used by the route handlers."""
    ensemble = _make_ensemble()
    set_ensemble(ensemble)
    return ensemble


def test_health(api_ready):
    response = health()
    assert response.status == "ok"
    assert response.sgd_loaded is True


def test_categories(api_ready):
    cats = categories()["categories"]
    assert len(cats) == 10
    assert "Food & Dining" in cats


def test_classify_batch(api_ready):
    response = classify(
        ClassifyRequest(
            transactions=[
                TransactionInput(description="TIM HORTONS"),
                TransactionInput(description="UBER TRIP"),
                TransactionInput(description="SOME RANDOM STORE"),
            ]
        )
    )
    assert len(response.results) == 3
    assert response.processing_time_ms >= 0
    assert response.model_version

    for result in response.results:
        assert result.description
        assert result.category
        assert result.confidence >= 0
        assert result.source
        assert isinstance(result.flagged_for_review, bool)


def test_classify_empty_batch(api_ready):
    response = classify(ClassifyRequest(transactions=[]))
    assert len(response.results) == 0


def test_classify_known_merchant_uses_rules(api_ready):
    response = classify(
        ClassifyRequest(transactions=[TransactionInput(description="TIM HORTONS")])
    )
    result = response.results[0]
    assert result.source == "rules"
    assert result.category == "Food & Dining"
