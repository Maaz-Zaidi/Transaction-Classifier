"""Tests for the ensemble orchestrator."""

import pytest

from transaction_classifier.models.ensemble import Ensemble, ClassificationResult
from transaction_classifier.models.sgd_model import SGDModel
from transaction_classifier.rules.engine import RulesEngine


@pytest.fixture
def ensemble():
    rules_engine = RulesEngine()
    # Train a minimal SGD model
    sgd = SGDModel()
    texts = [
        "TIM HORTONS", "STARBUCKS", "PIZZA",
        "UBER", "GAS STATION", "TAXI",
        "AMAZON", "STORE", "MALL",
        "NETFLIX", "CINEMA", "GAME",
        "PHARMACY", "DENTAL", "DOCTOR",
        "ROGERS", "HYDRO", "BELL",
        "MORTGAGE", "TRANSFER", "LOAN",
        "PAYROLL", "SALARY", "DEPOSIT",
        "CRA", "TAX", "GOVERNMENT",
        "CHARITY", "DONATION", "RED CROSS",
    ]
    labels = [
        "Food & Dining"] * 3 + [
        "Transportation"] * 3 + [
        "Shopping & Retail"] * 3 + [
        "Entertainment & Recreation"] * 3 + [
        "Healthcare & Medical"] * 3 + [
        "Utilities & Services"] * 3 + [
        "Financial Services"] * 3 + [
        "Income"] * 3 + [
        "Government & Legal"] * 3 + [
        "Charity & Donations"] * 3
    sgd.train(texts, labels)
    return Ensemble(rules_engine=rules_engine, sgd_model=sgd)


def test_classify_batch_returns_correct_length(ensemble):
    transactions = [
        "POS PURCHASE - 1234 TIM HORTONS OTTAWA ON",
        "SOME UNKNOWN MERCHANT",
        "INTERAC E-TRANSFER TO JOHN",
    ]
    results = ensemble.classify_batch(transactions)
    assert len(results) == 3


def test_known_merchant_uses_rules(ensemble):
    results = ensemble.classify_batch(["TIM HORTONS"])
    assert results[0].source == "rules"
    assert results[0].category == "Food & Dining"
    assert results[0].confidence == pytest.approx(0.98)
    assert results[0].flagged_for_review is False


def test_unknown_merchant_uses_sgd(ensemble):
    results = ensemble.classify_batch(["TOTALLY RANDOM UNKNOWN THING"])
    assert results[0].source == "sgd"


def test_classify_single(ensemble):
    result = ensemble.classify_single("NETFLIX.COM")
    assert isinstance(result, ClassificationResult)
    assert result.category is not None


def test_results_have_all_fields(ensemble):
    result = ensemble.classify_single("STARBUCKS COFFEE")
    assert result.transaction == "STARBUCKS COFFEE"
    assert result.cleaned  # not empty
    assert result.category
    assert result.confidence > 0
    assert result.source in ("rules", "sgd", "bert")
    assert isinstance(result.flagged_for_review, bool)
