"""Tests for the SGD model."""

import tempfile
from pathlib import Path

import pytest

from transaction_classifier.models.sgd_model import SGDModel


@pytest.fixture
def trained_model():
    """Small model trained on fixture data for testing."""
    texts = [
        "TIM HORTONS", "MCDONALD'S", "STARBUCKS", "PIZZA PIZZA", "SUBWAY",
        "UBER TRIP", "PETRO-CANADA", "PARKING", "OC TRANSPO", "PRESTO",
        "AMAZON", "IKEA", "HOME DEPOT", "BEST BUY", "CANADIAN TIRE",
        "NETFLIX", "SPOTIFY", "CINEPLEX", "STEAM", "GOODLIFE",
        "SHOPPERS DRUG", "REXALL", "PHARMACY", "DENTAL CLINIC", "HOSPITAL",
        "ROGERS", "BELL", "HYDRO", "ENBRIDGE", "FIDO",
        "MORTGAGE", "TRANSFER", "LOAN PAYMENT", "SERVICE CHARGE", "NSF FEE",
        "PAYROLL", "SALARY", "DEPOSIT", "REFUND", "CASH BACK",
        "CRA", "SERVICE ONTARIO", "TAX PAYMENT", "GOV OF CANADA", "LICENSE",
        "RED CROSS", "UNITED WAY", "SALVATION ARMY", "CHARITY", "DONATION",
    ]
    labels = [
        "Food & Dining"] * 5 + [
        "Transportation"] * 5 + [
        "Shopping & Retail"] * 5 + [
        "Entertainment & Recreation"] * 5 + [
        "Healthcare & Medical"] * 5 + [
        "Utilities & Services"] * 5 + [
        "Financial Services"] * 5 + [
        "Income"] * 5 + [
        "Government & Legal"] * 5 + [
        "Charity & Donations"] * 5

    model = SGDModel()
    model.train(texts, labels)
    return model


def test_train_returns_info(trained_model):
    assert trained_model.is_fitted


def test_predict_returns_valid_categories(trained_model):
    preds = trained_model.predict(["TIM HORTONS", "UBER", "AMAZON"])
    assert len(preds) == 3
    for p in preds:
        assert p.category in [
            "Food & Dining", "Transportation", "Shopping & Retail",
            "Entertainment & Recreation", "Healthcare & Medical",
            "Utilities & Services", "Financial Services", "Income",
            "Government & Legal", "Charity & Donations",
        ]
        assert 0.0 <= p.confidence <= 1.0


def test_predict_single(trained_model):
    pred = trained_model.predict_single("STARBUCKS COFFEE")
    assert pred.category is not None
    assert pred.confidence > 0


def test_save_and_load(trained_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model"
        trained_model.save(path)

        loaded = SGDModel()
        loaded.load(path)
        assert loaded.is_fitted

        # predictions match
        texts = ["TIM HORTONS", "UBER"]
        orig_preds = trained_model.predict(texts)
        loaded_preds = loaded.predict(texts)
        for o, l in zip(orig_preds, loaded_preds):
            assert o.category == l.category
            assert abs(o.confidence - l.confidence) < 1e-6
