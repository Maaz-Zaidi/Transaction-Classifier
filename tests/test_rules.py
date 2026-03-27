"""Tests for the rules engine."""

import pytest

from transaction_classifier.rules.engine import RulesEngine


@pytest.fixture
def engine():
    return RulesEngine()


@pytest.mark.parametrize(
    "cleaned_text, expected_category",
    [
        ("TIM HORTONS", "Food & Dining"),
        ("TIM HORTO", "Food & Dining"),
        ("MCDONALD'S", "Food & Dining"),
        ("STARBUCKS COFFEE", "Food & Dining"),
        ("UBER EATS", "Food & Dining"),
        ("SKIP THE DISHES", "Food & Dining"),
        ("DOORDASH", "Food & Dining"),
        ("LOBLAWS", "Food & Dining"),
        ("COSTCO", "Food & Dining"),
        ("FRESHCO", "Food & Dining"),
        ("OC TRANSPO", "Transportation"),
        ("PRESTO", "Transportation"),
        ("PETRO-CANADA", "Transportation"),
        ("UBER TRIP", "Transportation"),
        ("PARKING LOT", "Transportation"),
        ("AMAZON.CA", "Shopping & Retail"),
        ("CANADIAN TIRE STORE", "Shopping & Retail"),
        ("IKEA", "Shopping & Retail"),
        ("HOME DEPOT", "Shopping & Retail"),
        ("NETFLIX.COM", "Entertainment & Recreation"),
        ("SPOTIFY", "Entertainment & Recreation"),
        ("CINEPLEX", "Entertainment & Recreation"),
        ("SHOPPERS DRUG MART", "Healthcare & Medical"),
        ("REXALL PHARMACY", "Healthcare & Medical"),
        ("ROGERS WIRELESS", "Utilities & Services"),
        ("FIDO MOBILE", "Utilities & Services"),
        ("HYDRO OTTAWA", "Utilities & Services"),
        ("E-TRANSFER", "Financial Services"),
        ("MORTGAGE PAYMENT", "Financial Services"),
        ("PAYROLL", "Income"),
        ("CANADA REVENUE", "Government & Legal"),
        ("SERVICE ONTARIO", "Government & Legal"),
    ],
)
def test_known_merchants(engine, cleaned_text, expected_category):
    match = engine.match(cleaned_text)
    assert match is not None, f"No rule matched for {cleaned_text!r}"
    assert match.category == expected_category, (
        f"{cleaned_text!r}: got {match.category}, expected {expected_category}"
    )


def test_no_match_returns_none(engine):
    result = engine.match("XYZZY UNKNOWN MERCHANT")
    assert result is None


def test_match_confidence(engine):
    match = engine.match("TIM HORTONS")
    assert match is not None
    assert match.confidence == pytest.approx(0.98)


def test_uber_eats_vs_uber_trip(engine):
    """UBER EATS should be Food, UBER (ride) should be Transportation."""
    eats = engine.match("UBER EATS")
    ride = engine.match("UBER TRIP")
    assert eats.category == "Food & Dining"
    assert ride.category == "Transportation"


def test_match_batch(engine):
    texts = ["TIM HORTONS", "UNKNOWN", "NETFLIX"]
    results = engine.match_batch(texts)
    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None
