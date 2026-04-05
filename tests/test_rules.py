"""tests for the rules engine."""

import pytest

from transaction_classifier.rules.engine import RulesEngine


@pytest.fixture
def engine():
    return RulesEngine()


@pytest.mark.parametrize(
    "cleaned_text, expected_category",
    [
        # financial services structural patterns
        ("E-TRANSFER-OUT", "Financial Services"),
        ("TRANSFER TO SAVINGS", "Financial Services"),
        ("MORTGAGE PAYMENT", "Financial Services"),
        ("LOAN PAYMENT", "Financial Services"),
        ("CREDIT CARD PAYMENT", "Financial Services"),
        ("NSF FEE", "Financial Services"),
        ("OVERDRAFT", "Financial Services"),
        ("SERVICE CHARGE", "Financial Services"),
        ("MONTHLY FEE", "Financial Services"),
        # income structural patterns
        ("E-TRANSFER-IN", "Income"),
        ("PAYROLL", "Income"),
        ("SALARY", "Income"),
        ("DEPOSIT", "Income"),
        ("REFUND", "Income"),
        ("CASH BACK", "Income"),
        # government and legal agency patterns
        ("CANADA REVENUE", "Government & Legal"),
        ("SERVICE ONTARIO", "Government & Legal"),
        ("SERVICE CANADA", "Government & Legal"),
        ("CRA PAYMENT", "Government & Legal"),
        ("TAX PAYMENT", "Government & Legal"),
        # generic service descriptors
        ("PARKING LOT", "Transportation"),
        ("PHARMACY", "Healthcare & Medical"),
        ("DENTAL OFFICE", "Healthcare & Medical"),
        ("HOSPITAL", "Healthcare & Medical"),
        ("CLINIC", "Healthcare & Medical"),
        ("OPTOMETRY", "Healthcare & Medical"),
        ("GYM", "Entertainment & Recreation"),
        ("FITNESS CENTRE", "Entertainment & Recreation"),
    ],
)
def test_structural_rules(engine, cleaned_text, expected_category):
    match = engine.match(cleaned_text)
    assert match is not None, f"No rule matched for {cleaned_text!r}"
    assert match.category == expected_category, (
        f"{cleaned_text!r}: got {match.category}, expected {expected_category}"
    )


def test_no_match_returns_none(engine):
    result = engine.match("XYZZY UNKNOWN MERCHANT")
    assert result is None


def test_merchant_names_are_not_ruled(engine):
    """merchant-name rules are gone, so brands should go through kb/ml."""
    merchant_names = [
        "TIM HORTONS",
        "MCDONALD'S",
        "COSTCO",
        "AMAZON",
        "WALMART",
        "NETFLIX",
        "SPOTIFY",
        "CANADIAN TIRE",
        "UBER",
        "UBER EATS",
        "LOBLAWS",
        "SHELL",
        "ROGERS",
        "BELL CANADA",
    ]
    for name in merchant_names:
        result = engine.match(name)
        assert result is None, (
            f"Merchant {name!r} should NOT be handled by rules, "
            f"but matched rule {result.rule_pattern!r} -> {result.category}"
        )


def test_match_confidence(engine):
    match = engine.match("MORTGAGE PAYMENT")
    assert match is not None
    assert match.confidence == pytest.approx(0.98)


def test_match_batch(engine):
    texts = ["MORTGAGE PAYMENT", "UNKNOWN MERCHANT", "PARKING LOT"]
    results = engine.match_batch(texts)
    assert len(results) == 3
    assert results[0] is not None
    assert results[0].category == "Financial Services"
    assert results[1] is None
    assert results[2] is not None
    assert results[2].category == "Transportation"
