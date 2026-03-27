"""Tests for the text cleaning pipeline."""

import pytest

from transaction_classifier.data.preprocess import clean_transaction


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Bank prefix stripping
        ("POS PURCHASE - 1847 TIM HORTO OTTAWA ON", "TIM HORTO"),
        ("INTERAC PURCHASE-0423 COSTCO WHOLESALE NEPEAN ON", "COSTCO WHOLESALE"),
        ("PAY/PAIE EMPLOYER NAME", "EMPLOYER NAME"),
        ("INTERNET BILL PMT ROGERS", "ROGERS"),
        ("PREAUTHORIZED DEBIT NETFLIX.COM", "NETFLIX.COM"),
        ("PRE-AUTHORIZED PAYMENT SPOTIFY", "SPOTIFY"),
        ("BILL PAYMENT HYDRO OTTAWA", "HYDRO OTTAWA"),
        ("RECURRING PAYMENT GOODLIFE", "GOODLIFE"),
        ("MISC PAYMENT INSURANCE CO", "INSURANCE CO"),

        # E-transfers -> marker token
        ("INTERAC E-TRANSFER TO JOHN D", "E-TRANSFER"),
        ("INTERAC E-TRANSFER FROM JANE S", "E-TRANSFER"),
        ("INTERAC E TRANSFER TO SOME PERSON", "E-TRANSFER"),

        # Internal transfers
        ("TFR-TO C/C SAVINGS", "TRANSFER C/C SAVINGS"),
        ("TFR-FR SAVINGS", "TRANSFER SAVINGS"),

        # Numeric noise removal
        ("UBER *TRIP 284", "UBER TRIP"),
        ("AMAZON.CA*XXXXXXXXX AMAZON.CA ON", "AMAZON.CA AMAZON.CA"),
        ("TIM HORTONS #456 OTTAWA ON", "TIM HORTONS"),
        ("STARBUCKS 12345 TORONTO ON", "STARBUCKS"),

        # Location suffix stripping
        ("LOBLAWS OTTAWA ON", "LOBLAWS"),
        ("SHOPPERS DRUG MART TORONTO ON", "SHOPPERS DRUG MART"),
        ("SOME STORE VANCOUVER BC", "SOME STORE"),
        ("MERCHANT NAME MONTREAL QC", "MERCHANT NAME"),
        ("STORE HALIFAX NS CA", "STORE"),

        # Clean passthrough (already clean merchant names from dataset)
        ("Tim Hortons", "TIM HORTONS"),
        ("Starbucks Coffee", "STARBUCKS COFFEE"),
        ("McDonald's", "MCDONALD'S"),

        # Edge cases
        ("", ""),
        ("   ", ""),
        ("A", "A"),
    ],
)
def test_clean_transaction(raw, expected):
    result = clean_transaction(raw)
    assert result == expected, f"clean_transaction({raw!r}) = {result!r}, expected {expected!r}"


def test_clean_transaction_is_uppercase():
    assert clean_transaction("tim hortons").isupper()


def test_clean_transaction_no_leading_trailing_spaces():
    result = clean_transaction("  POS PURCHASE - 1234 STORE OTTAWA ON  ")
    assert result == result.strip()
