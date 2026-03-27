"""Shared test fixtures."""

import pytest


SAMPLE_TRANSACTIONS = [
    ("POS PURCHASE - 1847 TIM HORTO OTTAWA ON", "Food & Dining"),
    ("INTERAC PURCHASE-0423 COSTCO WHOLESALE NEPEAN ON", "Food & Dining"),
    ("PAY/PAIE EMPLOYER NAME", "Income"),
    ("INTERNET BILL PMT ROGERS", "Utilities & Services"),
    ("PREAUTHORIZED DEBIT NETFLIX.COM", "Entertainment & Recreation"),
    ("UBER *TRIP 284", "Transportation"),
    ("AMAZON.CA*XXXXXXXXX AMAZON.CA ON", "Shopping & Retail"),
    ("INTERAC E-TRANSFER TO JOHN D", "Financial Services"),
    ("CANADIAN TIRE STORE OTTAWA ON", "Shopping & Retail"),
    ("PETRO-CANADA KANATA ON", "Transportation"),
]

SAMPLE_CLEANED = [
    "TIM HORTO",
    "COSTCO WHOLESALE",
    "EMPLOYER NAME",
    "ROGERS",
    "NETFLIX.COM",
    "UBER TRIP",
    "AMAZON.CA AMAZON.CA",
    "E-TRANSFER",
    "CANADIAN TIRE STORE",
    "PETRO-CANADA",
]


@pytest.fixture
def sample_transactions():
    return SAMPLE_TRANSACTIONS


@pytest.fixture
def sample_cleaned():
    return SAMPLE_CLEANED
