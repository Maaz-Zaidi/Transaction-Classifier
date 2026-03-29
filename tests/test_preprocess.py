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

        # E-transfers -> directional markers
        ("INTERAC E-TRANSFER TO JOHN D", "E-TRANSFER-OUT"),
        ("INTERAC E-TRANSFER FROM JANE S", "E-TRANSFER-IN"),
        ("INTERAC E TRANSFER TO SOME PERSON", "E-TRANSFER-OUT"),

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

        # === New prefix patterns ===
        ("Contactless Interac purchase - 4471 HOT CRISPY CHIC", "HOT CRISPY CHIC"),
        ("Online Banking payment - 5824 UNI OTT TUITION", "UNI OTT TUITION"),
        ("ATM withdrawal - OI820090", ""),
        ("Mobile cheque deposit - 9842", ""),
        ("01339 MACS CONV. STORE KANATA ON", "MACS CONV. STORE"),
        # Toast POS prefix
        ("TST-TAHINIS - 1940 EA OTTAWA ON", "TAHINIS - EA"),

        # === Real RBC chequing patterns (PDF-extracted) ===
        # "VisaDebitpurchase- 3076 Amazon.caPrime" (PDF joins words)
        ("VisaDebitpurchase- 3076 Amazon.caPrime", "AMAZON.CAPRIME"),
        ("Visa Debit purchase - 4232 Amazon Web Serv", "AMAZON WEB SERV"),
        # "MiscPayment GOODLIFECLUBS"
        ("MiscPayment GOODLIFECLUBS", "GOODLIFECLUBS"),
        ("Misc Payment RBC CREDIT CARD", "RBC CREDIT CARD"),
        # "PayrollDeposit EricssonCanada"
        ("PayrollDeposit EricssonCanada", "ERICSSONCANADA"),
        ("Payroll Deposit Ericsson Canada", "ERICSSON CANADA"),
        # E-transfers (RBC chequing format) with direction
        ("e-Transfer- Autodeposit ALP MERT TATAR C1ADReTVQJhM", "E-TRANSFER-IN"),
        ("e-Transfersent gulreena UG49BT", "E-TRANSFER-OUT"),
        ("e-Transfer sent gulreena UG49BT", "E-TRANSFER-OUT"),
        ("e-Transfer received SOMEONE NAME", "E-TRANSFER-IN"),
        # Canada Carbon Rebate
        ("CanadaCarbon Rebate CANADA", "CANADA"),
        ("Canada Carbon Rebate CANADA", "CANADA"),

        # === Real RBC MasterCard patterns ===
        ("TIM HORTONS #1664 KANATA ON", "TIM HORTONS"),
        ("SHOPPERS DRUG MART 631 OTTAWA ON", "SHOPPERS DRUG MART"),
        ("THE HOME DEPOT #7108 KANATA ON", "THE HOME DEPOT"),
        ("T&T SUPERMARKET #039 KANATA ON", "T&T SUPERMARKET"),
        ("FRESHCO #9620 NEPEAN ON", "FRESHCO"),
        ("SQ *CHAIGUYS NEPEAN ON", "CHAIGUYS"),
        ("*RFBT-RIDEAU CENTRE OTTAWA ON", "RIDEAU CENTRE"),
        ("AMZN MKTP CA*Z10WY2A31 WWW.AMAZON.CAON", "WWW.AMAZON.CAON"),
        ("SHAWARMA PRINCE KANATA ON", "SHAWARMA PRINCE"),
        ("MARY BROWNS CHICKEN KANATA ON", "MARY BROWNS CHICKEN"),
        ("CANADA COMPUTERS #30 KANATA ON", "CANADA COMPUTERS"),

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
