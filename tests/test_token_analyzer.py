"""tests for token splitting in transaction strings."""

import pytest

from transaction_classifier.knowledge.token_analyzer import (
    DecomposedQuery,
    TokenAnalyzer,
    TokenRole,
)


@pytest.fixture
def analyzer():
    return TokenAnalyzer()


@pytest.mark.parametrize(
    "text, expected_brand, expected_descriptor, expected_noise",
    [
        # brand, descriptor, and noise
        ("COSTCO GAS W1263", ["COSTCO"], ["GAS"], ["W1263"]),
        # brand and downloads descriptor
        ("AMAZON DOWNLOADS", ["AMAZON"], ["DOWNLOADS"], []),
        # brand and tuition descriptor
        ("UNI OTT TUITION", ["UNI", "OTT"], ["TUITION"], []),
        # single brand token
        ("NETFLIX.COM", ["NETFLIX.COM"], [], []),
        # noise only, so brand_query falls back to the original text
        ("W1263", [], [], ["W1263"]),
        # descriptor only
        ("GAS", [], ["GAS"], []),
        # multi-word descriptor
        ("AMAZON WEB SERVICES", ["AMAZON"], ["WEB", "SERVICES"], []),
        # brand with a convenience-store descriptor
        ("MACS CONV. STORE", ["MACS", "STORE"], ["CONV."], []),
        # grocery descriptor
        ("FRESHCO GROCERY", ["FRESHCO"], ["GROCERY"], []),
        # wholesale descriptor
        ("COSTCO WHOLESALE", ["COSTCO"], ["WHOLESALE"], []),
    ],
)
def test_token_decomposition(analyzer, text, expected_brand, expected_descriptor, expected_noise):
    result = analyzer.analyze(text)
    assert result.brand_tokens == expected_brand, (
        f"brand_tokens for {text!r}: got {result.brand_tokens}, expected {expected_brand}"
    )
    assert result.descriptor_tokens == expected_descriptor, (
        f"descriptor_tokens for {text!r}: got {result.descriptor_tokens}, expected {expected_descriptor}"
    )
    assert result.noise_tokens == expected_noise, (
        f"noise_tokens for {text!r}: got {result.noise_tokens}, expected {expected_noise}"
    )


@pytest.mark.parametrize(
    "text, expected_locations",
    [
        ("ZARA BAYSHORE", ["BAYSHORE"]),
        ("STORE OTTAWA", ["OTTAWA"]),
        ("TIM HORTONS KANATA", ["KANATA"]),
        ("PLAIN MERCHANT", []),
    ],
)
def test_location_detection(analyzer, text, expected_locations):
    result = analyzer.analyze(text)
    assert result.location_tokens == expected_locations


def test_brand_query_construction(analyzer):
    result = analyzer.analyze("COSTCO GAS W1263")
    assert result.brand_query == "COSTCO"


def test_brand_query_fallback_to_original(analyzer):
    result = analyzer.analyze("W1263")
    assert result.brand_query == "W1263"


def test_descriptor_context_single(analyzer):
    result = analyzer.analyze("COSTCO GAS W1263")
    assert "gas station" in result.descriptor_context
    assert "fuel" in result.descriptor_context


def test_descriptor_context_multi_token(analyzer):
    result = analyzer.analyze("AMAZON WEB SERVICES")
    assert "cloud computing" in result.descriptor_context


def test_empty_input(analyzer):
    result = analyzer.analyze("")
    assert result.brand_tokens == []
    assert result.descriptor_tokens == []
    assert result.noise_tokens == []
    assert result.brand_query == ""


def test_whitespace_input(analyzer):
    result = analyzer.analyze("   ")
    assert result.brand_tokens == []
    assert result.brand_query == ""


def test_all_roles_present(analyzer):
    result = analyzer.analyze("COSTCO GAS OTTAWA W1263")
    assert "COSTCO" in result.brand_tokens
    assert "GAS" in result.descriptor_tokens
    assert "OTTAWA" in result.location_tokens
    assert "W1263" in result.noise_tokens


def test_decomposed_query_is_dataclass(analyzer):
    result = analyzer.analyze("TEST")
    assert isinstance(result, DecomposedQuery)
    assert result.original == "TEST"


def test_multi_token_descriptor_drug_mart(analyzer):
    result = analyzer.analyze("SHOPPERS DRUG MART")
    assert "DRUG" in result.descriptor_tokens
    assert "MART" in result.descriptor_tokens
    assert "pharmacy" in result.descriptor_context


def test_noise_patterns(analyzer):
    result = analyzer.analyze("STORE S123 P456 #789")
    for token in ["S123", "P456"]:
        assert token in result.noise_tokens, f"{token} should be noise"
