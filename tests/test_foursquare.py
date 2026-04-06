"""tests for the foursquare category mapper."""

import pytest

from transaction_classifier.knowledge.foursquare import (
    build_metadata_text,
    map_foursquare_labels,
)


@pytest.mark.parametrize(
    "labels, expected_category",
    [
        # food labels that previously tied at 0.50
        (["Retail > Food and Beverage Retail > Grocery Store"], "Food & Dining"),
        (["Retail > Convenience Store"], "Food & Dining"),
        (["Dining and Drinking > Cafe, Coffee, and Tea House > Bubble Tea Shop"], "Food & Dining"),
        (["Retail > Food and Beverage Retail > Supermarket"], "Food & Dining"),
        # pharmacy should be healthcare, not shopping
        (["Retail > Pharmacy"], "Healthcare & Medical"),
        # church should be charity, not government
        (["Community and Government > Spiritual Center > Church"], "Charity & Donations"),
        # straightforward single-category labels
        (["Travel and Transportation > Gas Station"], "Transportation"),
        (["Travel and Transportation > Rest Area"], "Transportation"),
        (["Retail > Electronics Store"], "Shopping & Retail"),
        (["Arts and Entertainment > Museum"], "Entertainment & Recreation"),
        (["Health and Medicine > Dental Clinic"], "Healthcare & Medical"),
        (["Retail > Clothing Store"], "Shopping & Retail"),
        # multi-label: most specific should win
        (
            [
                "Retail > Food and Beverage Retail > Grocery Store",
                "Retail > Food and Beverage Retail > Supermarket",
            ],
            "Food & Dining",
        ),
    ],
)
def test_map_foursquare_labels(labels, expected_category):
    category, confidence = map_foursquare_labels(labels)
    assert category == expected_category
    assert confidence >= 0.40


def test_map_foursquare_labels_no_match():
    category, confidence = map_foursquare_labels(["Completely Unknown Label"])
    assert category is None
    assert confidence == 0.0


def test_map_foursquare_labels_empty():
    category, confidence = map_foursquare_labels([])
    assert category is None
    assert confidence == 0.0


def test_automotive_service_is_transportation():
    category, _ = map_foursquare_labels(["Automotive Service > Tire Shop"])
    assert category == "Transportation"


def test_specificity_weighting():
    """GROCERY (2.0) should outweigh RETAIL (1.0) and STORE (1.0)."""
    category, confidence = map_foursquare_labels(
        ["Retail > Food and Beverage Retail > Grocery Store"]
    )
    assert category == "Food & Dining"
    assert confidence > 0.50  # should be well above the 0.40 threshold


def test_build_metadata_text():
    text = build_metadata_text(
        category_labels=["Dining and Drinking > Restaurant"],
        locality="Ottawa",
        region="ON",
        country="CA",
    )
    assert "place types:" in text
    assert "Restaurant" in text
    assert "Ottawa" in text


def test_build_metadata_text_empty():
    text = build_metadata_text(
        category_labels=[],
        locality=None,
        region=None,
        country=None,
    )
    assert text == ""
