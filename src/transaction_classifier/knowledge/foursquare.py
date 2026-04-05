"""Helpers for translating Foursquare metadata into model-friendly text."""

from __future__ import annotations

from collections import Counter


_KEYWORD_RULES: dict[str, tuple[str, ...]] = {
    "Food & Dining": (
        "DINING AND DRINKING",
        "RESTAURANT",
        "CAFE",
        "COFFEE",
        "TEA",
        "BAKERY",
        "BUBBLE TEA",
        "FOOD AND BEVERAGE RETAIL",
        "GROCERY",
        "SUPERMARKET",
        "DELI",
        "BUTCHER",
        "ICE CREAM",
        "PIZZA",
        "BURGER",
        "FAST FOOD",
        "SANDWICH",
        "CONVENIENCE STORE",
    ),
    "Transportation": (
        "TRANSPORTATION",
        "GAS STATION",
        "PARKING",
        "AUTOMOTIVE",
        "CAR WASH",
        "TRANSIT",
        "AIRPORT",
        "TRAIN STATION",
        "BUS STATION",
    ),
    "Shopping & Retail": (
        "RETAIL",
        "STORE",
        "SHOP",
        "CLOTHING",
        "ELECTRONICS",
        "DEPARTMENT STORE",
        "BOOKSTORE",
        "FURNITURE",
        "DISCOUNT STORE",
        "JEWELRY",
        "SHOPPING MALL",
    ),
    "Entertainment & Recreation": (
        "ARTS AND ENTERTAINMENT",
        "MOVIE THEATER",
        "MUSEUM",
        "GYM",
        "FITNESS",
        "SPORTS",
        "RECREATION",
        "ARCADE",
        "NIGHT CLUB",
        "MUSIC VENUE",
        "CASINO",
    ),
    "Healthcare & Medical": (
        "HEALTH AND MEDICINE",
        "PHARMACY",
        "DENTAL",
        "HOSPITAL",
        "DOCTOR",
        "CLINIC",
        "MEDICAL",
        "OPTICAL",
    ),
    "Utilities & Services": (
        "UTILITY",
        "TELECOMMUNICATION",
        "INTERNET SERVICE PROVIDER",
        "REPAIR",
        "LAUNDRY",
        "DRY CLEANER",
        "HOME SERVICE",
        "POST OFFICE",
        "SHIPPING",
        "BARBER",
        "SALON",
    ),
    "Financial Services": (
        "FINANCIAL SERVICE",
        "BANK",
        "ATM",
        "INSURANCE",
        "CREDIT UNION",
        "CURRENCY EXCHANGE",
        "LOAN",
    ),
    "Government & Legal": (
        "GOVERNMENT",
        "LEGAL SERVICE",
        "COURTHOUSE",
        "EMBASSY",
    ),
    "Charity & Donations": (
        "NONPROFIT",
        "SOCIAL SERVICE",
        "RELIGIOUS ORGANIZATION",
        "CHARITY",
        "COMMUNITY CENTER",
    ),
}


def map_foursquare_labels(labels: list[str]) -> tuple[str | None, float]:
    """Map Foursquare category paths into the project's coarse taxonomy."""
    scores: Counter[str] = Counter()
    for label in labels:
        upper = label.upper()
        depth_bonus = 1.0 + (upper.count(">") * 0.1)
        for category, keywords in _KEYWORD_RULES.items():
            if any(keyword in upper for keyword in keywords):
                scores[category] += depth_bonus

    if not scores:
        return None, 0.0

    best_category, best_score = scores.most_common(1)[0]
    total = float(sum(scores.values()))
    confidence = best_score / total if total else 0.0

    if confidence < 0.55:
        return None, confidence
    return best_category, confidence


def build_metadata_text(
    category_labels: list[str],
    locality: str | None,
    region: str | None,
    country: str | None,
) -> str:
    """Build a compact metadata string suitable for model enrichment."""
    parts: list[str] = []

    if category_labels:
        parts.append("place types: " + "; ".join(category_labels[:3]))

    location_bits = [bit for bit in (locality, region, country) if bit]
    if location_bits:
        parts.append("location: " + ", ".join(location_bits))

    return ". ".join(parts)

