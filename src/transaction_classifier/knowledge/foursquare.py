"""Helpers for translating Foursquare metadata into model-friendly text."""

from __future__ import annotations

from collections import Counter


# specific keywords get higher weight than generic structural ones.
# this ensures "Retail > Grocery Store" resolves to Food (GROCERY=2.0)
# rather than tying with Shopping (RETAIL=1.0).
_GENERIC_KEYWORDS: set[str] = {"RETAIL", "STORE", "SHOP", "SERVICE", "CENTER", "CENTRE"}

_KEYWORD_RULES: dict[str, tuple[str, ...]] = {
    "Food & Dining": (
        "DINING AND DRINKING",
        "RESTAURANT",
        "CAFE",
        "COFFEE",
        "TEA HOUSE",
        "BAKERY",
        "BUBBLE TEA",
        "FOOD AND BEVERAGE RETAIL",
        "FOOD AND BEVERAGE SERVICE",
        "FOOD SERVICE",
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
        "DESSERT",
        "JUICE",
        "BISTRO",
        "STEAKHOUSE",
        "SEAFOOD",
        "NOODLE",
        "SUSHI",
        "DONUT",
        "BAGEL",
        "BREAKFAST",
        "BRUNCH",
        "WINGS",
        "BBQ",
        "RAMEN",
        "POKE",
        "SOUP",
        "FOOD TRUCK",
        "CATERING",
    ),
    "Transportation": (
        "TRANSPORTATION",
        "GAS STATION",
        "FUEL STATION",
        "PARKING",
        "CAR WASH",
        "TRANSIT",
        "AIRPORT",
        "TRAIN STATION",
        "BUS STATION",
        "REST AREA",
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
        "WAREHOUSE",
        "WHOLESALE",
        "THRIFT",
        "VINTAGE",
        "GIFT",
        "PET STORE",
        "SPORTING GOODS",
        "TOY",
        "COSMETICS",
        "MISCELLANEOUS STORE",
    ),
    "Entertainment & Recreation": (
        "ARTS AND ENTERTAINMENT",
        "MOVIE THEATER",
        "MUSEUM",
        "GYM",
        "FITNESS",
        "SPORTS AND RECREATION",
        "RECREATION",
        "ARCADE",
        "NIGHT CLUB",
        "MUSIC VENUE",
        "CASINO",
        "BOWLING",
        "AMUSEMENT",
        "THEME PARK",
        "PERFORMING ARTS",
        "CONCERT",
        "STADIUM",
        "ARENA",
        "GAMING",
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
        "OPTOMETRIST",
        "VETERINARIAN",
        "PHYSIOTHERAPY",
        "CHIROPRACT",
        "MENTAL HEALTH",
    ),
    "Utilities & Services": (
        "UTILITY",
        "TELECOMMUNICATION",
        "INTERNET SERVICE PROVIDER",
        "REPAIR",
        "LAUNDRY",
        "DRY CLEANER",
        "HOME SERVICE",
        "HOME IMPROVEMENT SERVICE",
        "POST OFFICE",
        "SHIPPING",
        "BARBER",
        "SALON",
        "CLEANING SERVICE",
        "CONTRACTOR",
        "PLUMBER",
        "ELECTRICIAN",
        "LOCKSMITH",
        "TAILOR",
        "PRINTING",
        "MOVING",
        "STORAGE",
    ),
    "Financial Services": (
        "FINANCIAL SERVICE",
        "BANK",
        "ATM",
        "INSURANCE",
        "CREDIT UNION",
        "CURRENCY EXCHANGE",
        "LOAN",
        "ACCOUNTING",
        "TAX PREPARATION",
    ),
    "Government & Legal": (
        "GOVERNMENT",
        "LEGAL SERVICE",
        "COURTHOUSE",
        "EMBASSY",
        "EDUCATION",
        "COLLEGE",
        "UNIVERSITY",
        "SCHOOL",
    ),
    "Charity & Donations": (
        "NONPROFIT",
        "SOCIAL SERVICE",
        "RELIGIOUS ORGANIZATION",
        "CHARITY",
        "COMMUNITY CENTER",
        "SPIRITUAL CENTER",
        "CHURCH",
        "MOSQUE",
        "SYNAGOGUE",
        "TEMPLE",
        "PLACE OF WORSHIP",
    ),
}

# automotive service is transportation (repair shops, tire service, etc.)
# but generic "AUTOMOTIVE" in retail context is shopping (car dealership)
_AUTOMOTIVE_TRANSPORT_KEYWORDS: tuple[str, ...] = (
    "AUTOMOTIVE SERVICE",
    "AUTOMOTIVE REPAIR",
    "TIRE",
    "OIL CHANGE",
    "AUTO BODY",
)


def map_foursquare_labels(labels: list[str]) -> tuple[str | None, float]:
    """Map Foursquare category paths into the project's coarse taxonomy.

    Uses per-keyword scoring with specificity weighting so that specific
    category keywords (GROCERY, PHARMACY) outweigh generic structural
    keywords (RETAIL, STORE) when both appear in the same label.
    """
    scores: Counter[str] = Counter()
    for label in labels:
        upper = label.upper()
        depth_bonus = 1.0 + (upper.count(">") * 0.1)

        # check automotive service -> transportation
        for kw in _AUTOMOTIVE_TRANSPORT_KEYWORDS:
            if kw in upper:
                scores["Transportation"] += depth_bonus * 2.0

        for category, keywords in _KEYWORD_RULES.items():
            for keyword in keywords:
                if keyword in upper:
                    weight = 1.0 if keyword in _GENERIC_KEYWORDS else 2.0
                    scores[category] += depth_bonus * weight

    if not scores:
        return None, 0.0

    best_category, best_score = scores.most_common(1)[0]
    total = float(sum(scores.values()))
    confidence = best_score / total if total else 0.0

    if confidence < 0.40:
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

