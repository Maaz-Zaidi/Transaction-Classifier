"""split cleaned transaction text into brand, descriptor, location, and noise tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transaction_classifier.knowledge.merchant_kb import MerchantKnowledgeBase


@dataclass(frozen=True, slots=True)
class TokenRole:
    token: str
    role: str  # brand | descriptor | location | noise
    descriptor_hint: str | None = None


@dataclass(frozen=True, slots=True)
class DecomposedQuery:
    original: str
    token_roles: list[TokenRole]
    brand_tokens: list[str]
    descriptor_tokens: list[str]
    location_tokens: list[str]
    noise_tokens: list[str]
    brand_query: str
    descriptor_context: str


# descriptor words that add category hints
_DESCRIPTOR_WORDS: dict[str, str] = {
    "GAS": "gas station, fuel",
    "GASOLINE": "gas station, fuel",
    "FUEL": "gas station, fuel",
    "WHOLESALE": "wholesale warehouse shopping",
    "GROCERY": "grocery food retail",
    "GROCERIES": "grocery food retail",
    "SUPERMARKET": "supermarket grocery",
    "RESTAURANT": "dining restaurant",
    "CAFE": "cafe coffee shop",
    "COFFEE": "coffee shop cafe",
    "BAKERY": "bakery baked goods",
    "PIZZA": "pizza restaurant",
    "GRILL": "grill restaurant dining",
    "SHAWARMA": "shawarma restaurant dining",
    "SUSHI": "sushi restaurant dining",
    "CHICKEN": "chicken restaurant dining",
    "BURRITO": "burrito restaurant dining",
    "DOWNLOADS": "digital content downloads",
    "DOWNLOAD": "digital content downloads",
    "PRIME": "subscription membership service",
    "TUITION": "education tuition fees",
    "INSURANCE": "insurance coverage",
    "ELECTRONICS": "electronics retail",
    "CLOTHING": "clothing apparel retail",
    "FURNITURE": "furniture home retail",
    "HARDWARE": "hardware tools retail",
    "BOOKS": "bookstore retail",
    "SHIPPING": "shipping delivery service",
    "LAUNDRY": "laundry cleaning service",
    "BARBER": "barbershop hair service",
    "SALON": "salon beauty service",
    "VENDING": "vending machine snacks",
    "CONV.": "convenience store snacks",
    "CONV": "convenience store snacks",
    "CONVENIENCE": "convenience store snacks",
}

# multi-word descriptors, checked before single tokens
_MULTI_TOKEN_DESCRIPTORS: dict[str, str] = {
    "WEB SERVICES": "cloud computing, technology services",
    "WEB SERV": "cloud computing, technology services",
    "DRUG MART": "pharmacy drugstore",
    "GAS BAR": "gas station, fuel",
    "COIN WASH": "laundry cleaning service",
    "FINE FOODS": "specialty food retail",
}

# common canadian cities and neighbourhoods that show up in statements
_CANADIAN_LOCATIONS: set[str] = {
    "OTTAWA", "TORONTO", "MONTREAL", "VANCOUVER", "CALGARY", "EDMONTON",
    "WINNIPEG", "QUEBEC", "HAMILTON", "KITCHENER", "LONDON", "VICTORIA",
    "HALIFAX", "REGINA", "SASKATOON", "BARRIE", "GUELPH", "KINGSTON",
    "MISSISSAUGA", "BRAMPTON", "MARKHAM", "RICHMOND", "SURREY", "BURNABY",
    "KANATA", "NEPEAN", "ORLEANS", "BARRHAVEN", "STITTSVILLE", "GLOUCESTER",
    "GATINEAU", "HULL", "LAVAL", "LONGUEUIL", "SCARBOROUGH", "ETOBICOKE",
    "BAYSHORE", "RIDEAU", "DOWNTOWN", "MIDTOWN", "UPTOWN",
    "HALTON", "OAKVILLE", "BURLINGTON", "OSHAWA", "WHITBY", "AJAX",
    "PICKERING", "NEWMARKET", "VAUGHAN", "THORNHILL", "RICHMOND HILL",
    "CAMPBELL", "STITTSVILL",
}

# noise patterns
_NOISE_RE = re.compile(r"^[A-Z]?\d+$|^\d+$|^[A-Z]{1,2}$")
_BRANCH_RE = re.compile(r"^[A-Z]\d{2,}$|^#\d+$|^S\d+$|^W\d+$|^P\d+$")


class TokenAnalyzer:
    """split cleaned text into token roles."""

    def analyze(
        self,
        cleaned_text: str,
        kb: MerchantKnowledgeBase | None = None,
    ) -> DecomposedQuery:
        if not cleaned_text or not cleaned_text.strip():
            return DecomposedQuery(
                original=cleaned_text,
                token_roles=[],
                brand_tokens=[],
                descriptor_tokens=[],
                location_tokens=[],
                noise_tokens=[],
                brand_query="",
                descriptor_context="",
            )

        upper = cleaned_text.upper().strip()
        raw_tokens = upper.split()

        # check multi-word descriptors first and mark those token positions
        multi_consumed: set[int] = set()
        multi_descriptors: list[tuple[str, str]] = []  # (matched_text, hint)

        for phrase, hint in _MULTI_TOKEN_DESCRIPTORS.items():
            phrase_upper = phrase.upper()
            if phrase_upper in upper:
                phrase_tokens = phrase_upper.split()
                for start in range(len(raw_tokens) - len(phrase_tokens) + 1):
                    window = raw_tokens[start : start + len(phrase_tokens)]
                    if window == phrase_tokens:
                        for j in range(start, start + len(phrase_tokens)):
                            multi_consumed.add(j)
                        multi_descriptors.append((phrase_upper, hint))

        # check multi-word brand matches against kb aliases
        multi_brand_consumed: set[int] = set()
        if kb is not None and kb.is_loaded:
            for window_size in (3, 2):
                for start in range(len(raw_tokens) - window_size + 1):
                    if any(j in multi_consumed or j in multi_brand_consumed
                           for j in range(start, start + window_size)):
                        continue
                    candidate = " ".join(raw_tokens[start : start + window_size])
                    if kb._lookup_exact_candidate_ids(candidate):
                        for j in range(start, start + window_size):
                            multi_brand_consumed.add(j)

        # classify the remaining tokens
        roles: list[TokenRole] = []
        brand_tokens: list[str] = []
        descriptor_tokens: list[str] = []
        location_tokens: list[str] = []
        noise_tokens: list[str] = []

        for i, token in enumerate(raw_tokens):
            if i in multi_consumed:
                continue
            if i in multi_brand_consumed:
                roles.append(TokenRole(token=token, role="brand"))
                brand_tokens.append(token)
                continue

            # noise
            if _NOISE_RE.match(token) or _BRANCH_RE.match(token):
                roles.append(TokenRole(token=token, role="noise"))
                noise_tokens.append(token)
                continue

            # location
            if token in _CANADIAN_LOCATIONS:
                roles.append(TokenRole(token=token, role="location"))
                location_tokens.append(token)
                continue

            # single-word descriptor
            # try the token as-is, then without trailing punctuation
            hint = _DESCRIPTOR_WORDS.get(token)
            if hint is None:
                match_token = token.rstrip(".'")
                hint = _DESCRIPTOR_WORDS.get(match_token)
            if hint is not None:
                roles.append(TokenRole(token=token, role="descriptor", descriptor_hint=hint))
                descriptor_tokens.append(token)
                continue

            # single-word kb brand match
            if kb is not None and kb.is_loaded and len(token) >= 3:
                exact_ids = kb._lookup_exact_candidate_ids(token)
                if exact_ids:
                    roles.append(TokenRole(token=token, role="brand"))
                    brand_tokens.append(token)
                    continue

            # otherwise treat it as part of the brand
            roles.append(TokenRole(token=token, role="brand"))
            brand_tokens.append(token)

        # add the multi-word descriptors back into the result
        for phrase, hint in multi_descriptors:
            for part in phrase.split():
                roles.append(TokenRole(token=part, role="descriptor", descriptor_hint=hint))
            descriptor_tokens.extend(phrase.split())

        # build the brand-only query
        brand_query = " ".join(brand_tokens) if brand_tokens else cleaned_text

        # build descriptor context from the collected hints
        seen_hints: list[str] = []
        for phrase, hint in multi_descriptors:
            if hint not in seen_hints:
                seen_hints.append(hint)
        for role in roles:
            if role.role == "descriptor" and role.descriptor_hint and role.descriptor_hint not in seen_hints:
                seen_hints.append(role.descriptor_hint)
        descriptor_context = "; ".join(seen_hints)

        return DecomposedQuery(
            original=cleaned_text,
            token_roles=roles,
            brand_tokens=brand_tokens,
            descriptor_tokens=descriptor_tokens,
            location_tokens=location_tokens,
            noise_tokens=noise_tokens,
            brand_query=brand_query,
            descriptor_context=descriptor_context,
        )
