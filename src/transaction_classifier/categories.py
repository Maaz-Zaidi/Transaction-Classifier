from enum import IntEnum


class Category(IntEnum):
    FOOD_AND_DINING = 0
    TRANSPORTATION = 1
    SHOPPING_AND_RETAIL = 2
    ENTERTAINMENT_AND_RECREATION = 3
    HEALTHCARE_AND_MEDICAL = 4
    UTILITIES_AND_SERVICES = 5
    FINANCIAL_SERVICES = 6
    INCOME = 7
    GOVERNMENT_AND_LEGAL = 8
    CHARITY_AND_DONATIONS = 9


LABEL_MAP: dict[str, Category] = {
    "Food & Dining": Category.FOOD_AND_DINING,
    "Transportation": Category.TRANSPORTATION,
    "Shopping & Retail": Category.SHOPPING_AND_RETAIL,
    "Entertainment & Recreation": Category.ENTERTAINMENT_AND_RECREATION,
    "Healthcare & Medical": Category.HEALTHCARE_AND_MEDICAL,
    "Utilities & Services": Category.UTILITIES_AND_SERVICES,
    "Financial Services": Category.FINANCIAL_SERVICES,
    "Income": Category.INCOME,
    "Government & Legal": Category.GOVERNMENT_AND_LEGAL,
    "Charity & Donations": Category.CHARITY_AND_DONATIONS,
}

# Reverse mapping: enum -> display label
CATEGORY_NAMES: dict[Category, str] = {v: k for k, v in LABEL_MAP.items()}

# Ordered list of all label strings (for SGDClassifier classes= parameter)
ALL_LABELS: list[str] = [CATEGORY_NAMES[Category(i)] for i in range(len(Category))]
