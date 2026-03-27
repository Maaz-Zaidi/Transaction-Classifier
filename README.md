# Transaction Classifier

## Abstract:
Once data is digital (either from DocuParse or bank exports), it needs to be organized. FinClass is a local machine learning service that classifies bank transactions (e.g., "UBER *TRIP 284") into standardized budget categories ("Transport", "Groceries") with high accuracy. This ensures that user data remains private and is not sent to third-party cloud aggregators.

# Architecture Overview

1. Data Pipeline:
- Ingests a statement label (e.g.,  HOT CRISPY CHICKEN OTTAWA ON, PARKING PPL TORONTO ON, AMAZON.CA*XXXXXXXXX AMAZON.CA ON, etc.,) and categorizes it into one of 10 categories:

    1. Food & Dining (restaurants, fast food, cafes, bakeries, groceries ...)
    2. Transportation (Gas, Parking, OC Transpo, Uber, Presto ...)
    3. Shopping & Retail (clothing, furniture, malls, electronics, Amazon ...)
    4. Entertainment & Recreation (movies, games, cinema, concerts, sports ...)
    5. Healthcare & Medical (pharmacy, dental, vision, hospital ...)
    6. Utilities & Services (phone, internet, hydro, water, barber, repairs ...)
    7. Financial Services (credit payments, savings, loans, banking fees ...)
    8. Income (payroll, refunds, deposits, e-transfers in ...)
    9. Government & Legal (taxes, fines, licensing, legal fees ...)
    10. Charity & Donations (non-profits, religious, fundraisers ...)

- Dataset: [mitulshah/transaction-categorization](https://huggingface.co/datasets/mitulshah/transaction-categorization) (4.5M records, includes Canadian merchants)
- Range: Canada-focused (Canadian merchants, bank string formats, currency)

2. Classification Model (Hybrid):
- Layer 1: Rules engine — regex/keyword matching for known merchants (~98% precision)
- Layer 2: TF-IDF + SGDClassifier — fast ML fallback with incremental learning support
- Layer 3: Fine-tuned DistilBERT — transformer model for harder cases

3. Active Learning Loop:
- User corrections stored and used for periodic retraining (Reach)

# Tech Stack
- Python 3.10+
- scikit-learn (TF-IDF, SGDClassifier)
- HuggingFace Transformers (DistilBERT)
- SQLite (correction storage)
- FastAPI (classification service API)

# Deliverables
- Classification Service: A Python API that takes transaction strings (batch) and returns categories with confidence scores. Optimized to run on modest hardware.
- Self-Correction UI: A minimal interface to view and correct classifications to improve the model. (Reach)