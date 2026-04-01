# Transaction Classifier

A local ML service that classifies bank transactions (e.g. "UBER *TRIP 284") into budget categories ("Transportation", "Food & Dining", etc). Everything runs on-device so no transaction data leaves the machine.

## what it does

Takes a raw transaction string from a bank statement and maps it to one of 10 categories:

- **Food & Dining** (restaurants, fast food, groceries, cafes)
- **Transportation** (gas, parking, transit, rideshare)
- **Shopping & Retail** (Amazon, clothing, electronics, furniture)
- **Entertainment & Recreation** (streaming, games, movies, sports)
- **Healthcare & Medical** (pharmacy, dental, hospital)
- **Utilities & Services** (phone, internet, hydro, repairs)
- **Financial Services** (credit payments, e-transfers out, banking fees)
- **Income** (payroll, refunds, deposits, e-transfers in)
- **Government & Legal** (CRA, taxes, fines, licensing)
- **Charity & Donations** (non-profits, religious, fundraisers)

Training data comes from [mitulshah/transaction-categorization](https://huggingface.co/datasets/mitulshah/transaction-categorization) (3.6M records). Evaluated against 497 real labeled transactions from my own RBC statements.

## how the pipeline works

Classification happens in two stages.

**Stage 1: Direction detection** runs on the raw (uncleaned) text. It looks for income patterns (payroll, direct deposit, e-transfer autodeposit) and government patterns (CRA, tax refund). If it matches, it returns immediately with 0.99 confidence. This catches all income and most government transactions before anything else runs.

**Stage 2** runs on cleaned/preprocessed text and has two layers:
- First, a rules engine tries to match against ~120 known merchant patterns (regex and keyword). If a rule hits, it returns with 0.98 confidence. This handles about 91% of expense transactions.
- If no rule matches, the ML model takes over. The ensemble picks the best available model in this order: fine-tuned MiniLM, SetFit, FastText, SGD.

Each result includes the original text, cleaned text, predicted category, confidence score, which source made the call, and whether it's flagged for review.

## models

I trained four models across four phases, each one building on what I learned from the last.

**SGD (v1, baseline)** uses TF-IDF vectors fed into a SGDClassifier. Trains fast on all 3.6M samples and gets 98% on the synthetic validation set, but only 28% on real bank data where rules don't help. The synthetic-to-real domain gap is massive.

**FastText (v2)** uses subword character n-grams, which should help with truncated merchant names. Gets 99% on validation but actually performs worse than SGD on real unknowns (15%). It has a strong bias toward predicting "Income" for anything it hasn't seen.

**SetFit (v3)** uses all-MiniLM-L6-v2 with contrastive learning on 8K stratified samples. This was the breakthrough, jumping real-data accuracy to 80.5%. Pre-trained sentence embeddings already understand concepts like food, retail, and finance, so the model generalizes way better.

**Fine-tuned MiniLM (v4, current best)** takes the same base model but uses standard cross-entropy fine-tuning instead of contrastive learning. Trains 5x faster than SetFit and pushes real-data accuracy to 84.5%. On transactions where only ML is involved (no rules or direction), it hits 75%.

## results on real data

Overall accuracy across 497 real RBC transactions: **84.5%**

Broken down by source:
- Direction detection: 100% (53 transactions)
- Rules engine: 91.3% (207 transactions)
- ML (fine-tune): 75.1% (237 transactions)

Strongest categories are Income (97.8%), Healthcare (100%), and Financial Services (91.2%). Weakest are Government & Legal (54.5%, mostly edge cases with e-transfers) and Charity (too few samples to measure).

## preprocessing

The cleaning pipeline strips out a lot of noise that banks add to transaction strings:

- Bank prefixes like "CONTACTLESS INTERAC PURCHASE-1234" or "VISA DEBIT PURCHASE-1234"
- Location suffixes like "OTTAWA ON" or "TORONTO ON"
- Reference numbers, card numbers, long digit sequences
- Special characters (keeping &, ', ., -, / since those appear in merchant names)

It also detects e-transfer direction (in vs out) and payroll/deposit patterns before cleaning, so that information isn't lost.

Example: `CONTACTLESS INTERAC PURCHASE - 1234 TIM HORTONS KANATA ON` becomes `TIM HORTONS`.

## API

FastAPI service with three endpoints:

- **POST /classify** takes a batch of transaction strings and returns categories, confidence scores, and sources for each one
- **GET /categories** returns the list of 10 valid category labels
- **GET /health** reports model status, version, and rules count

Run it with: `uvicorn src.transaction_classifier.api.app:create_app --factory`

## tech stack

- Python 3.10+
- scikit-learn (TF-IDF, SGD)
- sentence-transformers, setfit, transformers, torch (the neural models)
- fasttext
- FastAPI + uvicorn (API)
- pandas, pyarrow (data processing)
- datasets (HuggingFace data loading)
- pymupdf (PDF text extraction from bank statements)

## project structure

- `src/transaction_classifier/` is the main package (models, rules, preprocessing, API)
- `scripts/` has training and evaluation scripts for each model phase
- `Research Notes/` has per-phase evaluation results and findings
- `Research Documents/` has background research and architecture notes
- `tests/` has unit tests for preprocessing, rules, ensemble, API
- `data/` (gitignored) holds raw datasets, processed splits, and real labeled data
- `models/` (gitignored) holds trained model artifacts
- `checkpoints/` (gitignored) holds training checkpoints

## deliverables

- **Classification service** (done): a Python API that takes transaction strings in batch and returns categories with confidence scores, optimized to run on modest hardware
- **Self-correction UI**: a minimal interface to view and correct classifications, feeding corrections back into retraining (placeholder exists, not yet built)
- More rules to close gaps on known merchants
- Better handling of ambiguous categories (e.g. e-transfers that could be anything)
