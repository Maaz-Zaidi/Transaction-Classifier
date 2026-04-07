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

Training data comes from [mitulshah/transaction-categorization](https://huggingface.co/datasets/mitulshah/transaction-categorization) (3.6M records). Merchant knowledge comes from [Foursquare OS Places](https://huggingface.co/datasets/foursquare/fsq-os-places) (1.8M Canadian entries). Evaluated against 505 real labeled transactions from RBC statements spanning 2019–2026.

## how the pipeline works

Classification happens in stages, each one more expensive than the last. Most transactions resolve early and never reach the ML model.

**Stage 1: Direction detection** runs on the raw (uncleaned) text. It looks for income patterns (payroll, direct deposit, e-transfer autodeposit) and government patterns (CRA, tax refund). If it matches, it returns immediately with 0.99 confidence. This catches all income and most government transactions before anything else runs.

**Stage 2: Rules engine** runs on cleaned text and tries to match against ~25 structural patterns (MORTGAGE, PARKING, PHARMACY, SERVICE CHARGE, etc). These are transaction-type patterns, not merchant names. If a rule hits, it returns with 0.98 confidence.

**Stage 3: Merchant knowledge base** runs a retrieve-and-rerank pipeline against a local store of 1.8M Canadian merchants built from Foursquare OS Places:
1. Exact alias lookup on full cleaned text and brand-only query
2. FTS5 lexical search (brand + descriptor tokens only, noise/location excluded)
3. Dense retrieval via ChromaDB using BAAI/bge-small-en-v1.5 embeddings
4. Candidate fusion (RRF) of lexical and dense results
5. Cross-encoder reranking with BAAI/bge-reranker-v2-m3
6. If the top reranked candidate is strong enough, its mapped category is returned directly. If plausible but not decisive, its metadata enriches the text before ML.

A token analyzer decomposes each transaction string into brand, descriptor, location, and noise tokens before retrieval. Descriptor words like GAS, PHARMACY, and WHOLESALE inject category hints into the ML input even when the KB can't find the merchant.

**Stage 4: ML model** takes over for anything the KB can't resolve. The ensemble picks the best available model: fine-tuned MiniLM (enriched), SetFit, FastText, SGD. The primary model (fine-tuned all-MiniLM-L6-v2) was continuation-trained on metadata-enriched inputs so it can parse Foursquare place types and descriptor context appended by the KB layer.

Each result includes the original text, cleaned text, predicted category, confidence score, which source made the call, and whether it's flagged for review.

## models

I trained six models across twelve phases, each one building on what I learned from the last.

**SGD (v1, baseline)** uses TF-IDF vectors fed into a SGDClassifier. Trains fast on all 3.6M samples and gets 98% on the synthetic validation set, but only 28% on real bank data where rules don't help. The synthetic-to-real domain gap is massive.

**FastText (v2)** uses subword character n-grams, which should help with truncated merchant names. Gets 99% on validation but actually performs worse than SGD on real unknowns (15%). It has a strong bias toward predicting "Income" for anything it hasn't seen.

**SetFit (v3)** uses all-MiniLM-L6-v2 with contrastive learning on 8K stratified samples. This was the breakthrough, jumping real-data accuracy to 80.5%. Pre-trained sentence embeddings already understand concepts like food, retail, and finance, so the model generalizes way better.

**Fine-tuned MiniLM (v4)** takes the same base model but uses standard cross-entropy fine-tuning instead of contrastive learning. Trains 5x faster than SetFit and pushes real-data accuracy to 84.5% on the early eval set.

**Augmented MiniLM (v5) and CANINE (v6)** were experiments in abbreviation augmentation and character-level modeling. Both regressed badly. The problem was not tokenization or abbreviation handling — it was entity knowledge. This failure motivated the pivot to external merchant knowledge.

**Enriched MiniLM (v7, current best)** continues from the v4 checkpoint with additional training on metadata-enriched inputs. 20K samples, 50% enriched with real Foursquare label paths, learning rate 5e-6, 2 epochs. The metadata path accuracy jumped from 68.6% to 77.1%.

## results on real data

Overall accuracy across 505 unique real RBC transactions (3,113 weighted occurrences): **73.9% unique / 83.6% weighted**

Broken down by source:
- Direction detection: 100.0% (9 transactions)
- Rules engine: 92.3% (13 transactions)
- Knowledge base direct: 94.3% (175 transactions)
- ML + metadata enrichment: 77.1% (153 transactions)
- ML only: 44.5% (155 transactions)

Strongest categories are Income (100%), Healthcare (100%), and Food & Dining (89.3%). Weakest are Government & Legal (37.5%) and Utilities & Services (40.0%), mostly due to missing training data for Canadian-specific patterns.

The ML model is the current bottleneck. Deep analysis showed the training dataset has only 847 unique base merchant names despite 3.6M rows — the rest is template noise. 97% of test merchants don't appear in training. The path forward is diverse synthetic training data generation and distant supervision from KB entries.

## preprocessing

The cleaning pipeline strips out a lot of noise that banks add to transaction strings:

- Bank prefixes like "CONTACTLESS INTERAC PURCHASE-1234" or "VISA DEBIT PURCHASE-1234"
- Location suffixes like "OTTAWA ON" or "TORONTO ON"
- Reference numbers, card numbers, long digit sequences
- Special characters (keeping &, ', ., -, / since those appear in merchant names)
- Refund prefixes (strips "REFUND -" so the merchant name routes to the KB)
- Toast POS prefixes ("TST-"), store number prefixes

It also detects e-transfer direction (in vs out) and payroll/deposit patterns before cleaning, so that information isn't lost.

Example: `CONTACTLESS INTERAC PURCHASE - 1234 TIM HORTONS KANATA ON` becomes `TIM HORTONS`.

## setup

```bash
# clone and install
git clone <repo-url>
cd 1A01_Transaction_Classifier
python -m venv .venv
source .venv/bin/activate
pip install -e ".[bert,retrieval,dev]"
```

Requires Python 3.10+. The `bert` extra pulls in transformers, torch, and accelerate. The `retrieval` extra is needed for the dense retrieval pipeline. The `dev` extra adds pytest, httpx, and ruff.

## ingestion

There are three things to ingest before the pipeline works end-to-end: the training dataset, the merchant knowledge store, and the trained models.

### 1. download and split the training data

```bash
python scripts/download_data.py
```

Downloads the mitulshah/transaction-categorization dataset from HuggingFace (gated, requires a HF token) and creates train/val/test splits under `data/processed/`.

### 2. build the merchant knowledge store

```bash
python scripts/build_full_merchant_knowledge_store.py
```

Streams the Canada subset of Foursquare OS Places from HuggingFace (gated, requires access), builds a SQLite store with 1.8M entries and FTS5 indexes, and populates a ChromaDB collection with dense embeddings. Also merges in curated public merchants.

Creates three artifacts under `data/external_kb/`:
- `merchant_knowledge.sqlite3` (~2.4 GB)
- `merchant_knowledge_base.json`
- `chroma/` directory (~7.1 GB)

This takes a while. Expect 30–60 minutes depending on network and hardware.

### 3. train the models

Train in this order (each one is independent, but the enriched model depends on the baseline checkpoint):

```bash
# baseline fine-tuned MiniLM
python scripts/train_finetune.py

# enriched fine-tuned MiniLM (continues from baseline checkpoint)
python scripts/train_finetune_enriched.py
```

Optional earlier models (not required for the current pipeline):

```bash
python scripts/train_sgd.py
python scripts/train_fasttext.py
python scripts/train_setfit.py
```

Trained models are saved under `models/`. The enriched model lands in `models/finetune_enriched/`.

### 4. rebuild KB categories (if needed)

If you change the Foursquare category mapper:

```bash
python scripts/rebuild_kb_categories.py
```

Updates mapped categories in-place in the SQLite store without a full rebuild.

## running the API

```bash
uvicorn src.transaction_classifier.api.app:create_app --factory
```

Three endpoints:

- **POST /classify** takes a batch of transaction strings and returns categories, confidence scores, and sources for each one
- **GET /categories** returns the list of 10 valid category labels
- **GET /health** reports model status, version, and rules count

## running tests

```bash
pytest
```

153 tests covering preprocessing, rules, token analysis, merchant KB, ensemble logic, and the API.

## evaluation

```bash
# evaluate against the openai-labeled real data benchmark
python scripts/evaluate_phase6.py
```

Evaluates the full pipeline (direction + rules + KB + ML) against `data/real/openai_labeled.csv` (505 unique transactions). Reports accuracy by source, by category, confidence distributions, and error analysis.

## pre-built artifacts not included

> [!Note]
> The trained models (`models/`), knowledge store artifacts (`data/external_kb/`), and evaluation data (`data/real/`) are **not included in this repo** due to size (~12 GB total). The ingestion and training scripts above will rebuild everything from scratch, but that requires HuggingFace access to both gated datasets and takes 1–2 hours.

> **If you'd like access to the pre-built knowledge store, trained model checkpoints, or the labeled evaluation data, contact me directly.** I'm happy to share them.

## tech stack

- Python 3.10+
- scikit-learn (TF-IDF, SGD)
- sentence-transformers, setfit, transformers, torch (the neural models)
- fasttext
- chromadb (local vector store for merchant retrieval)
- BAAI/bge-small-en-v1.5 (dense retrieval embeddings)
- BAAI/bge-reranker-v2-m3 (cross-encoder reranking)
- FastAPI + uvicorn (API)
- pandas, pyarrow (data processing)
- datasets (HuggingFace data loading)
- pymupdf (PDF text extraction from bank statements)
- pydantic-settings (configuration)
- SQLite + FTS5 (merchant knowledge store and lexical search)

## project structure

- `src/transaction_classifier/` is the main package
  - `models/` — ensemble, direction detection, SGD, FastText, SetFit, fine-tune, CANINE, zero-shot
  - `knowledge/` — merchant KB, Foursquare mapper, curated merchants, retrieval (dense + lexical + rerank), token analyzer
  - `rules/` — YAML-based rules engine (structural patterns only)
  - `data/` — preprocessing, download, splitting, augmentation, enrichment
  - `api/` — FastAPI app and endpoints
  - `config.py` — all pipeline settings (thresholds, model paths, retrieval params)
- `scripts/` has training, evaluation, ingestion, and data processing scripts
- `Research Notes/` has per-phase evaluation results and findings
- `Research Documents/` has background research and architecture notes
- `tests/` has unit tests for preprocessing, rules, token analysis, merchant KB, ensemble, API
- `data/` (gitignored) holds raw datasets, processed splits, real labeled data, and the external KB
- `models/` (gitignored) holds trained model artifacts
- `checkpoints/` (gitignored) holds training checkpoints

## deliverables

- **Classification service** (done): a Python API that takes transaction strings in batch and returns categories with confidence scores, running fully on-device
- **Merchant knowledge store** (done): 1.8M Canadian merchants with dense retrieval, lexical search, cross-encoder reranking, and token-aware query decomposition
- **Self-correction UI**: a minimal interface to view and correct classifications, feeding corrections back into retraining (placeholder exists, not yet built)
- Better ML accuracy through diverse synthetic training data and distant supervision from KB entries
- Multi-institution support beyond RBC
