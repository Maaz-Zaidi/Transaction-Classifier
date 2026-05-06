# Phase 2: Architecture research
**Date:** 2026-03-25

## The problem

The current TF-IDF + SGDClassifier gets 98% on synthetic test data but only 28.4% on real Canadian bank transactions. The model has no semantic understanding. It can't figure out that "SHAWARMA PALACE" is food or that "HOT CRISPY CHIC" (truncated "CHICKEN") is a restaurant. Adding rules for every merchant doesn't scale.

I need a model that:
- Has contextual/semantic understanding of English words
- Handles truncated merchant names (bank descriptions often get cut at ~25 chars)
- Runs on consumer hardware (RTX 3060 12GB) or CPU
- Stays lightweight (no 1B+ parameter models)
- Generalizes to merchants never seen in training

## Research findings

### 1. Two-stage architecture (industry standard)

Published research and production systems (Plaid, Firefly III, Swedbank) all use the same pattern.

**Stage 1: Direction detection (deterministic rules)**

Detect if a transaction is income or expense from bank prefixes. These are standardized by Payments Canada (CPA codes) and highly reliable:
- Payroll Deposit -> Income
- e-Transfer Autodeposit/received -> Income
- GST CANADA / Tax Refund / Canada Carbon Rebate -> Government & Legal
- Student Loan CANADA -> Financial Services
- Visa Debit purchase -> Expense (classify merchant)
- Contactless Interac purchase -> Expense (classify merchant)
- Misc Payment -> Expense (classify merchant)
- Online Banking payment -> Expense (classify merchant)
- e-Transfer sent -> Financial Services

This alone fixes the Income category (currently 2.2% accuracy) and Government & Legal (currently 0.0%).

**Stage 2: Merchant classification (semantic ML)**

For expense transactions, classify the cleaned merchant name into one of the remaining categories using a model with semantic understanding.

**Sources:**
- Plaid Transactions API uses direction as a first-class feature before categorization
- arXiv:2305.18430 encodes direction in the feature vector
- arXiv:2404.08664 two-stage combining similarity detection + SVM
- arXiv:2312.07730 hierarchical taxonomy with macro/micro categories
- Firefly III resolves transaction type (withdrawal/deposit/transfer) before any categorization

### 2. Candidate models for Stage 2

#### A. FastText supervised (recommended for phase 2)

Facebook's FastText uses subword character n-grams (length 3-6) to build word embeddings. Unlike Word2Vec/GloVe, it can represent words never seen in training by composing their character n-grams.

**Why it fits:**
- "CHIC" decomposes to n-grams `<CH, CHI, HIC, IC>` which overlap heavily with "CHICKEN" `<CH, CHI, HIC, ICK, CKE, KEN, EN>`. The model infers similarity from shared subwords.
- Pre-trained on Common Crawl, so it already knows "shawarma" = food, "depot" = retail, "pharmacy" = health.
- Supervised mode trains directly on labeled categories, learning which n-grams matter.

**Specs:**
- Model size: 2-50MB (supervised), up to 7GB pre-trained (compressible to 21MB)
- Inference: 500K+ samples/sec on CPU
- Training: minutes on CPU for millions of records
- Expected accuracy: 78-86% on real data

**Key finding:** arXiv:2305.18430 found domain-specific FastText embeddings (trained on a transaction corpus) outperformed character CNNs, off-the-shelf embeddings, and other approaches for bank transaction classification.

#### B. SetFit (higher accuracy alternative)

Few-shot fine-tuning framework for sentence transformers (all-MiniLM-L6-v2, 22M params). Uses contrastive learning to adapt the embedding space, then trains a classification head.

**Why it's interesting:**
- With only 8 labeled examples per class, matches RoBERTa-Large fine-tuned on 3,000 examples
- I have 4.5M labeled records, so results should be very strong
- Training takes ~30 seconds on GPU

**Specs:**
- Model size: ~80MB
- Inference: ~14K samples/sec on CPU
- Expected accuracy: 88-93%

#### C. DistilBERT fine-tuned (original phase 2 plan, dropped)

66M parameter distilled BERT, fine-tuned on transactions.

**Why I'm not using it:**
- 250MB model, 2-3K inferences/sec on CPU (35x slower than FastText)
- 5-15 hours training time on RTX 3060 for 4.5M records
- Comparable accuracy to SetFit (88-94%) but much heavier
- Was designed as a fallback for SGD, but if I replace SGD with a semantically-aware model, the fallback is unnecessary

#### D. Zero-shot classification (rejected)

BART-large-mnli or DeBERTa for NLI-based zero-shot classification.

**Why rejected:**
- I have 4.5M labeled records, zero-shot solves the wrong problem
- 10 forward passes per sample (one per category label)
- ~50-200 samples/sec, way too slow
- 60-72% accuracy without training

#### E. GloVe/Word2Vec averaging (rejected)

Average pre-trained word vectors as transaction features.

**Why rejected:**
- Word-level only, "CHIC" is completely out of vocabulary
- Strictly worse than FastText for this task
- TF-IDF actually outperforms it on short text in most benchmarks

### 3. Quick comparison

- Current TF-IDF+SGD: ~12MB, 100K+/sec, 28% real accuracy, partial truncation handling, trains in 60s
- **FastText supervised: 2-50MB, 500K+/sec, 78-86% expected, excellent truncation handling, trains in minutes**
- SetFit (MiniLM): ~80MB, 14K/sec, 88-93% expected, good truncation handling, 30s on GPU
- DistilBERT: ~250MB, 2-3K/sec, 88-94% expected, good truncation handling, 5-15h on GPU
- Zero-shot (BART): 1.6GB, 50/sec, 60-72% expected, good truncation handling, no training needed

## Chosen architecture

1. Input comes in (e.g., "Payroll Deposit Ericsson Canada")
2. **Stage 1: Direction detection** (deterministic prefix rules) - "Payroll Deposit" -> Income, strip prefix -> "Ericsson Canada"
3. If Income/Gov -> done. If Expense -> Stage 2
4. **Stage 2: FastText classifier** (semantic subword embeddings) - "HOT CRISPY CHIC" -> subwords -> Food & Dining, confidence check, flag if below threshold
5. Output: category + confidence + source

**Why FastText over SetFit/DistilBERT:**
1. Fastest to implement and iterate on
2. The most relevant paper (arXiv:2305.18430) found it was the best approach for this exact problem
3. Subword n-grams directly handle truncation, which is the core challenge
4. 500K+/sec inference, trains in minutes, lets me experiment faster as well
5. If accuracy isn't enough, I can upgrade to SetFit (same pipeline, just swap the embedding layer)

**What changes from Phase 1:**
- SGDClassifier replaced by FastText as primary ML model
- DistilBERT cascade removed (unnecessary with a better primary model)
- Direction detection added as Stage 1 (rules-based, deterministic)
- Preprocessing updated to handle all known Canadian bank prefixes
- Model versioning added (metadata + metrics saved with each training run)

**New flow:** Stage 1 direction rules -> Stage 2 FastText -> flag for review

(vs old: Rules -> SGD -> DistilBERT -> flag for review)
