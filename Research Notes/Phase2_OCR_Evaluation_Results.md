# Phase 2: OCR re-evaluation results
**Date:** 2026-03-28

## Context

Phase 1 showed 43.4% accuracy on real data, with pdfplumber's word-joining being the main blocker. This re-tests after switching PDF extraction from pdfplumber to pymupdf (fitz), which actually preserves word boundaries in RBC PDFs.

## Extraction

- **Tool:** pymupdf (fitz), native PDF text extraction, no OCR needed
- **Why it works:** pymupdf uses a different algorithm than pdfplumber to infer spaces from character positions. Handles RBC PDFs correctly where pdfplumber failed.
- **Script:** `scripts/extract_descriptions_ocr.py`
- **Total extracted:** 624 descriptions (430 MasterCard, 194 Chequing), 241 unique
- **Ground truth:** Gemini-labeled categories (497 of 624 labeled successfully)

## Results

Overall comparison (pdfplumber vs pymupdf):
- Overall accuracy: 43.4% -> **53.7%** (+10.3%)
- Rules accuracy: 77.2% -> **84.1%** (190/226, +6.9%)
- SGD accuracy: 22.3% -> **28.4%** (77/271, +6.1%)
- Flagged for review: 51.9% -> **43.9%** (218/497, -8.0%)

By account type:
- MasterCard: 58.2% -> **67.3%** (204/303, +9.1%)
- Chequing: 32.8% -> 32.5% (63/194, basically flat)

By category:
- Healthcare & Medical: 100.0% (13/13)
- Entertainment & Recreation: 88.6% (31/35)
- Transportation: 83.3% (15/18)
- Food & Dining: 61.5% (126/205)
- Shopping & Retail: 59.2% (42/71)
- Financial Services: 47.4% (27/57)
- Utilities & Services: 31.6% (12/38)
- Income: **2.2%** (1/46)
- Government & Legal: **0.0%** (0/11)
- Charity & Donations: **0.0%** (0/3)

## Analysis

### What improved

Fixing word spacing helped MasterCard a lot (+9.1%). Merchants like `TIM HORTONS`, `SHOPPERS DRUG MART`, `T&T SUPERMARKET` now get correctly parsed and matched by rules. Rules accuracy jumped from 77.2% to 84.1%.

### What didn't improve

Chequing accuracy stayed flat (32.8% -> 32.5%). Chequing failures aren't spacing-related, they're caused by:
1. Preprocessing gaps (prefixes not stripped)
2. SGD can't classify unknown merchants
3. Direction-dependent categories (e-transfers, payroll)

### The real problem: SGD can't generalize

The SGD model (TF-IDF char n-grams trained on synthetic US data) has a fundamental domain mismatch:
- **Vocabulary overlap:** only 19.7% of real-world words exist in training data
- **Format mismatch:** training = "Taco Bell Store TXN208382", real = "Visa Debit purchase - 4471 HOT CRISPY CHIC"
- **Distribution mismatch:** training is balanced 10% per category, real data = 41% Food & Dining
- **Confidently wrong:** "RBC CREDIT CARD" -> Income at 0.86 confidence (should be Financial Services)

### Top failure patterns

- Payroll Deposit not classified as Income (~20 cases): SGD doesn't know ERICSSON = employer
- E-Transfer autodeposit not classified as Income (~15 cases): rules treat all e-transfers as Financial Services
- Contactless Interac prefix not stripped (~15 cases): missing preprocessing pattern
- Local restaurants getting wrong category (~30 cases): SGD never saw these merchants
- OPENAI, Kindle, AWS wrong category (~20 cases): not in training vocabulary
- Online Banking payment prefix not stripped (~5 cases): missing preprocessing pattern

## Conclusion

Fixing PDF extraction was necessary but not enough. The +10.3% gain confirms spacing was a real problem, but the remaining 46.3% error rate comes from the SGD model not being able to generalize to unseen Canadian merchants. Adding rules for individual merchants would be a band-aid. The real fix needs either:
1. A model with semantic understanding (FastText subword embeddings, sentence transformers)
2. A two-stage architecture that separates direction detection from merchant classification
3. Training data augmentation with real bank patterns
