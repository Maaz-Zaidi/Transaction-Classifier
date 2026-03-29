# Phase 2: Training and evaluation results
**Date:** 2026-03-28

## Architecture change

Replaced the Phase 1 cascade (Rules -> SGD -> DistilBERT) with a two-stage setup:

**Stage 1: Direction detection** (deterministic prefix rules)
- Income / Government resolved directly
- Expenses passed to Stage 2

**Stage 2: Rules engine -> FastText classifier**
- Known merchants matched by rules
- Unknown merchants classified by FastText (subword embeddings)

## What changed

### Preprocessing
- Added: `Contactless Interac purchase -`, `Online Banking payment -`, `ATM withdrawal`, `Mobile cheque deposit`, `Client Card Replacement Fee`, `TST-` (Toast POS), store number prefixes (`01339 ...`)
- E-transfer direction: `E-TRANSFER-IN` (autodeposit/received) vs `E-TRANSFER-OUT` (sent)
- Rules updated: `E-TRANSFER-IN` -> Income, `E-TRANSFER-OUT` -> Financial Services

### FastText model
- Trained on mitulshah dataset (3.6M samples, 10 epochs, 125s on CPU)
- Subword character n-grams (3-6), dim=100, word bigrams, softmax loss
- Val accuracy: **99%** (avg confidence 0.987, only 2.9% below threshold)

### Direction detection
- Deterministic prefix matching on raw (un-preprocessed) text
- Handles: Payroll Deposit, Direct Deposit, e-Transfer Autodeposit/Received, GST, Tax Refund, Canada Carbon Rebate

## Results on real bank data (624 descriptions, 497 Gemini-labeled)

Overall progression:
- Phase 1 (SGD): 43.4%
- Phase 1b (pymupdf+SGD): 53.7%
- **Phase 2 (direction+FastText): 55.7%**

Key changes:
- Income: 2.2% -> **97.8%** (direction detection fixed this)
- Government & Legal: 0.0% -> **54.5%**
- Chequing account: 32.5% -> **58.2%**
- MasterCard: 67.3% -> 54.1% (actually got worse)
- Flagged for review: 43.9% -> **14.3%**

By source:
- Direction: **100.0%** (53/53)
- Rules: **91.3%** (189/207)
- FastText: **14.8%** (35/237)

By category:
- Income: 2.2% -> **97.8%**
- Healthcare & Medical: 100.0% (unchanged)
- Entertainment & Recreation: 88.6% (unchanged)
- Transportation: 83.3% (unchanged)
- Shopping & Retail: 59.2% (unchanged)
- Government & Legal: 0.0% -> **54.5%**
- Financial Services: 47.4% -> 35.1% (got worse)
- Food & Dining: 61.5% -> **45.4%** (got worse)
- Utilities & Services: 31.6% (unchanged)
- Charity & Donations: 0.0% (unchanged)

## Analysis

### What worked
1. **Direction detection is the biggest single win.** Income went from 2.2% to 97.8%, Government from 0% to 54.5%. Chequing account (mostly payroll/e-transfers) jumped from 32.5% to 58.2%.
2. **Preprocessing fixes** correctly strip Contactless Interac, Online Banking, TST- prefixes.
3. **Rules accuracy went up** from 84.1% to 91.3% because better preprocessing feeds cleaner text.

### What didn't work
1. **FastText has an "Income" bias** for unknown merchants. It predicts Income at 0.90+ confidence for restaurants like "SUSHI EKI", "RIDEAU CENTRE", "QUICKIE". The training data's Income category has generic patterns ("FRANCHISE STORE", "LICENSING CENTER") whose character n-grams overlap with proper noun merchant names.
2. **MasterCard accuracy dropped** from 67.3% to 54.1% because FastText's Income bias is worse than SGD's random-ish predictions for MasterCard merchants.
3. **Food & Dining dropped** from 61.5% to 45.4%. Most food merchants are local/unknown and FastText classifies them as Income.

### The core problem (still the same)
FastText and SGD share the same fundamental issue: they were trained on synthetic data (mitulshah) that doesn't represent real Canadian bank merchants. Subword embeddings help with truncation but don't help when the model has a strong prior that generic text = Income.

### Next steps
- **SetFit with MiniLM**: pre-trained sentence transformer (all-MiniLM-L6-v2, 22M params) that already has real-world knowledge of food/retail/transport concepts. The idea is that pre-trained embeddings know "shawarma" = food, "depot" = retail, even if the fine-tuning data is synthetic.
- **Real bank data stays test-only**: the 497 Gemini-labeled descriptions are reserved for testing, not training.

## Model registry

- Model 1 (SGD): val 98%, real 53.7% - Phase 1 baseline
- Model 2 (FastText): val 99%, real 55.7% - Phase 2, direction detection + FastText. Income bias on unknown merchants
