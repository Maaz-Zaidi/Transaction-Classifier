# Phase 1: Real bank data results
**Date:** 2026-03-27

## Dataset
- **Source:** 20 RBC bank statement PDFs (7 MasterCard, 13 Chequing), 2024-2026
- **Extraction:** pdfplumber text -> Gemini CLI for JSON structuring
- **Ground truth:** Gemini-labeled categories (320 of 370 labeled successfully)
- **Total transactions:** 370 (184 MasterCard, 186 Chequing)

## Results

- Overall accuracy: **43.4%** (was 95.7% on synthetic data, so a 52.3% drop)
- Rules accuracy: 77.2% (down from 80.5%, only -3.3%)
- SGD accuracy: **22.3%** (down from 98.3%, massive -76.0% drop)
- Flagged for review: **51.9%** (up from 6.2%)

By account type:
- MasterCard: **58.2%** (78/134)
- Chequing: **32.8%** (61/186)

## What went wrong

### 1. PDF space stripping (the biggest issue)

pdfplumber strips spaces in many RBC PDFs:
- `TIMHORTONS #1664KANATAON` instead of `TIM HORTONS #1664 KANATA ON`
- `SHOPPERSDRUGMART 631OTTAWAON` instead of `SHOPPERS DRUG MART 631 OTTAWA ON`
- `CONTACTLESSINTERACPURCHASE-4005 HOTCRISPYCHIC` instead of `Contactless Interac purchase - 4005 HOT CRISPY CHIC`

This breaks preprocessing (can't strip prefixes or locations), and the SGD model has never seen joined-word patterns.

About 40% of MasterCard descriptions and 30% of chequing descriptions are affected.

### 2. Unknown merchant names

The SGD model was trained on mitulshah synthetic data which has a completely different merchant vocabulary:
- `RCSS SOUTH KANATA` (Real Canadian Superstore) - model has no idea this is food
- `RIDEAU BOURBON ST. GRI` (a restaurant) - not in training data
- `TST-TAHINIS` (Toast POS prefix + restaurant name) - model sees "TST-" as noise
- `Z3 SPECIALTY COFFE` - model has no clue
- `SHAWARMA PALACE`, `SHAWARMA PRINCE` - not in training data at all

### 3. Missing preprocessing patterns

New prefixes found in real data that I didn't account for:
- `ContactlessInteracpurchase-` / `Contactless Interac purchase -`
- `OnlineBankingpayment-` / `Online Banking payment -`
- `StudentLoan`
- `e-Transferreceived`

### 4. Category disagreements (model vs Gemini)

Some mismatches are honestly debatable, Gemini's labels aren't always right either:
- E-transfers labeled "Income" by Gemini (autodeposits) vs "Financial Services" by rules
- E-transfers sent to shops (e.g., "e-Transfersent Auto Shop Shaheen") labeled "Transportation" by Gemini
- "Amazon Web Services" labeled "Utilities & Services" by Gemini vs "Shopping & Retail" by rules
- "OPENAI*CHATGPT" labeled "Utilities & Services" by Gemini, could go either way
- "Kindle Unltd" labeled inconsistently by Gemini between "Entertainment & Recreation" and "Shopping & Retail"

### 5. Rules over-matching
- "AMAZON" rule catches "Amazon Web Services" (should be Utilities & Services)
- "E-TRANSFER" always maps to "Financial Services" but autodeposits are often income

## Top failure patterns (by frequency)

1. **PayrollDeposit/Payroll Deposit -> "Income"** (14 cases): prefix stripping works but SGD classifies "ERICSSONCANADA" as Financial Services. Rules don't cover "ERICSSON".
2. **E-Transfer autodeposits -> "Income"** (12 cases): rules map ALL e-transfers to "Financial Services", but incoming ones are often income.
3. **OPENAI*CHATGPT -> "Utilities & Services"** (10 cases): SGD classifies as Healthcare & Medical (wrong). Not in rules.
4. **MiscPayment RBC CREDIT CARD -> "Financial Services"** (6 cases): SGD sees "RBC CREDIT CARD" and thinks Income.
5. **ContactlessInteracpurchase -> Food** (8 cases): prefix not stripped, so SGD sees garbage.
6. **KindleUnltd -> "Entertainment & Recreation"** (8 cases): SGD sees joined word "KINDLEUNLTD" and guesses Food & Dining.

## What to fix next

### Quick wins
1. Add `Contactless Interac purchase` and `Online Banking payment` to prefix patterns
2. Add `e-Transferreceived` to e-transfer patterns
3. Add rules for: OPENAI/CHATGPT, KINDLE, ERICSSON (payroll), RBC CREDIT CARD, CIBC CPD
4. Split E-Transfer rule: autodeposit/received = Income, sent = Financial Services

### Medium effort
5. Try a different PDF library that preserves spaces (or investigate pdfplumber word extraction settings)
6. Add a word boundary restoration step to preprocessing that tries to re-split joined words

### Phase 2 (DistilBERT)
7. Fine-tune DistilBERT on a mix of synthetic + real (manually corrected) data
8. The char_wb n-gram approach in SGD should theoretically handle joined words better, but the training data doesn't have examples of them

### Active learning
9. The 370 real transactions, once correctly labeled, become the most valuable training data I have
10. Priority: get these 370 correctly labeled, add to training set, retrain
