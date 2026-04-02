# Phase 5: Full corpus evaluation with Codex labels

**Date:** 2026-04-02

## What changed

Expanded test set from 497 descriptions (small 2025 sample) to full corpus: 114 PDFs (39 MasterCard 2022-2025, 76 Noirt/chequing 2019-2026). Re-labeled all descriptions using OpenAI Codex CLI instead of Gemini, with proper deduplication on cleaned text.

## Dataset

- **Total extracted:** 3,113 descriptions from 114 PDFs
- **Raw unique:** 1,380
- **Cleaned unique:** 505 (deduplication on preprocessed text, not raw)
- **Codex labeled:** 505/505 (100% coverage vs Gemini's 59%)
- **Codex-Gemini agreement:** 93.3% on overlap (329 descriptions)

### Key improvements over previous evaluation

1. **Proper deduplication:** Previous eval treated "Visa Debit purchase - 9467 AMZN Mktp CA" and "Visa Debit purchase - 2912 AMZN Mktp CA" as different. Now deduped on cleaned text.
2. **Full coverage:** Gemini failed to label 41% of descriptions. Codex labeled everything.
3. **Better labeling:** Codex correctly labeled GOODLIFE CLUBS as Entertainment (Gemini said Healthcare), convenience stores as Food (Gemini said Shopping).

## Results (Model #5, fine-tuned MiniLM)

### Overall

| Metric | Phase 4b (old test) | Phase 5 (full corpus) |
|--------|--------------------|-----------------------|
| Test descriptions | 497 (with dupes) | 505 (cleaned unique) |
| Overall accuracy | 86.5% | 65.3% unique / 72.7% weighted |
| Fine-tune only | 78.7% | **58.5%** |

### By classification source

| Source | Accuracy | N |
|--------|----------|---|
| Direction | 100.0% | 9 |
| Rules | 84.2% (87.2% weighted) | 120 |
| **Fine-tune** | **58.5%** (59.0% weighted) | **376** |

### By category (Codex ground truth)

| Category | Train Acc | Real Acc (ft-only) | Delta |
|----------|-----------|-------------------|-------|
| Government & Legal | 98.0% | **0.0%** | -98% |
| Shopping & Retail | 68.0% | **21.9%** | -46% |
| Utilities & Services | 100.0% | **30.0%** | -70% |
| Transportation | 71.4% | **34.8%** | -37% |
| Entertainment & Recreation | 99.8% | **45.0%** | -55% |
| Financial Services | 94.6% | **46.9%** | -48% |
| Healthcare & Medical | 100.0% | 50.0% | -50% |
| Charity & Donations | 97.0% | 60.0% | -37% |
| Food & Dining | 98.6% | **80.3%** | -18% |
| Income | 96.4% | 100.0% | +4% |

### Key failure patterns identified

1. **Abbreviated merchant codes:** WMT SUPRCTR, CDN TIRE, MACS CONV, RCSS — WordPiece tokenization destroys them into meaningless subwords
2. **Canadian merchants absent from training:** CRA, MTO, UNIV OF OTTAWA, CHATIME, DOLLARAMA
3. **Bilingual descriptions:** PAIEMENT, REMBOURSEMENT, CREDIT ADDITIONNEL
4. **Confidence nearly useless:** Correct avg 0.365, wrong avg 0.324 (only 4-point gap)
5. **Category distribution mismatch:** Training is 10% balanced, real is 48% Food & Dining

## Preprocessing issues found

- 9/1380 descriptions produce empty strings after cleaning (numeric-only MasterCard refs, Visa Debit/Interac with only transaction IDs)
- Chequing AMZN Mktp CA not caught by preprocessing (prefix strip happens after AMZN check)

## Scripts created

- `scripts/extract_and_evaluate_full.py` — Full pipeline: extract all PDFs, dedup, preprocess, Gemini label, evaluate
- `scripts/label_with_codex.py` — Codex CLI batch labeling (batches of 15, 100% coverage)

## Data files

- `data/real/full_descriptions.csv` — All 3,113 extracted descriptions
- `data/real/full_unique_labeled.csv` — 1,380 raw unique with Gemini + model predictions
- `data/real/codex_labeled.csv` — 505 cleaned unique with Codex labels
- `data/real/model_analysis_report.txt` — Detailed 735-line analysis (confusion matrices, confidence distributions, failure patterns)
