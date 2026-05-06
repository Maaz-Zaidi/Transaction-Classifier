# Phase 1: Training Results
**Date:** 2026-03-23

## Dataset

Used the mitulshah/transaction-categorization dataset from HuggingFace (gated). 4,501,043 total records, dropped 3,719 that were empty after preprocessing, leaving 4,497,324. Split 80/10/10 stratified (3.6M train / 450K val / 450K test). Category balance is nearly perfect, each one sits at roughly 10% of total.

Worth noting: this is synthetically generated data with clean merchant names (e.g., "Tim Hortons #456", "Petro-Canada"). It does NOT reflect how messy real bank strings actually are.

## SGD Model (TF-IDF + SGDClassifier)

### Config
- Vectorizer: TfidfVectorizer, analyzer=char_wb, ngram_range=(3,5), max_features=100,000, sublinear_tf=True
- Classifier: SGDClassifier, loss=modified_huber, alpha=1e-4, class_weight=balanced
- Training time: 60.9 seconds on CPU

### Validation Results (SGD only, no rules)
- **Overall accuracy: 98%**
- **Average confidence: 0.9358**
- **Below 0.70 confidence threshold: 7.4%**
- Worst category: Shopping & Retail (0.95 F1), gets confused with Food & Dining because of Costco/Walmart overlap
- Best categories: Charity & Donations, Entertainment & Recreation, Financial Services (all 1.00 F1)

### Per-Category (Validation)

- Charity & Donations: P 1.00 / R 1.00 / F1 1.00
- Entertainment & Recreation: P 1.00 / R 1.00 / F1 1.00
- Financial Services: P 0.99 / R 1.00 / F1 1.00
- Food & Dining: P 1.00 / R 0.98 / F1 0.99
- Government & Legal: P 0.98 / R 0.97 / F1 0.98
- Healthcare & Medical: P 0.95 / R 0.99 / F1 0.97
- Income: P 0.98 / R 0.99 / F1 0.98
- Shopping & Retail: P 0.96 / R 0.94 / F1 0.95
- Transportation: P 1.00 / R 0.98 / F1 0.99
- Utilities & Services: P 0.98 / R 1.00 / F1 0.99

## Ensemble Results (Rules + SGD)

### Test Set
- **Overall accuracy: 95.7%**
- **Throughput: 30,468 transactions/sec** (449,733 in 14.8s)
- **Flagged for review: 6.2%**

### Where predictions came from
- Rules matched: 14.5% of transactions (65,420)
- SGD handled: 85.5% (384,313)

### Accuracy by source
- **Rules: 80.5%** (52,693/65,420 correct)
- **SGD: 98.3%** (377,712/384,313 correct)

### Per-Category (Ensemble, Test Set)

- Charity & Donations: P 1.00 / R 0.98 / F1 0.99
- Entertainment & Recreation: P 1.00 / R 0.96 / F1 0.98
- Financial Services: P 0.92 / R 0.97 / F1 0.95
- Food & Dining: P 0.94 / R 0.95 / F1 0.94
- Government & Legal: P 0.96 / R 0.95 / F1 0.96
- Healthcare & Medical: P 0.88 / R 0.98 / F1 0.93
- Income: P 0.98 / R 0.97 / F1 0.98
- Shopping & Retail: P 0.94 / R 0.88 / F1 0.91
- Transportation: P 1.00 / R 0.94 / F1 0.97
- Utilities & Services: P 0.96 / R 0.98 / F1 0.97

## What I learned

### SGD alone actually outperforms the ensemble on this dataset
The rules engine drops overall accuracy from 98% to 95.7%, probs because my rules were designed for real messy Canadian bank strings (like "POS PURCHASE - 1847 TIM HORTO OTTAWA ON") but the dataset has clean synthetic strings. The rules over-match. For example, the "BELL" rule meant for Bell Canada telecom matches anything containing "BELL" (like "Bellview Restaurant"). On real bank data the rules should be more precise since bank strings are truncated and formulaic.

### The char_wb n-gram approach works really well
Character-level n-grams at word boundaries (3-5 chars) handle short, semi-structured transaction strings really well. This was the most impactful feature engineering choice. Word-level n-grams would struggle with the vocabulary overlap between categories.

### Shopping & Retail is the hardest category
Consistently the lowest F1 across both SGD-only and ensemble. Main confusion pairs:
- Shopping & Retail vs Food & Dining (Costco, Walmart sell both)
- Shopping & Retail vs Healthcare & Medical (Shoppers Drug Mart)
- Shopping & Retail vs Entertainment & Recreation (Best Buy)

### Phase 2 (DistilBERT) might not even be necessary on clean data
98% SGD accuracy already exceeds my Phase 2 target of 88%. DistilBERT only becomes worth it if real-world accuracy drops a lot on actual bank statements, which it probably will because of the gap between synthetic and real data.

### The real test is real bank data
These results are on synthetic, clean data. Real Canadian bank strings are messy, truncated, and noisy. Actual accuracy will be lower. The preprocessing pipeline and rules engine are built for this reality, but the SGD model has only ever seen clean training data. Priority is testing with real statements and building the active learning loop.

## TODO list
1. Test with real bank statement strings to measure the domain gap
2. Tune rules to reduce false positives on clean data (lower priority for ambiguous terms like "BELL", "SHELL", "METRO")
3. If real-world accuracy drops below ~85%, I'll continue (DistilBERT)
4. Build active learning loop (Phase 3) to keep improving from user corrections
