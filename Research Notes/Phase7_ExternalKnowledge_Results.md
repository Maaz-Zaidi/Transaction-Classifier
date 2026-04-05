# Phase 7 Results: External merchant knowledge + metadata enrichment

**Date:** 2026-04-04

## Implementation

Phase 7 adds a merchant knowledge path because Phase 6 identified merchant knowledge as the main gap.

The pipeline now looks like this:

- Stage 1: direction detection
- Stage 2a: rules engine
- Stage 2b: external merchant knowledge base
- Stage 2c: metadata-enriched fine-tuned MiniLM fallback

The implementation includes:

- `scripts/build_external_merchant_kb.py` to build a merchant knowledge base from public data
- `src/transaction_classifier/knowledge/merchant_kb.py` for lookup and matching
- `src/transaction_classifier/knowledge/foursquare.py` for mapping Foursquare place categories into the project's 10-category taxonomy
- `src/transaction_classifier/knowledge/curated_merchants.py` for a small public seed set
- runtime integration in `src/transaction_classifier/models/ensemble.py`

The builder uses the gated Foursquare OS Places dataset (`foursquare/fsq-os-places`) and scans only the Canada-relevant parquet shards. In this run, it matched 27,286 candidate merchant rows across 2,377,140 Canadian place rows and wrote a merged KB with 164 entries:

- 132 Foursquare-derived
- 32 curated public merchants

Runtime behavior:

1. If the KB finds a very strong merchant match with a confident mapped category, the category is returned directly.
2. If the KB finds a plausible merchant with useful metadata but not enough confidence for a direct label, that metadata is appended to the transaction text and passed to the fine-tuned classifier.
3. If there is no useful merchant match, the system falls back to the fine-tuned model alone.

## Results on the Codex test set

Evaluation was run on `data/real/codex_labeled.csv` with 505 cleaned unique merchants and 3113 weighted occurrences.

### Baseline vs knowledge-enabled pipeline

- No KB: 65.35% unique accuracy, 72.70% weighted accuracy
- Curated-only KB: 68.32% unique accuracy, 81.14% weighted accuracy
- Foursquare-only KB: 70.10% unique accuracy, 75.55% weighted accuracy
- Merged KB: **71.88% unique accuracy, 83.14% weighted accuracy**

### Final merged-KB run

- Unique accuracy: **71.88%** (363/505)
- Weighted accuracy: **83.14%** (2588/3113)
- Flagged for review: 260
- Average confidence: 0.650

### By classification source

- Direction: 100.0% (9/9)
- Rules: 84.17% (101/120)
- Knowledge base: **96.55% (112/116)**
- Fine-tune + metadata: 57.35% (39/68)
- Fine-tune only: 53.13% (102/192)

### Significance check

Compared with the clean no-KB baseline on the same 505-row benchmark:

- Unique accuracy went from 65.35% to 71.88% for a gain of 6.53 points
- Weighted accuracy went from 72.70% to 83.14% for a gain of 10.44 points
- 39 rows improved and 6 rows regressed
- McNemar exact p-value: about `5.4e-7`

The improvement is unlikely to be explained by rounding noise.

## Observed strengths

### 1. Merchant knowledge appears to be the correct focus

Phase 6 suggested that the main issue was merchant knowledge rather than tokenization. Phase 7 is consistent with that conclusion. Once the system has a better representation of the merchant entity, classification becomes substantially easier.

### 2. Direct KB hits were highly precise

The direct `knowledge_base` route got 112 out of 116 predictions right, or 96.6%. This was the most precise part of the pipeline.

Examples:

- `RCSS SOUTH KANATA` -> Food & Dining
- `KINDLE UNLTD` -> Entertainment & Recreation
- `FIT4LESS` -> Entertainment & Recreation
- `RIDEAU CENTRE` -> Shopping & Retail
- `BARBERHOLIC BARBER` -> Utilities & Services

### 3. Metadata enrichment still helped when direct resolution was uncertain

Even when the KB could not safely return a direct category, the added metadata still improved the classifier on ambiguous strings.

Examples:

- `MACS CONV. STORE`
- `SHOPPER'S DRUG`
- `RCSS SOUTH KANA`

### 4. Foursquare contributed useful signal on its own

Foursquare-only moved the benchmark from 65.35% / 72.70% to 70.10% / 75.55%. This indicates that the public place dataset contributes useful signal even without the curated seed set.

## Limitations and open issues

### 1. This is not a clean held-out generalization result

The current builder uses `data/real/full_descriptions.csv` to generate candidate merchant names for Foursquare lookup. In this repo, the cleaned merchant identities in `full_descriptions.csv` overlap the Codex test set merchant identities exactly:

- Codex cleaned unique merchants: 504
- Full descriptions cleaned unique merchants: 504
- Overlap: 504/504

This means the Phase 7 setup is **transductive**, not a strict unseen-merchant holdout.

Important nuance:

- It is **not** label leakage because the KB builder never reads `codex_category`
- It **does** shape external KB construction around the exact merchant names in the benchmark

This makes the result encouraging, but not final evidence of out-of-sample generalization.

### 2. The curated seed still drives a large share of the weighted lift

The curated public merchant file overlaps 36 benchmark merchant identities, and those merchants are frequent ones. That is why the weighted gain increases so much.

This is visible in the ablation:

- Curated-only KB: 68.32% unique / 81.14% weighted
- Foursquare-only KB: 70.10% unique / 75.55% weighted
- Merged KB: 71.88% unique / 83.14% weighted

The merged result remains useful, but much of the weighted gain appears to come from frequent known merchants.

### 3. Recall remains the main weakness

The current JSON KB plus char-ngram matcher is precise, but it does not catch enough merchants.

Across the 504 unique cleaned merchants in the local corpus:

- Exact alias hits: 168
- Any KB match above minimum similarity: 269
- Direct-route candidates: 148
- Metadata-route candidates: 91
- Below-threshold weak matches: 30
- No KB match at all: 235

Weighted by occurrence count:

- Direct-route candidates: 1653/3113 occurrences
- Metadata-route candidates: 495/3113
- Below-threshold weak matches: 84/3113
- No KB match at all: 872/3113

At this stage, recall is a larger issue than precision.

### 4. Some missed merchants already exist in Foursquare

Manual review of the misses showed that several merchants already exist in Foursquare under different raw or canonical names. The current builder misses them because it keeps only exact normalized candidate matches.

Examples confirmed in Foursquare:

- `GOODLIFE CLUBS` -> `GoodLife Fitness`
- `WENDY'S PF KANA` -> `Wendy's`
- `ZARA BAYSHORE` -> `Zara`
- `BAYSHORE SHOPPIN` -> `Bayshore Centre`
- `SHELBYS 43` -> `Shelby's Legendary Shawarma`
- `UNI OTT TUITION` -> `University of Ottawa`
- `A & W KA` -> `A&W`
- `GRILLADES POULET ROUGE` -> `Poulet Rouge`
- `PAYPAL PATREON` -> `PayPal Canada`

This is strong evidence that the current KB architecture is still missing reachable improvements.

### 5. Some direct KB mistakes still come from heuristic category mapping

The direct KB route had 4 wrong predictions:

- `THANK YOU` -> Shopping & Retail (truth: Financial Services)
- `ONROUTE` -> Transportation (truth: Food & Dining)
- `MACS CONVENIENCE` -> Shopping & Retail (truth: Food & Dining)
- `ARZ FINE FOODS` -> Shopping & Retail (truth: Food & Dining)

These errors appear to come from a combination of:

- weak merchant normalization or alias ambiguity
- heuristic Foursquare category-to-taxonomy mapping

## Category behavior

The gains were spread across multiple categories, not just one:

- Entertainment & Recreation: 53.3% -> 60.0%
- Financial Services: 53.8% -> 56.4%
- Food & Dining: 82.6% -> 89.3%
- Government & Legal: 25.0% -> 37.5%
- Healthcare & Medical: 83.3% -> 100.0%
- Shopping & Retail: 41.4% -> 49.5%
- Transportation: 61.7% -> 63.8%
- Utilities & Services: 33.3% -> 46.7%

Even so, the weakest classes are still weak. Merchant knowledge helps, but it does not fully solve:

- Government & Legal
- Utilities & Services
- Financial Services

## Conclusion

Phase 7 indicates that merchant knowledge is worth keeping. The external knowledge layer improved this benchmark, and direct merchant resolution was very accurate when it triggered.

At the same time, the result should be interpreted with clear limits:

1. It is better than the clean no-KB baseline.
2. It is not label cheating.
3. It is not yet a clean unseen-merchant evaluation.
4. The next main problem is merchant retrieval recall, not category reasoning.

The next step is to replace the current exact or fuzzy JSON lookup with a dense retrieval plus reranker system built from external sources.
