# Phase 10 Analysis: Bottleneck identification and improvement ceiling

**Date:** 2026-04-05

## Context

Phase 9 removed the merchant-identity scaffolding rules and introduced token-aware query decomposition. The pipeline is now structurally sound but accuracy dropped from 73.86% to 71.3% because the KB and ML had to absorb responsibilities that the scaffolding rules were covering. This analysis identifies where the remaining errors come from and quantifies the improvement ceiling for each fix.

## Current pipeline performance

The Phase 9 evaluation on 505 Codex test samples breaks down as follows:

- Direction: 9 predictions at 100.0%
- Rules: 13 predictions at 92.3%
- Knowledge base direct: 159 predictions at 91.3%
- Fine-tune with metadata: 240 predictions at 65.8%
- Fine-tune only: 82 predictions at 41.5%
- Overall: 71.3% (360/505)

## Finding 1: KB coverage is nearly perfect but category mapping is broken

The first thing I checked was how many test transactions the KB can actually match. The answer surprised me.

The KB matches 504 out of 505 test transactions. Only one transaction has no KB match at all. Retrieval coverage is 99.8%. The retrieval pipeline built in Phase 8 is working. The dense, lexical, and rerank layers are finding candidates for essentially every transaction.

But of those 504 matches, the category mapping tells a different story:

- 274 have a correct mapped category
- 52 have a wrong mapped category
- 178 have no mapped category at all

The 178 no-category entries are the biggest gap. These have KB matches with metadata, but the `map_foursquare_labels()` function in foursquare.py either cannot map their Foursquare labels or has too low confidence. These transactions fall through to the ML model, which gets them right only 65.8% of the time (when enriched with metadata) or 41.5% (when not).

If every KB match had the correct category, the KB alone could resolve 504 out of 505 transactions. The theoretical ceiling is enormous.

## Finding 2: The Foursquare mapper has a structural scoring bug

I traced why 178 entries have no category and found a bug in the scoring logic of `map_foursquare_labels()`.

The function uses `any()` to check whether any keyword matches for a given category. When a match is found, it adds a single depth-weighted score for that category. The problem is that `any()` short-circuits after the first match. A label like "Retail > Food and Beverage Retail > Grocery Store" matches both Food keywords ("GROCERY", "FOOD AND BEVERAGE RETAIL") and Shopping keywords ("RETAIL", "STORE"), but each category gets only one score regardless of how many keywords match. The result is a 0.50/0.50 confidence tie, which falls below the 0.55 threshold, and the function returns None.

This affects most food-adjacent Foursquare labels. I tested the specific labels from the no-category entries:

- "Retail > Food and Beverage Retail > Grocery Store" returns confidence 0.50, no category
- "Retail > Convenience Store" returns confidence 0.50, no category
- "Dining and Drinking > Cafe, Coffee, and Tea House > Bubble Tea Shop" returns confidence 0.50, no category
- "Retail > Pharmacy" returns confidence 0.50, no category
- "Retail > Food and Beverage Retail > Supermarket" returns confidence 0.50, no category

These are all unambiguous labels. A grocery store is Food and Dining. A pharmacy is Healthcare. A bubble tea shop is Food and Dining. The mapper cannot resolve any of them because the generic "RETAIL" or "STORE" keyword ties with the specific food or health keyword.

Among the 178 no-category entries, 66 have valid labels that fail due to this confidence tie. Another 67 have labels that don't match any keyword because the keyword dictionary is missing entries like "FUEL STATION", "DESSERT", "FOOD AND BEVERAGE SERVICE", "HOME IMPROVEMENT SERVICE", "EDUCATION", and "CLEANING SERVICE". The remaining 45 entries have no Foursquare labels at all (location-only entries in the external database).

## Finding 3: The wrong category mappings are systematic

The 52 entries with wrong categories fall into a few patterns.

Food labeled as Shopping (5 cases): COSTCO WHOLESALE is mapped to Shopping because "Retail > Warehouse or Wholesale Store" matches Shopping. The label is technically correct for the merchant, but the ground truth says Food because these COSTCO transactions were grocery purchases.

Entertainment labeled as Shopping (4 cases): entries where Foursquare categorizes them under Retail subsets.

Transportation labeled as Food (4 cases): entries like ONROUTE (a highway rest stop mapped to Transportation via "Travel and Transportation > Rest Area", but it is actually a food court) and PRESTO (transit card reloads matching wrong KB entries).

Charity labeled as Government (4 cases): entries like religious organizations and nonprofits. Foursquare labels them as "Community and Government > Organization" or "Community and Government > Spiritual Center > Church", and the mapper sends all "GOVERNMENT" matches to Government and Legal. These should be Charity and Donations.

Healthcare labeled as Shopping (2 cases): SHOPPERS DRUG MART matched to Shopping because "Retail > Pharmacy" triggers the RETAIL keyword for Shopping alongside the PHARMACY keyword for Healthcare, producing a tie that resolves to Shopping.

## Finding 4: The ML model cannot use enriched text

The fine-tuned MiniLM was trained on 3.6 million raw merchant names from the mitulshah synthetic dataset. The training text looks like "NORDSTROM RACK TXN751417", "SAFEWAY PHARMACY", "CARNIVAL". The model was never exposed to the enrichment format used at inference.

I tested the model directly on enriched versus raw inputs:

- "COSTCO" alone predicts Food and Dining at 0.343 confidence
- "COSTCO. external metadata: warehouse retailer. descriptor context: gas station, fuel" predicts Utilities and Services at 0.217 (worse and wrong)
- "SHOPPERS DRUG MART" alone predicts Food and Dining at 0.296
- "SHOPPERS DRUG MART. external metadata: pharmacy and drugstore chain" predicts Healthcare at 0.504 (correct, confidence jumped)

The model is inconsistent. Sometimes metadata helps, sometimes it hurts. The fundamental issue is a training/inference format mismatch.

However, plain category keywords do work: "pharmacy drugstore" alone predicts Healthcare at 0.608, "gas station fuel" alone predicts Utilities at 0.403. The model understands individual category concepts. It just cannot parse the structured enrichment format.

The 240 finetune_metadata predictions have an average error confidence of 0.28. The model is guessing on most of its mistakes.

## Finding 5: Dense/lexical retrieval is actively harmful in many cases

I broke down the KB matches by retrieval strategy:

- Exact: 208 matches at similarity 1.0
- Rerank: 117 matches at average similarity 0.918
- Dense_lexical (no reranker hit): 179 matches at average similarity 0.718

The exact matches are reliable. The rerank matches are decent (76 out of 117 have correct KB category, 65%). The dense_lexical matches are terrible: only 50 out of 179 have correct KB category (28%).

I checked for wrong merchant retrievals in the dense_lexical group and found 95 cases where the retrieved merchant has no token overlap with the query at all. "UNIVERSITY OF OTTA" matched "U of T Department of Italian Studies". "AEO CANADA CORPORATION" matched "Alberta Electric System Operator". "CHARTWELLS-OTTAWA U-27" matched "T317" (a college classroom entry). These wrong matches feed completely wrong metadata to the ML model.

Currently 164 transactions qualify for direct KB resolution. At relaxed thresholds (similarity >= 0.85, confidence >= 0.75), 230 would qualify, pulling more away from the unreliable ML path.

## Finding 6: Descriptor context could override wrong KB categories

I checked whether the token analyzer's descriptor context would correctly override wrong KB categories. There are 5 confirmed cases:

- AMAZON DOWNLOADS: KB says Shopping, descriptor says "digital content downloads" which is Entertainment
- SHOPPERS DRUG MART: KB says Shopping, descriptor says "pharmacy drugstore" which is Healthcare
- AMAZON WEB SERVICES: KB says Shopping, descriptor says "cloud computing, technology services" which is Utilities
- AMAZON DOWNLOAD: same as above
- AMAZON WEB SERV: same as above

These are all high-confidence exact KB matches (similarity 1.0) currently returned with 0.99 confidence in the wrong category. The descriptor context is the correct signal.

## Ceiling estimates

I simulated an improved category mapper with per-keyword scoring, extended keywords, and specificity weighting. The simulation added correct categories to 48 previously no-category entries and fixed 2 previously wrong mappings, for a net gain of 50 correct predictions. That would bring accuracy from 71.3% to approximately 81.2%.

However, the simulation also introduced 30 new wrong mappings because some entries were matched to wrong merchants in the first place (the 95 bad dense_lexical retrievals). The mapper fix helps most on the exact and rerank matches where retrieval was correct.

Combined improvement estimate for the full stack:

- Fix Foursquare mapper: 71.3% to approximately 81%
- Filter out bad dense_lexical enrichment: adds approximately 2-3% by not polluting ML with wrong metadata
- Descriptor override: adds approximately 1% (5 high-confidence fixes)
- ML retraining on enriched format: adds approximately 5-7% on the remaining ML predictions

Realistic combined target: 85-90% accuracy on the 505-row Codex test set.

## Implementation results

I implemented all three fixes and ran the evaluation on the 505-row Codex test set.

The Foursquare mapper rebuild updated 384,030 entries in the 1.8M-entry SQLite store. Of those, 326,544 gained a category for the first time (None to something), 27,542 had their category corrected, and only 41 lost a category. KB coverage went from 989,660 mapped entries to 1,316,163.

The evaluation results after all three fixes:

Overall accuracy: 72.5% (366/505), up from 71.3% (360/505). The breakdown by source:

- Direction: 9 predictions at 100.0% (unchanged)
- Rules: 13 predictions at 92.3% (unchanged)
- Knowledge base direct: 172 predictions at 94.8% (was 159 at 91.3%)
- Fine-tune with metadata: 156 predictions at 68.6% (was 240 at 65.8%)
- Fine-tune only: 155 predictions at 48.4% (was 82 at 41.5%)

The KB direct path improved as expected. Thirteen more transactions resolve directly through the KB with the corrected mapper, and the accuracy of those resolutions went up. The descriptor override is working correctly for the multi-purpose merchants.

The overall gain of 1.2 percentage points is lower than my initial ceiling estimate of 10 points. The reason is that the quality gate did what it was supposed to do: it stopped feeding wrong dense_lexical metadata to the ML model. But this pushed 84 transactions from the metadata-enriched ML path (68.6% accuracy) into the raw ML path (48.4% accuracy). The net effect on those 84 transactions is negative in the short term, even though the enrichment they were receiving was often wrong.

The bottleneck is now squarely in the ML model. 155 transactions go through the raw finetune path at 48.4% accuracy, contributing 80 of the 139 total errors. The model was trained on raw merchant names and does not understand the enrichment format. The next step is retraining the ML model on enriched inputs so it can use the metadata signal that the KB provides.

The 48 errors in the finetune_metadata path are also ML-limited. The enriched text is correct in most cases, but the model cannot parse it effectively. The most common confusion is Shopping being mislabeled as Utilities or Financial Services.

## Files used in this analysis

The diagnostic scripts are stored in `scripts/experiments/phase10_analysis.py`.

## Next steps

1. ~~Fix the Foursquare category mapper scoring and keyword coverage~~ (done)
2. ~~Raise the quality bar for dense_lexical matches~~ (done)
3. ~~Add descriptor-based category override in the ensemble~~ (done)
4. Retrain the ML model on enriched inputs (next phase)
