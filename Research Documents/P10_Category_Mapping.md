# Phase 10 Architecture: Category mapping overhaul and match quality gating

**Date:** 2026-04-05

## Problem statement

Phase 9 established that the retrieval pipeline has near-perfect coverage (504 out of 505 test transactions match a KB entry). The bottleneck is not finding the right merchant. The bottleneck is what happens after retrieval: translating a Foursquare place category into the project's 10-category taxonomy, and deciding how much to trust each match.

There are three distinct problems in the post-retrieval path.

The first is the category mapper itself. The function that translates Foursquare labels into the project's taxonomy has a scoring bug that produces tied confidence on most food-related labels, causing 178 out of 504 matches to have no category at all. This forces those transactions into the ML path, which gets them right only 65% of the time.

The second is match quality gating. The pipeline treats a dense_lexical match (where the reranker did not fire or did not produce a confident score) the same as a rerank match. In practice, dense_lexical matches have 28% category accuracy compared to 65% for rerank and near-100% for exact. Feeding wrong merchant metadata to the ML model is worse than feeding no metadata at all.

The third is multi-purpose merchant disambiguation. Some merchants like COSTCO and AMAZON span multiple categories depending on the transaction. The KB assigns them one category, which is correct for the merchant's primary business but wrong for specific transactions. The token analyzer already identifies the disambiguating descriptor (GAS, DOWNLOADS, WEB SERVICES), but the current pipeline does not use that signal to override the KB category.

## Category mapper redesign

### Root cause

The current `map_foursquare_labels()` function scores categories using `any()`:

```
if any(keyword in upper for keyword in keywords):
    scores[category] += depth_bonus
```

This adds at most one score per category per label, regardless of how many keywords match. When a Foursquare label like "Retail > Food and Beverage Retail > Grocery Store" contains keywords for both Food ("GROCERY", "FOOD AND BEVERAGE RETAIL") and Shopping ("RETAIL", "STORE"), each category gets one score, producing a 0.50 confidence tie. The 0.55 threshold rejects it.

### Proposed fix

I will change the scoring to count per-keyword matches, weighted by specificity. Generic structural keywords like "RETAIL", "STORE", "SHOP", and "SERVICE" get a base weight of 1.0. Specific category-identifying keywords like "GROCERY", "PHARMACY", "GAS STATION", and "RESTAURANT" get a higher weight of 2.0. This ensures that "Retail > Grocery Store" resolves to Food because GROCERY (2.0) outweighs RETAIL (1.0).

I will also expand the keyword dictionary to cover Foursquare labels that currently match nothing. The main gaps are:

- Food: "DESSERT", "JUICE", "BISTRO", "STEAKHOUSE", "FOOD AND BEVERAGE SERVICE", "FOOD SERVICE"
- Transportation: "FUEL STATION" (currently only "GAS STATION" is listed)
- Healthcare: "PHARMACY" needs to be added as a healthcare keyword, not just left to compete with "RETAIL"
- Charity: "SPIRITUAL CENTER", "CHURCH", "MOSQUE", "SYNAGOGUE" (currently mapped to Government via "GOVERNMENT" in the label path)
- Utilities: "HOME IMPROVEMENT SERVICE", "CLEANING SERVICE", "CONTRACTOR"
- Government: "EDUCATION", "COLLEGE", "UNIVERSITY", "SCHOOL"

I will also lower the confidence threshold from 0.55 to 0.40. With per-keyword scoring and specificity weighting, a 0.40 threshold is sufficient to resolve genuine ambiguity while still rejecting truly uncertain cases.

### Rebuilding KB categories

The category mapper is called during KB construction, not at query time. Changing the mapper means the existing KB entries retain their old (possibly None or wrong) categories. To apply the new mappings, I need to update the KB entries. For the SQLite store, this means running an UPDATE query that recomputes mapped_category and mapping_confidence for all entries that have raw_category_labels. This avoids a full rebuild of the 1.8M-entry store.

## Match quality gating

### Current behavior

The ensemble routes transactions through the KB based on two thresholds:

1. Direct KB resolution: strategy is not "rerank", similarity >= 0.90, mapping_confidence >= 0.82
2. Metadata enrichment: similarity >= 0.68 and entry has metadata_text
3. Otherwise: raw ML

This means dense_lexical matches with similarity >= 0.68 get their metadata fed to the ML model, even when the matched merchant is completely wrong. At 28% category accuracy, the dense_lexical path is doing more harm than good in the metadata enrichment role.

### Proposed fix

I will add a strategy-aware quality gate. Exact matches pass through as before. Rerank matches pass through as before (the cross-encoder provides a meaningful confidence signal). Dense_lexical matches should only contribute metadata enrichment if their similarity is above a higher threshold (0.80 instead of 0.68), because without the reranker's validation, a low-similarity dense_lexical hit is likely a wrong merchant.

This is a conservative change. It does not reject dense_lexical matches entirely. It raises the bar for when they are trusted enough to enrich the ML input. Transactions where the dense_lexical match is below 0.80 will still reach the ML model, but with raw text instead of misleading metadata.

## Descriptor override

### Current behavior

When the KB resolves a transaction directly (high similarity, high mapping confidence, exact or non-rerank strategy), the ensemble returns the KB category immediately. The descriptor context from the token analyzer is appended to the enriched text for the ML path, but it has no influence on the direct KB resolution path.

This means AMAZON (Shopping, exact match, 1.0 similarity) always resolves to Shopping, even when the full transaction is "AMAZON WEB SERVICES" and the descriptor says "cloud computing, technology services".

### Proposed fix

I will add a descriptor override check in the ensemble before the direct KB resolution return. The check works as follows:

1. If the transaction has a descriptor context from the token analyzer
2. And the descriptor context maps to a specific category (via a small descriptor-to-category dictionary)
3. And that category differs from the KB category
4. Then use the descriptor category instead of the KB category

The descriptor-to-category mapping is small and conservative. It only fires for unambiguous descriptors:

- "gas station, fuel" maps to Transportation
- "cloud computing, technology services" maps to Utilities and Services
- "digital content downloads" maps to Entertainment and Recreation
- "pharmacy drugstore" maps to Healthcare and Medical
- "grocery food retail" maps to Food and Dining

This is not a general-purpose override. It only applies when the descriptor is a known category signal that contradicts the KB. Descriptors that are consistent with or ambiguous relative to the KB category are ignored.

The reason for this conservative scope is that the KB category is usually correct. The override should only fire in the specific case of multi-purpose merchants where the transaction-level descriptor is more informative than the merchant-level category.

## Interaction between the three fixes

The three fixes operate on different parts of the pipeline and mostly do not interact.

The category mapper fix operates at KB build/update time. It changes which entries have mapped categories and what those categories are. This increases the number of transactions that can be resolved directly by the KB, reducing the load on the ML model.

The match quality gate operates at query time in the ensemble. It changes whether a low-quality KB match contributes metadata enrichment to the ML input. This improves the quality of ML inputs for the remaining transactions that cannot be resolved by the KB.

The descriptor override operates at query time in the ensemble. It changes the category returned by high-confidence KB matches for multi-purpose merchants. This fixes a small number of high-confidence wrong predictions.

The expected combined effect is cumulative. The mapper fix brings accuracy from 71% to approximately 81%. The quality gate adds approximately 2-3 points by cleaning up ML inputs. The descriptor override adds approximately 1 point. The total target is approximately 84-85% before any ML retraining.

## Files to modify

- `src/transaction_classifier/knowledge/foursquare.py` (category mapper rewrite)
- `src/transaction_classifier/models/ensemble.py` (quality gate, descriptor override)
- `scripts/rebuild_kb_categories.py` (new script to update SQLite store categories)
- `tests/test_foursquare.py` (new or updated tests for mapper)
- `tests/test_ensemble.py` (updated tests for override logic)
