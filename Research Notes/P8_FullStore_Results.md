# Phase 8 Results: Full external merchant store and retrieval audit

**Date:** 2026-04-05

## Summary

I moved the Phase 8 retrieval work from a small candidate-filtered merchant set to a full external store built from the Canada subset of Foursquare OS Places, plus the existing curated public merchants.

The main result is that the system now has broad external coverage locally, and the benchmark improvement is real. At the same time, the remaining misses are now mostly retrieval and ranking issues rather than corpus coverage issues.

## What I implemented

I replaced the small JSON-first merchant lookup path with a scalable local store:

- a SQLite knowledge store for merchant entries, aliases, and FTS search
- a persistent Chroma collection for dense retrieval
- retrieve and rerank logic in the runtime knowledge base
- a streaming builder that ingests the Canada Foursquare corpus without loading the full dataset into memory

The main implementation pieces are:

- `scripts/build_full_merchant_knowledge_store.py`
- `src/transaction_classifier/knowledge/merchant_kb.py`
- `src/transaction_classifier/knowledge/retrieval.py`
- `src/transaction_classifier/models/ensemble.py`
- `src/transaction_classifier/api/app.py`

I also kept the existing deterministic tiers in front of retrieval:

- direction detection
- rules engine

The runtime path is now:

1. Direction detection
2. Rules engine
3. Exact alias lookup
4. Lexical retrieval
5. Dense retrieval
6. Candidate fusion
7. Cross-encoder reranking
8. Direct KB decision or metadata-enriched fine-tuned classifier fallback

## Full ingest output

The full Canada ingest completed successfully.

- Canada rows scanned: 2,377,121
- Parquet files scanned: 50 out of 53
- Final merchant entries: 1,795,842
- SQLite store: `data/external_kb/merchant_knowledge.sqlite3`
- Chroma collection: `data/external_kb/chroma`

Artifact sizes after the build:

- SQLite store: about 2.4 GB
- Chroma store: about 7.1 GB

This removed the earlier need to prefilter the external corpus around a small candidate set before retrieval. Still however, it's a big issue of size that needs to be worked on. For now I'm still focusing on improving the actual results though. 

## Benchmark result on the openai test set

I evaluated the full-store pipeline on `data/real/openai_labeled.csv`.

Results on the 505-row unique test set:

- Baseline without KB: 65.35% unique accuracy, 72.70% weighted accuracy
- Earlier Phase 8 small-store run: 71.88% unique accuracy, 82.69% weighted accuracy
- Full-store run: **73.86% unique accuracy, 83.62% weighted accuracy**

Additional details from the full-store run:

- Correct predictions: 373 out of 505
- Weighted correct occurrences: 2603 out of 3113
- Flagged for review: 276
- Average confidence: about 0.626

Source breakdown:

- Direction: 9 rows at 100.0%
- Rules: 120 rows at 84.17%
- Knowledge base: 100 rows at 97.0%
- Fine-tune with metadata: 200 rows at 67.5%
- Fine-tune only: 76 rows at 40.8%

This is a meaningful improvement over the clean no-KB baseline, and it is also an improvement over the smaller full-ingest predecessor that used a limited store.

## What improved

The full store improved recall on merchant families that had previously been outside the smaller candidate-filtered KB.

I confirmed successful retrieval or resolution improvements for cases such as:

- `GOODLIFE CLUBS`
- `BAYSHORE SHOPPIN`
- `A & W KA`
- `GRILLADES POULET ROUGE`

The main gain from the full ingest is that the system now has access to the broader merchant corpus locally. This means I no longer need to treat missing external coverage as the default explanation for the remaining failures.

## What the investigation showed

After the full ingest, I checked the merchants that were still failing and traced the candidate lists, fusion order, and reranker behavior.

The main finding is that the remaining issues are mostly not missing-corpus issues.

For several failed test cases, the relevant merchant family already exists in the full store:

- `WENDY'S` exists and resolves correctly on its own
- `ZARA` exists and resolves correctly on its own
- `SHELBYS` exists and resolves correctly on its own
- `UNIVERSITY OF OTTAWA` exists and resolves correctly on its own
- `GOODLIFE FITNESS` exists and resolves correctly on its own

The current failures come from bank-style merchant variants being handled poorly during retrieval.

Examples:

- `WENDY'S PF KANA`
- `ZARA BAYSHORE`
- `SHELBYS 43`
- `UNI OTT TUITION`

In these cases, the store contains the right merchant family, but the retrieval stack still ranks the wrong candidate too highly.

## Remaining weaknesses

### 1. Lexical retrieval is too permissive

The current FTS query builder turns the cleaned query into a flat `OR` over tokens. This is too broad for bank strings with branch or location fragments.

That behavior pushes queries like:

- `WENDY'S PF KANA` toward `KANA` and `PF` noise
- `UNI OTT TUITION` toward `OTT` and `TUITION` noise

### 2. Dense retrieval is too influenced by location-heavy document text

The dense retrieval document currently mixes:

- canonical name
- aliases
- metadata
- place types
- location

This causes location terms like `BAYSHORE` to overpower the brand term in cases such as `ZARA BAYSHORE`.

### 3. Candidate fusion favors dense results too early

The current fusion weights allow the top dense result to outrank the top lexical brand hit even when the lexical side has the correct merchant family.

This is part of why:

- `ZARA BAYSHORE` drifts toward `BAYSHORE` venues
- `SHELBYS 43` drifts toward automotive `SHELBY` results

### 4. The fallback path returns the top fused candidate when rerank confidence is weak

If reranking does not produce a confident enough score, the search path falls back to the top fused candidate instead of the best lexical brand candidate.

That keeps some dense-location mistakes alive even when the lexical side already found the right brand family.

### 5. Alias generation is still conservative

The full-store ingest builds only a limited set of aliases from external rows. That is safe, but it does not generate enough bank-style variants for abbreviated or branch-specific strings.

### 6. Category mapping is still coarse in a few high-value cases

Some Foursquare categories still map too weakly or too generically.

Examples:

- `Retail > Pharmacy` does not currently become a strong healthcare signal
- `Retail > Food and Beverage Retail > Grocery Store` can remain too weak for direct use
- `Travel and Transportation > Rest Area` is often too broad to return directly as a final category

## Cleanliness and evaluation notes

This full-store setup is materially cleaner than the earlier candidate-filtered KB build because I ingested the external Canada corpus directly instead of shaping the KB around the local benchmark merchant identities.

I did not use `openai_category` labels in the runtime or in the store build.

The benchmark is still the same openai-labeled test set, so I should treat the result as a strong internal benchmark improvement rather than the final word on out-of-sample generalization.

## Verification

I ran the test suite after the full-store changes.

- `125` tests passed

## Conclusion

At this stage, I consider the full external ingest complete and successful.

The current system improved from 65.35% to 73.86% unique accuracy and from 72.70% to 83.62% weighted accuracy on the strict openai benchmark.

The remaining work is no longer about loading more of the external corpus. The remaining work is to improve query normalization, lexical search construction, fusion behavior, alias generation, and direct-category gating so the retrieval stack uses the existing corpus more effectively.
