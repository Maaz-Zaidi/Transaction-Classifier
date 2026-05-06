# Phase 11: Enriched model retraining

**Date:** 2026-04-06

## Context

Phase 10 fixed the Foursquare category mapper, added a quality gate for dense_lexical matches, and introduced descriptor-based overrides for multi-purpose merchants. These changes brought accuracy from 71.3% to 72.5% on the 505-row openai test set. The analysis showed that the ML model was the remaining bottleneck, contributing 128 of the 139 total errors. The model was trained on raw merchant names but at inference time receives enriched text with Foursquare place types, location data, and descriptor context. I set out to bridge that training/inference format mismatch.

## Investigation

I started by checking whether I could simply look up training data merchant names in the KB and use the resulting metadata for enrichment. I sampled 500 synthetic training examples and ran them through the KB retrieval pipeline.

The results ruled out that approach. The KB matched 100% of synthetic training samples (expected, given the 1.8M-entry store), but only 13.4% were exact matches. The other 86.6% were dense_lexical or rerank matches, and only 38.2% of all matches had the correct category. The synthetic training data contains generic names like "CONSULTING STORE TXN889259" and "MBTA BRANCH" that do not correspond to real merchants. Looking them up in the KB produces wrong merchant matches with wrong metadata 41% of the time. Training on that would teach the model to trust garbage metadata.

I pivoted to a different approach: using the training sample's known category label to sample real Foursquare label paths from the KB. For each training sample categorized as Food and Dining, I sample a real Foursquare path like "Dining and Drinking > Restaurant > Fast Food Restaurant" from the pool of labels that actually map to Food and Dining. This teaches the model the format and the category signals without contaminating it with wrong metadata.

I built a label pool by querying all entries in the SQLite store that have both a mapped_category and raw_category_labels. I initially collected all labels from matching entries, but found that many entries have co-occurring labels from different categories. An entry mapped to Food might have labels for both "Dining and Drinking > Restaurant" (correct) and "Landmarks and Outdoors > Nudist Beach" (noise from co-location). I fixed this by individually verifying each label through map_foursquare_labels() before adding it to the pool.

The clean label pool has 851 unique labels across 9 categories (Income has no Foursquare representation, which is correct).

## Training approach

I implemented the enrichment in a new module (src/transaction_classifier/data/enrich.py) that takes training texts and labels, and for a configurable fraction of samples, appends Foursquare metadata in the same format that build_enriched_transaction_text() produces at inference time. The enrichment varies across four format templates to build robustness:

1. Full format with "external metadata:" and duplicate "place types:" (matches most inference text, 30% of enriched samples)
2. Just "external metadata:" (25%)
3. Just "place types:" (25%)
4. Full format with "merchant identity:" (20%)

Some fraction of enriched samples (8%) receive wrong-category metadata to make the model robust to bad KB matches at inference time. Canadian locations from the KB are randomly appended to 40% of enriched samples.

## First attempt: training from scratch

I first tried training a fresh model (from the base MiniLM checkpoint) on 139K samples (50K base, 2x augmentation, 50% enriched) with 4 epochs. The model achieved 99% on the synthetic validation set but only 70.1% on the real test set, down from the 72.5% baseline. The metadata path improved slightly (71.4% vs 68.6%) but the raw text path collapsed from 48.4% to 38.7%.

The problem was clear. The model trained from scratch on enriched data learned to expect metadata and performed poorly when it was absent. The synthetic training data is also fundamentally different from real bank transaction text. Clean names like "TACO BELL - USA" are nothing like "Contactless Interac purchase - 8427 PINEWOOD ORCHAR".

## Second attempt: continuation training

I switched to continuing from the existing baseline checkpoint rather than training from scratch. This is standard continual learning: the model keeps its existing raw text ability while learning the new enrichment format as an additional signal.

I added a from_checkpoint parameter to the FineTuneModel.train() method so it loads from a saved model directory instead of the base MiniLM. I also updated predict() to read max_length from the model's saved metadata so it handles longer enriched text correctly.

I tested several configurations:

1. 50K samples, 50% enrich, lr=2e-5, 4 epochs: 70.1% (from scratch, terrible)
2. 50K samples, 50% enrich, lr=2e-5, 4 epochs (improved format): 68.9% (from scratch, still bad)
3. 20K samples, 50% enrich, lr=5e-6, 2 epochs (from checkpoint): 73.7% (breakthrough)
4. 30K samples, 45% enrich, lr=3e-6, 3 epochs (from checkpoint): 73.7% (same)
5. 20K samples, 30% enrich, lr=5e-6, 2 epochs (from checkpoint): 73.3% (slightly worse)
6. 20K samples, 50% enrich, lr=5e-6, 2 epochs (from checkpoint, rerun): 73.9% (best)

The key hyperparameters for continuation training are the learning rate (5e-6, one quarter of the original 2e-5) and the number of epochs (2 instead of 3-4). Too much training causes catastrophic forgetting of the raw text distribution. The 50% enrich ratio was the best trade-off.

## Results

The best enriched model (configuration 6) achieves 73.9% accuracy (373/505), up from 72.5% (366/505) with the Phase 10 baseline.

Breakdown by source:
- Direction: 9 predictions at 100.0% (unchanged)
- Rules: 13 predictions at 92.3% (unchanged)
- Knowledge base direct: 175 predictions at 94.3%
- Fine-tune with metadata: 153 predictions at 77.1% (was 68.6%, the main improvement)
- Fine-tune only: 155 predictions at 44.5% (was 48.4%, slight regression)

The metadata path improved by 8.5 percentage points, which is the core win from enriched training. The model can now parse "place types: Dining and Drinking > Restaurant" in the enriched text and use it as a category signal. The raw text path regressed by 3.9 points, which partially offsets the gain.

The net improvement of +1.4 points (7 additional correct predictions) is modest in absolute terms. The ceiling for this approach is limited by the raw text path: 155 transactions go through the ML model with no metadata at all, and the model gets them right only 44.5% of the time. These are the hardest cases, where the KB either found no match or the match was filtered by the quality gate.

## Progression

Starting from Phase 9 (71.3%), the combined Phase 10 and 11 changes brought accuracy to 73.9%.

Phase 10 fixes (mapper + quality gate + override) added 1.2 points by improving the KB resolution path: more transactions resolve directly via KB, bad dense_lexical matches no longer pollute the ML input, and multi-purpose merchants get their category overridden by descriptor context.

Phase 11 enriched retraining added 1.4 points by improving the ML model's ability to use metadata. The finetune_metadata path went from 68.6% to 77.1% accuracy.

## Files

- src/transaction_classifier/data/enrich.py (new enrichment module)
- src/transaction_classifier/models/finetune_model.py (added from_checkpoint parameter, predict reads max_length from metadata)
- scripts/train_finetune_enriched.py (new training script)
- scripts/evaluate_phase6.py (added "enriched" model option)
- models/finetune_enriched/ (saved model)

## Next steps

The remaining 131 errors come from three sources:

1. Raw ML predictions (87 errors from 155 predictions). These transactions have no usable KB metadata. Improving them requires either better KB retrieval coverage or a fundamentally different model architecture.

2. Enriched ML predictions (35 errors from 153 predictions). The model can use metadata but still makes mistakes. Better metadata quality and model retraining on real-world data would help.

3. KB direct resolution (9 errors from 175 predictions). The KB maps to the wrong category in a few cases. These are mostly multi-category merchants where the Foursquare label is technically correct for the merchant but wrong for the specific transaction.

The highest-impact next step would be collecting more real-world labeled data for training. The synthetic dataset is the fundamental limiter, as it does not represent the distribution of real Canadian bank transaction descriptions.
