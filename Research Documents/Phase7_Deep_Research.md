# Phase 7 Research: How to Actually Fix Transaction Classification

Date: 2026-04-04

After Phase 6 failed, I did deep research into how production fintech companies solve this, what open datasets exist, and what the latest ML approaches are. This document captures all of it.

## 1. How Production Fintech Companies Do It

Every single production system uses a hybrid approach. Nobody relies on ML alone.

**Plaid** processes 500M transactions daily at 90%+ accuracy. Their stack: light fuzzy matching for known merchants, a BERT-style MLM trained unsupervised on their entire transaction corpus, a BiLSTM for merchant name extraction (95% accuracy), a classification head, and crucially, SafeGraph Places data (6M+ POIs with MCC codes). They match about 50% of card-present transactions to verified merchant locations. Their taxonomy is 16 primary / 104 detailed categories. In 2025 they added AI-assisted label generation with targeted human review.

Source: https://plaid.com/blog/transaction-enrichment-engine/

**Ntropy** claims 95%+ accuracy using "humans, rules, small-language models, and large-language models." They've seen 100M unique merchants. Key finding from them: the top 500 merchants cover 50% of all transactions, but the other 50% is the long tail of millions of smaller merchants. They say rules-based approaches or ChatGPT alone typically achieve only 60-70%.

Source: https://www.ntropy.com/blog/transaction-categorization

**Spade** (YC, $40M Series B) claims 99.9% coverage of US and Canadian merchants with 99%+ accuracy at under 40ms latency. They go beyond MCC codes with their own merchant database.

Source: https://spade.com/

**Slope** built a LoRA-fine-tuned OPT-125M model for merchant name extraction. Training data: 6M transactions where Plaid auto-tagged 2.5M, filtered to 66K high-quality labels, augmented with 2K hand-labeled. Result: 72% exact match (vs Plaid's 62%), processing 500 txn/sec.

Source: https://medium.com/slope-stories/slope-transformer-the-first-llm-trained-to-understand-the-language-of-banks-88adbb6c8da9

**Takeaway:** Every production system has a merchant knowledge base. That's the missing piece in our pipeline.

## 2. Open Datasets

### Foursquare OS Places (the big find)

100M+ global POIs released under Apache 2.0 (commercial use OK). Available on HuggingFace as `foursquare/fsq-os-places`.

Key attributes: name, fsq_category_ids / fsq_category_labels (hierarchical up to 6 levels deep, e.g. "Dining and Drinking > Restaurant > Chinese Restaurant"), locality, region, country. 1000+ place categories.

This is directly usable as a merchant name -> category lookup table for Canadian businesses.

- HuggingFace: foursquare/fsq-os-places
- Schema: https://docs.foursquare.com/data-products/docs/places-os-data-schema
- Categories: https://docs.foursquare.com/data-products/docs/categories

### MCC Codes

Every card transaction has a Merchant Category Code assigned by the payment processor. Open mappings exist:

- greggles/mcc-codes on GitHub: CSV with MCC, description, IRS/USDA descriptions
- Mastercard: 879 MCCs grouped under 20 categories (Oct 2024)
- Visa: Merchant Data Standards Manual (Oct 2025)

I don't have MCC codes in my bank statement data (they're stripped before consumers see them), but the Foursquare category effectively serves as a substitute.

Sources:
- https://github.com/greggles/mcc-codes
- https://classification.codes/mcc-lookup

### Plaid Taxonomy (downloadable)

Their PFC taxonomy CSV is at: https://plaid.com/documents/transactions-personal-finance-category-taxonomy.csv (16 primary, 104 detailed categories). Could be useful for mapping Foursquare categories to our 10 categories.

### Other

- USA Banking Transactions Dataset (2023-2024) on Kaggle: pradeepkumar2424/usa-banking-transactions-dataset-2023-2024
- Most Kaggle transaction datasets are synthetic or fraud-focused
- World-POI dataset integrating Foursquare + OpenStreetMap: https://arxiv.org/html/2510.21342

## 3. Zero-Shot Classification

The BTZSC benchmark (March 2026, 22 datasets, multiple model families) provides the most comprehensive comparison. Best results:

- Qwen3-Reranker-8B: 0.72 macro F1 (new SOTA for zero-shot)
- Mistral-Nemo-12B: 0.67 (best instruction LLM)
- GTE-large-en-v1.5: 0.62 (best embedding model)
- DeBERTa-v3-large: 0.60 (best NLI cross-encoder)

Critical finding: "Empirical accuracy is highly sensitive to prompt and hypothesis engineering."

For practical use, MoritzLaurer/deberta-v3-large-zeroshot-v2.0 on HuggingFace is the current best NLI-based zero-shot classifier. Improved by 9.4% over v1. There's also a base variant (deberta-v3-base-zeroshot-v2.0) that's faster.

The hypothesis approach for transactions: premise = "LCBO #0008 TORONTO", hypothesis = "This transaction is for purchasing alcoholic beverages." But the entity knowledge problem persists. The model needs to know LCBO is a liquor store.

A Zurich paper (2024) confirmed that fine-tuned small LLMs significantly outperform zero-shot for specialized domains. Zero-shot is best as a fallback for the long tail.

Sources:
- https://arxiv.org/abs/2603.11991 (BTZSC benchmark)
- https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0

## 4. LLM-as-Classifier and LLM-Generated Training Data

### The key paper: SME Transaction Classification (arxiv 2508.05425)

Used GPT-4o (temp 0.7) to generate synthetic transaction descriptions by rephrasing real ones. Example: "biffa waste servic ltd b47391 bbp" -> "veolia refuse service payment ref ltd vrs b47392". Applied inverse-frequency scaling (up to 30x for minority classes). Fine-tuned FinBERT with focal loss (gamma=2).

Results: 73.4% standard accuracy, 90.4% high-confidence accuracy (conf > 0.8). GPT-4o zero-shot alone got only 60.4%.

This is the approach I want to adapt: use an LLM to generate realistic Canadian transaction descriptions, then fine-tune on them.

Source: https://arxiv.org/html/2508.05425v1

### Federal Reserve Paper: Active Knowledge Distillation (arxiv 2511.11574)

Proposes M-RARU for efficient LLM-to-small-model distillation. Teacher: Gemma-3-27B. Key result: up to 80% reduction in labeling requirements vs random sampling. GBDT was the best student: 44x training speedup, 35x inference speedup vs DistilBERT.

Source: https://arxiv.org/html/2511.11574v1

### "Better with Less" (arxiv 2509.25803)

Compared custom small models vs LLMs for financial transactions. A 1.7M parameter model achieved near-identical accuracy to Llama3-8b (72.07% vs 72.89%) at 1/10th the cost and 8x the speed.

Source: https://arxiv.org/html/2509.25803

### Ntropy on GPT-4

"GPT-4 with the right prompts is a reasoning engine that can solve transaction enrichment for nearly all cases. However, it is very slow and can get very expensive."

## 5. Embedding Models for Short Text

Based on MTEB benchmarks (2024-2025):

- all-MiniLM-L6-v2 (22M, 384d): fast, ~78% retrieval. What I'm using now.
- E5-base-v2 (110M, 768d): ~83% retrieval. Balanced.
- BGE-base-en-v1.5 (110M, 768d): ~84.7%. Best precision for its size.
- BGE-large-en-v1.5 (335M, 1024d): ~86%. Production RAG.
- BGE-M3 (568M, 1024d): very high, multilingual. Handles French, which matters for Canadian data.

Moving from MiniLM to BGE-base-en-v1.5 would give 5-8% accuracy boost at reasonable latency. BGE-M3 handles both English and French.

Sources:
- https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/
- https://github.com/embeddings-benchmark/mteb

## 6. Self-Training / Pseudo-Labeling

Self-training is relevant because I have a model trained on synthetic data (source domain) that needs to work on real Canadian transactions (target domain).

Best practices from 2024-2025 literature:
- High confidence threshold (0.90-0.95) for initial pseudo-labels
- Per-class dynamic thresholds (FlexMatch/CPL)
- Iterative: pseudo-label top 10-20%, retrain, repeat 3-5 times
- Prototype-guided selection: check embedding distance to class centroids
- The SME paper showed temperature scaling reduced Expected Calibration Error from 0.1091 to 0.0048

Main risk: confirmation bias. If the model consistently misclassifies CHATIME as Shopping with high confidence, pseudo-labeling cements the error.

Sources:
- https://arxiv.org/pdf/2408.07221 (Pseudo-labeling survey 2024)
- https://aclanthology.org/2024.acl-long.640.pdf (Self-training with pseudo-label scorer)

## 7. Canadian-Specific Challenges

Canada's Open Banking framework rolls out 2026. FDX standard includes Interac, BMO, CIBC, Desjardins, RBC, TD. Standardized transaction formats are coming but not here yet.

Canadian bank statements have specific patterns:
- Interac transaction markers (INTERAC PURCHASE, INTERAC E-TRF)
- Provincial liquor boards (LCBO, SAQ, BCLDB)
- Canadian-only chains (Shoppers Drug Mart, Canadian Tire, Dollarama, Loblaws, Metro, Sobeys)
- Bilingual chain names (JEAN COUTU / PJC)
- Crown corporations (Canada Post, VIA Rail)
- French transaction descriptions (Quebec banks especially Desjardins)

A paper on French Open Banking transactions (arxiv 2504.12319) trained on 94,356 French banking transactions across 84 categories. Best result: Word2Vec (300d) + Random Forest = 95% F1. Simple TF-IDF + LinearSVM got 94% F1.

No dedicated public Canadian bank transaction dataset exists. Foursquare OS Places filtered to Canada is the closest thing to a Canadian merchant knowledge base.

Sources:
- https://arxiv.org/html/2504.12319v1 (French banking paper)
- https://noda.live/articles/open-banking-in-canada

## Synthesis: What To Build

Based on all this research, the path forward has three pillars:

1. **Merchant Knowledge Base** from Foursquare OS Places (Canada subset). This directly solves the entity knowledge problem. Every production system has one.

2. **LLM-Generated Canadian Training Data.** Use Codex/GPT to generate realistic Canadian transaction descriptions. Seed with real Canadian merchant names from Foursquare. Fine-tune MiniLM on combined data with focal loss. The SME paper validated this exact approach.

3. **Zero-Shot NLI Fallback.** DeBERTa for the long tail of truly unknown merchants where neither the knowledge base nor the fine-tuned model has an answer. Low-confidence gated, only activates when nothing else works.

Each pillar is independently testable, additive (doesn't break existing tiers), and runs on CPU.
