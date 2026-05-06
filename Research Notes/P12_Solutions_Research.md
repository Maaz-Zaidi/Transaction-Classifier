# Phase 12: Solutions Research — How to Get +10% Accuracy

**Date:** 2026-04-06

## How the Industry Solves This

### Plaid (500M transactions/day, market leader)
Plaid built a **transaction foundation model** using contrastive learning on their massive transaction dataset. Positive pairs = transactions with the same financial meaning, hard negatives = similar-looking but different transactions. The encoder learns "financial intent" not lexical similarity. Results: income classification +48%, loan payment detection +14%, bank fee classification +22%. They then layer lightweight task-specific heads on top.

Key insight: **they solved entity resolution first** (who is this merchant?) and classification second (what category?). Their AI Annotator achieves >95% human alignment for labeling.

### Brex (credit card company)
Two-stage pipeline: (1) **Google Places API** to identify merchant from descriptor text + location, (2) ML classifier on the resolved entity. Google Places resolves ~50% of merchants. They clean descriptors by stripping card processor prefixes (TST*, SQ*, WPY*). InferSent embeddings outperformed bag-of-words. Overall error rate: **well below 5%**. Fallback: Amazon Mechanical Turk for unresolved merchants.

Key insight: **external entity resolution (Google Places) is the primary classifier**, ML is secondary.

### Ntropy (transaction enrichment API)
Uses LLMs + database of 100M+ entities. Pipeline: (1) entity identification from description, (2) query entities database + location service, (3) categorize using description + amount + resolved entity. Combines weak supervision, knowledge transfer, and transformer architecture. Key finding: **needed tens of crowd-sourced human labels per transaction** for the model to distinguish signal from noise.

### Meniga (80B transactions/year, 90%+ accuracy)
Multiple category detectors: MCC codes, text analysis, transaction amount, internal bank codes. ML learns subtle patterns in descriptions, merchant names, and amounts. Handles 30+ countries.

Key insight: **MCC codes + text + amount together** outperform text alone.

### Triqai
Advocates solving the problem **upstream via enrichment** rather than better classification rules. Their pipeline: parse raw string → recognize merchant identity → analyze context (channel, location, recurrence) → infer user intent → map to category. Achieves 95%+ on well-identified transactions. Uses "AI reasoning and web-derived context to identify merchants dynamically."

---

## The Pattern: Entity Resolution Before Classification

Every successful system follows the same architecture:

```
Raw transaction text
    ↓
[Stage 1] Merchant Identity Resolution
    - Match to known merchant database
    - Use external APIs (Google Places, Foursquare)  
    - Handle abbreviations, truncations
    ↓
[Stage 2] Category Classification
    - Known merchant → direct lookup (deterministic)
    - Unknown merchant → ML classification on enriched features
    - Use MCC codes, amount, location as additional features
```

**Our current architecture already follows this pattern** (rules → KB → ML), but our KB resolution is weak (rerank matches wrong merchants) and our ML fallback has no useful training data.

---

## Concrete Solutions Ranked by Expected Impact

### Solution 1: LLM-Generated Synthetic Training Data (+5-10%)

The most directly relevant paper: **"Categorising SME Bank Transactions with Machine Learning and Synthetic Data Generation"** (arXiv:2508.05425). They used GPT-4o (temperature 0.7) to generate semantic variations of transaction descriptions:

- Input: "biffa waste servic ltd" (Utilities)
- Generated: "veolia refuse service payment", "grundon rubbish collection fee"

Results: FinBERT fine-tuned on synthetic data achieved **73.49%** standard accuracy, **90.36% at high confidence (>0.8)**, and **89.63% top-2 accuracy**. Synthetic data had 94.2% uniqueness (no mode collapse) and 0.879 cosine similarity with real data.

**How we'd apply this:**
1. Prompt Claude/GPT-4o with our 10 categories + 20-30 Canadian examples each
2. Ask it to generate 500-1000 diverse Canadian merchant names per category
3. Include bank statement formatting (truncation, no spaces, bilingual)
4. Include descriptor words (GRILL, BAKERY, PHARMACY, GAS, etc.)
5. Generate 5K-10K total examples covering the long tail of merchants
6. Fine-tune from our existing checkpoint with low LR

Example prompt structure:
```
Generate 100 realistic Canadian bank transaction descriptions for the 
category "Food & Dining". Include:
- Local Canadian restaurants (shawarma shops, pho restaurants, poutine places)
- Truncated merchant names (max 25 chars, cut mid-word)
- Convenience stores that sell food
- Bilingual names (French/English)
- Various formats: "MERCHANT CITY PROV", "MERCHANT", "MERCHANT #1234"
```

**Expected cost:** ~$5-15 for 10K examples via Claude API.

### Solution 2: KB Entries as Training Data — Distant Supervision (+3-5%)

We have 1.3M KB entries with verified categories. Use them directly as training data:

1. Take each KB entry's `canonical_name` and `aliases` as training text
2. Use `mapped_category` as the label (only entries with confidence >= 0.5)
3. Apply augmentation: truncation, abbreviation, noise injection
4. Filter to Canadian entries for domain match (~300K entries)

This is **distant supervision** — using a knowledge base to automatically label training data. The key challenge is noisy labels (some KB categories are wrong). Mitigation: only use entries with high mapping confidence, and mix KB-derived data with existing synthetic data.

**Concrete numbers:**
- ~300K Canadian KB entries with categories
- After deduplication by canonical name: ~50-100K unique merchant names
- Each with 1-3 aliases = 100-300K training samples
- All with correct category labels derived from Foursquare taxonomy

This immediately solves the "847 merchant names" problem — we'd go from 847 to 50,000+ real merchant names in training.

### Solution 3: Contrastive Learning on Merchant Names (+3-5%)

Instead of classification (text → 1 of 10 categories), train a **sentence embedding model** that maps similar merchants close together. This is what Plaid does.

**How it works:**
1. Create positive pairs from KB: ("SHOPPERS DRUG MART", "SHOPPERS DRUG M") — same entity
2. Create negative pairs: ("SHOPPERS DRUG MART", "WALMART PHARMACY") — different entity
3. Fine-tune MiniLM with contrastive loss (MultipleNegativesRankingLoss)
4. At inference: embed transaction text, find nearest KB entry, use KB's category

This **replaces** the current Chroma dense retrieval with a domain-specific retrieval model. The current MiniLM embeddings are general-purpose (trained on NLI/paraphrase). A contrastively trained model would understand that "CDN TIRE" ≈ "CANADIAN TIRE" ≈ "Canadian Tire" even though they're lexically very different.

The Eridu project (HuggingFace) did exactly this for company name matching: fine-tuned MiniLM on 2M+ labeled pairs of matching/non-matching names using contrastive learning.

**How we'd build pairs:**
- KB has `canonical_name` and `aliases` — each alias is a positive pair with the canonical name
- We have 1.8M entries × 3-5 aliases each = millions of positive pairs
- Negative pairs: random entries from different categories

### Solution 4: Two-Stage Entity Resolution → Classification (+5-8%)

Restructure the pipeline to explicitly separate merchant identity resolution from category classification:

**Stage 1: WHO is this merchant?**
- Use the contrastively-trained embedding model (Solution 3) to match transaction text to KB entries
- This is pure entity resolution: "CDN TIRE STORE" → "Canadian Tire"
- If matched with high confidence → use KB category directly (deterministic)
- If no match → pass to Stage 2

**Stage 2: WHAT category is this transaction?**
- Only runs for truly unknown merchants
- Uses enriched text with any partial information (descriptor words, location)
- Trained on diverse data (Solutions 1+2) so it handles unknown merchants

This is Brex's architecture: Google Places resolves 50% deterministically, ML handles the rest. Our KB should resolve an even higher percentage given 1.8M entries.

**The key improvement over current pipeline:** currently our KB resolution fails on "CDN TIRE STORE" because the embedding model doesn't understand that CDN TIRE ≈ CANADIAN TIRE. Contrastive training on KB aliases would fix this.

### Solution 5: Fix the Foursquare → Category Mapping in Enrichment (+2-3%)

Already identified in Phase 12 analysis. Feed the model our mapped category name instead of raw Foursquare labels:

```
Current:  "SHOPPERS DRUG MART. place types: Retail > Pharmacy"
Fixed:    "SHOPPERS DRUG MART. category: Healthcare & Medical"
```

The model correctly maps "Healthcare and Medicine" → Healthcare (0.855 confidence) but "Retail > Pharmacy" → Shopping (0.579). Fix the input, fix the output.

### Solution 6: Expand Rules for Structural Patterns (+2%)

Bank statement notices (PAYMENT THANK YOU, OVERLIMIT FEE, CREDIT BALANCE) are structural text, not merchant names. Rules handle them perfectly:

```yaml
- pattern: "PAYMENT.*THANK"       → Financial Services
- pattern: "CREDIT.*BALANCE"      → Financial Services
- pattern: "OVERLIMIT"            → Financial Services
- pattern: "UBER\\s+EATS"        → Food & Dining
- pattern: "MTO\\b"              → Government & Legal
- pattern: "PRESTO.*RELOAD"      → Transportation
- pattern: "OCT[-*]"             → Transportation
```

### Solution 7: Use Transaction Amount as Feature (+1-2%)

Every industry system uses amount. A $5 transaction at COSTCO is more likely Food (gas station snack) than Shopping ($200 furniture). The openai test set may have amounts — if so, adding them as a feature would help disambiguate multi-category merchants.

---

## Recommended Implementation Order

| Phase | Solution | Expected Gain | Effort | Cumulative |
|-------|----------|--------------|--------|------------|
| 12A | Fix FS label mapping (#5) + Expand rules (#6) | +4% | Low | ~78% |
| 12B | KB entries as training data (#2) | +4% | Medium | ~82% |
| 12C | LLM synthetic data generation (#1) | +3% | Medium | ~85% |
| 12D | Contrastive learning for KB retrieval (#3) | +4% | Medium | ~89% |
| 12E | Two-stage entity resolution (#4) | +2% | Low (builds on 12D) | ~91% |

**Conservative estimate: 78-85% with phases 12A-12C.**
**Optimistic estimate: 85-91% with all phases.**

The key insight from industry is that **entity resolution IS the classification**. Once you know the merchant identity, the category is a lookup. Our KB has 1.8M merchants with categories — we just need to match transactions to them better.

---

## Sources

- [Plaid Transaction Foundation Model](https://plaid.com/blog/building-transaction-foundation-model-intelligent-finance/)
- [Plaid AI-Enhanced Transaction Categories](https://plaid.com/blog/ai-enhanced-transaction-categorization/)
- [Brex Merchant Classification System](https://medium.com/brexeng/how-we-built-a-mostly-automated-system-to-solve-credit-card-merchant-classification-f9108029e59b)
- [Ntropy Transaction Enrichment](https://www.ntropy.com/blog/understanding-financial-transactions)
- [Meniga Transaction Categorisation Guide](https://www.meniga.com/resources/transaction-categorisation/)
- [Triqai: Why Transaction Categorization Fails](https://www.triqai.com/article/why-transaction-categorization-is-hard)
- [SME Transaction Classification with Synthetic Data (arXiv:2508.05425)](https://arxiv.org/html/2508.05425v1)
- [Open Banking Transaction Classification (arXiv:2504.12319)](https://arxiv.org/html/2504.12319v1)
- [Weakly Supervised Bank Transaction Classification (arXiv:2305.18430)](https://arxiv.org/abs/2305.18430)
- [Hierarchical Financial Transaction Classification (arXiv:2312.07730)](https://arxiv.org/html/2312.07730v1)
- [MCC Codes Dataset (GitHub)](https://github.com/greggles/mcc-codes)
- [Eridu: Contrastive Learning for Entity Matching (HuggingFace)](https://huggingface.co/Graphlet-AI/eridu)
- [Sentence Transformers Training Guide](https://huggingface.co/blog/how-to-train-sentence-transformers)
- [LLM Synthetic Data Survey (arXiv:2503.14023)](https://arxiv.org/html/2503.14023v2)
- [Foursquare Open Source Places (HuggingFace)](https://huggingface.co/datasets/foursquare/fsq-os-places)
- [Self-Training Survey (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0925231224016758)
- [Capital One Merchant Industry Type Imputation](https://capitalone.com/tech/machine-learning/imputing-merchant-information-in-customer-transaction-data-using-sequence-classification)
- [ExpenseSorted: ML Powers Transaction Categorization](https://www.expensesorted.com/blog/ml-bank-transaction-categorization-explained)
