# Phase 3: SetFit (MiniLM) results
**Date:** 2026-03-28

## Architecture

Same two-stage pipeline, with SetFit replacing FastText as the Stage 2 classifier:

**Stage 1: Direction detection** (deterministic prefix rules)
- Income / Government resolved directly
- Expenses passed to Stage 2

**Stage 2: Rules engine -> SetFit classifier**
- Known merchants matched by rules
- Unknown merchants classified by SetFit (sentence-transformer embeddings)

## SetFit model

- **Base model**: `sentence-transformers/all-MiniLM-L6-v2` (22M params)
- **Training**: contrastive learning on 8,000 stratified samples from mitulshah dataset
- **Hyperparams**: 1 epoch, batch_size=32, num_iterations=20 (320K contrastive pairs)
- **Training time**: 54 min on CPU (no CUDA, PyTorch CPU-only installed)
- **Val accuracy**: 98% (5K sample), avg confidence 0.986

### Why SetFit works

The key idea was confirmed: **pre-trained sentence transformer embeddings already understand real-world concepts** that FastText/SGD can't learn from synthetic training data alone.

MiniLM was pre-trained on 1B+ sentence pairs from the internet. It already knows:
- "shawarma", "sushi", "chicken" are food concepts
- "supermarket", "store", "mart" are retail concepts
- "payment", "credit", "loan" are financial concepts

FastText/SGD only know what the synthetic mitulshah dataset taught them, which leads to the Income bias for unknown proper noun merchants.

## Results on real bank data (497 Gemini-labeled descriptions)

### Overall progression

- Phase 1 (SGD): 43.4%
- Phase 1b (pymupdf+SGD): 53.7%, ML-only 28.4%, flagged 51.9%
- Phase 2 (direction+FastText): 55.7%, ML-only 14.8%, flagged 14.3%
- **Phase 3 (direction+SetFit): 80.5%, ML-only 66.7%, flagged 0.8%**

### By source
- Direction: 100.0% (53/53)
- Rules: 91.3% (189/207)
- SetFit: 66.7% (158/237)

### By account type
- MasterCard: 54.1% -> **79.5%**
- Chequing: 58.2% -> **82.0%**

### By category
- Income: 97.8% (unchanged, direction detection handles this)
- Healthcare & Medical: 100.0% (unchanged)
- Entertainment & Recreation: 88.6% (unchanged)
- Transportation: 83.3% (unchanged)
- **Food & Dining: 45.4% -> 83.9%** (+38.5%)
- **Financial Services: 35.1% -> 91.2%** (+56.1%)
- **Shopping & Retail: 59.2% -> 74.6%** (+15.4%)
- Government & Legal: 54.5% (unchanged)
- Utilities & Services: 31.6% -> 34.2% (+2.6%)
- Charity & Donations: 0.0% (unchanged)

## Analysis

### What worked
1. **Pre-trained embeddings killed the Income bias.** SetFit doesn't predict Income for food merchants because MiniLM already knows food concepts from pre-training. FastText's 14.8% -> SetFit's 66.7% on unknown merchants.
2. **Food & Dining massive jump** (+38.5%). Most food merchants are local/unknown and FastText was classifying them as Income. SetFit gets them right.
3. **Financial Services huge jump** (+56.1%). SetFit understands payment/credit/loan concepts from pre-training.
4. **Almost nothing flagged** (0.8%). SetFit is very confident in its predictions.

### What's still not working
1. **Utilities & Services still low** (34.2%). OPENAI *CHATGPT gets classified as Transportation (confused by the abbreviation patterns). Barbershop goes to Shopping instead of Utilities.
2. **Convenience stores misclassified.** MACS CONV. STORE -> Shopping & Retail instead of Food & Dining. Gemini labeled it Food, which is debatable for convenience stores.
3. **Amazon Marketplace** -> Transportation (wrong). The AMZN MKTP pattern confuses SetFit.
4. **Rules engine conflicts**: Shoppers Drug Mart -> Healthcare (rules) but Gemini says Shopping. Amazon Web Services -> Shopping (rules) but should be Utilities.
5. **Charity & Donations**: 0%, only 1 sample in SetFit-only, and e-transfers to charities get caught by rules as Financial Services.

### Why the last 19.5% is wrong
1. **Rules mismatches** (~8 errors): Shoppers Drug Mart, Amazon Web Services, e-transfers to charities/auto shops matched to wrong category
2. **Software/SaaS not in training data**: OPENAI, Kindle Unlimited aren't represented in mitulshah categories
3. **Ambiguous categories**: is a convenience store Food or Shopping? Is a barber Utilities or Shopping?
4. **Abbreviations**: AMZN MKTP, RFBT-RIDEAU CENTRE, even MiniLM struggles with extreme abbreviations

### Next steps
- **Fix rules**: update for Shoppers Drug Mart, Amazon Web Services, tuition payments
- **Add direction detection**: Mobile cheque deposit -> Income, ATM withdrawal -> Financial Services
- **Consider CUDA PyTorch**: training would drop from 54 min to ~5 min with RTX 3060
- **Try more training data**: 8K samples is very small, try 20K-50K with reduced num_iterations

## Model registry

- Model 1 (SGD): val 98%, real 53.7%, ML-only 28.4% - Phase 1 baseline
- Model 2 (FastText): val 99%, real 55.7%, ML-only 14.8% - Phase 2, Income bias on unknowns
- **Model 3 (SetFit): val 98%, real 80.5%, ML-only 66.7% - Phase 3, pre-trained embeddings work**
