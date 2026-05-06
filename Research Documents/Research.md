# Transaction Classifier Research
**Date:** 2026-03-23

## 1. Datasets

### Best option

The **mitulshah/transaction-categorization** dataset on HuggingFace is the best starting point:
- 4.5M+ records across 5 countries including Canada
- 10 categories: Food & Dining, Transportation, Shopping & Retail, Entertainment & Recreation, Healthcare & Medical, Utilities & Services, Financial Services, Income, Government & Legal, Charity & Donations
- Has Canadian examples like "Tim Hortons #456", "Petro-Canada", "Loblaws"
- **Caveat:** It's synthetically generated. The merchant templates are realistic but won't capture the messiness of real bank strings (e.g., "POS PURCHASE - 1234 TIM HORTO OTTAWA ON")

Other datasets I found:
- **mgrella/autonlp-data-bank-transaction-classification** (HuggingFace) - Italian, not useful
- **Banking77** (HuggingFace) - 13K queries, 77 intent categories, but it's about customer intents not merchant classification
- **apoorvwatsky/bank-transaction-data** (Kaggle) - Small, not Canadian
- **GoMask.ai synthetic datasets** - Varied, unclear if Canadian

### What's missing

My README has 15 categories but the mitulshah dataset only has 10. Missing:
- **Groceries** (merged into Food & Dining in most datasets)
- **Subscriptions** (merged into Utilities or Entertainment)
- **E-Transfers** (very Canada-specific)
- **Banking** (credit payments, savings installments)
- **Education** (tuition, textbooks)
- **Misc**

No public dataset matches my 15-category schema exactly. Seems like a universal problem.

### Synthetic data generation

Probably necessary. Some approaches:
- A 2025 paper (arxiv:2508.05425) used GPT-4o to rephrase real transactions while keeping the meaning. They generated up to 30x examples for minority classes.
- The mitulshah dataset itself was made with templates + country-specific patterns + random variations.
- My plan: Start with mitulshah, then add manual labels from my own bank statements (even 500-1000 real examples help), generate templates for Canadian merchants I know (OC Transpo, Presto, Rogers, Fido, etc.), and maybe use an LLM for one-time generation of realistic variations.

### What real Canadian bank strings look like

Real bank statements look nothing like clean merchant names:
```
POS PURCHASE - 1847 TIM HORTO OTTAWA ON
INTERAC PURCHASE-0423 COSTCO WHOLESALE NEPEAN ON
PAY/PAIE EMPLOYER NAME
INTERNET BILL PMT ROGERS
PREAUTHORIZED DEBIT NETFLIX.COM
INTERAC E-TRANSFER TO JOHN D
MORTGAGE PAYMENT
TFR-TO C/C SAVINGS
```

Any model has to handle this noise. This is arguably the hardest part of the whole problem.

## 2. Model approaches

### Fine-tuned DistilBERT

Take pre-trained DistilBERT (66M params), add a classification head, fine-tune on labeled transaction data.

**Accuracy estimates:**
- 85-93% on clean/synthetic data
- 75-85% on real messy bank strings
- FreeAgent got 93% with BERT in production, but they had millions of real labeled examples from years of user corrections
- One paper (arxiv:2508.05425) got 73.5% standard accuracy, 90.4% on high-confidence predictions with FinBERT

**Hardware:**
- Training: 4-8GB GPU VRAM, 30-60 min on a modern GPU
- Inference: ~5-15ms/transaction on GPU, ~50-100ms on CPU

**Pros:** Understands context and word relationships, handles misspellings well, good transfer learning from English pretraining.

**Cons:** Overkill for obvious transactions ("TIM HORTONS" is always Food), slower inference, needs GPU for training, more complex to deploy.

**My take:** Strong choice for primary model, but not worth using alone. Better paired with rules.

### Traditional ML (TF-IDF + SGD/XGBoost/SVM)

Vectorize text with TF-IDF (bag of n-grams), feed it into a classical classifier.

**Accuracy estimates:**
- One paper (arxiv:2504.12319) got 94-95% F1 with Random Forest + Word2Vec on 84 categories
- A blog post got 86% with Random Forest alone
- TF-IDF + SVM: 88-94% on cleaner data
- On messy real data: 70-85%

**Hardware:**
- CPU only, trains in minutes
- Inference: 0.08-0.12ms per transaction (35x faster than DistilBERT)

**Pros:** Extremely fast, no GPU needed, simple to implement and debug, scikit-learn is mature, supports incremental learning via partial_fit.

**Cons:** No semantic understanding ("UBER EATS" and "SKIP THE DISHES" are unrelated tokens to it), struggles with unseen merchants, feature engineering matters more.

**My take:** Excellent as a fast fallback or primary model for v1. Surprisingly competitive with transformers on this task.

### Small Local LLMs (Phi-3, Llama 3, Mistral 7B)

Either prompt engineering (zero/few-shot) or fine-tune with QLoRA/LoRA.

**Accuracy estimates:**
- Zero-shot GPT-4o got only 60.4% on SME transactions (arxiv:2508.05425), which is surprisingly bad
- Few-shot with good prompts: 65-80% (highly variable)
- Fine-tuned small LLMs can potentially match DistilBERT but at way higher compute cost

**Hardware:**
- Phi-3-mini: ~4GB VRAM quantized, fine-tuning needs 8-12GB
- Llama 3 8B / Mistral 7B: ~6GB VRAM quantized, fine-tuning needs 16-24GB
- Inference: 200-2000ms per transaction
- 1000 transactions = 3-30 minutes (vs seconds for other approaches)

**Pros:** Can handle complex reasoning ("AMAZON FRESH" -> Groceries vs "AMAZON.CA" -> Retail), can adapt to new categories without retraining.

**Cons:** Inference speed is a dealbreaker for batch classification, high VRAM, massive overkill for 90%+ of transactions, inconsistent outputs.

**My take:** Not worth it as primary classifier. Maybe useful as an "oracle" for the hardest 5% of ambiguous transactions, or for one-time data generation.

### Hybrid (Rules + ML Fallback)

Layer 1: regex/keyword rules for known merchants. Layer 2: ML model for everything else. Optionally layer 3: LLM for lowest-confidence predictions.

**Accuracy estimates:**
- Rules alone can handle 40-60% of transactions with near-100% accuracy
- Rules + ML combined: 85-95%
- FreeAgent's production system uses this and gets 93%+ on 60% of all transactions

**Basic flow:**
1. Check exact-match dictionary (merchant -> category)
2. Check regex rules ("TIM HORT*" -> Food, "INTERAC E-TRANSFER" -> E-Transfers)
3. If no rule matches, run ML model
4. If ML confidence < threshold, flag for user review

**Pros:** Highest practical accuracy for a solo developer, rules are transparent and instantly updatable, ML handles the long tail, fastest inference since most transactions hit rules first.

**Cons:** Have to maintain a rule set that can grow unwieldy, rules are brittle to new formats, two systems to maintain.

**My take:** This is the way to go. It's what production systems actually use.

### Embedding-based (Sentence Transformers + kNN)

Embed all known transactions using something like all-MiniLM-L6-v2, then at inference embed the new transaction and find k-nearest neighbors to vote on category.

**Accuracy estimates:**
- 75-88% depending on embedding model and reference dataset quality
- Works well as a few-shot approach (only need 5-10 examples per category)

**Hardware:**
- Embedding model: ~100MB
- Inference: ~10-30ms per embedding on CPU

**Pros:** No training required, naturally handles new categories, good cold-start solution, can explain classifications.

**Cons:** Accuracy ceiling is lower than fine-tuned models, short transaction strings produce less informative embeddings.

**My take:** Good for prototyping and cold-start. Could be a component in a hybrid system.

## 3. Practical numbers

### Estimated accuracy on real Canadian bank data

- Rules only: 40-60% coverage, ~98% precision
- TF-IDF + XGBoost/SVM: 70-85% (85-92% with rules)
- DistilBERT fine-tuned: 75-85% (88-95% with rules)
- Local LLM zero-shot: 55-70% (75-85% with rules)
- Embeddings + kNN: 65-80% (80-90% with rules)
- **Hybrid rules + DistilBERT: 85-95%**

### Inference speed (per transaction, CPU)

- Rules (regex): <0.01ms
- TF-IDF + XGBoost: 0.08-0.12ms
- DistilBERT: 50-100ms
- Local LLM (quantized): 200-2000ms
- Embeddings + kNN: 10-30ms

### How much training data is needed

- Rules: none, just domain knowledge
- TF-IDF + XGBoost: minimum ~500-1000 labeled examples, good at 5-10K
- DistilBERT: minimum ~1-2K labeled examples, good at 10-20K
- Local LLM fine-tuned: minimum ~500-1K (QLoRA), good at 5-10K
- Embeddings + kNN: minimum ~50-100 per category, good at 500-1K per category

### Ambiguous merchants

This is a real problem. "AMAZON" could be Retail, Groceries (Amazon Fresh), Subscriptions (Prime), or Entertainment (Prime Video).

What works:
- Rules with sub-patterns: "AMAZON FRESH" -> Groceries, "AMAZON PRIME*" -> Subscriptions, "AMAZON.CA" -> Retail (default)
- DistilBERT can learn these if training data includes them
- Amount-based features: a $4.99 Amazon charge is probably a subscription, $87.43 is probably retail

What doesn't:
- TF-IDF: "AMAZON" is one token, can't distinguish sub-types without extra features
- Simple kNN: embedding of "AMAZON.CA*XXXXXXXXX" will be close to all Amazon categories

## 4. Open-source projects I looked at

**j-convey/BankTextCategorizer** - Uses BERT, 13 categories with ~60 subcategories, trained on 62K records. Takeaway: self-hosting for privacy works, category schema has to match training data.

**robintw/BankClassify** - Naive Bayes, interactive (asks user to categorize, learns from corrections). Takeaway: the simplest approach with a correction loop can be surprisingly effective.

**eli-goodfriend/banking-class** - Logistic Regression + Naive Bayes, uses merchant name AND dollar amount as features. Takeaway: amount is a useful feature.

**Foxel05/Finance-TransactionCategorizer** - Naive Bayes Gaussian. Takeaway: even simple models provide value for personal finance.

**GlenCrawford/bank_transaction_unsupervised_clustering** - k-prototypes clustering (unsupervised). Takeaway: unsupervised approaches can bootstrap categories when you have no labels.

### Common patterns across these

1. Everyone starts with rules and adds ML later. No successful project uses ML alone.
2. Amount is an underused feature that improves accuracy.
3. User correction loops are the key differentiator. Projects that learn from corrections converge to high accuracy over time.
4. 10-15 categories is the sweet spot. More than 20 and accuracy drops significantly.
5. Data preprocessing is 50% of the work (stripping prefixes like "POS PURCHASE -", normalizing whitespace, removing card numbers).

## 5. Active learning / self-correction

### How it should work

1. Transaction comes in
2. Check rules engine first (high confidence matches)
3. If no rule matches, run ML model
4. If ML confidence > 0.8, accept the prediction
5. If ML confidence < 0.8, flag for user review
6. User corrects it
7. Store correction in training DB
8. Periodically retrain or use partial_fit

### Incremental learning options

**For Traditional ML (my v1 plan):**
- Use SGDClassifier or MultinomialNB from scikit-learn, both support partial_fit()
- Use HashingVectorizer instead of TfidfVectorizer (stateless, handles unseen vocabulary)
- Call partial_fit(new_X, new_y) after each batch of corrections
- Important: pass all possible classes= in the first call

**For DistilBERT:**
- Save corrections to a database
- Periodically fine-tune on accumulated corrections (e.g., every 100 corrections)
- Use a low learning rate (1e-5 to 2e-5) to avoid catastrophic forgetting
- Train on a mix of original data + corrections (replay buffer)

**For Embeddings + kNN:**
- Simplest to update: just add the corrected transaction to the reference set
- No retraining needed, new examples are immediately available
- Easiest active learning implementation

### Smart querying (uncertainty sampling)

Instead of flagging all low-confidence predictions, prioritize:
1. Predictions near the decision boundary (e.g., 45% Food, 40% Groceries)
2. New merchants never seen in training data
3. Transactions where the model disagrees with rules

This maximizes the value of each user correction.

### Practical tips

- Store every correction with timestamp, original prediction, confidence, and user-provided label
- Track accuracy over time to measure improvement
- Allow bulk corrections ("all Tim Hortons transactions are Food & Dining")
- Do a full retrain periodically (e.g., monthly) since it's more stable than only using partial_fit
- Version your models so you can rollback if a retrain makes things worse

## 6. My plan

Based on all this, here's what I'm going with:

### Phase 1: MVP
1. Download mitulshah/transaction-categorization dataset, filter to Canadian data
2. Remap categories to my 15-category schema (manual mapping + gap-filling)
3. Build rules engine for known Canadian merchants (Tim Hortons, Loblaws, OC Transpo, etc.)
4. Train TF-IDF + SGDClassifier as ML fallback (supports incremental learning)
5. Classification flow: rules first, ML fallback, confidence threshold for flagging

### Phase 2: Accuracy Push
1. Generate synthetic data for missing categories (E-Transfers, Banking, Education) using templates or an LLM
2. Add transaction amount as a feature alongside text
3. Fine-tune DistilBERT on the combined dataset as primary ML model
4. Keep TF-IDF + SGD as fast fallback for batch processing

### Phase 3: Active Learning
1. Build minimal correction UI (even a CLI works)
2. Store corrections in SQLite
3. Implement periodic retraining with accumulated corrections
4. Track accuracy metrics per category over time

### Tech stack
- Python 3.10+
- scikit-learn (TF-IDF, SGDClassifier, evaluation)
- HuggingFace Transformers (DistilBERT fine-tuning)
- sentence-transformers (optional, for embedding approach)
- SQLite (correction storage)
- FastAPI or Flask (classification service API)

## 7. Takeaways

1. **No single approach wins.** The best production systems are all hybrids (rules + ML).
2. **Data quality matters more than model complexity.** A well-cleaned dataset with good Canadian merchant coverage beats a fancier model trained on generic data.
3. **Start simple.** TF-IDF + SGDClassifier + rules should get me to 85%+ with minimal effort.
4. **Local LLMs aren't worth it** as a primary classifier. Too slow, too resource-hungry, and GPT-4o only hit 60% zero-shot.
5. **DistilBERT is the sweet spot** for transformer-based classification. Small enough for my hardware, accurate enough for production.
6. **The correction loop is the real advantage.** Over time, user corrections will make any model converge to high accuracy on my specific transaction patterns.
7. **Canadian-specific data is scarce.** I'll need to generate or manually label some of it.
8. **Real bank strings are messy.** Need to budget time for preprocessing and normalization.

## Sources

### datasets
- [mitulshah/transaction-categorization (HuggingFace)](https://huggingface.co/datasets/mitulshah/transaction-categorization)
- [mgrella/autonlp-bank-transaction-classification (HuggingFace)](https://huggingface.co/mgrella/autonlp-bank-transaction-classification-5521155)
- [Banking77 Dataset (HuggingFace)](https://huggingface.co/datasets/PolyAI/banking77)
- [Bank Transaction Data (Kaggle)](https://www.kaggle.com/datasets/apoorvwatsky/bank-transaction-data)
- [GoMask.ai Synthetic Datasets](https://gomask.ai/marketplace/datasets/banking-transaction-categorization-dataset)

### papers
- [Specialized text classification for Open Banking transactions (arxiv:2504.12319)](https://arxiv.org/html/2504.12319v1)
- [Categorising SME Bank Transactions with ML and Synthetic Data (arxiv:2508.05425)](https://arxiv.org/html/2508.05425v1)
- [Instruction Fine-Tuning LLMs for Financial Text Classification (ACM)](https://dl.acm.org/doi/10.1145/3706119)
- [Comparing BERT against traditional ML for text classification (arxiv:2005.13012)](https://arxiv.org/abs/2005.13012)

### open-source projects
- [j-convey/BankTextCategorizer](https://github.com/j-convey/BankTextCategorizer)
- [robintw/BankClassify](https://github.com/robintw/BankClassify)
- [eli-goodfriend/banking-class](https://github.com/eli-goodfriend/banking-class)
- [Foxel05/Finance-TransactionCategorizer](https://github.com/Foxel05/Finance-TransactionCategorizer)
- [GlenCrawford/bank_transaction_unsupervised_clustering](https://github.com/GlenCrawford/bank_transaction_unsupervised_clustering)
- [LynnN-98/classification_Transaction_NLP](https://github.com/LynnN-98/classification_Transaction_NLP)

### other references
- [FreeAgent: Fine-Tuning BERT for Multiclass Categorisation](https://engineering.freeagent.com/2021/09/15/fine-tuning-bert-for-multiclass-categorisation-with-amazon-sagemaker/)
- [FreeAgent: Evolving Banking Automation with ML](https://freeagent.medium.com/evolving-banking-automation-how-we-developed-our-machine-learning-capabilities-to-supercharge-93534d4f02dc)
- [Hybrid Rules + ML Transaction Categorization](https://mvvenrooij.nl/2024/12/categorizing-transactions-with-machine-learning-and-rules/)
- [Deep Learning Hybrid Transaction Classification (Journal of Big Data)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00651-x)
- [CRIF Categorisation Engine Challenges](https://crif.co.uk/news-events/blog/categorisation-engine-main-challenges-of-applying-ml-and-ai/)
- [Text Classification with DistilBERT (Kaggle)](https://www.kaggle.com/code/pritishmishra/text-classification-with-distilbert-92-accuracy)
- [DistilBERT Banking77 Classification (HuggingFace)](https://huggingface.co/nickprock/distilbert-base-uncased-banking77-classification)
- [Scikit-learn Incremental Learning](https://scikit-learn.org/stable/computing/scaling_strategies.html)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Few-Shot Learning with SBERT (Medium)](https://medium.com/analytics-vidhya/few-shot-learning-using-sbert-95f8b08248bf)
- [Nubank: Automatic Retraining for ML Models](https://building.nubank.com/automatic-retraining-for-machine-learning-models/)
