# Phase 6 Results: Augmentation + CANINE Both Failed

Date: 2026-04-03

## What I tried

Two approaches to fix the 58.5% ML-only accuracy on real Canadian bank transactions:

**Phase 6a (Augmented MiniLM):** Took the same MiniLM base, trained it on abbreviation-augmented data. 50K stratified base samples from mitulshah, ran augment_dataset() with 3 variants each (vowel removal, consonant compression, truncation, acronyms), giving 173,761 total training examples. 4 epochs, batch 64, lr 2e-5.

**Phase 6b (CANINE):** Swapped MiniLM entirely for Google's CANINE-s (132M params, character-level transformer, no WordPiece). Same augmented data pipeline. 5 epochs, batch 32, lr 5e-5.

## Results

Both approaches regressed badly.

- Baseline: 65.3% overall, 58.5% ML-only
- Augmented MiniLM: 52.5% overall, 41.2% ML-only (17-point regression)
- CANINE: 39.4% overall, 23.7% ML-only (35-point regression)

Both models were confidently wrong. Augmented had 0.90 avg confidence at 41% accuracy. CANINE had 0.97 avg confidence at 24% accuracy. They memorized synthetic patterns and generalized to nothing.

Food & Dining (the largest category at 242 samples) went from 82.6% baseline to 58.3% augmented to 35.5% CANINE.

## What I learned

The original hypothesis was wrong. I thought the problem was WordPiece tokenization destroying abbreviated merchant names. The data proved otherwise:

1. Augmentation poisoned the model because the abbreviation variants of synthetic merchant names don't match how real banks actually truncate. The preprocessing pipeline already handles most real-world truncation.

2. CANINE's character-level processing didn't help because the problem isn't tokenization at all. MiniLM's sentence-level pre-training gives it better semantic understanding of merchant names than CANINE's character-level pre-training, even if CANINE handles the characters more cleanly.

3. The real bottleneck is domain mismatch. The synthetic training merchants (mitulshah dataset) simply don't represent what shows up on a Canadian bank statement. No amount of clever augmentation or alternative architectures can fix a data distribution problem.

## Codex Debate (3 rounds)

I ran a structured 3-round debate with Codex (GPT-5.4) about alternative approaches. Summary of each round:

**Round 1:** I proposed a semantic preprocessing pipeline (abbreviation expansion via external knowledge, word-level semantic understanding, RAG for brand lookup). Codex raised 5 objections: solving the wrong problem, vague external data dependency, char n-gram unreliability, error propagation without uncertainty strategy, and over-engineering.

**Round 2:** I countered that our preprocessing already handles artifacts (the 376 unmatched strings ARE clean merchant names), that abbreviation expansion isn't the main lever (many failures are readable names like FARM BOY), and proposed an ensemble approach (add semantic signal, don't overwrite). Codex agreed the ensemble idea was the strongest part, recommended a "frozen semantic scorer using pretrained embeddings against per-class prototypes" as a gated low-confidence fallback.

**Round 3:** I asked for concrete implementation details. Codex was blunt: the frozen semantic scorer would help on descriptive names (REAL FRUIT BUBBLE TEA) but fail on opaque brands (LCBO, CHATIME, FIVE GUYS). "Merchant classification is mostly entity resolution plus lexical matching, not pure semantics. The step change comes from real labeled merchant names, not from centroids."

## Conclusion

The problem is entity knowledge, not semantic reasoning. The model doesn't fail because it can't understand "supermarket" means food. It fails because it doesn't know what LCBO, CHATIME, or FARM BOY are. Every production fintech system (Plaid, Ntropy, Spade) solves this with a merchant knowledge base, not pure ML. That's the direction for Phase 7.
