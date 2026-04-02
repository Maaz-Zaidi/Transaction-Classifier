# Phase 6: Root-cause fix — abbreviation augmentation + CANINE

**Date:** 2026-04-02

## Problem diagnosis

The fine-tune model (all-MiniLM-L6-v2) gets 92.4% on training test data but only 58.5% on real Canadian bank transactions. The 33.9-point gap has two root causes:

### Root cause 1: WordPiece tokenization destroys abbreviated text

Bank statements truncate merchant names to ~25 characters per Visa Merchant Data Standards. WordPiece tokenizes "WMT SUPRCTR" as `["w", "##mt", "su", "##pr", "##ct", "##r"]` — meaningless fragments the model never saw during pretraining. The embedding layer produces garbage vectors for these tokens. This is NOT fixable by fine-tuning alone because the problem is at the tokenization layer, below the model.

### Root cause 2: Training data has no abbreviation patterns

The mitulshah dataset contains full merchant names ("Walmart Supercenter") but never abbreviated forms ("WMT SUPRCTR"). The model has zero signal for how abbreviations map to categories. Even if tokenization worked perfectly, the model hasn't learned that truncated/compressed names are valid inputs.

## Solution: Two-phase approach

### Phase 6a: Abbreviation augmentation (same MiniLM model)

**Goal:** Teach the model that abbreviations exist by training on augmented data.

**Method:** For each training example, generate 3-5 abbreviated variants:
- Truncate at random length (15-25 chars)
- Remove vowels: "WALMART" -> "WLMRT"
- Consonant compression: "SUPERCENTER" -> "SUPRCNTR"
- Random word truncation: "WALMART SUPERCENTER" -> "WMT SUPERCENTER"
- Drop trailing words: "CANADIAN TIRE STORE" -> "CANADIAN TIRE"

Train on all variants with the same label. The model learns the *pattern* of abbreviation, not specific instances.

Also generate Canadian-specific synthetic data (CRA, MTO, RCSS, CHATIME, etc.) and abbreviate those too.

**Expected outcome:** Fine-tune accuracy from 58.5% -> ~70-75%. Validates the approach without changing model architecture.

### Phase 6b: CANINE character-level model

**Goal:** Eliminate the tokenization problem entirely.

**Model:** `google/canine-s` (~132M params, character-level transformer, no tokenizer)
- Processes raw Unicode characters directly
- "WMT SUPRCTR" -> `['W','M','T',' ','S','U','P','R','C','T','R']` — every character is meaningful
- Uses stride-4 downsampling convolutions to keep efficiency
- Supports `AutoModelForSequenceClassification` (same HuggingFace Trainer workflow)

**Training:** Same augmented dataset from Phase 6a. CANINE + abbreviation-augmented data attacks both root causes simultaneously.

**Expected outcome:** Fine-tune accuracy -> 80%+

## Research sources

- [CANINE Paper (Clark et al., TACL 2022)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00448/109284)
- [BPE-Dropout (Provilkov et al., ACL 2020)](https://arxiv.org/abs/1910.13267)
- [Tokenization Falling Short (ACL Findings 2024)](https://arxiv.org/abs/2406.11687)
- [Data Augmentation for Robustness (MDPI 2023)](https://www.mdpi.com/1999-4893/16/1/59)
- [CharBERT: Character-aware Pre-trained LM](https://arxiv.org/abs/2011.01513)
- [SME Transaction Classification with Synthetic Data (arXiv 2508.05425)](https://arxiv.org/html/2508.05425v1)
- [Visa Merchant Data Standards Manual](https://usa.visa.com/content/dam/VCOM/download/merchants/visa-merchant-data-standards-manual.pdf)

## Success criteria

- Phase 6a: Fine-tune-only accuracy >= 70% on the 505 Codex-labeled test set
- Phase 6b: Fine-tune-only accuracy >= 80% on the same test set
- Both must maintain CPU inference capability
