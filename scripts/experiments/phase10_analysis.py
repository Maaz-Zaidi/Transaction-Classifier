#!/usr/bin/env python
"""Phase 10 diagnostic analysis: KB coverage, category mapping gaps, ML enrichment quality.

This script runs the full set of experiments used to identify the Phase 10 bottlenecks.
It is meant to be re-run after fixes to measure progress.

Usage:
    python scripts/experiments/phase10_analysis.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd

from transaction_classifier.config import settings
from transaction_classifier.data.preprocess import clean_transaction
from transaction_classifier.knowledge.foursquare import map_foursquare_labels
from transaction_classifier.knowledge.merchant_kb import (
    MerchantKnowledgeBase,
    build_enriched_transaction_text,
)
from transaction_classifier.knowledge.token_analyzer import TokenAnalyzer
from transaction_classifier.models.ensemble import Ensemble
from transaction_classifier.models.finetune_model import FineTuneModel
from transaction_classifier.rules.engine import RulesEngine


def load_test_data() -> pd.DataFrame:
    codex_path = settings.data_dir / "real" / "codex_labeled.csv"
    df = pd.read_csv(codex_path)
    df = df[df["codex_category"].notna() & (df["codex_category"] != "")].copy()
    return df


def load_pipeline():
    rules = RulesEngine()
    kb = MerchantKnowledgeBase()
    store_path = (
        settings.knowledge_store_path
        if settings.knowledge_store_path.exists()
        else settings.knowledge_base_path
    )
    kb.load(store_path)
    ft = FineTuneModel()
    ft.load(settings.model_dir / "finetune")
    analyzer = TokenAnalyzer()
    ens = Ensemble(rules_engine=rules, finetune_model=ft, knowledge_base=kb)
    return ens, kb, ft, analyzer


def experiment_kb_coverage(df, kb, analyzer):
    """Check KB match rate, category mapping accuracy, and gaps."""
    print("=" * 80)
    print("EXPERIMENT: KB COVERAGE AND CATEGORY MAPPING")
    print("=" * 80)

    correct, wrong, no_cat, miss = [], [], [], []

    for _, row in df.iterrows():
        cleaned = clean_transaction(row["raw_example"])
        true_cat = row["codex_category"]
        match, _ = kb.search_with_tokens(cleaned, token_analyzer=analyzer, min_similarity=0.58)

        if not match:
            miss.append({"cleaned": cleaned, "true": true_cat})
            continue
        if not match.entry.mapped_category:
            no_cat.append({
                "cleaned": cleaned, "true": true_cat,
                "kb_name": match.entry.canonical_name,
                "raw_labels": match.entry.raw_category_labels[:3],
                "strategy": match.strategy,
            })
            continue
        if match.entry.mapped_category == true_cat:
            correct.append({"cleaned": cleaned, "true": true_cat})
        else:
            wrong.append({
                "cleaned": cleaned, "true": true_cat,
                "kb_cat": match.entry.mapped_category,
                "kb_name": match.entry.canonical_name,
                "raw_labels": match.entry.raw_category_labels[:3],
                "strategy": match.strategy,
            })

    total = len(df)
    print(f"Total: {total}")
    print(f"KB match with correct category: {len(correct)}")
    print(f"KB match with wrong category:   {len(wrong)}")
    print(f"KB match but no category:       {len(no_cat)}")
    print(f"KB miss (no match at all):      {len(miss)}")
    print()

    ceiling_with_cat = len(correct) + len(wrong)
    ceiling_all = ceiling_with_cat + len(no_cat)
    print(f"CEILING: if all mapped categories correct: {ceiling_with_cat}/{total} = {ceiling_with_cat/total*100:.1f}%")
    print(f"CEILING: if all entries had correct categories: {ceiling_all}/{total} = {ceiling_all/total*100:.1f}%")
    print()

    # wrong category breakdown
    wrong_groups = Counter()
    for d in wrong:
        wrong_groups[(d["true"], d["kb_cat"])] += 1
    print(f"Wrong KB category breakdown ({len(wrong)} total):")
    for (true, kb_cat), count in wrong_groups.most_common(15):
        print(f"  true={true:30s} kb={kb_cat:25s}  x{count}")
    print()

    # no-category reason analysis
    no_labels = sum(1 for d in no_cat if not d["raw_labels"])
    has_labels = [d for d in no_cat if d["raw_labels"]]
    print(f"No-category entries: {len(no_cat)}")
    print(f"  No Foursquare labels at all: {no_labels}")
    print(f"  Has labels but no mapping:   {len(has_labels)}")

    unmapped_labels = Counter()
    for d in has_labels:
        for label in d["raw_labels"]:
            cat, conf = map_foursquare_labels([label])
            if cat is None:
                unmapped_labels[label] += 1
    print(f"\nTop unmapped Foursquare labels:")
    for label, count in unmapped_labels.most_common(20):
        print(f"  {label:60s} x{count}")

    return correct, wrong, no_cat, miss


def experiment_strategy_quality(df, kb, analyzer):
    """Break down match quality by retrieval strategy."""
    print()
    print("=" * 80)
    print("EXPERIMENT: RETRIEVAL STRATEGY QUALITY")
    print("=" * 80)

    strategies = Counter()
    strat_sims = defaultdict(list)
    strat_cat_correct = defaultdict(int)
    strat_cat_total = defaultdict(int)

    for _, row in df.iterrows():
        cleaned = clean_transaction(row["raw_example"])
        true_cat = row["codex_category"]
        match, _ = kb.search_with_tokens(cleaned, token_analyzer=analyzer, min_similarity=0.58)
        if not match:
            continue
        strategies[match.strategy] += 1
        strat_sims[match.strategy].append(match.similarity)
        if match.entry.mapped_category:
            strat_cat_total[match.strategy] += 1
            if match.entry.mapped_category == true_cat:
                strat_cat_correct[match.strategy] += 1

    for strat, count in strategies.most_common():
        sims = strat_sims[strat]
        correct = strat_cat_correct.get(strat, 0)
        total = strat_cat_total.get(strat, 0)
        acc = correct / total * 100 if total else 0
        print(
            f"  {strat:15s}: {count:4d} matches"
            f"  avg_sim={sum(sims)/len(sims):.3f}"
            f"  min_sim={min(sims):.3f}"
            f"  cat_acc={acc:.0f}% ({correct}/{total})"
        )


def experiment_ml_enrichment(ft):
    """Test ML model response to enriched vs raw text."""
    print()
    print("=" * 80)
    print("EXPERIMENT: ML MODEL ENRICHMENT SENSITIVITY")
    print("=" * 80)

    test_cases = [
        ("COSTCO", "COSTCO. external metadata: warehouse retailer. descriptor context: gas station, fuel"),
        ("SHOPPERS DRUG MART", "SHOPPERS DRUG MART. external metadata: pharmacy and drugstore chain"),
        ("AMAZON", "AMAZON. external metadata: online retailer. descriptor context: digital content downloads"),
        ("WMT SUPRCTR", "WMT SUPRCTR. external metadata: walmart supercenter grocery and retail"),
    ]

    for raw, enriched in test_cases:
        raw_cat, raw_conf = ft.predict([raw])[0]
        enr_cat, enr_conf = ft.predict([enriched])[0]
        print(f"  {raw:25s} raw={raw_cat:25s} ({raw_conf:.2f})  enriched={enr_cat:25s} ({enr_conf:.2f})")


def experiment_error_analysis(df, ens):
    """Run full ensemble and break down errors by source."""
    print()
    print("=" * 80)
    print("EXPERIMENT: ERROR ANALYSIS BY SOURCE")
    print("=" * 80)

    results = ens.classify_batch(df["raw_example"].tolist())
    errors_by_source = defaultdict(list)
    total_by_source = Counter()
    correct_by_source = Counter()

    for i, (r, true) in enumerate(zip(results, df["codex_category"])):
        total_by_source[r.source] += 1
        if r.category == true:
            correct_by_source[r.source] += 1
        else:
            errors_by_source[r.source].append({
                "cleaned": r.cleaned[:40],
                "predicted": r.category,
                "true": true,
                "confidence": r.confidence,
            })

    overall_correct = sum(correct_by_source.values())
    print(f"Overall: {overall_correct}/{len(df)} = {overall_correct/len(df)*100:.1f}%\n")

    for source in sorted(total_by_source, key=lambda s: -total_by_source[s]):
        total = total_by_source[source]
        correct = correct_by_source[source]
        errs = errors_by_source[source]
        print(f"  {source:20s}: {correct}/{total} correct ({correct/total*100:.1f}%), {len(errs)} errors")

    for source in ["finetune_metadata", "finetune", "knowledge_base"]:
        errs = errors_by_source.get(source, [])
        if not errs:
            continue
        print(f"\n  Top {source} confusions:")
        confusion = Counter()
        for e in errs:
            confusion[(e["true"], e["predicted"])] += 1
        for (true, pred), count in confusion.most_common(8):
            print(f"    {true:30s} -> {pred:25s} x{count}")


def main():
    print("Loading test data and pipeline...")
    df = load_test_data()
    ens, kb, ft, analyzer = load_pipeline()
    print(f"Loaded {len(df)} test samples, KB has {kb.size} entries\n")

    experiment_kb_coverage(df, kb, analyzer)
    experiment_strategy_quality(df, kb, analyzer)
    experiment_ml_enrichment(ft)
    experiment_error_analysis(df, ens)


if __name__ == "__main__":
    main()
