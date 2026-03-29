#!/usr/bin/env python
"""Evaluate fine-tuned MiniLM model against real bank data.

Uses pre-existing Gemini labels from ocr_labeled.csv.
Compares against all previous model versions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from transaction_classifier.config import settings
from transaction_classifier.models.ensemble import Ensemble
from transaction_classifier.models.finetune_model import FineTuneModel
from transaction_classifier.rules.engine import RulesEngine

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "real"


def main():
    labeled_path = OUTPUT_DIR / "ocr_labeled.csv"
    if not labeled_path.exists():
        print(f"ERROR: {labeled_path} not found. Run evaluate_real_ocr.py first.")
        sys.exit(1)

    df = pd.read_csv(labeled_path)
    df = df[df["gemini_category"].notna()].copy()
    print(f"Loaded {len(df)} Gemini-labeled descriptions")

    # Load fine-tuned model
    ft_path = settings.model_dir / "finetune"
    if not (ft_path / "model").exists():
        print(f"ERROR: No fine-tuned model at {ft_path}. Run train_finetune.py first.")
        sys.exit(1)

    print("Loading fine-tuned MiniLM...")
    ft_model = FineTuneModel()
    ft_model.load(ft_path)

    # Build ensemble
    rules_engine = RulesEngine()
    ensemble = Ensemble(rules_engine=rules_engine, finetune_model=ft_model)

    print("Running ensemble (direction + rules + fine-tuned MiniLM)...")
    results = ensemble.classify_batch(df["description"].tolist())

    df["ft_category"] = [r.category for r in results]
    df["ft_confidence"] = [r.confidence for r in results]
    df["ft_source"] = [r.source for r in results]
    df["ft_flagged"] = [r.flagged_for_review for r in results]
    df["ft_match"] = df["ft_category"] == df["gemini_category"]

    # Report
    report = []
    report.append("=" * 70)
    report.append("REAL DATA EVALUATION - Phase 4 (Direction + Rules + Fine-tuned MiniLM)")
    report.append("=" * 70)
    report.append(f"Gemini-labeled: {len(df)}")
    report.append(f"Overall accuracy: {df['ft_match'].mean():.1%} "
                  f"({df['ft_match'].sum()}/{len(df)})")

    report.append(f"\nComparison across models:")
    report.append(f"Phase 1  (pdfplumber + SGD):          43.4%")
    report.append(f"Phase 1b (pymupdf + SGD):             53.7%")
    report.append(f"Phase 2  (direction+rules+FT):        55.7%")
    report.append(f"Phase 3  (direction+rules+SetFit):    80.5%")
    report.append(f"Phase 4  (direction+rules+FineTune):  {df['ft_match'].mean():.1%}")

    # By source
    report.append(f"\nBy classification source:")
    for source in ["direction", "rules", "finetune"]:
        subset = df[df["ft_source"] == source]
        if len(subset):
            acc = subset["ft_match"].mean()
            report.append(f"  {source}: {acc:.1%} ({subset['ft_match'].sum()}/{len(subset)})")

    # By account type
    report.append(f"\nBy account type:")
    for acct in ["mastercard", "chequing"]:
        subset = df[df["account_type"] == acct]
        if len(subset):
            acc = subset["ft_match"].mean()
            report.append(f"  {acct}: {acc:.1%} ({subset['ft_match'].sum()}/{len(subset)})")

    # Flagged
    report.append(f"\nFlagged for review: {df['ft_flagged'].sum()} "
                  f"({df['ft_flagged'].mean():.1%})")

    # Category breakdown
    report.append(f"\nAccuracy by category:")
    report.append(f"{'Category':30s} | {'Phase 3 (SF)':12s} | {'Phase 4 (FT)':12s}")
    report.append("-" * 60)

    phase3_by_cat = {
        "Income": "97.8%", "Healthcare & Medical": "100.0%",
        "Entertainment & Recreation": "88.6%", "Transportation": "83.3%",
        "Food & Dining": "83.9%", "Shopping & Retail": "74.6%",
        "Financial Services": "91.2%", "Government & Legal": "54.5%",
        "Utilities & Services": "34.2%", "Charity & Donations": "0.0%",
    }

    for cat in sorted(df["gemini_category"].unique()):
        subset = df[df["gemini_category"] == cat]
        acc = subset["ft_match"].mean()
        prev = phase3_by_cat.get(cat, "N/A")
        report.append(f"  {cat:30s} | {prev:12s} | {acc:.1%}")

    # ML-only accuracy
    ml_only = df[df["ft_source"] == "finetune"]
    if len(ml_only):
        report.append(f"\nFine-tune-only breakdown (unknown merchants):")
        report.append(f"Total: {ml_only['ft_match'].mean():.1%} "
                      f"({ml_only['ft_match'].sum()}/{len(ml_only)})")
        report.append(f"  (SetFit was: 66.7%, FastText was: 14.8%, SGD was: 28.4%)")
        report.append(f"\nBy category:")
        for cat in sorted(ml_only["gemini_category"].unique()):
            subset = ml_only[ml_only["gemini_category"] == cat]
            acc = subset["ft_match"].mean()
            report.append(f"  {cat:30s}: {acc:.1%} ({subset['ft_match'].sum()}/{len(subset)})")

    # Mismatches
    mismatches = df[~df["ft_match"]].copy()
    if len(mismatches):
        report.append(f"\nMISMATCHES ({len(mismatches)}):")
        report.append(
            f"{'Description':50s} | {'Model':25s} | "
            f"{'Gemini':25s} | Src      | Conf"
        )
        report.append("-" * 145)
        for _, row in mismatches.iterrows():
            desc = str(row["description"])[:50]
            report.append(
                f"{desc:50s} | {row['ft_category']:25s} | "
                f"{row['gemini_category']:25s} | {row['ft_source']:8s} | "
                f"{row['ft_confidence']:.2f}"
            )

    report_text = "\n".join(report)
    print(f"\n{report_text}")

    # Save
    report_path = OUTPUT_DIR / "finetune_eval_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    df.to_csv(OUTPUT_DIR / "finetune_labeled.csv", index=False)
    print(f"\nReport: {report_path}")
    print(f"Data: {OUTPUT_DIR / 'finetune_labeled.csv'}")


if __name__ == "__main__":
    main()
