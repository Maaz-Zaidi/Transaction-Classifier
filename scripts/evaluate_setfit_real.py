#!/usr/bin/env python
"""Evaluate SetFit model against real bank data.

Uses pre-existing Gemini labels from ocr_labeled.csv (no re-labeling needed).
Runs the SetFit ensemble and compares against previous model versions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from transaction_classifier.config import settings
from transaction_classifier.models.ensemble import Ensemble
from transaction_classifier.models.setfit_model import SetFitTransactionModel
from transaction_classifier.rules.engine import RulesEngine

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "real"


def main():
    # Load pre-labeled data (Gemini labels from previous evaluation)
    labeled_path = OUTPUT_DIR / "ocr_labeled.csv"
    if not labeled_path.exists():
        print(f"ERROR: {labeled_path} not found. Run evaluate_real_ocr.py first.")
        sys.exit(1)

    df = pd.read_csv(labeled_path)
    print(f"Loaded {len(df)} descriptions")

    # Keep only rows with Gemini labels
    df = df[df["gemini_category"].notna()].copy()
    print(f"Gemini-labeled: {len(df)}")

    # Load SetFit model
    setfit_path = settings.model_dir / "setfit"
    if not (setfit_path / "setfit_model").exists():
        print(f"ERROR: No SetFit model at {setfit_path}. Run train_setfit.py first.")
        sys.exit(1)

    print("\nLoading SetFit model...")
    setfit_model = SetFitTransactionModel()
    setfit_model.load(setfit_path)

    # Build ensemble with SetFit
    rules_engine = RulesEngine()
    ensemble = Ensemble(rules_engine=rules_engine, setfit_model=setfit_model)

    # Classify
    print("Running ensemble (direction + rules + SetFit)...")
    results = ensemble.classify_batch(df["description"].tolist())

    df["sf_category"] = [r.category for r in results]
    df["sf_confidence"] = [r.confidence for r in results]
    df["sf_source"] = [r.source for r in results]
    df["sf_flagged"] = [r.flagged_for_review for r in results]

    df["sf_match"] = df["sf_category"] == df["gemini_category"]

    # Report
    report = []
    report.append("=" * 70)
    report.append("REAL DATA EVALUATION - Phase 3 (Direction + Rules + SetFit)")
    report.append("=" * 70)
    report.append(f"Gemini-labeled: {len(df)}")
    report.append(f"Overall accuracy: {df['sf_match'].mean():.1%} "
                  f"({df['sf_match'].sum()}/{len(df)})")

    report.append(f"\nComparison across models:")
    report.append(f"Phase 1  (pdfplumber + SGD):       43.4%")
    report.append(f"Phase 1b (pymupdf + SGD):          53.7%")
    report.append(f"Phase 2  (direction+rules+FT):     55.7%")
    report.append(f"Phase 3  (direction+rules+SetFit): {df['sf_match'].mean():.1%}")

    # By source
    report.append(f"\nBy classification source:")
    for source in ["direction", "rules", "setfit"]:
        subset = df[df["sf_source"] == source]
        if len(subset):
            acc = subset["sf_match"].mean()
            report.append(f"  {source}: {acc:.1%} ({subset['sf_match'].sum()}/{len(subset)})")

    # By account type
    report.append(f"\nBy account type:")
    for acct in ["mastercard", "chequing"]:
        subset = df[df["account_type"] == acct]
        if len(subset):
            acc = subset["sf_match"].mean()
            report.append(f"  {acct}: {acc:.1%} ({subset['sf_match'].sum()}/{len(subset)})")

    # Flagged
    report.append(f"\nFlagged for review: {df['sf_flagged'].sum()} "
                  f"({df['sf_flagged'].mean():.1%})")

    # Category breakdown
    report.append(f"\nAccuracy by category:")
    report.append(f"{'Category':30s} | {'Phase 2 (FT)':12s} | {'Phase 3 (SF)':12s}")
    report.append("-" * 60)

    # Phase 2 results for comparison (from Phase2_Training_Results.md)
    phase2_by_cat = {
        "Income": "97.8%",
        "Healthcare & Medical": "100.0%",
        "Entertainment & Recreation": "88.6%",
        "Transportation": "83.3%",
        "Shopping & Retail": "59.2%",
        "Government & Legal": "54.5%",
        "Financial Services": "35.1%",
        "Food & Dining": "45.4%",
        "Utilities & Services": "31.6%",
        "Charity & Donations": "0.0%",
    }

    for cat in sorted(df["gemini_category"].unique()):
        subset = df[df["gemini_category"] == cat]
        acc = subset["sf_match"].mean()
        prev = phase2_by_cat.get(cat, "N/A")
        report.append(f"  {cat:30s} | {prev:12s} | {acc:.1%}")

    # SetFit-only accuracy (how well does it do on unknown merchants?)
    setfit_only = df[df["sf_source"] == "setfit"]
    if len(setfit_only):
        report.append(f"\nSetFit-only breakdown (unknown merchants):")
        report.append(f"Total: {setfit_only['sf_match'].mean():.1%} "
                      f"({setfit_only['sf_match'].sum()}/{len(setfit_only)})")
        report.append(f"  (FastText was: 14.8%, SGD was: 28.4%)")
        report.append(f"\nBy category:")
        for cat in sorted(setfit_only["gemini_category"].unique()):
            subset = setfit_only[setfit_only["gemini_category"] == cat]
            acc = subset["sf_match"].mean()
            report.append(f"  {cat:30s}: {acc:.1%} ({subset['sf_match'].sum()}/{len(subset)})")

    # Mismatches
    mismatches = df[~df["sf_match"]].copy()
    if len(mismatches):
        report.append(f"\nMISMATCHES ({len(mismatches)}):")
        report.append(
            f"{'Description':50s} | {'Model':25s} | "
            f"{'Gemini':25s} | Src    | Conf"
        )
        report.append("-" * 140)
        for _, row in mismatches.iterrows():
            desc = str(row["description"])[:50]
            report.append(
                f"{desc:50s} | {row['sf_category']:25s} | "
                f"{row['gemini_category']:25s} | {row['sf_source']:6s} | "
                f"{row['sf_confidence']:.2f}"
            )

    report_text = "\n".join(report)
    print(f"\n{report_text}")

    # Save
    report_path = OUTPUT_DIR / "setfit_eval_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    df.to_csv(OUTPUT_DIR / "setfit_labeled.csv", index=False)
    print(f"\nReport: {report_path}")
    print(f"Data: {OUTPUT_DIR / 'setfit_labeled.csv'}")


if __name__ == "__main__":
    main()
