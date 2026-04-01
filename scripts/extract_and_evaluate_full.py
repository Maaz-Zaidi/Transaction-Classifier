#!/usr/bin/env python
"""Extract, label, deduplicate, and evaluate on full bank statement corpus.

1. Extracts descriptions from ALL MasterCard + Noirt/chequing PDFs
2. Deduplicates (labels and evaluates on unique descriptions only)
3. Labels unique descriptions with Gemini
4. Runs the ensemble classifier (direction + rules + fine-tuned MiniLM)
5. Produces a comprehensive evaluation report

Outputs:
    data/real/full_descriptions.csv     (all extracted descriptions)
    data/real/full_unique_labeled.csv   (deduplicated + gemini + model predictions)
    data/real/full_eval_report.txt      (evaluation report)
"""

import csv
import json
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from transaction_classifier.categories import ALL_LABELS
from transaction_classifier.config import settings
from transaction_classifier.data.preprocess import clean_transaction
from transaction_classifier.models.ensemble import Ensemble
from transaction_classifier.models.finetune_model import FineTuneModel
from transaction_classifier.rules.engine import RulesEngine

# Import extraction functions from the existing script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_descriptions_ocr import (
    extract_mastercard_descriptions,
    extract_chequing_descriptions,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "real"

MASTERCARD_DIR = Path(
    r"C:\Users\crims\Documents\0A04_Reference_Documents"
    r"\4B04_Bank\4C01_RBC\4D03_MasterCard"
)
CHEQUING_DIR = Path(
    r"C:\Users\crims\Documents\0A04_Reference_Documents"
    r"\4B04_Bank\4C01_RBC\4D02_Noirt"
)

LABEL_PROMPT = """You are labeling bank transactions into budget categories.

For each transaction description below, assign exactly ONE category from this list:
{categories}

Output a JSON array where each element is:
{{"description": "<original description>", "category": "<chosen category>"}}

Consider:
- The transaction description (from Canadian bank statements)
- Common Canadian merchants and their categories
- Prefixes like "Visa Debit purchase", "Misc Payment", "Payroll Deposit", "e-Transfer" indicate transaction type
- "Payroll Deposit" = Income
- "e-Transfer received/Autodeposit" = Income (money coming in)
- "e-Transfer sent" = Financial Services (money going out)
- "Misc Payment RBC CREDIT CARD" = Financial Services (credit card payment)
- "Online Banking payment - UNI OTT TUITION" = Government & Legal (tuition)
- "Student Loan" / "NSLSC" = Financial Services
- "GST CANADA" / "Tax Refund CANADA" / "Canada Carbon Rebate" = Government & Legal

Transactions to label:
{transactions}

Output ONLY valid JSON array. No markdown, no explanation."""


def label_batch(batch: list[str]) -> dict[str, str]:
    """Label a batch of descriptions using Gemini."""
    categories = "\n".join(f"- {c}" for c in ALL_LABELS)
    txn_lines = "\n".join(f'- "{d}"' for d in batch)
    prompt = LABEL_PROMPT.format(categories=categories, transactions=txn_lines)

    temp_file = OUTPUT_DIR / f"_eval_{uuid.uuid4().hex[:8]}.txt"
    temp_file.write_text(prompt, encoding="utf-8")

    try:
        result = subprocess.run(
            f'type "{temp_file}" | gemini -p ""',
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_DIR), shell=True,
        )
        output = result.stdout.strip()

        # Strip noise
        lines = output.split("\n")
        output = "\n".join(
            l for l in lines
            if "DeprecationWarning" not in l and "node --trace" not in l
            and "Loaded cached" not in l
        ).strip()

        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()

        try:
            labeled = json.loads(output)
        except json.JSONDecodeError:
            last_brace = output.rfind("}")
            if last_brace > 0:
                trimmed = output[: last_brace + 1].rstrip().rstrip(",") + "\n]"
                labeled = json.loads(trimmed)
            else:
                return {}

        if not isinstance(labeled, list):
            return {}

        return {
            item["description"]: item["category"]
            for item in labeled
            if "description" in item and "category" in item
            and item["category"] in ALL_LABELS
        }

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
        print(f"    ERROR: {e}")
        return {}
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except PermissionError:
            pass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract from ALL PDFs ──
    print("=" * 70)
    print("STEP 1: Extracting descriptions from all bank statement PDFs")
    print("=" * 70)

    all_rows = []
    for pdf_dir, account_type, extract_fn in [
        (MASTERCARD_DIR, "mastercard", extract_mastercard_descriptions),
        (CHEQUING_DIR, "chequing", extract_chequing_descriptions),
    ]:
        if not pdf_dir.exists():
            print(f"WARNING: {pdf_dir} not found, skipping")
            continue

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        print(f"\n  {account_type}: {len(pdf_files)} PDFs")
        for pdf_path in pdf_files:
            descs = extract_fn(pdf_path)
            print(f"    {pdf_path.name}: {len(descs)} descriptions")
            for d in descs:
                all_rows.append({
                    "description": d,
                    "account_type": account_type,
                    "source_file": pdf_path.name,
                })

    df_all = pd.DataFrame(all_rows)
    mc = (df_all["account_type"] == "mastercard").sum()
    chq = (df_all["account_type"] == "chequing").sum()
    print(f"\n  Total extracted: {len(df_all)} ({mc} MC, {chq} chequing)")

    # Save full extraction
    full_path = OUTPUT_DIR / "full_descriptions.csv"
    df_all.to_csv(full_path, index=False)

    # ── Step 2: Deduplicate ──
    print(f"\n{'=' * 70}")
    print("STEP 2: Deduplicating")
    print("=" * 70)

    # Unique descriptions with account_type (keep first occurrence)
    df_unique = df_all.drop_duplicates(subset=["description"]).copy()
    df_unique = df_unique.reset_index(drop=True)

    # Count occurrences of each description
    desc_counts = df_all["description"].value_counts().to_dict()
    df_unique["occurrence_count"] = df_unique["description"].map(desc_counts)

    print(f"  Total descriptions: {len(df_all)}")
    print(f"  Unique descriptions: {len(df_unique)}")
    print(f"  Dedup ratio: {len(df_all) / len(df_unique):.1f}x")
    print(f"  Top repeated:")
    for desc, count in sorted(desc_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {count:3d}x  {desc[:70]}")

    # ── Step 3: Preprocess and verify ──
    print(f"\n{'=' * 70}")
    print("STEP 3: Preprocessing verification")
    print("=" * 70)

    df_unique["cleaned"] = df_unique["description"].apply(clean_transaction)
    empty_cleaned = (df_unique["cleaned"] == "").sum()
    print(f"  Empty after cleaning: {empty_cleaned}/{len(df_unique)}")
    if empty_cleaned > 0:
        print("  Empty descriptions:")
        for _, row in df_unique[df_unique["cleaned"] == ""].iterrows():
            print(f"    [{row['account_type'][:2].upper()}] {row['description']}")

    # Show some preprocessing examples
    print(f"\n  Sample preprocessing:")
    samples = df_unique.sample(min(15, len(df_unique)), random_state=42)
    for _, row in samples.iterrows():
        print(f"    {row['description'][:55]:55s} => [{row['cleaned']}]")

    # ── Step 4: Label unique descriptions with Gemini ──
    print(f"\n{'=' * 70}")
    print("STEP 4: Labeling unique descriptions with Gemini")
    print("=" * 70)

    unique_descs = df_unique["description"].tolist()
    all_labels = {}
    batch_size = 30

    for i in range(0, len(unique_descs), batch_size):
        batch = unique_descs[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(unique_descs) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} "
              f"({i+1}-{min(i+batch_size, len(unique_descs))})...",
              end=" ", flush=True)

        labels = label_batch(batch)
        all_labels.update(labels)
        print(f"{len(labels)} labeled")

    df_unique["gemini_category"] = df_unique["description"].map(all_labels)
    unlabeled = df_unique["gemini_category"].isna().sum()
    labeled_count = len(df_unique) - unlabeled
    print(f"\n  Labeled: {labeled_count}/{len(df_unique)} ({unlabeled} unlabeled)")

    # ── Step 5: Run our model ──
    print(f"\n{'=' * 70}")
    print("STEP 5: Running ensemble classifier (direction + rules + fine-tuned MiniLM)")
    print("=" * 70)

    ft_path = settings.model_dir / "finetune"
    if not (ft_path / "model").exists():
        print("ERROR: No fine-tuned model. Run train_finetune.py first.")
        sys.exit(1)

    ft_model = FineTuneModel()
    ft_model.load(ft_path)
    rules_engine = RulesEngine()
    ensemble = Ensemble(rules_engine=rules_engine, finetune_model=ft_model)

    results = ensemble.classify_batch(df_unique["description"].tolist())
    df_unique["model_category"] = [r.category for r in results]
    df_unique["model_confidence"] = [r.confidence for r in results]
    df_unique["model_source"] = [r.source for r in results]
    df_unique["model_flagged"] = [r.flagged_for_review for r in results]

    # ── Step 6: Compare (on labeled subset only) ──
    print(f"\n{'=' * 70}")
    print("STEP 6: Evaluation (deduplicated, unique descriptions only)")
    print("=" * 70)

    labeled = df_unique[df_unique["gemini_category"].notna()].copy()
    labeled["match"] = labeled["model_category"] == labeled["gemini_category"]

    report = []
    report.append("=" * 70)
    report.append("FULL CORPUS EVALUATION — Deduplicated Unique Descriptions")
    report.append(f"Direction + Rules + Fine-tuned MiniLM (Model #5)")
    report.append("=" * 70)
    report.append(f"PDFs processed: {len(df_all['source_file'].unique())}")
    report.append(f"Total descriptions extracted: {len(df_all)}")
    report.append(f"Unique descriptions: {len(df_unique)}")
    report.append(f"Gemini-labeled: {len(labeled)}")
    report.append(f"Overall accuracy (unique): {labeled['match'].mean():.1%} "
                  f"({labeled['match'].sum()}/{len(labeled)})")

    # Weighted accuracy (accounting for duplicates)
    labeled_with_counts = labeled.copy()
    weighted_correct = (labeled_with_counts["match"] * labeled_with_counts["occurrence_count"]).sum()
    weighted_total = labeled_with_counts["occurrence_count"].sum()
    report.append(f"Weighted accuracy (with dupes): {weighted_correct / weighted_total:.1%} "
                  f"({int(weighted_correct)}/{int(weighted_total)})")

    report.append(f"\nPrevious results (smaller test set):")
    report.append(f"  Phase 4b (497 descs, with dupes):  86.5%")

    # By source
    report.append(f"\nBy classification source:")
    for source in ["direction", "rules", "finetune"]:
        subset = labeled[labeled["model_source"] == source]
        if len(subset):
            acc = subset["match"].mean()
            report.append(f"  {source}: {acc:.1%} ({subset['match'].sum()}/{len(subset)})")

    # By account type
    report.append(f"\nBy account type:")
    for acct in ["mastercard", "chequing"]:
        subset = labeled[labeled["account_type"] == acct]
        if len(subset):
            acc = subset["match"].mean()
            report.append(f"  {acct}: {acc:.1%} ({subset['match'].sum()}/{len(subset)})")

    # By category
    report.append(f"\nAccuracy by category:")
    report.append(f"{'Category':30s} | {'Accuracy':10s} | {'Correct/Total':15s}")
    report.append("-" * 60)
    for cat in sorted(labeled["gemini_category"].unique()):
        subset = labeled[labeled["gemini_category"] == cat]
        acc = subset["match"].mean()
        report.append(f"  {cat:30s} | {acc:10.1%} | {subset['match'].sum()}/{len(subset)}")

    # ML-only breakdown
    ml_only = labeled[labeled["model_source"] == "finetune"]
    if len(ml_only):
        report.append(f"\nFine-tune-only (unknown merchants):")
        report.append(f"  Accuracy: {ml_only['match'].mean():.1%} "
                      f"({ml_only['match'].sum()}/{len(ml_only)})")
        report.append(f"\n  By category:")
        for cat in sorted(ml_only["gemini_category"].unique()):
            subset = ml_only[ml_only["gemini_category"] == cat]
            acc = subset["match"].mean()
            report.append(f"    {cat:30s}: {acc:.1%} ({subset['match'].sum()}/{len(subset)})")

    # Mismatches
    mismatches = labeled[~labeled["match"]].copy()
    report.append(f"\nMISMATCHES ({len(mismatches)}):")
    report.append(
        f"{'Description':55s} | {'Model':25s} | "
        f"{'Gemini':25s} | Src      | Conf | #"
    )
    report.append("-" * 155)
    for _, row in mismatches.sort_values("occurrence_count", ascending=False).iterrows():
        desc = str(row["description"])[:55]
        report.append(
            f"{desc:55s} | {row['model_category']:25s} | "
            f"{row['gemini_category']:25s} | {row['model_source']:8s} | "
            f"{row['model_confidence']:.2f} | {row['occurrence_count']}"
        )

    report_text = "\n".join(report)
    print(f"\n{report_text}")

    # Save
    report_path = OUTPUT_DIR / "full_eval_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    df_unique.to_csv(OUTPUT_DIR / "full_unique_labeled.csv", index=False)
    print(f"\nReport: {report_path}")
    print(f"Data: {OUTPUT_DIR / 'full_unique_labeled.csv'}")
    print(f"Full extraction: {full_path}")


if __name__ == "__main__":
    main()
