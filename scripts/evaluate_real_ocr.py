#!/usr/bin/env python
"""Evaluate model against OCR-extracted real bank descriptions.

Labels descriptions with Gemini (ground truth), then runs the ensemble
classifier and compares.

Outputs:
    data/real/ocr_labeled.csv          (descriptions + gemini labels + model predictions)
    data/real/ocr_eval_report.txt      (comparison report)
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
from transaction_classifier.models.ensemble import Ensemble
from transaction_classifier.models.fasttext_model import FastTextModel
from transaction_classifier.rules.engine import RulesEngine

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "real"

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
- "Misc Payment CIBC CPD" = Financial Services (loan payment)
- "Online Banking payment - UNI OTT TUITION" = Government & Legal (tuition)
- "Student Loan CANADA" = Financial Services
- "GST CANADA" / "Tax Refund CANADA" / "Canada Carbon Rebate" = Government & Legal

Transactions to label:
{transactions}

Output ONLY valid JSON array. No markdown, no explanation."""


def label_batch(batch: list[str]) -> dict[str, str]:
    """Label a batch of descriptions using Gemini. Returns {description: category}."""
    categories = "\n".join(f"- {c}" for c in ALL_LABELS)
    txn_lines = "\n".join(f'- "{d}"' for d in batch)

    prompt = LABEL_PROMPT.format(categories=categories, transactions=txn_lines)

    temp_file = OUTPUT_DIR / f"_eval_{uuid.uuid4().hex[:8]}.txt"
    temp_file.write_text(prompt, encoding="utf-8")

    try:
        result = subprocess.run(
            f'type "{temp_file}" | gemini -p ""',
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_DIR),
            shell=True,
        )

        output = result.stdout.strip()

        # strip noise
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

        return {item["description"]: item["category"] for item in labeled
                if "description" in item and "category" in item}

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
        print(f"    ERROR: {e}")
        return {}
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except PermissionError:
            pass


def main():
    # load ocr descriptions
    csv_path = OUTPUT_DIR / "descriptions_ocr.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run extract_descriptions_ocr.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} descriptions ({df['account_type'].value_counts().to_dict()})")

    # deduplicate so only unique descriptions are labeled
    unique_descs = df["description"].unique().tolist()
    print(f"Unique descriptions: {len(unique_descs)}")

    # step 1: label with gemini
    print("\nStep 1: Labeling unique descriptions with Gemini...")
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

    df["gemini_category"] = df["description"].map(all_labels)
    unlabeled = df["gemini_category"].isna().sum()
    print(f"\nLabeled: {len(df) - unlabeled}/{len(df)}"
          f" ({unlabeled} could not be labeled)")

    # step 2: run the model (direction + rules + fasttext)
    print("\nStep 2: Running ensemble classifier (direction + rules + FastText)...")
    rules_engine = RulesEngine()
    ft_model = FastTextModel()
    ft_path = settings.model_dir / "fasttext"
    if not (ft_path / "fasttext_model.bin").exists():
        print("ERROR: No FastText model. Run train_fasttext.py first.")
        sys.exit(1)
    ft_model.load(ft_path)
    ensemble = Ensemble(rules_engine=rules_engine, fasttext_model=ft_model)

    results = ensemble.classify_batch(df["description"].tolist())
    df["model_category"] = [r.category for r in results]
    df["model_confidence"] = [r.confidence for r in results]
    df["model_source"] = [r.source for r in results]
    df["model_flagged"] = [r.flagged_for_review for r in results]
    df["cleaned"] = [r.cleaned for r in results]

    # step 3: compare
    labeled = df[df["gemini_category"].notna()].copy()
    labeled["match"] = labeled["model_category"] == labeled["gemini_category"]

    report = []
    report.append("=" * 70)
    report.append("REAL DATA MODEL EVALUATION - Phase 2 (Direction + Rules + FastText)")
    report.append("=" * 70)
    report.append(f"Total descriptions: {len(df)}")
    report.append(f"Gemini-labeled: {len(labeled)}")
    report.append(f"Overall accuracy: {labeled['match'].mean():.1%} "
                  f"({labeled['match'].sum()}/{len(labeled)})")

    # previous results for comparison
    report.append(f"\nComparison across evaluations:")
    report.append(f"Phase 1 (pdfplumber + SGD):      43.4% (139/320)")
    report.append(f"Phase 1b (pymupdf + SGD):         53.7% (267/497)")
    report.append(f"Phase 2  (direction+rules+FT):    {labeled['match'].mean():.1%} "
                  f"({labeled['match'].sum()}/{len(labeled)})")

    # by source
    report.append(f"\nBy classification source:")
    for source in ["direction", "rules", "fasttext"]:
        subset = labeled[labeled["model_source"] == source]
        if len(subset):
            acc = subset["match"].mean()
            report.append(f"  {source}: {acc:.1%} ({subset['match'].sum()}/{len(subset)})")

    # by account type
    report.append(f"\nBy account type:")
    for acct in ["mastercard", "chequing"]:
        subset = labeled[labeled["account_type"] == acct]
        if len(subset):
            acc = subset["match"].mean()
            report.append(f"  {acct}: {acc:.1%} ({subset['match'].sum()}/{len(subset)})")
            prev = {"mastercard": "67.3%", "chequing": "32.5%"}
            report.append(f"    (Phase 1b: {prev.get(acct, 'N/A')})")

    # flagged
    report.append(f"\nFlagged for review: {labeled['model_flagged'].sum()} "
                  f"({labeled['model_flagged'].mean():.1%})")
    report.append(f"  (previous: 51.9%)")

    # category breakdown
    report.append(f"\nAccuracy by Gemini category:")
    for cat in sorted(labeled["gemini_category"].unique()):
        subset = labeled[labeled["gemini_category"] == cat]
        acc = subset["match"].mean()
        report.append(f"  {cat:30s}: {acc:.1%} ({subset['match'].sum()}/{len(subset)})")

    # mismatches
    mismatches = labeled[~labeled["match"]].copy()
    if len(mismatches):
        report.append(f"\nMISMATCHES ({len(mismatches)}):")
        report.append(
            f"{'Description':55s} | {'Cleaned':30s} | {'Model':25s} | "
            f"{'Gemini':25s} | Src  | Conf"
        )
        report.append("-" * 180)
        for _, row in mismatches.iterrows():
            desc = str(row["description"])[:55]
            cleaned = str(row["cleaned"])[:30]
            report.append(
                f"{desc:55s} | {cleaned:30s} | {row['model_category']:25s} | "
                f"{row['gemini_category']:25s} | {row['model_source']:4s} | "
                f"{row['model_confidence']:.2f}"
            )

    report_text = "\n".join(report)
    print(f"\n{report_text}")

    # save
    report_path = OUTPUT_DIR / "ocr_eval_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    df.to_csv(OUTPUT_DIR / "ocr_labeled.csv", index=False)
    print(f"\nReport: {report_path}")
    print(f"Data: {OUTPUT_DIR / 'ocr_labeled.csv'}")


if __name__ == "__main__":
    main()
