#!/usr/bin/env python
"""Label real transactions using Gemini, then test the model against them.

Outputs:
    data/real/labeled_transactions.csv  (with gemini_category column)
    data/real/model_vs_real_report.txt  (comparison report)
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
from transaction_classifier.models.sgd_model import SGDModel
from transaction_classifier.rules.engine import RulesEngine

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "real"

LABEL_PROMPT_TEMPLATE = """You are labeling bank transactions into budget categories.

For each transaction below, assign exactly ONE category from this list:
{categories}

Output a JSON array where each element is:
{{"description": "<original description>", "category": "<chosen category>"}}

Consider:
- The transaction description (may have spaces stripped due to PDF extraction)
- The amount and type (debit=purchase/payment, credit=deposit/refund)
- Common Canadian merchants and their categories

Transactions to label:
{transactions}

Output ONLY valid JSON array. No markdown, no explanation."""


def label_batch_with_gemini(batch: list[dict]) -> list[dict]:
    """Label a batch of transactions using Gemini."""
    categories = "\n".join(f"- {c}" for c in ALL_LABELS)
    txn_lines = "\n".join(
        f"- \"{t['description']}\" (${t['amount']}, {t['type']})"
        for t in batch
    )

    prompt = LABEL_PROMPT_TEMPLATE.format(
        categories=categories,
        transactions=txn_lines,
    )

    temp_file = OUTPUT_DIR / f"_label_{uuid.uuid4().hex[:8]}.txt"
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
            # try repair
            last_brace = output.rfind("}")
            if last_brace > 0:
                trimmed = output[: last_brace + 1].rstrip().rstrip(",") + "\n]"
                labeled = json.loads(trimmed)
            else:
                return []

        return labeled if isinstance(labeled, list) else []

    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"    ERROR: {e}")
        return []
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except PermissionError:
            pass


def label_all_transactions(df: pd.DataFrame, batch_size: int = 25) -> pd.DataFrame:
    """Label all transactions in batches."""
    all_labels = {}

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        batch_dicts = batch[["description", "amount", "type"]].to_dict("records")

        print(f"  Labeling batch {i // batch_size + 1} "
              f"({i+1}-{min(i+batch_size, len(df))} of {len(df)})...",
              end=" ", flush=True)

        labeled = label_batch_with_gemini(batch_dicts)

        # match labels back to descriptions
        label_map = {item["description"]: item["category"] for item in labeled}
        for _, row in batch.iterrows():
            if row["description"] in label_map:
                all_labels[row.name] = label_map[row["description"]]

        print(f"{len(labeled)} labeled")

    df = df.copy()
    df["gemini_category"] = df.index.map(all_labels)
    return df


def test_model(df: pd.DataFrame) -> None:
    """Run the model on real transactions and compare to Gemini labels."""
    # load models
    rules_engine = RulesEngine()
    sgd_model = SGDModel()
    sgd_path = settings.model_dir / "sgd"
    if not (sgd_path / "sgd_pipeline.joblib").exists():
        print("ERROR: No SGD model. Run train_sgd.py first.")
        sys.exit(1)
    sgd_model.load(sgd_path)
    ensemble = Ensemble(rules_engine=rules_engine, sgd_model=sgd_model)

    # classify
    results = ensemble.classify_batch(df["description"].tolist())

    df = df.copy()
    df["model_category"] = [r.category for r in results]
    df["model_confidence"] = [r.confidence for r in results]
    df["model_source"] = [r.source for r in results]
    df["model_flagged"] = [r.flagged_for_review for r in results]
    df["cleaned"] = [r.cleaned for r in results]

    # compare only rows with gemini labels
    labeled = df[df["gemini_category"].notna()].copy()
    labeled["match"] = labeled["model_category"] == labeled["gemini_category"]

    # report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("REAL DATA MODEL EVALUATION")
    report_lines.append("=" * 70)
    report_lines.append(f"Total transactions: {len(df)}")
    report_lines.append(f"Gemini-labeled: {len(labeled)}")
    report_lines.append(f"Model accuracy: {labeled['match'].mean():.1%} "
                       f"({labeled['match'].sum()}/{len(labeled)})")

    # by source
    for source in ["rules", "sgd"]:
        subset = labeled[labeled["model_source"] == source]
        if len(subset):
            acc = subset["match"].mean()
            report_lines.append(f"  {source}: {acc:.1%} ({subset['match'].sum()}/{len(subset)})")

    # by account type
    for acct in ["mastercard", "chequing"]:
        subset = labeled[labeled["account_type"] == acct]
        if len(subset):
            acc = subset["match"].mean()
            report_lines.append(f"  {acct}: {acc:.1%} ({subset['match'].sum()}/{len(subset)})")

    report_lines.append(f"\nFlagged for review: {labeled['model_flagged'].sum()} "
                       f"({labeled['model_flagged'].mean():.1%})")

    # mismatches
    mismatches = labeled[~labeled["match"]]
    if len(mismatches):
        report_lines.append(f"\nMISMATCHES ({len(mismatches)}):")
        report_lines.append(f"{'Description':50s} | {'Cleaned':25s} | {'Model':25s} | {'Gemini':25s} | Src | Conf")
        report_lines.append("-" * 165)
        for _, row in mismatches.iterrows():
            desc = str(row["description"])[:50]
            cleaned = str(row["cleaned"])[:25]
            report_lines.append(
                f"{desc:50s} | {cleaned:25s} | {row['model_category']:25s} | "
                f"{row['gemini_category']:25s} | {row['model_source']:4s} | {row['model_confidence']:.2f}"
            )

    report = "\n".join(report_lines)
    print(report)

    # save report
    report_path = OUTPUT_DIR / "model_vs_real_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")

    # save full labeled data
    df.to_csv(OUTPUT_DIR / "labeled_transactions.csv", index=False)
    print(f"Full data saved to {OUTPUT_DIR / 'labeled_transactions.csv'}")


def main():
    csv_path = OUTPUT_DIR / "all_transactions.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run extract_bank_pdfs.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transactions")

    # step 1: label with gemini
    print("\nStep 1: Labeling with Gemini...")
    df = label_all_transactions(df)
    unlabeled = df["gemini_category"].isna().sum()
    if unlabeled:
        print(f"WARNING: {unlabeled} transactions could not be labeled")

    # step 2: test the model
    print("\nStep 2: Testing model against Gemini labels...")
    test_model(df)


if __name__ == "__main__":
    main()
