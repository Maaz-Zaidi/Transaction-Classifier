#!/usr/bin/env python
"""Evaluate the full ensemble (rules + SGD) on the test set."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from transaction_classifier.config import settings
from transaction_classifier.models.ensemble import Ensemble
from transaction_classifier.models.sgd_model import SGDModel
from transaction_classifier.rules.engine import RulesEngine


def main():
    test_path = settings.data_dir / "processed" / "test.parquet"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. Run download_data.py first.")
        sys.exit(1)

    print("Loading test data...")
    test_df = pd.read_parquet(test_path)
    print(f"  Test: {len(test_df)} samples")

    # load models
    rules_engine = RulesEngine()
    sgd_model = SGDModel()
    sgd_path = settings.model_dir / "sgd"
    if not (sgd_path / "sgd_pipeline.joblib").exists():
        print(f"ERROR: No SGD model at {sgd_path}. Run train_sgd.py first.")
        sys.exit(1)
    sgd_model.load(sgd_path)

    ensemble = Ensemble(rules_engine=rules_engine, sgd_model=sgd_model)

    # classify
    print("\nClassifying test set...")
    start = time.perf_counter()
    results = ensemble.classify_batch(test_df["transaction_description"].tolist())
    elapsed = time.perf_counter() - start
    print(f"  Classified {len(results)} transactions in {elapsed:.1f}s")
    print(f"  Throughput: {len(results) / elapsed:.0f} transactions/sec")

    # analyze sources
    sources = [r.source for r in results]
    for source in ["rules", "sgd", "bert"]:
        count = sources.count(source)
        if count:
            pct = count / len(sources) * 100
            print(f"  {source}: {count} ({pct:.1f}%)")

    flagged = sum(1 for r in results if r.flagged_for_review)
    print(f"  Flagged for review: {flagged} ({flagged / len(results) * 100:.1f}%)")

    # accuracy
    pred_labels = [r.category for r in results]
    true_labels = test_df["category"].tolist()

    print("\n" + "=" * 60)
    print("Classification Report (Full Ensemble)")
    print("=" * 60)
    print(classification_report(true_labels, pred_labels))

    # per-source accuracy
    for source in ["rules", "sgd"]:
        indices = [i for i, s in enumerate(sources) if s == source]
        if indices:
            correct = sum(
                1 for i in indices if pred_labels[i] == true_labels[i]
            )
            acc = correct / len(indices) * 100
            print(f"{source} accuracy: {acc:.1f}% ({correct}/{len(indices)})")

    # confusion matrix (abbreviated)
    print("\nConfusion matrix saved to stdout. Categories:")
    labels = sorted(set(true_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    print(f"  Shape: {cm.shape}")
    print(f"  Diagonal sum (correct): {cm.diagonal().sum()}")
    print(f"  Total: {cm.sum()}")
    print(f"  Overall accuracy: {cm.diagonal().sum() / cm.sum() * 100:.1f}%")


if __name__ == "__main__":
    main()
