#!/usr/bin/env python
"""Train the TF-IDF + SGDClassifier model."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sklearn.metrics import classification_report

from transaction_classifier.config import settings
from transaction_classifier.models.sgd_model import SGDModel


def main():
    processed_dir = settings.data_dir / "processed"
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run download_data.py first.")
        sys.exit(1)

    # load data
    print("Loading training data...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")

    # train
    print("\nTraining SGD model...")
    model = SGDModel()
    start = time.perf_counter()
    info = model.train(
        texts=train_df["cleaned"].tolist(),
        labels=train_df["category"].tolist(),
    )
    elapsed = time.perf_counter() - start
    print(f"  Trained in {elapsed:.1f}s")
    print(f"  Features: {info['n_features']}")
    print(f"  Classes: {len(info['classes'])}")

    # evaluate on validation set
    print("\nEvaluating on validation set...")
    val_preds = model.predict(val_df["cleaned"].tolist())
    pred_labels = [p.category for p in val_preds]
    pred_confidences = [p.confidence for p in val_preds]

    print(classification_report(val_df["category"].tolist(), pred_labels))

    avg_conf = sum(pred_confidences) / len(pred_confidences)
    print(f"Average confidence: {avg_conf:.4f}")

    low_conf = sum(1 for c in pred_confidences if c < settings.sgd_confidence_threshold)
    print(
        f"Below threshold ({settings.sgd_confidence_threshold}): "
        f"{low_conf} ({low_conf / len(pred_confidences) * 100:.1f}%)"
    )

    # save
    save_path = settings.model_dir / "sgd"
    print(f"\nSaving model to {save_path}...")
    model.save(save_path)
    print("Done.")


if __name__ == "__main__":
    main()
