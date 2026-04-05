#!/usr/bin/env python
"""Train the FastText supervised classifier.

Usage:
    python scripts/train_fasttext.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sklearn.metrics import classification_report

from transaction_classifier.config import settings
from transaction_classifier.models.fasttext_model import FastTextModel


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

    # filter empty cleaned texts
    train_df = train_df[train_df["cleaned"].str.len() > 0].copy()
    val_df = val_df[val_df["cleaned"].str.len() > 0].copy()

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")

    # train
    print("\nTraining FastText model...")
    model = FastTextModel()
    start = time.perf_counter()
    info = model.train(
        texts=train_df["cleaned"].tolist(),
        labels=train_df["category"].tolist(),
        epoch=10,
        lr=0.5,
        word_ngrams=2,
        dim=100,
        minn=3,
        maxn=6,
        loss="softmax",
    )
    elapsed = time.perf_counter() - start
    print(f"  Trained in {elapsed:.1f}s")
    print(f"  Labels: {info['labels']}")
    print(f"  Embedding dim: {info['dim']}")

    # save first (before evaluation, in case eval crashes)
    save_path = settings.model_dir / "fasttext"
    print(f"\nSaving model to {save_path}...")
    model.save(save_path)
    print(f"Metadata: {model.metadata}")

    # evaluate on validation set
    print("\nEvaluating on validation set...")
    predictions = model.predict(val_df["cleaned"].tolist())
    pred_labels = [p[0] for p in predictions]
    pred_confidences = [p[1] for p in predictions]

    print(classification_report(val_df["category"].tolist(), pred_labels))

    avg_conf = sum(pred_confidences) / len(pred_confidences)
    print(f"Average confidence: {avg_conf:.4f}")

    low_conf = sum(1 for c in pred_confidences if c < 0.70)
    print(f"Below 0.70 threshold: {low_conf} ({low_conf / len(pred_confidences) * 100:.1f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
