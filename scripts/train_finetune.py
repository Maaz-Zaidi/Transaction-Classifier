#!/usr/bin/env python
"""Train the fine-tuned MiniLM classifier.

Standard fine-tuning with cross-entropy loss. Outperforms SetFit's contrastive
approach when you have 800+ samples per class.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sklearn.metrics import classification_report

from transaction_classifier.config import settings
from transaction_classifier.models.finetune_model import FineTuneModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8000,
                        help="Max training samples (stratified)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--val-samples", type=int, default=5000,
                        help="Validation samples for eval during training")
    args = parser.parse_args()

    processed_dir = settings.data_dir / "processed"
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run download_data.py first.")
        sys.exit(1)

    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    train_df = train_df[train_df["cleaned"].str.len() > 0].copy()
    val_df = val_df[val_df["cleaned"].str.len() > 0].copy()

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")

    # Sample validation set for eval during training
    val_sample = val_df.sample(n=min(args.val_samples, len(val_df)), random_state=42)

    print(f"\nTraining fine-tuned MiniLM (max_samples={args.max_samples}, "
          f"epochs={args.epochs}, lr={args.lr})...")

    model = FineTuneModel()
    start = time.perf_counter()
    info = model.train(
        texts=train_df["cleaned"].tolist(),
        labels=train_df["category"].tolist(),
        val_texts=val_sample["cleaned"].tolist(),
        val_labels=val_sample["category"].tolist(),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
    )
    elapsed = time.perf_counter() - start
    print(f"  Trained in {elapsed:.1f}s")
    print(f"  Samples used: {info['train_samples']}")

    # Save before eval
    save_path = settings.model_dir / "finetune"
    print(f"\nSaving model to {save_path}...")
    model.save(save_path)

    # Evaluate on validation subset
    print(f"\nEvaluating on {len(val_sample)} validation samples...")
    eval_start = time.perf_counter()
    predictions = model.predict(val_sample["cleaned"].tolist())
    eval_elapsed = time.perf_counter() - eval_start

    pred_labels = [p[0] for p in predictions]
    pred_confidences = [p[1] for p in predictions]

    print(f"  Evaluation took {eval_elapsed:.1f}s")
    print(classification_report(val_sample["category"].tolist(), pred_labels))

    avg_conf = sum(pred_confidences) / len(pred_confidences)
    print(f"Average confidence: {avg_conf:.4f}")

    low_conf = sum(1 for c in pred_confidences if c < 0.70)
    print(f"Below 0.70 threshold: {low_conf} ({low_conf / len(pred_confidences) * 100:.1f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
