#!/usr/bin/env python
"""Train the SetFit (MiniLM) classifier.

SetFit uses contrastive learning on sentence-transformer embeddings.
The pre-trained all-MiniLM-L6-v2 already understands real-world concepts
(food, retail, transport) from its pre-training corpus, unlike FastText/SGD
which only know what synthetic training data taught them.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sklearn.metrics import classification_report

from transaction_classifier.config import settings
from transaction_classifier.models.setfit_model import SetFitTransactionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-samples", type=int, default=50_000,
        help="Max training samples (stratified subsample). Default 50K. "
             "SetFit generates contrastive pairs so full 3.6M is intractable."
    )
    parser.add_argument("--epochs", type=int, default=1, help="Contrastive epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--num-iterations", type=int, default=20,
                        help="Number of text pairs per class per epoch")
    args = parser.parse_args()

    processed_dir = settings.data_dir / "processed"
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run download_data.py first.")
        sys.exit(1)

    # Load data
    print("Loading training data...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Filter empty
    train_df = train_df[train_df["cleaned"].str.len() > 0].copy()
    val_df = val_df[val_df["cleaned"].str.len() > 0].copy()

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")

    # Train
    print(f"\nTraining SetFit (MiniLM) with max_samples={args.max_samples}...")
    model = SetFitTransactionModel()
    start = time.perf_counter()
    info = model.train(
        texts=train_df["cleaned"].tolist(),
        labels=train_df["category"].tolist(),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        max_samples=args.max_samples,
    )
    elapsed = time.perf_counter() - start
    print(f"  Trained in {elapsed:.1f}s")
    print(f"  Samples used: {info['train_samples']}")
    print(f"  Base model: {info['base_model']}")

    # Save first
    save_path = settings.model_dir / "setfit"
    print(f"\nSaving model to {save_path}...")
    model.save(save_path)

    # Evaluate on validation subset (full val set is too slow for sentence-transformers)
    val_sample = val_df.sample(n=min(5000, len(val_df)), random_state=42)
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
