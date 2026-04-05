#!/usr/bin/env python
"""Train fine-tuned MiniLM with abbreviation augmentation (Phase 6a).

Loads training data, applies abbreviation augmentation to teach the model
that truncated/compressed merchant names map to the same category, then
fine-tunes MiniLM with a classification head.

Phase 4 found 20K > 8K hurt without augmentation (overfitting to synthetic
patterns). With augmentation this inverts — more base merchants = more diverse
abbreviation variants. CLI arg allows A/B testing.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sklearn.metrics import classification_report

from transaction_classifier.config import settings
from transaction_classifier.data.augment import augment_dataset
from transaction_classifier.models.finetune_model import FineTuneModel
from transaction_classifier.models.registry import register_model


def stratified_sample(texts, labels, max_samples, seed=42):
    """Stratified subsample before augmentation."""
    import random
    from transaction_classifier.categories import ALL_LABELS

    random.seed(seed)
    label2idx: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        label2idx.setdefault(label, []).append(i)

    per_class = max(1, max_samples // len(ALL_LABELS))
    indices = []
    for cls_indices in label2idx.values():
        if len(cls_indices) <= per_class:
            indices.extend(cls_indices)
        else:
            indices.extend(random.sample(cls_indices, per_class))
    random.shuffle(indices)

    return [texts[i] for i in indices], [labels[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description="Train augmented MiniLM (Phase 6a)")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Base samples before augmentation (stratified)")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--augments-per-sample", type=int, default=3,
                        help="Abbreviation variants per sample")
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

    # stratified sample before augmentation
    texts = train_df["cleaned"].tolist()
    labels = train_df["category"].tolist()

    if args.max_samples and args.max_samples < len(texts):
        print(f"\nStratified sampling to {args.max_samples} base samples...")
        texts, labels = stratified_sample(texts, labels, args.max_samples)
        print(f"  After sampling: {len(texts)} samples")

    # apply abbreviation augmentation
    print(f"\nAugmenting with {args.augments_per_sample} variants per sample...")
    aug_texts, aug_labels = augment_dataset(
        texts, labels, augments_per_sample=args.augments_per_sample,
    )
    print(f"  After augmentation: {len(aug_texts)} samples "
          f"({len(aug_texts) - len(texts)} augmented)")

    # sample a validation set for training eval
    val_sample = val_df.sample(n=min(args.val_samples, len(val_df)), random_state=42)

    print(f"\nTraining augmented MiniLM (epochs={args.epochs}, "
          f"batch_size={args.batch_size}, lr={args.lr})...")

    model = FineTuneModel()
    start = time.perf_counter()
    info = model.train(
        texts=aug_texts,
        labels=aug_labels,
        val_texts=val_sample["cleaned"].tolist(),
        val_labels=val_sample["category"].tolist(),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=None,  # already sampled before augmentation
    )
    elapsed = time.perf_counter() - start
    print(f"  Trained in {elapsed:.1f}s")
    print(f"  Samples used: {info['train_samples']}")

    # save to a separate directory
    save_path = settings.model_dir / "finetune_augmented"
    print(f"\nSaving model to {save_path}...")
    model.save(save_path)

    # evaluate on validation subset
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

    # register in model registry
    from sklearn.metrics import accuracy_score
    val_accuracy = accuracy_score(val_sample["category"].tolist(), pred_labels)

    register_model(
        model_type="finetune_augmented",
        version_dir=save_path,
        metrics={
            "val_accuracy": round(val_accuracy, 4),
            "val_avg_confidence": round(avg_conf, 4),
            "val_below_threshold_pct": round(low_conf / len(pred_confidences) * 100, 1),
        },
        hyperparams={
            "base_model": FineTuneModel.MODEL_ID,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "warmup_ratio": 0.1,
            "max_length": 64,
            "base_samples": len(texts),
            "augments_per_sample": args.augments_per_sample,
            "total_train_samples": len(aug_texts),
            "loss": "cross-entropy",
        },
        train_samples=len(aug_texts),
        notes=f"Phase 6a: Augmented MiniLM. {len(texts)} base -> {len(aug_texts)} with abbreviation augmentation ({args.augments_per_sample} variants/sample). {args.epochs} epochs.",
    )
    print("\nRegistered in model registry.")
    print("Done.")


if __name__ == "__main__":
    main()
