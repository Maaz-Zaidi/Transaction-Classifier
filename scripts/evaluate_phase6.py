#!/usr/bin/env python
"""Phase 6 evaluation on real Canadian bank transactions.

Uses codex_labeled.csv as TEST-ONLY ground truth. Never trains on this data.
Evaluates: baseline MiniLM, augmented MiniLM, CANINE, or all three.
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from transaction_classifier.config import settings
from transaction_classifier.models.ensemble import Ensemble
from transaction_classifier.models.finetune_model import FineTuneModel
from transaction_classifier.rules.engine import RulesEngine


def load_ensemble(model_name: str) -> Ensemble:
    """Load ensemble with the specified ML model."""
    rules_engine = RulesEngine()

    if model_name == "baseline":
        model_path = settings.model_dir / "finetune"
        if not (model_path / "model").exists():
            print(f"ERROR: Baseline model not found at {model_path}")
            sys.exit(1)
        ft_model = FineTuneModel()
        ft_model.load(model_path)
        return Ensemble(rules_engine=rules_engine, finetune_model=ft_model)

    elif model_name == "augmented":
        model_path = settings.model_dir / "finetune_augmented"
        if not (model_path / "model").exists():
            print(f"ERROR: Augmented model not found at {model_path}")
            print("Run: python scripts/train_finetune_augmented.py")
            sys.exit(1)
        ft_model = FineTuneModel()
        ft_model.load(model_path)
        return Ensemble(rules_engine=rules_engine, finetune_model=ft_model)

    elif model_name == "canine":
        from transaction_classifier.models.canine_model import CanineModel
        model_path = settings.model_dir / "canine"
        if not (model_path / "model").exists():
            print(f"ERROR: CANINE model not found at {model_path}")
            print("Run: python scripts/train_canine.py")
            sys.exit(1)
        canine = CanineModel()
        canine.load(model_path)
        return Ensemble(rules_engine=rules_engine, canine_model=canine)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model_name: str, ensemble: Ensemble, df: pd.DataFrame) -> dict:
    """Run evaluation and return metrics dict."""
    raw_texts = df["raw_example"].tolist()
    true_labels = df["codex_category"].tolist()

    start = time.perf_counter()
    results = ensemble.classify_batch(raw_texts)
    elapsed = time.perf_counter() - start

    # overall accuracy
    pred_labels = [r.category for r in results]
    sources = [r.source for r in results]
    confidences = [r.confidence for r in results]

    correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
    total = len(true_labels)
    overall_acc = correct / total

    # per-source breakdown
    source_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, true, source in zip(pred_labels, true_labels, sources):
        source_counts[source]["total"] += 1
        if pred == true:
            source_counts[source]["correct"] += 1

    # ml-only accuracy; excludes direction and rules
    ml_sources = {"finetune", "setfit", "fasttext", "sgd", "canine"}
    ml_correct = sum(1 for p, t, s in zip(pred_labels, true_labels, sources)
                     if s in ml_sources and p == t)
    ml_total = sum(1 for s in sources if s in ml_sources)
    ml_acc = ml_correct / ml_total if ml_total > 0 else 0.0

    # per-category breakdown
    cat_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, true in zip(pred_labels, true_labels):
        cat_counts[true]["total"] += 1
        if pred == true:
            cat_counts[true]["correct"] += 1

    return {
        "model_name": model_name,
        "overall_accuracy": overall_acc,
        "ml_only_accuracy": ml_acc,
        "ml_correct": ml_correct,
        "ml_total": ml_total,
        "total": total,
        "correct": correct,
        "inference_time": elapsed,
        "avg_confidence": sum(confidences) / len(confidences),
        "source_breakdown": {
            s: {"accuracy": d["correct"] / d["total"] if d["total"] > 0 else 0.0,
                "count": d["total"]}
            for s, d in sorted(source_counts.items())
        },
        "category_breakdown": {
            c: {"accuracy": d["correct"] / d["total"] if d["total"] > 0 else 0.0,
                "count": d["total"]}
            for c, d in sorted(cat_counts.items())
        },
    }


def format_report(metrics: dict) -> str:
    """Format a single model's metrics as a readable report."""
    lines = []
    m = metrics
    lines.append(f"=== {m['model_name'].upper()} ===")
    lines.append(f"Overall accuracy:  {m['overall_accuracy']:.1%} ({m['correct']}/{m['total']})")
    lines.append(f"ML-only accuracy:  {m['ml_only_accuracy']:.1%} ({m['ml_correct']}/{m['ml_total']})")
    lines.append(f"Avg confidence:    {m['avg_confidence']:.4f}")
    lines.append(f"Inference time:    {m['inference_time']:.1f}s")
    lines.append("")

    lines.append("By source:")
    for source, data in m["source_breakdown"].items():
        lines.append(f"  {source:12s}  {data['accuracy']:6.1%}  (n={data['count']})")
    lines.append("")

    lines.append("By category:")
    for cat, data in m["category_breakdown"].items():
        lines.append(f"  {cat:30s}  {data['accuracy']:6.1%}  (n={data['count']})")

    return "\n".join(lines)


def format_comparison(all_metrics: list[dict]) -> str:
    """Format a comparison table across models."""
    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 6 MODEL COMPARISON")
    lines.append("=" * 70)
    lines.append("")

    # summary table
    header = f"{'Model':<20s} {'Overall':>10s} {'ML-only':>10s} {'ML n':>6s} {'Time':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    for m in all_metrics:
        lines.append(
            f"{m['model_name']:<20s} "
            f"{m['overall_accuracy']:>9.1%} "
            f"{m['ml_only_accuracy']:>9.1%} "
            f"{m['ml_total']:>6d} "
            f"{m['inference_time']:>7.1f}s"
        )
    lines.append("")

    # per-category comparison
    all_cats = sorted({c for m in all_metrics for c in m["category_breakdown"]})
    cat_header = f"{'Category':<30s}" + "".join(f" {m['model_name']:>12s}" for m in all_metrics)
    lines.append(cat_header)
    lines.append("-" * len(cat_header))
    for cat in all_cats:
        row = f"{cat:<30s}"
        for m in all_metrics:
            if cat in m["category_breakdown"]:
                row += f" {m['category_breakdown'][cat]['accuracy']:>11.1%}"
            else:
                row += f" {'N/A':>12s}"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 6 evaluation on real data")
    parser.add_argument("--model", type=str, default="all",
                        choices=["baseline", "augmented", "canine", "all"],
                        help="Which model(s) to evaluate")
    args = parser.parse_args()

    # load test data
    codex_path = settings.data_dir / "real" / "codex_labeled.csv"
    if not codex_path.exists():
        print(f"ERROR: {codex_path} not found.")
        sys.exit(1)

    df = pd.read_csv(codex_path)
    # filter rows with valid labels
    df = df[df["codex_category"].notna() & (df["codex_category"] != "")].copy()
    print(f"Loaded {len(df)} labeled test samples from codex_labeled.csv")
    print(f"Categories: {sorted(df['codex_category'].unique())}")
    print()

    models_to_eval = []
    if args.model == "all":
        # evaluate all available models
        for name in ["baseline", "augmented", "canine"]:
            try:
                ensemble = load_ensemble(name)
                models_to_eval.append((name, ensemble))
            except SystemExit:
                print(f"  Skipping {name} (not available)\n")
    else:
        ensemble = load_ensemble(args.model)
        models_to_eval.append((args.model, ensemble))

    if not models_to_eval:
        print("ERROR: No models available to evaluate.")
        sys.exit(1)

    all_metrics = []
    report_parts = []

    for name, ensemble in models_to_eval:
        print(f"Evaluating {name}...")
        metrics = evaluate_model(name, ensemble, df)
        all_metrics.append(metrics)
        report = format_report(metrics)
        report_parts.append(report)
        print(report)
        print()

    # comparison table if multiple models
    if len(all_metrics) > 1:
        comparison = format_comparison(all_metrics)
        report_parts.append(comparison)
        print(comparison)

    # save report
    report_path = settings.data_dir / "real" / "phase6_eval_report.txt"
    full_report = "\n\n".join(report_parts)
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
