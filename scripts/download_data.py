#!/usr/bin/env python
"""Download the dataset and create train/val/test splits."""

import sys
from pathlib import Path

# Add src to path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transaction_classifier.data.download import download_dataset
from transaction_classifier.data.split import process_and_split


def main():
    print("=" * 60)
    print("Step 1: Download dataset")
    print("=" * 60)
    raw_path = download_dataset()

    print()
    print("=" * 60)
    print("Step 2: Preprocess and split")
    print("=" * 60)
    paths = process_and_split(raw_path=raw_path)

    print()
    print("=" * 60)
    print("Done! Files created:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
