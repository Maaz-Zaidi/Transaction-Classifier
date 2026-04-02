"""Re-label all unique descriptions using Codex CLI in batches of 15."""

import csv
import json
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "real"
TEMP_DIR = Path("/tmp")

ALL_LABELS = [
    "Food & Dining",
    "Shopping & Retail",
    "Transportation",
    "Entertainment & Recreation",
    "Utilities & Services",
    "Financial Services",
    "Healthcare & Medical",
    "Government & Legal",
    "Income",
    "Charity & Donations",
]

LABEL_PROMPT = """You are labeling bank transactions into budget categories.

For each transaction description below, assign exactly ONE category from this list:
{categories}

Output a JSON array where each element is:
{{"description": "<original description>", "category": "<chosen category>"}}

Consider:
- These are Canadian bank transaction descriptions
- Common Canadian merchants and their categories
- Prefixes like "Visa Debit purchase", "Misc Payment", "Payroll Deposit", "e-Transfer" indicate transaction type
- "Payroll Deposit" = Income
- "e-Transfer received/Autodeposit" = Income (money coming in)
- "e-Transfer sent" = Financial Services (money going out)
- "Misc Payment RBC CREDIT CARD" = Financial Services (credit card payment)
- "Online Banking payment - UNI OTT TUITION" = Government & Legal (tuition)
- "Student Loan" / "NSLSC" = Financial Services
- "GST CANADA" / "Tax Refund CANADA" / "Canada Carbon Rebate" = Government & Legal
- RCSS = Real Canadian Superstore (grocery store) = Food & Dining
- GOODLIFE / Fit4less = gym/fitness = Entertainment & Recreation
- Shoppers Drug Mart = pharmacy = Healthcare & Medical

Transactions to label:
{transactions}

Output ONLY valid JSON array. No markdown, no explanation, no tool calls. Just the JSON."""

BATCH_SIZE = 15


def label_batch_codex(batch: list[str], batch_num: int) -> dict[str, str]:
    """Label a batch using codex exec."""
    categories = "\n".join(f"- {c}" for c in ALL_LABELS)
    txn_lines = "\n".join(f'- "{d}"' for d in batch)
    prompt = LABEL_PROMPT.format(categories=categories, transactions=txn_lines)

    prompt_file = TEMP_DIR / f"codex_batch_{batch_num}.txt"
    output_file = TEMP_DIR / f"codex_out_{batch_num}.txt"
    prompt_file.write_text(prompt, encoding="utf-8")

    try:
        result = subprocess.run(
            f'codex exec --sandbox read-only --ephemeral -o "{output_file}" -',
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
            shell=True,
        )

        if output_file.exists():
            output = output_file.read_text(encoding="utf-8").strip()
        else:
            output = result.stdout.strip()

        # Strip markdown fences if present
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()

        try:
            labeled = json.loads(output)
        except json.JSONDecodeError:
            # Try to salvage partial JSON
            last_brace = output.rfind("}")
            if last_brace > 0:
                trimmed = output[: last_brace + 1].rstrip().rstrip(",") + "\n]"
                labeled = json.loads(trimmed)
            else:
                return {}

        if not isinstance(labeled, list):
            return {}

        return {
            item["description"]: item["category"]
            for item in labeled
            if "description" in item
            and "category" in item
            and item["category"] in ALL_LABELS
        }

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
        print(f"    ERROR: {e}")
        return {}
    finally:
        prompt_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)


def main():
    # Load all rows from full extraction
    input_path = OUTPUT_DIR / "full_unique_labeled.csv"
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Deduplicate by cleaned text — group rows, keep all raw variants
    cleaned_groups = defaultdict(list)
    for r in rows:
        cleaned_groups[r["cleaned"]].append(r)

    # Get unique descriptions to label (use raw description for context)
    unique_items = []
    for cleaned, group in cleaned_groups.items():
        total_occ = sum(int(r.get("occurrence_count", 1)) for r in group)
        unique_items.append({
            "cleaned": cleaned,
            "raw_example": group[0]["description"],
            "occurrence_count": total_occ,
            "group": group,
        })

    descriptions = [item["raw_example"] for item in unique_items]
    print(f"Total unique cleaned descriptions: {len(descriptions)}")
    print(f"Labeling in batches of {BATCH_SIZE}...")

    # Label in batches
    all_labels = {}
    num_batches = (len(descriptions) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(descriptions), BATCH_SIZE):
        batch = descriptions[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        result = label_batch_codex(batch, batch_num)
        labeled_count = len(result)
        all_labels.update(result)
        print(f"  Batch {batch_num}/{num_batches} ({i+1}-{i+len(batch)})... {labeled_count} labeled")

    print(f"\nTotal labeled: {len(all_labels)}/{len(descriptions)}")

    # Map labels back to unique items
    codex_labels = {}
    for item in unique_items:
        raw = item["raw_example"]
        if raw in all_labels:
            codex_labels[item["cleaned"]] = all_labels[raw]

    # Write output: one row per unique cleaned description
    output_path = OUTPUT_DIR / "codex_labeled.csv"
    out_fields = [
        "cleaned", "raw_example", "occurrence_count",
        "codex_category", "model_category", "model_confidence",
        "model_source", "gemini_category",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for item in unique_items:
            cleaned = item["cleaned"]
            first = item["group"][0]

            # Get Gemini label (most common across raw variants)
            gemini_counts = defaultdict(int)
            for r in item["group"]:
                if r.get("gemini_category"):
                    gemini_counts[r["gemini_category"]] += 1
            best_gemini = max(gemini_counts, key=gemini_counts.get) if gemini_counts else ""

            writer.writerow({
                "cleaned": cleaned,
                "raw_example": item["raw_example"],
                "occurrence_count": item["occurrence_count"],
                "codex_category": codex_labels.get(cleaned, ""),
                "model_category": first["model_category"],
                "model_confidence": first["model_confidence"],
                "model_source": first["model_source"],
                "gemini_category": best_gemini,
            })

    print(f"\nOutput: {output_path}")

    # Quick accuracy summary
    labeled_items = [
        item for item in unique_items if item["cleaned"] in codex_labels
    ]
    correct = sum(
        1
        for item in labeled_items
        if codex_labels[item["cleaned"]] == item["group"][0]["model_category"]
    )
    total = len(labeled_items)
    print(f"\nModel accuracy vs Codex labels:")
    print(f"  Unique: {correct}/{total} = {correct/total*100:.1f}%")

    weighted_correct = sum(
        item["occurrence_count"]
        for item in labeled_items
        if codex_labels[item["cleaned"]] == item["group"][0]["model_category"]
    )
    weighted_total = sum(item["occurrence_count"] for item in labeled_items)
    print(f"  Weighted: {weighted_correct}/{weighted_total} = {weighted_correct/weighted_total*100:.1f}%")


if __name__ == "__main__":
    main()
