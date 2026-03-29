#!/usr/bin/env python
"""Extract transactions from RBC bank statement PDFs.

Uses pdfplumber for text extraction, then Gemini CLI to structure into JSON.
Outputs: data/real/all_transactions.csv
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pdfplumber

# Paths to bank statement PDFs
MASTERCARD_DIR = Path(
    r"C:\Users\crims\Documents\0A04_Reference_Documents"
    r"\4B04_Bank\4C01_RBC\4D03_MasterCard\4E01_2025"
)
CHEQUING_DIR = Path(
    r"C:\Users\crims\Documents\0A04_Reference_Documents"
    r"\4B04_Bank\4C01_RBC\4D02_Noirt\4E01_2025"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "real"

GEMINI_PROMPT = """You are extracting bank transactions from raw text of a PDF statement.

Parse the text below and extract EVERY transaction into a JSON array.
Each transaction object must have:
- "date": posting date as "YYYY-MM-DD" (infer year from the statement period header)
- "description": the EXACT activity description as printed (preserve original casing, store numbers, city names)
- "amount": dollar amount as a positive float
- "type": "debit" for purchases/withdrawals/payments-out, "credit" for deposits/refunds/payments-in

Rules:
- Extract ONLY actual transactions from the activity table
- Skip: opening/closing balances, summaries, totals, interest rates, contact info, page headers
- Skip: the long reference numbers below some transactions (e.g., "55181364338462646134684")
- Skip: foreign currency lines and exchange rate lines
- For credit cards: purchases = "debit", refunds (negative amounts) = "credit"
- For chequing: withdrawals column = "debit", deposits column = "credit"
- Output ONLY valid JSON array, nothing else. No markdown, no explanation.

STATEMENT TEXT:
"""


def extract_pages_from_pdf(pdf_path: Path) -> list[str]:
    """Extract text from each page of a PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                pages.append(text)
    return pages


def _repair_truncated_json(text: str) -> list[dict]:
    """Try to salvage transactions from truncated JSON output."""
    # Find the last complete JSON object (ends with "}")
    last_brace = text.rfind("}")
    if last_brace == -1:
        return []

    # Trim to last complete object and close the array
    trimmed = text[: last_brace + 1].rstrip().rstrip(",") + "\n]"

    try:
        result = json.loads(trimmed)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    return []


def structure_with_gemini(raw_text: str, pdf_name: str) -> list[dict]:
    """Send extracted text to Gemini CLI for structured extraction."""
    import uuid

    full_prompt = GEMINI_PROMPT + raw_text

    # Write prompt to temp file with unique name to avoid Windows file locks
    temp_prompt = OUTPUT_DIR / f"_prompt_{uuid.uuid4().hex[:8]}.txt"
    temp_prompt.write_text(full_prompt, encoding="utf-8")

    try:
        result = subprocess.run(
            f'type "{temp_prompt}" | gemini -p ""',
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_DIR),
            shell=True,
        )

        if result.returncode != 0:
            print(f"    ERROR: gemini returned code {result.returncode}")
            print(f"    stderr: {result.stderr[:300]}")
            return []

        output = result.stdout.strip()

        # Strip node deprecation warnings
        lines = output.split("\n")
        output = "\n".join(
            l for l in lines
            if "DeprecationWarning" not in l and "node --trace" not in l
            and "Loaded cached" not in l
        ).strip()

        # Strip markdown code blocks if present
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()

        try:
            transactions = json.loads(output)
        except json.JSONDecodeError:
            # Try to repair truncated JSON
            transactions = _repair_truncated_json(output)
            if transactions:
                print(f"(repaired {len(transactions)} txns) ", end="", flush=True)
            else:
                debug_path = OUTPUT_DIR / f"debug_{pdf_name}.txt"
                debug_path.write_text(output)
                print(f"\n    ERROR: JSON parse failed, saved to {debug_path}")
                return []

        if not isinstance(transactions, list):
            print(f"    ERROR: Expected list, got {type(transactions)}")
            return []

        return transactions

    except subprocess.TimeoutExpired:
        print(f"    ERROR: Gemini timed out")
        return []
    finally:
        try:
            temp_prompt.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows file lock, will be cleaned up later


def process_pdf(pdf_path: Path, account_type: str) -> list[dict]:
    """Extract and structure transactions from a single PDF."""
    print(f"  {pdf_path.name}...", end=" ", flush=True)

    pages = extract_pages_from_pdf(pdf_path)
    if not pages:
        print("EMPTY")
        return []

    all_transactions = []
    for i, page_text in enumerate(pages):
        # Skip pages that don't look like they contain transactions
        if not any(kw in page_text.lower() for kw in [
            "date", "description", "activity", "withdrawal", "deposit",
            "amount", "posting", "transaction",
        ]):
            continue

        txns = structure_with_gemini(page_text, f"{pdf_path.stem}_p{i+1}")
        all_transactions.extend(txns)

    for t in all_transactions:
        t["source_file"] = pdf_path.name
        t["account_type"] = account_type

    print(f"{len(all_transactions)} transactions")
    return all_transactions


def write_csv(transactions: list[dict], output_path: Path) -> None:
    """Write transactions to CSV."""
    fieldnames = ["date", "description", "amount", "type", "source_file", "account_type"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in transactions:
            row = {k: t.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"\nWrote {len(transactions)} transactions to {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_transactions = []

    for pdf_dir, account_type in [
        (MASTERCARD_DIR, "mastercard"),
        (CHEQUING_DIR, "chequing"),
    ]:
        if not pdf_dir.exists():
            print(f"WARNING: {pdf_dir} not found, skipping")
            continue

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        print(f"\n{'='*60}")
        print(f"Processing {len(pdf_files)} {account_type} statements")
        print(f"{'='*60}")

        for pdf_path in pdf_files:
            transactions = process_pdf(pdf_path, account_type)
            all_transactions.extend(transactions)

    if all_transactions:
        write_csv(all_transactions, OUTPUT_DIR / "all_transactions.csv")

        mc = sum(1 for t in all_transactions if t.get("account_type") == "mastercard")
        chq = sum(1 for t in all_transactions if t.get("account_type") == "chequing")
        debits = sum(1 for t in all_transactions if t.get("type") == "debit")
        credits = sum(1 for t in all_transactions if t.get("type") == "credit")

        print(f"\nSummary:")
        print(f"  Total: {len(all_transactions)}")
        print(f"  MasterCard: {mc}, Chequing: {chq}")
        print(f"  Debits: {debits}, Credits: {credits}")
    else:
        print("\nNo transactions extracted.")


if __name__ == "__main__":
    main()
