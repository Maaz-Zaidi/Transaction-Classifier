#!/usr/bin/env python
"""Extract transaction descriptions from RBC bank statement PDFs using pymupdf.

Uses pymupdf (fitz) instead of pdfplumber to get properly-spaced text.
Only extracts descriptions, no amounts/dates/structuring needed.
Outputs: data/real/descriptions_ocr.csv
"""

import re
import sys
from pathlib import Path

import fitz  # pymupdf

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "real"

MASTERCARD_DIR = Path(
    r"C:\Users\crims\Documents\0A04_Reference_Documents"
    r"\4B04_Bank\4C01_RBC\4D03_MasterCard\4E01_2025"
)
CHEQUING_DIR = Path(
    r"C:\Users\crims\Documents\0A04_Reference_Documents"
    r"\4B04_Bank\4C01_RBC\4D02_Noirt\4E01_2025"
)

# Line classifiers

# MasterCard dates: "DEC 08", "JAN 02", etc.
_MC_DATE = re.compile(
    r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{1,2}$"
)
# Reference numbers: long digit strings (18+ digits)
_REF_NUMBER = re.compile(r"^\d{15,}$")
# Amounts: $1,234.56 or -$1,234.56 or just 1,234.56
_AMOUNT = re.compile(r"^-?\$?[\d,]+\.\d{2}$")
# Foreign currency / exchange rate lines
_FOREX = re.compile(r"^(Foreign Currency|Exchange rate)", re.IGNORECASE)

# Chequing dates: "27 Dec", "2 Jan", etc.
_CHQ_DATE = re.compile(
    r"^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$",
    re.IGNORECASE,
)

# MasterCard header lines to skip
_MC_HEADERS = {
    "TRANSACTION POSTING", "ACTIVITY DESCRIPTION", "AMOUNT ($)", "DATE",
}

# Chequing header lines to skip
_CHQ_HEADERS = {
    "Date", "Description", "Withdrawals ($)", "Deposits ($)", "Balance ($)",
    "Opening Balance", "Closing Balance",
    "Details of your account activity",
    "Details of your account activity - continued",
}

# Generic skip patterns (page headers, boilerplate)
_SKIP_PATTERNS = [
    re.compile(r"^RBC.*Mastercard", re.IGNORECASE),
    re.compile(r"^\d+ OF \d+$"),
    re.compile(r"^MAAZ ZAIDI"),
    re.compile(r"^5415 90\*"),
    re.compile(r"^STATEMENT FROM"),
    re.compile(r"^Thank you for choosing"),
    re.compile(r"^PRIMARY \(continued\)"),
    re.compile(r"^\d+ of \d+$"),
    re.compile(r"^Royal Bank of Canada"),
    re.compile(r"^C\.P\. \d+"),
    re.compile(r"^Your RBC personal"),
    re.compile(r"^From \w+ \d+"),
    re.compile(r"^Your account number"),
    re.compile(r"^How to reach us"),
    re.compile(r"^1-800-"),
    re.compile(r"^www\."),
    re.compile(r"^https://"),
    re.compile(r"^RBPDA"),
    re.compile(r"^\*\d+[A-Z]"),
    re.compile(r"^Your (opening|closing) balance"),
    re.compile(r"^Total (deposits|withdrawals)"),
    re.compile(r"^[=+\-] ?\$"),
    re.compile(r"^Summary of your account"),
    re.compile(r"^RBC Advantage Banking"),
    re.compile(r"^\d+ [A-Z]+ RD,"),
    re.compile(r"^Please check this"),
    re.compile(r"^If you opted"),
    re.compile(r"^indicate that"),
    re.compile(r"^Please retain"),
    re.compile(r"^.Registered trade"),
    re.compile(r"^Royal Bank of Canada GST"),
    re.compile(r"^Royal Trust"),
    re.compile(r"^The Royal Trust"),
    re.compile(r"^Important information"),
    re.compile(r"^Protect your PIN"),
    re.compile(r"^Never share"),
    re.compile(r"^Cover the key"),
    re.compile(r"^Here are four ways"),
    re.compile(r"^[a-d]\) "),
    re.compile(r"^or send a text"),
    re.compile(r"^company or agency"),
    re.compile(r"^or a gift card"),
    re.compile(r"^of a higher amount"),
    re.compile(r"^Stay Informed"),
    re.compile(r"^Quick, convenient"),
    re.compile(r"^Other payment options"),
    re.compile(r"^RBC Royal Bank ATM"),
    re.compile(r"^RBC Online Banking"),
    re.compile(r"^RBC Mobile app"),
    re.compile(r"^Earn \d+%"),
    re.compile(r"^For all other purchases"),
    re.compile(r"^backI"),
    re.compile(r"^IRestrictions"),
    re.compile(r"^Visit www"),
    re.compile(r"^RBC ROYAL BANK$"),
    re.compile(r"^CREDIT CARD PAYMENT"),
    re.compile(r"^P\.O\.BOX"),
    re.compile(r"^TORONTO, ONTARIO"),
    re.compile(r"^NEW BALANCE$"),
    re.compile(r"^MINIMUM PAYMENT$"),
    re.compile(r"^PAYMENT DUE DATE$"),
    re.compile(r"^AMOUNT PAID$"),
    re.compile(r"^\$$"),
    re.compile(r"^01\d{4}$"),
    re.compile(r"^\d{2} [A-Z]+MEADOW"),
    # E-transfer reference codes (alphanumeric, ~12 chars)
    re.compile(r"^[A-Za-z0-9]{10,14}$"),
]


def _is_skip_line(line: str) -> bool:
    """Check if a line is boilerplate/header that should be skipped."""
    stripped = line.strip()
    if not stripped or stripped == ".":
        return True
    if stripped in _MC_HEADERS or stripped in _CHQ_HEADERS:
        return True
    for pattern in _SKIP_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


def extract_mastercard_descriptions(pdf_path: Path) -> list[str]:
    """Extract transaction descriptions from a MasterCard statement PDF."""
    descriptions = []
    doc = fitz.open(str(pdf_path))

    # Stop markers, anything after these on a page is not transactions
    stop_markers = {
        "TOTAL ACCOUNT BALANCE", "Time to Pay", "CASH BACK SUMMARY",
        "IMPORTANT INFORMATION", "CONTACT US", "PAYMENTS & INTEREST RATES",
        "Thank you for choosing",
    }

    # Non-transaction lines that appear inside the table
    skip_descriptions = re.compile(
        r"^(CASH (ADVANCE|BACK) (INTEREST|REWARD)|"
        r"Previous Cash Back|Cash Back on|New Cash Back)",
        re.IGNORECASE,
    )
    # pymupdf sometimes joins a posting date to the description on one line
    _leading_date = re.compile(
        r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{1,2}\s+",
    )

    for page in doc:
        text = page.get_text()
        lines = text.strip().split("\n")

        # Find where transaction data starts on this page
        start = None
        for idx, line in enumerate(lines):
            if "ACTIVITY DESCRIPTION" in line:
                # Skip past the DATE/DATE header lines after ACTIVITY DESCRIPTION
                start = idx + 1
                while start < len(lines) and lines[start].strip() in (
                    "AMOUNT ($)", "DATE", ""
                ):
                    start += 1
                break

        if start is None:
            continue

        i = start
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line or line == ".":
                continue

            # Stop at end-of-table markers
            if any(line.startswith(m) for m in stop_markers):
                break

            if _MC_DATE.match(line):
                continue
            if _REF_NUMBER.match(line):
                continue
            if _AMOUNT.match(line):
                continue
            if _FOREX.match(line):
                continue
            if skip_descriptions.match(line):
                continue

            # Strip leading date if pymupdf joined it with the description
            line = _leading_date.sub("", line).strip()
            if not line:
                continue

            # What remains should be a transaction description
            descriptions.append(line)

    doc.close()
    return descriptions


def extract_chequing_descriptions(pdf_path: Path) -> list[str]:
    """Extract transaction descriptions from a chequing statement PDF.

    Chequing descriptions can span multiple lines, so consecutive ones get joined.
    """
    descriptions = []
    doc = fitz.open(str(pdf_path))

    for page in doc:
        text = page.get_text()
        lines = text.strip().split("\n")

        # Find where transaction activity starts
        activity_start = None
        for idx, line in enumerate(lines):
            if "Opening Balance" in line or "Details of your account activity" in line:
                activity_start = idx + 1
                break

        if activity_start is None:
            continue

        # Parse the activity section
        current_desc_lines: list[str] = []
        i = activity_start
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line or line == ".":
                continue

            # Stop at closing balance or boilerplate
            if "Closing Balance" in line:
                break
            if line.startswith("Please check"):
                break

            if _is_skip_line(line):
                continue

            # Dates signal a new transaction
            if _CHQ_DATE.match(line):
                # Save any accumulated description
                if current_desc_lines:
                    descriptions.append(" ".join(current_desc_lines))
                    current_desc_lines = []
                continue

            # Amounts/balances
            if _AMOUNT.match(line):
                # Save accumulated description before this amount
                if current_desc_lines:
                    descriptions.append(" ".join(current_desc_lines))
                    current_desc_lines = []
                continue

            # Otherwise it's part of a description
            current_desc_lines.append(line)

        # Don't forget the last one
        if current_desc_lines:
            descriptions.append(" ".join(current_desc_lines))

    doc.close()
    return descriptions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_descriptions: list[dict] = []

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

        extract_fn = (
            extract_mastercard_descriptions
            if account_type == "mastercard"
            else extract_chequing_descriptions
        )

        for pdf_path in pdf_files:
            descs = extract_fn(pdf_path)
            print(f"  {pdf_path.name}: {len(descs)} descriptions")
            for d in descs:
                all_descriptions.append({
                    "description": d,
                    "account_type": account_type,
                    "source_file": pdf_path.name,
                })

    if not all_descriptions:
        print("\nNo descriptions extracted.")
        return

    # Write CSV
    import csv

    output_path = OUTPUT_DIR / "descriptions_ocr.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["description", "account_type", "source_file"])
        writer.writeheader()
        writer.writerows(all_descriptions)

    mc = sum(1 for d in all_descriptions if d["account_type"] == "mastercard")
    chq = sum(1 for d in all_descriptions if d["account_type"] == "chequing")

    print(f"\nTotal: {len(all_descriptions)} descriptions ({mc} MC, {chq} chequing)")
    print(f"Saved to {output_path}")

    # Print sample
    print(f"\nSample descriptions:")
    seen = set()
    for d in all_descriptions:
        desc = d["description"]
        if desc not in seen:
            seen.add(desc)
            print(f"  [{d['account_type'][:2].upper()}] {desc}")
        if len(seen) >= 25:
            break


if __name__ == "__main__":
    main()
