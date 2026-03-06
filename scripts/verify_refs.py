"""Verify that LLM answers contain valid file/line references.

Queries the server with a set of test questions, then checks each answer for:
  1. Fortran filename references (e.g. dgemm.f, saxpy.f)
  2. Line number patterns (e.g. "line 42", "lines 10-20", "L15")
  3. Filename existence on disk (cross-checked against blas_src/)

Usage:
    # Against local server
    python scripts/verify_refs.py

    # Against deployed server
    python scripts/verify_refs.py --url https://legacylens-production-fd39.up.railway.app

    # Skip disk verification (no blas_src/ needed)
    python scripts/verify_refs.py --skip-disk
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, ".")

# ── Test queries that should produce file/line references ──

REF_QUERIES = [
    {"query": "What does DGEMM do?", "expected_file": "dgemm.f"},
    {"query": "Explain how SAXPY works", "expected_file": "saxpy.f"},
    {"query": "What is the purpose of DTRSM?", "expected_file": "dtrsm.f"},
    {"query": "Generate documentation for DGEMV", "expected_file": "dgemv.f"},
    {"query": "Are there potential bugs in DSYRK?", "expected_file": "dsyrk.f"},
    {"query": "What routines does DGEMM call?", "expected_file": "dgemm.f"},
]

# ── Regex patterns for reference detection ──

# Fortran filenames: word.f, word.f90, etc.
RE_FILENAME = re.compile(r"\b(\w+\.f(?:90|95|or|pp)?)\b", re.IGNORECASE)

# Line references: "line 42", "lines 10-20", "line 5", "L15", ":42"
RE_LINE_REF = re.compile(
    r"(?:"
    r"lines?\s+(\d+)"       # "line 42" or "lines 10"
    r"|L(\d+)"              # "L15"
    r"|:(\d+)"              # ":42" (as in file:line)
    r"|line\s+(\d+)\s*[-–]\s*(\d+)"  # "line 10-20"
    r")",
    re.IGNORECASE,
)


def post_query(base_url: str, query: str, timeout: float = 120.0) -> dict:
    """POST to /api/query and return the parsed JSON response."""
    url = f"{base_url}/api/query"
    data = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def scan_disk_files(source_dir: str) -> set[str]:
    """Build a set of lowercase Fortran filenames on disk."""
    names = set()
    src = Path(source_dir)
    if not src.exists():
        return names
    for f in src.rglob("*"):
        if f.suffix.lower() in {".f", ".f90", ".f95", ".for", ".fpp"}:
            names.add(f.name.lower())
    return names


def verify_answer(answer: str, expected_file: str, disk_files: set[str]) -> dict:
    """Check a single answer for file/line references.

    Returns a dict with:
        has_filename:  bool — answer mentions any .f filename
        has_expected:  bool — answer mentions the expected file
        has_line_ref:  bool — answer contains a line number reference
        filenames:     list — all .f filenames found in the answer
        valid_files:   list — filenames that exist on disk
        invalid_files: list — filenames NOT found on disk (potential hallucinations)
    """
    # Extract filenames
    filenames = [m.lower() for m in RE_FILENAME.findall(answer)]
    unique_files = list(dict.fromkeys(filenames))  # dedupe, preserve order

    # Cross-check against disk
    valid = [f for f in unique_files if f in disk_files] if disk_files else unique_files
    invalid = [f for f in unique_files if f not in disk_files] if disk_files else []

    # Check for line references
    line_matches = RE_LINE_REF.findall(answer)

    return {
        "has_filename": len(unique_files) > 0,
        "has_expected": expected_file.lower() in [f.lower() for f in unique_files],
        "has_line_ref": len(line_matches) > 0,
        "filenames": unique_files,
        "valid_files": valid,
        "invalid_files": invalid,
        "line_ref_count": len(line_matches),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify file/line references in LLM answers")
    parser.add_argument(
        "--url",
        default=os.getenv("EVAL_URL", "http://localhost:8000"),
        help="Base URL of the LegacyLens server",
    )
    parser.add_argument(
        "--source-dir",
        default="blas_src",
        help="Path to BLAS source for disk verification (default: blas_src)",
    )
    parser.add_argument(
        "--skip-disk",
        action="store_true",
        help="Skip disk-based file verification",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    disk_files = set() if args.skip_disk else scan_disk_files(args.source_dir)

    print("\n" + "=" * 65)
    print("  LegacyLens Reference Verification")
    print(f"  Target: {base_url}")
    if disk_files:
        print(f"  Disk files: {len(disk_files)} Fortran files in {args.source_dir}/")
    else:
        print("  Disk verification: SKIPPED")
    print("=" * 65)

    results = []
    total = len(REF_QUERIES)

    for i, case in enumerate(REF_QUERIES, 1):
        query = case["query"]
        expected = case["expected_file"]

        print(f"\n  [{i}/{total}] {query}")

        if i > 1:
            time.sleep(2)  # Rate limit courtesy

        response = post_query(base_url, query)

        if "error" in response:
            print(f"    ERROR: {response['error']}")
            results.append({"query": query, "error": response["error"]})
            continue

        answer = response.get("answer", "")
        check = verify_answer(answer, expected, disk_files)

        # Status symbols
        file_ok = "✅" if check["has_expected"] else "❌"
        line_ok = "✅" if check["has_line_ref"] else "⚠️"
        disk_ok = "✅" if not check["invalid_files"] else f"⚠️  hallucinated: {check['invalid_files']}"

        print(f"    Expected file ({expected}): {file_ok}")
        print(f"    Files referenced: {check['filenames'][:5]}")
        print(f"    Line references:  {line_ok} ({check['line_ref_count']} found)")
        if disk_files:
            print(f"    Disk validation:  {disk_ok}")

        results.append({
            "query": query,
            "expected_file": expected,
            **check,
        })

    # ── Summary ──
    valid = [r for r in results if "error" not in r]
    has_file = sum(1 for r in valid if r["has_filename"])
    has_expected = sum(1 for r in valid if r["has_expected"])
    has_line = sum(1 for r in valid if r["has_line_ref"])
    hallucinated = sum(1 for r in valid if r["invalid_files"])

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"  Queries run:          {len(valid)}/{total}")
    print(f"  Has any file ref:     {has_file}/{len(valid)}  ({has_file/len(valid):.0%})" if valid else "")
    print(f"  Has expected file:    {has_expected}/{len(valid)}  ({has_expected/len(valid):.0%})" if valid else "")
    print(f"  Has line references:  {has_line}/{len(valid)}  ({has_line/len(valid):.0%})" if valid else "")
    if disk_files:
        print(f"  Hallucinated files:   {hallucinated}/{len(valid)}")
    print("=" * 65)

    # Exit code: 0 if >=80% have file refs, 1 otherwise
    if valid and has_file / len(valid) < 0.8:
        print("\n  FAIL: Less than 80% of answers contain file references")
        sys.exit(1)
    else:
        print("\n  PASS: Answers contain sufficient file/line references")
        sys.exit(0)


if __name__ == "__main__":
    main()
