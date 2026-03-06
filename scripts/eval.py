"""LegacyLens RAG Evaluation Script.

Tests all 8 code understanding features with 3 queries each (24 total).
Measures Precision@5 (retrieval quality) and Term Recall (generation quality).

Usage:
    # Against local server (default http://localhost:8000)
    python scripts/eval.py

    # Against deployed server
    python scripts/eval.py --url https://legacylens-production-fd39.up.railway.app

    # Custom pass thresholds
    python scripts/eval.py --p5-threshold 0.3 --recall-threshold 0.6
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

# ── Ground-truth evaluation dataset: 3 queries × 8 features = 24 ──

EVAL_CASES = [
    # ── Feature 1: Code Explanation ──
    {
        "feature": "Code Explanation",
        "query": "What does DGEMM do?",
        "expected_routines": ["DGEMM"],
        "expected_terms": ["matrix", "multiply", "double", "alpha", "beta", "Level 3"],
    },
    {
        "feature": "Code Explanation",
        "query": "Explain how SAXPY works",
        "expected_routines": ["SAXPY"],
        "expected_terms": ["vector", "scalar", "single", "Level 1"],
    },
    {
        "feature": "Code Explanation",
        "query": "What is the purpose of DTRSM?",
        "expected_routines": ["DTRSM"],
        "expected_terms": ["triangular", "solve", "matrix", "double", "Level 3"],
    },
    # ── Feature 2: Dependency Mapping ──
    {
        "feature": "Dependency Mapping",
        "query": "What routines does DGEMM call?",
        "expected_routines": ["DGEMM"],
        "expected_terms": ["XERBLA", "LSAME", "call"],
    },
    {
        "feature": "Dependency Mapping",
        "query": "Show the dependencies of DGEMV",
        "expected_routines": ["DGEMV"],
        "expected_terms": ["XERBLA", "LSAME", "call"],
    },
    {
        "feature": "Dependency Mapping",
        "query": "What subroutines does DTRSM depend on?",
        "expected_routines": ["DTRSM"],
        "expected_terms": ["XERBLA", "LSAME", "call"],
    },
    # ── Feature 3: Pattern Detection ──
    {
        "feature": "Pattern Detection",
        "query": "Find similar patterns across Level 3 routines",
        "expected_routines": ["DGEMM", "DSYMM", "DTRMM", "DTRSM"],
        "expected_terms": ["pattern", "Level 3", "matrix", "parameter"],
    },
    {
        "feature": "Pattern Detection",
        "query": "Compare SAXPY and DAXPY",
        "expected_routines": ["SAXPY", "DAXPY"],
        "expected_terms": ["single", "double", "precision", "vector", "Level 1"],
    },
    {
        "feature": "Pattern Detection",
        "query": "What patterns do the BLAS Level 1 routines share?",
        "expected_routines": ["DSCAL", "DAXPY", "DDOT", "DSWAP"],
        "expected_terms": ["Level 1", "vector", "loop", "increment"],
    },
    # ── Feature 4: Impact Analysis ──
    {
        "feature": "Impact Analysis",
        "query": "What would be affected if XERBLA changes?",
        "expected_routines": ["XERBLA"],
        "expected_terms": ["error", "called", "impact", "routines"],
    },
    {
        "feature": "Impact Analysis",
        "query": "What is the impact of changing LSAME?",
        "expected_routines": ["LSAME"],
        "expected_terms": ["character", "comparison", "used", "routines"],
    },
    {
        "feature": "Impact Analysis",
        "query": "If DSCAL is modified, what else is affected?",
        "expected_routines": ["DSCAL"],
        "expected_terms": ["scaling", "vector", "Level 1"],
    },
    # ── Feature 5: Documentation Generation ──
    {
        "feature": "Documentation Generation",
        "query": "Generate documentation for DGEMM",
        "expected_routines": ["DGEMM"],
        "expected_terms": ["Purpose", "Parameters", "BLAS Level", "matrix"],
    },
    {
        "feature": "Documentation Generation",
        "query": "Write documentation for DAXPY",
        "expected_routines": ["DAXPY"],
        "expected_terms": ["Purpose", "Parameters", "vector", "scalar"],
    },
    {
        "feature": "Documentation Generation",
        "query": "Create docs for DGEMV",
        "expected_routines": ["DGEMV"],
        "expected_terms": ["Purpose", "Parameters", "matrix", "Level 2"],
    },
    # ── Feature 6: Translation Hints ──
    {
        "feature": "Translation Hints",
        "query": "What is the modern equivalent of DGEMM?",
        "expected_routines": ["DGEMM"],
        "expected_terms": ["NumPy", "Python", "modern", "matrix"],
    },
    {
        "feature": "Translation Hints",
        "query": "How would I translate SAXPY to modern code?",
        "expected_routines": ["SAXPY"],
        "expected_terms": ["NumPy", "Python", "modern", "vector"],
    },
    {
        "feature": "Translation Hints",
        "query": "What modern libraries replace DGEMV?",
        "expected_routines": ["DGEMV"],
        "expected_terms": ["NumPy", "Python", "modern", "matrix"],
    },
    # ── Feature 7: Bug Pattern Search ──
    {
        "feature": "Bug Pattern Search",
        "query": "Are there potential bugs in DGEMM?",
        "expected_routines": ["DGEMM"],
        "expected_terms": ["parameter", "validation", "XERBLA"],
    },
    {
        "feature": "Bug Pattern Search",
        "query": "Find potential issues in DGEMV",
        "expected_routines": ["DGEMV"],
        "expected_terms": ["parameter", "validation", "error"],
    },
    {
        "feature": "Bug Pattern Search",
        "query": "What could go wrong in DTRSM?",
        "expected_routines": ["DTRSM"],
        "expected_terms": ["parameter", "validation", "error"],
    },
    # ── Feature 8: Business Logic Extraction ──
    {
        "feature": "Business Logic Extraction",
        "query": "What is the mathematical formula for DGEMM?",
        "expected_routines": ["DGEMM"],
        "expected_terms": ["C", "alpha", "beta", "matrix", "multiply"],
    },
    {
        "feature": "Business Logic Extraction",
        "query": "What algorithm does DGEMV compute?",
        "expected_routines": ["DGEMV"],
        "expected_terms": ["y", "alpha", "beta", "matrix", "vector"],
    },
    {
        "feature": "Business Logic Extraction",
        "query": "Extract the formula from DSYRK",
        "expected_routines": ["DSYRK"],
        "expected_terms": ["C", "alpha", "beta", "symmetric", "rank"],
    },
]


# ── HTTP helper ──


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
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"error": f"HTTP {e.code}: {body[:200]}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def check_health(base_url: str) -> bool:
    """Quick health check before running eval."""
    try:
        url = f"{base_url}/api/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "ok"
    except Exception:
        return False


# ── Metrics ──


def compute_p_at_5(sources: list[dict], expected_routines: list[str]) -> float:
    """Precision@5: fraction of top-5 retrieved sources that are relevant."""
    if not sources:
        return 0.0
    expected_upper = {r.upper() for r in expected_routines}
    top5 = sources[:5]
    relevant = sum(
        1
        for s in top5
        if s.get("chunk", {})
        .get("metadata", {})
        .get("subroutine_name", "")
        .upper()
        in expected_upper
    )
    return relevant / min(5, len(sources))


def compute_term_recall(answer: str, expected_terms: list[str]) -> float:
    """Term Recall: fraction of expected terms found in the answer."""
    if not expected_terms:
        return 1.0
    if not answer:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for term in expected_terms if term.lower() in answer_lower)
    return found / len(expected_terms)


# ── Evaluation runner ──


def evaluate_single(
    base_url: str, case: dict, index: int, total: int
) -> dict:
    """Run a single eval case and return metrics."""
    query = case["query"]
    feature = case["feature"]
    expected_routines = case["expected_routines"]
    expected_terms = case["expected_terms"]

    print(f"\n  [{index}/{total}] {feature}")
    print(f"    Query: {query}")

    start = time.time()
    response = post_query(base_url, query)
    wall_ms = (time.time() - start) * 1000

    # Handle errors
    if "error" in response:
        print(f"    ERROR: {response['error']}")
        return {
            "query": query,
            "feature": feature,
            "p_at_5": 0.0,
            "term_recall": 0.0,
            "passed": False,
            "latency_ms": wall_ms,
            "error": response["error"],
        }

    answer = response.get("answer", "")
    sources = response.get("sources", [])
    server_ms = response.get("query_time_ms", wall_ms)

    # Compute metrics
    p5 = compute_p_at_5(sources, expected_routines)
    recall = compute_term_recall(answer, expected_terms)

    # Find which terms were missing
    answer_lower = answer.lower()
    missing_terms = [t for t in expected_terms if t.lower() not in answer_lower]

    # Retrieved routine names
    retrieved = [
        s.get("chunk", {}).get("metadata", {}).get("subroutine_name", "?")
        for s in sources[:5]
    ]

    print(f"    Sources: {retrieved}")
    print(f"    P@5: {p5:.0%}  |  Term Recall: {recall:.0%}")
    if missing_terms:
        print(f"    Missing terms: {missing_terms}")
    print(f"    Latency: {server_ms:.0f}ms")

    return {
        "query": query,
        "feature": feature,
        "p_at_5": p5,
        "term_recall": recall,
        "passed": False,  # Set by caller based on thresholds
        "latency_ms": server_ms,
        "sources_count": len(sources),
        "answer_length": len(answer),
        "retrieved_routines": retrieved,
        "missing_terms": missing_terms,
    }


def run_eval(
    base_url: str, p5_threshold: float = 0.2, recall_threshold: float = 0.5
) -> list[dict]:
    """Run all 24 evaluation cases."""
    total = len(EVAL_CASES)
    results = []

    print(f"\n  Running {total} evaluation queries...")

    for i, case in enumerate(EVAL_CASES, 1):
        # Small delay between queries to avoid API rate limiting
        if i > 1:
            time.sleep(2)
        result = evaluate_single(base_url, case, i, total)

        # Apply pass/fail thresholds
        if "error" not in result:
            result["passed"] = (
                result["p_at_5"] >= p5_threshold
                and result["term_recall"] >= recall_threshold
            )

        status = "PASS" if result["passed"] else "FAIL"
        print(f"    Result: {status}")
        results.append(result)

    return results


# ── Reporting ──

FEATURES_ORDER = [
    "Code Explanation",
    "Dependency Mapping",
    "Pattern Detection",
    "Impact Analysis",
    "Documentation Generation",
    "Translation Hints",
    "Bug Pattern Search",
    "Business Logic Extraction",
]


def print_results_table(results: list[dict], base_url: str):
    """Print the formatted results table."""
    print("\n" + "=" * 80)
    print("  LegacyLens RAG Evaluation Results")
    print(f"  Target: {base_url}")
    print(f"  Date:   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Per-feature aggregation
    feature_stats = {}
    for feat in FEATURES_ORDER:
        feat_results = [r for r in results if r["feature"] == feat]
        if not feat_results:
            continue
        feature_stats[feat] = {
            "queries": len(feat_results),
            "passed": sum(1 for r in feat_results if r["passed"]),
            "failed": sum(1 for r in feat_results if not r["passed"]),
            "mean_p5": statistics.mean(r["p_at_5"] for r in feat_results),
            "mean_recall": statistics.mean(r["term_recall"] for r in feat_results),
        }

    # Table header
    header = f"{'Feature':<28} {'Queries':>7}  {'Passed':>6}  {'Failed':>6}  {'Mean P@5':>9}  {'Mean Term Recall':>17}"
    separator = "\u2500" * 80
    print(f"\n{header}")
    print(separator)

    # Per-feature rows
    for feat in FEATURES_ORDER:
        if feat not in feature_stats:
            continue
        s = feature_stats[feat]
        print(
            f"{feat:<28} {s['queries']:>7}  {s['passed']:>6}  {s['failed']:>6}"
            f"  {s['mean_p5']:>8.1%}  {s['mean_recall']:>16.1%}"
        )

    # Overall row
    print(separator)
    total_queries = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    total_failed = total_queries - total_passed
    overall_p5 = statistics.mean(r["p_at_5"] for r in results) if results else 0
    overall_recall = (
        statistics.mean(r["term_recall"] for r in results) if results else 0
    )
    print(
        f"{'OVERALL':<28} {total_queries:>7}  {total_passed:>6}  {total_failed:>6}"
        f"  {overall_p5:>8.1%}  {overall_recall:>16.1%}"
    )

    # Latency stats
    latencies = [r["latency_ms"] for r in results if "error" not in r]
    if latencies:
        mean_lat = statistics.mean(latencies)
        median_lat = statistics.median(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else max(latencies)
        print(
            f"\nLatency:  Mean: {mean_lat:.0f}ms  |  Median: {median_lat:.0f}ms  |  P95: {p95_lat:.0f}ms"
        )

    # Failures summary
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            reason = f.get("error", "")
            if not reason:
                parts = []
                if f["p_at_5"] < 0.2:
                    parts.append(f"P@5={f['p_at_5']:.0%}")
                if f["term_recall"] < 0.5:
                    missing = f.get("missing_terms", [])
                    parts.append(f"Recall={f['term_recall']:.0%} missing={missing}")
                reason = "; ".join(parts)
            print(f"    [{f['feature']}] \"{f['query']}\"")
            print(f"      → {reason}")

    print()


def save_results(results: list[dict], output_path: str):
    """Save raw results to JSON."""
    # Build summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "mean_p_at_5": statistics.mean(r["p_at_5"] for r in results) if results else 0,
        "mean_term_recall": (
            statistics.mean(r["term_recall"] for r in results) if results else 0
        ),
    }

    output = {"summary": summary, "results": results}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Raw results saved to {output_path}")


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="LegacyLens RAG Evaluation")
    parser.add_argument(
        "--url",
        default=os.getenv("EVAL_URL", "http://localhost:8000"),
        help="Base URL of the LegacyLens server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--p5-threshold",
        type=float,
        default=0.2,
        help="Minimum P@5 to pass (default: 0.2)",
    )
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=0.5,
        help="Minimum Term Recall to pass (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Output file for raw JSON results (default: eval_results.json)",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print("\n" + "=" * 80)
    print("  LegacyLens RAG Evaluation")
    print(f"  Target: {base_url}")
    print(f"  Queries: {len(EVAL_CASES)}")
    print(f"  Features: {len(FEATURES_ORDER)}")
    print(f"  Pass thresholds: P@5 >= {args.p5_threshold:.0%}, Term Recall >= {args.recall_threshold:.0%}")
    print("=" * 80)

    # Health check
    print("\n  Checking server health...")
    if not check_health(base_url):
        print(f"\n  ERROR: Server not reachable at {base_url}")
        print("  Start it with: uvicorn app.main:app --reload")
        sys.exit(1)
    print("  Server is healthy!")

    # Run evaluation
    total_start = time.time()
    results = run_eval(base_url, args.p5_threshold, args.recall_threshold)
    total_seconds = time.time() - total_start

    # Print results table
    print_results_table(results, base_url)

    print(f"  Total eval time: {total_seconds:.1f}s")

    # Save raw results
    save_results(results, args.output)

    # Exit code: 0 if all passed, 1 if any failed
    failed = sum(1 for r in results if not r["passed"])
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
