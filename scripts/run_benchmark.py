"""Benchmark script for LegacyLens RAG pipeline.

Runs 10 benchmark queries covering all 4 features and BLAS levels,
measures retrieval precision, and documents failure modes.

Usage:
    python scripts/run_benchmark.py           # retrieval-only (fast, no LLM calls)
    python scripts/run_benchmark.py --full    # retrieval + generation (uses Claude credits)
"""

import sys
import time
import json

# Allow imports from project root
sys.path.insert(0, ".")

from app.retrieval.search import search_codebase, build_context
from app.retrieval.query_classifier import classify_query, get_search_params

# ── Ground-truth benchmark dataset ──
BENCHMARK_CASES = [
    # ── Feature: EXPLAIN ──
    {
        "query": "What does DGEMM do?",
        "expected_type": "explain",
        "expected_routines": ["DGEMM"],
        "description": "Level 3 — flagship matrix multiply, exact name match",
    },
    {
        "query": "How does the dot product work in BLAS?",
        "expected_type": "explain",
        "expected_routines": ["DDOT", "SDOT", "CDOTU", "ZDOTU"],
        "description": "Level 1 — semantic search for 'dot product', should find xDOT variants",
    },
    {
        "query": "Explain how BLAS handles vector scaling",
        "expected_type": "explain",
        "expected_routines": ["DSCAL", "SSCAL", "CSCAL", "ZSCAL"],
        "description": "Level 1 — concept search (scaling), should find xSCAL family",
    },
    # ── Feature: DOCUMENT ──
    {
        "query": "Generate documentation for DTRSV",
        "expected_type": "document",
        "expected_routines": ["DTRSV"],
        "description": "Level 2 — triangular solve, exact name in query",
    },
    {
        "query": "Write documentation for the swap routines",
        "expected_type": "document",
        "expected_routines": ["SSWAP", "DSWAP", "CSWAP", "ZSWAP"],
        "description": "Level 1 — all 4 data type variants of SWAP",
    },
    # ── Feature: PATTERN ──
    {
        "query": "Find similar patterns across Level 3 BLAS routines",
        "expected_type": "pattern",
        "expected_routines": ["DGEMM", "DSYMM", "DTRMM", "DSYRK", "DTRSM"],
        "description": "Level 3 — should cluster multiple matrix-matrix ops",
    },
    {
        "query": "Compare the single and double precision AXPY routines",
        "expected_type": "pattern",
        "expected_routines": ["SAXPY", "DAXPY"],
        "description": "Level 1 — S vs D precision comparison",
    },
    # ── Feature: LOGIC ──
    {
        "query": "What is the mathematical formula for DGEMV?",
        "expected_type": "logic",
        "expected_routines": ["DGEMV"],
        "description": "Level 2 — matrix-vector multiply formula extraction",
    },
    {
        "query": "What algorithm does DSYRK use to compute the result?",
        "expected_type": "logic",
        "expected_routines": ["DSYRK"],
        "description": "Level 3 — symmetric rank-k update algorithm",
    },
    # ── Edge case ──
    {
        "query": "How are errors handled in BLAS?",
        "expected_type": "explain",
        "expected_routines": ["XERBLA"],
        "description": "Cross-cutting — error handler, tests semantic understanding",
    },
]


def measure_retrieval(case):
    """Measure retrieval precision for a single benchmark case."""
    query = case["query"]
    expected_type = case["expected_type"]
    expected_routines = [r.upper() for r in case["expected_routines"]]

    # Classify
    query_type = classify_query(query)
    search_params = get_search_params(query_type)
    classified_correctly = query_type.value == expected_type

    # Search
    results, search_time_ms = search_codebase(query, top_k=search_params["top_k"])

    # Measure precision: how many returned results match expected routines?
    returned_routines = [
        r.chunk.metadata.subroutine_name.upper() for r in results
    ]
    hits = [r for r in returned_routines if r in expected_routines]
    recall = len(set(hits)) / len(set(expected_routines)) if expected_routines else 0
    precision = len(hits) / len(returned_routines) if returned_routines else 0

    # Top-1 accuracy: is the best result one of the expected routines?
    top1_hit = returned_routines[0] in expected_routines if returned_routines else False

    # Scores
    scores = [r.score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    top_score = scores[0] if scores else 0

    return {
        "query": query,
        "description": case["description"],
        "classified_correctly": classified_correctly,
        "expected_type": expected_type,
        "actual_type": query_type.value,
        "expected_routines": expected_routines,
        "returned_routines": returned_routines,
        "top1_hit": top1_hit,
        "recall": recall,
        "precision": precision,
        "avg_score": avg_score,
        "top_score": top_score,
        "num_results": len(results),
        "search_time_ms": search_time_ms,
    }


def run_benchmark(full_mode=False):
    """Run all benchmark cases and print results."""
    print("=" * 70)
    print("LegacyLens RAG Benchmark")
    print("Mode: %s" % ("FULL (retrieval + generation)" if full_mode else "RETRIEVAL ONLY"))
    print("Cases: %d" % len(BENCHMARK_CASES))
    print("=" * 70)

    results = []
    total_start = time.time()

    for i, case in enumerate(BENCHMARK_CASES, 1):
        print("\n[%d/%d] %s" % (i, len(BENCHMARK_CASES), case["query"]))
        print("  Expected: %s -> %s" % (case["expected_type"], case["expected_routines"]))

        result = measure_retrieval(case)
        results.append(result)

        status = "PASS" if result["top1_hit"] else "FAIL"
        classify_status = "PASS" if result["classified_correctly"] else "WARN"
        print("  Classify: %s (%s)" % (classify_status, result["actual_type"]))
        print("  Top-1:    %s %s" % (status, result["returned_routines"][:3]))
        print("  Recall:   %.0f%% | Precision: %.0f%%" % (result["recall"] * 100, result["precision"] * 100))
        print("  Scores:   top=%.3f, avg=%.3f" % (result["top_score"], result["avg_score"]))
        print("  Time:     %.0fms" % result["search_time_ms"])

    # ── Full mode: also test generation for first 3 cases ──
    if full_mode:
        from app.retrieval.generator import generate_answer

        print("\n" + "=" * 70)
        print("GENERATION QUALITY (first 3 cases)")
        print("=" * 70)

        for i, case in enumerate(BENCHMARK_CASES[:3], 1):
            query = case["query"]
            query_type = classify_query(query)
            search_params = get_search_params(query_type)
            search_results, _ = search_codebase(query, top_k=search_params["top_k"])
            if search_results:
                context = build_context(search_results)
                gen_start = time.time()
                answer = generate_answer(query, context, query_type)
                gen_ms = (time.time() - gen_start) * 1000
                print("\n[%d] %s" % (i, query))
                print("  Generation time: %.0fms" % gen_ms)
                print("  Answer length: %d chars" % len(answer))
                print("  First 200 chars: %s..." % answer[:200])

    # ── Summary ──
    total_time = (time.time() - total_start) * 1000

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    classify_acc = sum(1 for r in results if r["classified_correctly"]) / len(results)
    top1_acc = sum(1 for r in results if r["top1_hit"]) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_top_score = sum(r["top_score"] for r in results) / len(results)

    print("  Classification accuracy:  %.0f%% (%d/%d)" % (classify_acc * 100, sum(1 for r in results if r["classified_correctly"]), len(results)))
    print("  Top-1 retrieval accuracy: %.0f%% (%d/%d)" % (top1_acc * 100, sum(1 for r in results if r["top1_hit"]), len(results)))
    print("  Average recall:           %.0f%%" % (avg_recall * 100))
    print("  Average precision:        %.0f%%" % (avg_precision * 100))
    print("  Average top score:        %.3f" % avg_top_score)
    print("  Total time:               %.0fms" % total_time)

    # ── Failure analysis ──
    failures = [r for r in results if not r["top1_hit"]]
    if failures:
        print("\n  FAILURE MODES (%d cases):" % len(failures))
        for f in failures:
            print('    FAIL "%s"' % f["query"])
            print("       Expected: %s" % f["expected_routines"])
            print("       Got: %s" % f["returned_routines"][:3])
            if f["top_score"] < 0.3:
                print("       -> Low similarity (%.3f) — query may be too abstract" % f["top_score"])
            else:
                print("       -> Wrong routine retrieved — embedding missed intent")
    else:
        print("\n  No failures!")

    # Save raw results
    with open("benchmark_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\n  Raw results saved to benchmark_results.json")

    return results


if __name__ == "__main__":
    full = "--full" in sys.argv
    run_benchmark(full_mode=full)
