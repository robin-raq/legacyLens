"""Tests for search re-ranking: routine name extraction and score boosting."""

import time
import pytest
from app.models import CodeChunk, ChunkMetadata, SearchResult


def _make_result(name: str, score: float) -> SearchResult:
    """Helper: create a SearchResult with the given subroutine name and score."""
    return SearchResult(
        chunk=CodeChunk(
            id=f"test::{name}",
            text=f"SUBROUTINE {name}",
            metadata=ChunkMetadata(
                file_path=f"SRC/{name.lower()}.f",
                start_line=1,
                end_line=100,
                subroutine_name=name,
                blas_level="3",
                data_type="double real",
                description=f"Test {name}",
                line_count=100,
            ),
        ),
        score=score,
    )


# ── extract_routine_names ──


class TestExtractRoutineNames:
    def test_single_routine_name(self):
        from app.retrieval.search import extract_routine_names
        assert extract_routine_names("What does DGEMM do?") == {"DGEMM"}

    def test_multiple_routine_names(self):
        from app.retrieval.search import extract_routine_names
        assert extract_routine_names("Compare SAXPY and DAXPY") == {"SAXPY", "DAXPY"}

    def test_no_routine_name(self):
        from app.retrieval.search import extract_routine_names
        assert extract_routine_names("How are errors handled in BLAS?") == set()

    def test_utility_routine_xerbla(self):
        from app.retrieval.search import extract_routine_names
        assert "XERBLA" in extract_routine_names("What if XERBLA changes?")

    def test_utility_routine_lsame(self):
        from app.retrieval.search import extract_routine_names
        assert "LSAME" in extract_routine_names("What does LSAME do?")

    def test_case_insensitive(self):
        from app.retrieval.search import extract_routine_names
        assert extract_routine_names("explain dgemm") == {"DGEMM"}

    def test_does_not_match_short_words(self):
        from app.retrieval.search import extract_routine_names
        # "BLAS", "DOES", "THE" should not match
        result = extract_routine_names("How does the dot product work?")
        assert result == set()

    def test_all_four_prefixes(self):
        from app.retrieval.search import extract_routine_names
        result = extract_routine_names("Compare SGEMM, DGEMM, CGEMM, ZGEMM")
        assert result == {"SGEMM", "DGEMM", "CGEMM", "ZGEMM"}

    def test_level1_routines(self):
        from app.retrieval.search import extract_routine_names
        result = extract_routine_names("Explain DSCAL and DSWAP")
        assert result == {"DSCAL", "DSWAP"}

    def test_level2_routines(self):
        from app.retrieval.search import extract_routine_names
        result = extract_routine_names("What does DGEMV compute?")
        assert result == {"DGEMV"}


# ── rerank_results ──


class TestRerankResults:
    def test_boost_exact_match_to_top(self):
        """Exact name match should be boosted above higher-scored non-matches."""
        from app.retrieval.search import rerank_results
        results = [
            _make_result("DGEMV", 0.90),
            _make_result("DGEMM", 0.80),
            _make_result("SGEMM", 0.70),
        ]
        reranked = rerank_results(results, "What does DGEMM do?", top_k=3)
        assert reranked[0].chunk.metadata.subroutine_name == "DGEMM"

    def test_no_names_no_change(self):
        """Conceptual query with no routine names preserves original order."""
        from app.retrieval.search import rerank_results
        results = [
            _make_result("DGEMV", 0.90),
            _make_result("DGEMM", 0.80),
            _make_result("SGEMM", 0.70),
        ]
        reranked = rerank_results(results, "How are errors handled?", top_k=3)
        names = [r.chunk.metadata.subroutine_name for r in reranked]
        assert names == ["DGEMV", "DGEMM", "SGEMM"]

    def test_trims_to_top_k(self):
        """Result list is trimmed to top_k after re-ranking."""
        from app.retrieval.search import rerank_results
        results = [_make_result(f"R{i}", 0.9 - i * 0.1) for i in range(6)]
        reranked = rerank_results(results, "general query", top_k=3)
        assert len(reranked) == 3

    def test_multiple_matches_both_boosted(self):
        """Two mentioned names should both appear in top positions."""
        from app.retrieval.search import rerank_results
        results = [
            _make_result("DGEMM", 0.90),
            _make_result("DGEMV", 0.85),
            _make_result("SAXPY", 0.60),
            _make_result("DAXPY", 0.55),
        ]
        reranked = rerank_results(results, "Compare SAXPY and DAXPY", top_k=4)
        top2_names = {r.chunk.metadata.subroutine_name for r in reranked[:2]}
        assert "SAXPY" in top2_names
        assert "DAXPY" in top2_names

    def test_boost_used_for_ordering_not_display(self):
        """Boost affects sort order but original score is preserved for display."""
        from app.retrieval.search import rerank_results
        results = [_make_result("DGEMV", 0.55), _make_result("DGEMM", 0.50)]
        reranked = rerank_results(results, "What does DGEMM do?", top_k=2)
        # DGEMM (0.50 * 2.0 = 1.0 sort score) should sort first
        assert reranked[0].chunk.metadata.subroutine_name == "DGEMM"
        # But displayed score stays at the original 0.50, not boosted 1.0
        assert reranked[0].score == pytest.approx(0.50)
        assert reranked[1].score == pytest.approx(0.55)

    def test_preserves_chunk_data(self):
        """Re-ranking should not corrupt chunk metadata."""
        from app.retrieval.search import rerank_results
        results = [_make_result("DGEMM", 0.80)]
        reranked = rerank_results(results, "DGEMM", top_k=1)
        assert reranked[0].chunk.metadata.file_path == "SRC/dgemm.f"
        assert reranked[0].chunk.metadata.blas_level == "3"
        assert reranked[0].chunk.id == "test::DGEMM"

    def test_empty_results(self):
        """Empty list should return empty list."""
        from app.retrieval.search import rerank_results
        assert rerank_results([], "DGEMM", top_k=5) == []

    def test_performance_under_10ms(self):
        """Re-ranking 10 results should take < 10ms."""
        from app.retrieval.search import rerank_results
        results = [_make_result(f"DGEMM{i}", 0.9 - i * 0.05) for i in range(10)]
        start = time.time()
        for _ in range(100):
            rerank_results(results, "What does DGEMM do?", top_k=5)
        elapsed_ms = (time.time() - start) * 1000 / 100
        assert elapsed_ms < 10, f"Re-ranking took {elapsed_ms:.1f}ms (target: <10ms)"


# ── filter_by_score_gap ──


class TestFilterByScoreGap:
    def test_filters_low_scoring_results(self):
        """Results far below top score should be removed."""
        from app.retrieval.search import filter_by_score_gap
        results = [
            _make_result("DGEMM", 0.99),
            _make_result("DGEMV", 0.47),
            _make_result("SGEMM", 0.43),
        ]
        filtered = filter_by_score_gap(results)
        names = [r.chunk.metadata.subroutine_name for r in filtered]
        assert "DGEMM" in names
        assert "DGEMV" not in names
        assert "SGEMM" not in names

    def test_keeps_close_scores(self):
        """Results close to top score should be kept."""
        from app.retrieval.search import filter_by_score_gap
        results = [
            _make_result("DGEMM", 0.90),
            _make_result("DGEMV", 0.85),
            _make_result("SGEMM", 0.80),
        ]
        filtered = filter_by_score_gap(results)
        assert len(filtered) == 3

    def test_empty_results(self):
        """Empty list should return empty."""
        from app.retrieval.search import filter_by_score_gap
        assert filter_by_score_gap([]) == []

    def test_single_result_kept(self):
        """A single result should always be kept."""
        from app.retrieval.search import filter_by_score_gap
        results = [_make_result("DGEMM", 0.50)]
        assert len(filter_by_score_gap(results)) == 1

    def test_custom_ratio(self):
        """Custom ratio adjusts the threshold."""
        from app.retrieval.search import filter_by_score_gap
        results = [
            _make_result("DGEMM", 1.0),
            _make_result("DGEMV", 0.75),
        ]
        # With strict ratio=0.8, DGEMV (0.75 < 1.0*0.8=0.8) gets filtered
        strict = filter_by_score_gap(results, ratio=0.8)
        assert len(strict) == 1
        # With loose ratio=0.5, DGEMV (0.75 >= 1.0*0.5=0.5) survives
        loose = filter_by_score_gap(results, ratio=0.5)
        assert len(loose) == 2


# ── detect_query_metadata_filters ──


class TestDetectQueryMetadataFilters:
    def test_level_3_detected(self):
        """'Level 3' in query should produce blas_level filter."""
        from app.retrieval.search import detect_query_metadata_filters
        result = detect_query_metadata_filters("Find similar patterns across Level 3 routines")
        assert result == {"blas_level": {"$eq": "3"}}

    def test_level_1_detected(self):
        from app.retrieval.search import detect_query_metadata_filters
        result = detect_query_metadata_filters("Show Level 1 operations")
        assert result == {"blas_level": {"$eq": "1"}}

    def test_level_2_detected(self):
        from app.retrieval.search import detect_query_metadata_filters
        result = detect_query_metadata_filters("Explain level 2 matrix-vector routines")
        assert result == {"blas_level": {"$eq": "2"}}

    def test_no_level_returns_none(self):
        """Queries without a BLAS level should return None."""
        from app.retrieval.search import detect_query_metadata_filters
        assert detect_query_metadata_filters("What does DGEMM do?") is None

    def test_case_insensitive(self):
        from app.retrieval.search import detect_query_metadata_filters
        result = detect_query_metadata_filters("LEVEL 3 routines")
        assert result == {"blas_level": {"$eq": "3"}}

    def test_level_without_number_returns_none(self):
        """'level' alone without 1/2/3 should not trigger filtering."""
        from app.retrieval.search import detect_query_metadata_filters
        assert detect_query_metadata_filters("What level of optimization?") is None
