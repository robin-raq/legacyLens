"""Tests for query type classification and feature-specific prompts."""

import pytest
from app.retrieval.query_classifier import classify_query, QueryType, get_search_params


class TestClassifyQuery:
    """Test that queries are routed to the correct feature."""

    # ── Code Explanation ──
    def test_explain_keyword(self):
        assert classify_query("Explain what DGEMM does") == QueryType.EXPLAIN

    def test_what_does_keyword(self):
        assert classify_query("What does SSWAP do?") == QueryType.EXPLAIN

    def test_how_does_keyword(self):
        assert classify_query("How does ZGEMV work?") == QueryType.EXPLAIN

    def test_describe_keyword(self):
        assert classify_query("Describe the CTRMM subroutine") == QueryType.EXPLAIN

    # ── Documentation Generation ──
    def test_generate_docs_keyword(self):
        assert classify_query("Generate docs for DSWAP") == QueryType.DOCUMENT

    def test_document_keyword(self):
        assert classify_query("Document the SAXPY function") == QueryType.DOCUMENT

    def test_write_documentation_keyword(self):
        assert classify_query("Write documentation for DGEMM") == QueryType.DOCUMENT

    def test_docstring_keyword(self):
        assert classify_query("Create a docstring for ZCOPY") == QueryType.DOCUMENT

    # ── Pattern Detection ──
    def test_find_similar_keyword(self):
        assert classify_query("Find similar patterns to DGEMM") == QueryType.PATTERN

    def test_patterns_keyword(self):
        assert classify_query("What patterns are common across Level 3 routines?") == QueryType.PATTERN

    def test_similar_to_keyword(self):
        assert classify_query("Show subroutines similar to DSWAP") == QueryType.PATTERN

    def test_compare_keyword(self):
        assert classify_query("Compare SGEMM and DGEMM") == QueryType.PATTERN

    # ── Business Logic Extraction ──
    def test_math_keyword(self):
        assert classify_query("What math does DTRSM compute?") == QueryType.LOGIC

    def test_algorithm_keyword(self):
        assert classify_query("What algorithm does DGEMV use?") == QueryType.LOGIC

    def test_formula_keyword(self):
        assert classify_query("What is the formula in DSYRK?") == QueryType.LOGIC

    def test_calculation_keyword(self):
        assert classify_query("What calculation does DDOT perform?") == QueryType.LOGIC

    # ── Dependency Mapping ──
    def test_dependency_keyword(self):
        assert classify_query("What are the dependencies of DGEMM?") == QueryType.DEPENDENCY

    def test_what_calls_keyword(self):
        assert classify_query("What does DGEMM call?") == QueryType.DEPENDENCY

    def test_call_graph_keyword(self):
        assert classify_query("Show the call graph for DTRSM") == QueryType.DEPENDENCY

    def test_external_keyword(self):
        assert classify_query("What EXTERNAL declarations does DGEMV have?") == QueryType.DEPENDENCY

    # ── Impact Analysis ──
    def test_impact_keyword(self):
        assert classify_query("What is the impact of changing XERBLA?") == QueryType.IMPACT

    def test_affected_keyword(self):
        assert classify_query("What routines are affected if DSCAL changes?") == QueryType.IMPACT

    def test_what_would_break_keyword(self):
        assert classify_query("What would break if LSAME is modified?") == QueryType.IMPACT

    # ── Translation Hints ──
    def test_modern_keyword(self):
        assert classify_query("What is the modern equivalent of DGEMM?") == QueryType.TRANSLATION

    def test_translate_keyword(self):
        assert classify_query("How would I translate SAXPY to Python?") == QueryType.TRANSLATION

    def test_numpy_keyword(self):
        assert classify_query("What NumPy function replaces DGEMV?") == QueryType.TRANSLATION

    def test_migrate_keyword(self):
        assert classify_query("How to migrate DTRSM to modern code?") == QueryType.TRANSLATION

    # ── Bug Pattern Search ──
    def test_bug_keyword(self):
        assert classify_query("Are there bugs in DGEMM?") == QueryType.BUG_PATTERN

    def test_potential_problem_keyword(self):
        assert classify_query("Find potential problems in DGEMV") == QueryType.BUG_PATTERN

    def test_validation_keyword(self):
        assert classify_query("Does DTRSM validate its inputs?") == QueryType.BUG_PATTERN

    def test_xerbla_keyword(self):
        assert classify_query("Which routines use XERBLA for error handling?") == QueryType.BUG_PATTERN

    # ── Default (general search) ──
    def test_general_query_defaults_to_explain(self):
        assert classify_query("DGEMM") == QueryType.EXPLAIN

    def test_simple_question(self):
        assert classify_query("matrix multiplication") == QueryType.EXPLAIN

    # ── Case insensitivity ──
    def test_case_insensitive(self):
        assert classify_query("EXPLAIN what dgemm does") == QueryType.EXPLAIN
        assert classify_query("generate DOCS for dswap") == QueryType.DOCUMENT


class TestGetSearchParams:
    """Test that each query type gets appropriate search parameters."""

    def test_explain_uses_top5(self):
        params = get_search_params(QueryType.EXPLAIN)
        assert params["top_k"] == 5

    def test_pattern_uses_top10(self):
        """Pattern detection needs more results to find clusters."""
        params = get_search_params(QueryType.PATTERN)
        assert params["top_k"] == 10

    def test_document_uses_top3(self):
        """Doc gen focuses on one subroutine, fewer results needed."""
        params = get_search_params(QueryType.DOCUMENT)
        assert params["top_k"] == 3

    def test_logic_uses_top5(self):
        params = get_search_params(QueryType.LOGIC)
        assert params["top_k"] == 5

    def test_dependency_uses_top5(self):
        params = get_search_params(QueryType.DEPENDENCY)
        assert params["top_k"] == 5

    def test_impact_uses_top8(self):
        params = get_search_params(QueryType.IMPACT)
        assert params["top_k"] == 8

    def test_translation_uses_top5(self):
        params = get_search_params(QueryType.TRANSLATION)
        assert params["top_k"] == 5

    def test_bug_pattern_uses_top8(self):
        params = get_search_params(QueryType.BUG_PATTERN)
        assert params["top_k"] == 8

    def test_all_types_have_system_prompt(self):
        """Every query type must have a system prompt."""
        for qt in QueryType:
            params = get_search_params(qt)
            assert "system_prompt" in params
            assert len(params["system_prompt"]) > 50
