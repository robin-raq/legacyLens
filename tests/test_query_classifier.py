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
        assert classify_query("What patterns are used in error handling?") == QueryType.PATTERN

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

    def test_all_types_have_system_prompt(self):
        """Every query type must have a system prompt."""
        for qt in QueryType:
            params = get_search_params(qt)
            assert "system_prompt" in params
            assert len(params["system_prompt"]) > 50
