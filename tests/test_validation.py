"""Tests for input validation and sanitization."""

import pytest
from pydantic import ValidationError
from app.models import QueryRequest


class TestQueryRequestValidation:
    def test_valid_query(self):
        req = QueryRequest(query="What does DGEMM do?")
        assert req.query == "What does DGEMM do?"

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="   ")

    def test_query_stripped(self):
        req = QueryRequest(query="  What does DGEMM do?  ")
        assert req.query == "What does DGEMM do?"

    def test_query_too_long_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="a" * 2001)

    def test_query_at_max_length_accepted(self):
        req = QueryRequest(query="a" * 2000)
        assert len(req.query) == 2000

    def test_single_char_accepted(self):
        req = QueryRequest(query="x")
        assert req.query == "x"

    def test_newlines_preserved(self):
        """Newlines in queries are valid (multi-line code questions)."""
        req = QueryRequest(query="Explain this:\nSUBROUTINE DGEMM")
        assert "\n" in req.query
