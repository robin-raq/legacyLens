"""Tests for data models."""

import pytest
from app.models import CodeChunk, ChunkMetadata, SearchResult, QueryRequest, QueryResponse


class TestChunkMetadata:
    def test_defaults(self):
        m = ChunkMetadata(file_path="test.f", start_line=1, end_line=10)
        assert m.subroutine_name == ""
        assert m.blas_level == "unknown"
        assert m.data_type == "unknown"
        assert m.description == ""
        assert m.line_count == 0

    def test_full_metadata(self):
        m = ChunkMetadata(
            file_path="SRC/dgemm.f",
            start_line=213,
            end_line=262,
            subroutine_name="DGEMM",
            blas_level="3",
            data_type="double real",
            description="matrix multiplication",
            line_count=50,
        )
        assert m.file_path == "SRC/dgemm.f"
        assert m.subroutine_name == "DGEMM"


class TestCodeChunk:
    def test_creation(self):
        chunk = CodeChunk(
            id="test::FOO",
            text="SUBROUTINE FOO",
            metadata=ChunkMetadata(file_path="test.f", start_line=1, end_line=5),
        )
        assert chunk.id == "test::FOO"
        assert "FOO" in chunk.text


class TestQueryRequest:
    def test_query_field(self):
        req = QueryRequest(query="What does DGEMM do?")
        assert req.query == "What does DGEMM do?"


class TestQueryResponse:
    def test_creation(self):
        resp = QueryResponse(answer="test", sources=[], query_time_ms=100.0)
        assert resp.answer == "test"
        assert resp.sources == []
        assert resp.query_time_ms == 100.0
