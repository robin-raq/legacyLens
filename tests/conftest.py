"""Shared test fixtures for LegacyLens test suite."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """FastAPI test client."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def mock_search():
    """Mock search_codebase to avoid Pinecone/OpenAI calls in tool tests."""
    with patch("app.agent.tools.search_codebase") as mock:
        from app.models import CodeChunk, ChunkMetadata, SearchResult

        mock.return_value = (
            [
                SearchResult(
                    chunk=CodeChunk(
                        id="dgemm.f::DGEMM",
                        text="SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)\n*  Purpose: DGEMM performs C := alpha*op(A)*op(B) + beta*C",
                        metadata=ChunkMetadata(
                            file_path="SRC/dgemm.f",
                            start_line=1,
                            end_line=50,
                            subroutine_name="DGEMM",
                            blas_level="3",
                            data_type="double real",
                            description="matrix-matrix multiply",
                            line_count=50,
                        ),
                    ),
                    score=0.85,
                )
            ],
            15.0,
        )
        yield mock


@pytest.fixture
def mock_anthropic_end_turn():
    """Mock Anthropic client returning a direct text answer (no tool calls)."""
    with patch("app.agent.agent._client") as mock:
        response = MagicMock()
        response.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "DGEMM performs double-precision matrix multiplication: C := alpha*A*B + beta*C"
        response.content = [text_block]
        mock.messages.create.return_value = response
        yield mock


@pytest.fixture
def mock_anthropic_tool_then_answer():
    """Mock Anthropic client: first call returns tool_use, second returns end_turn."""
    with patch("app.agent.agent._client") as mock:
        # First response: tool_use
        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me search for that."
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_123"
        tool_block.name = "search_codebase"
        tool_block.input = {"query": "DGEMM matrix multiply", "top_k": 5}
        tool_response.content = [text_block, tool_block]

        # Second response: end_turn
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_text = MagicMock()
        final_text.type = "text"
        final_text.text = "DGEMM performs double-precision matrix-matrix multiplication."
        final_response.content = [final_text]

        mock.messages.create.side_effect = [tool_response, final_response]
        yield mock
