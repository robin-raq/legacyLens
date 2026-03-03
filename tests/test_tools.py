"""Tests for agent tool definitions and executors — TDD RED phase."""

import pytest
from unittest.mock import patch, MagicMock
from app.models import CodeChunk, ChunkMetadata, SearchResult


# ── Tool Definition Tests ──


class TestToolDefinitions:
    def test_returns_list(self):
        from app.agent.tools import TOOL_DEFINITIONS

        assert isinstance(TOOL_DEFINITIONS, list)

    def test_has_three_tools(self):
        from app.agent.tools import TOOL_DEFINITIONS

        assert len(TOOL_DEFINITIONS) == 3

    def test_each_has_required_fields(self):
        from app.agent.tools import TOOL_DEFINITIONS

        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_tool_names_are_unique(self):
        from app.agent.tools import TOOL_DEFINITIONS

        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_expected_tool_names(self):
        from app.agent.tools import TOOL_DEFINITIONS

        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert names == {"search_codebase", "get_routine_info", "list_routines_by_level"}


# ── search_codebase executor ──


class TestExecuteSearchCodebase:
    def test_returns_results(self, mock_search):
        from app.agent.tools import execute_search_codebase

        result = execute_search_codebase("matrix multiply")
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["subroutine_name"] == "DGEMM"

    def test_results_have_expected_fields(self, mock_search):
        from app.agent.tools import execute_search_codebase

        result = execute_search_codebase("DGEMM")
        item = result["results"][0]
        assert "subroutine_name" in item
        assert "file_path" in item
        assert "blas_level" in item
        assert "data_type" in item
        assert "score" in item
        assert "code_snippet" in item

    def test_empty_query_returns_error(self):
        from app.agent.tools import execute_search_codebase

        result = execute_search_codebase("")
        assert "error" in result

    def test_top_k_clamped(self, mock_search):
        from app.agent.tools import execute_search_codebase

        execute_search_codebase("test", top_k=100)
        # Should have been clamped to 15
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[1]["top_k"] <= 15

    def test_exception_returns_error(self):
        from app.agent.tools import execute_search_codebase

        with patch("app.agent.tools.search_codebase", side_effect=Exception("Pinecone down")):
            result = execute_search_codebase("test")
            assert "error" in result


# ── get_routine_info executor ──


class TestExecuteGetRoutineInfo:
    def test_known_routine_returns_info(self, mock_search):
        from app.agent.tools import execute_get_routine_info

        result = execute_get_routine_info("DGEMM")
        assert result["found"] is True
        assert result["routine_name"] == "DGEMM"
        assert "source_code" in result

    def test_unknown_routine_returns_not_found(self, mock_search):
        """Domain verification — validates routine exists."""
        from app.agent.tools import execute_get_routine_info

        # Override mock to return non-matching results
        mock_search.return_value = (
            [
                SearchResult(
                    chunk=CodeChunk(
                        id="test::DGEMM",
                        text="SUBROUTINE DGEMM",
                        metadata=ChunkMetadata(
                            file_path="SRC/dgemm.f",
                            start_line=1,
                            end_line=50,
                            subroutine_name="DGEMM",
                            blas_level="3",
                            data_type="double real",
                        ),
                    ),
                    score=0.5,
                )
            ],
            10.0,
        )
        result = execute_get_routine_info("XYZFOO")
        assert result["found"] is False
        assert "not found" in result["error"].lower()

    def test_case_insensitive(self, mock_search):
        from app.agent.tools import execute_get_routine_info

        result = execute_get_routine_info("dgemm")
        assert result["found"] is True

    def test_empty_name_returns_error(self):
        from app.agent.tools import execute_get_routine_info

        result = execute_get_routine_info("")
        assert "error" in result

    def test_exception_returns_error(self):
        from app.agent.tools import execute_get_routine_info

        with patch("app.agent.tools.search_codebase", side_effect=Exception("fail")):
            result = execute_get_routine_info("DGEMM")
            assert "error" in result


# ── list_routines_by_level executor ──


class TestExecuteListRoutines:
    def test_returns_routines_list(self, mock_search):
        from app.agent.tools import execute_list_routines_by_level

        result = execute_list_routines_by_level(blas_level="3")
        assert "routines" in result
        assert isinstance(result["routines"], list)

    def test_filters_by_level(self, mock_search):
        from app.agent.tools import execute_list_routines_by_level

        # Mock returns level 3, so filtering by level 1 should return empty
        result = execute_list_routines_by_level(blas_level="1")
        assert result["total"] == 0

    def test_matching_level_returns_results(self, mock_search):
        from app.agent.tools import execute_list_routines_by_level

        result = execute_list_routines_by_level(blas_level="3")
        assert result["total"] >= 1

    def test_no_filters_returns_all(self, mock_search):
        from app.agent.tools import execute_list_routines_by_level

        result = execute_list_routines_by_level()
        assert result["total"] >= 1

    def test_exception_returns_error(self):
        from app.agent.tools import execute_list_routines_by_level

        with patch("app.agent.tools.search_codebase", side_effect=Exception("fail")):
            result = execute_list_routines_by_level(blas_level="3")
            assert "error" in result


# ── Dispatcher ──


class TestDispatchTool:
    def test_dispatch_search(self, mock_search):
        from app.agent.tools import dispatch_tool

        result = dispatch_tool("search_codebase", {"query": "DGEMM"})
        assert "results" in result

    def test_dispatch_routine_info(self, mock_search):
        from app.agent.tools import dispatch_tool

        result = dispatch_tool("get_routine_info", {"routine_name": "DGEMM"})
        assert "found" in result

    def test_dispatch_list(self, mock_search):
        from app.agent.tools import dispatch_tool

        result = dispatch_tool("list_routines_by_level", {"blas_level": "3"})
        assert "routines" in result

    def test_dispatch_unknown_returns_error(self):
        from app.agent.tools import dispatch_tool

        result = dispatch_tool("nonexistent_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]
