"""Tests for the agentic loop — TDD RED phase."""

import pytest
from unittest.mock import patch, MagicMock


class TestAgentEndTurn:
    """Tests for when the model returns a direct answer (no tool calls)."""

    def test_returns_answer_text(self, mock_anthropic_end_turn):
        from app.agent.agent import run_agent

        messages = []
        answer, sources, tool_calls = run_agent("What is DGEMM?", messages)
        assert "DGEMM" in answer
        assert "matrix multiplication" in answer.lower()

    def test_returns_empty_sources_when_no_tools(self, mock_anthropic_end_turn):
        from app.agent.agent import run_agent

        messages = []
        answer, sources, tool_calls = run_agent("Hello", messages)
        assert sources == []

    def test_returns_empty_tool_calls_when_no_tools(self, mock_anthropic_end_turn):
        from app.agent.agent import run_agent

        messages = []
        answer, sources, tool_calls = run_agent("Hello", messages)
        assert tool_calls == []

    def test_appends_to_messages(self, mock_anthropic_end_turn):
        from app.agent.agent import run_agent

        messages = []
        run_agent("What is DGEMM?", messages)
        # Should have user message + assistant response
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


class TestAgentToolUse:
    """Tests for when the model calls tools before answering."""

    def test_tool_call_is_executed(self, mock_anthropic_tool_then_answer, mock_search):
        from app.agent.agent import run_agent

        messages = []
        answer, sources, tool_calls = run_agent("Tell me about DGEMM", messages)
        assert "matrix" in answer.lower()
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "search_codebase"

    def test_sources_collected_from_tool(self, mock_anthropic_tool_then_answer, mock_search):
        from app.agent.agent import run_agent

        messages = []
        answer, sources, tool_calls = run_agent("Tell me about DGEMM", messages)
        assert len(sources) >= 1
        assert sources[0].chunk.metadata.subroutine_name == "DGEMM"

    def test_messages_include_tool_results(self, mock_anthropic_tool_then_answer, mock_search):
        from app.agent.agent import run_agent

        messages = []
        run_agent("Tell me about DGEMM", messages)
        # user → assistant(tool_use) → user(tool_result) → assistant(end_turn)
        assert len(messages) == 4
        # The tool_result message
        tool_result_msg = messages[2]
        assert tool_result_msg["role"] == "user"
        assert any(
            item.get("type") == "tool_result"
            for item in tool_result_msg["content"]
        )


class TestAgentMaxIterations:
    def test_stops_at_max_iterations(self):
        """Agent stops looping after MAX_ITERATIONS."""
        from app.agent.agent import run_agent

        with patch("app.agent.agent._client") as mock_client:
            # Always return tool_use — should hit max iterations
            tool_response = MagicMock()
            tool_response.stop_reason = "tool_use"
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.id = "toolu_loop"
            tool_block.name = "search_codebase"
            tool_block.input = {"query": "test"}
            tool_response.content = [tool_block]
            mock_client.messages.create.return_value = tool_response

            with patch("app.agent.agent.dispatch_tool", return_value={"results": []}):
                messages = []
                answer, sources, tool_calls = run_agent("test", messages)
                # Should have given up
                assert "maximum" in answer.lower() or len(tool_calls) > 0
                # Should not loop infinitely
                assert mock_client.messages.create.call_count <= 10


class TestAgentErrorHandling:
    def test_api_timeout_returns_message(self):
        from app.agent.agent import run_agent
        import anthropic

        with patch("app.agent.agent._client") as mock_client:
            mock_client.messages.create.side_effect = anthropic.APITimeoutError(
                request=MagicMock()
            )
            messages = []
            answer, sources, tool_calls = run_agent("test", messages)
            assert "timed out" in answer.lower()

    def test_api_error_returns_message(self):
        from app.agent.agent import run_agent
        import anthropic

        with patch("app.agent.agent._client") as mock_client:
            mock_client.messages.create.side_effect = anthropic.APIError(
                message="Server error",
                request=MagicMock(),
                body=None,
            )
            messages = []
            answer, sources, tool_calls = run_agent("test", messages)
            assert "error" in answer.lower()

    def test_tool_error_does_not_crash(self, mock_search):
        """If a tool throws, the error is fed back to the model."""
        from app.agent.agent import run_agent

        with patch("app.agent.agent._client") as mock_client:
            # First call: tool_use
            tool_response = MagicMock()
            tool_response.stop_reason = "tool_use"
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.id = "toolu_err"
            tool_block.name = "search_codebase"
            tool_block.input = {"query": "test"}
            tool_response.content = [tool_block]

            # Second call: end_turn
            final_response = MagicMock()
            final_response.stop_reason = "end_turn"
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "I encountered an error but recovered."
            final_response.content = [text_block]

            mock_client.messages.create.side_effect = [tool_response, final_response]

            with patch(
                "app.agent.agent.dispatch_tool",
                return_value={"error": "Pinecone is down"},
            ):
                messages = []
                answer, sources, tool_calls = run_agent("test", messages)
                # Should not crash, should return the recovery answer
                assert "recovered" in answer.lower() or answer != ""


class TestAgentConversationHistory:
    def test_existing_messages_sent_to_model(self, mock_anthropic_end_turn):
        from app.agent.agent import run_agent

        messages = [
            {"role": "user", "content": "What is BLAS?"},
            {"role": "assistant", "content": [MagicMock(type="text", text="BLAS is...")]},
        ]
        run_agent("Tell me more", messages)
        # Model should receive the full history
        call_args = mock_anthropic_end_turn.messages.create.call_args
        sent_messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        # History (2) + new user message (1) = 3 before response
        assert len(sent_messages) >= 3
