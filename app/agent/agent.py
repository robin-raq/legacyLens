"""Agentic loop: Claude decides which tools to call and when."""

import json
import anthropic
from app.config import settings
from app.agent.tools import TOOL_DEFINITIONS, dispatch_tool
from app.models import SearchResult, CodeChunk, ChunkMetadata

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=90.0)

MAX_ITERATIONS = 8
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are LegacyLens, an expert assistant for understanding the BLAS (Basic Linear Algebra Subprograms) Fortran codebase. You help developers understand legacy Fortran code through clear, precise explanations.

You have access to tools that let you search and inspect the BLAS codebase:
- **search_codebase**: Semantic search for routines by concept or keyword
- **get_routine_info**: Look up a specific routine by name (e.g., DGEMM)
- **list_routines_by_level**: Browse routines by BLAS level (1/2/3) or data type

When answering questions:
1. Use tools to retrieve relevant source code before answering
2. Reference specific file paths and line numbers from retrieved code
3. Explain the BLAS level (1=vector, 2=matrix-vector, 3=matrix-matrix)
4. Explain data types from naming convention (S=single, D=double, C=complex single, Z=complex double)
5. Describe mathematical operations being performed
6. Note performance optimizations when present

If a user asks about a specific routine, use get_routine_info first to verify it exists.
If the retrieved code doesn't contain relevant information, say so honestly.
Keep answers concise but thorough. Use `ROUTINE_NAME` formatting for subroutine references."""


def run_agent(
    query: str, messages: list[dict]
) -> tuple[str, list[SearchResult], list[dict]]:
    """Run the agentic loop.

    Args:
        query: The user's current query.
        messages: Full conversation history (modified in place).

    Returns:
        (answer_text, sources, tool_calls_log)
    """
    messages.append({"role": "user", "content": query})

    sources: list[SearchResult] = []
    tool_calls_log: list[dict] = []

    for iteration in range(MAX_ITERATIONS):
        try:
            response = _client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
        except anthropic.APITimeoutError:
            return "Request timed out. Please try again.", sources, tool_calls_log
        except anthropic.APIError as e:
            return f"An error occurred: {str(e)}", sources, tool_calls_log

        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": response.content})
            answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer += block.text
            return answer, sources, tool_calls_log

        elif response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)

                    # Collect sources from search/lookup results
                    if block.name == "search_codebase" and "results" in result:
                        sources.extend(_extract_sources_from_search(result))
                    elif block.name == "get_routine_info" and result.get("found"):
                        sources.extend(_extract_sources_from_routine(result))

                    tool_calls_log.append(
                        {
                            "tool_name": block.name,
                            "tool_input": block.input,
                            "tool_result": result,
                        }
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

        else:
            messages.append({"role": "assistant", "content": response.content})
            return "Unexpected response from model.", sources, tool_calls_log

    return (
        "I reached the maximum number of tool calls. Here's what I found so far.",
        sources,
        tool_calls_log,
    )


def _extract_sources_from_search(result: dict) -> list[SearchResult]:
    """Convert search tool results back to SearchResult objects."""
    sources = []
    for r in result.get("results", []):
        lines = r.get("lines", "0-0").split("-")
        chunk = CodeChunk(
            id=f"{r['file_path']}::{r['subroutine_name']}",
            text=r.get("code_snippet", ""),
            metadata=ChunkMetadata(
                file_path=r["file_path"],
                start_line=int(lines[0]),
                end_line=int(lines[1]),
                subroutine_name=r["subroutine_name"],
                blas_level=r["blas_level"],
                data_type=r["data_type"],
                description=r.get("description", ""),
            ),
        )
        sources.append(SearchResult(chunk=chunk, score=r["score"]))
    return sources


def _extract_sources_from_routine(result: dict) -> list[SearchResult]:
    """Convert get_routine_info result to SearchResult."""
    lines = result.get("lines", "0-0").split("-")
    chunk = CodeChunk(
        id=f"{result['file_path']}::{result['routine_name']}",
        text=result.get("source_code", ""),
        metadata=ChunkMetadata(
            file_path=result["file_path"],
            start_line=int(lines[0]),
            end_line=int(lines[1]),
            subroutine_name=result["routine_name"],
            blas_level=result["blas_level"],
            data_type=result["data_type"],
            description=result.get("description", ""),
            line_count=result.get("line_count", 0),
        ),
    )
    return [SearchResult(chunk=chunk, score=1.0)]
