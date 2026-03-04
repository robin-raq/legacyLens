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

## Available Tools
- **search_codebase**: Semantic search for routines by concept or keyword
- **get_routine_info**: Look up a specific routine by name (e.g., DGEMM)
- **list_routines_by_level**: Browse routines by BLAS level (1/2/3) or data type
- **analyze_dependencies**: Show what a routine calls (CALL statements, EXTERNAL refs)

## Code Understanding Features
Detect the user's intent and respond using the appropriate feature mode:

### 1. Code Explanation (default)
Explain what a function or section does in plain English. Cover:
- Purpose and mathematical operation
- BLAS level (1=vector ops, 2=matrix-vector, 3=matrix-matrix)
- Data type from naming convention (S=single, D=double, C=complex single, Z=complex double)
- Key algorithm steps and loop structure
- Performance optimizations (loop unrolling, blocking, etc.)

### 2. Dependency Mapping
When asked about dependencies, call graphs, or "what calls what":
- Use `analyze_dependencies` to find CALL and EXTERNAL references
- Explain the role of each dependency (e.g., XERBLA for error handling, LSAME for character comparison)
- Describe the data flow: what parameters are passed between routines
- Show the call chain hierarchy

### 3. Pattern Detection
When asked to find similar code, compare routines, or identify patterns:
- Use `search_codebase` to find structurally similar routines
- Group results by shared characteristics (same operation across data types)
- Explain the BLAS naming convention patterns (S/D/C/Z prefix = same operation, different precision)
- Highlight common code structures: parameter validation via XERBLA, early-return optimizations, loop ordering

### 4. Impact Analysis
When asked "what would be affected if X changes" or about ripple effects:
- Use `analyze_dependencies` to find what the routine calls
- Use `search_codebase` to find routines that might call the target
- Explain upstream (callers) and downstream (callees) impact
- Identify shared patterns that would need consistent changes (e.g., all S/D/C/Z variants)

### 5. Documentation Generation
When asked to generate docs, document, or create documentation:
- Produce structured documentation with:
  - **Purpose**: One-line description
  - **Mathematical Operation**: Formula (e.g., C := alpha*op(A)*op(B) + beta*C)
  - **Parameters**: Table with Name, Type, Direction (in/out/inout), Description
  - **BLAS Level & Data Type**
  - **Notes**: Edge cases, performance characteristics
- Reference file path and line numbers

### 6. Translation Hints
When asked about modern equivalents, modernization, or migration:
- Explain what the Fortran code does, then suggest modern equivalents
- Map to NumPy/SciPy (Python), LAPACK bindings, Eigen (C++), or BLAS wrappers
- Note Fortran-specific idioms: column-major order, 1-based indexing, COMMON blocks, implicit typing
- Highlight what would change in a modern implementation (memory management, error handling, generics)

### 7. Bug Pattern Search
When asked about potential bugs, issues, or code quality:
- Look for common Fortran pitfalls: off-by-one errors, uninitialized variables, integer overflow in loop bounds
- Check parameter validation patterns (does it call XERBLA for all invalid inputs?)
- Identify potential numerical issues: division by zero guards, overflow/underflow handling
- Note missing IMPLICIT NONE declarations, unused variables, or suspicious control flow

### 8. Business Logic Extraction
When asked about business rules, algorithms, or computational logic:
- Extract the core mathematical formula and express it clearly
- Explain the algorithm step-by-step (not just what the code says, but the mathematical reasoning)
- Identify special cases and optimizations (alpha=0, beta=1, transpose flags)
- Explain computational complexity (O(n) for Level 1, O(n²) for Level 2, O(n³) for Level 3)
- Describe numerical stability considerations

## General Guidelines
1. ALWAYS use tools to retrieve relevant source code before answering
2. Reference specific file paths and line numbers in every answer
3. If a user asks about a specific routine, use `get_routine_info` first
4. If the retrieved code doesn't answer the question, say so honestly
5. Keep answers concise but thorough
6. Use `ROUTINE_NAME` backtick formatting for subroutine references"""


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


def run_agent_stream(query: str, messages: list[dict]):
    """Streaming version of run_agent. Yields SSE event dicts.

    Events yielded:
        {"event": "tool_start", "data": {"tool_name": ..., "tool_input": ...}}
        {"event": "tool_result", "data": {"tool_name": ..., "status": "ok"|"error"}}
        {"event": "text_delta", "data": {"chunk": "..."}}
        {"event": "sources", "data": [serialized SearchResult list]}
        {"event": "error", "data": {"message": "..."}}
    """
    messages.append({"role": "user", "content": query})

    sources: list[SearchResult] = []

    for iteration in range(MAX_ITERATIONS):
        try:
            # Use regular create for tool-use phases, stream for final text
            response = _client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
        except anthropic.APITimeoutError:
            yield {"event": "error", "data": {"message": "Request timed out."}}
            return
        except anthropic.APIError as e:
            yield {"event": "error", "data": {"message": str(e)}}
            return

        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": response.content})

            # Stream the text in small chunks for typewriter effect
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            # Yield text in chunks
            chunk_size = 12
            for i in range(0, len(text), chunk_size):
                yield {
                    "event": "text_delta",
                    "data": {"chunk": text[i : i + chunk_size]},
                }

            # Yield sources
            source_dicts = [s.model_dump() for s in sources]
            yield {"event": "sources", "data": source_dicts}
            return

        elif response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Notify frontend that tool is starting
                    yield {
                        "event": "tool_start",
                        "data": {
                            "tool_name": block.name,
                            "tool_input": block.input,
                        },
                    }

                    result = dispatch_tool(block.name, block.input)

                    # Collect sources
                    if block.name == "search_codebase" and "results" in result:
                        sources.extend(_extract_sources_from_search(result))
                    elif block.name == "get_routine_info" and result.get("found"):
                        sources.extend(_extract_sources_from_routine(result))

                    # Notify frontend of result
                    yield {
                        "event": "tool_result",
                        "data": {
                            "tool_name": block.name,
                            "status": "error" if "error" in result else "ok",
                        },
                    }

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
            yield {"event": "error", "data": {"message": "Unexpected response."}}
            return

    yield {
        "event": "error",
        "data": {"message": "Reached maximum tool calls."},
    }


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
