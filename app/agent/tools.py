"""Tool definitions and executors for the LegacyLens agent."""

import re
import json
from app.retrieval.search import search_codebase

# ── Anthropic tool_use definitions ──

TOOL_DEFINITIONS = [
    {
        "name": "search_codebase",
        "description": (
            "Semantic search across the BLAS Fortran codebase. "
            "Use this to find subroutines related to a concept, operation, or keyword. "
            "Returns matching routines with relevance scores, code snippets, and metadata "
            "(BLAS level, data type, file path, line numbers)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (e.g., 'matrix multiplication', 'vector swap')",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 15)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_routine_info",
        "description": (
            "Look up a specific BLAS routine by its exact name (e.g., 'DGEMM', 'SAXPY'). "
            "Returns the full source code, metadata, BLAS level, data type, and description. "
            "Use this when the user asks about a specific named routine. "
            "Returns an error if the routine does not exist in the BLAS codebase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "routine_name": {
                    "type": "string",
                    "description": "Exact BLAS routine name (e.g., 'DGEMM', 'SSWAP', 'ZGEMV')",
                },
            },
            "required": ["routine_name"],
        },
    },
    {
        "name": "list_routines_by_level",
        "description": (
            "List BLAS routines filtered by level and/or data type. "
            "BLAS Level 1 = vector operations, Level 2 = matrix-vector, Level 3 = matrix-matrix. "
            "Data types: 'single real', 'double real', 'single complex', 'double complex'. "
            "Returns routine names with brief descriptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "blas_level": {
                    "type": "string",
                    "description": "BLAS level: '1', '2', or '3'",
                    "enum": ["1", "2", "3"],
                },
                "data_type": {
                    "type": "string",
                    "description": "Data type filter",
                    "enum": [
                        "single real",
                        "double real",
                        "single complex",
                        "double complex",
                    ],
                },
            },
            "required": [],
        },
    },
    {
        "name": "analyze_dependencies",
        "description": (
            "Analyze what other routines a BLAS subroutine calls (dependency mapping). "
            "Parses CALL statements and EXTERNAL declarations from the Fortran source. "
            "Use this for dependency mapping, impact analysis, and understanding call relationships. "
            "Returns the list of routines called and external references."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "routine_name": {
                    "type": "string",
                    "description": "The BLAS routine name to analyze (e.g., 'DGEMM')",
                },
            },
            "required": ["routine_name"],
        },
    },
]


# ── Executors ──


def execute_search_codebase(query: str, top_k: int = 5) -> dict:
    """Semantic search across BLAS routines."""
    if not query or not query.strip():
        return {"error": "Query must be a non-empty string."}
    top_k = max(1, min(top_k, 15))

    try:
        results, search_time_ms = search_codebase(query, top_k=top_k)
        return {
            "results": [
                {
                    "subroutine_name": r.chunk.metadata.subroutine_name,
                    "file_path": r.chunk.metadata.file_path,
                    "lines": f"{r.chunk.metadata.start_line}-{r.chunk.metadata.end_line}",
                    "blas_level": r.chunk.metadata.blas_level,
                    "data_type": r.chunk.metadata.data_type,
                    "description": r.chunk.metadata.description[:200],
                    "score": round(r.score, 4),
                    "code_snippet": r.chunk.text[:2000],
                }
                for r in results
            ],
            "total_results": len(results),
            "search_time_ms": round(search_time_ms, 1),
        }
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


def execute_get_routine_info(routine_name: str) -> dict:
    """Look up a specific routine by name. Validates the routine exists (domain verification)."""
    name = routine_name.strip().upper()
    if not name:
        return {"error": "Routine name must be non-empty."}

    try:
        results, _ = search_codebase(name, top_k=10)

        # Find exact match by subroutine name
        for r in results:
            if r.chunk.metadata.subroutine_name.upper() == name:
                return {
                    "found": True,
                    "routine_name": r.chunk.metadata.subroutine_name,
                    "file_path": r.chunk.metadata.file_path,
                    "lines": f"{r.chunk.metadata.start_line}-{r.chunk.metadata.end_line}",
                    "blas_level": r.chunk.metadata.blas_level,
                    "data_type": r.chunk.metadata.data_type,
                    "description": r.chunk.metadata.description,
                    "line_count": r.chunk.metadata.line_count,
                    "source_code": r.chunk.text,
                }

        # Domain verification: routine does not exist
        return {
            "found": False,
            "error": f"Routine '{name}' not found in the BLAS codebase.",
            "suggestion": "Use search_codebase to find routines by description.",
        }
    except Exception as e:
        return {"error": f"Lookup failed: {str(e)}"}


def execute_list_routines_by_level(
    blas_level: str | None = None, data_type: str | None = None
) -> dict:
    """List routines filtered by BLAS level and/or data type."""
    try:
        # Build a contextual query for the level
        level_queries = {
            "1": "BLAS Level 1 vector operations",
            "2": "BLAS Level 2 matrix-vector operations",
            "3": "BLAS Level 3 matrix-matrix operations",
        }
        query_text = level_queries.get(blas_level, "BLAS routines")
        if data_type:
            query_text += f" {data_type}"

        results, _ = search_codebase(query_text, top_k=50)

        # Post-filter by requested criteria
        filtered = []
        seen = set()
        for r in results:
            m = r.chunk.metadata
            if blas_level and m.blas_level != blas_level:
                continue
            if data_type and m.data_type != data_type:
                continue
            if m.subroutine_name in seen:
                continue
            seen.add(m.subroutine_name)
            filtered.append(
                {
                    "routine_name": m.subroutine_name,
                    "blas_level": m.blas_level,
                    "data_type": m.data_type,
                    "description": m.description[:150],
                    "file_path": m.file_path,
                }
            )

        return {
            "routines": filtered,
            "total": len(filtered),
            "filters_applied": {
                "blas_level": blas_level,
                "data_type": data_type,
            },
        }
    except Exception as e:
        return {"error": f"List failed: {str(e)}"}


def parse_fortran_calls(source_code: str) -> dict:
    """Parse CALL statements and EXTERNAL declarations from Fortran source.

    Returns dict with 'calls' (list of routine names invoked via CALL)
    and 'external_refs' (list of names declared EXTERNAL).
    """
    calls: set[str] = set()
    externals: set[str] = set()

    for line in source_code.split("\n"):
        if not line.strip():
            continue

        # Fortran fixed-form: column 1 determines comments (C, c, *, !)
        first_char = line[0] if line else ""
        if first_char in ("C", "c", "*", "!"):
            continue

        # Find CALL ROUTINE_NAME(...)
        for match in re.finditer(r"\bCALL\s+(\w+)", line, re.IGNORECASE):
            calls.add(match.group(1).upper())

        # Find EXTERNAL ROUTINE1, ROUTINE2, ...
        ext_match = re.match(r"\s*EXTERNAL\s+(.+)", line.strip(), re.IGNORECASE)
        if ext_match:
            names = [n.strip().upper() for n in ext_match.group(1).split(",")]
            externals.update(n for n in names if n)

    return {
        "calls": sorted(calls),
        "external_refs": sorted(externals),
    }


def execute_analyze_dependencies(routine_name: str) -> dict:
    """Analyze dependencies for a BLAS routine by parsing CALL/EXTERNAL statements."""
    name = routine_name.strip().upper()
    if not name:
        return {"error": "Routine name must be non-empty."}

    try:
        info = execute_get_routine_info(name)
        if not info.get("found"):
            return info  # Pass through the not-found response

        source = info.get("source_code", "")
        deps = parse_fortran_calls(source)

        return {
            "routine_name": info["routine_name"],
            "file_path": info["file_path"],
            "blas_level": info["blas_level"],
            "calls": deps["calls"],
            "external_refs": deps["external_refs"],
            "total_dependencies": len(set(deps["calls"]) | set(deps["external_refs"])),
        }
    except Exception as e:
        return {"error": f"Dependency analysis failed: {str(e)}"}


# ── Dispatcher ──


def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Route a tool call to its executor. Returns error dict for unknown tools."""
    executors = {
        "search_codebase": lambda inp: execute_search_codebase(
            query=inp.get("query", ""),
            top_k=inp.get("top_k", 5),
        ),
        "get_routine_info": lambda inp: execute_get_routine_info(
            routine_name=inp.get("routine_name", ""),
        ),
        "list_routines_by_level": lambda inp: execute_list_routines_by_level(
            blas_level=inp.get("blas_level"),
            data_type=inp.get("data_type"),
        ),
        "analyze_dependencies": lambda inp: execute_analyze_dependencies(
            routine_name=inp.get("routine_name", ""),
        ),
    }

    executor = executors.get(tool_name)
    if executor is None:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        return executor(tool_input)
    except Exception as e:
        return {"error": f"Tool '{tool_name}' failed: {str(e)}"}
