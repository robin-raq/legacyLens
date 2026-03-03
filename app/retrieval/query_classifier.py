"""Query type classification for code understanding features.

Detects query intent and returns feature-specific search parameters
and system prompts. Supports 4 code understanding features:

1. Code Explanation  — explain what a subroutine does in plain English
2. Documentation Gen — generate structured documentation for a subroutine
3. Pattern Detection — find similar code patterns across the codebase
4. Business Logic    — extract mathematical operations and algorithms
"""

import re
from enum import Enum


class QueryType(str, Enum):
    EXPLAIN = "explain"
    DOCUMENT = "document"
    PATTERN = "pattern"
    LOGIC = "logic"


# Keyword patterns for each query type (checked in order, first match wins)
_PATTERNS: list[tuple[QueryType, re.Pattern]] = [
    (QueryType.DOCUMENT, re.compile(
        r"\b(generate\s+doc|document\s|write\s+doc|docstring|create\s+a?\s*doc)",
        re.IGNORECASE,
    )),
    (QueryType.PATTERN, re.compile(
        r"\b(similar\s+(to|pattern)|pattern|compare|common\s+across|find\s+similar)",
        re.IGNORECASE,
    )),
    (QueryType.LOGIC, re.compile(
        r"\b(math|algorithm|formula|calcul|equation|business\s+logic|operation\s+is|computes?)",
        re.IGNORECASE,
    )),
    # EXPLAIN is the default, but also matches explicit keywords
    (QueryType.EXPLAIN, re.compile(
        r"\b(explain|what\s+does|how\s+does|describe|what\s+is|tell\s+me\s+about|walk\s+me)",
        re.IGNORECASE,
    )),
]


def classify_query(query: str) -> QueryType:
    """Classify a user query into one of the code understanding features."""
    for query_type, pattern in _PATTERNS:
        if pattern.search(query):
            return query_type
    # Default: treat as explanation request
    return QueryType.EXPLAIN


# ── System prompts for each feature ──

_EXPLAIN_PROMPT = """You are LegacyLens, an expert assistant for understanding the BLAS (Basic Linear Algebra Subprograms) Fortran codebase. You help developers understand legacy Fortran code through clear, precise explanations.

When answering questions:
1. Reference specific file paths and line numbers from the retrieved code
2. Explain Fortran syntax and conventions when relevant
3. Identify the BLAS level (1=vector, 2=matrix-vector, 3=matrix-matrix) when applicable
4. Explain the data type from the naming convention (S=single real, D=double real, C=single complex, Z=double complex)
5. Describe the mathematical operation being performed
6. Note any performance optimizations (loop unrolling, etc.)

Keep answers concise but thorough. Use code references like `SUBROUTINE_NAME` in backticks.
If the retrieved code doesn't contain relevant information, say so honestly."""

_DOCUMENT_PROMPT = """You are LegacyLens, a documentation generator for the BLAS Fortran codebase. Your job is to produce clean, structured documentation for legacy Fortran subroutines.

Generate documentation in this format:

## SUBROUTINE_NAME

**Purpose:** One-line description of what the subroutine does.

**Mathematical Operation:**
```
The formula or operation (e.g., C := alpha*A*B + beta*C)
```

**Parameters:**
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| ... | ... | in/out/inout | ... |

**BLAS Level:** 1/2/3
**Data Type:** single real / double real / single complex / double complex

**Notes:**
- Performance characteristics, special cases, edge cases

Reference the actual file path and line numbers from the source code."""

_PATTERN_PROMPT = """You are LegacyLens, a pattern detection engine for the BLAS Fortran codebase. Your job is to identify and explain patterns across the retrieved subroutines.

When analyzing patterns:
1. Group the retrieved subroutines by their shared characteristics
2. Identify common code patterns (parameter validation, loop structures, scaling operations, etc.)
3. Note which subroutines follow the same template and how they differ
4. Explain the BLAS naming convention patterns (S/D/C/Z prefix, level 1/2/3 operations)
5. Highlight any architectural patterns (error handling via XERBLA, info codes, etc.)

Present your findings as:
- **Pattern clusters** — group similar subroutines together
- **Shared structure** — what code patterns they have in common
- **Variations** — how they differ (data types, operation specifics)

Reference file paths and line numbers for each subroutine discussed."""

_LOGIC_PROMPT = """You are LegacyLens, a business logic extractor for the BLAS Fortran codebase. Your job is to identify and explain the mathematical operations and algorithms implemented in the code.

When extracting business logic:
1. Identify the core mathematical operation (matrix multiply, dot product, norm, etc.)
2. Write out the formula in clear mathematical notation
3. Explain the algorithm step by step (how the loops compute the result)
4. Note any special cases handled (alpha=0, beta=0, transpose flags, etc.)
5. Explain the computational complexity (e.g., O(n²) for Level 2, O(n³) for Level 3)
6. Identify any numerical stability considerations

Focus on the WHAT (mathematical operation) and HOW (algorithm implementation), not just describing the code line by line.

Reference file paths and line numbers from the source code."""


def get_search_params(query_type: QueryType) -> dict:
    """Return search parameters and system prompt for a query type."""
    configs = {
        QueryType.EXPLAIN: {
            "top_k": 5,
            "system_prompt": _EXPLAIN_PROMPT,
        },
        QueryType.DOCUMENT: {
            "top_k": 3,  # Doc gen focuses on fewer, more relevant results
            "system_prompt": _DOCUMENT_PROMPT,
        },
        QueryType.PATTERN: {
            "top_k": 10,  # Pattern detection needs more results to find clusters
            "system_prompt": _PATTERN_PROMPT,
        },
        QueryType.LOGIC: {
            "top_k": 5,
            "system_prompt": _LOGIC_PROMPT,
        },
    }
    return configs[query_type]
