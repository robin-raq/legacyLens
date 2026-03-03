"""Answer generation using Claude."""

import anthropic
from app.config import settings

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """You are LegacyLens, an expert assistant for understanding the BLAS (Basic Linear Algebra Subprograms) Fortran codebase. You help developers understand legacy Fortran code through clear, precise explanations.

When answering questions:
1. Reference specific file paths and line numbers from the retrieved code
2. Explain Fortran syntax and conventions when relevant
3. Identify the BLAS level (1=vector, 2=matrix-vector, 3=matrix-matrix) when applicable
4. Explain the data type from the naming convention (S=single real, D=double real, C=single complex, Z=double complex)
5. Describe the mathematical operation being performed
6. Note any performance optimizations (loop unrolling, etc.)

Keep answers concise but thorough. Use code references like `SUBROUTINE_NAME` in backticks.
If the retrieved code doesn't contain relevant information, say so honestly."""


def generate_answer(query: str, context: str) -> str:
    """Generate an answer using Claude with retrieved context."""
    user_message = f"""Based on the following retrieved BLAS source code, answer this question:

**Question:** {query}

**Retrieved Code:**
{context}

Provide a clear, detailed answer referencing the specific source files and line numbers."""

    response = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text
