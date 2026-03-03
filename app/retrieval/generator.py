"""Answer generation using Claude with feature-specific system prompts."""

import anthropic
from app.config import settings
from app.retrieval.query_classifier import QueryType, get_search_params

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=60.0)

# User message templates per feature
_USER_TEMPLATES = {
    QueryType.EXPLAIN: """Based on the following retrieved BLAS source code, answer this question:

**Question:** {query}

**Retrieved Code:**
{context}

Provide a clear, detailed answer referencing the specific source files and line numbers.""",

    QueryType.DOCUMENT: """Generate structured documentation for the following BLAS subroutine(s):

**Request:** {query}

**Retrieved Code:**
{context}

Produce complete documentation in the format specified (Purpose, Parameters table, Notes).""",

    QueryType.PATTERN: """Analyze the following BLAS subroutines and identify patterns:

**Request:** {query}

**Retrieved Code:**
{context}

Group similar subroutines, identify shared patterns, and explain variations.""",

    QueryType.LOGIC: """Extract and explain the mathematical operations and algorithms from:

**Request:** {query}

**Retrieved Code:**
{context}

Focus on the mathematical formula, algorithm steps, special cases, and computational complexity.""",
}


def generate_answer(query: str, context: str, query_type: QueryType = QueryType.EXPLAIN) -> str:
    """Generate an answer using Claude with a feature-specific system prompt."""
    params = get_search_params(query_type)
    system_prompt = params["system_prompt"]
    user_template = _USER_TEMPLATES[query_type]
    user_message = user_template.format(query=query, context=context)

    response = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text
