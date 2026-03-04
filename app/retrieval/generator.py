"""Answer generation using Claude or Gemini with feature-specific system prompts."""

from __future__ import annotations

import time
from typing import Iterator

from app.config import settings
from app.retrieval.query_classifier import QueryType, get_search_params

# ── Lazy client initialisation ──
# Clients are created on first use so missing API keys for the unused
# provider don't crash the app at import time.

_anthropic_client = None
_gemini_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=60.0
        )
    return _anthropic_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=settings.google_api_key)
    return _gemini_client


# ── User message templates per feature ──

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

    QueryType.DEPENDENCY: """Analyze the dependencies and call structure of:

**Request:** {query}

**Retrieved Code:**
{context}

Extract CALL statements, EXTERNAL declarations, and explain the data flow between routines.""",

    QueryType.IMPACT: """Analyze the potential impact of changes to:

**Request:** {query}

**Retrieved Code:**
{context}

Identify callers and callees. Explain upstream and downstream impact if this code changes.""",

    QueryType.TRANSLATION: """Suggest modern equivalents for:

**Request:** {query}

**Retrieved Code:**
{context}

Explain the Fortran code, then map it to modern languages (NumPy/SciPy, Eigen). Note Fortran idioms.""",

    QueryType.BUG_PATTERN: """Analyze the following code for potential issues:

**Request:** {query}

**Retrieved Code:**
{context}

Look for: off-by-one errors, missing validation, uninitialized vars, overflow risks. Reference file:line.""",
}


# ── Generation backends ──


def _generate_anthropic(system_prompt: str, user_message: str) -> str:
    """Generate answer via Anthropic Claude."""
    client = _get_anthropic_client()
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


_MAX_RETRIES = 3
_RETRY_DELAY = 4.0  # seconds, matches Gemini's retry hint


def _generate_gemini(system_prompt: str, user_message: str) -> str:
    """Generate answer via Google Gemini, with retry on rate-limit errors."""
    client = _get_gemini_client()
    from google.genai import types
    from google.genai.errors import ClientError

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=2000,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text
        except ClientError as e:
            if "429" in str(e) and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
                continue
            raise


def _stream_anthropic(system_prompt: str, user_message: str) -> Iterator[str]:
    """Stream answer via Anthropic Claude."""
    client = _get_anthropic_client()
    with client.messages.stream(
        model=settings.anthropic_model,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        yield from stream.text_stream


def _stream_gemini(system_prompt: str, user_message: str) -> Iterator[str]:
    """Stream answer via Google Gemini, with retry on rate-limit errors."""
    client = _get_gemini_client()
    from google.genai import types
    from google.genai.errors import ClientError

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content_stream(
                model=settings.gemini_model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=2000,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return  # Success
        except ClientError as e:
            if "429" in str(e) and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
                continue
            raise


# ── Public API ──


def _build_prompt(query: str, context: str, query_type: QueryType) -> tuple[str, str]:
    """Build system prompt and user message for a given query type."""
    params = get_search_params(query_type)
    system_prompt = params["system_prompt"]
    user_template = _USER_TEMPLATES[query_type]
    user_message = user_template.format(query=query, context=context)
    return system_prompt, user_message


def generate_answer(
    query: str, context: str, query_type: QueryType = QueryType.EXPLAIN
) -> str:
    """Generate an answer using the configured LLM provider."""
    system_prompt, user_message = _build_prompt(query, context, query_type)

    if settings.llm_provider == "gemini":
        return _generate_gemini(system_prompt, user_message)
    return _generate_anthropic(system_prompt, user_message)


def generate_answer_stream(
    query: str, context: str, query_type: QueryType = QueryType.EXPLAIN
) -> Iterator[str]:
    """Stream an answer using the configured LLM provider.

    Yields text deltas as they arrive from the API.
    """
    system_prompt, user_message = _build_prompt(query, context, query_type)

    if settings.llm_provider == "gemini":
        yield from _stream_gemini(system_prompt, user_message)
    else:
        yield from _stream_anthropic(system_prompt, user_message)
