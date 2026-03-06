"""API routes for LegacyLens query endpoint with SSE streaming support."""

import asyncio
import json
import logging
import time
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse
from app.models import QueryRequest, QueryResponse
from app.retrieval.search import search_codebase, build_context
from app.retrieval.generator import generate_answer, generate_answer_stream
from app.retrieval.query_classifier import classify_query, get_search_params, QueryType
from app.session import store as session_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "legacylens"}


def format_sse_event(event: str, data: dict | str) -> str:
    """Format a Server-Sent Event string.

    SSE spec: each event is `event: <type>\\ndata: <json>\\n\\n`
    """
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def _resolve_session(session_id: str | None) -> tuple[str, list[dict]]:
    """Resolve or create a session, returning (session_id, history)."""
    if session_id:
        history = session_store.get_messages(session_id)
        if history is not None:
            return session_id, history
    new_id = session_store.create_session()
    return new_id, []


# ── Response handlers ──


async def _handle_sync(
    request: QueryRequest,
    session_id: str,
    query_type: QueryType,
    results: list,
    start: float,
) -> QueryResponse | JSONResponse:
    """Generate answer synchronously and return a JSON response."""
    try:
        answer = ""
        if results:
            context = build_context(results)
            answer = await asyncio.to_thread(
                generate_answer, request.query, context, query_type,
            )

        total_ms = (time.time() - start) * 1000

        if answer:
            session_store.add_turn(session_id, request.query, answer)

        return QueryResponse(
            answer=answer,
            sources=results,
            query_type=query_type.value,
            query_time_ms=total_ms,
            session_id=session_id,
        )
    except Exception as e:
        logger.error("Generation failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Generation failed: {e}", "status": "error"},
        )


def _handle_stream(
    request: QueryRequest,
    session_id: str,
    query_type: QueryType,
    results: list,
    search_time_ms: float,
    start: float,
) -> StreamingResponse:
    """Stream answer tokens via Server-Sent Events."""
    context = build_context(results) if results else ""
    sources_data = [r.model_dump() for r in results]

    async def event_generator():
        # 1. Send sources + metadata first
        yield format_sse_event("sources", {
            "query_type": query_type.value,
            "search_time_ms": round(search_time_ms, 1),
            "sources": sources_data,
            "session_id": session_id,
        })

        # 2. Stream answer tokens, collecting for session storage
        full_answer = []
        try:
            if context:
                gen = generate_answer_stream(request.query, context, query_type)
                # Use a sentinel instead of catching StopIteration because
                # PEP 479 converts it into RuntimeError inside async generators.
                _sentinel = object()
                loop = asyncio.get_event_loop()
                gen_iter = iter(gen)
                while True:
                    chunk = await loop.run_in_executor(
                        None, lambda: next(gen_iter, _sentinel)
                    )
                    if chunk is _sentinel:
                        break
                    full_answer.append(chunk)
                    yield format_sse_event("token", {"content": chunk})
        except Exception as e:
            logger.error("Streaming generation failed: %s", e)
            yield format_sse_event("error", {"message": str(e)})

        # Store the turn in session memory
        answer_text = "".join(full_answer)
        if answer_text:
            session_store.add_turn(session_id, request.query, answer_text)

        # 3. Send done event
        total_ms = (time.time() - start) * 1000
        yield format_sse_event("done", {"total_time_ms": round(total_ms, 1)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Main query endpoint ──


@router.post("/query")
async def query(
    request: QueryRequest,
    stream: bool = Query(default=False),
    session_id: str | None = Query(default=None),
):
    """Classify the query, search the codebase, and generate an answer."""
    start = time.time()

    session_id, _history = _resolve_session(session_id)
    query_type = classify_query(request.query)
    search_params = get_search_params(query_type)

    try:
        results, search_time_ms = await asyncio.to_thread(
            search_codebase, request.query, search_params["top_k"]
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        return JSONResponse(
            status_code=503,
            content={"error": f"Search service unavailable: {e}", "status": "error"},
        )

    if stream:
        return _handle_stream(request, session_id, query_type, results, search_time_ms, start)
    return await _handle_sync(request, session_id, query_type, results, start)
