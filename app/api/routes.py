import asyncio
import json
import time
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.models import QueryRequest, QueryResponse
from app.retrieval.search import search_codebase, build_context
from app.retrieval.generator import generate_answer, generate_answer_stream
from app.retrieval.query_classifier import classify_query, get_search_params

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "legacylens"}


def format_sse_event(event: str, data: dict | str) -> str:
    """Format a Server-Sent Event string.

    SSE spec: each event is `event: <type>\\ndata: <json>\\n\\n`
    """
    payload = json.dumps(data) if isinstance(data, dict) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/query")
async def query(request: QueryRequest, stream: bool = Query(default=False)):
    start = time.time()

    # Classify the query intent (CPU-only, fast — no thread needed)
    query_type = classify_query(request.query)
    search_params = get_search_params(query_type)

    # Run blocking I/O in thread pool to avoid blocking the event loop
    results, search_time_ms = await asyncio.to_thread(
        search_codebase, request.query, search_params["top_k"]
    )

    if not stream:
        # ── Non-streaming path (backward compatible) ──
        answer = ""
        if results:
            context = build_context(results)
            answer = await asyncio.to_thread(
                generate_answer, request.query, context, query_type
            )

        total_ms = (time.time() - start) * 1000

        return QueryResponse(
            answer=answer,
            sources=results,
            query_type=query_type.value,
            query_time_ms=total_ms,
        )

    # ── Streaming path ──
    context = build_context(results) if results else ""

    # Serialise sources for SSE (Pydantic models → dicts)
    sources_data = [r.model_dump() for r in results]

    async def event_generator():
        # 1. Send sources + metadata first
        yield format_sse_event("sources", {
            "query_type": query_type.value,
            "search_time_ms": round(search_time_ms, 1),
            "sources": sources_data,
        })

        # 2. Stream answer tokens
        if context:
            gen = generate_answer_stream(request.query, context, query_type)
            # Run the blocking generator in a thread, yielding chunks.
            # NOTE: We use a sentinel instead of catching StopIteration
            # because PEP 479 converts StopIteration inside async generators
            # into RuntimeError, silently killing the generator.
            _sentinel = object()
            loop = asyncio.get_event_loop()
            gen_iter = iter(gen)
            while True:
                chunk = await loop.run_in_executor(
                    None, lambda: next(gen_iter, _sentinel)
                )
                if chunk is _sentinel:
                    break
                yield format_sse_event("token", {"content": chunk})

        # 3. Send done event
        total_ms = (time.time() - start) * 1000
        yield format_sse_event("done", {"total_time_ms": round(total_ms, 1)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
