import asyncio
import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.models import QueryRequest, QueryResponse, ChatRequest, ChatResponse, ToolCall
from app.retrieval.search import search_codebase, build_context
from app.retrieval.generator import generate_answer
from app.retrieval.query_classifier import classify_query, get_search_params
from app.agent.session import store
from app.agent.agent import run_agent

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "legacylens"}


@router.post("/query")
async def query(request: QueryRequest):
    start = time.time()

    # Classify the query intent (CPU-only, fast — no thread needed)
    query_type = classify_query(request.query)
    search_params = get_search_params(query_type)

    # Run blocking I/O in thread pool to avoid blocking the event loop
    results, search_time_ms = await asyncio.to_thread(
        search_codebase, request.query, search_params["top_k"]
    )

    # Build context and generate answer (also blocking I/O)
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


@router.post("/chat")
async def chat(request: ChatRequest):
    start = time.time()

    # Session management: reuse or create
    session_id = request.session_id
    messages = None
    if session_id:
        messages = store.get_messages(session_id)

    if messages is None:
        session_id = store.create_session()
        messages = []

    try:
        # Run agentic loop in thread pool (blocking I/O)
        answer, sources, tool_calls_log = await asyncio.to_thread(
            run_agent, request.query, messages
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Agent error: {str(e)}"},
        )

    # Persist updated messages back to session
    store.set_messages(session_id, messages)

    total_ms = (time.time() - start) * 1000

    return ChatResponse(
        answer=answer,
        sources=sources,
        tool_calls=[
            ToolCall(
                tool_name=tc["tool_name"],
                tool_input=tc["tool_input"],
                tool_result=tc["tool_result"],
            )
            for tc in tool_calls_log
        ],
        session_id=session_id,
        query_time_ms=total_ms,
    )
