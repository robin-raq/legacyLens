import time
from fastapi import APIRouter
from app.models import QueryRequest, QueryResponse
from app.retrieval.search import search_codebase, build_context
from app.retrieval.generator import generate_answer

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "legacylens"}


@router.post("/query")
async def query(request: QueryRequest):
    start = time.time()

    # Search for relevant chunks
    results, search_time_ms = search_codebase(request.query, top_k=5)

    # Build context and generate answer
    answer = ""
    if results:
        context = build_context(results)
        answer = generate_answer(request.query, context)

    total_ms = (time.time() - start) * 1000

    return QueryResponse(
        answer=answer,
        sources=results,
        query_time_ms=total_ms,
    )
