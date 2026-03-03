import time
from fastapi import APIRouter
from app.models import QueryRequest, QueryResponse
from app.retrieval.search import search_codebase, build_context
from app.retrieval.generator import generate_answer
from app.retrieval.query_classifier import classify_query, get_search_params

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "legacylens"}


@router.post("/query")
async def query(request: QueryRequest):
    start = time.time()

    # Classify the query intent → picks the right feature
    query_type = classify_query(request.query)
    search_params = get_search_params(query_type)

    # Search with feature-specific top_k
    results, search_time_ms = search_codebase(
        request.query, top_k=search_params["top_k"]
    )

    # Build context and generate answer with feature-specific prompt
    answer = ""
    if results:
        context = build_context(results)
        answer = generate_answer(request.query, context, query_type)

    total_ms = (time.time() - start) * 1000

    return QueryResponse(
        answer=answer,
        sources=results,
        query_type=query_type.value,
        query_time_ms=total_ms,
    )
