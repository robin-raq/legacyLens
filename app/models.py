from pydantic import BaseModel, Field, field_validator


class ChunkMetadata(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    subroutine_name: str = ""
    blas_level: str = "unknown"
    data_type: str = "unknown"
    description: str = ""
    line_count: int = 0


class CodeChunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata


class SearchResult(BaseModel):
    chunk: CodeChunk
    score: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or whitespace-only")
        return v


class QueryResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    query_type: str = "explain"
    query_time_ms: float
    session_id: str = ""


