from pydantic import BaseModel


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
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    query_type: str = "explain"
    query_time_ms: float
