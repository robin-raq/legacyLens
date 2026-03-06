import dotenv

# Load .env first so it fills in any missing env vars
dotenv.load_dotenv(override=True)

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API keys
    openai_api_key: str
    anthropic_api_key: str = ""
    google_api_key: str = ""
    pinecone_api_key: str
    pinecone_index_name: str = "legacylens"

    # LLM provider & models
    llm_provider: str = "gemini"  # "gemini" or "anthropic"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    gemini_model: str = "gemini-2.5-flash"

    # LLM generation
    max_tokens: int = 2000
    max_retries: int = 3
    retry_delay: float = 4.0
    llm_timeout: float = 60.0

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_timeout: float = 30.0
    embedding_batch_size: int = 100

    # Search tuning
    score_threshold: float = 0.2
    score_gap_ratio: float = 0.6
    exact_match_boost: float = 2.0
    context_max_chars: int = 30000

    # Pinecone
    pinecone_batch_size: int = 100
    pinecone_metadata_max_chars: int = 10000

    # Validation
    max_query_length: int = 2000

    # Session
    session_ttl: int = 3600
    max_sessions: int = 200
    max_messages_per_session: int = 50

    # Server
    port: int = 3000

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
