import dotenv
import os

# Load .env first so it fills in any missing env vars
dotenv.load_dotenv(override=True)

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str = ""
    google_api_key: str = ""
    pinecone_api_key: str
    pinecone_index_name: str = "legacylens"
    llm_provider: str = "gemini"  # "gemini" or "anthropic"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    gemini_model: str = "gemini-2.5-flash"
    port: int = 3000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
