from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # Local databases (auto-created, no install needed)
    sqlite_db_path: str = "./hubmicro.db"
    chroma_db_path: str = "./chroma_data"

    # Embedding model (downloads ~80 MB on first run, then cached)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 51   # ~10% of 512

    # RAG retrieval
    top_k_chunks: int = 5


settings = Settings()  # type: ignore[call-arg]
