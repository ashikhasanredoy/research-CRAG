import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # App Information
    PROJECT_NAME: str = "ResearchCRAG"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Ollama Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "llama3.2")
    DEFAULT_EMBED_MODEL: str = os.getenv("DEFAULT_EMBED_MODEL", "nomic-embed-text")
    LLM_TEMPERATURE: float = 0.1

    # Storage Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    PAPERS_DIR: Path = BASE_DIR / "data" / "papers"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "chroma_db"

    # Ingestion & Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval & Reranking
    RETRIEVAL_TOP_K: int = 6
    RERANKER_TOP_K: int = 4
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    USE_RERANKER: bool = False
    SIMILARITY_THRESHOLD: float = 0.35  # Threshold for grader

    # CRAG Control
    CONFIDENCE_THRESHOLD: float = 0.012  # Threshold for High Confidence Fast Path
    MAX_QUERY_REWRITES: int = 2
    ENABLE_WEB_SEARCH_FALLBACK: bool = True
    STRICT_VERIFICATION: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
