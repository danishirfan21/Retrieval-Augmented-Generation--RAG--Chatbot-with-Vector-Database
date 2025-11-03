"""
Configuration module for RAG Chatbot
Handles environment variables and application settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # OpenAI Configuration
    openai_api_key: str

    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str = "gcp-starter"
    pinecone_index_name: str = "financial-rag-chatbot"

    # Model Configuration
    # Default embedding model is set to 1024-dim to align with common Pinecone index configs
    # Change via EMBEDDING_MODEL in .env if needed
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    # Preferred provider for embeddings: 'huggingface' or 'openai'
    embedding_provider: str = "huggingface"
    # If True, automatically fall back to OpenAI embeddings when HF API fails (e.g., 401)
    allow_fallback_to_openai: bool = True
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500

    # Hugging Face Inference API (for embeddings via API)
    huggingface_api_token: str | None = None

    # Retrieval Configuration
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Application Info
    app_name: str = "RAG Financial Chatbot"
    app_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
