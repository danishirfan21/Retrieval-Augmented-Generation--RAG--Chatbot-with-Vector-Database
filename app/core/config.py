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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500

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
