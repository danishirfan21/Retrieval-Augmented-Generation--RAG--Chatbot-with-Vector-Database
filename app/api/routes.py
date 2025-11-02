"""
API routes for RAG chatbot
Defines endpoints for querying and health checks
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from app.rag.chain import RAGChain
from app.rag.retriever import PineconeRetriever
from app.core.config import get_settings
from app.core.embeddings import EmbeddingGenerator
from app.services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances (initialized in main.py)
rag_chain: Optional[RAGChain] = None


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str = Field(..., description="The question to ask", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    question: str
    answer: str
    sources: List[str]
    retrieved_docs: List[Dict[str, Any]]


class ChatMessage(BaseModel):
    """Chat message model"""
    question: str
    answer: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint with history"""
    question: str = Field(..., description="The question to ask", min_length=1)
    chat_history: List[ChatMessage] = Field(default=[], description="Previous chat messages")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    service: str
    version: str


class IndexStatsResponse(BaseModel):
    """Response model for index statistics"""
    total_vector_count: int
    dimension: int
    index_fullness: float


# Initialize RAG components
def initialize_rag_chain():
    """Initialize RAG chain and dependencies"""
    global rag_chain

    try:
        settings = get_settings()
        logger.info("Initializing RAG components...")

        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator(settings.embedding_model)

        # Initialize Pinecone service
        pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index_name=settings.pinecone_index_name
        )

        # Connect to existing index
        pinecone_service.get_index()

        # Initialize retriever
        retriever = PineconeRetriever(
            pinecone_service=pinecone_service,
            embedding_generator=embedding_gen,
            top_k=settings.top_k
        )

        # Initialize RAG chain
        rag_chain = RAGChain(
            retriever=retriever,
            openai_api_key=settings.openai_api_key,
            model_name=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )

        logger.info("RAG chain initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}")
        raise


# Endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question

    Args:
        request: Query request with question and optional top_k

    Returns:
        Answer with sources and retrieved documents
    """
    try:
        if rag_chain is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG chain not initialized"
            )

        logger.info(f"Received query: {request.question[:50]}...")

        result = rag_chain.invoke(
            question=request.question,
            top_k=request.top_k
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/chat", response_model=QueryResponse)
async def chat(request: ChatRequest):
    """
    Chat with the RAG system with conversation history

    Args:
        request: Chat request with question, history, and optional top_k

    Returns:
        Answer with sources and retrieved documents
    """
    try:
        if rag_chain is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG chain not initialized"
            )

        logger.info(f"Received chat query: {request.question[:50]}...")

        # Convert chat history to dict format
        history = [
            {"question": msg.question, "answer": msg.answer}
            for msg in request.chat_history
        ]

        result = rag_chain.invoke_with_chat_history(
            question=request.question,
            chat_history=history,
            top_k=request.top_k
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {str(e)}"
        )


@router.get("/stats", response_model=IndexStatsResponse)
async def get_stats():
    """Get statistics about the vector index"""
    try:
        settings = get_settings()

        pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index_name=settings.pinecone_index_name
        )

        stats = pinecone_service.get_index_stats()

        return IndexStatsResponse(
            total_vector_count=stats.get("total_vector_count", 0),
            dimension=stats.get("dimension", 0),
            index_fullness=stats.get("index_fullness", 0.0)
        )

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )
