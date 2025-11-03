"""
API routes for RAG chatbot
Defines endpoints for querying and health checks
"""
from fastapi import APIRouter, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import shutil
import tempfile
import os

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

        # Pick the best available index (most vectors) among common dimension suffixes, then align embeddings
        base_index_name = settings.pinecone_index_name
        candidates = [
            (base_index_name, None),
            (f"{base_index_name}-3072", 3072),
            (f"{base_index_name}-1536", 1536),
            (f"{base_index_name}-1024", 1024),
            (f"{base_index_name}-768", 768),
            (f"{base_index_name}-384", 384),
        ]
        best_name = None
        best_dim = None
        best_count = -1
        # Probe candidates
        for name, dim_hint in candidates:
            svc = PineconeService(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment,
                index_name=name,
            )
            try:
                if not svc.index_exists():
                    continue
                stats = svc.get_index_stats()
                count = int(stats.get("total_vector_count", 0))
                # If base (no suffix) and no vectors, deprioritize
                if count > best_count:
                    best_name = name
                    # Try to get accurate dimension; fall back to hint
                    try:
                        best_dim = svc.get_index_dimension()
                    except Exception:
                        best_dim = dim_hint
                    best_count = count
            except Exception:
                continue

        # Decide target index and embedding configuration
        if best_name and best_count > 0 and best_dim:
            index_name_to_use = best_name
            target_dim = best_dim
            # Map dimension to model/provider
            dim_to_model = {
                3072: ("openai", "text-embedding-3-large"),
                1536: ("openai", "text-embedding-3-small"),
                1024: ("huggingface", "BAAI/bge-large-en-v1.5"),
                768: ("huggingface", "sentence-transformers/all-mpnet-base-v2"),
                384: ("huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
            }
            provider, model_name = dim_to_model.get(target_dim, (settings.embedding_provider, settings.embedding_model))
            embedding_gen = EmbeddingGenerator(model_name)
            # Force provider if needed by temporarily overriding via attribute
            if hasattr(embedding_gen, "_provider"):
                embedding_gen._provider = provider  # type: ignore[attr-defined]
            pinecone_service = PineconeService(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment,
                index_name=index_name_to_use,
            )
            # Ensure index connectivity (no-op if exists)
            try:
                pinecone_service.create_index(dimension=target_dim, metric="cosine")
            except Exception:
                pinecone_service.get_index()
            logger.info(f"Using index '{index_name_to_use}' (dim={target_dim}, vectors={best_count})")
        else:
            # No populated index found; fall back to configured model/provider and base index
            embedding_gen = EmbeddingGenerator(settings.embedding_model)
            desired_dim = embedding_gen.get_dimension()
            pinecone_service = PineconeService(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment,
                index_name=base_index_name,
            )
            pinecone_service.create_index(dimension=desired_dim, metric="cosine")

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
        # Prefer the index used by the running RAG chain if available
        if rag_chain is not None:
            stats = rag_chain.retriever.pinecone_service.get_index_stats()
        else:
            settings = get_settings()
            # Align stats to embedding dimension similar to initialization
            embedding_gen = EmbeddingGenerator(settings.embedding_model)
            base_index_name = settings.pinecone_index_name
            service = PineconeService(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment,
                index_name=base_index_name,
            )
            try:
                desired_dim = embedding_gen.get_dimension()
                if service.index_exists():
                    try:
                        existing_dim = service.get_index_dimension()
                    except Exception:
                        existing_dim = desired_dim
                    if existing_dim != desired_dim:
                        service = PineconeService(
                            api_key=settings.pinecone_api_key,
                            environment=settings.pinecone_environment,
                            index_name=f"{base_index_name}-{desired_dim}",
                        )
            except Exception:
                pass
            stats = service.get_index_stats()

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


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload one or more files for ingestion."""
    settings = get_settings()
    temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    try:
        saved = []
        for f in files:
            dest = os.path.join(temp_dir, f.filename)
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(dest)
        # Call ingestion script logic directly
        from scripts.ingest_documents import ingest_documents
        ingest_documents(temp_dir)
        return JSONResponse({"success": True, "files": [os.path.basename(x) for x in saved]})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})
    finally:
        # Optionally: clean up temp_dir after ingestion
        pass
