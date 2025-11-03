"""
FastAPI application for RAG Financial Chatbot
Main entry point for the API service
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from app.core.config import get_settings
from app.api.routes import router, initialize_rag_chain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-based chatbot for answering financial questions using vector database retrieval",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve a minimal static frontend at /ui (and static assets under /static)
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except Exception:
    # If directory doesn't exist yet, ignore mounting error during module import; it will be available at runtime
    pass


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting up RAG Financial Chatbot API...")
    try:
        initialize_rag_chain()
        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG Financial Chatbot API...")


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG Chatbot"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to RAG Financial Chatbot API",
        "version": settings.app_version,
        "docs": "/docs"
    }


@app.get("/ui")
async def ui():
    """Serve the minimal chat UI"""
    return FileResponse("app/static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )
