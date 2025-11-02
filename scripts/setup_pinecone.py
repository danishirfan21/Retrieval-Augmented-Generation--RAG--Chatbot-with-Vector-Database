"""
Setup script for Pinecone index
Creates the index if it doesn't exist
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.core.embeddings import EmbeddingGenerator
from app.services.pinecone_service import PineconeService
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_pinecone():
    """Setup Pinecone index"""
    try:
        # Load settings
        settings = get_settings()
        logger.info("Settings loaded successfully")

        # Initialize embedding generator to get dimension
        logger.info("Initializing embedding generator...")
        embedding_gen = EmbeddingGenerator(settings.embedding_model)
        dimension = embedding_gen.get_dimension()
        logger.info(f"Embedding dimension: {dimension}")

        # Initialize Pinecone service
        logger.info("Initializing Pinecone service...")
        pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index_name=settings.pinecone_index_name
        )

        # Create index
        logger.info(f"Creating index '{settings.pinecone_index_name}'...")
        pinecone_service.create_index(dimension=dimension, metric="cosine")

        # Verify index creation
        stats = pinecone_service.get_index_stats()
        logger.info(f"Index created successfully!")
        logger.info(f"Index stats: {stats}")

        print("\n" + "="*80)
        print("✓ Pinecone setup completed successfully!")
        print("="*80)
        print(f"Index Name: {settings.pinecone_index_name}")
        print(f"Dimension: {dimension}")
        print(f"Metric: cosine")
        print("\nNext steps:")
        print("  1. Run 'python scripts/ingest_documents.py' to add documents")
        print("  2. Start the API server with 'uvicorn app.main:app --reload'")
        print("="*80)

    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise


if __name__ == "__main__":
    setup_pinecone()
