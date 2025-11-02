"""
Document ingestion script
Processes documents, generates embeddings, and stores them in Pinecone
"""
import sys
import os
from pathlib import Path
import logging
from typing import List
import hashlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.core.embeddings import EmbeddingGenerator
from app.services.pinecone_service import PineconeService
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_documents(data_dir: str) -> List:
    """
    Load documents from a directory

    Args:
        data_dir: Path to directory containing documents

    Returns:
        List of loaded documents
    """
    logger.info(f"Loading documents from {data_dir}")

    # Load text files
    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    txt_docs = txt_loader.load()
    logger.info(f"Loaded {len(txt_docs)} text documents")

    # Load PDF files
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    pdf_docs = pdf_loader.load()
    logger.info(f"Loaded {len(pdf_docs)} PDF documents")

    all_docs = txt_docs + pdf_docs
    logger.info(f"Total documents loaded: {len(all_docs)}")

    return all_docs


def split_documents(documents: List, chunk_size: int, chunk_overlap: int) -> List:
    """
    Split documents into chunks

    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks
    """
    logger.info(f"Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    return chunks


def generate_doc_id(text: str, index: int) -> str:
    """
    Generate a unique ID for a document chunk

    Args:
        text: Text content
        index: Chunk index

    Returns:
        Unique document ID
    """
    hash_object = hashlib.md5(text.encode())
    return f"doc_{index}_{hash_object.hexdigest()[:8]}"


def ingest_documents(data_dir: str):
    """
    Main ingestion pipeline

    Args:
        data_dir: Path to directory containing documents
    """
    try:
        # Load settings
        settings = get_settings()

        # Initialize components
        logger.info("Initializing embedding generator...")
        embedding_gen = EmbeddingGenerator(settings.embedding_model)

        logger.info("Initializing Pinecone service...")
        pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index_name=settings.pinecone_index_name
        )

        # Create index if it doesn't exist
        pinecone_service.create_index(
            dimension=embedding_gen.get_dimension(),
            metric="cosine"
        )

        # Load and split documents
        documents = load_documents(data_dir)

        if not documents:
            logger.warning("No documents found to ingest")
            return

        chunks = split_documents(
            documents,
            settings.chunk_size,
            settings.chunk_overlap
        )

        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_gen.generate_embeddings(texts)

        # Prepare metadata
        metadata_list = []
        ids = []
        for i, chunk in enumerate(chunks):
            doc_id = generate_doc_id(chunk.page_content, i)
            ids.append(doc_id)

            metadata = {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_index": i
            }
            metadata_list.append(metadata)

        # Upsert to Pinecone
        logger.info("Uploading to Pinecone...")
        pinecone_service.upsert_vectors(
            vectors=embeddings,
            ids=ids,
            metadata=metadata_list
        )

        # Get index stats
        stats = pinecone_service.get_index_stats()
        logger.info(f"Index stats: {stats}")

        logger.info("Document ingestion completed successfully!")

    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise


if __name__ == "__main__":
    # Get data directory from command line or use default
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
    else:
        data_directory = str(Path(__file__).parent.parent / "data" / "sample_docs")

    logger.info(f"Starting document ingestion from: {data_directory}")
    ingest_documents(data_directory)
