"""
Custom retriever for Pinecone vector store
Implements retrieval logic for RAG pipeline
"""
from typing import List, Dict, Any
import logging
from app.core.embeddings import EmbeddingGenerator
from app.services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)


class PineconeRetriever:
    """Custom retriever that fetches relevant documents from Pinecone"""

    def __init__(
        self,
        pinecone_service: PineconeService,
        embedding_generator: EmbeddingGenerator,
        top_k: int = 5
    ):
        """
        Initialize the retriever

        Args:
            pinecone_service: Pinecone service instance
            embedding_generator: Embedding generator instance
            top_k: Number of documents to retrieve
        """
        self.pinecone_service = pinecone_service
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        logger.info(f"Retriever initialized with top_k={top_k}")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides default)

        Returns:
            List of retrieved documents with metadata and scores
        """
        k = top_k if top_k is not None else self.top_k

        logger.info(f"Retrieving top {k} documents for query: {query[:50]}...")

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)

        # Query Pinecone
        results = self.pinecone_service.query(
            query_vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        # Format results
        retrieved_docs = []
        for match in results:
            doc = {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source": match.metadata.get("source", "unknown"),
                "metadata": match.metadata
            }
            retrieved_docs.append(doc)

        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

    def format_docs_for_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "unknown")
            text = doc.get("text", "")
            score = doc.get("score", 0.0)

            context_parts.append(
                f"[Document {i}] (Source: {source}, Relevance: {score:.3f})\n{text}\n"
            )

        return "\n".join(context_parts)
