"""
Embedding generation module using sentence-transformers
Generates vector embeddings for text documents and queries
"""
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles text embedding generation using sentence-transformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        logger.info(f"Generating embedding for query: {query[:50]}...")
        embedding = self.model.encode([query])[0]
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.dimension
