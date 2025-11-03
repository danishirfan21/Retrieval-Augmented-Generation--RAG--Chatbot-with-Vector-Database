"""
Pinecone vector database service
Handles connection, indexing, and querying operations
"""
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for managing Pinecone vector database operations"""

    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initialize Pinecone service

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the index to use
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)
        self.index = None

        logger.info(f"Pinecone service initialized for index: {index_name}")

    def create_index(self, dimension: int, metric: str = "cosine"):
        """
        Create a new Pinecone index

        Args:
            dimension: Dimension of the vectors
            metric: Distance metric (cosine, euclidean, or dotproduct)
        """
        try:
            # Check if index already exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name in existing_indexes:
                logger.info(f"Index '{self.index_name}' already exists")
            else:
                logger.info(f"Creating index '{self.index_name}' with dimension {dimension}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                time.sleep(1)
                logger.info(f"Index '{self.index_name}' created successfully")

            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index '{self.index_name}'")

        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise

    def index_exists(self) -> bool:
        """Return True if the index exists in Pinecone."""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            return self.index_name in existing_indexes
        except Exception as e:
            logger.error(f"Error checking index existence: {str(e)}")
            raise

    def get_index_dimension(self) -> int:
        """Return the dimension of the existing index, if available."""
        try:
            # Describe index to get metadata including dimension
            desc = self.pc.describe_index(self.index_name)
            # desc may be a dict-like object depending on SDK version
            if hasattr(desc, "dimension"):
                return desc.dimension  # type: ignore[attr-defined]
            if isinstance(desc, dict) and "dimension" in desc:
                return int(desc["dimension"])  # type: ignore[index]

            # Fallback: connect and try stats (not guaranteed to include dimension)
            idx = self.pc.Index(self.index_name)
            stats = idx.describe_index_stats()
            dim = stats.get("dimension") if isinstance(stats, dict) else None
            if dim is not None:
                return int(dim)

            raise RuntimeError("Unable to determine index dimension from Pinecone API response")
        except Exception as e:
            logger.error(f"Error getting index dimension: {str(e)}")
            raise

    def get_index(self):
        """Get the current index instance"""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)
        return self.index

    def upsert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]]
    ):
        """
        Upsert vectors into the index

        Args:
            vectors: List of embedding vectors
            ids: List of unique IDs for each vector
            metadata: List of metadata dictionaries for each vector
        """
        try:
            if self.index is None:
                self.index = self.get_index()

            # Prepare data for upsert
            data = []
            for i, (vec_id, vec, meta) in enumerate(zip(ids, vectors, metadata)):
                data.append({
                    "id": vec_id,
                    "values": vec,
                    "metadata": meta
                })

            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors)")

            logger.info(f"Successfully upserted {len(vectors)} vectors")

        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Dict[str, Any] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the index for similar vectors

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            include_metadata: Whether to include metadata in results

        Returns:
            List of matching results with scores and metadata
        """
        try:
            if self.index is None:
                self.index = self.get_index()

            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata
            )

            return results.matches

        except Exception as e:
            logger.error(f"Error querying index: {str(e)}")
            raise

    def delete_index(self):
        """Delete the current index"""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Index '{self.index_name}' deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        try:
            if self.index is None:
                self.index = self.get_index()

            stats = self.index.describe_index_stats()
            return stats

        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            raise
