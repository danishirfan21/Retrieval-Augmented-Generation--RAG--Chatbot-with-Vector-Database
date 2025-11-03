"""
Embedding generation with provider support.

Primary: Hugging Face Inference API (no local model install).
Fallback: OpenAI Embeddings (if configured) when HF fails or is unavailable.
"""
from typing import List, Optional, Sequence
import logging
import httpx
from openai import OpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles text embedding generation with HF Inference API and optional OpenAI fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator

        Args:
            model_name: Hugging Face model id to use for embeddings
        """
        self.model_name = model_name
        settings = get_settings()
        self._provider: str = (settings.embedding_provider or "huggingface").lower()
        self._allow_fallback = bool(getattr(settings, "allow_fallback_to_openai", True))

        # Known model dimension hints to avoid probing before index creation
        self._model_dim_hints = {
            "BAAI/bge-large-en-v1.5": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        # HF settings
        self.api_token: Optional[str] = settings.huggingface_api_token
        if self.api_token:
            # Sanitize common mistakes: tokens pasted with quotes/spaces
            self.api_token = self.api_token.strip().strip('"').strip("'")
        self.api_url = (
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        )
        self.headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}

        # OpenAI settings
        self._openai_api_key: Optional[str] = getattr(settings, "openai_api_key", None)
        self._openai_client: Optional[OpenAI] = None
        self._openai_model: str = "text-embedding-3-small"

        # Dimension (prefer hint; probe lazily if unknown)
        self.dimension = self._model_dim_hints.get(self.model_name)

        logger.info(
            f"Embedding generator initialized (provider={self._provider}, model={self.model_name})"
        )

    def _post_feature_extraction(self, inputs: Sequence[str]) -> List[List[float]]:
        """Call HF Inference API for feature extraction and return pooled embeddings.

        Handles both already-pooled outputs and per-token outputs by averaging tokens.
        """
        payload = {
            "inputs": list(inputs),
            "options": {"wait_for_model": True},
        }
        timeout = httpx.Timeout(60.0, connect=30.0)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(self.api_url, headers=self.headers, json=payload)
            if resp.status_code == 503:
                # Model loading; brief backoff and retry once
                logger.warning("Model loading on HF Inference, retrying once...")
                resp = client.post(self.api_url, headers=self.headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"HF Inference API error {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()

        # data can be:
        # - List[float] when a single input returns a pooled embedding (rare)
        # - List[List[float]] when single input returns token embeddings (we pool)
        # - List[List[List[float]]] when multiple inputs return token embeddings

        def pool_token_embeddings(tokens: List[List[float]]) -> List[float]:
            # mean-pool across tokens
            if not tokens:
                return []
            dim = len(tokens[0])
            sums = [0.0] * dim
            for t in tokens:
                # guard different lengths
                if len(t) != dim:
                    dim = min(dim, len(t))
                    t = t[:dim]
                    sums = sums[:dim]
                for i, v in enumerate(t):
                    sums[i] += float(v)
            n = max(1, len(tokens))
            return [s / n for s in sums]

        embeddings: List[List[float]] = []
        # Normalize shapes
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            # Single pooled vector
            embeddings = [list(map(float, data))]
        elif isinstance(data, list) and data and isinstance(data[0], list) and data and data and data and isinstance(data[0][0] if data[0] else [], (int, float)):
            # Single input -> token-level outputs
            embeddings = [pool_token_embeddings(data)]
        elif isinstance(data, list):
            # Multiple inputs case
            for item in data:
                if isinstance(item, list) and item and isinstance(item[0], (int, float)):
                    embeddings.append(list(map(float, item)))
                elif isinstance(item, list):
                    embeddings.append(pool_token_embeddings(item))
                else:
                    raise RuntimeError("Unexpected response format from HF Inference API")
        else:
            raise RuntimeError("Unexpected response format from HF Inference API")

        return embeddings

    def _openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        if self._openai_client is None:
            if not self._openai_api_key:
                raise RuntimeError("OpenAI API key not configured for fallback embeddings")
            self._openai_client = OpenAI(api_key=self._openai_api_key)
        res = self._openai_client.embeddings.create(model=self._openai_model, input=texts)
        vectors = [d.embedding for d in res.data]
        # Set/override dimension from result
        if vectors and (self.dimension is None or self.dimension == 0):
            self.dimension = len(vectors[0])
        return vectors

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts via HF Inference API (batched)."""
        logger.info(f"Generating embeddings for {len(texts)} texts (provider={self._provider})")
        try:
            if self._provider == "openai":
                return self._openai_embeddings(texts)

            # Default to HF
            results: List[List[float]] = []
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embs = self._post_feature_extraction(batch)
                results.extend(embs)
            # set dimension if unknown
            if results and getattr(self, "dimension", None) in (None, 0):
                self.dimension = len(results[0])
            return results
        except Exception as e:
            msg = str(e)
            is_auth_error = ("401" in msg) or ("Unauthorized" in msg)
            if self._provider == "huggingface" and self._allow_fallback and is_auth_error:
                logger.warning(
                    "HF embeddings failed with 401. Falling back to OpenAI embeddings (text-embedding-3-small)."
                )
                # Switch provider and model
                self._provider = "openai"
                self.model_name = self._openai_model
                self.dimension = self._model_dim_hints.get(self.model_name)
                return self._openai_embeddings(texts)
            raise

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        logger.info(f"Generating embedding for query: {query[:50]} (provider={self._provider})")
        if self._provider == "openai":
            return self._openai_embeddings([query])[0]
        return self._post_feature_extraction([query])[0]

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings (probes HF API if unknown)."""
        if getattr(self, "dimension", None) in (None, 0):
            # Prefer hint; otherwise, probe using provider
            hint = self._model_dim_hints.get(self.model_name)
            if hint:
                self.dimension = hint
            else:
                try:
                    self.dimension = len(self.generate_query_embedding("dimension probe"))
                except Exception as e:
                    logger.error(f"Failed to determine embedding dimension: {e}")
                    raise
        return int(self.dimension)
