"""Embedding model wrapper for OpenAI, Cohere, HuggingFace, and Ollama providers."""

from __future__ import annotations

from aiblocks.rag.config import EmbeddingConfig


class EmbeddingModel:
    """Generates dense vector embeddings for documents and queries."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._client = None   # openai / cohere / ollama client
        self._model = None    # sentence-transformers model

    def build(self) -> EmbeddingModel:
        provider = self.config.provider

        if provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                raise ImportError("Run: pip install aiblocks[rag]")

        elif provider == "ollama":
            # Ollama exposes an OpenAI-compatible embeddings endpoint
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.config.ollama_base_url,
                    api_key="ollama",
                )
            except ImportError:
                raise ImportError("Run: pip install aiblocks[rag]")

        elif provider == "cohere":
            try:
                import cohere
                self._client = cohere.Client()
            except ImportError:
                raise ImportError("Run: pip install aiblocks[rag-cohere]")

        elif provider == "huggingface":
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.model)
            except ImportError:
                raise ImportError("Run: pip install aiblocks[rag-huggingface]")

        return self

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches. Returns a list of float vectors."""
        self._assert_built()

        if self.config.provider == "huggingface":
            return self._model.encode(
                texts,
                batch_size=self.config.batch_size,
            ).tolist()

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            all_embeddings.extend(self._embed_batch(batch, input_type="search_document"))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        self._assert_built()

        if self.config.provider == "huggingface":
            return self._model.encode([query])[0].tolist()

        return self._embed_batch([query], input_type="search_query")[0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        provider = self.config.provider

        if provider in ("openai", "ollama"):
            response = self._client.embeddings.create(input=texts, model=self.config.model)
            return [item.embedding for item in response.data]

        if provider == "cohere":
            response = self._client.embed(
                texts=texts,
                model=self.config.model,
                input_type=input_type,
            )
            return [list(e) for e in response.embeddings]

        raise ValueError(f"Unknown provider: {provider}")

    def _assert_built(self) -> None:
        if self._client is None and self._model is None:
            raise RuntimeError("Call build() before embed()")
