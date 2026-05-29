"""RAGPipeline — end-to-end retrieval-augmented generation pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from aiblocks.core.base import BaseModule
from aiblocks.rag.config import RAGConfig
from aiblocks.rag.exceptions import (
    GenerationError,
    IngestionError,
    ModelNotAvailableError,
)

# Maps provider name → (env-var name, key URL, whether to suggest ollama fallback)
_API_KEY_REQUIREMENTS: dict[str, tuple[str, str, bool]] = {
    "openai":    ("OPENAI_API_KEY",    "https://platform.openai.com/api-keys",  True),
    "anthropic": ("ANTHROPIC_API_KEY", "https://console.anthropic.com/",        False),
    "cohere":    ("COHERE_API_KEY",    "https://dashboard.cohere.com/api-keys", False),
}


class RAGPipeline(BaseModule):
    """
    Orchestrates the full RAG flow:
      ingest:  load → chunk → embed → store
      run:     retrieve → (rerank) → generate
    """

    def __init__(self, config: RAGConfig | None = None, **kwargs) -> None:
        if config is None:
            config = RAGConfig(**kwargs)
        super().__init__(config)

        # Fail fast on missing API keys before any network or disk I/O.
        cfg: RAGConfig = self.config  # type: ignore[assignment]
        self._check_api_keys(cfg)

        self._built = False
        self._ingested = False
        self._loader = None
        self._chunker = None
        self._embedding = None
        self._vectorstore = None
        self._retriever = None
        self._reranker = None
        self._generator = None

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self) -> RAGPipeline:
        """Instantiate and wire all pipeline components. Lazy-imports underlying libs."""
        cfg: RAGConfig = self.config  # type: ignore[assignment]

        # Check Ollama reachability before spending time on component builds.
        if cfg.embedding.provider == "ollama" or cfg.generator.provider == "ollama":
            self._check_ollama(cfg)

        from aiblocks.rag.loaders.document_loader import DocumentLoader
        from aiblocks.rag.chunkers.text_chunker import TextChunker
        from aiblocks.rag.embeddings.embedding_model import EmbeddingModel
        from aiblocks.rag.vectorstores.vector_store import VectorStore
        from aiblocks.rag.retrievers.retriever import Retriever
        from aiblocks.rag.rerankers.reranker import Reranker
        from aiblocks.rag.generators.generator import Generator

        print("Building loader...")
        self._loader = DocumentLoader(cfg.loader).build()

        print("Building chunker...")
        self._chunker = TextChunker(cfg.chunker).build()

        print("Building embedding model...")
        self._embedding = EmbeddingModel(cfg.embedding).build()

        print("Building vector store...")
        self._vectorstore = VectorStore(cfg.vectorstore).build()

        print("Building retriever...")
        self._retriever = Retriever(cfg.retriever).build(self._vectorstore, self._embedding)

        print("Building reranker...")
        self._reranker = Reranker(cfg.reranker).build()

        print("Building generator...")
        self._generator = Generator(cfg.generator).build()

        self._built = True
        print("Pipeline ready.")
        return self

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------

    def ingest(self, source: str | list[str]) -> dict:
        """
        Index documents into the vector store.

        Args:
            source: A file path, directory path, or list of either.

        Returns:
            {"chunks_indexed": int, "source": source}
        """
        self._require_built()

        cfg: RAGConfig = self.config  # type: ignore[assignment]
        sources = [source] if isinstance(source, str) else list(source)

        # Validate all paths up-front before touching the loader.
        for s in sources:
            p = Path(s)
            if not p.exists():
                raise FileNotFoundError(f"Source path not found: {p.resolve()}")
            if p.is_dir():
                pattern = "**/*" if cfg.loader.recursive else "*"
                supported = [
                    f for f in p.glob(pattern)
                    if f.is_file() and f.suffix.lower() in cfg.loader.supported_extensions
                ]
                if not supported:
                    raise ValueError(
                        f"No supported files found in {p}. "
                        f"Supported: {cfg.loader.supported_extensions}"
                    )

        print(f"Loading from {source!r}...")
        documents = self._loader.load(source)
        print(f"  Loaded {len(documents)} document(s).")

        print("Chunking...")
        chunks = self._chunker.chunk(documents)
        print(f"  Created {len(chunks)} chunk(s).")

        print("Embedding...")
        texts = [c.page_content for c in chunks]
        embeddings = self._embedding.embed(texts)

        print("Storing in vector store...")
        self._vectorstore.store(chunks, embeddings)

        self._retriever.index_corpus(chunks)

        n = len(chunks)
        if n == 0:
            raise IngestionError(
                "Ingestion produced 0 chunks. Check your source files are not empty."
            )
        if n < 3:
            print(
                f"Warning: only {n} chunk(s) indexed. "
                "Retrieval quality may be low."
            )

        self._ingested = True
        print(f"Ingestion complete: {n} chunks indexed.")
        return {"chunks_indexed": n, "source": source}

    # ------------------------------------------------------------------
    # run / query
    # ------------------------------------------------------------------

    def run(self, query: str) -> dict:
        """
        Answer a question using the indexed documents.

        Args:
            query: Natural-language question.

        Returns:
            {"answer": str, "sources": list[str], "chunks_used": int}
        """
        self._require_built()

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not self._ingested:
            raise IngestionError(
                "No documents ingested. Call pipeline.ingest(source) before querying."
            )

        results = self._retriever.retrieve(query)

        if self.config.reranker.enabled:  # type: ignore[union-attr]
            results = self._reranker.rerank(query, results)

        context = "\n\n".join(text for text, _, _ in results)
        sources = [str(meta.get("source", "")) for _, meta, _ in results]

        try:
            answer = self._generator.generate(query, context)
        except Exception as exc:
            model = self.config.generator.model  # type: ignore[union-attr]
            raise GenerationError(
                f"Generation failed: {exc}\n"
                f"If using Ollama, make sure the model is pulled: ollama pull {model}"
            ) from exc

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(results),
        }

    # Both names work: pipeline.run(q) and pipeline.query(q)
    query = run

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("Call build() before using the pipeline.")

    @staticmethod
    def _check_api_keys(cfg: RAGConfig) -> None:
        """Raise EnvironmentError early if a required API key is absent."""
        seen: set[str] = set()
        for provider in (cfg.embedding.provider, cfg.generator.provider):
            if provider in _API_KEY_REQUIREMENTS and provider not in seen:
                seen.add(provider)
                env_var, url, suggest_ollama = _API_KEY_REQUIREMENTS[provider]
                if not os.environ.get(env_var):
                    suffix = (
                        " Or use provider='ollama' for local models."
                        if suggest_ollama else ""
                    )
                    raise EnvironmentError(
                        f"{env_var} environment variable is not set. "
                        f"Get a key at {url}.{suffix}"
                    )

    @staticmethod
    def _check_ollama(cfg: RAGConfig) -> None:
        """Verify Ollama is reachable; raise ModelNotAvailableError if not."""
        import urllib.request

        base_url = (
            cfg.embedding.ollama_base_url
            if cfg.embedding.provider == "ollama"
            else cfg.generator.ollama_base_url
        )
        health_url = base_url.rstrip("/").removesuffix("/v1")

        try:
            urllib.request.urlopen(health_url, timeout=3)  # noqa: S310
        except Exception:
            raise ModelNotAvailableError(
                "Ollama is not running. Start it with: ollama serve\n"
                "Or download from https://ollama.com"
            )
