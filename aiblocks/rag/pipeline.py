"""RAGPipeline — end-to-end retrieval-augmented generation pipeline."""

from __future__ import annotations

from aiblocks.core.base import BaseModule
from aiblocks.rag.config import RAGConfig


class RAGPipeline(BaseModule):
    """
    Orchestrates the full RAG flow:
      ingest:  load → chunk → embed → store
      run:     retrieve → (rerank) → generate
    """

    def __init__(self, config: RAGConfig | None = None, **kwargs) -> None:
        if config is None:
            # Allow partial overrides:  RAGPipeline(chunker={"strategy": "semantic"})
            # Pydantic coerces nested dicts to the correct sub-config type automatically.
            config = RAGConfig(**kwargs)
        super().__init__(config)
        self._built = False
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
        from aiblocks.rag.loaders.document_loader import DocumentLoader
        from aiblocks.rag.chunkers.text_chunker import TextChunker
        from aiblocks.rag.embeddings.embedding_model import EmbeddingModel
        from aiblocks.rag.vectorstores.vector_store import VectorStore
        from aiblocks.rag.retrievers.retriever import Retriever
        from aiblocks.rag.rerankers.reranker import Reranker
        from aiblocks.rag.generators.generator import Generator

        cfg: RAGConfig = self.config  # type: ignore[assignment]

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

        # BM25 index for sparse / hybrid retrieval
        self._retriever.index_corpus(chunks)

        print(f"Ingestion complete: {len(chunks)} chunks indexed.")
        return {"chunks_indexed": len(chunks), "source": source}

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

        results = self._retriever.retrieve(query)

        if self.config.reranker.enabled:  # type: ignore[union-attr]
            results = self._reranker.rerank(query, results)

        context = "\n\n".join(text for text, _, _ in results)
        sources = [str(meta.get("source", "")) for _, meta, _ in results]

        answer = self._generator.generate(query, context)

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(results),
        }

    # Both names work: pipeline.run(q) and pipeline.query(q)
    query = run

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("Call build() before using the pipeline.")
