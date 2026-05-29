"""Retriever supporting dense (ANN), sparse (BM25), and hybrid strategies."""

from __future__ import annotations

from aiblocks.rag.config import RetrieverConfig


class Retriever:
    """Retrieves the most relevant document chunks for a given query."""

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        self._vectorstore = None
        self._embedding_model = None
        # BM25 state — populated by index_corpus()
        self._corpus_texts: list[str] = []
        self._bm25 = None

    def build(self, vectorstore, embedding_model) -> Retriever:
        """Wire up the shared vectorstore and embedding model."""
        self._vectorstore = vectorstore
        self._embedding_model = embedding_model
        return self

    def index_corpus(self, chunks: list) -> None:
        """Build a BM25 index from ingested chunks (required for sparse/hybrid)."""
        if self.config.strategy not in ("sparse", "hybrid"):
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Run: pip install rank-bm25  # needed for sparse/hybrid retrieval")

        self._corpus_texts = [c.page_content for c in chunks]
        tokenized = [text.lower().split() for text in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str) -> list[tuple]:
        """Return a ranked list of (text, metadata, score) tuples."""
        strategy = self.config.strategy
        k = self.config.top_k

        if strategy == "dense":
            return self._dense(query, k)
        if strategy == "sparse":
            return self._sparse(query, k)
        if strategy == "hybrid":
            return self._hybrid(query, k)
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _dense(self, query: str, k: int) -> list[tuple]:
        qvec = self._embedding_model.embed_query(query)
        return self._vectorstore.search(qvec, k)

    def _sparse(self, query: str, k: int) -> list[tuple]:
        if self._bm25 is None:
            raise RuntimeError(
                "BM25 index not built. Call index_corpus() after ingest() for sparse retrieval."
            )
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            (self._corpus_texts[i], {}, float(scores[i]))
            for i in top_indices
        ]

    def _hybrid(self, query: str, k: int) -> list[tuple]:
        dense_results = self._dense(query, k)
        sparse_results = self._sparse(query, k)

        # Normalise dense scores (already cosine similarity in [0,1] for Chroma)
        max_dense = max((s for _, _, s in dense_results), default=1.0) or 1.0
        # Normalise sparse scores
        max_sparse = max((s for _, _, s in sparse_results), default=1.0) or 1.0

        combined: dict[str, list] = {}
        for text, meta, score in dense_results:
            combined[text] = [meta, score / max_dense, 0.0]
        for text, meta, score in sparse_results:
            norm = score / max_sparse
            if text in combined:
                combined[text][2] = norm
            else:
                combined[text] = [meta, 0.0, norm]

        # Equal-weight fusion
        scored = [
            (text, data[0], 0.5 * data[1] + 0.5 * data[2])
            for text, data in combined.items()
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]
