"""Reranker supporting Cohere Rerank and sentence-transformers CrossEncoder."""

from __future__ import annotations

from aiblocks.rag.config import RerankerConfig


class Reranker:
    """Re-scores retrieved chunks to push the most relevant ones to the top."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._client = None    # cohere client
        self._model = None     # CrossEncoder model

    def build(self) -> Reranker:
        if not self.config.enabled:
            return self

        try:
            if self.config.provider == "cohere":
                import cohere
                self._client = cohere.Client()

            elif self.config.provider == "cross-encoder":
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.config.model)

        except ImportError:
            raise ImportError("Run: pip install aiblocks[rag]")

        return self

    def rerank(self, query: str, results: list[tuple]) -> list[tuple]:
        """Rerank (text, metadata, score) tuples; return top_n."""
        if not self.config.enabled or not results:
            return results

        if self.config.provider == "cohere":
            return self._rerank_cohere(query, results)
        if self.config.provider == "cross-encoder":
            return self._rerank_cross_encoder(query, results)
        raise ValueError(f"Unknown reranker provider: {self.config.provider}")

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _rerank_cohere(self, query: str, results: list[tuple]) -> list[tuple]:
        texts = [text for text, _, _ in results]
        response = self._client.rerank(
            query=query,
            documents=texts,
            model=self.config.model,
            top_n=self.config.top_n,
        )
        reranked = []
        for item in response.results:
            text, meta, _ = results[item.index]
            reranked.append((text, meta, float(item.relevance_score)))
        return reranked

    def _rerank_cross_encoder(self, query: str, results: list[tuple]) -> list[tuple]:
        texts = [text for text, _, _ in results]
        pairs = [[query, text] for text in texts]
        scores = self._model.predict(pairs)
        scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [
            (r[0], r[1], float(s))
            for r, s in scored[: self.config.top_n]
        ]
