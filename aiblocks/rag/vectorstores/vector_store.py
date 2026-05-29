"""Vector store wrapper supporting Chroma (default), FAISS, Pinecone, and Qdrant."""

from __future__ import annotations

import hashlib

from aiblocks.rag.config import VectorStoreConfig
from aiblocks.rag.exceptions import RetrievalError


class VectorStore:
    """Stores document embeddings and answers nearest-neighbour queries."""

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self._store = None
        # FAISS keeps documents in memory alongside its index
        self._faiss_docs: list = []
        self._faiss_module = None

    def build(self) -> VectorStore:
        provider = self.config.provider
        try:
            if provider == "chroma":
                import chromadb
                client = chromadb.PersistentClient(path=self.config.persist_dir)
                self._store = client.get_or_create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

            elif provider == "faiss":
                import faiss  # noqa: F401 — just confirm it's importable
                import faiss as faiss_mod
                self._faiss_module = faiss_mod
                # Index is created lazily on first store() call (need embedding dim)

            elif provider == "pinecone":
                import pinecone  # noqa: F401
                raise NotImplementedError(
                    "Pinecone support requires API key configuration. "
                    "Construct the index manually and pass it via a custom VectorStore subclass."
                )

            elif provider == "qdrant":
                from qdrant_client import QdrantClient
                self._store = QdrantClient(path=self.config.persist_dir)

        except ImportError:
            raise ImportError("Run: pip install aiblocks[rag]")

        return self

    def store(self, chunks: list, embeddings: list[list[float]]) -> None:
        """Persist chunks and their pre-computed embeddings."""
        if not chunks:
            raise ValueError("Cannot store empty chunks list.")
        if not embeddings:
            raise ValueError(
                f"Embeddings count (0) does not match chunks count ({len(chunks)}). "
                "This is likely an embedding model error."
            )
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) does not match "
                f"chunks count ({len(chunks)}). "
                "This is likely an embedding model error."
            )

        provider = self.config.provider

        if provider == "chroma":
            ids = [self._make_id(c.page_content, i) for i, c in enumerate(chunks)]
            self._store.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=[c.page_content for c in chunks],
                metadatas=[self._sanitize_metadata(c.metadata) for c in chunks],
            )

        elif provider == "faiss":
            import numpy as np
            vectors = np.array(embeddings, dtype="float32")
            if self._store is None:
                dim = vectors.shape[1]
                self._store = self._faiss_module.IndexFlatL2(dim)
            self._store.add(vectors)
            self._faiss_docs.extend(chunks)

        else:
            raise NotImplementedError(f"store() not implemented for provider '{provider}'")

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple]:
        """Return up to top_k results as (text, metadata, score) tuples."""
        if not query_embedding:
            raise ValueError("Query vector is empty.")

        provider = self.config.provider

        try:
            if provider == "chroma":
                results = self._store.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self._store.count()),
                    include=["documents", "metadatas", "distances"],
                )
                return [
                    (doc, meta, 1.0 - dist)
                    for doc, meta, dist in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ]

            elif provider == "faiss":
                import numpy as np
                vec = np.array([query_embedding], dtype="float32")
                distances, indices = self._store.search(vec, top_k)
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    if 0 <= idx < len(self._faiss_docs):
                        chunk = self._faiss_docs[idx]
                        results.append((chunk.page_content, chunk.metadata, float(dist)))
                return results

            raise NotImplementedError(f"search() not implemented for provider '{provider}'")

        except (NotImplementedError, ValueError):
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Vector search failed: {exc}. "
                f"If using Chroma, the collection may be corrupted. "
                f"Try deleting {self.config.persist_dir} and re-ingesting."
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(content: str, index: int) -> str:
        digest = hashlib.md5(content.encode()).hexdigest()[:10]
        return f"{index}_{digest}"

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """Coerce metadata values to types accepted by Chroma (str/int/float/bool)."""
        out = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif v is None:
                out[k] = ""
            else:
                out[k] = str(v)
        return out
