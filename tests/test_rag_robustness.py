"""Robustness and error-handling tests for the aiblocks RAG module.

Run with:  pytest tests/test_rag_robustness.py -v
No rag dependencies (langchain, chromadb, etc.) are required — all I/O
is either mocked or exercised through pure-Python code paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Allow running from any working directory without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from aiblocks.rag import (
    RAGPipeline,
    IngestionError,
    GenerationError,
    ModelNotAvailableError,
)
from aiblocks.rag.config import LoaderConfig, VectorStoreConfig
from aiblocks.rag.loaders.document_loader import DocumentLoader
from aiblocks.rag.vectorstores.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _built_pipeline(monkeypatch, *, ingested: bool = False, **kwargs) -> RAGPipeline:
    """Return a RAGPipeline with _built=True and optional _ingested, no real build."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = RAGPipeline(**kwargs)
    pipeline._built = True
    pipeline._ingested = ingested
    return pipeline


# ---------------------------------------------------------------------------
# API key checks  (fire in __init__ before any I/O)
# ---------------------------------------------------------------------------

class TestApiKeyChecks:
    def test_missing_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            RAGPipeline()  # default provider is openai

    def test_missing_openai_key_message_suggests_ollama(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ollama"):
            RAGPipeline()

    def test_missing_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")  # embedding is openai
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            RAGPipeline(generator={"provider": "anthropic"})

    def test_missing_cohere_key_on_embedding(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")  # generator is openai
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="COHERE_API_KEY"):
            RAGPipeline(embedding={"provider": "cohere"})

    def test_ollama_provider_needs_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Should NOT raise — ollama has no env-var requirement
        pipeline = RAGPipeline(
            embedding={"provider": "ollama", "model": "nomic-embed-text"},
            generator={"provider": "ollama", "model": "llama3.1:8b"},
        )
        assert pipeline is not None


# ---------------------------------------------------------------------------
# Ollama connectivity check  (fires in build())
# ---------------------------------------------------------------------------

class TestOllamaConnectivity:
    def test_ollama_not_running(self):
        pipeline = RAGPipeline(
            embedding={"provider": "ollama", "model": "nomic-embed-text"},
            generator={"provider": "ollama", "model": "llama3.1:8b"},
        )
        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            with pytest.raises(ConnectionError, match="Ollama is not running"):
                pipeline.build()

    def test_ollama_not_running_raises_model_not_available(self):
        pipeline = RAGPipeline(
            embedding={"provider": "ollama", "model": "nomic-embed-text"},
            generator={"provider": "ollama", "model": "llama3.1:8b"},
        )
        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            with pytest.raises(ModelNotAvailableError):
                pipeline.build()


# ---------------------------------------------------------------------------
# Query validation  (fires in run() / query())
# ---------------------------------------------------------------------------

class TestQueryValidation:
    def test_empty_query_raises(self, monkeypatch):
        pipeline = _built_pipeline(monkeypatch, ingested=True)
        with pytest.raises(ValueError, match="Query cannot be empty"):
            pipeline.run("")

    def test_whitespace_only_query_raises(self, monkeypatch):
        pipeline = _built_pipeline(monkeypatch, ingested=True)
        with pytest.raises(ValueError, match="Query cannot be empty"):
            pipeline.run("   ")

    def test_query_before_ingest_raises(self, monkeypatch):
        pipeline = _built_pipeline(monkeypatch, ingested=False)
        with pytest.raises((RuntimeError, IngestionError), match="No documents ingested"):
            pipeline.run("What is AI?")

    def test_query_alias_also_checks_empty(self, monkeypatch):
        pipeline = _built_pipeline(monkeypatch, ingested=True)
        with pytest.raises(ValueError, match="Query cannot be empty"):
            pipeline.query("")


# ---------------------------------------------------------------------------
# Ingest source validation  (fires in ingest())
# ---------------------------------------------------------------------------

class TestIngestValidation:
    def test_missing_source_directory(self, monkeypatch, tmp_path):
        pipeline = _built_pipeline(monkeypatch)
        missing = str(tmp_path / "does_not_exist")
        with pytest.raises(FileNotFoundError):
            pipeline.ingest(missing)

    def test_missing_source_file(self, monkeypatch, tmp_path):
        pipeline = _built_pipeline(monkeypatch)
        missing_file = str(tmp_path / "ghost.txt")
        with pytest.raises(FileNotFoundError):
            pipeline.ingest(missing_file)

    def test_empty_source_directory(self, monkeypatch, tmp_path):
        pipeline = _built_pipeline(monkeypatch)
        # tmp_path exists but contains no supported files
        with pytest.raises(ValueError, match="No supported files found"):
            pipeline.ingest(str(tmp_path))

    def test_directory_with_only_unsupported_files(self, monkeypatch, tmp_path):
        pipeline = _built_pipeline(monkeypatch)
        (tmp_path / "file.xyz").write_text("ignored")
        with pytest.raises(ValueError, match="No supported files found"):
            pipeline.ingest(str(tmp_path))


# ---------------------------------------------------------------------------
# Document loader error handling
# ---------------------------------------------------------------------------

class TestDocumentLoader:
    def _loader(self) -> DocumentLoader:
        loader = DocumentLoader(LoaderConfig())
        loader._built = True
        return loader

    def test_unsupported_file_type(self, tmp_path):
        bad_file = tmp_path / "report.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            self._loader().load(str(bad_file))

    def test_missing_single_file(self, tmp_path):
        missing = str(tmp_path / "ghost.txt")
        with pytest.raises(FileNotFoundError, match="File not found"):
            self._loader().load(missing)

    def test_zero_documents_raises(self, tmp_path):
        # Build loader then mock _load_file to return []
        loader = self._loader()
        loader._load_file = MagicMock(return_value=[])
        # Create a real .txt file so the path check passes
        real_file = tmp_path / "empty.txt"
        real_file.write_text("")
        with pytest.raises(ValueError, match="No documents could be loaded"):
            loader.load(str(real_file))

    def test_bad_file_in_list_is_skipped_not_crashed(self, tmp_path):
        """One bad file in a list should warn, not kill the whole load."""
        good_file = tmp_path / "good.txt"
        good_file.write_text("real content")

        loader = self._loader()

        def fake_load_file(path):
            if "good" in str(path):
                doc = MagicMock()
                doc.page_content = "real content"
                return [doc]
            raise RuntimeError("corrupted")

        loader._load_file = fake_load_file
        docs = loader.load([str(good_file)])
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# VectorStore validation
# ---------------------------------------------------------------------------

class TestVectorStoreValidation:
    def _store(self) -> VectorStore:
        return VectorStore(VectorStoreConfig())

    def _chunk(self) -> MagicMock:
        c = MagicMock()
        c.page_content = "sample text"
        c.metadata = {}
        return c

    def test_empty_chunks_raises(self):
        with pytest.raises(ValueError, match="empty chunks"):
            self._store().store([], [[0.1, 0.2]])

    def test_empty_embeddings_raises(self):
        with pytest.raises(ValueError, match="Embeddings count"):
            self._store().store([self._chunk()], [])

    def test_chunks_embeddings_mismatch_raises(self):
        chunks = [self._chunk(), self._chunk()]   # 2 chunks
        embeddings = [[0.1, 0.2]]                 # 1 embedding
        with pytest.raises(ValueError, match="Embeddings count"):
            self._store().store(chunks, embeddings)

    def test_empty_query_vector_raises(self):
        with pytest.raises(ValueError, match="Query vector is empty"):
            self._store().search([], top_k=5)
