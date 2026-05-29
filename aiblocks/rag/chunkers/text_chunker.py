"""Text chunker wrapping langchain text splitters."""

from __future__ import annotations

from aiblocks.rag.config import ChunkerConfig


class TextChunker:
    """Splits documents into chunks using configurable strategies."""

    def __init__(self, config: ChunkerConfig) -> None:
        self.config = config
        self._splitter = None

    def build(self) -> TextChunker:
        strategy = self.config.strategy
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        try:
            if strategy == "fixed":
                from langchain_text_splitters import CharacterTextSplitter
                self._splitter = CharacterTextSplitter(
                    chunk_size=size,
                    chunk_overlap=overlap,
                    separator="\n",
                )

            elif strategy == "recursive":
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                self._splitter = RecursiveCharacterTextSplitter(
                    chunk_size=size,
                    chunk_overlap=overlap,
                )

            elif strategy == "sentence":
                from langchain_text_splitters import NLTKTextSplitter
                # NLTK punkt tokenizer must be downloaded once: nltk.download("punkt")
                self._splitter = NLTKTextSplitter(chunk_size=size)

            elif strategy == "token":
                from langchain_text_splitters import TokenTextSplitter
                self._splitter = TokenTextSplitter(
                    chunk_size=size,
                    chunk_overlap=overlap,
                )

            elif strategy == "semantic":
                # SemanticChunker determines boundaries via embedding similarity.
                # Uses OpenAI embeddings by default; requires OPENAI_API_KEY.
                from langchain_text_splitters import SemanticChunker
                from langchain_openai import OpenAIEmbeddings
                self._splitter = SemanticChunker(OpenAIEmbeddings())

        except ImportError:
            raise ImportError("Run: pip install aiblocks[rag]")

        return self

    def chunk(self, documents: list) -> list:
        """Split a list of langchain Documents into smaller chunks."""
        if self._splitter is None:
            raise RuntimeError("Call build() before chunk()")
        return self._splitter.split_documents(documents)
