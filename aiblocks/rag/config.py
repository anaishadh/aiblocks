"""Configuration classes for every stage of the RAG pipeline."""

from typing import Literal

from aiblocks.core.base import BaseConfig


class LoaderConfig(BaseConfig):
    supported_extensions: list[str] = [".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".json"]
    encoding: str = "utf-8"
    recursive: bool = False


class ChunkerConfig(BaseConfig):
    strategy: Literal["fixed", "recursive", "sentence", "token", "semantic"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50


class EmbeddingConfig(BaseConfig):
    provider: Literal["openai", "cohere", "huggingface"] = "openai"
    model: str = "text-embedding-3-small"
    batch_size: int = 32


class VectorStoreConfig(BaseConfig):
    provider: Literal["chroma", "faiss", "pinecone", "qdrant"] = "chroma"
    collection_name: str = "aiblocks_default"
    persist_dir: str = "./chroma_db"


class RetrieverConfig(BaseConfig):
    strategy: Literal["dense", "sparse", "hybrid"] = "dense"
    top_k: int = 5


class RerankerConfig(BaseConfig):
    enabled: bool = False
    provider: Literal["cohere", "cross-encoder"] = "cohere"
    model: str = "rerank-english-v3.0"
    top_n: int = 3


class GeneratorConfig(BaseConfig):
    provider: Literal["openai", "anthropic", "huggingface"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    system_prompt: str = "You are a helpful assistant. Answer using only the provided context."


class RAGConfig(BaseConfig):
    loader: LoaderConfig = LoaderConfig()
    chunker: ChunkerConfig = ChunkerConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    reranker: RerankerConfig = RerankerConfig()
    generator: GeneratorConfig = GeneratorConfig()
