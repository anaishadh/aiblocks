# Fully local RAG — no API keys required
# Requirements: Ollama running locally (https://ollama.com)
# Pull models first:
#   ollama pull llama3.1:8b
#   ollama pull nomic-embed-text
# Run: python examples/local_rag_quickstart.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from aiblocks.rag import RAGPipeline

pipeline = RAGPipeline(
    embedding={
        "provider": "ollama",
        "model": "nomic-embed-text",
    },
    vectorstore={
        "provider": "chroma",
        "collection_name": "local_demo",
        "persist_dir": "./chroma_local",
    },
    chunker={
        "strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 50,
    },
    retriever={
        "strategy": "dense",
        "top_k": 3,
    },
    generator={
        "provider": "ollama",
        "model": "llama3.1:8b",
        "temperature": 0.1,
        "max_tokens": 512,
    },
).build()

print("Ingesting documents...")
result = pipeline.ingest("examples/sample_docs/")
print(f"Indexed {result['chunks_indexed']} chunks")
print()

queries = [
    "What is artificial intelligence?",
    "What are the main applications of AI?",
    "What are the risks of AI?",
]

print("=" * 60)
print("QUERY RESULTS (fully local, no API key)")
print("=" * 60)

for query in queries:
    response = pipeline.query(query)
    print(f"\nQ: {query}")
    print(f"A: {response['answer']}")
    print(f"Chunks used: {response['chunks_used']}")
    print("-" * 60)
