# Run: python examples/rag_quickstart.py
# Requires: OPENAI_API_KEY environment variable set

import os
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from aiblocks.rag import RAGPipeline

# ------------------------------------------------------------------
# Guard: require API key before doing anything
# ------------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY is not set.\n"
        "Export it before running:  export OPENAI_API_KEY=sk-..."
    )

# ------------------------------------------------------------------
# 1. Configure and build the pipeline
# ------------------------------------------------------------------

SAMPLE_DOC = Path(__file__).parent / "sample_docs" / "ai_overview.txt"

pipeline = RAGPipeline(
    chunker={"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 40},
    embedding={"provider": "openai", "model": "text-embedding-3-small"},
    vectorstore={"provider": "chroma", "collection_name": "rag_quickstart", "persist_dir": "./chroma_quickstart"},
    retriever={"strategy": "dense", "top_k": 3},
    generator={"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
)

pipeline.build()

# ------------------------------------------------------------------
# 2. Ingest the sample document
# ------------------------------------------------------------------

print("\n" + "=" * 60)
result = pipeline.ingest(str(SAMPLE_DOC))
print(f"Indexed {result['chunks_indexed']} chunks from {Path(result['source']).name}")

# ------------------------------------------------------------------
# 3. Run queries and print results
# ------------------------------------------------------------------

QUERIES = [
    "What is artificial intelligence?",
    "What are the main applications of AI?",
    "What are the risks of AI?",
]

print("\n" + "=" * 60)
print("QUERY RESULTS")
print("=" * 60)

for query in QUERIES:
    response = pipeline.run(query)

    print(f"\nQ: {query}")
    print(f"A: {response['answer']}")

    sources = [s for s in response["sources"] if s]
    if sources:
        unique_sources = sorted(set(sources))
        print(f"Sources: {', '.join(unique_sources)}")

    print(f"Chunks used: {response['chunks_used']}")
    print("-" * 60)
