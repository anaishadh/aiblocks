# aiblocks

A unified Python library that wraps LangChain, LlamaIndex, HuggingFace, RAGAS and vLLM under a single clean interface — so you can focus on AI, not tooling.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
[![Install from GitHub](https://img.shields.io/badge/install-from%20github-blue)](https://github.com/anaishadh/aiblocks)

---

## Why aiblocks

Enterprise AI systems require RAG, fine-tuning, agents, evaluation and deployment — each with its own ecosystem of tools. Developers waste weeks learning LangChain, HuggingFace, RAGAS, and vLLM separately, just to wire them together. aiblocks wraps all of it behind one consistent interface with sensible defaults and composable config. You bring the AI knowledge, we handle the tooling.

---

## Modules

| Module      | Status       | What it does                              |
|-------------|--------------|-------------------------------------------|
| RAG         | Available    | Ingest, retrieve, generate                |
| Evaluation  | Available    | Score with RAGAS, BERTScore, LLM-judge    |
| Fine-tuning | Coming soon  | LoRA, QLoRA, DPO, SFT                     |
| Agent       | Coming soon  | ReAct, multi-agent, memory                |
| Deployment  | Coming soon  | vLLM serving, quantization                |

---

## Installation

> **Note:** aiblocks is not yet on PyPI. Install directly from GitHub:
> ```bash
> pip install git+https://github.com/anaishadh/aiblocks.git[rag]
> ```

```bash
# default RAG + core
pip install aiblocks[rag]

# evaluation
pip install aiblocks[evaluation]

# everything
pip install aiblocks[all]
```

---

## Quickstart

**RAG in 8 lines:**

```python
from aiblocks.rag import RAGPipeline

pipeline = RAGPipeline(
    embedding={"provider": "openai"},
    retriever={"strategy": "dense"},
    generator={"model": "gpt-4o-mini"}
).build()

pipeline.ingest("your_docs/")
response = pipeline.query("What is your question?")
print(response["answer"])
```

**Evaluate in 8 lines:**

```python
from aiblocks.evaluation import EvaluationPipeline

evaluator = EvaluationPipeline(
    framework="ragas",
    rag={"faithfulness": True, "answer_relevancy": True}
).build()

scores = evaluator.evaluate(dataset)
print(scores)
```

---

## Local / Open-Source Models

Run aiblocks with zero API keys using Ollama.

**Setup (one time):**
1. Install Ollama from https://ollama.com
2. Pull models:
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

**Run fully local:**
```python
from aiblocks.rag import RAGPipeline

pipeline = RAGPipeline(
    embedding={"provider": "ollama", "model": "nomic-embed-text"},
    generator={"provider": "ollama", "model": "llama3.1:8b"}
).build()

pipeline.ingest("your_docs/")
print(pipeline.query("Your question?")["answer"])
```

Any model available in Ollama works — llama3.1:8b, mistral, gemma2, qwen2.5, phi3 and more.
See available models at https://ollama.com/library

---

## Configuration

All config fields have sensible defaults — you only specify what you want to change.

**Full RAGConfig:**

```python
from aiblocks.rag import RAGConfig

config = RAGConfig(
    loader={"supported_extensions": [".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".json"],
            "encoding": "utf-8",
            "recursive": False},
    chunker={"strategy": "recursive",   # fixed | recursive | sentence | token | semantic
             "chunk_size": 512,
             "chunk_overlap": 50},
    embedding={"provider": "openai",    # openai | cohere | huggingface
               "model": "text-embedding-3-small",
               "batch_size": 32},
    vectorstore={"provider": "chroma",  # chroma | faiss | pinecone | qdrant
                 "collection_name": "aiblocks_default",
                 "persist_dir": "./chroma_db"},
    retriever={"strategy": "dense",     # dense | sparse | hybrid
               "top_k": 5},
    reranker={"enabled": False,
              "provider": "cohere",     # cohere | cross-encoder
              "model": "rerank-english-v3.0",
              "top_n": 3},
    generator={"provider": "openai",    # openai | anthropic | huggingface
               "model": "gpt-4o-mini",
               "temperature": 0.0,
               "max_tokens": 1024,
               "system_prompt": "You are a helpful assistant. Answer using only the provided context."},
)
```

**Full EvaluationConfig:**

```python
from aiblocks.evaluation import EvaluationConfig

config = EvaluationConfig(
    framework="ragas",              # ragas | deepeval | custom
    output_format="dict",           # dict | dataframe | json
    statistical={"bleu": False,
                 "rouge": False,
                 "bertscore": False,
                 "meteor": False},
    rag={"faithfulness": True,
         "answer_relevancy": True,
         "context_precision": False,
         "context_recall": False,
         "answer_correctness": False},
    llm_judge={"enabled": False,
               "provider": "openai",    # openai | anthropic
               "model": "gpt-4o-mini",
               "rubric": "Rate the response on faithfulness, relevance, and completeness on a scale of 1-5."},
    safety={"hallucination_rate": False,
            "toxicity": False},
    operational={"latency": True,
                 "cost_per_query": False},
)
```

---

## Environment Variables

```bash
OPENAI_API_KEY=your-key-here
COHERE_API_KEY=your-key-here        # optional, for Cohere reranker
ANTHROPIC_API_KEY=your-key-here     # optional, for Anthropic generator
```

---

## Roadmap

- [ ] Fine-tuning module (LoRA, QLoRA, DPO)
- [ ] Agent module (ReAct, multi-agent, memory)
- [ ] Deployment module (vLLM, quantization, serving)
- [ ] YAML config support
- [ ] CLI interface
- [ ] More vector stores (Pinecone, Qdrant)

---

## License

MIT

Built with the belief that strong AI theory should not require strong DevOps.
