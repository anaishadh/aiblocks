# Run: python examples/eval_quickstart.py
# Requires: OPENAI_API_KEY environment variable set

import os
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from aiblocks.evaluation import EvaluationPipeline

# ------------------------------------------------------------------
# Guard: require API key before doing anything
# ------------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY is not set.\n"
        "Export it before running:  export OPENAI_API_KEY=sk-..."
    )

# ------------------------------------------------------------------
# Hardcoded dataset — representative outputs from rag_quickstart.py
# contexts are excerpts from examples/sample_docs/ai_overview.txt
# ------------------------------------------------------------------

DATASET = [
    {
        "question": "What is artificial intelligence?",
        "answer": (
            "Artificial intelligence is a branch of computer science focused on building "
            "systems that can perform tasks typically requiring human intelligence, such as "
            "reasoning, learning, problem-solving, perception, and language understanding. "
            "Modern AI is largely powered by machine learning and deep learning."
        ),
        "contexts": [
            (
                "Artificial intelligence (AI) is a branch of computer science focused on "
                "building systems that can perform tasks typically requiring human intelligence. "
                "These tasks include reasoning, learning, problem-solving, perception, language "
                "understanding, and decision-making."
            ),
            (
                "Modern AI systems are largely powered by machine learning, a subfield where "
                "algorithms learn patterns from large datasets rather than being explicitly "
                "programmed with rules. Deep learning, a further specialization using layered "
                "neural networks, has driven many recent breakthroughs."
            ),
        ],
        "ground_truth": (
            "Artificial intelligence is a branch of computer science that builds systems "
            "capable of performing tasks that typically require human intelligence, including "
            "reasoning, learning, and language understanding."
        ),
    },
    {
        "question": "What are the main applications of AI?",
        "answer": (
            "AI is applied across many industries. In healthcare it assists with disease "
            "diagnosis and drug discovery. In transportation, autonomous vehicles rely on AI. "
            "Financial institutions use it for fraud detection and trading. Consumer technology "
            "uses AI for recommendation engines and virtual assistants."
        ),
        "contexts": [
            (
                "The applications of AI span nearly every industry. In healthcare, AI assists "
                "in diagnosing diseases from medical images, predicting patient outcomes, and "
                "accelerating drug discovery."
            ),
            (
                "In transportation, autonomous vehicles use AI to perceive their surroundings "
                "and navigate safely. Financial institutions deploy AI for fraud detection, "
                "algorithmic trading, and personalized banking services."
            ),
            (
                "In consumer technology, recommendation engines power what users see on "
                "streaming platforms and e-commerce sites. Natural language processing has "
                "enabled virtual assistants, chatbots, and real-time translation services."
            ),
        ],
        "ground_truth": (
            "AI is applied in healthcare, transportation, finance, and consumer technology, "
            "enabling use cases such as disease diagnosis, autonomous vehicles, fraud detection, "
            "and recommendation systems."
        ),
    },
    {
        "question": "What are the risks of AI?",
        "answer": (
            "Key risks of AI include biased training data leading to discriminatory outcomes, "
            "lack of transparency in autonomous decision-making, job displacement through "
            "automation, and the alignment problem — the risk that advanced AI systems behave "
            "in unintended ways if not carefully aligned with human values."
        ),
        "contexts": [
            (
                "Despite its promise, AI carries significant risks and challenges. Bias in "
                "training data can lead to discriminatory outcomes in hiring, lending, and "
                "criminal justice systems."
            ),
            (
                "Autonomous systems making high-stakes decisions raise serious questions about "
                "accountability and transparency. The rapid pace of automation threatens to "
                "displace workers in sectors ranging from manufacturing to knowledge work."
            ),
            (
                "At a broader level, advanced AI systems could behave in unintended ways if "
                "their objectives are not carefully aligned with human values — a challenge "
                "researchers call the alignment problem."
            ),
        ],
        "ground_truth": (
            "AI risks include data bias causing discriminatory outcomes, opacity in automated "
            "decisions, job displacement from automation, and the alignment problem."
        ),
    },
]

# ------------------------------------------------------------------
# Build the evaluation pipeline
# ------------------------------------------------------------------

pipeline = EvaluationPipeline(
    framework="ragas",
    rag={"faithfulness": True, "answer_relevancy": True},
    statistical={"bleu": False, "rouge": False},
    llm_judge={"enabled": False},
    operational={"latency": True, "cost_per_query": False},
)

pipeline.build()

# ------------------------------------------------------------------
# Run evaluation and print results
# ------------------------------------------------------------------

scores = pipeline.evaluate(DATASET)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

metadata = scores.pop("metadata", {})

for metric, value in scores.items():
    if isinstance(value, float):
        print(f"  {metric:<35} {value:.4f}")
    else:
        print(f"  {metric:<35} {value}")

print("-" * 60)
print(f"  Samples evaluated: {metadata.get('samples_evaluated', len(DATASET))}")
print(f"  Framework:         {metadata.get('framework', 'ragas')}")
print("=" * 60)
