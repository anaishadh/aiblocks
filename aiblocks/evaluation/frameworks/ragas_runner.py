"""RagasRunner — wraps ragas.evaluate() for the full async-batched evaluation flow."""

from __future__ import annotations

from aiblocks.evaluation.config import RAGMetricsConfig


class RagasRunner:
    """
    Converts an aiblocks dataset to a ragas EvaluationDataset and calls
    ragas.evaluate(), returning a plain {metric_name: float} dict.

    Expected dataset item keys:
        question    (str)         — user query
        answer      (str)         — generated answer
        contexts    (list[str])   — retrieved chunks
        ground_truth (str, opt.)  — reference answer
    """

    def __init__(self, config: RAGMetricsConfig) -> None:
        self.config = config
        self._metrics: list = []
        self._built = False

    def build(self) -> RagasRunner:
        try:
            from ragas.metrics import (  # noqa: F401
                Faithfulness,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                AnswerCorrectness,
            )
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")

        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            AnswerCorrectness,
        )

        if self.config.faithfulness:
            self._metrics.append(Faithfulness())
        if self.config.answer_relevancy:
            self._metrics.append(AnswerRelevancy())
        if self.config.context_precision:
            self._metrics.append(ContextPrecision())
        if self.config.context_recall:
            self._metrics.append(ContextRecall())
        if self.config.answer_correctness:
            self._metrics.append(AnswerCorrectness())

        self._built = True
        return self

    def evaluate(self, dataset: list[dict]) -> dict[str, float]:
        """Run ragas 0.1.x evaluate() and return averaged scores."""
        if not self._built:
            raise RuntimeError("Call build() before evaluate()")

        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            )
            from datasets import Dataset
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")

        metrics_list = []
        if self.config.faithfulness:
            metrics_list.append(faithfulness)
        if self.config.answer_relevancy:
            metrics_list.append(answer_relevancy)
        if self.config.context_precision:
            metrics_list.append(context_precision)
        if self.config.context_recall:
            metrics_list.append(context_recall)
        if self.config.answer_correctness:
            metrics_list.append(answer_correctness)

        if not metrics_list:
            return {}

        hf_dataset = Dataset.from_list([
            {
                "question": item["question"],
                "answer": item["answer"],
                "contexts": item.get("contexts", []),
                "ground_truth": item.get("ground_truth", ""),
            }
            for item in dataset
        ])

        result = evaluate(hf_dataset, metrics=metrics_list)
        return dict(result)

    @staticmethod
    def _to_dict(result) -> dict[str, float]:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            return {col: float(df[col].mean()) for col in df.columns}
        if isinstance(result, dict):
            return {k: float(v) for k, v in result.items()}
        return dict(result)
