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
        """Run ragas.evaluate() and return averaged scores."""
        if not self._built:
            raise RuntimeError("Call build() before evaluate()")
        if not self._metrics:
            return {}

        try:
            from ragas import evaluate
            from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")

        samples = [
            SingleTurnSample(
                user_input=item["question"],
                response=item["answer"],
                retrieved_contexts=item.get("contexts", []),
                reference=item.get("ground_truth"),
            )
            for item in dataset
        ]

        result = evaluate(
            EvaluationDataset(samples=samples),
            metrics=self._metrics,
        )
        return self._to_dict(result)

    @staticmethod
    def _to_dict(result) -> dict[str, float]:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            return {col: float(df[col].mean()) for col in df.columns}
        if isinstance(result, dict):
            return {k: float(v) for k, v in result.items()}
        return dict(result)
