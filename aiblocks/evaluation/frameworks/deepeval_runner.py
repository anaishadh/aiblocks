"""DeepEvalRunner — wraps deepeval test cases and metrics."""

from __future__ import annotations

from aiblocks.evaluation.config import RAGMetricsConfig


class DeepEvalRunner:
    """
    Converts an aiblocks dataset to deepeval LLMTestCase objects, runs each
    enabled metric individually, and returns a plain {metric_name: float} dict.

    Expected dataset item keys:
        question     (str)         — user query
        answer       (str)         — generated answer
        contexts     (list[str])   — retrieved chunks
        ground_truth (str, opt.)   — reference answer
    """

    def __init__(self, config: RAGMetricsConfig) -> None:
        self.config = config
        self._metric_pairs: list[tuple[str, object]] = []  # (name, metric_instance)
        self._built = False

    def build(self) -> DeepEvalRunner:
        try:
            from deepeval.metrics import (  # noqa: F401
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                CorrectnessMetric,
            )
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")

        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            CorrectnessMetric,
        )

        if self.config.faithfulness:
            self._metric_pairs.append(("faithfulness", FaithfulnessMetric()))
        if self.config.answer_relevancy:
            self._metric_pairs.append(("answer_relevancy", AnswerRelevancyMetric()))
        if self.config.context_precision:
            self._metric_pairs.append(("context_precision", ContextualPrecisionMetric()))
        if self.config.context_recall:
            self._metric_pairs.append(("context_recall", ContextualRecallMetric()))
        if self.config.answer_correctness:
            self._metric_pairs.append(("answer_correctness", CorrectnessMetric()))

        self._built = True
        return self

    def evaluate(self, dataset: list[dict]) -> dict[str, float]:
        """Measure each metric over every test case and return averaged scores."""
        if not self._built:
            raise RuntimeError("Call build() before evaluate()")
        if not self._metric_pairs:
            return {}

        try:
            from deepeval.test_case import LLMTestCase
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")

        test_cases = [
            LLMTestCase(
                input=item["question"],
                actual_output=item["answer"],
                expected_output=item.get("ground_truth", ""),
                retrieval_context=item.get("contexts", []),
            )
            for item in dataset
        ]

        results: dict[str, float] = {}
        for name, metric in self._metric_pairs:
            scores: list[float] = []
            for case in test_cases:
                metric.measure(case)
                scores.append(float(metric.score))
            results[name] = round(sum(scores) / len(scores), 4) if scores else 0.0

        return results
