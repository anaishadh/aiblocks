"""EvaluationPipeline — orchestrates all enabled metric groups over a dataset."""

from __future__ import annotations

import json

from aiblocks.core.base import BaseModule
from aiblocks.evaluation.config import EvaluationConfig


class EvaluationPipeline(BaseModule):
    """
    Runs the full evaluation suite over an aiblocks-format dataset:

        dataset item = {
            "question":    str,
            "answer":      str,
            "contexts":    list[str],
            "ground_truth": str,   # optional — required for some metrics
        }

    Enabled metric groups are determined by EvaluationConfig.
    """

    def __init__(self, config: EvaluationConfig | None = None, **kwargs) -> None:
        if config is None:
            config = EvaluationConfig(**kwargs)
        super().__init__(config)
        self._built = False
        self._statistical = None
        self._rag_runner = None
        self._llm_judge = None
        self._operational = None

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self) -> EvaluationPipeline:
        """Instantiate only the metric components that are enabled in config."""
        cfg: EvaluationConfig = self.config  # type: ignore[assignment]

        stat_cfg = cfg.statistical
        if any([stat_cfg.bleu, stat_cfg.rouge, stat_cfg.bertscore, stat_cfg.meteor]):
            print("Building statistical metrics...")
            from aiblocks.evaluation.metrics.statistical import StatisticalMetrics
            self._statistical = StatisticalMetrics(stat_cfg).build()

        rag_cfg = cfg.rag
        if any([
            rag_cfg.faithfulness,
            rag_cfg.answer_relevancy,
            rag_cfg.context_precision,
            rag_cfg.context_recall,
            rag_cfg.answer_correctness,
        ]):
            print("Building RAG metrics...")
            if cfg.framework == "ragas":
                from aiblocks.evaluation.frameworks.ragas_runner import RagasRunner
                self._rag_runner = RagasRunner(rag_cfg).build()
            elif cfg.framework == "deepeval":
                from aiblocks.evaluation.frameworks.deepeval_runner import DeepEvalRunner
                self._rag_runner = DeepEvalRunner(rag_cfg).build()
            else:  # custom
                from aiblocks.evaluation.metrics.rag_metrics import RAGMetrics
                self._rag_runner = RAGMetrics(rag_cfg).build()

        if cfg.llm_judge.enabled:
            print("Building LLM judge...")
            from aiblocks.evaluation.metrics.llm_judge import LLMJudge
            self._llm_judge = LLMJudge(cfg.llm_judge).build()

        op_cfg = cfg.operational
        if any([op_cfg.latency, op_cfg.cost_per_query]):
            print("Building operational metrics...")
            from aiblocks.evaluation.metrics.operational import OperationalMetrics
            self._operational = OperationalMetrics(op_cfg)

        self._built = True
        print("Evaluation pipeline ready.")
        return self

    # ------------------------------------------------------------------
    # run / evaluate
    # ------------------------------------------------------------------

    def run(self, dataset: list[dict]) -> dict:
        """
        Evaluate the dataset and return a results dict.

        Args:
            dataset: List of dicts with keys question, answer, contexts,
                     and optionally ground_truth.

        Returns:
            Flat dict of {metric_name: score, ..., "metadata": {...}}.
            When output_format is "json"      → JSON string.
            When output_format is "dataframe" → pandas DataFrame.
        """
        if not self._built:
            raise RuntimeError("Call build() before run()")

        cfg: EvaluationConfig = self.config  # type: ignore[assignment]
        results: dict = {}

        if self._operational and cfg.operational.latency:
            self._operational.start_timer()

        # --- Statistical metrics ---
        if self._statistical is not None:
            results.update(self._statistical.compute(dataset))

        # --- RAG metrics ---
        if self._rag_runner is not None:
            results.update(self._rag_runner.evaluate(dataset))

        # --- LLM judge ---
        if self._llm_judge is not None:
            results["llm_judge_score"] = self._llm_judge.judge(dataset)

        # --- Operational metrics ---
        if self._operational is not None:
            if cfg.operational.latency:
                results.update(self._operational.measure_latency(len(dataset)))
            if cfg.operational.cost_per_query:
                results.update(self._operational.estimate_cost(dataset))

        results["metadata"] = {
            "samples_evaluated": len(dataset),
            "framework": cfg.framework,
        }

        return self._format(results, cfg.output_format)

    evaluate = run  # alias

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format(results: dict, fmt: str):
        if fmt == "json":
            return json.dumps(results, indent=2, default=str)
        if fmt == "dataframe":
            try:
                import pandas as pd
                flat = {k: v for k, v in results.items() if k != "metadata"}
                df = pd.DataFrame([flat])
                df.attrs["metadata"] = results.get("metadata", {})
                return df
            except ImportError:
                # pandas not installed; fall back to dict silently
                return results
        return results
