"""Operational metrics: latency timing and cost estimation."""

from __future__ import annotations

import time

from aiblocks.evaluation.config import OperationalMetricsConfig

# Per-token pricing in USD (input, output) for common models.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":          (0.15e-6,  0.60e-6),
    "gpt-4o":               (5.00e-6, 15.00e-6),
    "gpt-4-turbo":          (10.0e-6, 30.0e-6),
    "claude-3-5-sonnet":    (3.00e-6, 15.00e-6),
    "claude-3-haiku":       (0.25e-6,  1.25e-6),
}
# Fallback when the model is not in the table
_DEFAULT_PRICING = (1.00e-6, 3.00e-6)
# Rough words-to-tokens conversion factor
_WORDS_PER_TOKEN = 0.75


class OperationalMetrics:
    """Measures evaluation latency and estimates per-query LLM costs."""

    def __init__(self, config: OperationalMetricsConfig) -> None:
        self.config = config
        self._start: float | None = None

    # ------------------------------------------------------------------
    # Timer helpers (called by EvaluationPipeline around run())
    # ------------------------------------------------------------------

    def start_timer(self) -> None:
        self._start = time.perf_counter()

    def measure_latency(self, n_samples: int) -> dict[str, float]:
        """Return total and per-sample latency in milliseconds."""
        elapsed_ms = (time.perf_counter() - self._start) * 1_000
        return {
            "eval_latency_ms": round(elapsed_ms, 2),
            "mean_eval_latency_ms_per_sample": round(elapsed_ms / max(n_samples, 1), 2),
        }

    # ------------------------------------------------------------------
    # Cost estimation (no external deps)
    # ------------------------------------------------------------------

    def estimate_cost(self, dataset: list[dict], model: str = "gpt-4o-mini") -> dict[str, float]:
        """
        Estimate total and per-query LLM cost from token counts.

        Token counts are approximated from word counts when not present in dataset items.
        If a dataset item contains 'input_tokens' / 'output_tokens' keys those are used
        directly (e.g. when calling aiblocks RAGPipeline with token tracking).
        """
        input_rate, output_rate = _PRICING.get(model, _DEFAULT_PRICING)
        total_cost = 0.0

        for item in dataset:
            if "input_tokens" in item and "output_tokens" in item:
                i_tok = item["input_tokens"]
                o_tok = item["output_tokens"]
            else:
                i_words = len(item.get("question", "").split()) + sum(
                    len(c.split()) for c in item.get("contexts", [])
                )
                o_words = len(item.get("answer", "").split())
                i_tok = i_words / _WORDS_PER_TOKEN
                o_tok = o_words / _WORDS_PER_TOKEN

            total_cost += i_tok * input_rate + o_tok * output_rate

        n = len(dataset)
        return {
            "estimated_total_cost_usd": round(total_cost, 6),
            "estimated_cost_per_query_usd": round(total_cost / n, 6) if n else 0.0,
        }
