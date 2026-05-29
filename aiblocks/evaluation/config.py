"""Configuration classes for every stage of the Evaluation pipeline."""

from typing import Literal

from aiblocks.core.base import BaseConfig


class StatisticalMetricsConfig(BaseConfig):
    bleu: bool = False
    rouge: bool = False        # computes rouge1, rouge2, rougeL
    bertscore: bool = False
    meteor: bool = False


class RAGMetricsConfig(BaseConfig):
    faithfulness: bool = True
    answer_relevancy: bool = True
    context_precision: bool = False
    context_recall: bool = False
    answer_correctness: bool = False


class LLMJudgeConfig(BaseConfig):
    enabled: bool = False
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    rubric: str = (
        "Rate the response on faithfulness, relevance, "
        "and completeness on a scale of 1-5."
    )


class SafetyMetricsConfig(BaseConfig):
    hallucination_rate: bool = False
    toxicity: bool = False


class OperationalMetricsConfig(BaseConfig):
    latency: bool = True         # measures total evaluation time in ms
    cost_per_query: bool = False  # estimates cost from token counts × model pricing


class EvaluationConfig(BaseConfig):
    statistical: StatisticalMetricsConfig = StatisticalMetricsConfig()
    rag: RAGMetricsConfig = RAGMetricsConfig()
    llm_judge: LLMJudgeConfig = LLMJudgeConfig()
    safety: SafetyMetricsConfig = SafetyMetricsConfig()
    operational: OperationalMetricsConfig = OperationalMetricsConfig()
    framework: Literal["ragas", "deepeval", "custom"] = "ragas"
    output_format: Literal["dict", "dataframe", "json"] = "dict"
