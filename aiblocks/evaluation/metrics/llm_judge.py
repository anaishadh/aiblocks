"""LLM-as-judge scorer using OpenAI or Anthropic."""

from __future__ import annotations

import re

from aiblocks.evaluation.config import LLMJudgeConfig


class LLMJudge:
    """Prompts an LLM to score each sample against a rubric and returns the mean."""

    def __init__(self, config: LLMJudgeConfig) -> None:
        self.config = config
        self._client = None

    def build(self) -> LLMJudge:
        try:
            if self.config.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI()
            elif self.config.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
        except ImportError:
            raise ImportError("Run: pip install aiblocks[evaluation]")
        return self

    def judge(self, dataset: list[dict]) -> float:
        """Return the mean rubric score (1–5) across all samples."""
        scores = [
            self._score_single(
                question=item["question"],
                answer=item["answer"],
                context="\n".join(item.get("contexts", [])),
            )
            for item in dataset
        ]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_single(self, question: str, answer: str, context: str) -> float:
        prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer: {answer}\n\n"
            f"Rubric: {self.config.rubric}\n\n"
            "Respond with ONLY a single number between 1 and 5."
        )
        if self.config.provider == "openai":
            raw = self._call_openai(prompt)
        else:
            raw = self._call_anthropic(prompt)
        return self._parse_score(raw)

    def _call_openai(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    @staticmethod
    def _parse_score(text: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            return max(1.0, min(5.0, float(match.group(1))))
        return 0.0
