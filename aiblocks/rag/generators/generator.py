"""LLM generator wrapping OpenAI, Anthropic, and HuggingFace inference."""

from __future__ import annotations

from aiblocks.rag.config import GeneratorConfig


class Generator:
    """Generates a grounded answer from retrieved context using an LLM."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._client = None      # openai / anthropic client
        self._pipeline = None    # transformers pipeline

    def build(self) -> Generator:
        provider = self.config.provider
        try:
            if provider == "openai":
                from openai import OpenAI
                self._client = OpenAI()

            elif provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()

            elif provider == "huggingface":
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.config.model,
                )

        except ImportError:
            raise ImportError("Run: pip install aiblocks[rag]")

        return self

    def generate(self, query: str, context: str) -> str:
        """Return a grounded answer for query given the retrieved context."""
        self._assert_built()
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        if self.config.provider == "openai":
            return self._generate_openai(user_message)
        if self.config.provider == "anthropic":
            return self._generate_anthropic(user_message)
        if self.config.provider == "huggingface":
            return self._generate_hf(user_message)
        raise ValueError(f"Unknown provider: {self.config.provider}")

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _generate_openai(self, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    def _generate_anthropic(self, user_message: str) -> str:
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _generate_hf(self, user_message: str) -> str:
        full_prompt = f"{self.config.system_prompt}\n\n{user_message}"
        result = self._pipeline(
            full_prompt,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature if self.config.temperature > 0 else None,
            do_sample=self.config.temperature > 0,
        )
        # Strip the input prompt from the generated text
        generated = result[0]["generated_text"]
        return generated[len(full_prompt):].strip()

    def _assert_built(self) -> None:
        if self._client is None and self._pipeline is None:
            raise RuntimeError("Call build() before generate()")
