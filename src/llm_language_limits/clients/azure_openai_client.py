# src/llm_language_limits/clients/azure_openai_client.py
from __future__ import annotations
import os
from .base import ChatResult
from ..config import ModelSpec, Provider
from ..ratelimit import throttle


class AzureOpenAIClient:
    def __init__(self, spec: ModelSpec):
        from openai import AzureOpenAI
        self.spec = spec
        self._c = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            timeout=90.0,
            max_retries=0,
        )

    def chat(self, messages, system, temperature, max_tokens) -> ChatResult:
        throttle(Provider.AZURE_OPENAI)
        full = [{"role": "system", "content": system}, *messages]
        kwargs = {"model": self.spec.id, "messages": full}
        if self.spec.id.startswith("gpt-5"):
            # GPT-5 reasoning models: use max_completion_tokens and only the
            # default temperature (they reject a custom temperature). Set
            # reasoning_effort="minimal" so the model emits a visible assistant
            # reply instead of spending the whole token budget on hidden
            # reasoning — this study measures the visible RESPONSE to repetition,
            # not the reasoning trace.
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        resp = self._c.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return ChatResult(
            text=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            raw={"finish_reason": choice.finish_reason,
                 "content_filter": getattr(choice, "content_filter_results", None)},
        )
