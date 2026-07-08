# src/llm_language_limits/clients/azure_openai_client.py
from __future__ import annotations
import os
from .base import ChatResult
from ..config import ModelSpec


class AzureOpenAIClient:
    def __init__(self, spec: ModelSpec):
        from openai import AzureOpenAI
        self.spec = spec
        self._c = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )

    def chat(self, messages, system, temperature, max_tokens) -> ChatResult:
        full = [{"role": "system", "content": system}, *messages]
        resp = self._c.chat.completions.create(
            model=self.spec.id,           # Azure deployment name
            messages=full,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        return ChatResult(
            text=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            raw={"finish_reason": choice.finish_reason,
                 "content_filter": getattr(choice, "content_filter_results", None)},
        )
