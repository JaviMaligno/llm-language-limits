# src/llm_language_limits/clients/anthropic_client.py
from __future__ import annotations
import os
from .base import ChatResult
from ..config import ModelSpec


class AnthropicClient:
    def __init__(self, spec: ModelSpec):
        import anthropic
        self.spec = spec
        key = os.environ["ANTHROPIC_API_KEY"]
        # OAuth subscription tokens (sk-ant-oat*) authenticate via Bearer, not x-api-key.
        if key.startswith("sk-ant-oat"):
            self._c = anthropic.Anthropic(auth_token=key)
        else:
            self._c = anthropic.Anthropic(api_key=key)

    def chat(self, messages, system, temperature, max_tokens) -> ChatResult:
        resp = self._c.messages.create(
            model=self.spec.id,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return ChatResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw={"stop_reason": resp.stop_reason},
        )
