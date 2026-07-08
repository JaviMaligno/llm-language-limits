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
            self._c = anthropic.Anthropic(auth_token=key, timeout=90.0, max_retries=0)
        else:
            self._c = anthropic.Anthropic(api_key=key, timeout=90.0, max_retries=0)

    def chat(self, messages, system, temperature, max_tokens) -> ChatResult:
        # Claude 5-family models deprecate/reject `temperature`; they run at their
        # default temperature. `temperature` is accepted for protocol symmetry but
        # not forwarded — greedy (temperature=0) is only available on the open
        # (Modal) models, where we control generation directly.
        resp = self._c.messages.create(
            model=self.spec.id,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return ChatResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw={"stop_reason": resp.stop_reason},
        )
