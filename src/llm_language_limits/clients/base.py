from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol

Message = dict


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    raw: dict | None = None


class ModelClient(Protocol):
    def chat(self, messages: list[Message], system: str,
             temperature: float, max_tokens: int) -> ChatResult: ...


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


class FakeClient:
    """Deterministic client for tests. No network."""

    def __init__(self, reply_fn: Callable[[list[Message]], str] | None = None):
        self._reply_fn = reply_fn

    def chat(self, messages: list[Message], system: str,
             temperature: float, max_tokens: int) -> ChatResult:
        if self._reply_fn is not None:
            text = self._reply_fn(messages)
        else:
            last_user = next((m["content"] for m in reversed(messages)
                              if m["role"] == "user"), "")
            text = f"ack: {last_user}"
        in_toks = sum(_approx_tokens(m["content"]) for m in messages) + _approx_tokens(system)
        return ChatResult(text=text, input_tokens=in_toks,
                          output_tokens=_approx_tokens(text))
