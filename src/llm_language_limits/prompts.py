from __future__ import annotations

Message = dict  # {"role": str, "content": str}


def build_single_turn(text: str, n: int, sep: str = " ") -> list[Message]:
    if n < 1:
        raise ValueError("n must be >= 1")
    return [{"role": "user", "content": sep.join([text] * n)}]


def build_multi_turn(text: str, n: int, prior_assistant: list[str]) -> list[Message]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(prior_assistant) != n - 1:
        raise ValueError(
            f"prior_assistant must have {n - 1} items, got {len(prior_assistant)}")
    msgs: list[Message] = []
    for i in range(n):
        msgs.append({"role": "user", "content": text})
        if i < len(prior_assistant):
            msgs.append({"role": "assistant", "content": prior_assistant[i]})
    return msgs
