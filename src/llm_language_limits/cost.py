# src/llm_language_limits/cost.py
from __future__ import annotations

# ($ per 1M input tokens, $ per 1M output tokens). Update at run time.
PRICES: dict[str, tuple[float, float]] = {
    "claude": (3.0, 15.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    "qwen": (0.0, 0.0),   # Modal billed by GPU-hour, not tokens; track separately
}


def _price_for(label: str) -> tuple[float, float]:
    for prefix, price in PRICES.items():
        if label.startswith(prefix):
            return price
    return (0.0, 0.0)


def estimate_cost(records: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in records:
        pin, pout = _price_for(r["model"])
        cost = r["input_tokens"] / 1e6 * pin + r["output_tokens"] / 1e6 * pout
        out[r["model"]] = out.get(r["model"], 0.0) + cost
    return out


def print_estimate(label: str, n_calls: int, avg_in: int, avg_out: int) -> None:
    pin, pout = _price_for(label)
    est = n_calls * (avg_in / 1e6 * pin + avg_out / 1e6 * pout)
    print(f"[estimate] {label}: {n_calls} calls ≈ ${est:.2f} "
          f"(Modal GPU-hours tracked separately if $0)")
