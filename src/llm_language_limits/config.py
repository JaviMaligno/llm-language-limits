from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    MODAL = "modal"


@dataclass(frozen=True)
class ModelSpec:
    id: str                      # provider-native model id / deployment name
    provider: Provider
    label: str                   # short stable key used in storage + plots
    is_base: bool = False        # True for non-RLHF base models
    exposes_hidden_states: bool = False  # True only for Modal-hosted open models


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "claude-sonnet": ModelSpec("claude-sonnet-5", Provider.ANTHROPIC, "claude-sonnet"),
    "claude-opus": ModelSpec("claude-opus-4-8", Provider.ANTHROPIC, "claude-opus"),
    # id = real Azure deployment name (from code-world-models); label stays stable for pricing/plots.
    "gpt-5": ModelSpec("gpt-5.4", Provider.AZURE_OPENAI, "gpt-5"),
    "gpt-5-mini": ModelSpec("gpt-5.4-mini", Provider.AZURE_OPENAI, "gpt-5-mini"),
    "gpt-5-nano": ModelSpec("gpt-5-nano", Provider.AZURE_OPENAI, "gpt-5-nano"),
    "qwen7b-instruct": ModelSpec(
        "Qwen/Qwen2.5-7B-Instruct", Provider.MODAL, "qwen7b-instruct",
        exposes_hidden_states=True),
    "qwen7b-base": ModelSpec(
        "Qwen/Qwen2.5-7B", Provider.MODAL, "qwen7b-base",
        is_base=True, exposes_hidden_states=True),
    "qwen72b-instruct": ModelSpec(
        "Qwen/Qwen2.5-72B-Instruct", Provider.MODAL, "qwen72b-instruct",
        exposes_hidden_states=True),
}

DEFAULT_N_GRID: list[int] = [1, 3, 10, 30, 100, 300, 1000]
MULTITURN_MAX_TURNS: int = 300
MULTITURN_DEFAULT_CAP: int = 100

# Minimal controlled system prompt used for EVERY model (methodological invariant).
SYSTEM_PROMPT: str = "You are a helpful assistant."


def models_for(tier: str) -> list[ModelSpec]:
    if tier == "smoke":
        return [MODEL_REGISTRY["gpt-5-nano"]]
    if tier == "pilot":
        return [MODEL_REGISTRY[k] for k in
                ("gpt-5-nano", "claude-sonnet", "qwen7b-instruct")]
    if tier == "full":
        return list(MODEL_REGISTRY.values())
    raise ValueError(f"unknown tier: {tier}")
