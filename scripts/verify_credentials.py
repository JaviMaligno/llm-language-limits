# scripts/verify_credentials.py
"""Cheap liveness check for every provider. Run FIRST, before any sweep."""
from __future__ import annotations
import os, sys
from llm_language_limits.config import MODEL_REGISTRY, SYSTEM_PROMPT
from llm_language_limits.clients import get_client

CHECKS = {
    "ANTHROPIC_API_KEY": "claude-sonnet",
    "AZURE_OPENAI_API_KEY": "gpt-5-nano",
    "MODAL_CHAT_URL": "qwen7b-instruct",
}


def main() -> int:
    any_fail = False
    for env_key, label in CHECKS.items():
        if not os.environ.get(env_key):
            print(f"[skip] {label}: {env_key} not set")
            continue
        try:
            client = get_client(MODEL_REGISTRY[label])
            res = client.chat([{"role": "user", "content": "ping"}],
                              system=SYSTEM_PROMPT, temperature=0.0, max_tokens=8)
            print(f"[ok]   {label}: '{res.text[:30]}' "
                  f"({res.input_tokens}+{res.output_tokens} tok)")
        except Exception as e:  # noqa: BLE001 — report all provider errors
            any_fail = True
            print(f"[FAIL] {label}: {type(e).__name__}: {e}")
    if any_fail:
        print("\nAt least one provider failed. If Anthropic failed, fall back to Azure.")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
