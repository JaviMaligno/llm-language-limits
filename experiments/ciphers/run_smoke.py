# experiments/ciphers/run_smoke.py
"""Smoke: validate the cipher pipeline end-to-end against one cheap model."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from llm_language_limits.clients import get_client
from llm_language_limits.config import MODEL_REGISTRY
from llm_language_limits.storage import read_records
from sweep import run_sweep

OUT = Path("data/ciphers_smoke.jsonl")


def main():
    spec = MODEL_REGISTRY["gpt-5-nano"]
    run_sweep(get_client, [spec], ["rot13", "random_substitution"], ["pure"], 1,
              OUT, turn_cap=6, resume=False)
    for r in read_records(OUT):
        print(f"{r['cipher']:20} action@{r['first_action_turn']} "
              f"decode@{r['first_explicit_turn']} prod@{r['first_production_turn']}")


if __name__ == "__main__":
    main()
