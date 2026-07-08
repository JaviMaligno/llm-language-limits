# llm-language-limits

Experiments on what LLMs do at the **edges of language**.

## Experiment 1 — Repetition (`experiments/repetition/`)
How LLMs respond to absurd repetition of a phrase — single-turn (wall of text)
and multi-turn (conversational insistence) — swept over N, categories, and models.
Design: `personal-website/docs/superpowers/specs/2026-07-07-llm-repetition-breakdown-design.md`.

## Setup
```bash
uv sync --extra dev            # core + tests
uv sync --extra open           # + Modal / transformers for open models
cp .env.example .env           # fill in secrets (never commit .env)
```

## Run order (staged)
1. `uv run python scripts/verify_credentials.py`  — verify keys, detect limits.
2. `uv run --extra open modal deploy modal_app/deploy_open_models.py` — set `MODAL_CHAT_URL`.
3. `uv run python experiments/repetition/run_smoke.py`  — validate pipeline + cost.
4. `uv run python experiments/repetition/run_pilot.py --yes`  — inspect, RE-TUNE.
5. `uv run python experiments/repetition/run_full.py --yes`  — full sweep.
6. Open `experiments/repetition/analysis.ipynb`  — plots.

## Methodological invariant
Every model gets the SAME minimal system prompt via raw API. The core experiment
is NEVER routed through an agent harness; the harness-bias probe
(`run_harness_bias.py`) is a separate, isolated appendix.
