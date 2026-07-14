# llm-language-limits

Experiments on what LLMs do at the **edges of language**.

## Experiment 1 — Repetition (`experiments/repetition/`)
How LLMs respond to absurd repetition of a phrase — single-turn (wall of text)
and multi-turn (conversational insistence) — swept over N, categories, and models.
Design and methodology are documented in the author's project notes (not vendored here yet).

## Setup
```bash
uv sync --extra dev            # core + tests
uv sync --extra open           # + Modal / transformers for open models
cp .env.example .env           # fill in secrets (never commit .env)
```

The local experiment entry points load this repository's `.env` explicitly
and let it override inherited shell variables. This prevents a globally
exported key from another Azure subscription from being used accidentally.

## Run order (staged)
1. `uv run python scripts/verify_credentials.py`  — verify keys, detect limits.
2. `uv run --extra open modal deploy modal_app/deploy_open_models.py` — set `MODAL_CHAT_URL`.
3. `uv run python experiments/repetition/run_smoke.py`  — validate pipeline + cost.
4. `uv run python experiments/repetition/run_pilot.py --yes`  — inspect, RE-TUNE.
5. `uv run python experiments/repetition/run_full.py --yes`  — full sweep.
6. `uv run python experiments/repetition/analyze.py` — coverage report, tables,
   and plots in `data/analysis/`.

The full sweep deliberately uses different grids per delivery mode:
single-turn runs N=`1,3,10,30,100,300,1000`; multi-turn stops at
N=`1,3,10,30,100`. The multi-turn N=300/1000 cells are out of scope because
they add hundreds of sequential generations without improving the comparison.

To run the open-model sweep entirely in Modal (safe to close the local Mac):

```bash
uv run --extra open modal run --detach modal_app/run_sweep_cloud.py \
  --models qwen7b-instruct,qwen7b-base
```

If the configured judge credential is temporarily unavailable, add
`--skip-judge`. Generation and automatic metrics are persisted with
`judge_pending=true`; the analysis excludes them from judge breakdown rates
instead of silently treating them as normal.

Deferred judgments can be filled later without repeating generation:

```bash
uv run python scripts/backfill_judgments.py --judge claude-sonnet
```

The command checkpoints after every successful annotation and records the
judge identity in `judge_model`. Use a different judge only as an explicit
methodological choice; do not mix judge models silently.

Interrupted runs resume by matrix key. To resume one provider without calling
the others, pass explicit labels, for example:

```bash
uv run python experiments/repetition/run_full.py \
  --models qwen7b-instruct qwen7b-base --yes
```

Before resuming Modal models, verify that the Modal workspace is active and
that `MODAL_CHAT_URL` points at the currently deployed `chat_endpoint`. A 404
response containing `workspace ... is disabled` requires account reactivation;
retries cannot recover it.

## Analysis coverage

`analyze.py` checks the full expected matrix before producing results. Its
`data/analysis/REPORT.md` explicitly separates complete models from partial
models; partial-model pooled means are diagnostic and must not be used as
headline comparisons. Re-run the command after any resumed sweep to refresh
the coverage note and figures.

## Methodological invariant
Every model gets the SAME minimal system prompt via raw API. The core experiment
is NEVER routed through an agent harness; the harness-bias probe
(`run_harness_bias.py`) is a separate, isolated appendix.
