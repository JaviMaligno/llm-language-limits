# llm-language-limits

Experiments on what LLMs do at the **edges of language**.

## Experiment 1 — Repetition (`experiments/repetition/`)
How LLMs respond to absurd repetition of a phrase — single-turn (wall of text)
and multi-turn (conversational insistence) — swept over N, categories, and models.
Design and methodology are documented in the author's project notes (not vendored here yet).

## Experiment 2 — Ciphers (`experiments/ciphers/`)
How fast a model infers that it is being addressed in a non-human encoding
(comprehension) and whether it starts replying in it (production), over ten
ciphers x three exposure protocols. Both core metrics are programmatic — a task
oracle plus the inverse cipher — so no LLM judge is involved. Design:
`docs/superpowers/specs/2026-07-15-ciphers-experiment-design.md`.

```bash
uv run python experiments/ciphers/run_smoke.py            # pipeline check
uv run python experiments/ciphers/run_pilot.py --yes      # inspect, prune, re-tune
uv run python experiments/ciphers/run_full.py --yes       # tuned matrix
uv run python experiments/ciphers/analyze.py              # tables + figures + REPORT.md
```

`analyze.py` writes `data/analysis/ciphers/`, checks the expected matrix, and
keeps three things separable that must never be pooled: censored latency vs
comprehension rate, memorised schemes vs keyed-novel ciphers, and in-code output
that comes with comprehension vs mere form mimicry.

The bounded jailbreak sub-probe (`jailbreak_probe.py`, design in
`docs/superpowers/specs/2026-07-17-*`) is a separate robustness appendix with a
**two-part result**, and both parts are load-bearing:

- **On the 7B models the uplift is not identifiable.** Two independent judges agree on
  plaintext (kappa 0.66–0.74) and *worse than chance* on their ciphered replies
  (−0.17 and −0.05), so the uplift `analyze_jailbreak.py` prints for them must not be
  reported — including its uniformly negative sign, which reflects compliances the
  judge failed to recognise in half-decoded text rather than protection.
- **On gpt-5 the labels are usable and a bounded bypass is real.** Judges agree
  perfectly on its plaintext cells and at 0.818 raw on its ciphered ones. It refused
  all 30 plaintext cells, and complied with two distinct prompts under two distinct
  ciphers (letters→digits, reverse_all), reproduced across replicates and confirmed by
  both judges. Reported as an existence result, never as a rate: ten prompts cannot
  estimate frequency, and every interval includes zero.

Always compute agreement per subject and per cipher before reporting any rate from
this probe — that check is what separates the two halves above.

```bash
GPT5_DEPLOYMENT=<no-shield deployment> \
  uv run python experiments/ciphers/jailbreak_probe.py --yes \
    --models qwen7b-instruct qwen7b-base --judge <label> \
    --audit-dump data/jailbreak_audit.jsonl        # opt-in dump, delete after use
uv run python scripts/rejudge_jailbreak.py prepare --per-label 10000 --batch-size 30
uv run python scripts/rejudge_jailbreak.py judge-azure --model gpt-4o
uv run python scripts/rejudge_jailbreak.py merge   # agreement, per condition
uv run python experiments/ciphers/analyze_jailbreak.py
```

`PROBE_VERSION 1` records are an invalid measurement (they judged the decoded reply
unconditionally) and are ignored both on resume and in analysis; only v2 may be
analysed. Harmful prompts and raw completions never leave the gitignored `data/`
directory: the results file stores labels and numeric metrics only, and the audit
dump is the single opt-in exception, to be deleted once agreement is computed.

Note on gateway configuration: the default Azure content filter rejects encoded
prompts before the model sees them. A deployment whose jailbreak shield is set to
annotate-only fixes that for most ciphers, but base64 stays blocked by the
**self-harm** classifier, which assigns medium severity to base64 of an innocuous
sentence. Harm categories are not configurable without limited-access approval.

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
