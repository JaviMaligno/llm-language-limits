# AGENTS.md — llm-language-limits

Experiments on what LLMs do at the edges of language. `README.md` is the primary
methodology document; read it before changing an experiment. This file summarises how to
work in the repo and the gotchas that are not obvious from the code.

## Layout

- `src/llm_language_limits/` — shared library (clients, rendering, OCR readers, metrics,
  oracle, cost estimates, storage, retry/rate limiting). Installed package.
- `experiments/repetition/` — part 1: absurd repetition (single- and multi-turn).
- `experiments/ciphers/` — part 2: cipher comprehension/production + jailbreak sub-probe.
- `experiments/perception/` — part 3: multimodal reading under image degradation, OCR
  baselines, invisible-instruction probe. Degradation ranges live in `render_manifest.yaml`.
- `scripts/` — helpers (credential check, judgment backfill, re-judging, human staircase,
  provenance freeze). Importable as `scripts.*` from tests.
- `modal_app/` — Modal deployments for open models (text and VL) and cloud sweeps.
- `tests/` — pytest suite. `docs/` — design specs (`docs/superpowers/specs/`) and blog handoffs.
- `data/` — gitignored outputs and analysis (`data/analysis/...`). Treat as output, never commit.

## Setup and commands

```bash
uv sync --extra dev                 # core + tests
uv sync --extra open                # + Modal / transformers for open models
cp .env.example .env                # fill in keys; never commit .env
uv run pytest                       # test suite
uv run python scripts/verify_credentials.py
```

Each experiment follows the same staged flow: `run_smoke.py` -> `run_pilot.py --yes` ->
`run_full.py --yes` -> `analyze.py` (perception adds `calibrate.py --yes` before the pilot).
Exact invocations per experiment are in `README.md`. Entry points load the repo `.env`
explicitly and override inherited shell variables on purpose.

Paid sweeps resume by matrix key; in `repetition` and `perception`, `run_full.py --models
<labels>` resumes a subset of providers (`ciphers/run_full.py` has no such flag).

## Invariants (do not break)

- Every model gets the same minimal system prompt via raw API; the core experiment never
  goes through an agent harness (`run_harness_bias.py` is an isolated appendix).
- Do not mix judge models silently; `judge_model` is recorded per row.
- Harmful prompts and raw jailbreak completions never leave `data/`.
- Programmatic metrics (CER, oracle, inverse cipher) are preferred over LLM judges.

## Perception experiment pitfalls

Learned by running the pipeline; none is obvious from the already-fixed code.

1. **Match stimulus width, not character count.** Nonsense strings rendered at ~44% of
   their meaningful sibling's width get far more pixels per glyph on a fixed canvas, which
   swamps the prior effect. `config.items_for` pairs items 1:1 by rendered width.
2. **Prompts wider than the canvas are excluded, not clipped** (`render.fits_canvas`); a
   clipped string's CER measures the right margin.
3. **A gate that always passes verifies nothing.** The rescaling check only discriminates at
   a level the model barely reads, so calibrate first (`calibrate.py`) and probe there.
4. **Classic OCR is a poor calibrator.** Tesseract never fails on contrast and fails abruptly
   on rotation. Each reader in `ocr.py` declares a `generation`; report readers per
   generation, never pooled. Apple Vision runs with language correction disabled on purpose
   (correction would supply exactly the prior nonsense stimuli exist to deny).
5. **Every record writer must stamp `manifest_version`** (and `reasoning_effort`). Both are
   part of `sweep.cell_key`; recalibrating a range keeps the coordinates but changes the
   degradation value, so a resume key without the version mixes two experiments in one file.
   This was once fixed in `sweep` but missed in `run_pilot.run_ocr_baselines`, which builds
   records by hand — any new writer must do the same.
6. **Contrast has an 8-bit physical floor.** Ratio 0.004 renders the glyph at grey 254 on
   255; anything lower produces an identical image. A reader still reading there means the
   family has no threshold — that is the result.
7. **One reasoning budget: `reasoning_effort="none"`** for every gpt-5 generation (all accept
   it; gpt-4o and Qwen-VL do not reason). Higher effort measures the token budget, not
   perception (empty answers at `medium`, rambling at `low`). Do not let client code
   renegotiate effort per model; `sweep._guard_uniform_effort` refuses mixed files.
8. **Sign the prior gap by advantage, not subtraction.** In `size` and `contrast` the small
   value is the hard end; plain `meaningful - nonsense` flips the sign there.
9. **Do not present an ordering inside the noise.** If confidence intervals overlap, subjects
   are indistinguishable; do not tabulate them as a ranking.
10. **Keep the bootstrap vectorized.** The naive per-resample Python loop took ~10 min per
    analysis; the vectorized version runs in seconds and a performance test guards it.

Operational notes: a 768x192 image at `detail: high` costs ~246 input tokens on gpt-5-nano.
Azure returns frequent 429s even with few workers; the sweep's retry with backoff absorbs
them. `cost.print_estimate` warns when a model label is missing from `cost.PRICES` (an
unpriced label would otherwise estimate $0.00) — add the price before trusting an estimate.
