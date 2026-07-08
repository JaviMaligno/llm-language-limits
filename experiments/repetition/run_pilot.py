# experiments/repetition/run_pilot.py
"""Pilot: 3 models × all categories × reduced N. Inspect results, then RE-TUNE
the N grid / categories / roster before the full sweep."""
from __future__ import annotations
import argparse, time
from pathlib import Path
from llm_language_limits.config import models_for
from llm_language_limits.stimuli import load_stimuli
from llm_language_limits.clients import get_client
from llm_language_limits.config import MODEL_REGISTRY
from llm_language_limits.runner import run_matrix
from llm_language_limits.storage import read_records, to_parquet
from llm_language_limits.cost import estimate_cost, print_estimate

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "data" / "pilot.jsonl"
JUDGE_LABEL = "claude-sonnet"  # fixed primary judge (see spec §4.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()
    specs = models_for("pilot")
    stimuli = load_stimuli(HERE / "stimuli.yaml")
    n_grid = [1, 10, 100]
    replicates = 1
    # Calls per (spec, stimulus): single mode = 1 call per N; multi mode = min(N, cap)
    # calls per N (the sweep fans out multi-turn into that many sequential calls).
    from llm_language_limits.config import MULTITURN_DEFAULT_CAP
    calls_per_stim = sum(1 + min(n, MULTITURN_DEFAULT_CAP) for n in n_grid)
    calls_per_spec = len(stimuli) * calls_per_stim * replicates
    for s in specs:
        print_estimate(s.label, calls_per_spec, avg_in=400, avg_out=80)
    print(f"[estimate] ~{calls_per_spec * len(specs)} total API calls; APPROXIMATE — "
          f"multi-turn late turns carry far more input than avg_in, so treat as a lower bound.")
    from llm_language_limits.config import Provider
    if any(s.provider is Provider.MODAL for s in specs):
        print("[warn] Modal (open) models bill by GPU-hour and are NOT included in the "
              "$ estimate above — budget A10G (7B) / 2xA100-80GB (72B) separately.")
    if not args.yes:
        print("Re-run with --yes to execute the pilot.")
        return
    judge = get_client(MODEL_REGISTRY[JUDGE_LABEL])
    t0 = time.time()
    run_matrix(lambda s: get_client(s), judge, specs, stimuli,
               n_grid=n_grid, modes=["single", "multi"], replicates=1, out_path=OUT)
    to_parquet(OUT, OUT.with_suffix(".parquet"))
    print(f"[pilot] {len(read_records(OUT))} records in {time.time()-t0:.1f}s")
    print(f"[pilot] cost by model: {estimate_cost(read_records(OUT))}")


if __name__ == "__main__":
    main()
