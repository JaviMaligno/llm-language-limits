# experiments/repetition/run_full.py
"""Full sweep over the tuned matrix. Prints cost estimate; requires --yes."""
from __future__ import annotations
import argparse, time
from pathlib import Path
from llm_language_limits.config import models_for, DEFAULT_N_GRID, MODEL_REGISTRY
from llm_language_limits.stimuli import load_stimuli
from llm_language_limits.clients import get_client
from llm_language_limits.runner import run_matrix
from llm_language_limits.storage import read_records, to_parquet
from llm_language_limits.cost import estimate_cost, print_estimate

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "data" / "full.jsonl"
JUDGE_LABEL = "claude-sonnet"
REPLICATES = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    specs = models_for("full")
    stimuli = load_stimuli(HERE / "stimuli.yaml")
    # Calls per (spec, stimulus): single = 1 call per N; multi = min(N, cap) calls per N.
    from llm_language_limits.config import MULTITURN_DEFAULT_CAP, Provider
    calls_per_stim = sum(1 + min(n, MULTITURN_DEFAULT_CAP) for n in DEFAULT_N_GRID)
    calls_per_spec = len(stimuli) * calls_per_stim * REPLICATES
    for s in specs:
        print_estimate(s.label, calls_per_spec, avg_in=400, avg_out=150)
    print(f"[estimate] ~{calls_per_spec * len(specs)} total API calls; APPROXIMATE — "
          f"multi-turn late turns carry far more input than avg_in, so treat as a lower bound.")
    if any(s.provider is Provider.MODAL for s in specs):
        print("[warn] Modal (open) models bill by GPU-hour and are NOT included in the "
              "$ estimate above — budget A10G (7B) / 2xA100-80GB (72B) separately.")
    if not args.yes:
        print("Re-run with --yes to execute the FULL sweep.")
        return
    judge = get_client(MODEL_REGISTRY[JUDGE_LABEL])
    t0 = time.time()
    run_matrix(lambda s: get_client(s), judge, specs, stimuli,
               n_grid=DEFAULT_N_GRID, modes=["single", "multi"],
               replicates=REPLICATES, out_path=OUT)
    to_parquet(OUT, OUT.with_suffix(".parquet"))
    print(f"[full] {len(read_records(OUT))} records in {time.time()-t0:.1f}s")
    print(f"[full] cost by model: {estimate_cost(read_records(OUT))}")


if __name__ == "__main__":
    main()
