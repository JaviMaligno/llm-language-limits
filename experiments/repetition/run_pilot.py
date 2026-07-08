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
    n_calls = len(specs) * len(stimuli) * len(n_grid) * 2 * 1
    for s in specs:
        print_estimate(s.label, n_calls // len(specs), avg_in=200, avg_out=120)
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
