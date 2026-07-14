# experiments/repetition/run_full.py
"""Full sweep over the tuned matrix. Prints cost estimate; requires --yes."""
from __future__ import annotations
import argparse, time
from pathlib import Path
from llm_language_limits.config import (
    MODEL_REGISTRY,
    MULTI_TURN_N_GRID,
    SINGLE_TURN_N_GRID,
    models_for,
)
from llm_language_limits.stimuli import load_stimuli
from llm_language_limits.clients import get_client
from llm_language_limits.runner import run_matrix
from llm_language_limits.storage import read_records, to_parquet
from llm_language_limits.cost import estimate_cost, print_estimate
from llm_language_limits.environment import load_project_env

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "data" / "full.jsonl"
JUDGE_LABEL = "claude-sonnet"
REPLICATES = 3
MODE_GRIDS = {
    "single": SINGLE_TURN_N_GRID,
    "multi": MULTI_TURN_N_GRID,
}


def run_full_matrix(client_factory, judge, specs, stimuli, out_path=OUT):
    """Run each mode separately so their N grids cannot be mixed accidentally."""
    for mode, n_grid in MODE_GRIDS.items():
        print(f"[full] starting {mode}-turn grid: {n_grid}")
        run_matrix(
            client_factory, judge, specs, stimuli,
            n_grid=n_grid, modes=[mode], replicates=REPLICATES,
            out_path=out_path,
        )


def main():
    load_project_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_REGISTRY),
        help="resume only these model labels (default: the full registry)",
    )
    args = ap.parse_args()
    specs = (
        [MODEL_REGISTRY[label] for label in args.models]
        if args.models else models_for("full")
    )
    stimuli = load_stimuli(HERE / "stimuli.yaml")
    # Single cells cost one generation; multi cells cost N sequential generations.
    from llm_language_limits.config import Provider
    calls_per_stim = len(SINGLE_TURN_N_GRID) + sum(MULTI_TURN_N_GRID)
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
    run_full_matrix(lambda s: get_client(s), judge, specs, stimuli)
    to_parquet(OUT, OUT.with_suffix(".parquet"))
    print(f"[full] {len(read_records(OUT))} records in {time.time()-t0:.1f}s")
    print(f"[full] cost by model: {estimate_cost(read_records(OUT))}")


if __name__ == "__main__":
    main()
