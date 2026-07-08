# experiments/repetition/run_smoke.py
"""Smoke run: validate the full pipeline end-to-end against a live provider.
Run AFTER scripts/verify_credentials.py passes."""
from __future__ import annotations
import time
from pathlib import Path
from llm_language_limits.config import models_for
from llm_language_limits.stimuli import load_stimuli
from llm_language_limits.clients import get_client
from llm_language_limits.clients.base import FakeClient
from llm_language_limits.runner import run_matrix
from llm_language_limits.storage import read_records
from llm_language_limits.cost import estimate_cost

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "data" / "smoke.jsonl"


def main():
    spec = models_for("smoke")[0]
    stimuli = [s for s in load_stimuli(HERE / "stimuli.yaml") if s.category == "greeting"]
    judge = get_client(spec)  # reuse the cheap model as judge for smoke only
    t0 = time.time()
    run_matrix(lambda s: get_client(s), judge, [spec], stimuli,
               n_grid=[1, 10], modes=["single", "multi"], replicates=1,
               out_path=OUT, resume=False)
    dt = time.time() - t0
    recs = read_records(OUT)
    print(f"[smoke] {len(recs)} records in {dt:.1f}s")
    print(f"[smoke] cost by model: {estimate_cost(recs)}")


if __name__ == "__main__":
    main()
