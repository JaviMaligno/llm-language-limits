# experiments/ciphers/run_pilot.py
"""Pilot: broad roster × all ciphers × all protocols. Inspect, then prune."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from llm_language_limits.clients import get_client
from llm_language_limits.storage import read_records
from llm_language_limits.cost import print_estimate
from config import models_for, CIPHER_SET, PROTOCOLS, TURN_CAP, REPLICATES
from sweep import run_sweep

OUT = Path("data/ciphers_pilot.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    specs = models_for("pilot")
    cells = len(specs) * len(CIPHER_SET) * len(PROTOCOLS) * REPLICATES
    for s in specs:
        print_estimate(s.label, (cells // len(specs)) * TURN_CAP, avg_in=300, avg_out=120)
    print(f"[estimate] ~{cells} cells x up to {TURN_CAP} turns each.")
    if not args.yes:
        print("Re-run with --yes to execute the pilot.")
        return
    run_sweep(get_client, specs, CIPHER_SET, PROTOCOLS, REPLICATES, OUT, turn_cap=TURN_CAP)
    print(f"[pilot] {len(read_records(OUT))} cells done")


if __name__ == "__main__":
    main()
