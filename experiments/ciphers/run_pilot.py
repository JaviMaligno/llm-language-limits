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

# Rough token model for the pre-run cost estimate. Multi-turn conversations re-send a
# GROWING history every turn, so input tokens grow ~linearly with the turn index — a flat
# per-call average (the old avg_in=300) undercounts input by ~(turn_cap+1)/2 (≈8x at
# turn_cap=15). We estimate per-CONVERSATION totals with the arithmetic series instead.
AVG_TURN_IN = 60      # tokens of one short coded task message
AVG_TURN_OUT = 120    # tokens of one model reply
SYS_TOKENS = 15       # fixed system prompt


def _conversation_tokens(turn_cap: int) -> tuple[int, int]:
    """Approx (input, output) tokens for ONE conversation, accounting for context growth.

    Turn t's input carries the system prompt + the (t-1) prior exchanges + the current
    user message; output is one reply per turn. Preamble overhead (fewshot/escalating)
    is NOT included here and makes this a mild UNDER-estimate for those protocols.
    """
    total_in = sum(SYS_TOKENS + (t - 1) * (AVG_TURN_IN + AVG_TURN_OUT) + AVG_TURN_IN
                   for t in range(1, turn_cap + 1))
    total_out = turn_cap * AVG_TURN_OUT
    return total_in, total_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    specs = models_for("pilot")
    cells = len(specs) * len(CIPHER_SET) * len(PROTOCOLS) * REPLICATES
    cells_per_spec = cells // len(specs)
    conv_in, conv_out = _conversation_tokens(TURN_CAP)
    for s in specs:
        # one "call" here = one whole conversation; avg_in/out are per-conversation totals.
        print_estimate(s.label, cells_per_spec, avg_in=conv_in, avg_out=conv_out)
    print(f"[estimate] ~{cells} conversations x up to {TURN_CAP} turns; "
          f"~{conv_in} in / {conv_out} out tokens per conversation (context grows each "
          f"turn). fewshot/escalating add preamble overhead not counted here.")
    if not args.yes:
        print("Re-run with --yes to execute the pilot.")
        return
    run_sweep(get_client, specs, CIPHER_SET, PROTOCOLS, REPLICATES, OUT, turn_cap=TURN_CAP)
    print(f"[pilot] {len(read_records(OUT))} cells done")


if __name__ == "__main__":
    main()
