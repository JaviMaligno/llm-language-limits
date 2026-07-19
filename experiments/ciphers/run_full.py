# experiments/ciphers/run_full.py
"""Full sweep (original-direction: comprehension/production latency + difficulty ranking)
over the models that ENGAGE with ciphers (gpt-5 + Qwen x2). Claude refuses encoded input
(~87%) and is reported separately from the pilot data — see models_for("full").

gpt-5 routing: Azure's default content filter blocks ciphers as jailbreak. Set env
GPT5_DEPLOYMENT to a deployment whose content filter has Prompt Shields disabled
(e.g. a temporary noshield deployment) to let gpt-5 receive ciphered prompts.
"""
import argparse
import os
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from llm_language_limits.clients import get_client
from llm_language_limits.storage import read_records
from llm_language_limits.cost import print_estimate
from config import models_for, CIPHER_SET, PROTOCOLS, TURN_CAP, REPLICATES
from sweep import run_sweep
from run_pilot import _conversation_tokens

OUT = Path("data/ciphers_full.jsonl")


def _resolve_specs():
    gpt5_dep = os.environ.get("GPT5_DEPLOYMENT")
    specs = []
    for s in models_for("full"):
        if s.label == "gpt-5" and gpt5_dep:
            s = replace(s, id=gpt5_dep)   # route gpt-5 to a Prompt-Shields-off deployment
        specs.append(s)
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    specs = _resolve_specs()
    cells = len(specs) * len(CIPHER_SET) * len(PROTOCOLS) * REPLICATES
    conv_in, conv_out = _conversation_tokens(TURN_CAP)
    print(f"roster: {[(s.label, s.id) for s in specs]}")
    for s in specs:
        print_estimate(s.label, cells // len(specs), avg_in=conv_in, avg_out=conv_out)
    print(f"[estimate] ~{cells} conversations x up to {TURN_CAP} turns "
          f"(~{conv_in} in / {conv_out} out tokens per conversation).")
    if not args.yes:
        print("Re-run with --yes to execute the full sweep.")
        return
    run_sweep(get_client, specs, CIPHER_SET, PROTOCOLS, REPLICATES, OUT,
              turn_cap=TURN_CAP, resume=True, max_workers=6)
    print(f"[full] {len(read_records(OUT))} cells done")


if __name__ == "__main__":
    main()
