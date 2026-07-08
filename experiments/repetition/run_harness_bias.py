# experiments/repetition/run_harness_bias.py
"""APPENDIX (isolated): repeat a message at a Claude Code agent harness and
compare with the raw-API baseline. NOT part of the core dataset."""
from __future__ import annotations
import subprocess, json, time
from pathlib import Path
from llm_language_limits.stimuli import load_stimuli
from llm_language_limits.storage import append_record

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "data" / "harness_bias.jsonl"
TURNS = 30  # small


def ask_harness(prompt: str) -> str:
    """One-shot query to the Claude Code CLI (non-interactive)."""
    p = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True, timeout=120)
    return p.stdout.strip()


def main():
    stim = next(s for s in load_stimuli(HERE / "stimuli.yaml")
                if s.category == "greeting")
    for t in range(1, TURNS + 1):
        reply = ask_harness(stim.text)
        append_record(OUT, {"probe": "harness_bias", "category": stim.category,
                            "turn": t, "text": reply, "ts": time.time()})
        print(f"[harness-bias] turn {t}: {reply[:60]!r}")


if __name__ == "__main__":
    main()
