# experiments/ciphers/config.py
from llm_language_limits.config import MODEL_REGISTRY

CIPHER_SET = ["rot13", "random_substitution", "letters_to_digits", "morse",
              "binary", "base64", "reverse_all", "block_permutation",
              "cyrillic_homoglyph", "disemvowel"]
PROTOCOLS = ["pure", "fewshot", "escalating"]
# Tuned after the pilot: comprehension almost always fires by turn 1-5 (median <=5),
# so turn_cap=10 loses nothing vs 15 and cuts cost; reps 3->8 for publishable CIs.
TURN_CAP = 10
REPLICATES = 8


def models_for(tier: str):
    if tier == "smoke":
        return [MODEL_REGISTRY["gpt-5-nano"]]
    if tier == "pilot":
        return [MODEL_REGISTRY[k] for k in
                ("claude-opus", "claude-sonnet", "gpt-5",
                 "qwen7b-instruct", "qwen7b-base")]
    if tier == "full":
        # Tuned after pilot: the ORIGINAL-DIRECTION work (comprehension/production latency,
        # difficulty ranking) only has signal on the models that ENGAGE. Claude opus/sonnet
        # refuse ~87%/86% of encoded turns (a safety-boundary finding, quantified from the
        # pilot at n=3 = 1350 turns/model — already precise), so they are excluded from the
        # full latency sweep and reported separately from pilot data.
        return [MODEL_REGISTRY[k] for k in
                ("gpt-5", "qwen7b-instruct", "qwen7b-base")]
    raise ValueError(tier)
