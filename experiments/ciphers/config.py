# experiments/ciphers/config.py
from llm_language_limits.config import MODEL_REGISTRY

CIPHER_SET = ["rot13", "random_substitution", "letters_to_digits", "morse",
              "binary", "base64", "reverse_all", "block_permutation",
              "cyrillic_homoglyph", "disemvowel"]
PROTOCOLS = ["pure", "fewshot", "escalating"]
TURN_CAP = 15
REPLICATES = 3


def models_for(tier: str):
    if tier == "smoke":
        return [MODEL_REGISTRY["gpt-5-nano"]]
    if tier == "pilot":
        return [MODEL_REGISTRY[k] for k in
                ("claude-opus", "claude-sonnet", "gpt-5",
                 "qwen7b-instruct", "qwen7b-base")]
    if tier == "full":
        # tuned after pilot; default to the pilot roster
        return [MODEL_REGISTRY[k] for k in
                ("claude-opus", "claude-sonnet", "gpt-5",
                 "qwen7b-instruct", "qwen7b-base")]
    raise ValueError(tier)
