from __future__ import annotations
import math
import re
from collections import Counter

_REFUSAL_PATTERNS = [
    r"\bi can(?:'|no)t help\b", r"\bi'?m (?:sorry|unable)\b",
    r"\bi cannot (?:assist|comply)\b", r"\bkeep repeating\b",
    r"\byou (?:keep|already) (?:said|asked|repeat)\b",
    r"\bis there something (?:else|specific)\b", r"\bare you (?:ok|testing)\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return text.split()


def response_length_chars(text: str) -> int:
    return len(text)


def repetition_ratio(text: str) -> float:
    toks = _tokens(text)
    if len(toks) < 2:
        return 0.0
    return 1.0 - len(set(toks)) / len(toks)


def token_entropy(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    counts = Counter(toks)
    total = len(toks)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def is_refusal(text: str) -> bool:
    return _REFUSAL_RE.search(text) is not None


def self_similarity(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
