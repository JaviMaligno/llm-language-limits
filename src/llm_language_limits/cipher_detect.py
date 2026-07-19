from __future__ import annotations
import re
from .ciphers import Cipher
from .oracle import Task, explicit_decode_ok, normalize

_LETTERS = re.compile(r"[a-zA-Z]")

_ENGLISH_MARGIN = 0.3

_COMMON_WORDS: frozenset[str] = frozenset({
    # function words
    "the", "a", "an", "is", "are", "was", "were", "it", "this", "that",
    "to", "of", "and", "or", "in", "on", "for", "you", "your", "i", "me",
    "my", "we", "us", "our", "he", "she", "they", "them", "their", "here",
    "there", "yes", "no", "not", "so", "ok", "okay", "sure", "hello", "hi",
    "hey", "thanks", "thank", "please", "what", "why", "how", "who", "when",
    "where", "which", "with", "as", "at", "be", "do", "does", "did", "have",
    "has", "had", "will", "would", "can", "could", "should", "must", "may",
    "might", "am", "been", "being", "if", "then", "else", "but", "because",
    "from", "by", "up", "down", "out", "about", "into", "over", "under",
    "again", "all", "any", "some", "more", "most", "other", "such", "only",
    "own", "same", "than", "too", "very", "just", "now", "also", "well",
    "get", "got", "give", "go", "come", "let", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve",
    # content words used in tasks / tests
    "answer", "question", "reply", "word", "words", "name", "capital",
    "color", "colour", "animal", "number", "banana", "apple", "paris",
    "france", "red", "blue", "yellow", "green", "cold", "hot", "dog",
    "cat", "sky", "opposite", "primary", "count", "correct", "true",
    "false", "good", "bad", "day", "night", "time", "year", "way",
    "man", "woman", "people", "thing", "think", "know", "see", "look",
    "want", "need", "use", "find", "tell", "ask", "work", "call",
    "try", "help", "show", "make", "made", "take", "took", "sun",
    "moon", "star", "water", "fire", "earth", "air", "food", "house",
    "car", "book", "school", "friend", "family", "life", "world",
    "hand", "eye", "head", "heart", "mind", "example", "answering",
})


def _looks_like_text(s: str) -> bool:
    if len(s.strip()) < 3:
        return False
    letters = len(_LETTERS.findall(s))
    return letters / max(1, len(s)) > 0.5


def _safe_decode(reply: str, cipher: Cipher) -> str | None:
    try:
        return cipher.decode(reply)
    except Exception:
        return None


def _english_score(s: str) -> float:
    toks = normalize(s).split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in _COMMON_WORDS) / len(toks)


def comprehension_action(reply: str, task: Task, cipher: Cipher) -> bool:
    if task.check(reply):
        return True
    dec = _safe_decode(reply, cipher)
    return bool(dec) and task.check(dec)


def produced_in_code(reply: str, cipher: Cipher, task: Task | None = None, *, min_chars: int = 8) -> bool:
    if cipher.lossy:
        return False
    dec = _safe_decode(reply, cipher)
    if not dec or dec.strip() == reply.strip():
        return False
    # (component 3) oracle anchor: decoded reply satisfies the task, raw does not
    #   -> the model expressed a valid answer that only reads correctly after decoding.
    #   Works even for very short coded answers (e.g. rot13("banana") == "onanan").
    if task is not None and task.check(dec) and not task.check(reply):
        return True
    # (component 1) englishness delta: decoded text reads clearly more like natural
    #   English than the raw reply, and the decoded text looks like text.
    if len(reply.strip()) < min_chars:
        return False
    return (
        _looks_like_text(dec)
        and _english_score(dec) >= _english_score(reply) + _ENGLISH_MARGIN
    )


def explicit_decode(reply: str, plaintext: str) -> bool:
    return explicit_decode_ok(reply, plaintext)
