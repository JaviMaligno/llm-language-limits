from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable


def normalize(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def _tokens(s: str) -> set[str]:
    return set(normalize(s).split())


def _contains(word: str) -> Callable[[str], bool]:
    needed = normalize(word).split()          # phrase -> all tokens must be present
    return lambda resp: needed != [] and all(tok in _tokens(resp) for tok in needed)


def _contains_any(words) -> Callable[[str], bool]:
    normed = [normalize(w).split() for w in words]
    return lambda resp: any(toks and all(t in _tokens(resp) for t in toks) for toks in normed)


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    check: Callable[[str], bool]


TASK_BANK: list[Task] = [
    Task("banana", "reply with the name of the long yellow fruit", _contains("banana")),
    Task("add_7_5", "what is seven plus five", _contains_any(["12", "twelve"])),
    Task("capital_france", "name the capital of france", _contains("paris")),
    Task("sky_blue", "answer with a single word is the daytime sky blue", _contains("yes")),
    Task("color", "name a primary color", _contains_any(["red", "blue", "yellow"])),
    Task("count_three", "count from one to three",
         lambda r: all(n in _tokens(r) for n in ("one", "two", "three"))),
    Task("opposite_hot", "what is the opposite of hot", _contains("cold")),
    Task("animal_bark", "what animal says woof", _contains("dog")),
]


def explicit_decode_ok(model_output: str, plaintext: str) -> bool:
    return normalize(plaintext) in normalize(model_output)
