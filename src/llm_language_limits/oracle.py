from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable


def normalize(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def _contains(word: str) -> Callable[[str], bool]:
    w = normalize(word)
    return lambda resp: w in normalize(resp)


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    check: Callable[[str], bool]


TASK_BANK: list[Task] = [
    Task("banana", "reply with the single word banana", _contains("banana")),
    Task("add_7_5", "what is seven plus five", _contains("12")),
    Task("capital_france", "name the capital of france", _contains("paris")),
    Task("sky_blue", "reply yes or no is the sky blue", _contains("yes")),
    Task("color", "name a primary color",
         lambda r: any(c in normalize(r) for c in ("red", "blue", "yellow"))),
    Task("count_three", "count from one to three",
         lambda r: all(n in normalize(r) for n in ("one", "two", "three"))),
    Task("opposite_hot", "what is the opposite of hot", _contains("cold")),
    Task("animal_bark", "what animal says woof", _contains("dog")),
]


def explicit_decode_ok(model_output: str, plaintext: str) -> bool:
    return normalize(plaintext) in normalize(model_output)
