from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

EXPECTED_CATEGORIES: frozenset[str] = frozenset({
    "greeting", "answerable_question", "command", "insult",
    "threat_distress", "praise", "single_word", "gibberish",
    "nonlexical_noise",
})


@dataclass(frozen=True)
class Stimulus:
    category: str
    text: str
    note: str


def load_stimuli(path: str | Path) -> list[Stimulus]:
    data = yaml.safe_load(Path(path).read_text())
    out: list[Stimulus] = []
    for item in data["stimuli"]:
        cat = item["category"]
        if cat not in EXPECTED_CATEGORIES:
            raise ValueError(f"unknown category: {cat}")
        out.append(Stimulus(category=cat, text=item["text"], note=item.get("note", "")))
    return out
