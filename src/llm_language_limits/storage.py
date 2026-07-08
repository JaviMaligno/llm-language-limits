# src/llm_language_limits/storage.py
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def append_record(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_records(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def to_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> None:
    df = pd.DataFrame(read_records(jsonl_path))
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)


def record_key(record: dict) -> tuple:
    return (record["model"], record["category"], record["n"],
            record["mode"], record["replicate"])
