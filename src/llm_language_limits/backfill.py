"""Durable backfill of deferred LLM-judge annotations."""
from __future__ import annotations

import time
from pathlib import Path

from .judge import judge_response
from .storage import read_records, write_records


def _judge_with_retry(client, text: str, *, max_attempts: int = 5):
    for attempt in range(1, max_attempts + 1):
        try:
            return judge_response(client, text)
        except Exception as exc:  # noqa: BLE001 - preserve the rest of the dataset
            if attempt == max_attempts:
                print(f"[judge-backfill] failed after {max_attempts} attempts: "
                      f"{type(exc).__name__}: {exc}")
                return None
            backoff = min(30, 2 ** attempt)
            print(f"[judge-backfill] attempt {attempt} failed "
                  f"({type(exc).__name__}: {exc}); backoff {backoff}s")
            time.sleep(backoff)


def backfill_judgments(path: str | Path, judge_client, judge_label: str, *,
                       max_records: int | None = None) -> tuple[int, int]:
    """Fill pending judgments in place, checkpointing after every success.

    Returns ``(completed, remaining)``. Generations and automatic metrics are
    never recomputed or modified.
    """
    records = read_records(path)
    pending = [
        i for i, record in enumerate(records)
        if record.get("judge_pending") is True
    ]
    if max_records is not None:
        pending = pending[:max_records]

    completed = 0
    for index in pending:
        verdict = _judge_with_retry(judge_client, records[index]["text"])
        if verdict is None:
            continue
        records[index]["judge_labels"] = verdict.labels
        records[index]["judge_confidence"] = verdict.confidence
        records[index]["judge_pending"] = False
        records[index]["judge_model"] = judge_label
        records[index]["judge_rationale"] = verdict.rationale
        completed += 1
        write_records(path, records)
        if completed <= 3 or completed % 10 == 0:
            print(f"[judge-backfill] {completed} judgments checkpointed")

    remaining = sum(record.get("judge_pending") is True for record in records)
    return completed, remaining
