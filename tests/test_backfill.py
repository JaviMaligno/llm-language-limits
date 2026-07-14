from __future__ import annotations

import json

from llm_language_limits.backfill import backfill_judgments
from llm_language_limits.clients.base import FakeClient
from llm_language_limits.storage import read_records, write_records


def test_backfill_updates_only_pending_judgments(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {"model": "base", "text": "loop", "judge_pending": True,
         "judge_labels": [], "judge_confidence": None, "length": 4},
        {"model": "instruct", "text": "ok", "judge_pending": False,
         "judge_labels": ["normal"], "judge_confidence": 1.0, "length": 2},
    ]
    write_records(path, records)
    judge = FakeClient(reply_fn=lambda _: json.dumps({
        "labels": ["degeneration_loop"], "confidence": 0.9,
        "rationale": "repeats",
    }))

    completed, remaining = backfill_judgments(path, judge, "test-judge")

    actual = read_records(path)
    assert (completed, remaining) == (1, 0)
    assert actual[0]["judge_labels"] == ["degeneration_loop"]
    assert actual[0]["judge_confidence"] == 0.9
    assert actual[0]["judge_model"] == "test-judge"
    assert actual[0]["judge_rationale"] == "repeats"
    assert actual[0]["length"] == 4
    assert actual[1] == records[1]


def test_backfill_limit_checkpoints_partial_progress(tmp_path):
    path = tmp_path / "records.jsonl"
    write_records(path, [
        {"text": "a", "judge_pending": True},
        {"text": "b", "judge_pending": True},
    ])
    judge = FakeClient(reply_fn=lambda _: json.dumps({
        "labels": ["normal"], "confidence": 1.0, "rationale": "ok",
    }))

    completed, remaining = backfill_judgments(
        path, judge, "test-judge", max_records=1,
    )

    assert (completed, remaining) == (1, 1)
    assert [r["judge_pending"] for r in read_records(path)] == [False, True]
