"""Backfill records generated with ``judge_pending=true``."""
from __future__ import annotations

import argparse
from pathlib import Path

from llm_language_limits.backfill import backfill_judgments
from llm_language_limits.clients import get_client
from llm_language_limits.config import MODEL_REGISTRY
from llm_language_limits.environment import load_project_env
from llm_language_limits.storage import to_parquet

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "full.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--judge", choices=sorted(MODEL_REGISTRY), default="claude-sonnet",
        help="judge model label; stored on every backfilled record",
    )
    parser.add_argument(
        "--max-records", type=int,
        help="limit work for a credential/cost smoke test",
    )
    args = parser.parse_args()

    load_project_env()
    client = get_client(MODEL_REGISTRY[args.judge])
    completed, remaining = backfill_judgments(
        args.input, client, args.judge, max_records=args.max_records,
    )
    to_parquet(args.input, args.input.with_suffix(".parquet"))
    print(f"[judge-backfill] completed={completed} remaining={remaining} "
          f"judge={args.judge}")
    return 0 if remaining == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
