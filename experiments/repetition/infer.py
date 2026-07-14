"""Cluster-bootstrap and exact sign-flip inference for the repetition sweep.

The stimulus category is the resampling unit: cells sharing a category are
not treated as independent observations. This is intentionally conservative
with only one item per category.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODE_N = {"single": {1, 3, 10, 30, 100, 300, 1000},
          "multi": {1, 3, 10, 30, 100}}
COMPLETE = ["claude-sonnet", "qwen7b-instruct", "qwen7b-base"]


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["breakdown"] = df["judge_labels"].map(
        lambda labels: bool(labels) and any(x != "normal" for x in labels))
    return df[df.apply(lambda r: r["mode"] in MODE_N and r["n"] in MODE_N[r["mode"]], axis=1)]


def bootstrap(values_by_category: dict[str, np.ndarray], rng: np.random.Generator,
              draws: int = 10000) -> tuple[float, float, float]:
    keys = sorted(values_by_category)
    matrix = np.stack([values_by_category[k] for k in keys])
    indices = rng.integers(0, len(keys), size=(draws, len(keys)))
    means = matrix[indices].mean(axis=(1, 2))
    return float(matrix.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        total += 1
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (total + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/full.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/analysis/inference.csv"))
    args = ap.parse_args()
    df = load(args.input)
    rng = np.random.default_rng(20260714)
    rows: list[dict] = []

    for model in COMPLETE:
        for mode in MODE_N:
            sub = df[(df["model"] == model) & (df["mode"] == mode)]
            for metric in ("repetition_ratio", "breakdown"):
                groups = {cat: g[metric].to_numpy(dtype=float)
                          for cat, g in sub.groupby("category")}
                mean, lo, hi = bootstrap(groups, rng)
                rows.append({"comparison": model, "mode": mode, "metric": metric,
                             "estimate": mean, "ci_low": lo, "ci_high": hi,
                             "p_value": ""})

    for mode in MODE_N:
        sub = df[df["mode"] == mode]
        for left, right in (("qwen7b-base", "qwen7b-instruct"),
                            ("qwen7b-base", "claude-sonnet"),
                            ("qwen7b-instruct", "claude-sonnet")):
            for metric in ("repetition_ratio", "breakdown"):
                paired = sub[sub["model"].isin((left, right))].groupby(
                    ["category", "n", "replicate", "model"])[metric].first().unstack("model").dropna()
                diffs = paired[left].astype(float) - paired[right].astype(float)
                category_diffs = diffs.groupby(level="category").mean().to_numpy()
                rows.append({"comparison": f"{left} - {right}", "mode": mode,
                             "metric": metric, "estimate": float(category_diffs.mean()),
                             "ci_low": "", "ci_high": "",
                             "p_value": sign_flip_p(category_diffs)})

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
