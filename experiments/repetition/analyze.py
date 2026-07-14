"""Generate coverage, summary tables, and publication-ready plots.

The analysis is deliberately coverage-aware: partial models are reported, but
never silently pooled into headline comparisons with complete models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_language_limits.config import (
    MULTI_TURN_N_GRID,
    SINGLE_TURN_N_GRID,
    models_for,
)
from llm_language_limits.stimuli import load_stimuli

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EXPECTED_MODES = ("single", "multi")
EXPECTED_REPLICATES = 3
MODE_GRIDS = {
    "single": SINGLE_TURN_N_GRID,
    "multi": MULTI_TURN_N_GRID,
}


def load_results(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no records in {path}")
    df = pd.DataFrame(rows)
    keys = ["model", "category", "n", "mode", "replicate"]
    duplicates = df.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(f"{duplicates.sum()} duplicate rows for matrix keys")
    df["breakdown"] = pd.array(
        [
            pd.NA if not labels else any(label != "normal" for label in labels)
            for labels in df["judge_labels"]
        ],
        dtype="boolean",
    )
    return df


def target_matrix_mask(df: pd.DataFrame, stimuli_path: Path) -> pd.Series:
    categories = {s.category for s in load_stimuli(stimuli_path)}
    mask = df["category"].isin(categories)
    mode_mask = pd.Series(False, index=df.index)
    for mode, n_grid in MODE_GRIDS.items():
        mode_mask |= df["mode"].eq(mode) & df["n"].isin(n_grid)
    return mask & mode_mask & df["replicate"].between(0, EXPECTED_REPLICATES - 1)


def coverage_table(df: pd.DataFrame, stimuli_path: Path) -> pd.DataFrame:
    categories = [s.category for s in load_stimuli(stimuli_path)]
    expected_per_model = (
        len(categories) * sum(len(grid) for grid in MODE_GRIDS.values())
        * EXPECTED_REPLICATES
    )
    target = target_matrix_mask(df, stimuli_path)
    counts = df[target].groupby("model").size()
    unexpected_counts = df[~target].groupby("model").size()
    rows = []
    for spec in models_for("full"):
        observed = int(counts.get(spec.label, 0))
        rows.append({
            "model": spec.label,
            "observed": observed,
            "expected": expected_per_model,
            "coverage": observed / expected_per_model,
            "complete": observed == expected_per_model,
            "out_of_scope": int(unexpected_counts.get(spec.label, 0)),
        })
    return pd.DataFrame(rows)


def metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    group = df.groupby(["model", "mode", "n"], as_index=False)
    return group.agg(
        cells=("breakdown", "size"),
        breakdown_rate=("breakdown", "mean"),
        repetition_ratio=("repetition_ratio", "mean"),
        entropy=("entropy", "mean"),
        response_length=("length", "mean"),
    )


def _plot_metric(summary: pd.DataFrame, metric: str, label: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, mode in zip(axes, EXPECTED_MODES, strict=True):
        subset = summary[summary["mode"] == mode]
        for model, group in subset.groupby("model"):
            group = group.sort_values("n")
            ax.plot(group["n"], group[metric], marker="o", label=model)
        ax.set(xscale="log", xlabel="Input repetitions (N)", title=mode.title())
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_breakdown_heatmap(df: pd.DataFrame, out: Path) -> None:
    pivot = df.pivot_table(
        index="category", columns="model", values="breakdown", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(max(7, 1.35 * len(pivot.columns)), 6))
    heatmap_values = pivot.astype("Float64").fillna(0).to_numpy(dtype=float)
    image = ax.imshow(heatmap_values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    fig.colorbar(image, ax=ax, label="Breakdown rate")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(coverage: pd.DataFrame, df: pd.DataFrame, path: Path) -> None:
    total_observed = int(coverage["observed"].sum())
    total_expected = int(coverage["expected"].sum())
    lines = [
        "# Repetition experiment: analysis status", "",
        f"Overall coverage: **{total_observed}/{total_expected} "
        f"({total_observed / total_expected:.1%})**.", "",
        "| Model | Target cells | Coverage | Out of scope | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in coverage.itertuples():
        status = "complete" if row.complete else "partial"
        lines.append(
            f"| `{row.model}` | {row.observed}/{row.expected} | "
            f"{row.coverage:.1%} | {row.out_of_scope} | {status} |"
        )
    complete = coverage.loc[coverage["complete"], "model"].tolist()
    lines += ["", "## Interpretation guardrail", ""]
    if complete:
        lines.append(
            "Headline comparisons may currently use only the complete models: "
            + ", ".join(f"`{model}`" for model in complete) + "."
        )
    else:
        lines.append("No model is complete; all findings are provisional.")
    lines.append(
        "Partial-model plots are diagnostic only. Do not compare their pooled "
        "means because missing cells are not balanced across N and mode."
    )
    parse_errors = int(df["judge_confidence"].eq(0).sum())
    pending_judgments = int(df["breakdown"].isna().sum())
    lines += [
        "", f"Judge parse fallbacks (confidence 0): **{parse_errors}**.",
        f"Pending judgments: **{pending_judgments}**. Automatic metrics remain usable; "
        "breakdown rates exclude these rows.", "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/full.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/analysis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input)
    coverage = coverage_table(df, HERE / "stimuli.yaml")
    target_df = df[target_matrix_mask(df, HERE / "stimuli.yaml")].copy()
    summary = metric_summary(target_df)
    coverage.to_csv(args.output_dir / "coverage.csv", index=False)
    summary.to_csv(args.output_dir / "metrics_by_model_mode_n.csv", index=False)
    _plot_metric(summary, "breakdown_rate", "Breakdown rate", args.output_dir / "breakdown_vs_n.png")
    _plot_metric(summary, "repetition_ratio", "Output repetition ratio", args.output_dir / "repetition_vs_n.png")
    _plot_metric(summary, "entropy", "Token entropy", args.output_dir / "entropy_vs_n.png")
    _plot_metric(summary, "response_length", "Response length (characters)", args.output_dir / "length_vs_n.png")
    _plot_breakdown_heatmap(target_df, args.output_dir / "breakdown_by_category.png")
    write_report(coverage, df, args.output_dir / "REPORT.md")
    print(coverage.to_string(index=False, formatters={"coverage": "{:.1%}".format}))


if __name__ == "__main__":
    main()
