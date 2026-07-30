"""Analysis for the cipher experiment (part 2): latency, difficulty, production.

Three methodological choices are load-bearing and are reported side by side rather
than collapsed into one number:

1. **Censoring.** A cell where comprehension never fires is right-censored at
   `n_turns + 1`, never dropped. But a censored mean ALONE cannot tell "no model ever
   cracked this cipher" from "every model cracked it on the last turn", so every table
   also carries the comprehension RATE and the median turn among SOLVED cells.
2. **Contamination.** Keyed ciphers (random substitution, block permutation) are the only
   ones whose key must be inferred in context; the rest are standard schemes the models
   have plausibly memorised. `kind` keeps them separable so latency on `known` ciphers is
   never read as evidence about inference.
3. **Clustering.** The cipher is the resampling unit for uncertainty: cells sharing a
   cipher are not independent observations. With ten ciphers the exact sign-flip test
   bottoms out near 1/1025, so p values are exploratory.

The bootstrap/sign-flip helpers are deliberately re-implemented here instead of imported
from `experiments/repetition/infer.py`: the cluster unit differs (cipher vs stimulus
category) and part 1 is already published — its code stays untouched.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CELL_KEYS = ["model", "cipher", "protocol", "replicate"]
LATENCY_COLS = ["first_action_turn", "first_explicit_turn", "first_production_turn"]
# Keyed per run, so the model cannot have memorised the mapping (spec 6).
NOVEL_CIPHERS = frozenset({"random_substitution", "block_permutation"})


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Markdown table without pulling in `tabulate` (part 1 hand-rolls these too)."""
    def cell(value) -> str:
        if isinstance(value, float):
            return "n/a" if pd.isna(value) else format(value, floatfmt)
        return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cap = int(df["n_turns"].max())
    for col in LATENCY_COLS:
        df[col + "_c"] = df[col].astype("float").fillna(cap + 1)
    df["kind"] = np.where(df["cipher"].isin(NOVEL_CIPHERS), "novel", "known")
    df["comprehended"] = df["first_action_turn"].notna()
    df["produced"] = df["first_production_turn"].notna()
    df["explicit_decoded"] = df["first_explicit_turn"].notna()
    return df


def load_results(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no records in {path}")
    df = pd.DataFrame(rows)
    duplicates = df.duplicated(CELL_KEYS, keep=False)
    if duplicates.any():
        raise ValueError(f"{int(duplicates.sum())} duplicate rows for cell keys {CELL_KEYS}")
    return _augment(df)


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        solved = g["comprehended"]
        return pd.Series({
            "n": len(g),
            "comprehension_rate": solved.mean(),
            "median_turn_if_solved": g.loc[solved, "first_action_turn"].median(),
            "mean_action_censored": g["first_action_turn_c"].mean(),
            "explicit_decode_rate": g["explicit_decoded"].mean(),
            "production_rate": g["produced"].mean(),
            "production_consistency": g.loc[g["produced"], "production_consistency"].mean(),
        })

    out = df.groupby(by, dropna=False).apply(agg, include_groups=False).reset_index()
    return out.astype({"n": int})


def coverage_table(df: pd.DataFrame, *, ciphers: list[str], protocols: list[str],
                   replicates: int) -> pd.DataFrame:
    expected = len(protocols) * replicates
    rows = []
    for model in sorted(df["model"].unique()):
        for cipher in ciphers:
            cells = int(((df["model"] == model) & (df["cipher"] == cipher)).sum())
            rows.append({"model": model, "cipher": cipher, "cells": cells,
                         "expected": expected, "missing": max(0, expected - cells)})
    return pd.DataFrame(rows)


def survival_curve(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Fraction of cells comprehended by turn t (the latency curve of the article)."""
    cap = int(df["n_turns"].max())
    groups = df.groupby(by) if by else [((), df)]
    rows = []
    for key, g in groups:
        for turn in range(1, cap + 1):
            solved = (g["first_action_turn"].astype("float") <= turn).sum()
            row = {"turn": turn, "cum_comprehended": solved / len(g), "n": len(g)}
            if by:
                row |= dict(zip(by, key if isinstance(key, tuple) else (key,)))
            rows.append(row)
    return pd.DataFrame(rows)


def refusal_rates(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Per-turn refusal rate — the safety-boundary finding that shaped the full sweep.

    Only turns that actually reported a stop signal count in the denominator. Clients that
    do not surface one (Azure, Modal) store nulls, and a null is "not observed", not
    "did not refuse" — otherwise the rate is silently diluted by the models that cannot
    report it at all.
    """
    by = by or ["model"]
    rows = []
    for key, g in df.groupby(by, dropna=False):
        signals = [s for cell in g["stop_signals"].dropna() for s in cell if s is not None]
        refused = sum(1 for s in signals if s == "refusal")
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rows.append(row | {
            "cells": len(g),
            "turns_with_signal": len(signals),
            "refused_turns": refused,
            "refusal_rate": refused / len(signals) if signals else float("nan"),
        })
    return pd.DataFrame(rows)


def bootstrap_ci(values_by_cluster: dict[str, np.ndarray], rng: np.random.Generator,
                 draws: int = 10000) -> tuple[float, float, float]:
    """Cluster bootstrap: resample whole ciphers, not individual cells."""
    keys = sorted(values_by_cluster)
    means = np.array([values_by_cluster[k].mean() for k in keys])
    indices = rng.integers(0, len(keys), size=(draws, len(keys)))
    draws_mean = means[indices].mean(axis=1)
    return (float(means.mean()), float(np.quantile(draws_mean, .025)),
            float(np.quantile(draws_mean, .975)))


def sign_flip_p(values: np.ndarray) -> float:
    """Exact sign-flip test over per-cluster differences (conservative, few clusters)."""
    observed = abs(float(values.mean()))
    extreme = total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        total += 1
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (total + 1)


def model_contrasts(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Per-model comprehension/production rates + pairwise contrasts, clustered by cipher."""
    rows = []
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        for metric in ("comprehended", "produced"):
            clusters = {c: g[metric].to_numpy(dtype=float) for c, g in sub.groupby("cipher")}
            mean, lo, hi = bootstrap_ci(clusters, rng)
            rows.append({"comparison": model, "metric": metric, "estimate": mean,
                         "ci_low": lo, "ci_high": hi, "p_value": ""})
    models = sorted(df["model"].unique())
    for left, right in itertools.combinations(models, 2):
        for metric in ("comprehended", "produced"):
            paired = (df[df["model"].isin((left, right))]
                      .groupby(["cipher", "protocol", "replicate", "model"])[metric]
                      .first().unstack("model").dropna())
            if paired.empty or left not in paired or right not in paired:
                continue
            diffs = paired[left].astype(float) - paired[right].astype(float)
            per_cipher = diffs.groupby(level="cipher").mean().to_numpy()
            rows.append({"comparison": f"{left} - {right}", "metric": metric,
                         "estimate": float(per_cipher.mean()), "ci_low": "", "ci_high": "",
                         "p_value": sign_flip_p(per_cipher)})
    return pd.DataFrame(rows)


def plot_latency_heatmap(df: pd.DataFrame, out: Path) -> None:
    piv = df.pivot_table(index="cipher", columns="model",
                         values="first_action_turn_c", aggfunc="mean")
    order = df.groupby("cipher")["first_action_turn_c"].mean().sort_values().index
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i, cipher in enumerate(piv.index):
        for j, model in enumerate(piv.columns):
            value = piv.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if value > piv.values[~np.isnan(piv.values)].mean() else "black")
    ax.set_title("Turns to comprehension (cap+1 = never)")
    fig.colorbar(im, label="mean turn of first correct action")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_survival(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in df.groupby("model"):
        curve = survival_curve(g)
        ax.step(curve["turn"], curve["cum_comprehended"], where="post", label=model)
    ax.set_xlabel("turn")
    ax.set_ylabel("fraction of cells comprehended")
    ax.set_ylim(0, 1)
    ax.set_title("Comprehension latency")
    ax.legend()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_protocol_effect(df: pd.DataFrame, out: Path) -> None:
    table = summarize(df, ["protocol", "kind"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, title in ((axes[0], "comprehension_rate", "comprehension"),
                              (axes[1], "production_rate", "production")):
        piv = table.pivot(index="protocol", columns="kind", values=metric)
        piv.plot.bar(ax=ax, rot=0)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("rate")
    fig.suptitle("Protocol effect, split by known vs keyed-novel ciphers")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_report(df: pd.DataFrame, coverage: pd.DataFrame, out_dir: Path,
                 contrasts: pd.DataFrame, refusals: pd.DataFrame | None = None) -> None:
    by_cipher = summarize(df, ["cipher"]).sort_values("mean_action_censored")
    by_model = summarize(df, ["model"])
    by_protocol = summarize(df, ["protocol"])
    by_kind = summarize(df, ["kind", "protocol"])
    missing = coverage[coverage["missing"] > 0]
    mimicry = summarize(df, ["model", "cipher"])
    mimicry = mimicry[(mimicry["comprehension_rate"] == 0) & (mimicry["production_rate"] > 0)]

    lines = [
        "# Cipher experiment (part 2): analysis status",
        "",
        f"Cells analysed: **{len(df)}** "
        f"({df['model'].nunique()} models x {df['cipher'].nunique()} ciphers x "
        f"{df['protocol'].nunique()} protocols x {df['replicate'].nunique()} replicates), "
        f"turn cap {int(df['n_turns'].max())}.",
        "",
        "## Coverage",
        "",
        (f"Incomplete matrix cells:\n\n{_md_table(missing)}\n\n"
         "Those model x cipher combinations must not appear in a headline comparison; the "
         "per-cipher tables below pool over the models that actually ran. **The absence is "
         "not random**: a whole model x cipher block failing (rather than scattered cells) "
         "indicates a systematic gateway rejection, so the affected cipher's pooled row is "
         "computed over a different model set than the others."
         if not missing.empty else "The matrix is complete for every model x cipher pair."),
        "",
        "## Difficulty ranking (pooled over models and protocols)",
        "",
        _md_table(by_cipher, ".3f"),
        "",
        "## By model",
        "",
        _md_table(by_model, ".3f"),
        "",
        "## By protocol",
        "",
        _md_table(by_protocol, ".3f"),
        "",
        "## Known vs keyed-novel ciphers",
        "",
        _md_table(by_kind, ".3f"),
        "",
        "## Clustered inference (cipher = resampling unit)",
        "",
        _md_table(contrasts, ".4f"),
        "",
    ]
    if refusals is not None and not refusals.empty:
        lines += [
            "## Safety boundary: refusal of encoded turns (pilot roster)",
            "",
            "Measured on the pilot, the only tier where the Claude models ran. A model that "
            "refuses the encoded turn produces no latency data at all, which is why the full "
            "sweep excludes those models from the latency matrix and reports them here "
            "instead. Rates are over turns that reported a stop signal.",
            "",
            _md_table(refusals),
            "",
        ]
    lines += [
        "## Interpretation guardrails",
        "",
        "- `mean_action_censored` treats never-comprehended cells as `cap + 1`; read it "
        "next to `comprehension_rate`, never alone.",
        "- Latency on `known` ciphers conflates inference with memorised schemes. Only the "
        "`novel` rows speak to in-context inference.",
        "- `production_rate` relies on decoding the reply and checking that decoding makes it "
        "more English-like. It is unverifiable for lossy ciphers (disemvowel) and weakest "
        "for letter-level ciphers, whose coded output still looks like letters.",
    ]
    if not mimicry.empty:
        lines += [
            "- **Form mimicry without comprehension**: the cells below produce in-code output "
            "while never once acting correctly on the decoded task. For a base (non-instruct) "
            "model this is the expected completion behaviour — it continues the surface "
            "pattern of the prompt. It must not be reported as 'adopting the code'.",
            "",
            _md_table(mimicry, ".3f"),
        ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "ciphers_full.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "analysis" / "ciphers")
    ap.add_argument("--pilot", type=Path, default=ROOT / "data" / "ciphers_pilot.jsonl",
                    help="pilot records, used only for the refusal-rate (safety) table")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(HERE))
    from config import CIPHER_SET, PROTOCOLS, REPLICATES

    df = load_results(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    coverage = coverage_table(df, ciphers=CIPHER_SET, protocols=PROTOCOLS,
                              replicates=REPLICATES)
    rng = np.random.default_rng(20260729)
    contrasts = model_contrasts(df, rng)

    coverage.to_csv(args.out_dir / "coverage.csv", index=False)
    summarize(df, ["cipher", "model"]).to_csv(args.out_dir / "by_cipher_model.csv", index=False)
    summarize(df, ["protocol", "kind"]).to_csv(args.out_dir / "by_protocol_kind.csv", index=False)
    survival_curve(df, ["model"]).to_csv(args.out_dir / "survival_by_model.csv", index=False)
    contrasts.to_csv(args.out_dir / "inference.csv", index=False)

    refusals = None
    if args.pilot.exists():
        refusals = refusal_rates(load_results(args.pilot), ["model"])
        refusals.to_csv(args.out_dir / "refusal_rates_pilot.csv", index=False)

    plot_latency_heatmap(df, args.out_dir / "comprehension_heatmap.png")
    plot_survival(df, args.out_dir / "comprehension_survival.png")
    plot_protocol_effect(df, args.out_dir / "protocol_effect.png")
    write_report(df, coverage, args.out_dir, contrasts, refusals)

    print(f"[analyze] {len(df)} cells -> {args.out_dir}")
    print(summarize(df, ["cipher"]).sort_values("mean_action_censored").to_string(index=False))


if __name__ == "__main__":
    main()
