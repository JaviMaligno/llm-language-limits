import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments/repetition/analyze.py"
SPEC = importlib.util.spec_from_file_location("repetition_analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_load_results_marks_any_non_normal_label_as_breakdown(tmp_path):
    path = tmp_path / "results.jsonl"
    records = [
        {"model": "m", "category": "c", "n": 1, "mode": "single",
         "replicate": 0, "judge_labels": ["normal"], "judge_confidence": 1},
        {"model": "m", "category": "c", "n": 1, "mode": "single",
         "replicate": 1, "judge_labels": ["normal", "meta_complaint"],
         "judge_confidence": 1},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records))
    assert analysis.load_results(path)["breakdown"].tolist() == [False, True]


def test_load_results_keeps_unjudged_rows_out_of_breakdown_rate(tmp_path):
    path = tmp_path / "results.jsonl"
    record = {
        "model": "m", "category": "c", "n": 1, "mode": "single",
        "replicate": 0, "judge_labels": [], "judge_confidence": None,
        "judge_pending": True,
    }
    path.write_text(json.dumps(record))
    assert pd.isna(analysis.load_results(path).iloc[0]["breakdown"])


def test_load_results_rejects_duplicate_cells(tmp_path):
    path = tmp_path / "results.jsonl"
    record = {"model": "m", "category": "c", "n": 1, "mode": "single",
              "replicate": 0, "judge_labels": ["normal"]}
    path.write_text(json.dumps(record) + "\n" + json.dumps(record))
    with pytest.raises(ValueError, match="duplicate"):
        analysis.load_results(path)


def test_metric_summary_counts_cells():
    df = pd.DataFrame({
        "model": ["m", "m"], "mode": ["single", "single"], "n": [1, 1],
        "breakdown": [False, True], "repetition_ratio": [0.1, 0.3],
        "entropy": [1.0, 3.0], "length": [10, 30],
    })
    row = analysis.metric_summary(df).iloc[0]
    assert row["cells"] == 2
    assert row["breakdown_rate"] == 0.5
    assert row["repetition_ratio"] == pytest.approx(0.2)


def test_target_matrix_excludes_expensive_multiturn_cells(tmp_path):
    stimuli = tmp_path / "stimuli.yaml"
    stimuli.write_text("stimuli:\n  - category: greeting\n    text: hello\n")
    df = pd.DataFrame({
        "category": ["greeting"] * 4,
        "mode": ["single", "single", "multi", "multi"],
        "n": [1000, 3000, 100, 300],
        "replicate": [0, 0, 2, 0],
    })
    assert analysis.target_matrix_mask(df, stimuli).tolist() == [True, False, True, False]


def test_coverage_uses_asymmetric_mode_grids(tmp_path, monkeypatch):
    stimuli = tmp_path / "stimuli.yaml"
    stimuli.write_text("stimuli:\n  - category: greeting\n    text: hello\n")
    monkeypatch.setattr(
        analysis, "models_for",
        lambda tier: [type("Spec", (), {"label": "m"})()],
    )
    rows = []
    for mode, n_grid in analysis.MODE_GRIDS.items():
        for n in n_grid:
            for replicate in range(analysis.EXPECTED_REPLICATES):
                rows.append({
                    "model": "m", "category": "greeting", "mode": mode,
                    "n": n, "replicate": replicate,
                })
    # A stale N=300 multi-turn row is retained in raw data but is not expected.
    rows.append({
        "model": "m", "category": "greeting", "mode": "multi",
        "n": 300, "replicate": 0,
    })

    row = analysis.coverage_table(pd.DataFrame(rows), stimuli).iloc[0]
    assert row["observed"] == 36
    assert row["expected"] == 36
    assert bool(row["complete"])
    assert row["out_of_scope"] == 1


def test_breakdown_heatmap_accepts_nullable_boolean_values(tmp_path):
    df = pd.DataFrame({
        "category": ["greeting", "greeting"],
        "model": ["judged", "pending"],
        "breakdown": pd.array([True, pd.NA], dtype="boolean"),
    })
    out = tmp_path / "heatmap.png"
    analysis._plot_breakdown_heatmap(df, out)
    assert out.exists()
