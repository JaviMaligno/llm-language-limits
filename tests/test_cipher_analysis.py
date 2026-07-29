import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "ciphers" / "analyze.py"
SPEC = importlib.util.spec_from_file_location("cipher_analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _cell(**kw):
    base = {"model": "gpt-5", "cipher": "rot13", "protocol": "pure", "replicate": 0,
            "n_turns": 10, "first_action_turn": 1, "first_explicit_turn": 3,
            "first_production_turn": None, "production_consistency": 0.0}
    return base | kw


def _write(tmp_path, rows):
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_censors_never_solved_at_cap_plus_one(tmp_path):
    df = analysis.load_results(_write(tmp_path, [
        _cell(first_action_turn=None), _cell(replicate=1, first_action_turn=4)]))
    assert sorted(df["first_action_turn_c"]) == [4.0, 11.0]


def test_load_tags_keyed_ciphers_as_novel(tmp_path):
    df = analysis.load_results(_write(tmp_path, [
        _cell(cipher="rot13"), _cell(cipher="random_substitution"),
        _cell(cipher="block_permutation")]))
    assert dict(zip(df["cipher"], df["kind"])) == {
        "rot13": "known", "random_substitution": "novel", "block_permutation": "novel"}


def test_load_rejects_duplicate_cells(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        analysis.load_results(_write(tmp_path, [_cell(), _cell()]))


def test_summarize_separates_impossible_from_barely_solved():
    # A cipher nobody ever cracks and one cracked on the last turn have nearly the same
    # censored mean; the comprehension rate is what distinguishes them.
    rows = [_cell(cipher="hard", first_action_turn=None, replicate=i) for i in range(4)]
    rows += [_cell(cipher="late", first_action_turn=10, replicate=i) for i in range(4)]
    frame = analysis._augment(pd.DataFrame(rows))
    out = analysis.summarize(frame, ["cipher"]).set_index("cipher")
    assert out.loc["hard", "comprehension_rate"] == 0.0
    assert out.loc["late", "comprehension_rate"] == 1.0
    assert abs(out.loc["hard", "mean_action_censored"] - 11.0) < 1e-9
    assert abs(out.loc["late", "mean_action_censored"] - 10.0) < 1e-9
    assert np.isnan(out.loc["hard", "median_turn_if_solved"])


def test_coverage_reports_missing_cells(tmp_path):
    df = analysis.load_results(_write(tmp_path, [_cell()]))
    cov = analysis.coverage_table(df, ciphers=["rot13"], protocols=["pure"], replicates=2)
    row = cov.set_index(["model", "cipher"]).loc[("gpt-5", "rot13")]
    assert row["cells"] == 1 and row["expected"] == 2 and row["missing"] == 1


def test_survival_curve_is_cumulative_and_bounded():
    rows =[_cell(first_action_turn=1, replicate=0), _cell(first_action_turn=5, replicate=1),
            _cell(first_action_turn=None, replicate=2)]
    curve = analysis.survival_curve(analysis._augment(pd.DataFrame(rows)))
    frac = curve["cum_comprehended"].tolist()
    assert frac == sorted(frac)
    assert frac[0] == pytest.approx(1 / 3) and frac[-1] == pytest.approx(2 / 3)


def test_refusal_rates_use_only_turns_that_reported_a_signal():
    # Anthropic replies carry per-turn stop signals; Azure/Modal ones are null. A null must
    # count as "not observed", never as "did not refuse", or the rate is silently diluted.
    rows = [
        _cell(model="claude-opus", stop_signals=["refusal"] * 8 + ["stop"] * 2),
        _cell(model="claude-opus", replicate=1, stop_signals=["refusal"] * 10),
        _cell(model="qwen7b-base", stop_signals=[None] * 10),
    ]
    out = analysis.refusal_rates(analysis._augment(pd.DataFrame(rows))).set_index("model")
    assert out.loc["claude-opus", "turns_with_signal"] == 20
    assert out.loc["claude-opus", "refusal_rate"] == pytest.approx(0.9)
    assert out.loc["qwen7b-base", "turns_with_signal"] == 0
    assert np.isnan(out.loc["qwen7b-base", "refusal_rate"])


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    clusters = {f"c{i}": np.array([0.0, 1.0]) for i in range(8)}
    mean, lo, hi = analysis.bootstrap_ci(clusters, rng, draws=500)
    assert mean == pytest.approx(0.5) and lo <= mean <= hi


def test_sign_flip_p_is_smallest_when_all_differences_agree():
    same = analysis.sign_flip_p(np.array([0.3] * 8))
    mixed = analysis.sign_flip_p(np.array([0.3, -0.3] * 4))
    # with 8 clusters only the all-same-sign flips are as extreme -> 3/(2^8+1)
    assert same == pytest.approx(3 / 257)
    assert same < mixed and mixed == pytest.approx(1.0)
