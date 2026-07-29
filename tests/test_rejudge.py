import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rejudge_jailbreak.py"
SPEC = importlib.util.spec_from_file_location("rejudge", MODULE_PATH)
rejudge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rejudge)


def _view(i, label="REFUSED", channel="plain"):
    return {"model": "qwen7b-base", "cipher": "rot13", "prompt_id": i % 10,
            "replicate": i % 3, "channel": channel, "label": label,
            "judged_text": f"response number {i}"}


def test_view_key_identifies_a_judged_view_uniquely():
    a = _view(1)
    b = dict(a, channel="coded")
    assert rejudge.view_key(a) != rejudge.view_key(b)
    assert rejudge.view_key(a) == rejudge.view_key(dict(a))


def test_sample_is_stratified_by_original_label_and_deterministic():
    views = ([_view(i, "REFUSED") for i in range(40)]
             + [_view(i, "COMPLIED") for i in range(40, 45)]
             + [_view(i, "GARBLED") for i in range(45, 90)])
    sample = rejudge.sample_views(views, per_label=10, seed=7)
    counts = {}
    for v in sample:
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    assert counts["REFUSED"] == 10 and counts["GARBLED"] == 10
    assert counts["COMPLIED"] == 5          # fewer than per_label -> take all
    assert [rejudge.view_key(v) for v in sample] == [
        rejudge.view_key(v) for v in rejudge.sample_views(views, per_label=10, seed=7)]


def test_batches_cover_every_view_exactly_once():
    views = [_view(i) for i in range(25)]
    batches = rejudge.make_batches(views, size=10)
    assert [len(b) for b in batches] == [10, 10, 5]
    seen = [rejudge.view_key(v) for b in batches for v in b]
    assert len(seen) == len(set(seen)) == 25


def test_rendered_batch_numbers_cases_and_includes_both_sides():
    batch = [_view(1), _view(2)]
    text = rejudge.render_batch(batch, requests={0: "do something bad",
                                                1: "do something else bad",
                                                2: "third"})
    assert "CASE 1" in text and "CASE 2" in text
    assert "response number 1" in text and "response number 2" in text
    assert "do something" in text


def test_parse_labels_maps_positions_back_to_view_keys():
    batch = [_view(1), _view(2)]
    out = rejudge.parse_labels('{"1": "COMPLIED", "2": "GARBLED"}', batch)
    assert out[rejudge.view_key(batch[0])] == "COMPLIED"
    assert out[rejudge.view_key(batch[1])] == "GARBLED"


def test_parse_labels_tolerates_surrounding_prose():
    batch = [_view(1)]
    out = rejudge.parse_labels('Here you go:\n{"1": "REFUSED"}\nDone.', batch)
    assert out[rejudge.view_key(batch[0])] == "REFUSED"


def test_parse_labels_rejects_unknown_labels_and_missing_cases():
    batch = [_view(1), _view(2)]
    with pytest.raises(ValueError, match="unknown label"):
        rejudge.parse_labels('{"1": "MAYBE", "2": "REFUSED"}', batch)
    with pytest.raises(ValueError, match="missing"):
        rejudge.parse_labels('{"1": "REFUSED"}', batch)


def test_agreement_reports_raw_and_kappa():
    a = {"x": "REFUSED", "y": "COMPLIED", "z": "REFUSED", "w": "GARBLED"}
    b = {"x": "REFUSED", "y": "COMPLIED", "z": "GARBLED", "w": "GARBLED"}
    out = rejudge.agreement(a, b)
    assert out["n"] == 4
    assert out["raw_agreement"] == pytest.approx(0.75)
    assert -1.0 <= out["cohens_kappa"] <= 1.0
    assert out["confusion"][("REFUSED", "GARBLED")] == 1


def test_agreement_is_perfect_when_labels_match():
    a = {"x": "REFUSED", "y": "COMPLIED"}
    out = rejudge.agreement(a, dict(a))
    assert out["raw_agreement"] == 1.0 and out["cohens_kappa"] == 1.0
