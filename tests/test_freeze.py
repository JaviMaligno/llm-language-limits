import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "freeze_provenance.py"
SPEC = importlib.util.spec_from_file_location("freeze", MODULE_PATH)
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def test_file_digest_is_content_addressed(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"x": 1}\n')
    b.write_text('{"x": 1}\n')
    assert freeze.file_digest(a) == freeze.file_digest(b)
    b.write_text('{"x": 2}\n')
    assert freeze.file_digest(a) != freeze.file_digest(b)


def test_describe_dataset_reports_records_and_digest(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n')
    out = freeze.describe_dataset(path)
    assert out["records"] == 2
    assert out["sha256"] == freeze.file_digest(path)
    assert out["bytes"] == path.stat().st_size
    assert out["exists"] is True


def test_describe_dataset_records_when_the_data_was_generated(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"a": 1}\n')
    out = freeze.describe_dataset(path)
    # the API date a publication has to pin: when these calls actually ran
    assert out["last_modified_utc"].endswith("Z") and out["last_modified_utc"][:2] == "20"


def test_describe_dataset_breaks_down_probe_versions(tmp_path):
    # the appendix file holds an invalid v1 run alongside the valid v2 one; a freeze that
    # reports one record count would overstate what is analysable
    path = tmp_path / "jb.jsonl"
    path.write_text('{"label": "X"}\n'
                    '{"label": "X", "probe_version": 2}\n'
                    '{"label": "X", "probe_version": 2}\n')
    out = freeze.describe_dataset(path)
    assert out["records"] == 3
    assert out["by_probe_version"] == {"1": 1, "2": 2}


def test_describe_dataset_omits_version_breakdown_when_absent(tmp_path):
    path = tmp_path / "plain.jsonl"
    path.write_text('{"a": 1}\n')
    assert "by_probe_version" not in freeze.describe_dataset(path)


def test_describe_dataset_marks_absent_files_instead_of_failing(tmp_path):
    out = freeze.describe_dataset(tmp_path / "missing.jsonl")
    assert out["exists"] is False and out["records"] == 0


def test_sampling_parameters_are_read_from_code_not_retyped():
    params = freeze.sampling_parameters()
    # the values that must be frozen for publication, sourced from the modules that use them
    assert params["cipher_turn_cap"] == 10
    assert params["cipher_replicates"] == 8
    assert params["cipher_explicit_every"] == 3
    assert params["cipher_fewshot_k"] == 3
    assert params["cipher_seed"] == 0
    assert params["temperature"] == 0.0


def test_model_identities_include_provider_and_id():
    models = freeze.model_identities()
    assert models["gpt-5"]["id"] == "gpt-5.4"
    assert models["gpt-5"]["provider"] == "azure_openai"
    assert models["qwen7b-base"]["is_base"] is True


def test_render_is_deterministic_and_lists_every_section():
    payload = {
        "commit": "abc1234", "generated_utc": "2026-07-30T00:00:00Z",
        "datasets": {"data/x.jsonl": {"exists": True, "records": 2, "bytes": 10,
                                     "sha256": "ff"}},
        "models": {"gpt-5": {"id": "gpt-5.4", "provider": "azure_openai",
                             "is_base": False}},
        "sampling": {"temperature": 0.0},
        "deployments": {"gpt-5.4": "2026-03-05"},
        "notes": ["something worth freezing"],
    }
    first = freeze.render(payload)
    assert first == freeze.render(payload)
    for heading in ("Commit", "Datasets", "Model identities", "Sampling", "sha256"):
        assert heading in first
    assert "something worth freezing" in first
