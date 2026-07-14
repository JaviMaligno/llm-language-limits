import importlib.util
from pathlib import Path
from unittest.mock import Mock, call


MODULE_PATH = Path(__file__).parents[1] / "experiments/repetition/run_full.py"
SPEC = importlib.util.spec_from_file_location("repetition_run_full", MODULE_PATH)
run_full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_full)


def test_model_registry_labels_are_valid_cli_choices():
    assert "qwen7b-instruct" in run_full.MODEL_REGISTRY
    assert "qwen7b-base" in run_full.MODEL_REGISTRY


def test_full_matrix_runs_separate_mode_specific_grids(monkeypatch, tmp_path):
    run_matrix = Mock()
    monkeypatch.setattr(run_full, "run_matrix", run_matrix)
    client_factory = Mock()
    judge = Mock()
    specs = [Mock()]
    stimuli = [Mock()]
    out = tmp_path / "full.jsonl"

    run_full.run_full_matrix(client_factory, judge, specs, stimuli, out)

    assert run_matrix.call_args_list == [
        call(client_factory, judge, specs, stimuli,
             n_grid=[1, 3, 10, 30, 100, 300, 1000], modes=["single"],
             replicates=3, out_path=out),
        call(client_factory, judge, specs, stimuli,
             n_grid=[1, 3, 10, 30, 100], modes=["multi"],
             replicates=3, out_path=out),
    ]
