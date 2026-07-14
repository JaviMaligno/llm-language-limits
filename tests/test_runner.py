# tests/test_runner.py
from llm_language_limits.clients.base import FakeClient
from llm_language_limits.config import MODEL_REGISTRY
from llm_language_limits.stimuli import Stimulus
from llm_language_limits.runner import run_cell, run_matrix
from llm_language_limits.storage import read_records
import json

SPEC = MODEL_REGISTRY["gpt-5-nano"]
STIM = Stimulus("greeting", "hi", "n")
JUDGE = FakeClient(reply_fn=lambda m: json.dumps(
    {"labels": ["normal"], "confidence": 1.0, "rationale": "ok"}))

def test_run_cell_single_turn_record_shape():
    rec = run_cell(FakeClient(), JUDGE, SPEC, STIM, n=3, mode="single", replicate=0)
    assert rec["model"] == "gpt-5-nano"
    assert rec["n"] == 3 and rec["mode"] == "single"
    assert "length" in rec and "judge_labels" in rec
    assert rec["judge_labels"] == ["normal"]

def test_run_cell_can_defer_judging_without_losing_generation():
    rec = run_cell(FakeClient(), None, SPEC, STIM, n=3, mode="single", replicate=0)
    assert rec["text"]
    assert rec["judge_pending"] is True
    assert rec["judge_labels"] == []
    assert rec["judge_confidence"] is None

def test_run_cell_multiturn_respects_cap():
    rec = run_cell(FakeClient(), JUDGE, SPEC, STIM, n=1000, mode="multi",
                   replicate=0, multiturn_cap=5)
    assert rec["turns_run"] == 5
    assert "self_similarity_last" in rec

def test_run_matrix_writes_and_resumes(tmp_path):
    out = tmp_path / "raw.jsonl"
    run_matrix(lambda s: FakeClient(), JUDGE, [SPEC], [STIM],
               n_grid=[1, 3], modes=["single"], replicates=1, out_path=out)
    first = len(read_records(out))
    assert first == 2
    # second run with resume should add nothing
    run_matrix(lambda s: FakeClient(), JUDGE, [SPEC], [STIM],
               n_grid=[1, 3], modes=["single"], replicates=1, out_path=out)
    assert len(read_records(out)) == first


class _BoomClient:
    def chat(self, messages, system, temperature, max_tokens):
        raise RuntimeError("boom")

def test_run_matrix_skips_failing_cell_without_crashing(tmp_path):
    out = tmp_path / "raw.jsonl"
    run_matrix(lambda s: _BoomClient(), JUDGE, [SPEC], [STIM],
               n_grid=[1], modes=["single"], replicates=1, out_path=out)
    assert read_records(out) == []  # failing cell skipped, no crash, nothing written


def test_run_matrix_concurrent_writes_all_cells(tmp_path):
    out = tmp_path / "raw.jsonl"
    stim2 = Stimulus("insult", "x", "n")
    stimuli = [STIM, stim2]
    n_grid = [1, 3]
    modes = ["single", "multi"]
    replicates = 1
    run_matrix(lambda s: FakeClient(), JUDGE, [SPEC], stimuli,
               n_grid=n_grid, modes=modes, replicates=replicates, out_path=out,
               max_workers=4)
    from llm_language_limits.storage import record_key
    records = read_records(out)
    expected_keys = {
        (SPEC.label, stim.category, n, mode, rep)
        for stim in stimuli
        for n in n_grid
        for mode in modes
        for rep in range(replicates)
    }
    actual_keys = [record_key(r) for r in records]
    assert len(actual_keys) == len(expected_keys)  # no dupes
    assert set(actual_keys) == expected_keys  # no missing


def test_run_matrix_concurrent_skip_failing(tmp_path):
    out = tmp_path / "raw.jsonl"
    run_matrix(lambda s: _BoomClient(), JUDGE, [SPEC], [STIM],
               n_grid=[1], modes=["single"], replicates=1, out_path=out,
               max_workers=4)
    assert read_records(out) == []  # failing cells skipped, no exception propagates
