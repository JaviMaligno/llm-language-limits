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
