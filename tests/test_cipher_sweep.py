from llm_language_limits.config import MODEL_REGISTRY
from llm_language_limits.clients.base import FakeClient
from llm_language_limits.storage import read_records
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("experiments/ciphers")))
from sweep import run_sweep, cell_key  # noqa: E402
from run_pilot import _conversation_tokens  # noqa: E402

SPEC = MODEL_REGISTRY["gpt-5-nano"]


def test_sweep_writes_and_resumes(tmp_path):
    out = tmp_path / "c.jsonl"
    run_sweep(lambda s: FakeClient(reply_fn=lambda m: "banana"),
              [SPEC], ["rot13"], ["pure"], 1, out, turn_cap=3)
    n1 = len(read_records(out))
    assert n1 == 1
    run_sweep(lambda s: FakeClient(reply_fn=lambda m: "banana"),
              [SPEC], ["rot13"], ["pure"], 1, out, turn_cap=3)
    assert len(read_records(out)) == n1  # resume skips the done cell


def test_cell_key():
    assert cell_key({"model": "m", "cipher": "rot13", "protocol": "pure",
                     "replicate": 0}) == ("m", "rot13", "pure", 0)


def test_run_sweep_retries_then_succeeds(tmp_path, monkeypatch):
    import sweep as sweep_mod
    monkeypatch.setattr(sweep_mod.time, "sleep", lambda *_: None)  # no real backoff
    calls = {"n": 0}

    def flaky_reply(messages):
        calls["n"] += 1
        if calls["n"] <= 2:            # fail the first two chat calls, then succeed
            raise RuntimeError("transient 429")
        return "banana"

    out = tmp_path / "r.jsonl"
    run_sweep(lambda s: FakeClient(reply_fn=flaky_reply),
              [SPEC], ["rot13"], ["pure"], 1, out, turn_cap=1)
    assert len(read_records(out)) == 1  # cell eventually recorded despite transient failures


def test_run_sweep_skips_permanently_failing_cell(tmp_path, monkeypatch):
    import sweep as sweep_mod
    monkeypatch.setattr(sweep_mod.time, "sleep", lambda *_: None)

    def always_fail(messages):
        raise RuntimeError("nope")

    out = tmp_path / "f.jsonl"
    run_sweep(lambda s: FakeClient(reply_fn=always_fail),
              [SPEC], ["rot13"], ["pure"], 1, out, turn_cap=1)
    assert read_records(out) == []      # exhausted retries -> no record, but no crash


def test_conversation_tokens_grow_superlinearly_in_input():
    in1, out1 = _conversation_tokens(1)
    in15, out15 = _conversation_tokens(15)
    assert out15 == 15 * out1           # output scales linearly with turns
    assert in15 > 15 * in1              # input grows faster (context re-sent each turn)
