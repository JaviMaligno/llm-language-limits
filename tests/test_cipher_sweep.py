from llm_language_limits.config import MODEL_REGISTRY
from llm_language_limits.clients.base import FakeClient
from llm_language_limits.storage import read_records
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("experiments/ciphers")))
from sweep import run_sweep, cell_key  # noqa: E402

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
