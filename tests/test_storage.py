# tests/test_storage.py
from llm_language_limits.storage import (
    append_record, read_records, to_parquet, record_key,
)

def test_append_and_read_roundtrip(tmp_path):
    p = tmp_path / "sub" / "raw.jsonl"
    append_record(p, {"model": "m", "n": 1})
    append_record(p, {"model": "m", "n": 3})
    recs = read_records(p)
    assert len(recs) == 2 and recs[1]["n"] == 3

def test_record_key():
    r = {"model": "m", "category": "greeting", "n": 10,
         "mode": "single", "replicate": 2}
    assert record_key(r) == ("m", "greeting", 10, "single", 2)

def test_to_parquet(tmp_path):
    p = tmp_path / "raw.jsonl"
    append_record(p, {"model": "m", "n": 1, "length": 5})
    pq = tmp_path / "agg.parquet"
    to_parquet(p, pq)
    import pandas as pd
    df = pd.read_parquet(pq)
    assert list(df["n"]) == [1]
