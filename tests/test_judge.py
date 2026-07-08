import json
from llm_language_limits.clients.base import FakeClient
from llm_language_limits.judge import (
    JudgeVerdict, build_judge_prompt, judge_response, RUBRIC,
)

def test_prompt_is_blind_to_metadata():
    msgs = build_judge_prompt("some response")
    joined = " ".join(m["content"] for m in msgs)
    assert "some response" in joined
    assert "N=" not in joined and "model" not in joined.lower()

def test_rubric_defines_all_labels():
    for label in ("meta_complaint", "refusal", "degeneration_loop",
                  "glitch_incoherence", "character_break", "divergence"):
        assert label in RUBRIC

def test_judge_parses_structured_output():
    payload = json.dumps({"labels": ["meta_complaint", "refusal"],
                          "confidence": 0.8, "rationale": "asks why repeating"})
    client = FakeClient(reply_fn=lambda msgs: payload)
    v = judge_response(client, "why do you keep saying that?")
    assert isinstance(v, JudgeVerdict)
    assert set(v.labels) == {"meta_complaint", "refusal"}
    assert v.confidence == 0.8

def test_judge_handles_bad_json_gracefully():
    client = FakeClient(reply_fn=lambda msgs: "not json")
    v = judge_response(client, "x")
    assert v.labels == ["normal"]
    assert v.rationale == "parse_error"
