from llm_language_limits.prompts import build_single_turn, build_multi_turn

def test_single_turn_repeats_text_n_times():
    msgs = build_single_turn("hi", 3)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hi hi hi"

def test_single_turn_custom_separator():
    assert build_single_turn("x", 3, sep="")[0]["content"] == "xxx"

def test_multi_turn_interleaves_user_and_assistant():
    msgs = build_multi_turn("hi", 3, prior_assistant=["a1", "a2"])
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[0]["content"] == "hi" and msgs[-1]["content"] == "hi"
    assert msgs[1]["content"] == "a1" and msgs[3]["content"] == "a2"

def test_multi_turn_first_turn_has_no_history():
    msgs = build_multi_turn("hi", 1, prior_assistant=[])
    assert msgs == [{"role": "user", "content": "hi"}]

def test_multi_turn_rejects_mismatched_history():
    import pytest
    with pytest.raises(ValueError):
        build_multi_turn("hi", 3, prior_assistant=["only-one"])
