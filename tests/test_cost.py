from llm_language_limits.cost import _price_for, estimate_cost


def test_price_for_matches_most_specific_prefix():
    assert _price_for("gpt-5-nano") == (0.05, 0.40)
    assert _price_for("gpt-5-mini") == (0.25, 2.0)
    assert _price_for("gpt-5") == (1.25, 10.0)
    assert _price_for("claude-sonnet") == (3.0, 15.0)
    assert _price_for("qwen7b-base") == (0.0, 0.0)
    assert _price_for("unknown-model") == (0.0, 0.0)


def test_estimate_cost_uses_nano_price_for_nano():
    recs = [{"model": "gpt-5-nano", "input_tokens": 1_000_000, "output_tokens": 1_000_000}]
    est = estimate_cost(recs)
    assert abs(est["gpt-5-nano"] - (0.05 + 0.40)) < 1e-9
