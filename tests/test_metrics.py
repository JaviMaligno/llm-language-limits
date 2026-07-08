import math
from llm_language_limits.metrics import (
    response_length_chars, repetition_ratio, token_entropy,
    is_refusal, self_similarity,
)

def test_length():
    assert response_length_chars("abc") == 3

def test_repetition_ratio_all_same():
    assert repetition_ratio("a a a a") == 0.75  # 1 unique / 4 total

def test_repetition_ratio_all_unique():
    assert repetition_ratio("a b c d") == 0.0

def test_repetition_ratio_empty():
    assert repetition_ratio("") == 0.0

def test_entropy_uniform_two_tokens():
    assert math.isclose(token_entropy("a b"), 1.0, rel_tol=1e-9)

def test_entropy_single_token_is_zero():
    assert token_entropy("a a a") == 0.0

def test_refusal_detects_common_phrases():
    assert is_refusal("I can't help with that.")
    assert is_refusal("Why do you keep repeating yourself?")
    assert not is_refusal("Hello! How can I help?")

def test_self_similarity_identical():
    assert self_similarity("a b c", "a b c") == 1.0

def test_self_similarity_disjoint():
    assert self_similarity("a b", "c d") == 0.0
