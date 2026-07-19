import pytest
from llm_language_limits.ciphers import make_ciphers, CIPHERS

LOSSLESS = ["rot13", "random_substitution", "letters_to_digits", "morse",
            "binary", "base64", "reverse_all", "block_permutation"]

@pytest.mark.parametrize("name", LOSSLESS)
def test_roundtrip_lossless(name):
    c = CIPHERS[name]
    for text in ["hello world", "the quick brown fox", "reply with banana"]:
        assert c.decode(c.encode(text)) == text, name

def test_rot13_known():
    assert CIPHERS["rot13"].encode("abc") == "nop"

def test_random_substitution_is_keyed_and_not_identity():
    c = CIPHERS["random_substitution"]
    assert c.encode("the quick brown fox jumps") != "the quick brown fox jumps"
    # deterministic for a given seed
    assert make_ciphers(0)["random_substitution"].encode("hello") == c.encode("hello")
    assert make_ciphers(1)["random_substitution"].encode("hello") != c.encode("hello")

def test_lossy_flagged():
    assert CIPHERS["disemvowel"].lossy is True
    assert CIPHERS["cyrillic_homoglyph"].lossy is False
    assert CIPHERS["rot13"].lossy is False

def test_binary_shape():
    enc = CIPHERS["binary"].encode("A")   # 'A' = 65 = 1000001
    assert enc == "1000001"

def test_cyrillic_homoglyph_preserves_case():
    c = CIPHERS["cyrillic_homoglyph"]
    for s in ["Apple Pie", "COPY that", "Oxygen", "hello"]:
        assert c.decode(c.encode(s)) == s
