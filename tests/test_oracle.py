import re as _re

from llm_language_limits.ciphers import CIPHERS
from llm_language_limits.oracle import TASK_BANK, normalize, explicit_decode_ok


def test_bank_nonempty_and_checkable():
    assert len(TASK_BANK) >= 6
    for t in TASK_BANK:
        assert t.prompt and callable(t.check) and t.id


def test_banana_task_checks_action():
    t = next(t for t in TASK_BANK if t.id == "banana")
    assert t.check("BANANA") is True
    assert t.check("sure, here it is: banana!") is True
    assert t.check("apple") is False


def test_math_task():
    t = next(t for t in TASK_BANK if t.id == "add_7_5")
    assert t.check("the answer is 12") is True
    assert t.check("13") is False


def test_normalize():
    assert normalize("  Hello,   WORLD! ") == "hello world"


def test_explicit_decode_ok():
    assert explicit_decode_ok("It decodes to: reply with banana", "reply with banana")
    assert not explicit_decode_ok("no idea", "reply with banana")


def test_all_prompts_are_letters_and_spaces_only():
    for t in TASK_BANK:
        assert _re.fullmatch(r"[a-z ]+", t.prompt), (t.id, t.prompt)


def test_prompts_roundtrip_through_all_lossless_ciphers():
    for t in TASK_BANK:
        for name, c in CIPHERS.items():
            if c.lossy:
                continue
            assert c.decode(c.encode(t.prompt)) == t.prompt, (t.id, name)
