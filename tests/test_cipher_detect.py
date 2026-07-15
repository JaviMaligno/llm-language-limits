from llm_language_limits.ciphers import CIPHERS
from llm_language_limits.oracle import TASK_BANK
from llm_language_limits.cipher_detect import comprehension_action, produced_in_code

BANANA = next(t for t in TASK_BANK if t.id == "banana")


def test_action_plain_reply():
    assert comprehension_action("banana", BANANA, CIPHERS["rot13"]) is True


def test_action_coded_reply():
    coded = CIPHERS["rot13"].encode("banana")   # 'onanan'
    assert comprehension_action(coded, BANANA, CIPHERS["rot13"]) is True


def test_action_wrong():
    assert comprehension_action("apple", BANANA, CIPHERS["rot13"]) is False


def test_produced_in_code_true():
    coded = CIPHERS["rot13"].encode("here is your answer banana")
    assert produced_in_code(coded, CIPHERS["rot13"]) is True


def test_produced_in_code_false_for_plain():
    assert produced_in_code("here is your answer banana", CIPHERS["rot13"]) is False


def test_produced_in_code_false_for_lossy():
    assert produced_in_code("nythng", CIPHERS["disemvowel"]) is False


def test_produced_in_code_oracle_anchor_short_coded_answer():
    # rot13("banana") == "onanan": too short for the englishness path, caught by the oracle anchor
    coded = CIPHERS["rot13"].encode("banana")
    assert produced_in_code(coded, CIPHERS["rot13"], BANANA) is True


def test_produced_in_code_short_without_task_is_false():
    coded = CIPHERS["rot13"].encode("banana")
    assert produced_in_code(coded, CIPHERS["rot13"]) is False


def test_produced_in_code_nonletter_cipher():
    # a non-letter cipher family (letters->digits) is still detected via englishness
    coded = CIPHERS["letters_to_digits"].encode("here is your answer banana")
    assert produced_in_code(coded, CIPHERS["letters_to_digits"]) is True


def test_produced_in_code_plain_with_task_still_false():
    # a correct PLAINTEXT answer must not be counted as "produced in code"
    assert produced_in_code("banana", CIPHERS["rot13"], BANANA) is False
