from llm_language_limits.ciphers import CIPHERS
from llm_language_limits.oracle import TASK_BANK
from llm_language_limits.clients.base import FakeClient
from llm_language_limits.cipher_runner import run_conversation, build_fewshot_preamble

ROT13 = CIPHERS["rot13"]


def test_fewshot_preamble_shape():
    msgs = build_fewshot_preamble(ROT13, TASK_BANK, 2)
    assert len(msgs) == 4 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"


def test_run_conversation_all_correct():
    # a fake model that always answers correctly IN plaintext, regardless of the coded prompt
    client = FakeClient(reply_fn=lambda m: "banana 12 paris yes red one two three cold dog")
    rec = run_conversation(client, ROT13, TASK_BANK, "pure", turn_cap=6)
    assert rec["first_action_turn"] == 1          # correct from turn 1
    assert rec["n_turns"] == 6
    assert "first_production_turn" in rec


def test_run_conversation_never_comprehends():
    client = FakeClient(reply_fn=lambda m: "I do not understand")
    rec = run_conversation(client, ROT13, TASK_BANK, "pure", turn_cap=4)
    assert rec["first_action_turn"] is None
    assert rec["first_production_turn"] is None


def test_production_detected():
    # fake model replies in ROT13 code -> production should fire
    client = FakeClient(reply_fn=lambda m: ROT13.encode("your answer is banana indeed"))
    rec = run_conversation(client, ROT13, TASK_BANK, "pure", turn_cap=3)
    assert rec["first_production_turn"] == 1
    assert rec["production_consistency"] == 1.0


def test_short_coded_production_detected_via_task():
    # model replies with ONLY the coded answer word (too short for the englishness path);
    # detected as production only because run_conversation passes task= to produced_in_code.
    # turn 1's task is TASK_BANK[0]; reply with that task's coded answer.
    first_task = TASK_BANK[0]
    # find a plaintext answer the checker accepts, then code it
    answer = "banana" if first_task.id == "banana" else first_task.prompt
    client = FakeClient(reply_fn=lambda m: ROT13.encode(answer))
    rec = run_conversation(client, ROT13, TASK_BANK, "pure", turn_cap=2)
    assert rec["first_production_turn"] == 1


def test_escalating_no_consecutive_user_messages():
    # the fake asserts alternation on every history it receives;
    # before the fix, the hint2 turn produced two consecutive user messages.
    def reply_fn(messages):
        roles = [m["role"] for m in messages]
        for a, b in zip(roles, roles[1:]):
            assert not (a == b == "user"), f"consecutive user messages: {roles}"
        return "ok"
    client = FakeClient(reply_fn=reply_fn)
    run_conversation(client, ROT13, TASK_BANK, "escalating", turn_cap=6)


def test_first_explicit_turn_anchors_on_task_prompt():
    # a fake that replies with ONLY the decoded task prompt (not the meta-instruction)
    # must still register an explicit-decode success on the marked turn (default explicit_every=3).
    third = TASK_BANK[2]  # turn 3's task (rotation is (turn-1) % len)
    client = FakeClient(reply_fn=lambda m: third.prompt)
    rec = run_conversation(client, ROT13, TASK_BANK, "pure", turn_cap=3)
    assert rec["first_explicit_turn"] == 3


def test_fewshot_preamble_has_plaintext_key():
    # the user side must be readable plaintext (the task prompt), the assistant side its encoding
    msgs = build_fewshot_preamble(ROT13, TASK_BANK, 2)
    assert len(msgs) == 4
    for i in range(0, 4, 2):
        plain = msgs[i]["content"]
        coded = msgs[i + 1]["content"]
        assert plain in [t.prompt for t in TASK_BANK]      # user side is real plaintext
        assert coded == ROT13.encode(plain)                # assistant side is that text, coded
        assert coded != plain                              # and they actually differ (a real key)


def test_record_carries_token_totals_and_stops():
    client = FakeClient(reply_fn=lambda m: "banana")
    rec = run_conversation(client, ROT13, TASK_BANK, "pure", turn_cap=3)
    assert isinstance(rec["input_tokens"], int) and rec["input_tokens"] > 0
    assert isinstance(rec["output_tokens"], int) and rec["output_tokens"] > 0
    assert len(rec["stop_signals"]) == 3
