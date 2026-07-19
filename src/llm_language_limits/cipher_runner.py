from __future__ import annotations
from .config import SYSTEM_PROMPT
from .ciphers import Cipher
from .oracle import Task
from .cipher_detect import comprehension_action, produced_in_code, explicit_decode

Message = dict


def _stop_signal(raw):
    """Best-effort per-turn stop/finish reason from a provider's raw payload."""
    if not isinstance(raw, dict):
        return None
    return raw.get("stop_reason") or raw.get("finish_reason")


def build_fewshot_preamble(cipher: Cipher, tasks: list[Task], k: int) -> list[Message]:
    # A genuine Rosetta stone: the user side is PLAINTEXT, the assistant side is that
    # same text ENCODED. This shows a real plain<->coded pair (a decoding key) AND
    # demonstrates the assistant replying in code (induces production). Without the
    # plaintext side the few-shot protocol would be indistinguishable from pure inference.
    msgs: list[Message] = []
    for t in tasks[:k]:
        msgs.append({"role": "user", "content": t.prompt})
        msgs.append({"role": "assistant", "content": cipher.encode(t.prompt)})
    return msgs


def run_conversation(client, cipher: Cipher, tasks: list[Task], protocol: str, *,
                      turn_cap: int, explicit_every: int = 3, fewshot_k: int = 3,
                      temperature: float = 0.0, max_tokens: int = 256) -> dict:
    history: list[Message] = []
    if protocol == "fewshot":
        history += build_fewshot_preamble(cipher, tasks, fewshot_k)

    hint1 = max(1, turn_cap // 3)
    hint2 = max(2, 2 * turn_cap // 3)

    first_action = first_explicit = first_production = None
    prod_flags: list[bool] = []
    total_input = total_output = 0
    stop_signals: list = []

    for turn in range(1, turn_cap + 1):
        if protocol == "escalating" and turn == hint1:
            history += build_fewshot_preamble(cipher, tasks, 1)
        hint_text = ""
        if protocol == "escalating" and turn == hint2:
            hint_text = f"(the code is {cipher.name}; reply in the same code)\n"

        task = tasks[(turn - 1) % len(tasks)]
        ask_decode = (turn % explicit_every == 0)
        prompt_plain = task.prompt + ("  (also: write this message in plain English)" if ask_decode else "")
        history.append({"role": "user", "content": hint_text + cipher.encode(prompt_plain)})

        res = client.chat(history, SYSTEM_PROMPT, temperature, max_tokens)
        reply = res.text
        history.append({"role": "assistant", "content": reply})
        total_input += res.input_tokens
        total_output += res.output_tokens
        stop_signals.append(_stop_signal(res.raw))

        if first_action is None and comprehension_action(reply, task, cipher):
            first_action = turn
        if ask_decode and first_explicit is None and explicit_decode(reply, task.prompt):
            first_explicit = turn
        # CRITICAL OVERRIDE (task 4 brief): pass task= so the oracle-anchor path in
        # produced_in_code can fire on short coded answers (e.g. rot13("banana") == "onanan",
        # too short for the englishness-delta path alone).
        prod = produced_in_code(reply, cipher, task=task)
        prod_flags.append(prod)
        if first_production is None and prod:
            first_production = turn

    if first_production is not None:
        tail = prod_flags[first_production - 1:]
        consistency = sum(tail) / len(tail)
    else:
        consistency = 0.0

    return {
        "cipher": cipher.name, "protocol": protocol, "n_turns": turn_cap,
        "first_action_turn": first_action, "first_explicit_turn": first_explicit,
        "first_production_turn": first_production,
        "production_consistency": consistency,
        "input_tokens": total_input, "output_tokens": total_output,
        "stop_signals": stop_signals,
    }
