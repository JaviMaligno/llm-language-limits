# Ciphers Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how fast LLMs infer a novel encoding (comprehension) and start replying in it (production), across a difficulty ladder of ciphers and three exposure protocols, using a deterministic programmatic oracle.

**Architecture:** New pure modules `ciphers.py` (encode/decode registry) and `oracle.py` (verifiable-task bank) + detection functions, driven by a new multi-turn `cipher_conversation` runner that reuses the part-1 clients/storage. Each experiment "cell" is one (model × cipher × protocol × replicate) conversation; per-turn evaluation yields comprehension/production latencies. Entrypoints run staged (smoke → pilot → full), reusing the part-1 rate limiter, resume, and cloud sweep.

**Tech Stack:** Python 3.12, `uv`, `pytest`, existing `llm_language_limits` package (clients, storage, ratelimit, config), `pandas`/`matplotlib` for analysis.

## Global Constraints

- **Repo:** `~/Documents/repos/llm-language-limits`; new code under `src/llm_language_limits/` (shared modules) and `experiments/ciphers/` (manifests, entrypoints, analysis). Reuse part-1's `clients/`, `storage.py`, `ratelimit.py`, `config.py`.
- **Spec:** `docs/superpowers/specs/2026-07-15-ciphers-experiment-design.md`.
- **Deterministic oracle:** cipher encode/decode and task checks are pure/deterministic — NO LLM judge for core comprehension-action or production metrics. (A light LLM judge is allowed ONLY as a fallback for fuzzy explicit-decode matching, not for the action metric.)
- **Contamination guard:** the cipher set MUST include at least two novel/keyed ciphers (random substitution, keyed block permutation) whose key is generated per-run, so results separate real inference from memorized well-known schemes.
- **Two comprehension signals:** (a) correct ACTION on the decoded task (language-agnostic), (b) correct EXPLICIT DECODE on marked turns.
- **Production:** decode the model's reply with the inverse cipher; valid ⇒ producing in-code. Track first-valid turn + consistency.
- **Staged:** cipher round-trip unit tests → smoke → pilot (prune roster/ciphers/protocols, calibrate turn cap) → full. Resume + per-provider cost estimate before each live run.
- **Commit trailer:** end commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**Reference for existing interfaces** (do not reimplement): `clients.get_client(spec)`, `clients.base.ChatResult(text, input_tokens, output_tokens)`, `config.MODEL_REGISTRY`, `storage.append_record/read_records`, `ratelimit.throttle(provider)`.

## File Structure

```
src/llm_language_limits/
├── ciphers.py          # Cipher dataclass, ~10 codecs, CIPHERS registry, is_lossy
├── oracle.py           # Task dataclass, TASK_BANK, explicit_decode_ok, normalize
├── cipher_detect.py    # per-reply detection: comprehension_action, produced_in_code, explicit_decode_ok
├── cipher_runner.py    # run_conversation() (multi-turn, per-protocol) + latency reduction
experiments/ciphers/
├── config.py           # CIPHER_SET, PROTOCOLS, TURN_CAP, FEWSHOT_KS, pilot/full tiers
├── run_smoke.py        # 1 cheap model × 1-2 ciphers × short cap
├── run_pilot.py        # broad roster × subset → prune
├── run_full.py         # tuned matrix
└── analysis.ipynb      # latency curves, difficulty ranking, comp-vs-prod
tests/
├── test_ciphers.py
├── test_oracle.py
├── test_cipher_detect.py
└── test_cipher_runner.py
```

---

### Task 1: Cipher codecs (`ciphers.py`)

**Files:**
- Create: `src/llm_language_limits/ciphers.py`
- Test: `tests/test_ciphers.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Cipher: name: str; encode: Callable[[str], str]; decode: Callable[[str], str]; lossy: bool = False`
  - `def make_ciphers(seed: int = 0) -> dict[str, Cipher]` — builds the registry; keyed ciphers (random substitution, block permutation) derive their key deterministically from `seed`.
  - `CIPHERS: dict[str, Cipher] = make_ciphers(0)` — default registry.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ciphers.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ciphers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_language_limits.ciphers'`.

- [ ] **Step 3: Write `ciphers.py`**

```python
# src/llm_language_limits/ciphers.py
from __future__ import annotations
import base64 as _b64
import random
import string
from dataclasses import dataclass
from typing import Callable

_LOWER = string.ascii_lowercase


@dataclass(frozen=True)
class Cipher:
    name: str
    encode: Callable[[str], str]
    decode: Callable[[str], str]
    lossy: bool = False


def _rot(text: str, n: int) -> str:
    out = []
    for ch in text:
        if ch.islower():
            out.append(chr((ord(ch) - 97 + n) % 26 + 97))
        elif ch.isupper():
            out.append(chr((ord(ch) - 65 + n) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def _sub_maps(perm: str):
    fwd = {a: b for a, b in zip(_LOWER, perm)}
    fwd.update({a.upper(): b.upper() for a, b in zip(_LOWER, perm)})
    inv = {v: k for k, v in fwd.items()}
    enc = lambda t: "".join(fwd.get(c, c) for c in t)
    dec = lambda t: "".join(inv.get(c, c) for c in t)
    return enc, dec


_MORSE = {
    "a": ".-", "b": "-...", "c": "-.-.", "d": "-..", "e": ".", "f": "..-.",
    "g": "--.", "h": "....", "i": "..", "j": ".---", "k": "-.-", "l": ".-..",
    "m": "--", "n": "-.", "o": "---", "p": ".--.", "q": "--.-", "r": ".-.",
    "s": "...", "t": "-", "u": "..-", "v": "...-", "w": ".--", "x": "-..-",
    "y": "-.--", "z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
}
_MORSE_INV = {v: k for k, v in _MORSE.items()}


def _morse_enc(t: str) -> str:
    words = t.lower().split(" ")
    return " / ".join(" ".join(_MORSE.get(c, c) for c in w) for w in words)


def _morse_dec(t: str) -> str:
    words = t.split(" / ")
    return " ".join("".join(_MORSE_INV.get(sym, sym) for sym in w.split(" ")) for w in words)


def _digits_enc(t: str) -> str:
    # letters -> 1..26 separated by '-', spaces -> '/', other chars kept
    out = []
    for c in t:
        if c.isalpha():
            out.append(str(ord(c.lower()) - 96))
        elif c == " ":
            out.append("/")
        else:
            out.append(c)
    return "-".join(out)


def _digits_dec(t: str) -> str:
    out = []
    for tok in t.split("-"):
        if tok.isdigit():
            out.append(chr(int(tok) + 96))
        elif tok == "/":
            out.append(" ")
        else:
            out.append(tok)
    return "".join(out)


def _binary_enc(t: str) -> str:
    return " ".join(format(ord(c), "b") for c in t)


def _binary_dec(t: str) -> str:
    return "".join(chr(int(b, 2)) for b in t.split(" "))


def _b64_enc(t: str) -> str:
    return _b64.b64encode(t.encode()).decode()


def _b64_dec(t: str) -> str:
    return _b64.b64decode(t.encode()).decode()


def _block_perm(perm: list[int]):
    k = len(perm)
    inv = [perm.index(i) for i in range(k)]

    def apply(order, t):
        out = []
        for i in range(0, len(t), k):
            block = list(t[i:i + k])
            if len(block) == k:
                out.append("".join(block[order[j]] for j in range(k)))
            else:
                out.append("".join(block))  # short trailing block unchanged
        return "".join(out)

    return (lambda t: apply(perm, t)), (lambda t: apply(inv, t))


_CYR = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у"}
_CYR_INV = {v: k for k, v in _CYR.items()}
_VOWELS = set("aeiouAEIOU")


def make_ciphers(seed: int = 0) -> dict[str, Cipher]:
    rng = random.Random(seed)
    perm_letters = list(_LOWER)
    rng.shuffle(perm_letters)
    sub_enc, sub_dec = _sub_maps("".join(perm_letters))

    block = list(range(5))
    rng.shuffle(block)
    blk_enc, blk_dec = _block_perm(block)

    cyr_enc = lambda t: "".join(_CYR.get(c.lower(), c) for c in t)
    cyr_dec = lambda t: "".join(_CYR_INV.get(c, c) for c in t)

    ciphers = [
        Cipher("rot13", lambda t: _rot(t, 13), lambda t: _rot(t, 13)),
        Cipher("random_substitution", sub_enc, sub_dec),
        Cipher("letters_to_digits", _digits_enc, _digits_dec),
        Cipher("morse", _morse_enc, _morse_dec),
        Cipher("binary", _binary_enc, _binary_dec),
        Cipher("base64", _b64_enc, _b64_dec),
        Cipher("reverse_all", lambda t: t[::-1], lambda t: t[::-1]),
        Cipher("block_permutation", blk_enc, blk_dec),
        Cipher("cyrillic_homoglyph", cyr_enc, cyr_dec, lossy=False),
        Cipher("disemvowel", lambda t: "".join(c for c in t if c not in _VOWELS),
               lambda t: t, lossy=True),
    ]
    return {c.name: c for c in ciphers}


CIPHERS: dict[str, Cipher] = make_ciphers(0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ciphers.py -v`
Expected: PASS. (Note: `letters_to_digits`, `morse`, `binary` lowercase or reshape text, so their round-trip tests use lowercase inputs — the provided test strings are already lowercase.)

- [ ] **Step 5: Commit**

```bash
git add src/llm_language_limits/ciphers.py tests/test_ciphers.py
git commit -m "feat: cipher codec registry (10 ciphers, keyed novel ones)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Verifiable-task oracle (`oracle.py`)

**Files:**
- Create: `src/llm_language_limits/oracle.py`
- Test: `tests/test_oracle.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass(frozen=True) class Task: id: str; prompt: str; check: Callable[[str], bool]`
  - `TASK_BANK: list[Task]` — verifiable tasks (plaintext prompts + checkers over a response string).
  - `def normalize(s: str) -> str` — lowercase, strip, collapse whitespace/punct for matching.
  - `def explicit_decode_ok(model_output: str, plaintext: str) -> bool` — True if the normalized plaintext is contained in the normalized model output.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_oracle.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `oracle.py`**

```python
# src/llm_language_limits/oracle.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() and \
        " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()) or ""


def _contains(word: str) -> Callable[[str], bool]:
    w = normalize(word)
    return lambda resp: w in normalize(resp)


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    check: Callable[[str], bool]


TASK_BANK: list[Task] = [
    Task("banana", "reply with the single word banana", _contains("banana")),
    Task("add_7_5", "what is 7 plus 5", _contains("12")),
    Task("capital_france", "name the capital of france", _contains("paris")),
    Task("sky_blue", "reply yes or no is the sky blue", _contains("yes")),
    Task("color", "name a primary color",
         lambda r: any(c in normalize(r) for c in ("red", "blue", "yellow"))),
    Task("count_three", "count from one to three",
         lambda r: all(n in normalize(r) for n in ("one", "two", "three"))),
    Task("opposite_hot", "what is the opposite of hot", _contains("cold")),
    Task("animal_bark", "what animal says woof", _contains("dog")),
]


def explicit_decode_ok(model_output: str, plaintext: str) -> bool:
    return normalize(plaintext) in normalize(model_output)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_oracle.py -v`
Expected: PASS. (Simplify `normalize` if the ternary reads poorly — it must return the collapsed string or `""`.)

- [ ] **Step 5: Commit**

```bash
git add src/llm_language_limits/oracle.py tests/test_oracle.py
git commit -m "feat: verifiable-task oracle bank + normalization

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Per-reply detection (`cipher_detect.py`)

**Files:**
- Create: `src/llm_language_limits/cipher_detect.py`
- Test: `tests/test_cipher_detect.py`

**Interfaces:**
- Consumes: `ciphers.Cipher` (Task 1), `oracle.Task` / `normalize` (Task 2).
- Produces:
  - `def comprehension_action(reply: str, task: Task, cipher: Cipher) -> bool` — decode the reply with the cipher's inverse if it round-trips to sensible text, then oracle-check BOTH the raw reply and the decoded reply; True if either passes. (Model may answer in plain or in code.)
  - `def produced_in_code(reply: str, cipher: Cipher, *, min_chars: int = 8) -> bool` — True if decoding the reply yields text that is meaningfully different from the raw reply AND looks like natural language (mostly ascii letters/spaces), i.e. the reply was itself encoded. Always False for lossy ciphers (can't verify).
  - `def explicit_decode(reply: str, plaintext: str) -> bool` — thin re-export of `oracle.explicit_decode_ok` for symmetry.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cipher_detect.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cipher_detect.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `cipher_detect.py`**

```python
# src/llm_language_limits/cipher_detect.py
from __future__ import annotations
import re
from .ciphers import Cipher
from .oracle import Task, explicit_decode_ok

_LETTERS = re.compile(r"[a-zA-Z]")


def _looks_like_text(s: str) -> bool:
    if len(s.strip()) < 3:
        return False
    letters = len(_LETTERS.findall(s))
    return letters / max(1, len(s)) > 0.5


def _safe_decode(reply: str, cipher: Cipher) -> str | None:
    try:
        return cipher.decode(reply)
    except Exception:
        return None


def comprehension_action(reply: str, task: Task, cipher: Cipher) -> bool:
    if task.check(reply):
        return True
    dec = _safe_decode(reply, cipher)
    return bool(dec) and task.check(dec)


def produced_in_code(reply: str, cipher: Cipher, *, min_chars: int = 8) -> bool:
    if cipher.lossy:
        return False
    if len(reply.strip()) < min_chars:
        return False
    if _looks_like_text(reply):        # already plaintext -> not produced in code
        return False
    dec = _safe_decode(reply, cipher)
    return bool(dec) and _looks_like_text(dec) and dec.strip() != reply.strip()


def explicit_decode(reply: str, plaintext: str) -> bool:
    return explicit_decode_ok(reply, plaintext)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cipher_detect.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_language_limits/cipher_detect.py tests/test_cipher_detect.py
git commit -m "feat: per-reply comprehension/production detection

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Protocol message builders + conversation runner (`cipher_runner.py`)

**Files:**
- Create: `src/llm_language_limits/cipher_runner.py`
- Test: `tests/test_cipher_runner.py`

**Interfaces:**
- Consumes: `ciphers.Cipher`, `oracle.Task`, `cipher_detect.*`, `clients.base.ModelClient`/`FakeClient` (part-1), `config.SYSTEM_PROMPT`.
- Produces:
  - `def build_fewshot_preamble(cipher, tasks, k) -> list[Message]` — k user/assistant example pairs showing plaintext↔coded (for the few-shot protocol).
  - `def run_conversation(client, cipher, tasks, protocol, *, turn_cap, explicit_every=3, fewshot_k=3, temperature=0.0, max_tokens=256) -> dict` — runs a multi-turn conversation (one coded task per turn, history grows), evaluating each turn; returns a flat record with per-turn results and reduced latencies: `first_action_turn`, `first_explicit_turn`, `first_production_turn`, `production_consistency`, `n_turns`, plus `model`/`cipher`/`protocol` fields (filled by the caller).
  - protocol ∈ `{"pure", "fewshot", "escalating"}`. `"pure"`: no scaffolding. `"fewshot"`: prepend `build_fewshot_preamble`. `"escalating"`: start pure, inject a hint message at turn `turn_cap//3` (a fewshot pair) and `2*turn_cap//3` (the cipher name + "reply in the same code").
  - `first_*_turn` is the 1-indexed turn of the first success, or `None` if never.
  - `production_consistency` = fraction of turns at/after `first_production_turn` that are `produced_in_code`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cipher_runner.py
from llm_language_limits.ciphers import CIPHERS
from llm_language_limits.oracle import TASK_BANK
from llm_language_limits.clients.base import FakeClient
from llm_language_limits.cipher_runner import run_conversation, build_fewshot_preamble

ROT13 = CIPHERS["rot13"]

def test_fewshot_preamble_shape():
    msgs = build_fewshot_preamble(ROT13, TASK_BANK, 2)
    assert len(msgs) == 4 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"

def test_run_conversation_all_correct():
    # a fake model that always decodes and answers correctly IN plaintext
    def reply(messages):
        coded = messages[-1]["content"]
        return ROT13.decode(coded)  # echo the decoded task text -> contains the answer words? no
    # Use a smarter fake: answer 'banana ... 12 ... paris ...' so oracle passes broadly
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cipher_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `cipher_runner.py`**

```python
# src/llm_language_limits/cipher_runner.py
from __future__ import annotations
from .config import SYSTEM_PROMPT
from .ciphers import Cipher
from .oracle import Task
from .cipher_detect import comprehension_action, produced_in_code, explicit_decode

Message = dict


def build_fewshot_preamble(cipher: Cipher, tasks: list[Task], k: int) -> list[Message]:
    msgs: list[Message] = []
    for t in tasks[:k]:
        msgs.append({"role": "user", "content": cipher.encode(t.prompt)})
        msgs.append({"role": "assistant", "content": cipher.encode("ok: " + t.prompt)})
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

    for turn in range(1, turn_cap + 1):
        if protocol == "escalating" and turn == hint1:
            history += build_fewshot_preamble(cipher, tasks, 1)
        if protocol == "escalating" and turn == hint2:
            history.append({"role": "user",
                            "content": f"(the code is {cipher.name}; reply in the same code)"})

        task = tasks[(turn - 1) % len(tasks)]
        ask_decode = (turn % explicit_every == 0)
        prompt_plain = task.prompt + ("  (also: write this message in plain English)" if ask_decode else "")
        history.append({"role": "user", "content": cipher.encode(prompt_plain)})

        res = client.chat(history, SYSTEM_PROMPT, temperature, max_tokens)
        reply = res.text
        history.append({"role": "assistant", "content": reply})

        if first_action is None and comprehension_action(reply, task, cipher):
            first_action = turn
        if ask_decode and first_explicit is None and explicit_decode(reply, prompt_plain):
            first_explicit = turn
        prod = produced_in_code(reply, cipher)
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
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cipher_runner.py -v`
Expected: PASS. If `test_run_conversation_all_correct`'s reply string doesn't satisfy every task's checker, that's fine — the test only asserts `first_action_turn == 1`, which holds because turn 1's task (`banana`) is satisfied by the reply containing "banana".

- [ ] **Step 5: Commit**

```bash
git add src/llm_language_limits/cipher_runner.py tests/test_cipher_runner.py
git commit -m "feat: multi-turn cipher conversation runner + protocols

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Experiment config + sweep driver (`experiments/ciphers/config.py`, `sweep.py`)

**Files:**
- Create: `experiments/ciphers/config.py`
- Create: `experiments/ciphers/sweep.py`
- Test: `tests/test_cipher_sweep.py`

**Interfaces:**
- Consumes: `config.MODEL_REGISTRY`, `clients.get_client`, `ciphers.CIPHERS`, `oracle.TASK_BANK`, `cipher_runner.run_conversation`, `storage.append_record/read_records/record_key`.
- Produces:
  - `experiments/ciphers/config.py`: `CIPHER_SET: list[str]`, `PROTOCOLS = ["pure", "fewshot", "escalating"]`, `TURN_CAP: int = 15`, `REPLICATES = 3`, `models_for(tier)` (smoke→[gpt-5-nano], pilot→broad roster, full→tuned).
  - `experiments/ciphers/sweep.py`: `run_sweep(client_factory, specs, cipher_names, protocols, replicates, out_path, *, turn_cap, resume=True, max_workers=4)` — nested sweep over (spec × cipher × protocol × replicate); each cell calls `run_conversation`, augments the record with `model`/`replicate`, appends via storage. Reuses part-1's `ThreadPoolExecutor` + resume pattern (mirror `runner.run_matrix`: build done-set from `record_key`, submit, append in main thread). `record_key` for cipher cells = `(model, cipher, protocol, replicate)` — add a small local key function since part-1's `record_key` expects `category/n/mode`.

- [ ] **Step 1: Write the failing test** (drives the sweep + resume with a FakeClient)

```python
# tests/test_cipher_sweep.py
from llm_language_limits.config import MODEL_REGISTRY
from llm_language_limits.clients.base import FakeClient
from llm_language_limits.storage import read_records
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("experiments/ciphers")))
from sweep import run_sweep, cell_key  # noqa: E402

SPEC = MODEL_REGISTRY["gpt-5-nano"]

def test_sweep_writes_and_resumes(tmp_path):
    out = tmp_path / "c.jsonl"
    run_sweep(lambda s: FakeClient(reply_fn=lambda m: "banana"),
              [SPEC], ["rot13"], ["pure"], 1, out, turn_cap=3)
    n1 = len(read_records(out))
    assert n1 == 1
    run_sweep(lambda s: FakeClient(reply_fn=lambda m: "banana"),
              [SPEC], ["rot13"], ["pure"], 1, out, turn_cap=3)
    assert len(read_records(out)) == n1  # resume skips the done cell

def test_cell_key():
    assert cell_key({"model": "m", "cipher": "rot13", "protocol": "pure",
                     "replicate": 0}) == ("m", "rot13", "pure", 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cipher_sweep.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'sweep'` until created).

- [ ] **Step 3: Write `config.py` and `sweep.py`**

```python
# experiments/ciphers/config.py
from llm_language_limits.config import MODEL_REGISTRY

CIPHER_SET = ["rot13", "random_substitution", "letters_to_digits", "morse",
              "binary", "base64", "reverse_all", "block_permutation",
              "cyrillic_homoglyph", "disemvowel"]
PROTOCOLS = ["pure", "fewshot", "escalating"]
TURN_CAP = 15
REPLICATES = 3


def models_for(tier: str):
    if tier == "smoke":
        return [MODEL_REGISTRY["gpt-5-nano"]]
    if tier == "pilot":
        return [MODEL_REGISTRY[k] for k in
                ("claude-opus", "claude-sonnet", "gpt-5",
                 "qwen7b-instruct", "qwen7b-base")]
    if tier == "full":
        # tuned after pilot; default to the pilot roster
        return [MODEL_REGISTRY[k] for k in
                ("claude-opus", "claude-sonnet", "gpt-5",
                 "qwen7b-instruct", "qwen7b-base")]
    raise ValueError(tier)
```

```python
# experiments/ciphers/sweep.py
from __future__ import annotations
import concurrent.futures
from llm_language_limits.ciphers import CIPHERS
from llm_language_limits.oracle import TASK_BANK
from llm_language_limits.cipher_runner import run_conversation
from llm_language_limits.storage import append_record, read_records


def cell_key(rec: dict) -> tuple:
    return (rec["model"], rec["cipher"], rec["protocol"], rec["replicate"])


def _run_cell(client, spec, cipher_name, protocol, replicate, turn_cap):
    rec = run_conversation(client, CIPHERS[cipher_name], TASK_BANK, protocol,
                           turn_cap=turn_cap)
    rec["model"] = spec.label
    rec["replicate"] = replicate
    return rec


def run_sweep(client_factory, specs, cipher_names, protocols, replicates, out_path,
              *, turn_cap, resume=True, max_workers=4):
    done = {cell_key(r) for r in read_records(out_path)} if resume else set()
    clients = {s.label: client_factory(s) for s in specs}
    pending = []
    for spec in specs:
        for cn in cipher_names:
            for proto in protocols:
                for rep in range(replicates):
                    if (spec.label, cn, proto, rep) in done:
                        continue
                    pending.append((spec, cn, proto, rep))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_run_cell, clients[s.label], s, cn, proto, rep, turn_cap)
                for (s, cn, proto, rep) in pending]
        for fut in concurrent.futures.as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:  # keep the sweep alive
                print(f"[skip] cell failed: {type(e).__name__}: {e}")
                continue
            append_record(out_path, rec)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cipher_sweep.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full new-module suite**

Run: `uv run pytest tests/test_ciphers.py tests/test_oracle.py tests/test_cipher_detect.py tests/test_cipher_runner.py tests/test_cipher_sweep.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/ciphers/config.py experiments/ciphers/sweep.py tests/test_cipher_sweep.py
git commit -m "feat: cipher experiment config + concurrent resumable sweep

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Staged entrypoints (`run_smoke.py`, `run_pilot.py`)

**Files:**
- Create: `experiments/ciphers/run_smoke.py`
- Create: `experiments/ciphers/run_pilot.py`

**Interfaces:**
- Consumes: `experiments/ciphers/{config,sweep}`, `clients.get_client`, `storage.read_records`.
- Produces: CLI entrypoints. `run_smoke.py`: 1 cheap model × 2 ciphers (`rot13`, `random_substitution`) × 1 protocol (`pure`) × turn_cap 6 → `data/ciphers_smoke.jsonl`, prints per-cell latencies. `run_pilot.py`: broad roster × all ciphers × all protocols × `REPLICATES` → `data/ciphers_pilot.jsonl`, `--yes` gate + per-provider cost estimate (reuse `cost.print_estimate`; a "call" here ≈ `turn_cap` model calls per cell, so estimate `len(cells) * turn_cap`).

- [ ] **Step 1: Write `run_smoke.py`**

```python
# experiments/ciphers/run_smoke.py
"""Smoke: validate the cipher pipeline end-to-end against one cheap model."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from llm_language_limits.clients import get_client
from llm_language_limits.config import MODEL_REGISTRY
from llm_language_limits.storage import read_records
from sweep import run_sweep

OUT = Path("data/ciphers_smoke.jsonl")


def main():
    spec = MODEL_REGISTRY["gpt-5-nano"]
    run_sweep(get_client, [spec], ["rot13", "random_substitution"], ["pure"], 1,
              OUT, turn_cap=6, resume=False)
    for r in read_records(OUT):
        print(f"{r['cipher']:20} action@{r['first_action_turn']} "
              f"decode@{r['first_explicit_turn']} prod@{r['first_production_turn']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `run_pilot.py`**

```python
# experiments/ciphers/run_pilot.py
"""Pilot: broad roster × all ciphers × all protocols. Inspect, then prune."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from llm_language_limits.clients import get_client
from llm_language_limits.storage import read_records
from llm_language_limits.cost import print_estimate
from config import models_for, CIPHER_SET, PROTOCOLS, TURN_CAP, REPLICATES
from sweep import run_sweep

OUT = Path("data/ciphers_pilot.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    specs = models_for("pilot")
    cells = len(specs) * len(CIPHER_SET) * len(PROTOCOLS) * REPLICATES
    for s in specs:
        print_estimate(s.label, (cells // len(specs)) * TURN_CAP, avg_in=300, avg_out=120)
    print(f"[estimate] ~{cells} cells x up to {TURN_CAP} turns each.")
    if not args.yes:
        print("Re-run with --yes to execute the pilot.")
        return
    run_sweep(get_client, specs, CIPHER_SET, PROTOCOLS, REPLICATES, OUT, turn_cap=TURN_CAP)
    print(f"[pilot] {len(read_records(OUT))} cells done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Offline import guard**

Run:
```bash
uv run python -c "import sys; sys.path.insert(0,'experiments/ciphers'); import run_smoke, run_pilot; print('import ok')"
```
Expected: prints `import ok` (no `main()` executed).

- [ ] **Step 4: Commit**

```bash
git add experiments/ciphers/run_smoke.py experiments/ciphers/run_pilot.py
git commit -m "feat: cipher experiment smoke + pilot entrypoints

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: LIVE GATES (human).** After creds are set: run `run_smoke.py` (validate + inspect latencies), then `run_pilot.py --yes`; inspect `data/ciphers_pilot.jsonl` and RE-TUNE `CIPHER_SET` / roster / `TURN_CAP` / protocols (drop redundant protocol if 2≈3) before any full sweep. Also decide the jailbreak sub-probe here (spec §4).

---

### Task 7: Analysis notebook (`analysis.ipynb`)

**Files:**
- Create: `experiments/ciphers/analysis.ipynb`

**Interfaces:**
- Consumes: `data/ciphers_pilot.jsonl` (or full).
- Produces: a notebook computing, per (model × cipher × protocol): mean `first_action_turn` (with non-comprehension = right-censored at `n_turns`+1), `first_production_turn`, production consistency, and a **difficulty ranking** of ciphers by mean comprehension latency. Figures: comprehension-latency heatmap (cipher × model), comprehension-vs-production scatter, latency-by-protocol comparison.

- [ ] **Step 1: Build the notebook (nbformat) with these cells**

Cell 1 — load + right-censor:
```python
import json, pandas as pd, numpy as np
rows=[json.loads(l) for l in open("../../data/ciphers_pilot.jsonl")]
df=pd.DataFrame(rows)
CAP=df["n_turns"].max()
for col in ["first_action_turn","first_explicit_turn","first_production_turn"]:
    df[col+"_c"]=df[col].fillna(CAP+1)   # censored: never = cap+1
df.head()
```

Cell 2 — difficulty ranking (by mean comprehension latency, pooled over models/protocols):
```python
rank=df.groupby("cipher")["first_action_turn_c"].mean().sort_values()
print(rank)
```

Cell 3 — comprehension-latency heatmap (cipher × model):
```python
import matplotlib.pyplot as plt
piv=df.pivot_table(index="cipher",columns="model",values="first_action_turn_c",aggfunc="mean")
fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(piv.values,aspect="auto",cmap="viridis_r")
ax.set_xticks(range(len(piv.columns)));ax.set_xticklabels(piv.columns,rotation=45,ha="right")
ax.set_yticks(range(len(piv.index)));ax.set_yticklabels(piv.index)
fig.colorbar(im,label="mean turns to comprehension (cap+1 = never)")
fig.savefig("../../data/ciphers_comprehension_heatmap.png",dpi=140,bbox_inches="tight")
```

Cell 4 — comprehension vs production + protocol comparison:
```python
g=df.groupby(["protocol"])[["first_action_turn_c","first_production_turn_c"]].mean()
print(g)
```

- [ ] **Step 2: Validate the notebook is well-formed**

Run:
```bash
uv run python -c "import nbformat; nb=nbformat.read('experiments/ciphers/analysis.ipynb',as_version=4); nbformat.validate(nb); print('nb ok', len(nb.cells))"
```
Expected: `nb ok 4`.

- [ ] **Step 3: Commit**

```bash
git add experiments/ciphers/analysis.ipynb
git commit -m "docs: cipher experiment analysis notebook

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- §2 IV1 ciphers (~8-10, incl novel/keyed) → Task 1 (10 ciphers, random_substitution + block_permutation keyed). ✓
- §2 IV2 protocols (pure/fewshot/escalating) → Task 4 `run_conversation` + Task 5 PROTOCOLS. ✓
- §2 IV3 turns/latency → Task 4 latency reduction; Task 5 turn_cap; Task 7 censoring. ✓
- §2 models (broad pilot → prune) → Task 5 `models_for`, Task 6 pilot gate re-tune. ✓
- §3 verifiable-task oracle + both comprehension signals → Task 2 (bank) + Task 3 (`comprehension_action` action-signal; `explicit_decode` decode-signal). ✓
- §3 production via inverse-decode + consistency → Task 3 `produced_in_code` + Task 4 `production_consistency`. ✓
- §3 no LLM judge for core / light judge only for fuzzy decode → Task 2/3 are pure string logic; no judge wired. (If explicit-decode fuzzy match proves too strict in the pilot, add a light judge THEN — logged as a pilot-gate action, not built now.) ✓
- §4 reuse harness + staged + resume + cost → Task 5 sweep (ThreadPool+resume mirrors part-1), Task 6 smoke/pilot + cost gate. ✓
- §4 jailbreak deferred → Task 6 Step 5 pilot-gate decision. ✓
- §6 contamination guard → Global Constraint + Task 1 keyed ciphers. ✓
- §5 human mirror + §articles → article-writing is a later phase (after data), not a code task. Noted.

**Gap:** the full-sweep entrypoint (`run_full.py`) is intentionally deferred — the spec mandates pruning the matrix at the pilot gate before a full run, so `run_full.py` is written after the pilot re-tune (a one-file follow-up mirroring `run_pilot.py` with the tuned `models_for("full")`/`CIPHER_SET`). Added as an explicit pilot-gate follow-up, not a pre-built task.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step has complete code.

**3. Type consistency:** `Cipher(name, encode, decode, lossy)` used consistently (Tasks 1,3,4). `run_conversation(...) -> dict` keys (`cipher/protocol/n_turns/first_*_turn/production_consistency`) match Task 5 `_run_cell` augmentation (`model/replicate`) and Task 7 columns. `cell_key` tuple `(model, cipher, protocol, replicate)` matches the record fields. ✓
