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
