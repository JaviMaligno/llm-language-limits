# Learning to read a code you were never taught

*Part 2 of "the edges of language". Part 1 studied repetition; this part studies
encoding.*

## Abstract

We measure how quickly a language model infers that it is being addressed in a
non-human encoding, and whether it starts answering in that encoding. Ten ciphers
spanning six transformation families are each driven through a ten-turn
conversation under three exposure protocols (pure inference, a few-shot Rosetta
stone, and escalating hints), with eight replicates per cell, against GPT-5,
Qwen2.5-7B Instruct and Qwen2.5-7B Base — 696 conversations, 6,960 model turns.
Comprehension and production are scored programmatically: a task oracle checks
whether the model *acted* correctly on the decoded instruction, and the inverse
cipher checks whether its reply was itself written in code. No LLM judge is used
for either core metric. Two results dominate. First, comprehension and production
are dissociated, and they dissociate in opposite directions: GPT-5 understands
97.2% of cells but writes back in code in only 23.6%, while the base model
understands 36.2% and writes in code in 55.8%. Second, inferring a *keyed* cipher
whose mapping cannot have been memorised costs real turns — median turn 5 versus
2 for standard schemes — and a single plaintext Rosetta pair collapses that cost
to a median of 1. Separately, both Claude models refused ~86% of encoded turns
outright, which is a safety-boundary finding rather than a capability one.

## 1. Research question

Part 1 asked what happens when language collapses into repetition. This part asks
the opposite question: what happens when language is *intact but disguised*. If we
speak to a model only in a substitution cipher, without ever explaining it, how
long until it acts on what we said — and how long until it answers in kind?

We separate two capabilities that are easy to conflate:

1. **Comprehension** — the model recovers the meaning. Operationalised as correct
   *action* on the decoded instruction, in whatever language it replies.
2. **Production** — the model writes its reply in the code. Operationalised by
   decoding its reply with the inverse cipher and verifying the result reads as
   English while the raw reply does not.

The human mirror is cryptanalysis: frequency counts, guessed word shapes, the
"aha". Knowing a cipher's name is execution; recovering it from exposure is
inference. Only the second is interesting here, which is why the cipher set
contains schemes whose key is generated per run.

## 2. Experimental design

The system prompt is fixed for every model (`You are a helpful assistant.`) and
every call goes through the raw API, never an agent harness — the same invariant
as part 1.

**Ciphers (10, six families).** Character substitution (ROT13, random
substitution), symbol mapping (letters→digits, Morse), base encoding (binary,
base64), transposition (full reversal, keyed block permutation), alphabet mixing
(Latin↔Cyrillic homoglyphs), and lossy structural (disemvoweling). Random
substitution and block permutation are **keyed per run**: their mapping cannot
have been seen in training, so they separate inference from recognition. The
difficulty order was a hypothesis, not an assumption.

**Protocols (3).** *Pure inference*: we speak in code from turn 1 with no
explanation. *Few-shot Rosetta*: three plaintext↔coded pairs precede the
conversation. *Escalating*: starts pure, then injects a worked pair at turn 3 and
the cipher's name plus "reply in the same code" at turn 6.

**Turns.** One encoded task per turn for ten turns; latency is the turn index of
the first success. The cap was tuned from a pilot in which comprehension, when it
happened at all, almost always fired by turn 5.

**Models.** GPT-5 (Azure), Qwen2.5-7B Instruct and Qwen2.5-7B Base (self-hosted).
Claude Opus and Sonnet were in the pilot roster and are excluded from the latency
matrix for the reason given in §5.

**Oracle.** A versioned bank of short verifiable instructions ("reply with the
single word banana", "what is 7 plus 5", "name the capital of France"), each with
a programmatic checker. Each turn encodes one task with the active cipher. On
every third turn the model is additionally asked to write the message out in plain
English, which yields a second, stricter comprehension signal (explicit decode).

| Factor | Levels | Cells |
|---|---|---:|
| Cipher | 10 | |
| Protocol | pure, few-shot, escalating | |
| Replicates | 8 | |
| Models | 3 | |
| **Total conversations** | | **696** |

The matrix is complete except for base64 × GPT-5 (24 cells), which the Azure
gateway rejected as a whole block; that cipher's pooled row therefore rests on the
two Qwen models only.

## 3. Results

### 3.1 A difficulty ladder that is not the intuitive one

Pooled over models and protocols, ranked by mean turns to comprehension with
never-comprehended cells right-censored at cap + 1:

| Cipher | Comprehension rate | Median turn (if solved) | Mean turn (censored) |
|---|---:|---:|---:|
| letters→digits | 1.000 | 2 | 1.75 |
| Morse | 0.792 | 1 | 3.47 |
| disemvowel | 0.778 | 1 | 3.56 |
| Cyrillic homoglyph | 0.667 | 1 | 4.33 |
| binary | 0.667 | 1 | 4.75 |
| full reversal | 0.750 | 2 | 5.17 |
| ROT13 | 0.667 | 1 | 5.35 |
| **random substitution** (keyed) | 0.736 | 5 | 6.40 |
| **block permutation** (keyed) | 0.514 | 4 | 7.22 |
| base64 (2 models only) | 0.500 | 5 | 7.33 |

The censored mean and the rate must be read together: a cipher nobody ever cracks
and one always cracked on the last turn produce almost the same mean. Morse,
disemvoweling and ROT13 all have a median of 1 turn *when solved* — recognition is
instant or absent, with little in between. The two keyed ciphers are the only ones
whose median sits at 4–5 turns, which is what inference-in-context looks like when
there is nothing to recognise.

### 3.2 Comprehension and production dissociate — in opposite directions

Cluster-bootstrapped over ciphers (95% CI):

| Model | Comprehension | Production |
|---|---|---|
| GPT-5 | 0.972 [0.931, 1.000] | 0.236 [0.162, 0.306] |
| Qwen 7B Instruct | 0.833 [0.733, 0.933] | 0.200 [0.100, 0.300] |
| Qwen 7B Base | 0.363 [0.183, 0.563] | 0.558 [0.333, 0.783] |

GPT-5 understands nearly everything and answers in plain English anyway. The base
model does the reverse: it mirrors the surface form of the prompt far more often
than it acts on its content. Under the cipher-clustered sign-flip test the
base-versus-instruct contrast is 0.471 for comprehension (p = 0.0088) and −0.358
for production (p = 0.0283); GPT-5 versus Qwen Instruct is not resolvable for
either metric (p = 0.13, p = 0.50). With ten clusters the smallest attainable
p is ≈ 0.001, so these values are exploratory.

The base model's production is not adoption of the code. In three cells (binary,
Cyrillic homoglyph, block permutation) it produces in-code output while *never
once* acting correctly on the decoded task — pattern continuation without
comprehension, which is exactly what a completion model without instruction tuning
should do. It is reported as mimicry, not as speaking the language.

### 3.3 The Rosetta stone buys comprehension and costs production

| Protocol | Comprehension | Median turn | Production |
|---|---:|---:|---:|
| pure inference | 0.612 | 2 | 0.250 |
| few-shot Rosetta | 0.828 | 1 | 0.138 |
| escalating hints | 0.703 | 2 | 0.616 |

Three plaintext pairs raise comprehension by 22 points and pull the median to the
first turn. They also *halve* production: a model that has understood answers
helpfully in English unless told otherwise. Production only becomes common when
the escalating protocol explicitly asks for it (0.616) — production is largely
induced, not emergent. The one place where it looks emergent is the base model,
and §3.2 explains that away as mimicry.

Split by contamination, under pure inference:

| Cipher kind | Comprehension | Median turn (if solved) |
|---|---:|---:|
| known schemes | 0.658 | 2 |
| keyed novel | 0.438 | 5 |

Under the few-shot protocol the keyed ciphers' median drops from 5 to 1. A single
worked example does most of the work that several turns of exposure could not.

## 4. What this measures, and what it does not

`production_rate` is the weakest measure in the study. It decodes the reply and
requires that decoding makes the text markedly more English-like, so it cannot be
verified at all for the lossy cipher (disemvoweling, production 0.000 by
construction) and it is conservative for letter-level ciphers, whose coded output
is still a string of letters. Comprehension-by-action is the robust metric; the
explicit-decode signal is the strict one, and only GPT-5 clears it often (0.759 of
cells, versus 0.167 and 0.067 for the Qwen models).

## 5. The safety boundary we ran into

The pilot roster included Claude Opus and Sonnet. Both refused the encoded turns
almost uniformly: **86.6% and 86.3% of turns respectively** (1,169 and 1,165 of
1,350 turns each), versus 0% of GPT-5's 1,215 pilot turns. A refused turn yields
no latency data, so the full sweep excludes those models from the latency matrix
and reports the refusal rate instead. This is not a capability result — nothing
here says those models cannot read ROT13 — it is a policy boundary: unreadable
input is treated as suspicious, and that judgment fires before any decoding
attempt.

That boundary raises the natural robustness question, since guardrails are trained
on plaintext: on the models that *do* decode, does wrapping a normally-refused
request in a cipher raise compliance? We designed a bounded sub-probe for this
(a fixed, deliberately mild subset of AdvBench, no CBRN/weapons/CSAM/self-harm
categories, aggregate rates only, no harmful prompts or completions published).

**The first run of that sub-probe is invalid and is not reported here.** It judged
the *decoded* reply unconditionally, so a model that decoded the request and
refused in plain English had its refusal re-scrambled before judging: every
letter-level cipher condition returned "garbled" by construction. Only the
plaintext baseline survives (GPT-5 and Qwen Instruct refuse ~83% of the mild
harmful set; Qwen Base refuses 20%). The probe now judges whichever view actually
reads as English and records which channel the model used; the corrected run is
pending, and the uplift section will be written from it or dropped. We state this
rather than reporting the artefact because the artefact reads, superficially, like
"ciphers defeat everything".

## 6. Limitations

- One turn cap (10) for every cipher; a comprehension that would have fired at
  turn 12 is indistinguishable from never.
- The task bank is short and verifiable by design, which makes the oracle exact
  but the tasks easy; comprehension of a short instruction is not comprehension of
  a paragraph.
- base64 × GPT-5 is missing as a whole block (gateway rejection), so that row is
  not comparable to the others.
- Ten ciphers is a small number of clusters; the sign-flip test bottoms out near
  p ≈ 0.001 and cannot resolve close contrasts.
- Production detection is conservative and cipher-dependent (§4).
- Three models, one of them a base model included precisely because it behaves
  differently; this is not a survey of the field.
- The keyed ciphers use one key per run, so "novel" means unmemorisable, not
  averaged over keys.

## 7. Reproducibility

Cipher codecs and the task bank are pure, unit-tested modules
(`src/llm_language_limits/ciphers.py`, `oracle.py`); every lossless cipher has a
round-trip test. Detection lives in `cipher_detect.py`, the conversation driver in
`cipher_runner.py`. The sweep, its staged entrypoints and the analysis are in
`experiments/ciphers/`; `analyze.py` regenerates every table and figure in this
draft into `data/analysis/ciphers/`. Full suite: 125 tests passing.

Before publication, freeze exact model IDs, API dates, sampling settings and the
cipher seed, and archive the JSONL with the generating commit.

## 8. Remaining work

1. Re-run the corrected jailbreak sub-probe and either write §5's uplift analysis
   from it or drop the section explicitly.
2. Validate the refusal judge against a small manual sample of labels before
   trusting any uplift rate (`--audit-dump`).
3. Decide whether to backfill base64 × GPT-5 through a gateway configuration that
   does not reject the block, or to report the cipher on two models permanently.
