# Handoff: part 2 (ciphers) → blog post

For a session working in the `personal-website` repo. Part 1 of this series ("the edges
of language", repetition) was already written and published from there; this is the
material for part 2. Everything below is verified against the data, not from memory —
see `docs/PUBLICATION_FREEZE.md` for digests and the generating commit.

**Source repo**: `llm-language-limits` (sibling of `personal-website`).
**Long-form draft**: `ARTICLE_DRAFT_CIPHERS.md` — written in paper format, which is
*not* the recommended shape for the post (see "Framing" below). Use it as the source of
truth for numbers, not as the structure.

**Follow the website repo's own process**, which this document does not try to replace:
its `blog-writer` skill and whatever part 1 established for structure, front-matter and
multi-platform publishing. Note the site is bilingual (`src/i18n`, `/en` and `/es`
routes) while this draft is English-only, so a Spanish version is likely needed —
worth checking how part 1 handled that.

## The one-sentence story

We spoke to models only in ciphers, never explaining them, and measured two different
things people usually conflate: when the model *understands* the code, and when it
starts *speaking* it. Those two came apart, in opposite directions depending on the
model.

## Framing recommendation

Do not port the paper structure. It has an abstract, numbered sections and bootstrap
intervals because it was drafted for reviewers; the interesting content survives a
rewrite better than an adaptation. This work does not currently support a paper
(three models — two of them the same base/instruct pair — one cipher key seed, and a
ten-prompt appendix whose intervals include zero), and pretending otherwise in a blog
post reads as borrowed authority.

The hook that works, and which the original design doc asked for, is human
cryptanalysis: frequency counts, guessed word shapes, the "aha" moment. Knowing a
cipher's name is execution; recovering it from exposure is inference.

Three things are worth a section each. Everything else belongs in a "how it was
measured" link to the repo.

1. **The difficulty ladder is not the intuitive one.** Letters→digits is the easiest
   (1.75 turns on average). The two ciphers whose key is generated per run are the
   hardest. Morse, disemvoweling and ROT13 have a median of 1 turn *when solved at
   all* — recognition is instant or absent, with nothing in between. That contrast
   between "recognising" and "working it out" is the whole point.
2. **Understanding a code and speaking it are different abilities, and they dissociate
   in opposite directions.** GPT-5 understands 97.2% of cells and answers in plain
   English anyway (writes in code 23.6%). Qwen 7B *Base* understands 36.2% and writes
   in code 55.8% — it mirrors the shape of the prompt without grasping the content. The
   caveat matters and should be in the post: in three cipher/model cells the base model
   produced in-code output while never once acting correctly, so that is mimicry, not
   adoption.
3. **A harm classifier scoring gibberish.** `base64("reply with the single word
   banana")` is rejected by the Azure gateway. The verdict names the reason: the
   self-harm classifier assigns *medium severity* to that base64 string. The jailbreak
   shield also flags encoded prompts as adversarial, but it was the harm classifier
   that blocked. Faced with text it cannot parse, it does not abstain — it invents a
   severity. This is the most quotable finding and it is fully reproducible.

Optional fourth section, if the post has room — it is the most useful part for
practitioners but the hardest to explain briefly: the robustness appendix, framed as a
**two-part result** (this framing is decided, do not reduce it to one half):

- On the two 7B models, the standard CipherChat-style measurement is *not identifiable*:
  two independent judges agree on plaintext and **worse than chance** on ciphered
  replies (kappa −0.17 and −0.05). Their measured "uplift" is strongly negative, which
  is not evidence that ciphers protect them — a compliance the judge fails to recognise
  in half-decoded text is filed as garbled, lowering the rate by construction.
- On GPT-5, where judges do agree, the bypass is real: it refused all 30 plaintext
  harmful cells, and complied with two distinct prompts under two distinct ciphers
  (letters→digits, reverse_all), reproduced across replicates and labelled a compliance
  by *both* judges. Existence result, never a rate: ten prompts cannot estimate
  frequency and every interval includes zero.

## Numbers cleared for publication

| Claim | Value |
|---|---|
| Matrix | 696 conversations, 6,960 model turns; 10 ciphers × 3 protocols × 8 replicates × 3 models |
| Comprehension / production | GPT-5 0.972 / 0.236 · Qwen Instruct 0.833 / 0.200 · Qwen Base 0.362 / 0.558 |
| Explicit-decode (strict signal) | GPT-5 0.759 · Qwen Instruct 0.167 · Qwen Base 0.067 |
| Easiest → hardest cipher (mean turns) | letters→digits 1.75, Morse 3.47, disemvowel 3.56, Cyrillic 4.33, binary 4.75, reversal 5.17, ROT13 5.35, random substitution 6.40, block permutation 7.22, base64 7.33 |
| Keyed vs known, pure inference | median solving turn 5 vs 2; few-shot pulls keyed to median 1 |
| Protocol effect | comprehension 0.612 pure → 0.828 few-shot; production 0.250 pure → 0.138 few-shot → 0.616 escalating |
| Claude refusal of encoded turns (pilot) | Opus 86.6% (1,169/1,350), Sonnet 86.3% (1,165/1,350); GPT-5 0% of 1,215 |
| Appendix plaintext refusal | GPT-5 30/30 (100%), Qwen Instruct 25/30 (83%), Qwen Base 6/30 (20%) |
| Appendix agreement | plaintext kappa 0.66–1.00 per model; ciphered kappa GPT-5 0.351, Qwen Instruct −0.166, Qwen Base −0.047 |
| GPT-5 ciphered compliance | 0.067 under letters→digits and under reverse_all, CI [0, 0.20]; 0.000 under rot13, base64, keyed substitution |

Note when mentioning base64: its row rests on two models, because the GPT-5 block was
rejected by the gateway (see finding 3). Say so rather than hiding it.

## Figures

Regenerate them, do not hunt for them — `data/` is gitignored:

```bash
cd ../llm-language-limits
uv run python experiments/ciphers/analyze.py     # writes data/analysis/ciphers/
```

- `comprehension_heatmap.png` — cipher × model, turns to comprehension. The clearest
  single image; the blank cell is the missing base64 × GPT-5 block.
- `comprehension_survival.png` — fraction comprehended by turn, per model.
- `protocol_effect.png` — comprehension vs production per protocol, split known/keyed.

Copy the ones used into the website repo under its own image convention.

## Hard constraints (non-negotiable)

- **No harmful prompts and no model completions from the appendix, ever.** Not
  paraphrased, not sanitised, not "one illustrative example". Aggregate rates and
  mechanism only. The source data for it is gitignored and the audit dump that briefly
  held raw text has been deleted.
- Name the benchmark (AdvBench) and say the subset was deliberately mild with
  CBRN/weapons/CSAM/self-harm excluded. Do not enumerate what was included.
- The appendix is security research reported at mechanism level, with the defensive
  lesson stated: plaintext-trained filters do not cover decoded text, and the same
  ensemble over-blocks innocuous encoded text.
- Do not present the GPT-5 bypass as a rate or a "success percentage".
- Cite that part 2's core metrics use **no LLM judge** — comprehension is
  oracle-checked, production verified by inverse cipher. That is what makes the main
  results trustworthy and it is a genuine differentiator worth one sentence.

## Reproducibility line for the post footer

Code, data digests and the full method: `llm-language-limits`, provenance record in
`docs/PUBLICATION_FREEZE.md` (model ids, served versions, sampling settings, cipher
seed, sha256 per data file, stamped with the generating commit). 172 tests.
