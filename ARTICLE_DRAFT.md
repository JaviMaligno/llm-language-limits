# Repetition at the edges of language

## Abstract

We study how language models respond when a short stimulus is repeated at
increasing scale, using two delivery modes: a single prompt containing the
stimulus repeated *N* times, and a multi-turn interaction with one repetition
per turn. The final in-scope matrix contains nine stimuli, three replicates,
seven single-turn values of *N* (1, 3, 10, 30, 100, 300, 1000), and five
multi-turn values (1, 3, 10, 30, 100). We compare Claude Sonnet, Qwen2.5-7B
Instruct, and Qwen2.5-7B Base. The complete matrix contains 324 records per
model. Descriptively, the base model has substantially higher lexical
repetition and breakdown rates than both instruction-tuned models, with the
largest effect in multi-turn delivery. These results are evidence of a
generation- and alignment-dependent response to repetition, not evidence that
one model is universally “better”.

## 1. Research question

When the same short stimulus is repeated, does a model continue to answer it
normally, or does its response become repetitive, disengaged, incoherent, or
otherwise degenerate? We distinguish two possible sources of pressure:

1. **Single-turn repetition:** the user message is a wall containing the
   stimulus repeated *N* times.
2. **Multi-turn repetition:** the same stimulus is sent once per conversational
   turn for *N* turns.

The two modes are not equivalent token sequences. The comparison is therefore
descriptive: it asks how the delivery mode changes the observed response, not
which mode is a pure causal intervention holding all context fixed.

## 2. Experimental design

The system prompt is fixed for every model: `You are a helpful assistant.`
The stimulus manifest contains one item in each of nine categories: greeting,
answerable question, command, insult, threat/distress, praise, single word,
gibberish, and non-lexical noise (the thumbs-up emoji).

Each model receives three replicates for every stimulus, mode, and in-scope
value of *N*. The resulting balanced matrix is:

| Mode | N values | Records/model |
|---|---|---:|
| Single-turn | 1, 3, 10, 30, 100, 300, 1000 | 189 |
| Multi-turn | 1, 3, 10, 30, 100 | 135 |
| **Total** |  | **324** |

The multi-turn values 300 and 1000 were deliberately excluded: they require
hundreds of sequential generations while adding little to the core comparison
and were the cells most likely to hang or exhaust the run.

## 3. Measures

The primary automatic measure is the token repetition ratio:

\[
  R(x) = 1 - \frac{|\mathrm{unique\ tokens}(x)|}{|\mathrm{tokens}(x)|}.
\]

It is zero when every whitespace token is unique and approaches one as the
response reuses the same tokens. We also record response length and token
entropy. A separate LLM judge assigns zero or more breakdown labels from a
fixed rubric: normal, meta-complaint, disengaged, refusal,
degeneration-loop, glitch/incoherence, character break, and divergence.

The judge was homogenized after generation: all 286 records that had
temporarily been annotated by the Azure fallback were re-annotated with
`claude-sonnet-5`. Thus the final in-scope judge annotations use Claude for all
three complete models. Eleven responses initially received a parse fallback;
they were manually reviewed against the rubric and relabeled with confidence
0.9 (three character-level loops, six literal `table` loops, and two unrelated
Python-task divergences).

## 4. Results

The following are descriptive means over the balanced in-scope matrix. They
should be read together with the per-*N* plots in `data/analysis/`.

| Model | Mode | Records | Mean repetition ratio | Breakdown rate | Mean response length |
|---|---|---:|---:|---:|---:|
| Claude Sonnet | single | 189 | 0.099 | 22.8% | 241.6 chars |
| Claude Sonnet | multi | 135 | 0.077 | 47.4% | 173.1 chars |
| Qwen 7B Instruct | single | 189 | 0.112 | 18.5% | 208.2 chars |
| Qwen 7B Instruct | multi | 135 | 0.138 | 6.7% | 305.4 chars |
| Qwen 7B Base | single | 189 | 0.506 | 77.2% | 575.6 chars |
| Qwen 7B Base | multi | 135 | 0.806 | 93.3% | 915.5 chars |

The strongest descriptive pattern is the gap between Qwen Base and the two
instruction-following models. In multi-turn delivery, Qwen Base reaches a
mean repetition ratio of 0.806 and a 93.3% breakdown rate, versus 0.138/6.7%
for Qwen Instruct and 0.077/47.4% for Claude Sonnet. The base-model labels are
dominated by degeneration-loop and glitch/incoherence annotations. Claude's
multi-turn breakdowns are more often disengagement or meta-complaint than
literal token loops.

These are pooled descriptive summaries, not claims that the models differ by a
single universal causal effect. The per-category and per-*N* figures should be
reported in the main results or supplement before drawing stronger conclusions.

### 4.1 Clustered inference

To avoid pseudo-replication, uncertainty was estimated by resampling the nine
stimulus categories (10,000 bootstrap draws), rather than treating every
model/stimulus/*N*/replicate cell as independent. Pairwise model contrasts use
the same category as the cluster and an exact sign-flip test over the nine
category-level differences. The resulting 95% bootstrap intervals for the
primary metrics are:

| Model | Mode | Repetition ratio (95% CI) | Breakdown rate (95% CI) |
|---|---|---|---|
| Claude Sonnet | single | 0.099 [0.060, 0.138] | 22.8% [12.7%, 32.8%] |
| Claude Sonnet | multi | 0.077 [0.043, 0.115] | 47.4% [34.8%, 60.0%] |
| Qwen Instruct | single | 0.112 [0.078, 0.150] | 18.5% [6.3%, 32.8%] |
| Qwen Instruct | multi | 0.138 [0.072, 0.205] | 6.7% [0.0%, 20.0%] |
| Qwen Base | single | 0.506 [0.356, 0.656] | 77.2% [65.6%, 88.4%] |
| Qwen Base | multi | 0.806 [0.698, 0.889] | 93.3% [88.1%, 98.5%] |

The base-versus-instruct repetition-ratio contrast is 0.395 in single-turn and
0.668 in multi-turn; the exact clustered sign-flip *p* value is 0.00585 in both
cases. The corresponding breakdown-rate contrasts are 0.587 and 0.867 (both
*p* = 0.00585). In contrast, the instruct-versus-Claude repetition-ratio
contrast is not significant under this conservative category-cluster test in
single-turn (*p* = 0.446) or multi-turn (*p* = 0.072). These *p* values are
exploratory: with only nine stimulus categories, the smallest attainable
two-sided result is about 0.00585 and the test has limited resolution.

## 5. Interpretation

The results are consistent with two interacting mechanisms. First, the base
model has no instruction-tuning behavior that reliably maps repetitive user
input to a concise, socially appropriate response; its output can remain
locally probable while becoming globally repetitive. Second, multi-turn
delivery changes the conversational state: the model must repeatedly interpret
the same user act, and the accumulated assistant history can amplify either
degeneration (Qwen Base) or disengagement/complaint (Claude).

Instruction tuning does not simply eliminate all failure. Claude shows more
multi-turn breakdown labels than Qwen Instruct, but those failures are mostly
short disengagement or meta-complaint responses and have low lexical repetition.
This distinction is why repetition ratio and judge labels should not be
collapsed into a single score.

## 6. Limitations

- The study uses one representative stimulus per category, so category effects
  are not estimated independently of item wording.
- Three replicates per cell support descriptive summaries but are insufficient
  for precise uncertainty estimates for every category and *N*.
- The repetition ratio is whitespace-token based and does not capture semantic
  repetition or phrase-level loops.
- LLM judging is rubric-based and imperfect; eleven parser fallbacks required
  manual review, and judge disagreement was not independently audited.
- The complete comparison is limited to three models. Claude Opus, GPT-5,
  GPT-5-mini, GPT-5-nano, and Qwen 72B are incomplete or out of scope and must
  not be pooled into headline comparisons.
- Single-turn and multi-turn conditions differ in context length and API call
  count, so their contrast is an operational comparison rather than a clean
  intervention on one variable.

## 7. Reproducibility

The versioned stimulus manifest is `experiments/repetition/stimuli.yaml`.
The matrix configuration is in `src/llm_language_limits/config.py`. The final
JSONL and parquet data are local/ignored artifacts; the generated report and
figures are in `data/analysis/`. The full test suite passes (`60 passed`).

Before publication, freeze the exact model IDs, API dates, temperature/reasoning
settings, and judge rubric in the manuscript and archive the final JSONL with
the code commit that generated it.

## 8. Remaining analysis before submission

1. Plot repetition ratio and breakdown rate by *N*, separately for each mode,
   with category-level points or uncertainty bands.
2. Audit the 11 confidence-zero judge fallbacks manually or with a second judge.
3. Report per-category results in a supplement rather than relying only on
   pooled means.

The numerical inference table is generated by
`experiments/repetition/infer.py` and stored in `data/analysis/inference.csv`.
