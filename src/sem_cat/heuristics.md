# Heuristics Reference for `sem_cat`

This document summarizes the main heuristics currently used in the translation QA and multi-model comparison pipeline.

Scope:

- step 02 translation QA,
- gloss complexity scoring,
- comparison risk aggregation,
- proposal selection,
- expert review queue selection.

The descriptions below are intentionally practical. They are meant to help a maintainer understand what the code is trying to detect and why a row becomes suspicious.

## QA heuristics

The QA layer works on one `(gloss_ru, gloss_en)` pair at a time. It returns:

- `qa_keep`
- `qa_score`
- `qa_flags`
- optional `roundtrip_distance`

Important distinction:

- some heuristics are fatal and immediately set `qa_keep=False`;
- others are non-fatal and only add flags and/or score.

| Heuristic / flag | Where | Trigger | Effect | Notes |
|---|---|---|---|---|
| `empty_translation` | `qa/translation_qa.py` | `gloss_en` is blank or becomes blank after stripping | Fatal, `qa_keep=False`, `qa_score=1.0` | These rows are treated as true blanks in step 02 sidecar logic. |
| `punctuation_only` | `qa/translation_qa.py` | Output contains only punctuation | Fatal, `qa_keep=False`, `qa_score=1.0` | Nonblank string, but linguistically useless. |
| `repeated_token_loop` | `qa/translation_qa.py` + `qa/translation_flags.py` | Strong unigram / bigram / trigram repetition patterns | Fatal, `qa_keep=False`, `qa_score=1.0` | Designed to catch loops such as repeated words or repeated punctuation chunks. |
| `quoted_trivial_output` | `qa/translation_flags.py` | Output is just quotes around a very short fragment | Non-fatal flag only | Useful for catching junk like `"x"` or `'?'`. |
| `repeated_hyphens` | `qa/translation_flags.py` | Regex pattern `(-\\s*){3,}` | Non-fatal flag only | Marks broken or placeholder-looking outputs. |
| `no_ascii_letters` | `qa/translation_qa.py` | Output contains no ASCII letters | Adds `+0.30` | Useful because the target is English gloss text. |
| `multiword_for_singleword` | `qa/translation_flags.py` | Russian gloss has 1 token and English output has at least 3 tokens | Adds `+0.20` | Not always wrong, but suspicious for dictionary gloss translation. |
| `token_inflation` | `qa/translation_flags.py` | Russian gloss has 1 token and English output has at least 4 tokens | Adds `+0.20` | Stronger inflation signal than the previous one. |
| `too_long_for_gloss` | `qa/translation_flags.py` | `len(gloss_en) > max(80, len(gloss_ru) * 5)` | Adds `+0.40` | Catches overgenerated explanations. |
| `sentence_like_singleword_expansion` | `qa/translation_flags.py` | Single-word Russian gloss and English starts with sentence-like prefixes such as `it is`, `there is`, `the city of`, `it is located` | Adds `+0.35` | Focused on dictionary glosses that turned into mini-definitions. |
| `probable_name_overexpansion` | `qa/translation_flags.py` | Title-case single Cyrillic token plus multiword or sentence-like English expansion | Adds `+0.20` | Targets proper names that get explanatory prose instead of a clean name translation. |
| `roundtrip_far` | `qa/translation_qa.py` | Back-translation distance exceeds `0.50` | Adds `+0.30` | Only applies when round-trip QA is enabled and back-translation exists. |

### Sentence-like prefixes currently checked

The sentence-like detector uses a small prefix list for single-word glosses, including:

- `it is `
- `it's `
- `there is `
- `there are `
- `this is `
- `we can `
- `it was `
- `the city of `
- `it is called `
- `it is located `

### Repetition detector summary

The repetition detector is deliberately simple and pattern-based:

- unigram dominance on at least 4 tokens,
- bigram dominance on at least 6 tokens,
- trigram dominance on at least 8 tokens.

It is intended to catch obvious generation failure, not to be a full semantic validator.

## Comparison heuristics

The comparison layer combines outputs from multiple models for the same `gloss_ru`. It uses three heuristic families:

1. gloss complexity heuristics,
2. total risk heuristics,
3. proposal and review heuristics.

### Comparison heuristics table

| Layer | Heuristic / signal | Trigger | Effect | Notes |
|---|---|---|---|---|
| Gloss complexity | `singleword_gloss` | Russian gloss has exactly 1 token | `+0.05` complexity | Single words are harder because they provide less context. |
| Gloss complexity | `very_short_gloss` | Stripped gloss length is at most 3 characters | `+0.05` complexity | Very short glosses are often ambiguous. |
| Gloss complexity | `hyphenated_gloss` | Gloss contains `-` | `+0.05` complexity | Often marks clitics, particles, or compact lexicographic notation. |
| Gloss complexity | `probable_proper_name` | Single-token title-case Cyrillic gloss | `+0.05` complexity | Proper names are special because models often over-explain them. |
| Gloss complexity | `punctuation_heavy_gloss` | At least 2 punctuation characters from `.,!?;:"'()-/\\` | `+0.05` complexity | Punctuation often means shorthand or non-canonical gloss formatting. |
| Gloss complexity | `particle_or_clitic` | Standalone token `ни` or `ли`, token ending in `-то`, `-либо`, `-нибудь`, or token starting with `кое-` | `+0.05` complexity | This logic was tightened so `ни` and `ли` only match as whole tokens, not as substrings inside normal lexical words. |
| Gloss complexity | complexity cap | Sum of complexity contributions | capped at `0.30` | Prevents gloss complexity from dominating the whole risk score. |
| Total risk | `all_blank` | No model produced a nonblank output | `+0.40` risk | Hard failure for comparison usefulness. |
| Total risk | `only_one_output` | Exactly one nonblank output exists | `+0.20` risk | Coverage is weak even if the one output looks okay. |
| Total risk | `low_coverage` | Nonblank outputs are fewer than half of compared models | `+0.10` risk | Partial-coverage warning. |
| Total risk | `high_max_qa_score` | Max nonblank `qa_score` is above `0.5` | `+0.25` risk | A single very suspicious model output raises row risk. |
| Total risk | `elevated_avg_qa_score` | Average nonblank `qa_score` is above `0.3` | `+0.125` risk | Softer QA warning spread across models. |
| Total risk | `some_qa_keep_false` | At least one nonblank output has `qa_keep=False` | `+0.075` risk | Important because step 02 now keeps nonblank rejected outputs in the main CSV. |
| Total risk | `severe_disagreement` | `disagreement_score > 0.5` | `+0.25` risk | Strong cross-model conflict. |
| Total risk | `moderate_disagreement` | `disagreement_score > 0.3` | `+0.125` risk | Milder conflict. |
| Total risk | `multiple_sentence_like_outputs` | At least 2 models produced `sentence_like_singleword_expansion` | `+0.20` risk | Signals systematic overgeneration. |
| Total risk | `sentence_like_output` | Exactly 1 model produced `sentence_like_singleword_expansion` | `+0.10` risk | Weaker version of the previous signal. |
| Total risk | round-trip distance contribution | One or more outputs have large `roundtrip_distance` | adds round-trip risk | This only matters for models/runs where round-trip data exists. |
| Total risk | complexity contribution | `compute_gloss_complexity()` returned a nonzero score | adds complexity score directly | Complexity score is multiplied by `complexity_weight`, currently `1.0`. |
| Total risk | `strong_consensus` discount | Largest cluster size >= 3, good outputs >= 3, consensus ratio >= 0.60 | `-0.20` risk | Consensus lowers risk when it is both broad and clean. |
| Proposal | strong consensus pick | Largest cluster has 3 models | choose lowest-`qa_score` output inside the largest cluster | Decision reason is strong-consensus style selection. |
| Proposal | near-match consensus pick | Largest cluster has 2 models and total risk is below threshold | choose lowest-`qa_score` output inside the largest cluster | Allows a proposal when two models nearly agree and risk is still manageable. |
| Proposal | only one good output | Exactly one nonblank output has `qa_keep=True` and sufficiently low `qa_score` | choose that output | Useful when most models fail but one model behaves. |
| Proposal | low-confidence single output | Only one nonblank output exists, but confidence is weaker | keep a tentative proposal | Intended to preserve signal without pretending certainty. |
| Proposal | `all_blank` | No nonblank outputs at all | force manual review | No automatic proposal is possible. |
| Proposal | `no_clear_winner` | Outputs stay too far apart or no stable cluster wins | force manual review | Main fallback when disagreement stays unresolved. |
| Review routing | high-risk manual-review override | Total risk >= `0.65` and decision is not already fatal/manual-review | switch preferred source to manual review | Prevents fragile automatic picks on very risky rows. |
| Review routing | strict expert queue by risk | `risk_level` is `high` or `medium` | `needs_expert_review=True` | Low-risk benign disagreements should not flood the queue. |
| Review routing | strict expert queue by fatal comparison outcome | `decision_reason` is `all_blank` or `no_clear_winner` | `needs_expert_review=True` | These rows always need a human. |
| Review routing | strict expert queue by QA failure | Any nonblank model output has `qa_keep=False` | `needs_expert_review=True` | Nonblank rejected outputs are treated as meaningful review signals. |

## Practical reading notes

### Why the `particle_or_clitic` update matters

Earlier substring matching could treat ordinary lexical words as if they contained standalone particles such as `ни` or `ли`. The current logic is narrower:

- standalone `ни` and `ли` must match whole tokens,
- suffix patterns are checked with `endswith(...)`,
- `кое-` is checked with `startswith(...)`.

This reduces false complexity inflation on ordinary Russian words.

### Why step 02 and step 03 now fit together better

Step 02 now distinguishes:

- true empty outputs,
- rejected-but-nonblank outputs,
- suspicious-but-kept outputs.

That matters because step 03 can now treat nonblank `qa_keep=False` outputs as real comparison evidence and real review signals, rather than accidentally conflating them with blanks.

### What to inspect first in practice

When triaging a difficult row, the most useful fields are usually:

- `qa_keep`
- `qa_score`
- `qa_flags`
- `gloss_complexity_reasons`
- `risk_reasons`
- `decision_reason`
- `needs_expert_review`

Those fields usually explain why the row is in the queue much faster than reading all model outputs cold.
