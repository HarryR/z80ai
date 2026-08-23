# Training data

```bash
python data/lint.py data/adventure/commands.txt      # check before you train
python data/adventure/commands.txt | ...             # pipe-separated: query|RESPONSE
cat data/adventure/commands.txt | ./feedme.py --epochs 150
```

## Check the data before you train it

`data/lint.py` reports what a dataset will actually do to a 2-bit model. Run it
first — it takes a second, training takes an hour, and most of what goes wrong
is visible up front.

```
pairs                               11,900
unique queries                      11,900
unique responses                        17
charset                                 20  ' ACDEGHIKLNOPRSTUVW' + EOS
output layer weights                 2,560
exact duplicate pairs                    0  (0.0%)
pairs in a contradiction                 0  (0.0%)
accuracy ceiling                   100.0%
```

`--strict` exits non-zero if anything is flagged, so it can gate a build.

The two numbers that matter most are the ones nobody counts:

**Unique responses**, not line count. The character decoder is really a label
decoder — a model with four responses is doing four-way classification, however
many lines you feed it. Each additional response costs capacity, and each
additional *character* costs 128 output weights and forces a full retrain to
add. `tinychat` has 502 responses and a 40-character charset; it is fighting the
architecture.

**The accuracy ceiling.** If the same query appears with two different
responses, no model can be right about both. `tinychat` is capped at 87.0% and
`guess` at 97.8% before training starts. Without this number a model that has
learned everything learnable looks like it is underperforming, and the fix looks
like "train longer" when it is "fix the data".

## What suits this architecture

A query is hashed into 128 trigram buckets and classified. That rewards:

- **A small closed label set** — 4 to ~30 responses. This is the real complexity
  parameter.
- **Many phrasings per label.** Volume belongs here, not in more labels.
- **Short, keyword-bearing queries.** Long sentences that differ subtly ("bigger
  than a cat" vs "smaller than a cat") share most of their trigrams and the
  distinguishing signal gets diluted.
- **Responses that share characters.** `GO N`/`GO S`/`TAKE`/`DROP` keep the
  charset small. One stray `/` in one line costs an output neuron.
- **An explicit catch-all.** Out-of-distribution input produces *something*;
  training an `IDK` class decides what.

It does not suit open-ended chat, arithmetic, or anything needing more than a
couple of dozen distinct answers.

## Datasets

| | pairs | responses | charset | ceiling | held-out accuracy |
|---|---:|---:|---:|---:|---:|
| [`adventure/`](adventure/) | 11,900 | 17 | 20 | 100% | **99.8%** |
| `examples/guess/` | 28,718 | 4 | 11 | 97.8% | 90.6% (40 epochs) |
| `examples/tinychat/` | 2,990 | 502 | 40 | 87.0% | 61.0% |

Held-out accuracy is `IntAcc` on queries the model never saw, which is what
`feedme.py` now reports. `guess` is undertrained at 40 epochs rather than badly
shaped; `tinychat` reaches 78.7% on its *training* data and 61.0% off it, which
is the gap that motivated measuring any of this. The adventure figure is a
range because training is not seeded — see TRAINING.md.

## Generating data

`data/adventure/generate.py` is deterministic and dependency-free: same seed,
same bytes, no network. Its output is checked in, and a test asserts the two
still agree.

The two older generators are not like this. `examples/guess/gendata.py` needs
Ollama or an `ANTHROPIC_API_KEY`, `examples/tinychat/genpairs.py` needs three
local Ollama models, neither is seeded, and neither one's output is checked in —
so the data behind the shipped models cannot be reproduced. They are still
useful for bootstrapping a new topic from an LLM; just commit what comes out.

Whatever produced it, run `data/lint.py` on the result.
