# Training Z80-μLM

```bash
pip install torch                          # training only; building needs just NumPy
python data/lint.py data/adventure/commands.txt --strict
cat data/adventure/commands.txt | ./feedme.py --epochs 150 --chat
./exportmodel.py -m command_model_autoreg.pt -o model.npz
./build.py -m model.npz -o CHAT.COM
```

Requirements: Python 3.10+, PyTorch, NumPy. No emulator needed — `libz80emu.py`
runs the generated code in-process.

`--chat` opens a REPL after training. Ctrl+C stops training gracefully and keeps
the best checkpoint, and a later run continues from it as long as the
architecture and charset match.

**Export to `.npz` when you are done.** `build-examples.sh`, `verify_artifacts.py`,
`bench.py` and the release all read `model.npz`; the `.pt` checkpoint is only an
intermediate. Skipping the export leaves them on the old weights.

## Data format

Pipe-separated, case-insensitive, `#` for comments:

```
# movement
go north|GO N
head north|GO N
n|GO N
take the lamp|TAKE
asdf|IDK
```

Both sides are uppercased. Queries shorter than 2 characters are dropped;
queries over 60 and responses over 50 are **truncated, not rejected**, so data
past those limits silently changes meaning.

## Read this before collecting data

Two numbers decide whether a model can learn your dataset, and neither is the
line count. `data/lint.py` reports both.

### Distinct responses, not lines

The character decoder is really a **label decoder**. A model with four responses
is doing four-way classification no matter how many lines you feed it. Each
additional response costs capacity; each additional *character* costs 128 output
weights and forces a full retrain to add, because the output layer changes shape.

Aim for **4 to ~30 distinct responses**, 1–2 words each. Put your volume into
*phrasings per response*, not into more responses.

```
are you smart|YES        smart bot|YES        youre smart|YES
```

### The accuracy ceiling

If the same query appears with two different responses, no model can be right
about both. That caps accuracy before training starts:

| dataset | responses | ceiling |
|---|---:|---:|
| `data/adventure/` | 17 | 100% |
| `examples/guess/` | 4 | 97.8% |
| `examples/tinychat/` | 502 | 87.0% |

Without this number a model that has learned everything learnable looks like it
is underperforming, and the fix looks like "train longer" when it is "fix the
data".

### Also worth knowing

- **Balance.** No response should be more than ~40% of the data. A small model
  falls back to the majority class whenever it is unsure, so a dominant label
  becomes its default answer. `guess` ships `NO` at 56.2%, and four of six
  nonsense queries (`ASDF QWERTY`, `BANANA PHONE`, …) come back `NO`.
- **Short, distinctive queries.** Long sentences differing subtly — "bigger than
  a cat" vs "smaller than a cat" — share most of their trigrams, and the
  distinguishing signal gets diluted across 128 buckets.
- **Train a catch-all.** Out-of-distribution input produces *something*
  regardless; an explicit `nonsense → IDK` class decides what. Note that this
  only works if your charset can spell it — `guess`'s cannot.
- **Watch the charset.** One line containing a `/` costs an output neuron for
  the whole model. `data/lint.py` flags characters appearing in three or fewer
  pairs.

## Reading the training output

```
Loaded 11900 pairs → 10710 train / 1190 val → 1 chunks of 10710
Ceiling from contradictory labels: train 100.0%, val 100.0%
  Epoch 120: CE=0.0122, Acc=99.8%, IntAcc=99.7%, ValAcc=99.6% *BEST*
```

| column | meaning |
|---|---|
| `Acc` | float accuracy on training data. Ignore it. |
| `IntAcc` | quantized accuracy on training data — what the Z80 will compute |
| `ValAcc` | quantized accuracy on **held-out queries**. This is the real one. |

**`IntAcc` is not `Acc`.** Float accuracy can sit at 99% while integer accuracy
is 60%; only the integer number survives to the Z80.

**`ValAcc` is not `IntAcc`.** The gap between them is memorization:

| dataset | IntAcc | ValAcc | gap |
|---|---:|---:|---:|
| `data/adventure/` | 100.0% | 99.8% | 0.2 |
| `examples/guess/` | 90.7% | 90.6% | 0.1 |
| `examples/tinychat/` | 78.7% | **61.0%** | **17.7** |

Tinychat's 502 responses are being memorized, not learned. Until recently every
number this project reported was measured on the training set, so nothing could
have told you.

The split holds out 10% of unique *queries* (`--val-frac`, `--seed`). Note that
templated paraphrases of a held-out query still appear in training, so `ValAcc`
is optimistic — `guess`'s ~0 gap is partly that effect. `--val-frac 0` restores
the old training-set-only behaviour, and then the summary says so.

`--save-best` keeps the checkpoint with the best `ValAcc`. Worth using: training
is not monotonic, and a late epoch can be several points worse than the peak.

## Quantization-aware training

The model trains with progressive quantization and never in pure float:

```python
quant_temp = 0.3 + 0.7 * min(1.0, epoch / (epochs * 0.8))
```

30% quantized at epoch 0, fully quantized at 80% of the run, then refinement.
Starting in float lets the model find solutions that collapse when quantized —
if `IntAcc` suddenly drops, that is what happened.

Weights land on `{-2, -1, 0, +1}` — four values, asymmetric, not `±2`. The scale
is a per-layer 95th percentile of `|w|`, recomputed each forward pass. Gradients
pass through a straight-through estimator; an overflow penalty (`libqat.py`)
discourages accumulator values the Z80's 16 bits could not hold.

**Give it time after `QTemp` reaches 1.0.** The model needs to refine once fully
quantized; stopping the moment it gets there wastes the run.

## Architecture

Default `256 → 256 → 192 → 128 → charset_size`, ~143K weights at 2 bits ≈ 36KB
of information.

| component | size | purpose |
|---|---:|---|
| query buckets | 128 | trigram hash of the user's query |
| context buckets | 128 | hash of the last 8 characters generated |
| hidden | 256, 192, 128 | `--hidden-sizes 256,192,128` |
| output | `len(charset)` | one neuron per character, derived from responses |

Wider layers mean more capacity, a larger binary and slower inference. The Z80
backends count neurons in `B`, so `DJNZ` caps a layer at 256; anything wider
builds only for eZ80 (`buildez80.py`) — see [EZ80.md](EZ80.md). `feedme.py`
warns when you cross that line and the Z80 builders raise rather than
mis-assemble.

Memory, for the default shape: 65,536 + 49,152 + 24,576 weights in the hidden
layers, plus `128 × len(charset)` for the output. About 35KB packed.

## Debugging

| symptom | cause |
|---|---|
| Garbage output | `IntAcc` too low. Check it, not `Acc`. Quantization only reaches full strength 80% of the way through the run, so give it more epochs — or shrink the vocabulary. |
| Always the same answer | Class imbalance — one response above ~40%. Run `data/lint.py`. |
| High `IntAcc`, low `ValAcc` | Memorizing. Too many distinct responses, or too few phrasings each. |
| Accuracy plateaus below 100% | Check the ceiling. Contradictory labels may make the rest unreachable. |
| Similar inputs, same output | Trigram collision. Use shorter, more distinctive phrasings. |
| Binary too big | Fewer/narrower layers, or a smaller charset. Check what `data/lint.py` says about rare characters. |

## Example run

```
$ ./feedme.py --file data/adventure/commands.txt --epochs 150 --save-best

============================================================
Loading training data...
Loaded 11900 pairs → 10710 train / 1190 val → 1 chunks of 10710
Ceiling from contradictory labels: train 100.0%, val 100.0%
Epochs per chunk: 150
============================================================
Charset (20 chars): ' ACDEGHIKLNOPRSTUVW' + EOS
Validation: 1190 pairs → 5654 character examples
No checkpoint found, starting fresh
Model: 256 → 256 → 192 → 128 → 20
Parameters: 142,420

--- Chunk 1/1: 10710 pairs ---
Generated 51046 character examples
  Epoch 10: CE=1.5362, Acc=52.6%, IntAcc=51.4%, ValAcc=51.1% *BEST*
  Epoch 50: CE=0.0451, Acc=98.8%, IntAcc=97.1%, ValAcc=96.6% *BEST*
  Epoch 90: CE=0.0072, Acc=99.9%, IntAcc=99.6%, ValAcc=99.5% *BEST*
  Epoch 100: CE=0.0070, Acc=99.9%, IntAcc=99.9%, ValAcc=99.7% *BEST*
  Epoch 110: CE=0.0069, Acc=99.9%, IntAcc=99.2%, ValAcc=98.7%
  Epoch 150: CE=0.0046, Acc=100.0%, IntAcc=100.0%, ValAcc=99.8% *BEST*
Saved best (epochs: 150, best: 99.8% @ 150)

============================================================
Finished: 1/1 chunks, 150 total epochs
Best IntAcc (held-out): 99.8% at epoch 150
============================================================
```

(Every 10th epoch shown; nothing else altered.)

Two things to notice. Accuracy is **not monotonic** — epoch 110 lands below
epoch 100 — which is what `--save-best` is for. And **training is not seeded**:
there is no `torch.manual_seed` anywhere, so the same command on the same data
lands somewhere around 98.5–99.8% run to run. Compare runs with that in mind,
and do not read a half-point difference as a real difference.

See [data/README.md](data/README.md) for what makes a dataset suit this
architecture, and [data/adventure/](data/adventure/) for a worked example.
