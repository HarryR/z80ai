# Training Z80-μLM

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- CP/M Z80 emulator for testing

```bash
# Train on included conversation data
cat data/tinychat*.txt | ./feedme.py --epochs 300 --chat
# This will launch a > chat shell to test after training

# Build the Z80 binary
./buildz80com.py -m model.pt -o chat.com
```

## Training Data Format

Training data is pipe-separated and case-insensitive: `input|response`

For example:

```
hello|HI
HI|yo!
are you real|MAYBE
i hate you|ok
im sure about this|R U?
youre|you're
im|I'm
theyre|they're
```

The `data/` folder contains some themed training files. You can pipe or cat training data to the `feedme.py` script, for example:

```
cat data/*.txt | ./feedme.py
```

It will continue from the last snapshot (if any), meaning you can do multi-step training or fine-tuning, just pipe the training data to the script.

At any point you can hit Ctrl+C to gracefully stop training, and take a binary snapshot of your model using `buildz80com.py`.

## Quantization-Aware Training (QAT)

The model trains with progressive quantization, but never pure float:

```python
quant_temp = 0.3 + 0.7 * min(1.0, epoch / (epochs * 0.8))
```

- **Epoch 0**: 30% quantized (soft start)
- **Epoch 80%**: Fully quantized
- **Remaining 20%**: Pure quantized refinement

This prevents the model from finding float-only solutions that collapse when quantized. The number of epochs adjusts the linear slope angle, meaning more epochs give you longer training time with more precise weights.

Both floating point & integer quantized models are run in parallel, meaning the floating point model is being scored on how well its knowledge survives quantization.

### Key Training Features

- **Straight-Through Estimator (STE)**: Gradient flows through quantization
- **Overflow penalty**: Regularization to keep accumulator values safe
- **Quantization loss**: Encourages weights toward {-2,-1,0,+1} grid
- **Best model tracking**: Saves highest IntAcc checkpoint
- **Graceful Ctrl+C**: Saves best model on interrupt

## Training Tips

### 1. Small Vocabulary Wins

Fewer unique responses = easier for 2-bit weights to learn.

 * **Bad:** 200 different multi-word responses (like this sentence), the entropy is far too high

 * **Good:** 30-50 terse responses (OK, YES, NO, LS, DIR, X, Y, B3:, etc.)

### 2. Watch IntAcc, Not Acc

Float accuracy (Acc) can hit 99% while integer accuracy (IntAcc) is 60%. Only IntAcc matters for the Z80.

### 3. Never Pure Float

Starting at 30% quantized prevents catastrophic collapse mid-training. If you see IntAcc suddenly drop, your model found and fixated on a float-only solution.

### 4. More Epochs at QTemp=1.0

The model needs time to refine after reaching full quantization. Don't stop training right when QTemp hits 1.0.

### 5. Vary Your Phrasings

Same response should trigger from multiple input styles:
```
are you smart|yay
smart bot|yay
youre smart|yay
you are smart|yay
```

### 6. Echo Back Keywords

When appropriate, echo words from the input:
```
tell me about you|ME?
what about me|YOU?
is it good|GOOD?
```

### 7. Keep Responses Short

1-2 words, 5-10 characters max. Longer responses are harder to learn and slower to generate.

### 8. Match Data Size to Model Capacity

Rule of thumb: aim for roughly 1 training example per parameter. A 150K parameter model wants ~150K input/output pairs. Too few examples = overfitting. Too many = underfitting (model can't memorize everything).

Note: each query-response pair generates multiple character-level training examples during autoregressive training, so the effective example count is higher than your line count.

### 9. Out-of-Distribution Inputs Are Unpredictable

Inputs the model never saw during training will produce arbitrary outputs. With only 2-bit weights and limited capacity, there's no graceful "I don't know" - the model just fires whatever neurons activate strongest for the unfamiliar trigram pattern.

If you need reliable behavior on novel inputs, train explicit catch-all patterns (e.g., nonsense → IDK).


## Model Architecture

Default architecture: `256 → 256 → 192 → 128 → charset_size`

- Input: 256 (128 query buckets + 128 context buckets)
- Three hidden layers with ReLU
- Output: one neuron per character in charset

Larger hidden layers = more capacity but bigger .COM file.

### Architecture Parameters in feedme.py

These parameters in `feedme.py` control the model structure:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | 256 | Total input dimension (query + context buckets) |
| `hidden_sizes` | [256, 192, 128] | List of hidden layer widths |
| `num_chars` | ~35 | Output size (auto-derived from charset) |

**Input Encoding (256 dimensions):**

| Component | Buckets | Purpose |
|-----------|---------|---------|
| `TrigramEncoder.num_buckets` | 128 | Hash buckets for query text trigrams |
| `ContextEncoder.num_buckets` | 128 | Hash buckets for recent output n-grams |
| `ContextEncoder.context_len` | 8 | Characters of context to consider |

The input is split evenly: the first 128 values encode the user's query using trigram hashing, and the second 128 encode the characters generated so far (for autoregressive generation).

**Modifying the Architecture:**

To change layer sizes, edit line 382 in `feedme.py`:
```python
hidden_sizes = [256, 192, 128]  # Three hidden layers
```

Trade-offs:
- **Wider layers** → More capacity, larger .COM file, slower Z80 inference
- **Deeper layers** → Better feature extraction, but diminishing returns
- **Fewer layers** → Faster inference, less expressive

To change bucket counts, modify the encoder instantiations:
```python
query_encoder = TrigramEncoder(num_buckets=128)      # Query encoding
context_encoder = ContextEncoder(num_buckets=128, context_len=8)  # Context
```

**Memory Impact:**

Each layer adds `input × output × 2 bits` of weights plus `output × 16 bits` of biases. For the default architecture:
- Layer 1: 256×256 = 65,536 weights (16KB packed)
- Layer 2: 256×192 = 49,152 weights (12KB packed)
- Layer 3: 192×128 = 24,576 weights (6KB packed)
- Layer 4: 128×35 = 4,480 weights (1KB packed)
- **Total: ~35KB** for a typical model

## Charset

The charset is automatically derived from training data responses. Smaller charset = easier classification.

Typical charset: ` ABCDEFGHIJKLMNOPQRSTUVWXYZ?!'` + EOS (~30-35 chars)

## Debugging

### Model produces garbage

- Check IntAcc during training (should be >90%)
- Ensure you trained long enough at QTemp=1.0
- Try smaller vocabulary

### Model always outputs same thing

- Not enough training variety
- Try more epochs
- Check for class imbalance in responses

**Class imbalance causes default behavior.** If 60% of your training data has the same response, the model learns to output that response when uncertain. Small models are especially prone to this - they lack capacity to learn nuanced decision boundaries, so they fall back to the statistical majority.

Fix: count your response distribution and balance before training. Aim for roughly equal representation of each response class, or at least no class dominating above 40%.

### Similar inputs produce same output

Trigram hashing with 128 buckets means long similar phrases collide. For example, "bigger than elephant" and "smaller than elephant" share most trigrams - the distinguishing "big"/"sma" signal gets drowned out.

**Short distinctive phrases work best.** Single keywords or 2-3 word phrases with unique character patterns. Long sentences with subtle differences will confuse the model.

### .COM file too big

- Reduce hidden layer sizes
- Reduce charset size
- Prune training data to smaller vocabulary

## Example Training Run

```
$ cat data/tinychat*.txt | python3 feedme.py --epochs 300 --chat

============================================================
Autoregressive Character Model Training
============================================================

Building charset from training data...
Charset (35 chars): " ABCDEFGHIJKLMNOPQRSTUVWXYZ?!'" + EOS

Loading training data...
Loaded 1847 query-response pairs

Model: 256 → 256 → 192 → 128 → 35
Parameters: 143,395

Training for 300 epochs...
Epoch  50: CE=0.0892, Acc=97.2%, IntAcc=84.3%, QTemp=0.42
Epoch 100: CE=0.0234, Acc=99.1%, IntAcc=91.7%, QTemp=0.58
Epoch 150: CE=0.0156, Acc=99.4%, IntAcc=95.2%, QTemp=0.73
Epoch 200: CE=0.0098, Acc=99.6%, IntAcc=97.8%, QTemp=0.88
Epoch 250: CE=0.0076, Acc=99.7%, IntAcc=99.1%, QTemp=1.00
Epoch 300: CE=0.0065, Acc=99.7%, IntAcc=99.4%, QTemp=1.00 *BEST*

============================================================
Best IntAcc: 99.4% at epoch 300
============================================================

Interactive Chat (type '!' to exit)
============================================================
> hello
HI
> are you real
MAYBE
> !

Bye!
```
