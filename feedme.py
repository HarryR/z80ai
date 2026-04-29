#!/usr/bin/env python3
"""
Autoregressive character-level model for Z80.

Instead of classifying into response categories, this model generates
responses character-by-character:

1. Input: query_trigrams[128] + context[128] = 256 dimensions
2. Output: next_char probabilities[64]
3. Loop: run inference, emit char, update context, repeat

The context encodes the last few output characters using the same
trigram hashing approach as the query.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from collections import Counter

from libqat import OverflowAwareLinear


# Character set - built dynamically from training data
# EOS is always last character
EOS_CHAR = '\x00'

def build_charset_from_pairs(pairs: List[Tuple[str, str]]) -> str:
    """Build minimal charset from loaded query-response pairs."""
    chars = set()
    for query, response in pairs:
        chars.update(response.upper())  # Normalize to uppercase

    # Sort for consistency: space first, then A-Z, then 0-9, then punctuation
    chars.discard(EOS_CHAR)  # Remove if present, we add it last

    letters = sorted(c for c in chars if c.isalpha())
    digits = sorted(c for c in chars if c.isdigit())
    space = [' '] if ' ' in chars else []
    punct = sorted(c for c in chars if not c.isalnum() and c != ' ')

    charset = ''.join(space + letters + digits + punct) + EOS_CHAR
    return charset


# These are set dynamically from training data
CHARSET = ""
CHAR_TO_IDX = {}
IDX_TO_CHAR = {}
EOS_IDX = 0
NUM_CHARS = 0


def char_to_idx(c: str) -> int:
    """Convert character to index, defaulting to space for unknown."""
    c_upper = c.upper()
    if c_upper in CHAR_TO_IDX:
        return CHAR_TO_IDX[c_upper]
    elif c in CHAR_TO_IDX:
        return CHAR_TO_IDX[c]
    else:
        return 0  # space for unknown


def idx_to_char(i: int) -> str:
    """Convert index to character."""
    return IDX_TO_CHAR.get(i, ' ')


class TrigramEncoder:
    """Encode text into trigram hash buckets (integer-friendly, no normalization)."""

    def __init__(self, num_buckets: int = 128):
        self.num_buckets = num_buckets

    def _hash_trigram(self, trigram: str) -> int:
        """Hash a trigram to a bucket index."""
        h = 0
        for c in trigram:
            h = (h * 31 + ord(c)) & 0xFFFF
        return h % self.num_buckets

    def encode(self, text: str) -> np.ndarray:
        """Encode text into bucket counts (raw counts, Z80-compatible)."""
        vec = np.zeros(self.num_buckets, dtype=np.float32)
        text = text.lower()
        text = ' ' + text + ' '  # Pad for boundary trigrams

        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            bucket = self._hash_trigram(trigram)
            vec[bucket] += 1.0

        # No normalization - use raw counts for Z80 compatibility
        return vec


class ContextEncoder:
    """Encode recent output characters into hash buckets (integer-friendly)."""

    def __init__(self, num_buckets: int = 128, context_len: int = 8):
        self.num_buckets = num_buckets
        self.context_len = context_len

    def _hash_ngram(self, ngram: str, offset: int = 0) -> int:
        """Hash an n-gram with position offset."""
        h = offset * 7
        for c in ngram:
            h = (h * 31 + ord(c)) & 0xFFFF
        return h % self.num_buckets

    def encode(self, recent_chars: str) -> np.ndarray:
        """Encode recent output characters (raw counts, Z80-compatible)."""
        vec = np.zeros(self.num_buckets, dtype=np.float32)

        # Pad to context_len
        recent = recent_chars[-self.context_len:].lower()
        recent = recent.rjust(self.context_len)

        # Hash character n-grams with position info
        for n in [1, 2, 3]:  # Unigrams, bigrams, trigrams
            for i in range(len(recent) - n + 1):
                ngram = recent[i:i+n]
                bucket = self._hash_ngram(ngram, offset=i)
                vec[bucket] += 1.0

        # No normalization - use raw counts for Z80 compatibility
        return vec


def create_training_examples(query: str, response: str,
                            query_encoder: TrigramEncoder,
                            context_encoder: ContextEncoder) -> List[Tuple[np.ndarray, int]]:
    """
    Create training examples from a (query, response) pair.

    For response "hello", creates:
    - (query + context(""), 'h')
    - (query + context("h"), 'e')
    - (query + context("he"), 'l')
    - ...
    - (query + context("hello"), EOS)
    """
    examples = []
    query_vec = query_encoder.encode(query)

    # Add EOS to response
    response_with_eos = response + "\x00"

    output_so_far = ""
    for char in response_with_eos:
        # Encode current context
        context_vec = context_encoder.encode(output_so_far)

        # Combine query and context
        full_input = np.concatenate([query_vec, context_vec])

        # Target is next character (or EOS)
        target = char_to_idx(char) if char != "\x00" else EOS_IDX

        examples.append((full_input, target))
        output_so_far += char

    return examples




class AutoregressiveModel(nn.Module):
    """Autoregressive character model with configurable depth."""

    def __init__(self, input_size: int = 256, hidden_sizes: list = [128, 128],
                 num_chars: int = 64):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_chars = num_chars

        # Build layers dynamically
        self.layers = nn.ModuleList()
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(OverflowAwareLinear(prev_size, hidden_size))
            prev_size = hidden_size
        self.layers.append(OverflowAwareLinear(prev_size, num_chars))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, use_int: bool = False, quant_temp: float = 1.0) -> torch.Tensor:
        if use_int:
            return self._forward_int(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, quant_temp=quant_temp)
            x = self.relu(x)
        x = self.layers[-1](x, quant_temp=quant_temp)
        return x

    def _forward_int(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass simulating Z80 integer inference (16-bit accumulator)."""
        # Scale input like Z80 does
        x = (x * 32).round()

        for i, layer in enumerate(self.layers):
            # Quantize weights to {-2, -1, 0, +1} (4 values for 2 bits)
            w = layer.weight
            scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
            w_quant = torch.clamp(torch.round(w / scale), -2, 1)

            # Quantize bias
            b_quant = torch.round(layer.bias * 32)

            # Integer matmul with 16-bit overflow simulation
            x = x @ w_quant.T + b_quant
            # Simulate 16-bit signed overflow (wrap around)
            x = ((x + 32768) % 65536) - 32768

            # Shift down (divide by 4, arithmetic right shift)
            x = torch.div(x, 4, rounding_mode='trunc')

            # ReLU (except last layer)
            if i < len(self.layers) - 1:
                x = torch.relu(x)

        return x

    def get_overflow_stats(self) -> dict:
        return {f'layer{i+1}': layer.get_overflow_risk()
                for i, layer in enumerate(self.layers)}

    def reset_overflow_stats(self):
        for layer in self.layers:
            layer.reset_overflow_stats()

    def compute_quantization_loss(self) -> torch.Tensor:
        return sum(layer.get_quantization_loss() for layer in self.layers)

    def compute_total_overflow_penalty(self, x: torch.Tensor) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=x.device)
        for i, layer in enumerate(self.layers[:-1]):
            penalty = penalty + layer.compute_overflow_penalty(x)
            x = self.relu(layer(x))
        penalty = penalty + self.layers[-1].compute_overflow_penalty(x)
        return penalty

    def get_quantized_params(self) -> dict:
        """Extract 2-bit quantized weights."""
        params = {}

        for i, layer in enumerate(self.layers):
            name = f'fc{i+1}'
            with torch.no_grad():
                w = layer.weight
                w_scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
                w_scaled = w / w_scale
                w_quant = torch.clamp(torch.round(w_scaled), -2, 1).cpu().numpy().astype(np.int8)

                b = layer.bias
                b_quant = torch.round(b * 32).cpu().numpy().astype(np.int16)

                params[f'{name}_weight'] = w_quant
                params[f'{name}_bias'] = b_quant

        return params


def generate_response(model: AutoregressiveModel, query: str,
                     query_encoder: TrigramEncoder,
                     context_encoder: ContextEncoder,
                     max_len: int = 50, use_int: bool = True) -> str:
    """Generate a response character by character."""
    model.eval()

    query_vec = query_encoder.encode(query)
    output = ""

    with torch.no_grad():
        for _ in range(max_len):
            context_vec = context_encoder.encode(output)
            full_input = np.concatenate([query_vec, context_vec])
            x = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0)

            logits = model(x, use_int=use_int)
            next_char_idx = logits.argmax(dim=1).item()

            # Stop on EOS
            if next_char_idx == EOS_IDX:
                break

            next_char = idx_to_char(next_char_idx)
            output += next_char

    return output.strip()




def parse_pair(line: str) -> Tuple[str, str] | None:
    """Parse a single line into (query, response) or None if invalid."""
    line = line.strip()
    if '|' not in line:
        return None

    parts = line.split('|', 1)
    if len(parts) != 2:
        return None

    query = parts[0].strip().upper()
    response = parts[1].strip().upper()

    if len(query) >= 2 and len(response) >= 1:
        # Truncate smartly
        if len(query) > 60:
            query = query[:60].rsplit(' ', 1)[0] if ' ' in query[40:60] else query[:60]
        if len(response) > 50:
            response = response[:50].rsplit(' ', 1)[0] if ' ' in response[30:50] else response[:50]
        return (query, response)

    return None


def load_chunk(stdin, chunk_size: int = 0) -> List[Tuple[str, str]]:
    """Load up to chunk_size pairs from stdin (0 = all)."""
    pairs = []
    for line in stdin:
        pair = parse_pair(line)
        if pair:
            pairs.append(pair)
            if chunk_size > 0 and len(pairs) >= chunk_size:
                break
    return pairs


def validate_charset(pairs: List[Tuple[str, str]], charset: str) -> None:
    """Error if pairs contain characters not in charset."""
    allowed = set(charset)
    for query, response in pairs:
        for c in response:
            if c not in allowed:
                raise ValueError(f"Character '{c}' (ord {ord(c)}) in response '{response}' not in charset. "
                               f"Charset was built from first chunk and cannot change.")


def parse_hidden_sizes(spec: str) -> list[int]:
    vals = [int(x.strip()) for x in spec.split(',') if x.strip()]
    if not vals:
        raise ValueError("hidden size list cannot be empty")
    if any(v <= 0 or v > 255 for v in vals):
        raise ValueError("hidden sizes must be in range 1..255")
    return vals


def train_chunked(chunk_size: int = 1000, epochs_per_chunk: int = 100, lr: float = 0.01,
                  save_best: bool = False, hidden_sizes: list[int] | None = None,
                  checkpoint_file: str = 'command_model_autoreg.pt'):
    """Train incrementally on chunks of data from stdin."""
    global CHARSET, CHAR_TO_IDX, IDX_TO_CHAR, EOS_IDX, NUM_CHARS
    import sys
    import time

    print("=" * 60)
    print("Loading training data...")

    # Load all pairs upfront (cheap) to know totals
    all_pairs = load_chunk(sys.stdin, 0)  # 0 = load all
    total_pairs = len(all_pairs)

    if total_pairs == 0:
        print("No training data!")
        return None

    # Calculate chunks
    if chunk_size <= 0:
        chunk_size = total_pairs
    total_chunks = (total_pairs + chunk_size - 1) // chunk_size

    print(f"Loaded {total_pairs} pairs → {total_chunks} chunks of {chunk_size}")
    print(f"Epochs per chunk: {epochs_per_chunk}")
    print("=" * 60)

    # Build charset from ALL pairs (ensures consistency)
    CHARSET = build_charset_from_pairs(all_pairs)
    CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
    IDX_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}
    EOS_IDX = len(CHARSET) - 1
    NUM_CHARS = len(CHARSET)
    print(f"Charset ({NUM_CHARS} chars): {repr(CHARSET[:-1])} + EOS")

    query_encoder = TrigramEncoder(num_buckets=128)
    context_encoder = ContextEncoder(num_buckets=128, context_len=8)
    if hidden_sizes is None:
        hidden_sizes = [256, 192, 128]

    model = None
    total_epochs = 0
    best_int_acc = 0.0
    best_epoch = 0
    best_state = None

    # Try to resume from checkpoint
    try:
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        arch = checkpoint.get('architecture', {})
        if arch.get('num_classes') == NUM_CHARS and arch.get('hidden_sizes') == hidden_sizes:
            model = AutoregressiveModel(input_size=256, hidden_sizes=hidden_sizes, num_chars=NUM_CHARS)
            model.load_state_dict(checkpoint['model_state'])
            total_epochs = checkpoint.get('total_epochs', 0)
            best_int_acc = checkpoint.get('best_int_acc', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            print(f"Resumed from checkpoint: {total_epochs} epochs, best IntAcc: {best_int_acc:.1%}")
        else:
            print("Architecture changed, starting fresh")
    except FileNotFoundError:
        print("No checkpoint found, starting fresh")
    except Exception as e:
        print(f"Couldn't load checkpoint: {e}, starting fresh")

    # Initialize model if needed
    if model is None:
        model = AutoregressiveModel(input_size=256, hidden_sizes=hidden_sizes, num_chars=NUM_CHARS)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: 256 → {' → '.join(map(str, hidden_sizes))} → {NUM_CHARS}")
        print(f"Parameters: {total_params:,}")

    # Process in chunks
    for chunk_num in range(total_chunks):
        start_idx = chunk_num * chunk_size
        end_idx = min(start_idx + chunk_size, total_pairs)
        chunk = all_pairs[start_idx:end_idx]

        print(f"\n--- Chunk {chunk_num + 1}/{total_chunks}: {len(chunk)} pairs ---")

        # Generate examples for this chunk
        all_examples = []
        for query, response in chunk:
            examples = create_training_examples(query, response, query_encoder, context_encoder)
            all_examples.extend(examples)

        print(f"Generated {len(all_examples)} character examples")

        X = torch.tensor(np.stack([ex[0] for ex in all_examples]), dtype=torch.float32)
        y = torch.tensor(np.array([ex[1] for ex in all_examples]), dtype=torch.long)

        # Train on this chunk
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        interrupted = False
        for epoch in range(epochs_per_chunk):
            try:
                model.train()
                model.reset_overflow_stats()
                optimizer.zero_grad()

                quant_temp = 0.3 + 0.7 * min(1.0, epoch / (epochs_per_chunk * 0.8))

                outputs = model(X, quant_temp=quant_temp)
                ce_loss = criterion(outputs, y)
                quant_loss = model.compute_quantization_loss() * 0.10
                overflow_loss = model.compute_total_overflow_penalty(X) * 0.03

                loss = ce_loss + quant_loss + overflow_loss
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    with torch.no_grad():
                        preds = outputs.argmax(dim=1)
                        acc = (preds == y).float().mean()
                        int_outputs = model(X, use_int=True)
                        int_preds = int_outputs.argmax(dim=1)
                        int_acc = (int_preds == y).float().mean()

                        current_epoch = total_epochs + epoch + 1
                        if int_acc.item() > best_int_acc:
                            best_int_acc = int_acc.item()
                            best_epoch = current_epoch
                            best_state = {k: v.clone() for k, v in model.state_dict().items()}
                            marker = " *BEST*"
                        else:
                            marker = ""

                        print(f"  Epoch {current_epoch}: CE={ce_loss.item():.4f}, Acc={acc:.1%}, IntAcc={int_acc:.1%}{marker}")

            except KeyboardInterrupt:
                print("\nInterrupted!")
                interrupted = True
                break

        total_epochs += epoch + 1  # Count actual epochs completed

        # Save after each chunk
        if save_best and best_state:
            save_state = best_state
            save_note = "best"
        else:
            save_state = model.state_dict()
            save_note = "latest"

        torch.save({
            'model_state': save_state,
            'architecture': {
                'input_size': 256,
                'hidden_sizes': hidden_sizes,
                'num_classes': NUM_CHARS,
            },
            'charset': CHARSET,
            'total_epochs': total_epochs,
            'best_int_acc': best_int_acc,
            'best_epoch': best_epoch,
        }, checkpoint_file)
        print(f"Saved {save_note} (epochs: {total_epochs}, best: {best_int_acc:.1%} @ {best_epoch})")

        if interrupted:
            break

    print(f"\n{'=' * 60}")
    print(f"Finished: {chunk_num + 1}/{total_chunks} chunks, {total_epochs} total epochs")
    print(f"Best IntAcc: {best_int_acc:.1%} at epoch {best_epoch}")
    print("=" * 60)

    return model


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Train autoregressive model')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Epochs to train (per chunk if chunked)')
    parser.add_argument('--file', '-f', type=str, default=None, help='Training data file (default: stdin)')
    parser.add_argument('--chunk', '-c', type=int, default=0, help='Chunk size for streaming (0 = load all as one chunk)')
    parser.add_argument('--save-best', action='store_true', help='Save best model instead of latest')
    parser.add_argument('--hidden-sizes', type=str, default='256,192,128',
                        help='Comma-separated hidden layer sizes (e.g. 128,96,64)')
    parser.add_argument('--output', '-o', type=str, default='command_model_autoreg.pt',
                        help='Checkpoint output path')
    parser.add_argument('--chat', action='store_true', help='Interactive chat after training')
    args = parser.parse_args()

    # If file specified, redirect stdin from file
    if args.file:
        import io
        with open(args.file) as f:
            sys.stdin = io.StringIO(f.read())

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    model = train_chunked(
        chunk_size=args.chunk,
        epochs_per_chunk=args.epochs,
        save_best=args.save_best,
        hidden_sizes=hidden_sizes,
        checkpoint_file=args.output,
    )

    # Interactive chat session
    if args.chat:
        print("\n" + "=" * 60)
        print("Interactive Chat (type '!' to exit)")
        print("=" * 60)

        query_encoder = TrigramEncoder()
        context_encoder = ContextEncoder()

        while True:
            try:
                query = input("> ").strip()
                if not query:
                    continue
                if query == '!':
                    break
                response = generate_response(model, query, query_encoder, context_encoder, max_len=50)
                print(response)
            except (EOFError, KeyboardInterrupt):
                break

        print("\nBye!")
