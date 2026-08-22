"""
Reference implementation of the Z80 inference pipeline, in NumPy.

This is the golden model: it reproduces exactly what the generated Z80 code is
supposed to compute, including 16-bit accumulator wrap-around and arithmetic
(floor) right shifts.  Tests compare emulator output against it.

It deliberately does *not* import torch — the build scripts and CI only need
NumPy, and the semantics modelled here are integer semantics, not training ones.

Note the difference from ``feedme.AutoregressiveModel._forward_int``: that one
truncates towards zero when shifting down, whereas ``SRA H / RR L`` on real
hardware floors.  For negative accumulators the two disagree by one.  This
module implements the hardware behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

NUM_BUCKETS = 128
CONTEXT_LEN = 8
BUCKET_WEIGHT = 32  # fixed-point scale: one occurrence == 32
MAX_OUTPUT_LEN = 50
SHIFT = 2  # per-layer arithmetic right shift


def _lower(ch: str) -> str:
    """Lowercase A-Z only, matching the Z80's `cp 'A' / add 20h` test."""
    return chr(ord(ch) + 0x20) if "A" <= ch <= "Z" else ch


def hash16(chars: str, seed: int = 0) -> int:
    h = seed
    for ch in chars:
        h = (h * 31 + ord(ch)) & 0xFFFF
    return h


def trigram_encode(text: str, num_buckets: int = NUM_BUCKETS) -> np.ndarray:
    """Encode a query into trigram-hash buckets, exactly as TOKENIZE does.

    The query is treated as if padded with a space at each end, so a query of
    n characters contributes n trigrams.
    """
    vec = np.zeros(num_buckets, dtype=np.int32)
    text = text.lstrip(" ")
    if not text:
        return vec
    chars = [_lower(c) for c in text]
    padded = [" ", *chars, " "]
    for i in range(len(padded) - 2):
        vec[hash16("".join(padded[i : i + 3])) % num_buckets] += BUCKET_WEIGHT
    return vec


def context_encode(
    recent: str, num_buckets: int = NUM_BUCKETS, context_len: int = CONTEXT_LEN
) -> np.ndarray:
    """Encode the last ``context_len`` emitted characters, as ENCODE_CTX does."""
    vec = np.zeros(num_buckets, dtype=np.int32)
    window = "".join(_lower(c) for c in recent)[-context_len:].rjust(context_len)
    for n in (1, 2, 3):
        for pos in range(context_len - n + 1):
            bucket = hash16(window[pos : pos + n], seed=pos * 7) % num_buckets
            vec[bucket] += BUCKET_WEIGHT
    return vec


@dataclass
class Model:
    """A quantized model: 2-bit weights in {-2,-1,0,1} plus int16 biases."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    charset: str

    @property
    def num_layers(self) -> int:
        return len(self.weights)

    @property
    def input_size(self) -> int:
        return int(self.weights[0].shape[1])

    @property
    def output_size(self) -> int:
        return int(self.weights[-1].shape[0])

    @property
    def eos_idx(self) -> int:
        return len(self.charset) - 1

    @property
    def layer_sizes(self) -> list[int]:
        return [self.input_size] + [int(w.shape[0]) for w in self.weights]

    @classmethod
    def from_params(cls, params: dict, charset: str) -> Model:
        names = sorted(
            {k.replace("_weight", "").replace("_bias", "") for k in params},
            key=lambda n: int(n[2:]),
        )
        return cls(
            weights=[np.asarray(params[f"{n}_weight"], dtype=np.int32) for n in names],
            biases=[np.asarray(params[f"{n}_bias"], dtype=np.int32) for n in names],
            charset=charset,
        )

    @classmethod
    def load(cls, path: str) -> Model:
        from loadmodel import load_model_params

        params, _arch, charset = load_model_params(path)
        return cls.from_params(params, charset)

    def architecture(self) -> dict:
        sizes = self.layer_sizes
        return {
            "input_size": sizes[0],
            "hidden_sizes": sizes[1:-1],
            "num_classes": sizes[-1],
        }

    def save_npz(self, path: str) -> None:
        out: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True), start=1):
            out[f"fc{i}_weight"] = w.astype(np.int8)
            out[f"fc{i}_bias"] = b.astype(np.int16)
        out["_architecture"] = np.array(json.dumps(self.architecture()).encode())
        out["_charset"] = np.array(self.charset.encode())
        np.savez(path, **out)


def wrap(v: np.ndarray | int, bits: int) -> np.ndarray | int:
    """Wrap to a signed value of ``bits`` width, as the accumulator does."""
    half = 1 << (bits - 1)
    return ((np.asarray(v, dtype=np.int64) + half) & ((1 << bits) - 1)) - half


def forward(model: Model, x: np.ndarray, accum_bits: int = 16) -> np.ndarray:
    """Run integer inference; returns the final layer.

    ``accum_bits`` is 16 for the Z80 targets and 24 for the eZ80, which has
    room for a wider accumulator and so never wraps in practice.  The two agree
    on any model whose activations stay inside 16 bits, which is exactly what
    the QAT overflow penalty trains for.
    """
    acc = np.asarray(x, dtype=np.int64)
    last = model.num_layers - 1
    for i, (w, bias) in enumerate(zip(model.weights, model.biases, strict=True)):
        acc = wrap(w.astype(np.int64) @ acc + bias.astype(np.int64), accum_bits)
        acc = acc >> SHIFT  # arithmetic shift: floors, like SRA H / RR L
        if i != last:
            acc = np.maximum(acc, 0)
    return acc.astype(np.int64)


#: Widest layer a Z80 backend can emit: its neuron loop counts in B, and DJNZ
#: treats a zero start as 256. The eZ80 backend uses sentinels, so it has no cap.
Z80_MAX_LAYER = 256


def discover_layers(params: dict) -> tuple[list[str], list[int]]:
    """Return (ordered layer names, [input, hidden..., output] sizes)."""
    names = sorted(
        {k.replace("_weight", "").replace("_bias", "") for k in params},
        key=lambda n: int(n[2:]),
    )
    sizes: list[int] = []
    for i, name in enumerate(names):
        w = params[f"{name}_weight"]
        if i == 0:
            sizes.append(int(w.shape[1]))
        sizes.append(int(w.shape[0]))
    return names, sizes


def validate_z80_layers(layer_sizes: list[int]) -> None:
    """Reject models a Z80 backend would mis-assemble rather than emit them."""
    oversized = [n for n in layer_sizes if n > Z80_MAX_LAYER]
    if oversized:
        raise ValueError(
            f"layer sizes {oversized} exceed the Z80 limit of {Z80_MAX_LAYER}; "
            f"build for eZ80 with buildez80.py, which has no such limit"
        )


def argmax(values: np.ndarray) -> int:
    """First-wins argmax, matching the strict `>` comparison in ARGMAX."""
    return int(np.argmax(values))


def generate(
    model: Model, query: str, max_len: int = MAX_OUTPUT_LEN, accum_bits: int = 16
) -> str:
    """Autoregressively generate a response, exactly as GENERATE does."""
    query_vec = trigram_encode(query)
    ctx_chars = " " * CONTEXT_LEN
    out: list[str] = []
    for _ in range(max_len):
        x = np.concatenate([query_vec, context_encode(ctx_chars)])
        idx = argmax(forward(model, x, accum_bits))
        if idx == model.eos_idx:
            break
        ch = model.charset[idx]
        out.append(ch)
        ctx_chars = (ctx_chars + _lower(ch))[-CONTEXT_LEN:]
    return "".join(out)


# --- weight packing ----------------------------------------------------------
#
# Two on-disk encodings exist.  "plain" is the natural one (value+2, LSB pair
# first).  "rotated" reorders the pairs so the unpack loop in buildz80com.py can
# fold the {-2,-1,0,+1} test into two DECs, putting the most common value (0) on
# the fastest path.

# Each output neuron's weights start on a fresh byte, because the unpack loop
# reloads a packed byte whenever its per-neuron counter hits a multiple of four.
# Rows whose length isn't a multiple of four are padded with zero weights.

_ROT_ENCODE = {-2: 0, -1: 3, 0: 1, 1: 2}
_ROT_DECODE = {v: k for k, v in _ROT_ENCODE.items()}


def row_stride(in_features: int) -> int:
    """Packed bytes per output neuron."""
    return (in_features + 3) // 4


def pack_2bit(weights: np.ndarray, layout: str = "rotated") -> bytes:
    """Pack [out, in] weights to 2 bits each, one row per whole number of bytes."""
    w = np.clip(np.atleast_2d(np.asarray(weights)), -2, 1).astype(np.int64)
    pad = (-w.shape[1]) % 4
    if pad:
        w = np.pad(w, ((0, 0), (0, pad)))
    out = bytearray()
    for row in w:
        for i in range(0, len(row), 4):
            q = row[i : i + 4]
            if layout == "plain":
                c = [int(v) + 2 for v in q]
                out.append(c[0] | (c[1] << 2) | (c[2] << 4) | (c[3] << 6))
            else:
                c = [_ROT_ENCODE[int(v)] for v in q]
                out.append((c[2] << 6) | (c[1] << 4) | (c[0] << 2) | c[3])
    return bytes(out)


def unpack_2bit(data: bytes, shape: tuple[int, int], layout: str = "rotated") -> np.ndarray:
    """Inverse of :func:`pack_2bit`, dropping each row's padding."""
    out_features, in_features = shape
    stride = row_stride(in_features)
    vals = np.zeros(shape, dtype=np.int8)
    for r in range(out_features):
        base = r * stride
        for i in range(in_features):
            byte = data[base + i // 4]
            slot = i % 4
            if layout == "plain":
                vals[r, i] = ((byte >> (2 * slot)) & 3) - 2
            else:
                vals[r, i] = _ROT_DECODE[(byte >> (2, 4, 6, 0)[slot]) & 3]
    return vals
