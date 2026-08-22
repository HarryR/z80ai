# Testing

```bash
pip install numpy pytest
pytest                  # everything
pytest -m "not slow"    # skip the full-size model runs (~20s instead of ~45s)
```

Nothing here needs PyTorch. Six tests that check the training-time encoders
against the reference are skipped if torch is absent.

## How it works

The suite does not check that the build scripts emit *some* bytes. It executes
the bytes.

```
  model.npz ──┬──> build script ──> Z80 machine code ──> libz80emu ──> text
              │                                                          ║
              └──> libinfer (NumPy reference) ─────────────────────────> text
                                                                     compare
```

- **`libz80emu.py`** is a pure-Python Z80 / eZ80 CPU. It decodes structurally
  (the `x/y/z/p/q` decomposition), so the base instruction set, CB/ED/DD/FD
  prefixes and ADL mode all fall out of a few hundred lines. Cycle counts come
  from the M-cycle structure — 4T per opcode fetch, 3T per memory access, plus
  explicit internal cycles — and are pinned against the documented Zilog
  timings in `test_emulator.py`.
- **`libhost.py`** supplies the CP/M BDOS, ZX Spectrum ROM and Agon MOS entry
  points the three platforms call, turning console traffic into Python strings.
- **`libinfer.py`** is the golden model: the same tokenizer, context encoder,
  quantized inference and argmax in NumPy, with the hardware's exact integer
  semantics (16-bit accumulator wrap, arithmetic-shift flooring).

## What each file covers

| File | What it pins down |
|---|---|
| `test_emulator.py` | The emulator itself: flags, block moves, indexed loads, DJNZ, documented cycle counts |
| `test_libz80.py` | Assembler: label resolution, relative-jump range, `align`, instruction encodings |
| `test_packing.py` | Both 2-bit weight layouts round-trip; every neuron starts on a byte boundary |
| `test_encoders.py` | Trigram/context hashing against an independent restatement of the spec |
| `test_kernels.py` | Emulator memory vs. reference, value by value: tokenizer buckets, context buckets, output logits |
| `test_end_to_end.py` | Generated text from all three Z80 targets vs. the reference |
| `test_ez80.py` | ADL encoding, the Agon header, and the same numeric comparisons for the eZ80 |
| `test_builders.py` | TAP container layout, TPA size limits, embedded weight/bias/charset data |
| `test_model_shapes.py` | Layer discovery order, and the widths a Z80 backend can actually assemble |
| `test_build_frontend.py` | Automatic target selection |

## Why comparing text is not enough

`test_kernels.py` reads the actual 16-bit values out of emulator memory rather
than comparing generated strings. With a small charset two quite different logit
vectors often argmax to the same character, so a text comparison will happily
pass over a broken kernel. Both of the arithmetic bugs found while writing these
tests — the `MULADD` borrow and the packed-weight row desync — produced correct
*text* on the models that first exercised them.

## Speed

End-to-end tests use a synthetic 256→16→12 model and generate 8 characters, so a
full forward pass is a few hundred thousand emulated instructions rather than a
few million. The shipped examples are covered by the `slow`-marked tests.

To time a target rather than check it:

```bash
python bench.py --model examples/guess/model.npz --target com fast ez80
```
