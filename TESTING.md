# Testing

```bash
pip install -r requirements-dev.txt
pytest                  # everything
pytest -m "not slow"    # skip the full-size model runs (~20s instead of ~45s)
```

Nothing here needs PyTorch. Six tests that check the training-time encoders
against the reference are skipped if torch is absent. Python 3.10 or newer.

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
| `test_ez80_kernels.py` | All three eZ80 kernels against the reference *and* against each other, plus the flooring/ReLU boundary, the active-column list, and kernel selection |
| `test_ez80_argmax.py` | That the output layer is scanned in full however wide it is |
| `test_ez80_emulator.py` | eZ80-only opcodes and the two flag facts the kernels branch on |
| `test_builders.py` | TAP container layout, TPA size limits, embedded weight/bias/charset data |
| `test_model_shapes.py` | Layer discovery order, and the widths a Z80 backend can actually assemble |
| `test_build_frontend.py` | Automatic target selection |
| `test_verify_artifacts.py` | The release verifier's own failure paths |
| `test_libnn.py` | Layer planning, the Platform contract, and that the public API stays annotated |
| `test_codegen_stability.py` | The exact bytes each backend emits, by hash |

## In CI

`.github/workflows/ci.yml` runs four jobs. Beyond the test suite, the ones that
matter are in `build`:

| Check | What it stops |
|---|---|
| `ruff check` | pycodestyle, pyflakes, bugbear, comprehensions, simplifications, modern syntax, import order and performance rules. Two justified `noqa`s remain, both explained inline - see `ruff.toml` |
| Test matrix | Python 3.10 through 3.13, `fail-fast: false` so one version failing doesn't hide the rest |
| Reproducible build | Builds everything twice and compares byte for byte. A build that picks up dict ordering or a hash seed would stop matching what anyone can rebuild |
| Artifact verification | Boots every release binary in the emulator and compares its output to the reference. **A binary that assembles but computes the wrong answer cannot reach a release** |

That last one is the reason any of this exists. Run it yourself:

```bash
./build-examples.sh dist
python verify_artifacts.py --dist dist
```

It found the ZX Spectrum load-address bug — both shipped `.TAP` files were
assembled past the top of RAM and could not load on any Spectrum.

## Why comparing text is not enough

`test_kernels.py` reads the actual 16-bit values out of emulator memory rather
than comparing generated strings. With a small charset two quite different logit
vectors often argmax to the same character, so a text comparison will happily
pass over a broken kernel. Both of the arithmetic bugs found while writing these
tests — the `MULADD` borrow and the packed-weight row desync — produced correct
*text* on the models that first exercised them.

The strongest technique available needs no reference model at all. The eZ80
backend emits the same network three ways — a runtime weight stream, unrolled
weight-major, and unrolled column-major — and
`test_ez80_kernels.py::test_all_kernels_agree_bit_for_bit` requires all three to
produce a byte-identical `OUTBUF`. Three independently generated programs
agreeing exactly is very hard to achieve by accident, and it stays meaningful
for models where writing down the expected answer by hand would not.

## When a correct test is not enough

Some optimizations are invisible to correctness tests, because getting them
wrong only makes the program slower. The column-major kernel skips inputs whose
activation is zero; appending one anyway is harmless, since its column adds zero
everywhere. Every numeric test still passes, and the speedup is gone.

`test_hidden_layer_list_holds_exactly_its_nonzero_activations` and
`test_activation_that_floors_to_zero_is_not_listed` check the *list* rather than
the answer. The second exists because the first missed the case: it used a
natural model, and the only way a neuron reaches the shift and still comes out
zero is a pre-shift accumulator in `[0, 3]` — ReLU catches everything more
negative earlier. That band has to be arranged deliberately.

If an optimization can regress without failing anything, it needs a test that
looks at the mechanism, not the output.

## Speed

End-to-end tests use a synthetic 256→16→12 model and generate 8 characters, so a
full forward pass is a few hundred thousand emulated instructions rather than a
few million. The shipped examples are covered by the `slow`-marked tests.

To time a target rather than check it:

```bash
python bench.py --model examples/guess/model.npz --target com fast ez80
```
