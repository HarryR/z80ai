"""Every eZ80 kernel must compute exactly the same numbers.

The backend emits the same network several ways - walked at runtime, or
unrolled into straight-line code - and they are only interchangeable if they
agree bit for bit.  Two things are checked here:

  * each kernel against ``libinfer`` at ``accum_bits=24``, and
  * the kernels against *each other*, which needs no reference model at all
    and is the strongest signal available: two independently generated
    programs producing byte-identical OUTBUF is very hard to achieve by
    accident.

Splitting the accumulator into positive and negative halves, and reassociating
the sum, is exact rather than approximate: the reference wraps to 24 bits, and
addition mod 2**24 is associative and commutative.  These tests are what turn
that argument into evidence.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import read24, reference_input, run_ez80_until

import buildez80
import libinfer
from libez80 import AGON_MAX_IMAGE
from libhost import run_agon
from libinfer import Model

KERNELS = ["row", "compact"]
QUERIES = ["HELLO", "ARE YOU A ROBOT", "X"]
GEN_LEN = 6


def build(path: str, kernel: str, max_output_len: int = GEN_LEN):
    return buildez80.build_autoreg(path, max_output_len=max_output_len, kernel=kernel)


# --- against the reference ---------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("query", QUERIES)
def test_logits_match_reference(tiny_model_path, tiny_model, kernel, query):
    builder = build(tiny_model_path, kernel)
    cpu = run_ez80_until(builder, query, "ARGMAX")
    got = read24(cpu, builder.labels["OUTBUF"], tiny_model.output_size)
    want = libinfer.forward(tiny_model, reference_input(query), accum_bits=24)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("kernel", KERNELS)
def test_odd_widths_match_reference(odd_model_path, odd_model, kernel):
    """Widths that are not multiples of four, where packing bugs like to hide."""
    builder = build(odd_model_path, kernel)
    cpu = run_ez80_until(builder, "HELLO", "ARGMAX")
    got = read24(cpu, builder.labels["OUTBUF"], odd_model.output_size)
    want = libinfer.forward(odd_model, reference_input("HELLO"), accum_bits=24)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("kernel", KERNELS)
def test_hidden_layers_match_reference(tmp_path, model_factory, kernel):
    """Compare every intermediate buffer, so a failure names the layer."""
    model = model_factory([256, 20, 14, 9], seed=41)
    path = str(tmp_path / "deep.npz")
    model.save_npz(path)
    builder = build(path, kernel)

    expected = libinfer.forward_layers(model, reference_input("HELLO"), accum_bits=24)
    sizes = model.layer_sizes[1:]
    for i in range(len(expected) - 1):
        # Stopping at the next layer's entry means layer i has just finished.
        cpu = run_ez80_until(builder, "HELLO", f"LAYER{i + 2}")
        buf = "BUF_A" if (i + 1) % 2 == 1 else "BUF_B"
        got = read24(cpu, builder.labels[buf], sizes[i])
        np.testing.assert_array_equal(got, expected[i], err_msg=f"layer {i + 1}")


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("query", QUERIES)
def test_generated_text_matches_reference(tiny_model_path, tiny_model, kernel, query):
    image = build(tiny_model_path, kernel).build()
    out, _host = run_agon(image, stdin=[query, "!"], max_cycles=400_000_000)
    assert libinfer.generate(tiny_model, query, GEN_LEN, accum_bits=24) in out


# --- against each other ------------------------------------------------------


@pytest.mark.parametrize("query", QUERIES)
def test_all_kernels_agree_bit_for_bit(tiny_model_path, tiny_model, query):
    """The strongest check: no reference model involved, just two programs."""
    outputs = {}
    for kernel in KERNELS:
        builder = build(tiny_model_path, kernel)
        cpu = run_ez80_until(builder, query, "ARGMAX")
        outputs[kernel] = read24(
            cpu, builder.labels["OUTBUF"], tiny_model.output_size
        )
    first, *rest = KERNELS
    for kernel in rest:
        np.testing.assert_array_equal(
            outputs[kernel], outputs[first],
            err_msg=f"{kernel} disagrees with {first}",
        )


# --- arithmetic edges --------------------------------------------------------


def _single_neuron_model(bias: int, charset: str = " A\x00") -> Model:
    """One layer, all weights zero, so each logit is exactly its bias >> 2."""
    n = len(charset)
    weights = [np.zeros((n, 256), dtype=np.int32)]
    biases = [np.full(n, bias, dtype=np.int32)]
    return Model(weights=weights, biases=biases, charset=charset)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("bias", [-8, -5, -4, -3, -1, 0, 1, 3, 4, 7])
def test_shift_and_relu_at_the_boundary(tmp_path, kernel, bias):
    """>>2 floors, so a negative accumulator relus to zero - pin both edges.

    The unrolled kernel short-circuits the whole shift chain when the
    accumulator is negative.  That is only valid because floor(v/4) < 0 for
    every v < 0, which is exactly what these values probe.
    """
    model = _single_neuron_model(bias)
    path = str(tmp_path / f"bias{bias}.npz")
    model.save_npz(path)
    builder = build(path, kernel, max_output_len=1)
    cpu = run_ez80_until(builder, "HELLO", "ARGMAX")
    got = read24(cpu, builder.labels["OUTBUF"], model.output_size)
    want = libinfer.forward(model, reference_input("HELLO"), accum_bits=24)
    np.testing.assert_array_equal(got, want)
    # The output layer has no ReLU, so a negative bias must survive.
    assert (got < 0).any() == (bias < 0)


@pytest.mark.parametrize("kernel", KERNELS)
def test_output_layer_keeps_negative_logits(tmp_path, model_factory, kernel):
    model = model_factory([256, 12], charset=" AB\x00", seed=77)
    model.biases[-1][:] = -900
    path = str(tmp_path / "neg.npz")
    model.save_npz(path)

    builder = build(path, kernel, max_output_len=1)
    cpu = run_ez80_until(builder, "HELLO", "ARGMAX")
    got = read24(cpu, builder.labels["OUTBUF"], model.output_size)
    want = libinfer.forward(model, reference_input("HELLO"), accum_bits=24)
    assert (want < 0).any(), "the fixture failed to produce negative logits"
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("kernel", KERNELS)
def test_neuron_with_no_nonzero_weights(tmp_path, model_factory, kernel):
    """An all-zero row emits two epilogues back to back with no MACs between."""
    model = model_factory([256, 10], charset=" AB\x00", seed=53)
    model.weights[0][3, :] = 0
    model.weights[0][4, :] = 0
    path = str(tmp_path / "holes.npz")
    model.save_npz(path)

    builder = build(path, kernel, max_output_len=1)
    cpu = run_ez80_until(builder, "HELLO", "ARGMAX")
    got = read24(cpu, builder.labels["OUTBUF"], model.output_size)
    want = libinfer.forward(model, reference_input("HELLO"), accum_bits=24)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("kernel", KERNELS)
def test_single_layer_model(tmp_path, model_factory, kernel):
    """One layer means the output epilogue is the only one used."""
    model = model_factory([256], charset=" ABCD\x00", seed=61)
    path = str(tmp_path / "flat.npz")
    model.save_npz(path)

    builder = build(path, kernel, max_output_len=1)
    cpu = run_ez80_until(builder, "HELLO", "ARGMAX")
    got = read24(cpu, builder.labels["OUTBUF"], model.output_size)
    want = libinfer.forward(model, reference_input("HELLO"), accum_bits=24)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("kernel", KERNELS)
def test_layer_wider_than_the_z80_limit(tmp_path, model_factory, kernel):
    """DJNZ caps Z80 layers at 256 neurons; neither eZ80 kernel may."""
    model = model_factory([256, 300, 40], charset=" ABC\x00", seed=23)
    path = str(tmp_path / "wide.npz")
    model.save_npz(path)

    builder = build(path, kernel, max_output_len=2)
    cpu = run_ez80_until(builder, "HELLO", "ARGMAX")
    got = read24(cpu, builder.labels["OUTBUF"], model.output_size)
    want = libinfer.forward(model, reference_input("HELLO"), accum_bits=24)
    np.testing.assert_array_equal(got, want)


# --- the encoding ------------------------------------------------------------


def test_neuron_ops_roundtrips():
    rng = np.random.default_rng(3)
    for _ in range(20):
        row = rng.choice([-2, -1, 0, 1], size=37)
        rebuilt = np.zeros(37, dtype=np.int64)
        for col, weight in buildez80.neuron_ops(row):
            rebuilt[col] = weight
        np.testing.assert_array_equal(rebuilt, np.clip(row, -2, 1))


def test_neuron_ops_handles_degenerate_rows():
    assert buildez80.neuron_ops(np.zeros(16, dtype=np.int8)) == []
    allneg = buildez80.neuron_ops(np.full(4, -2, dtype=np.int8))
    assert allneg == [(0, -2), (1, -2), (2, -2), (3, -2)]


def test_neuron_ops_clips_out_of_range_weights():
    """Anything outside {-2,-1,0,1} is a training bug, not a new weight value."""
    assert buildez80.neuron_ops(np.array([7, -9, 0], dtype=np.int32)) == [
        (0, 1), (1, -2)
    ]


def test_neuron_ops_is_ascending_and_deterministic():
    """Column order is what makes the build reproducible - never iterate a set."""
    rng = np.random.default_rng(9)
    row = rng.choice([-2, -1, 0, 1], size=200)
    first = buildez80.neuron_ops(row)
    assert first == buildez80.neuron_ops(row)
    assert [c for c, _ in first] == sorted(c for c, _ in first)
    assert len({c for c, _ in first}) == len(first)


@pytest.mark.parametrize("kernel", KERNELS)
def test_build_is_byte_reproducible(tiny_model_path, kernel):
    assert build(tiny_model_path, kernel).build() == build(
        tiny_model_path, kernel
    ).build()


# --- layout ------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
def test_activation_buffers_are_contiguous(tiny_model_path, kernel):
    """Layer 0 addresses all 256 inputs through the INBUF label alone."""
    builder = build(tiny_model_path, kernel)
    assert builder.labels["CTXBUF"] == builder.labels["INBUF"] + 128 * 3


def test_unrolled_build_has_no_out_of_range_relative_jumps(guess_model_path):
    """resolve() raises on an unreachable JR; this pins the emission order."""
    build(guess_model_path, "row", max_output_len=1).build()


# --- kernel selection --------------------------------------------------------


def test_auto_picks_the_unrolled_kernel_when_it_fits(tiny_model_path):
    builder = buildez80.build_autoreg(tiny_model_path, max_output_len=1)
    assert "NEUREND" in builder.labels
    assert "WTS1" not in builder.labels, "auto should not have chosen compact"


def test_auto_falls_back_to_compact_when_unrolling_would_not_fit(
    tmp_path, model_factory
):
    """The compact kernel exists precisely for models too large to unroll."""
    model = model_factory([256, 700, 700], charset=" ABC\x00", seed=3)
    path = str(tmp_path / "huge.npz")
    model.save_npz(path)

    assert len(build(path, "row", max_output_len=1).build()) > AGON_MAX_IMAGE
    builder = buildez80.build_autoreg(path, max_output_len=1)
    assert "WTS1" in builder.labels, "auto should have fallen back to compact"


def test_unknown_kernel_is_rejected(tiny_model_path):
    with pytest.raises(ValueError, match="unknown kernel"):
        buildez80.build_autoreg(tiny_model_path, kernel="nope")


@pytest.mark.slow
@pytest.mark.parametrize("example", ["guess", "tinychat"])
def test_shipped_examples_fit_agon_sram(examples_dir, example):
    import os

    path = os.path.join(examples_dir, example, "model.npz")
    if not os.path.exists(path):
        pytest.skip(f"{example} model not present")
    image = buildez80.build_autoreg(path).build()
    assert len(image) <= AGON_MAX_IMAGE, (
        f"{example} is {len(image):,} bytes, more than the "
        f"{AGON_MAX_IMAGE:,} an Agon can load"
    )


@pytest.mark.slow
def test_full_model_matches_reference_on_every_kernel(guess_model_path):
    model = libinfer.Model.load(guess_model_path)
    expected = libinfer.generate(model, "IS IT AN ANIMAL", 3, accum_bits=24)
    for kernel in KERNELS:
        image = build(guess_model_path, kernel, max_output_len=3).build()
        out, _host = run_agon(
            image, stdin=["IS IT AN ANIMAL", "!"], max_cycles=2_000_000_000
        )
        assert expected in out, f"{kernel}: {out!r}"
