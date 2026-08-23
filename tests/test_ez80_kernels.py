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

KERNELS = ["column", "row", "compact"]
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


def test_auto_picks_the_fastest_kernel_when_it_fits(tiny_model_path):
    builder = buildez80.build_autoreg(tiny_model_path, max_output_len=1)
    assert builder.kernel == buildez80.KERNELS[0]


def test_each_kernel_records_which_one_it_is(tiny_model_path):
    for kernel in KERNELS:
        assert build(tiny_model_path, kernel).kernel == kernel


def test_auto_falls_back_to_compact_when_unrolling_would_not_fit(
    tmp_path, model_factory
):
    """The compact kernel exists precisely for models too large to unroll."""
    model = model_factory([256, 700, 700], charset=" ABC\x00", seed=3)
    path = str(tmp_path / "huge.npz")
    model.save_npz(path)

    for kernel in KERNELS[:-1]:
        assert len(build(path, kernel, max_output_len=1).build()) > AGON_MAX_IMAGE
    assert buildez80.build_autoreg(path, max_output_len=1).kernel == "compact"


def test_auto_steps_down_one_rung_at_a_time(tmp_path, model_factory, monkeypatch):
    """A ceiling that rules out only the fastest kernel must not skip to the last."""
    model = model_factory([256, 24, 18], seed=13)
    path = str(tmp_path / "mid.npz")
    model.save_npz(path)

    row_size = len(build(path, "row", max_output_len=1).build())
    monkeypatch.setattr(buildez80, "AGON_MAX_IMAGE", row_size)
    assert buildez80.build_autoreg(path, max_output_len=1).kernel == "row"


def test_unknown_kernel_is_rejected(tiny_model_path):
    with pytest.raises(ValueError, match="unknown kernel"):
        buildez80.build_autoreg(tiny_model_path, kernel="nope")


# --- column-major specifics --------------------------------------------------


def test_active_column_list_holds_exactly_the_nonzero_inputs(tiny_model_path):
    """The list is the whole optimization: wrong contents means wrong answers."""
    builder = build(tiny_model_path, "column")
    cpu = run_ez80_until(builder, "HELLO", "LAYER1")

    x = reference_input("HELLO")
    want = [
        builder.labels[f"COL1_{j}"] for j in range(len(x)) if x[j] != 0
    ]
    base = builder.labels["COLLIST"]
    got = [cpu.peek_word(base + 3 * i, 3) for i in range(len(want))]
    assert got == want

    terminator = cpu.peek_word(base + 3 * len(want), 3)
    assert terminator == builder.labels["LEPI1"], "list is not terminated"


def test_active_column_list_has_no_duplicates(tiny_model_path):
    """BUCKET_ADD can hit one bucket twice; appending there would double it."""
    builder = build(tiny_model_path, "column")
    cpu = run_ez80_until(builder, "ABABABAB", "LAYER1")

    base = builder.labels["COLLIST"]
    entries = []
    for i in range(257):
        value = cpu.peek_word(base + 3 * i, 3)
        if value == builder.labels["LEPI1"]:
            break
        entries.append(value)
    else:
        pytest.fail("never found the terminator")
    assert len(entries) == len(set(entries))


def test_hidden_layer_list_holds_exactly_its_nonzero_activations(
    tmp_path, model_factory
):
    """The later lists are built by the epilogue, and that is the whole win.

    Appending a zero activation is numerically harmless - its column adds zero
    everywhere - so no correctness test can notice it.  Only this one can, and
    without it the sparsity could regress silently into a slower kernel that
    still passes everything else.
    """
    model = model_factory([256, 32, 20], seed=91)
    path = str(tmp_path / "two.npz")
    model.save_npz(path)
    builder = build(path, "column", max_output_len=1)

    # Stop once layer 1's epilogue has finished building layer 2's list.
    cpu = run_ez80_until(builder, "HELLO", "LAYER2")

    activations = libinfer.forward_layers(
        model, reference_input("HELLO"), accum_bits=24
    )[0]
    want = [
        builder.labels[f"COL2_{j}"]
        for j in range(len(activations))
        if activations[j] != 0
    ]
    assert want, "the fixture produced an all-zero hidden layer"
    assert len(want) < len(activations), "nothing was skipped, so nothing is tested"

    base = builder.labels["COLLIST"]
    got = [cpu.peek_word(base + 3 * i, 3) for i in range(len(want))]
    assert got == want
    assert cpu.peek_word(base + 3 * len(want), 3) == builder.labels["LEPI2"]


def test_activation_that_floors_to_zero_is_not_listed(tmp_path, model_factory):
    """A small *positive* accumulator shifts to zero, and must not be listed.

    ReLU catches the negative case earlier, so this narrow band - pre-shift
    accumulator in [0, 3] - is the only way a neuron reaches the shift and
    still comes out zero.  Testing the list on a natural model misses it
    entirely, so the accumulator is arranged to land there on purpose.
    """
    model = model_factory([256, 24, 16], seed=5)
    # Neuron 7 of the hidden layer: no inputs at all, bias 2, so 2 >> 2 == 0.
    model.weights[0][7, :] = 0
    model.biases[0][7] = 2
    path = str(tmp_path / "floor.npz")
    model.save_npz(path)

    activations = libinfer.forward_layers(
        model, reference_input("HELLO"), accum_bits=24
    )[0]
    assert activations[7] == 0, "the fixture did not land in the flooring band"

    builder = build(path, "column", max_output_len=1)
    cpu = run_ez80_until(builder, "HELLO", "LAYER2")

    base = builder.labels["COLLIST"]
    listed = []
    for i in range(len(activations) + 1):
        value = cpu.peek_word(base + 3 * i, 3)
        if value == builder.labels["LEPI2"]:
            break
        listed.append(value)
    assert builder.labels["COL2_7"] not in listed


def test_column_major_with_an_all_zero_input(tmp_path, model_factory):
    """No active columns at all: the accumulators must still hold the biases."""
    model = model_factory([256, 10], charset=" AB\x00", seed=67)
    path = str(tmp_path / "zeroin.npz")
    model.save_npz(path)
    builder = build(path, "column", max_output_len=1)

    # Stop before the input scan, blank the whole input vector, and let INFER
    # rebuild the column list over it - which should come out empty.
    cpu = run_ez80_until(builder, "HELLO", "INFER")
    for addr in range(builder.labels["INBUF"], builder.labels["INBUF"] + 256 * 3):
        cpu.poke(addr, 0)

    # The scan should produce a list holding nothing but its terminator. Check
    # that before the epilogue reuses the buffer for the next layer's list.
    cpu.run(max_cycles=10_000_000, stop_pc=builder.labels["LAYER1"])
    assert cpu.pc == builder.labels["LAYER1"]
    assert cpu.peek_word(builder.labels["COLLIST"], 3) == builder.labels["LEPI1"]

    cpu.run(max_cycles=10_000_000, stop_pc=builder.labels["ARGMAX"])
    assert cpu.pc == builder.labels["ARGMAX"]

    got = read24(cpu, builder.labels["OUTBUF"], model.output_size)
    want = libinfer.forward(model, np.zeros(256, dtype=np.int64), accum_bits=24)
    np.testing.assert_array_equal(got, want)


def test_column_feeding_more_than_256_neurons(tmp_path, model_factory):
    """No DJNZ-shaped cap on how many neurons one input may feed."""
    model = model_factory([256, 300], charset=" AB\x00", seed=71)
    model.weights[0][:, 5] = 1  # column 5 feeds all 300 neurons
    path = str(tmp_path / "fan.npz")
    model.save_npz(path)

    builder = build(path, "column", max_output_len=1)
    cpu = run_ez80_until(builder, "HELLO", "ARGMAX")
    got = read24(cpu, builder.labels["OUTBUF"], model.output_size)
    want = libinfer.forward(model, reference_input("HELLO"), accum_bits=24)
    np.testing.assert_array_equal(got, want)


def test_column_kernel_leaves_the_stack_where_it_found_it(tiny_model_path):
    """Threading on IY rather than SP is what keeps CALL and RET usable."""
    builder = build(tiny_model_path, "column")
    before = run_ez80_until(builder, "HELLO", "LAYER1").sp
    after = run_ez80_until(builder, "HELLO", "ARGMAX").sp
    assert before == after


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
