"""ARGMAX must scan the whole output layer, however wide it is.

The eZ80 backend advertises unlimited layer widths, but ARGMAX used to count
neurons in B and DJNZ: a 299-entry charset assembled to ``LD B,43`` and picked
the largest of the first 44 logits.  Nothing failed - it just answered with the
wrong character.  These tests pin the width the docs promise.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import read24, run_ez80_until

import buildez80
import libinfer

# A charset wider than a byte counter can index.  The last entry is EOS.
WIDE_CHARSET = "".join(chr(0x21 + (i % 90)) for i in range(299)) + "\x00"


@pytest.fixture(scope="module")
def wide_output_model(model_factory):
    """256 -> 24 -> 300 outputs: only the output layer is oversized."""
    return model_factory([256, 24], charset=WIDE_CHARSET, seed=17)


@pytest.fixture(scope="module")
def wide_output_build(tmp_path_factory, wide_output_model):
    path = str(tmp_path_factory.mktemp("wide") / "wide.npz")
    wide_output_model.save_npz(path)
    return buildez80.build_autoreg(path, max_output_len=2)


def test_argmax_selects_the_true_maximum_over_a_300_wide_output(
    wide_output_build, wide_output_model
):
    """The regression: the winner must be findable past index 255."""
    cpu = run_ez80_until(wide_output_build, "HELLO", "PRINTCH")
    logits = read24(cpu, wide_output_build.labels["OUTBUF"], wide_output_model.output_size)

    x = np.concatenate(
        [libinfer.trigram_encode("HELLO"), libinfer.context_encode(" " * 8)]
    )
    np.testing.assert_array_equal(
        logits, libinfer.forward(wide_output_model, x, accum_bits=24)
    )

    want = libinfer.argmax(logits)
    got = read24(cpu, wide_output_build.labels["RESULT"], 1)[0]
    assert got == want


def test_argmax_result_is_not_truncated_to_a_byte(wide_output_build, wide_output_model):
    """Plant a maximum beyond 255 directly, so the test does not depend on the model."""
    cpu = run_ez80_until(wide_output_build, "HELLO", "ARGMAX")

    out = wide_output_build.labels["OUTBUF"]
    size = wide_output_model.output_size
    for i in range(size):
        for k in range(3):
            cpu.poke(out + 3 * i + k, 0)
    winner = 280
    assert winner < size
    cpu.poke(out + 3 * winner, 0x7F)  # comfortably the largest value present

    cpu.run(max_cycles=1_000_000, stop_pc=wide_output_build.labels["PRINTCH"])
    assert read24(cpu, wide_output_build.labels["RESULT"], 1)[0] == winner


def test_argmax_is_first_wins_on_ties(wide_output_build, wide_output_model):
    """libinfer.argmax takes the earliest maximum; ARGMAX must agree."""
    cpu = run_ez80_until(wide_output_build, "HELLO", "ARGMAX")

    out = wide_output_build.labels["OUTBUF"]
    for i in range(wide_output_model.output_size):
        for k in range(3):
            cpu.poke(out + 3 * i + k, 0)
    for i in (37, 200, 291):
        cpu.poke(out + 3 * i, 0x11)

    cpu.run(max_cycles=1_000_000, stop_pc=wide_output_build.labels["PRINTCH"])
    assert read24(cpu, wide_output_build.labels["RESULT"], 1)[0] == 37


def test_argmax_handles_an_all_negative_output_layer(wide_output_build, wide_output_model):
    """No ReLU on the output layer, so every logit may be negative."""
    cpu = run_ez80_until(wide_output_build, "HELLO", "ARGMAX")

    out = wide_output_build.labels["OUTBUF"]
    size = wide_output_model.output_size
    for i in range(size):
        value = (-1000 - i) & 0xFFFFFF
        for k in range(3):
            cpu.poke(out + 3 * i + k, (value >> (8 * k)) & 0xFF)
    # index 0 is the least negative, so it wins
    cpu.run(max_cycles=1_000_000, stop_pc=wide_output_build.labels["PRINTCH"])
    assert read24(cpu, wide_output_build.labels["RESULT"], 1)[0] == 0


def test_byte_immediates_are_range_checked():
    """The assembler-level guard that would have caught the ARGMAX bug."""
    from libez80 import EZ80Builder

    b = EZ80Builder()
    with pytest.raises(ValueError, match="out of range"):
        b.ld_b_n(299)
    with pytest.raises(ValueError, match="out of range"):
        b.cp_n(299)
    b.ld_b_n(255)  # the boundary is still legal
