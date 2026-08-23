"""eZ80 ADL-mode instruction behaviour the unrolled kernels depend on.

The kernels branch on flags left behind by SBC and deliberately do *not*
branch after ADD.  Those two facts are load-bearing and were previously
unpinned, so a plausible-looking "optimization" that moved a JP M after an ADD
would have passed every test in the suite while silently changing the model's
output.
"""

from __future__ import annotations

import pytest

from libez80 import AGON_LOAD_ADDR, EZ80Builder
from libz80emu import FLAG_S, FLAG_Z, Z80

ORG = AGON_LOAD_ADDR
SCRATCH = 0x060000


def run(fn, setup=None, max_cycles: int = 100_000) -> Z80:
    b = EZ80Builder(org=ORG)
    fn(b)
    b.halt()
    cpu = Z80(adl=True, mem_size=0x080000)
    cpu.load(ORG, b.build())
    cpu.pc = ORG
    cpu.sp = 0x070000
    if setup:
        setup(cpu)
    cpu.run(max_cycles=max_cycles)
    return cpu


def poke24(cpu: Z80, addr: int, val: int) -> None:
    for k in range(3):
        cpu.poke(addr + k, (val >> (8 * k)) & 0xFF)


def peek24(cpu: Z80, addr: int) -> int:
    return cpu.peek_word(addr, 3)


# --- register-pair indexed loads (eZ80 only) ---------------------------------


@pytest.mark.parametrize("disp", [0, 3, 127, -1, -128])
def test_ld_hl_ixd_reads_24_bits(disp):
    cpu = run(
        lambda b: (b.ld_ix_nn(SCRATCH), b.ld_hl_ixd(disp)),
        lambda c: poke24(c, SCRATCH + disp, 0xABCDEF),
    )
    assert cpu.hl == 0xABCDEF


@pytest.mark.parametrize("disp", [0, 5, -7])
def test_ld_ixd_hl_writes_24_bits(disp):
    cpu = run(lambda b: (b.ld_ix_nn(SCRATCH), b.ld_hl_nn(0x123456), b.ld_ixd_hl(disp)))
    assert peek24(cpu, SCRATCH + disp) == 0x123456


def test_iy_forms_use_iy_not_ix():
    """FD selects IY; a decoder that ignored the prefix would read from IX."""
    cpu = run(
        lambda b: (b.ld_ix_nn(SCRATCH), b.ld_iy_nn(SCRATCH + 0x100), b.ld_hl_iyd(0)),
        lambda c: (poke24(c, SCRATCH, 0x111111), poke24(c, SCRATCH + 0x100, 0x222222)),
    )
    assert cpu.hl == 0x222222


def test_ld_iyd_hl_round_trips():
    cpu = run(
        lambda b: (
            b.ld_iy_nn(SCRATCH),
            b.ld_hl_nn(0x778899),
            b.ld_iyd_hl(3),
            b.ld_hl_nn(0),
            b.ld_hl_iyd(3),
        )
    )
    assert cpu.hl == 0x778899


def test_de_forms_round_trip():
    cpu = run(
        lambda b: (
            b.ld_ix_nn(SCRATCH),
            b.ld_de_nn(0x0F0E0D),
            b.ld_ixd_de(0),
            b.ld_de_nn(0),
            b.ld_de_ixd(0),
        )
    )
    assert cpu.de == 0x0F0E0D


def test_indexed_pair_loads_do_not_exist_outside_adl():
    """On a plain Z80 these encodings are RLCA/RRCA with the prefix ignored."""
    b = EZ80Builder(org=0x0100)
    b.ld_hl_ixd(0)
    b.halt()
    cpu = Z80(adl=False)  # 16-bit mode
    cpu.load(0x0100, bytes(b.code) + b"\x76")
    cpu.pc = 0x0100
    cpu.hl = 0x1234
    cpu.run(max_cycles=1000)
    assert cpu.hl == 0x1234, "the ADL-only decode leaked into Z80 mode"


# --- 24-bit index arithmetic -------------------------------------------------


def test_add_ix_de_wraps_at_24_bits():
    cpu = run(lambda b: (b.ld_ix_nn(0xFFFFFF), b.ld_de_nn(2), b.add_ix_de()))
    assert cpu.ix == 0x000001


def test_add_iy_de_advances_the_output_cursor():
    cpu = run(lambda b: (b.ld_iy_nn(0x001000), b.ld_de_nn(3), b.add_iy_de()))
    assert cpu.iy == 0x001003


# --- the two flag facts NEUREND rests on -------------------------------------


def test_sbc_hl_de_sets_sign_from_bit_23_in_adl_mode():
    """`JP M` after the accumulator subtraction is how ReLU short-circuits."""
    cpu = run(lambda b: (b.ld_hl_nn(1), b.ld_de_nn(2), b.or_a(), b.sbc_hl_de()))
    assert cpu.hl == 0xFFFFFF
    assert cpu.f & FLAG_S, "S must come from bit 23, not bit 15"


def test_sbc_hl_de_leaves_sign_clear_on_a_positive_result():
    cpu = run(lambda b: (b.ld_hl_nn(5), b.ld_de_nn(2), b.or_a(), b.sbc_hl_de()))
    assert cpu.hl == 3
    assert not cpu.f & FLAG_S


def test_sbc_hl_de_sets_zero_on_equality():
    """First-wins argmax depends on distinguishing equal from greater."""
    cpu = run(lambda b: (b.ld_hl_nn(9), b.ld_de_nn(9), b.or_a(), b.sbc_hl_de()))
    assert cpu.f & FLAG_Z


def test_add_hl_de_does_not_disturb_sign_or_zero():
    """A JP M must never be moved to sit after an ADD - it would read stale flags."""
    cpu = run(
        lambda b: (
            b.ld_hl_nn(1),
            b.ld_de_nn(2),
            b.or_a(),
            b.sbc_hl_de(),  # sets S
            b.ld_de_nn(2),
            b.add_hl_de(),  # result is positive, but S must survive
        )
    )
    assert cpu.hl == 1
    assert cpu.f & FLAG_S


def test_ld_ix_nn_does_not_disturb_flags():
    """NEUREND resets the positive accumulator between the SBC and the JP M."""
    cpu = run(
        lambda b: (
            b.ld_hl_nn(1),
            b.ld_de_nn(2),
            b.or_a(),
            b.sbc_hl_de(),
            b.ld_ix_nn(0),
        )
    )
    assert cpu.f & FLAG_S
