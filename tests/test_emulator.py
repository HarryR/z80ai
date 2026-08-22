"""Unit tests for the Z80 emulator itself.

Everything downstream trusts this, so the flag-level behaviour of the
instructions the builders emit is pinned here explicitly.
"""

from __future__ import annotations

import pytest

from libz80 import Z80Builder
from libz80emu import FLAG_C, FLAG_N, FLAG_PV, FLAG_S, FLAG_Z, Z80


def run(code: bytes, org: int = 0x0100, setup=None, max_cycles: int = 100_000) -> Z80:
    cpu = Z80()
    cpu.load(org, code)
    cpu.pc = org
    cpu.sp = 0xF000
    if setup:
        setup(cpu)
    cpu.run(max_cycles=max_cycles)
    return cpu


def asm(fn, org: int = 0x0100) -> bytes:
    b = Z80Builder(org=org)
    fn(b)
    b.halt()
    return b.build()


def test_ld_and_arithmetic():
    cpu = run(asm(lambda b: (b.ld_hl_nn(0x1234), b.ld_de_nn(0x1111), b.add_hl_de())))
    assert cpu.hl == 0x2345


def test_sbc_hl_de_borrow():
    def prog(b):
        b.ld_hl_nn(0x0000)
        b.ld_de_nn(0x0001)
        b.or_a()  # clear carry
        b.sbc_hl_de()

    cpu = run(asm(prog))
    assert cpu.hl == 0xFFFF
    assert cpu.f & FLAG_C  # borrow out


def test_sra_h_rr_l_is_arithmetic_shift():
    """SRA H / RR L must floor, not truncate toward zero."""
    for value, expected in ((-9 & 0xFFFF, -3 & 0xFFFF), (9, 2), (-1 & 0xFFFF, 0xFFFF)):
        def prog(b, v=value):
            b.ld_hl_nn(v)
            b.sra_h()
            b.rr_l()
            b.sra_h()
            b.rr_l()

        cpu = run(asm(prog))
        assert cpu.hl == expected, f"{value:04X} -> {cpu.hl:04X}, want {expected:04X}"


def test_djnz_counts_256_when_b_is_zero():
    def prog(b):
        b.ld_b_n(0)
        b.ld_c_n(0)
        b.label("LP")
        b.inc_hl()
        b.djnz("LP")

    cpu = run(asm(prog))
    assert cpu.hl == 256


def test_ldir_block_copy():
    def prog(b):
        b.ld_hl_label("SRC")
        b.ld_de_label("DST")
        b.ld_bc_nn(4)
        b.ldir()
        b.halt()
        b.label("SRC")
        b.db(1, 2, 3, 4)
        b.label("DST")
        b.db(0, 0, 0, 0)

    bld = Z80Builder(org=0x0100)
    prog(bld)
    cpu = run(bld.build())
    dst = bld.labels["DST"]
    assert [cpu.peek(dst + i) for i in range(4)] == [1, 2, 3, 4]


def test_exx_swaps_all_three_pairs():
    def prog(b):
        b.ld_bc_nn(0x1122)
        b.ld_de_nn(0x3344)
        b.ld_hl_nn(0x5566)
        b.exx()
        b.ld_bc_nn(0xAABB)
        b.ld_de_nn(0xCCDD)
        b.ld_hl_nn(0xEEFF)
        b.exx()

    cpu = run(asm(prog))
    assert (cpu.bc, cpu.de, cpu.hl) == (0x1122, 0x3344, 0x5566)
    cpu2 = run(asm(lambda b: (prog(b), b.exx())))
    assert (cpu2.bc, cpu2.de, cpu2.hl) == (0xAABB, 0xCCDD, 0xEEFF)


def test_indexed_load_store():
    def prog(b):
        b.ld_ix_nn(0x2000)
        b.ld_hl_nn(0xBEEF)
        b.ld_ixd_l(0)
        b.ld_ixd_h(1)

    cpu = run(asm(prog))
    assert cpu.peek(0x2000) == 0xEF and cpu.peek(0x2001) == 0xBE


def test_bit_7_d_detects_negative():
    cpu = run(asm(lambda b: (b.ld_de_nn(0x8000), b.bit_7_d())))
    assert not (cpu.f & FLAG_Z)
    cpu = run(asm(lambda b: (b.ld_de_nn(0x7F00), b.bit_7_d())))
    assert cpu.f & FLAG_Z


def test_cp_sets_flags_without_touching_a():
    cpu = run(asm(lambda b: (b.ld_a_n(5), b.cp_n(7))))
    assert cpu.a == 5
    assert cpu.f & FLAG_C and cpu.f & FLAG_N and cpu.f & FLAG_S


def test_jp_m_uses_sign_flag():
    def prog(b):
        b.ld_hl_nn(1)
        b.ld_de_nn(2)
        b.or_a()
        b.sbc_hl_de()
        b.ld_a_n(0)
        b.jp_m("NEG")
        b.ld_a_n(1)
        b.label("NEG")

    cpu = run(asm(prog))
    assert cpu.a == 0


def test_call_and_ret_roundtrip():
    def prog(b):
        b.ld_a_n(1)
        b.call("SUB")
        b.halt()
        b.label("SUB")
        b.inc_a()
        b.ret()

    bld = Z80Builder(org=0x0100)
    prog(bld)
    cpu = run(bld.build())
    assert cpu.a == 2
    assert cpu.sp == 0xF000  # stack balanced


def test_rst_vectors_to_low_memory():
    cpu = Z80()
    cpu.load(0x0100, bytes([0xCF]))  # RST 08h
    cpu.poke(0x0008, 0x76)  # HALT
    cpu.pc = 0x0100
    cpu.sp = 0xF000
    cpu.run(max_cycles=100)
    assert cpu.pc == 0x0009 and cpu.halted


def test_parity_flag_on_logic_ops():
    cpu = run(asm(lambda b: (b.ld_a_n(0x03), b.and_n(0xFF))))
    assert cpu.f & FLAG_PV  # 0x03 has even parity
    cpu = run(asm(lambda b: (b.ld_a_n(0x07), b.and_n(0xFF))))
    assert not (cpu.f & FLAG_PV)


def test_cycle_counts_match_documented_timings():
    """Spot-check the M-cycle model against the Zilog timings."""
    cases = [
        (lambda b: b.nop(), 4),
        (lambda b: b.ld_a_n(1), 7),
        (lambda b: b.ld_hl_nn(0), 10),
        (lambda b: b.ld_a_hl(), 7),
        (lambda b: b.add_hl_de(), 11),
        (lambda b: b.push_hl(), 11),
        (lambda b: b.pop_hl(), 10),
        (lambda b: b.inc_hl(), 6),
        (lambda b: b.sbc_hl_de(), 15),
        (lambda b: b.ld_l_ixd(0), 19),
    ]
    for emit, expected in cases:
        bld = Z80Builder(org=0x0100)
        emit(bld)
        cpu = Z80()
        cpu.load(0x0100, bld.build())
        cpu.pc = 0x0100
        cpu.sp = 0xF000
        cpu.step()
        assert cpu.tstates == expected, f"{bld.code.hex()} took {cpu.tstates}, want {expected}"


def test_runaway_program_raises():
    from libz80emu import Z80Error

    with pytest.raises(Z80Error):
        run(bytes([0x18, 0xFE]), max_cycles=1000)  # JR -2, forever
