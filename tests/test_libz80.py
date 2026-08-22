"""Tests for the Z80Builder assembler."""

from __future__ import annotations

import pytest

from libz80 import Z80Builder, add_debug_utils, add_print_decimal, add_print_string


def test_org_and_addr_track_emitted_bytes():
    b = Z80Builder(org=0x0100)
    assert b.addr() == 0x0100
    b.nop()
    b.nop()
    assert b.addr() == 0x0102


def test_absolute_fixups_resolve_to_label_addresses():
    b = Z80Builder(org=0x0100)
    b.jp("TARGET")
    b.nop()
    b.label("TARGET")
    b.ret()
    image = b.build()
    assert image[0] == 0xC3
    assert image[1] | (image[2] << 8) == 0x0104


def test_relative_fixups_are_offsets_from_the_next_instruction():
    b = Z80Builder(org=0x0100)
    b.label("TOP")
    b.nop()
    b.jr("TOP")
    image = b.build()
    assert image[2] == 0xFD  # -3


def test_forward_relative_jump():
    b = Z80Builder(org=0x0100)
    b.jr("AHEAD")
    b.nop()
    b.nop()
    b.label("AHEAD")
    image = b.build()
    assert image[1] == 2


def test_unknown_label_is_an_error():
    b = Z80Builder()
    b.jp("NOWHERE")
    with pytest.raises(ValueError, match="Unknown label"):
        b.build()


def test_relative_jump_out_of_range_is_an_error():
    b = Z80Builder()
    b.jr("FAR")
    b.ds(200)
    b.label("FAR")
    with pytest.raises(ValueError, match="out of range"):
        b.build()


def test_align_is_a_no_op_when_already_aligned():
    b = Z80Builder(org=0x0100)
    b.align(256)
    assert b.addr() == 0x0100
    b.nop()
    b.align(256)
    assert b.addr() == 0x0200


def test_emit_masks_to_bytes():
    b = Z80Builder()
    b.emit(0x1FF)
    assert b.code == bytearray([0xFF])


def test_emit_word_is_little_endian():
    b = Z80Builder()
    b.emit_word(0xBEEF)
    assert b.code == bytearray([0xEF, 0xBE])


@pytest.mark.parametrize(
    "emit,expected",
    [
        (lambda b: b.nop(), b"\x00"),
        (lambda b: b.ret(), b"\xc9"),
        (lambda b: b.halt(), b"\x76"),
        (lambda b: b.di(), b"\xf3"),
        (lambda b: b.ei(), b"\xfb"),
        (lambda b: b.exx(), b"\xd9"),
        (lambda b: b.ex_de_hl(), b"\xeb"),
        (lambda b: b.ldir(), b"\xed\xb0"),
        (lambda b: b.ld_hl_nn(0x1234), b"\x21\x34\x12"),
        (lambda b: b.ld_de_nn(0x1234), b"\x11\x34\x12"),
        (lambda b: b.ld_bc_nn(0x1234), b"\x01\x34\x12"),
        (lambda b: b.ld_ix_nn(0x1234), b"\xdd\x21\x34\x12"),
        (lambda b: b.ld_iy_nn(0x1234), b"\xfd\x21\x34\x12"),
        (lambda b: b.ld_a_n(0x42), b"\x3e\x42"),
        (lambda b: b.ld_hl_n(0x42), b"\x36\x42"),
        (lambda b: b.sbc_hl_de(), b"\xed\x52"),
        (lambda b: b.sbc_hl_bc(), b"\xed\x42"),
        (lambda b: b.sra_h(), b"\xcb\x2c"),
        (lambda b: b.rr_l(), b"\xcb\x1d"),
        (lambda b: b.srl_e(), b"\xcb\x3b"),
        (lambda b: b.bit_7_d(), b"\xcb\x7a"),
        (lambda b: b.ld_iyd_l(1), b"\xfd\x75\x01"),
        (lambda b: b.ld_l_ixd(-1), b"\xdd\x6e\xff"),
        (lambda b: b.rst(0x10), b"\xd7"),
        (lambda b: b.ld_mem_label_sp("X"), b"\xed\x73\x00\x00"),
    ],
)
def test_instruction_encodings(emit, expected):
    b = Z80Builder()
    emit(b)
    assert bytes(b.code) == expected


def test_ascii_and_data_directives():
    b = Z80Builder()
    b.ascii("HI")
    b.db(1, 2)
    b.dw(0x1234)
    b.ds(2)
    assert bytes(b.code) == b"HI\x01\x02\x34\x12\x00\x00"


def test_debug_helpers_assemble_cleanly():
    b = Z80Builder()
    add_debug_utils(b)
    add_print_string(b)
    add_print_decimal(b)
    b.build()  # resolves without raising
    for label in ("PRHEX", "PRNYB", "DBGBUF", "PRCRLF", "PRMSG", "PRDEC"):
        assert label in b.labels
