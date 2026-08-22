"""eZ80 (ADL mode) backend tests.

The eZ80 build runs the same model as the Z80 builds but with 24-bit pointers
and a 24-bit accumulator, so it is checked against the reference at
``accum_bits=24`` and cross-checked against the Z80 build wherever the 16-bit
accumulator never wraps.
"""

from __future__ import annotations

import buildez80
import buildz80com
import libinfer
import numpy as np
import pytest
from libez80 import AGON_LOAD_ADDR, EZ80Builder
from libhost import AgonHost, run_agon, run_cpm
from libz80emu import Z80

GEN_LEN = 6
QUERIES = ["HELLO", "ARE YOU A ROBOT", "X"]


# --- the builder itself ------------------------------------------------------


def test_adl_immediates_are_three_bytes():
    b = EZ80Builder(org=0x040000)
    b.ld_hl_nn(0x123456)
    assert bytes(b.code) == b"\x21\x56\x34\x12"


def test_adl_label_fixups_are_three_bytes():
    b = EZ80Builder(org=0x040000)
    b.jp("T")
    b.label("T")
    b.ret()
    image = b.build()
    assert image[:4] == b"\xc3\x04\x00\x04"


@pytest.mark.parametrize(
    "emit,setup,read",
    [
        (lambda b: b.mlt_hl(), lambda c: setattr(c, "hl", 0x0C0D), lambda c: c.hl),
        (lambda b: b.mlt_de(), lambda c: setattr(c, "de", 0xFF02), lambda c: c.de),
        (lambda b: b.mlt_bc(), lambda c: setattr(c, "bc", 0x1001), lambda c: c.bc),
    ],
)
def test_mlt_multiplies_the_register_halves(emit, setup, read):
    """MLT is what a wider weight format would be built on."""
    b = EZ80Builder(org=0x040000)
    emit(b)
    b.halt()
    cpu = Z80(adl=True, mem_size=0x050000)
    cpu.load(0x040000, b.build())
    cpu.pc = 0x040000
    setup(cpu)
    before = read(cpu)
    cpu.run(max_cycles=1000)
    assert read(cpu) == (before >> 8) * (before & 0xFF)


def test_data_words_stay_sixteen_bit_in_adl_mode():
    b = EZ80Builder()
    b.dw(0x1234)
    assert bytes(b.code) == b"\x34\x12"
    b2 = EZ80Builder()
    b2.d24(0x123456)
    assert bytes(b2.code) == b"\x56\x34\x12"


def test_agon_header_is_present_and_well_formed(tiny_model_path):
    image = buildez80.build_autoreg(tiny_model_path).build()
    assert image[0] == 0xC3  # JP entry
    assert image[0x40:0x43] == b"MOS"
    assert image[0x43] == 0x00  # header version
    assert image[0x44] == 0x01  # ADL mode


def test_weight_stream_encoding_roundtrips():
    w = np.array([[-2, -1, 0, 1], [1, 1, -2, 0]], dtype=np.int8)
    blob = buildez80.encode_weights(w)
    rows, cur = [], []
    for byte in blob:
        if byte == buildez80.W_END_LAYER:
            break
        if byte == buildez80.W_END_NEURON:
            rows.append(cur)
            cur = []
        else:
            cur.append(byte - 256 if byte > 127 else byte)
    np.testing.assert_array_equal(np.array(rows), w)


def test_biases_are_sign_extended_to_24_bits():
    blob = buildez80.encode_biases(np.array([-1, 1, -300], dtype=np.int16))
    assert blob[0:3] == b"\xff\xff\xff"
    assert blob[3:6] == b"\x01\x00\x00"
    assert blob[6:9] == bytes(((-300) & 0xFFFFFF).to_bytes(3, "little"))


# --- execution ---------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_ez80(tiny_model_path):
    return buildez80.build_autoreg(tiny_model_path, max_output_len=GEN_LEN)


def agon_reply(builder, query: str) -> str:
    out, host = run_agon(builder.build(), stdin=[query, "!"], max_cycles=400_000_000)
    return out


@pytest.mark.parametrize("query", QUERIES)
def test_ez80_matches_reference(tiny_ez80, tiny_model, query):
    expected = libinfer.generate(tiny_model, query, GEN_LEN, accum_bits=24)
    assert expected in agon_reply(tiny_ez80, query)


def test_ez80_and_z80_agree_when_the_accumulator_does_not_wrap(
    tiny_model, tiny_model_path
):
    narrow = [libinfer.generate(tiny_model, q, GEN_LEN, 16) for q in QUERIES]
    wide = [libinfer.generate(tiny_model, q, GEN_LEN, 24) for q in QUERIES]
    if narrow != wide:
        pytest.skip("this model overflows 16 bits, so the targets must differ")

    ez80 = buildez80.build_autoreg(tiny_model_path, max_output_len=GEN_LEN)
    z80 = buildz80com.build_autoreg(tiny_model_path, max_output_len=GEN_LEN).build()
    for query in QUERIES:
        cpm_out, _ = run_cpm(z80, cmdline=query, max_cycles=400_000_000)
        assert cpm_out in agon_reply(ez80, query)


def _read24(cpu: Z80, addr: int, count: int) -> np.ndarray:
    vals = []
    for i in range(count):
        v = sum(cpu.peek(addr + 3 * i + k) << (8 * k) for k in range(3))
        vals.append(v - 0x1000000 if v & 0x800000 else v)
    return np.array(vals, dtype=np.int64)


def run_until(builder, query: str, label: str) -> Z80:
    host = AgonHost(stdin=[query, "!"])
    cpu = host.cpu
    cpu.load(AGON_LOAD_ADDR, builder.build())
    cpu.pc = AGON_LOAD_ADDR
    cpu.run(max_cycles=400_000_000, stop_pc=builder.labels[label])
    assert cpu.pc == builder.labels[label]
    return cpu


@pytest.mark.parametrize("query", QUERIES)
def test_ez80_tokenizer_matches_reference(tiny_ez80, query):
    cpu = run_until(tiny_ez80, query, "ARGMAX")
    got = _read24(cpu, tiny_ez80.labels["INBUF"], 128)
    np.testing.assert_array_equal(got, libinfer.trigram_encode(query))


def test_ez80_context_matches_reference(tiny_ez80):
    cpu = run_until(tiny_ez80, "HELLO", "ARGMAX")
    got = _read24(cpu, tiny_ez80.labels["CTXBUF"], 128)
    np.testing.assert_array_equal(got, libinfer.context_encode(" " * 8))


@pytest.mark.parametrize("query", QUERIES)
def test_ez80_logits_match_reference(tiny_ez80, tiny_model, query):
    cpu = run_until(tiny_ez80, query, "ARGMAX")
    got = _read24(cpu, tiny_ez80.labels["OUTBUF"], tiny_model.output_size)
    x = np.concatenate([libinfer.trigram_encode(query), libinfer.context_encode(" " * 8)])
    np.testing.assert_array_equal(got, libinfer.forward(tiny_model, x, accum_bits=24))


def test_ez80_handles_layers_wider_than_the_z80_limit(tmp_path, model_factory):
    """DJNZ caps Z80 layers at 256 neurons; ADL mode has no such limit."""
    model = model_factory([256, 300, 40], charset=" ABC\x00", seed=23)
    path = str(tmp_path / "wide.npz")
    model.save_npz(path)

    builder = buildez80.build_autoreg(path, max_output_len=2)
    cpu = run_until(builder, "HELLO", "ARGMAX")
    got = _read24(cpu, builder.labels["OUTBUF"], model.output_size)
    x = np.concatenate([libinfer.trigram_encode("HELLO"), libinfer.context_encode(" " * 8)])
    np.testing.assert_array_equal(got, libinfer.forward(model, x, accum_bits=24))


def test_ez80_builder_accepts_models_the_z80_builders_reject(tmp_path, model_factory):
    """The Z80 builders raise on a layer wider than 256; the eZ80 one must not."""
    model = model_factory([256, 300, 8], charset=" AB\x00", seed=31)
    path = str(tmp_path / "wide.npz")
    model.save_npz(path)
    with pytest.raises(ValueError, match="exceed the Z80 limit"):
        buildz80com.build_autoreg(path)
    buildez80.build_autoreg(path, max_output_len=1).build()


def test_ez80_binary_may_exceed_64k(guess_model_path):
    """The whole point of the port: models that cannot fit a Z80 at all."""
    image = buildez80.build_autoreg(guess_model_path).build()
    assert len(image) > 0x10000
    assert AGON_LOAD_ADDR + len(image) < 0x1000000


@pytest.mark.slow
def test_ez80_full_model_matches_reference(guess_model_path):
    model = libinfer.Model.load(guess_model_path)
    builder = buildez80.build_autoreg(guess_model_path, max_output_len=3)
    expected = libinfer.generate(model, "IS IT AN ANIMAL", 3, accum_bits=24)
    assert expected in agon_reply(builder, "IS IT AN ANIMAL")
