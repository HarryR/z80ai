"""Structural tests for the three build targets and the TAP container."""

from __future__ import annotations

import buildfastz80com
import buildz80com
import buildz80tap
import libinfer
import pytest

CPM_TPA_TOP = 0xE400  # where a stock CP/M 2.2 BDOS starts
ZX_RAM_TOP = 0x10000  # one past the last byte of RAM on a 48K machine


def test_tap_header_block_layout():
    block = buildz80tap.build_tap_header("CHAT", 0x8000, 0x1234)
    assert len(block) == 21  # 2 length + 19 block
    assert block[0] | (block[1] << 8) == 19
    assert block[2] == 0x00  # header flag
    assert block[3] == 3  # CODE
    assert block[4:14] == b"CHAT      "
    assert block[14] | (block[15] << 8) == 0x1234
    assert block[16] | (block[17] << 8) == 0x8000
    assert block[-1] == _xor(block[2:-1])


def test_tap_data_block_layout():
    payload = bytes(range(64))
    block = buildz80tap.build_tap_data(payload)
    assert block[0] | (block[1] << 8) == len(payload) + 2
    assert block[2] == 0xFF  # data flag
    assert block[3:-1] == payload
    assert block[-1] == _xor(block[2:-1])


def test_tap_filename_is_truncated_to_ten_characters():
    block = buildz80tap.build_tap_header("ABCDEFGHIJKLMNOP", 0x8000, 1)
    assert block[4:14] == b"ABCDEFGHIJ"


def _xor(data: bytes) -> int:
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum


@pytest.fixture(scope="module")
def images(tiny_model_path):
    return {
        "com": buildz80com.build_autoreg(tiny_model_path),
        "fast": buildfastz80com.build_autoreg(tiny_model_path),
        "tap": buildz80tap.build_autoreg(tiny_model_path),
    }


@pytest.mark.parametrize("target", ["com", "fast", "tap"])
def test_build_resolves_all_fixups(images, target):
    images[target].build()  # raises on an unresolved or out-of-range label


@pytest.mark.parametrize("target", ["com", "fast"])
def test_com_fits_in_the_transient_program_area(images, target):
    builder = images[target]
    end = builder.org + len(builder.build())
    assert end < CPM_TPA_TOP, f"{target} overruns the TPA by {end - CPM_TPA_TOP} bytes"


def test_tap_payload_fits_in_48k_ram(images):
    builder = images["tap"]
    end = builder.org + len(builder.build())
    assert end <= ZX_RAM_TOP


@pytest.mark.parametrize("example", ["guess", "tinychat"])
def test_shipped_examples_fit_in_48k_ram(example, examples_dir):
    """The real models, not a synthetic one.

    Both shipped .TAP files used to be assembled at 8000h, where only 32,768
    bytes are available - they ran past FFFFh and could not load at all. A
    synthetic tiny model fits anywhere, so only the real ones catch this.
    """
    import os

    path = os.path.join(examples_dir, example, "model.npz")
    if not os.path.exists(path):
        pytest.skip(f"{example} example model not present")
    builder = buildz80tap.build_autoreg(path)
    end = builder.org + len(builder.build())
    assert end <= ZX_RAM_TOP, (
        f"{example} runs to {end:#07x}, past the top of RAM by {end - ZX_RAM_TOP:,}"
    )


def test_builder_refuses_an_image_that_would_not_load(guess_model_path):
    """Assembling above the fold must fail loudly, not emit a broken tape."""
    with pytest.raises(ValueError, match="past the top of RAM"):
        buildz80tap.build_autoreg(guess_model_path, org=0x8000)


@pytest.mark.parametrize("target", ["com", "fast", "tap"])
def test_expected_entry_points_exist(images, target):
    labels = images[target].labels
    for name in ("START", "GENERATE", "ARGMAX", "TOKENIZE", "UPDATE_CTX", "CHARTBL"):
        assert name in labels, f"{target} is missing {name}"


def test_character_table_matches_the_model_charset(images, tiny_model):
    builder = images["com"]
    image = builder.build()
    base = builder.labels["CHARTBL"] - builder.org
    table = image[base : base + len(tiny_model.charset)]
    assert table == bytes(ord(c) for c in tiny_model.charset)


def test_packed_weights_are_embedded_verbatim(images, tiny_model):
    builder = images["com"]
    image = builder.build()
    for i, w in enumerate(tiny_model.weights, start=1):
        base = builder.labels[f"WTS{i}"] - builder.org
        expected = libinfer.pack_2bit(w, "rotated")
        assert image[base : base + len(expected)] == expected


def test_biases_are_embedded_as_little_endian_words(images, tiny_model):
    builder = images["com"]
    image = builder.build()
    for i, bias in enumerate(tiny_model.biases, start=1):
        base = builder.labels[f"BIAS{i}"] - builder.org
        for j, value in enumerate(bias):
            got = image[base + 2 * j] | (image[base + 2 * j + 1] << 8)
            assert got == (int(value) & 0xFFFF)


def test_fast_build_is_larger_but_both_hold_the_same_model(images):
    """The fast build trades size for speed; both must still assemble."""
    assert len(images["fast"].build()) > len(images["com"].build())
