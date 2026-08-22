"""Tests for the release-artifact verifier.

CI gates releases on this script, so its failure paths need to actually fail.
A verifier that silently passes everything is worse than no verifier.
"""

from __future__ import annotations

import pytest

import buildz80tap
import verify_artifacts as va


@pytest.fixture(scope="module")
def tap_bytes(tiny_model_path) -> tuple[bytes, bytes, int]:
    """A real .TAP container plus the image and load address inside it."""
    builder = buildz80tap.build_autoreg(tiny_model_path, max_output_len=2)
    image = builder.build()
    tap = buildz80tap.build_tap_header("CHAT", builder.org, len(image))
    tap += buildz80tap.build_tap_data(image)
    return tap, image, builder.org


def test_parse_tap_recovers_image_and_load_address(tap_bytes):
    tap, image, org = tap_bytes
    got_image, got_org = va.parse_tap(tap)
    assert got_image == image
    assert got_org == org


def test_parse_tap_rejects_a_corrupted_data_checksum(tap_bytes):
    tap, _image, _org = tap_bytes
    broken = bytearray(tap)
    broken[-1] ^= 0xFF
    with pytest.raises(va.VerificationError, match="checksum mismatch"):
        va.parse_tap(bytes(broken))


def test_parse_tap_rejects_a_corrupted_payload(tap_bytes):
    tap, _image, _org = tap_bytes
    broken = bytearray(tap)
    broken[40] ^= 0xFF  # somewhere inside the data block
    with pytest.raises(va.VerificationError, match="checksum mismatch"):
        va.parse_tap(bytes(broken))


def test_parse_tap_rejects_a_truncated_container(tap_bytes):
    tap, _image, _org = tap_bytes
    with pytest.raises(va.VerificationError, match="truncated"):
        va.parse_tap(tap[:-5])


def test_parse_tap_rejects_a_header_that_lies_about_the_length(tap_bytes):
    tap, _image, _org = tap_bytes
    broken = bytearray(tap)
    broken[14] = (broken[14] + 1) & 0xFF  # declared length, inside the header
    checksum = 0
    for byte in broken[2:20]:
        checksum ^= byte
    broken[20] = checksum  # keep the header checksum valid so length is what fails
    with pytest.raises(va.VerificationError, match="declares"):
        va.parse_tap(bytes(broken))


def test_check_fits_accepts_an_image_that_ends_exactly_at_the_top():
    va.check_fits(0x6000, 0x10000 - 0x6000, 0x10000, "the top of RAM")


def test_check_fits_rejects_one_byte_too_many():
    with pytest.raises(va.VerificationError, match="past the top of RAM"):
        va.check_fits(0x6000, 0x10000 - 0x6000 + 1, 0x10000, "the top of RAM")


def test_check_fits_reports_the_overrun_size():
    with pytest.raises(va.VerificationError, match="by 100 bytes"):
        va.check_fits(0x8000, 0x8000 + 100, 0x10000, "the top of RAM")


def test_every_known_artifact_names_a_model_that_exists(examples_dir):
    import os

    for name, (model_path, platform) in va.ARTIFACTS.items():
        assert platform in {"cpm", "zx", "agon"}, name
        assert os.path.exists(os.path.join(os.path.dirname(examples_dir), model_path)), (
            f"{name} refers to a missing model {model_path}"
        )


def test_verify_reports_nothing_for_an_empty_directory(tmp_path):
    assert va.verify(str(tmp_path), "HELLO") == []
