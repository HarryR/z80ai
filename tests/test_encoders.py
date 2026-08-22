"""Tests for the trigram / context hashing.

The reference implementations in :mod:`libinfer` are checked against an
independent restatement of the spec here, so a shared bug can't pass unnoticed.
Equality with what the Z80 actually computes is covered in ``test_kernels``.
"""

from __future__ import annotations

import numpy as np
import pytest

import libinfer

BUCKETS = libinfer.NUM_BUCKETS
W = libinfer.BUCKET_WEIGHT


def naive_trigram(text: str) -> np.ndarray:
    """Independent restatement: pad with spaces, hash every 3-window."""
    vec = np.zeros(BUCKETS, dtype=np.int32)
    padded = " " + text.lower().lstrip(" ") + " "
    if len(padded) < 3:
        return vec
    for i in range(len(padded) - 2):
        h = 0
        for ch in padded[i : i + 3]:
            h = (h * 31 + ord(ch)) % 65536
        vec[h % BUCKETS] += W
    return vec


def naive_context(recent: str) -> np.ndarray:
    vec = np.zeros(BUCKETS, dtype=np.int32)
    window = recent.lower()[-8:].rjust(8)
    for n in (1, 2, 3):
        for pos in range(8 - n + 1):
            h = pos * 7
            for ch in window[pos : pos + n]:
                h = (h * 31 + ord(ch)) % 65536
            vec[h % BUCKETS] += W
    return vec


@pytest.mark.parametrize(
    "text", ["HELLO", "hello", "a", "are you a robot", "WHAT?", "12 34", ""]
)
def test_trigram_matches_independent_implementation(text):
    np.testing.assert_array_equal(libinfer.trigram_encode(text), naive_trigram(text))


@pytest.mark.parametrize("recent", ["", "a", "hello", "abcdefghij", "  x "])
def test_context_matches_independent_implementation(recent):
    np.testing.assert_array_equal(libinfer.context_encode(recent), naive_context(recent))


def test_trigram_count_scales_with_length():
    """An n-character query contributes exactly n trigrams."""
    for text in ("A", "AB", "HELLO", "A LONGER QUERY"):
        assert libinfer.trigram_encode(text).sum() == len(text) * W


def test_context_always_contributes_the_same_total():
    """8 unigrams + 7 bigrams + 6 trigrams, regardless of content."""
    for recent in ("", "x", "hello", "abcdefghijkl"):
        assert libinfer.context_encode(recent).sum() == (8 + 7 + 6) * W


def test_trigram_is_case_insensitive():
    np.testing.assert_array_equal(
        libinfer.trigram_encode("Hello There"), libinfer.trigram_encode("HELLO THERE")
    )


def test_trigram_ignores_word_order_for_symmetric_inputs():
    """The documented 'tag cloud' property: shared trigrams overlap heavily."""
    a = libinfer.trigram_encode("hello there")
    b = libinfer.trigram_encode("there hello")
    overlap = np.minimum(a, b).sum() / a.sum()
    assert overlap > 0.7


def test_empty_and_blank_queries_encode_to_zero():
    assert libinfer.trigram_encode("").sum() == 0
    assert libinfer.trigram_encode("    ").sum() == 0


def test_only_az_is_lowercased():
    """The Z80 lowercases with a range check, so '[' (0x5B) must pass through."""
    assert libinfer._lower("[") == "["
    assert libinfer._lower("A") == "a"
    assert libinfer._lower("Z") == "z"
    assert libinfer._lower("@") == "@"


def test_context_window_keeps_only_the_last_eight_characters():
    np.testing.assert_array_equal(
        libinfer.context_encode("0123456789abcdefgh"),
        libinfer.context_encode("abcdefgh"),
    )


def test_hash_is_16_bit():
    assert libinfer.hash16("z" * 20) < 65536


# feedme imports torch; these two pin that training sees the same features the
# Z80 does, which is the whole reason the encoders live in one place.
needs_torch = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("torch") is None,
    reason="training-time encoders need torch",
)


@needs_torch
@pytest.mark.parametrize("text", ["HELLO", "ARE YOU A ROBOT", "X"])
def test_training_encoder_matches_reference(text):
    import feedme

    got = feedme.TrigramEncoder(BUCKETS).encode(text)
    np.testing.assert_allclose(got, libinfer.trigram_encode(text) / W)


@needs_torch
@pytest.mark.parametrize("recent", ["", "he", "hello"])
def test_training_context_encoder_matches_reference(recent):
    import feedme

    got = feedme.ContextEncoder(BUCKETS, 8).encode(recent)
    np.testing.assert_allclose(got, libinfer.context_encode(recent) / W)
