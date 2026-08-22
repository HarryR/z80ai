"""End-to-end equivalence: generated machine code vs. the NumPy reference.

These are the tests that actually matter.  Each build script is compiled for a
model, executed instruction-by-instruction in the emulator, and its console
output compared character-for-character against :func:`libinfer.generate`.
Any divergence -- packing, hashing, accumulator overflow, shift direction --
shows up here.
"""

from __future__ import annotations

import buildfastz80com
import buildz80com
import buildz80tap
import libinfer
import pytest
from libhost import run_cpm, run_zx

QUERIES = ["HELLO", "ARE YOU A ROBOT", "X", "WHAT IS THIS THING"]
GEN_LEN = 8  # keep emulated runs short; the code path is identical


def cpm_reply(image: bytes, query: str) -> str:
    """Run a .COM in single-query mode and return what it printed."""
    out, host = run_cpm(image, cmdline=query, max_cycles=400_000_000)
    assert host.finished, "program did not return to CP/M"
    return out


@pytest.fixture(scope="module")
def tiny_com(tiny_model_path):
    return buildz80com.build_autoreg(tiny_model_path, max_output_len=GEN_LEN).build()


@pytest.fixture(scope="module")
def tiny_fast_com(tiny_model_path):
    return buildfastz80com.build_autoreg(tiny_model_path, max_output_len=GEN_LEN).build()


@pytest.fixture(scope="module")
def tiny_tap(tiny_model_path):
    return buildz80tap.build_autoreg(tiny_model_path, max_output_len=GEN_LEN).build()


@pytest.mark.parametrize("query", QUERIES)
def test_com_matches_reference(tiny_com, tiny_model, query):
    assert cpm_reply(tiny_com, query) == libinfer.generate(tiny_model, query, GEN_LEN)


@pytest.mark.parametrize("query", QUERIES)
def test_fast_com_matches_reference(tiny_fast_com, tiny_model, query):
    assert cpm_reply(tiny_fast_com, query) == libinfer.generate(tiny_model, query, GEN_LEN)


@pytest.mark.parametrize("query", QUERIES)
def test_tap_matches_reference(tiny_tap, tiny_model, query):
    image = tiny_tap
    out, _host = run_zx(image, stdin=[query, "!"], max_cycles=400_000_000)
    # The ZX build is chat-only; strip the prompt/echo chrome around the reply.
    expected = libinfer.generate(tiny_model, query, GEN_LEN)
    assert expected in out, f"{expected!r} not in {out!r}"


def test_all_three_builds_agree(tiny_com, tiny_fast_com, tiny_model):
    for query in QUERIES:
        assert cpm_reply(tiny_com, query) == cpm_reply(tiny_fast_com, query)


@pytest.mark.parametrize("query", QUERIES[:2])
def test_layer_widths_not_multiple_of_four(odd_model_path, odd_model, query):
    """Packed weights must stay aligned when a layer width isn't a multiple of 4."""
    image = buildz80com.build_autoreg(odd_model_path, max_output_len=GEN_LEN).build()
    assert cpm_reply(image, query) == libinfer.generate(odd_model, query, GEN_LEN)


def test_chat_mode_roundtrip(tiny_com, tiny_model):
    out, host = run_cpm(tiny_com, cmdline="", stdin=["HELLO", "!"], max_cycles=400_000_000)
    assert host.finished
    assert libinfer.generate(tiny_model, "HELLO", GEN_LEN) in out


def test_empty_query_is_handled(tiny_com, tiny_model):
    """A blank command line must not hang or crash - it enters chat mode."""
    out, host = run_cpm(tiny_com, cmdline="", stdin=["!"], max_cycles=50_000_000)
    assert host.finished
    assert ">" in out


@pytest.mark.slow
@pytest.mark.parametrize("query", ["IS IT AN ANIMAL", "HELLO"])
def test_full_model_matches_reference(guess_model_path, query):
    model = libinfer.Model.load(guess_model_path)
    image = buildz80com.build_autoreg(guess_model_path, max_output_len=4).build()
    assert cpm_reply(image, query) == libinfer.generate(model, query, 4)
