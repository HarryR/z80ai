"""Numeric tests on the generated kernels.

Comparing generated text is a weak signal: with few output classes two
different logit vectors often argmax to the same character.  These tests read
the actual buffers out of emulator memory and compare every 16-bit value
against the reference, so a packing or accumulation bug cannot hide.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import read_words
from helpers import run_cpm_until as run_until

import buildz80com
import libinfer


@pytest.fixture(scope="module")
def tiny_builder(tiny_model_path):
    return buildz80com.build_autoreg(tiny_model_path, max_output_len=1)


@pytest.fixture(scope="module")
def odd_builder(odd_model_path):
    return buildz80com.build_autoreg(odd_model_path, max_output_len=1)


@pytest.mark.parametrize("query", ["HELLO", "A", "IS IT AN ANIMAL"])
def test_tokenizer_matches_reference(tiny_builder, query):
    cpu = run_until(tiny_builder, query, "ARGMAX")
    got = read_words(cpu, tiny_builder.labels["INBUF"], 128)
    np.testing.assert_array_equal(got, libinfer.trigram_encode(query))


def test_initial_context_matches_reference(tiny_builder):
    cpu = run_until(tiny_builder, "HELLO", "ARGMAX")
    got = read_words(cpu, tiny_builder.labels["INBUF"] + 256, 128)
    np.testing.assert_array_equal(got, libinfer.context_encode(" " * 8))


@pytest.mark.parametrize("query", ["HELLO", "WHAT IS THIS"])
def test_output_logits_match_reference(tiny_builder, tiny_model, query):
    cpu = run_until(tiny_builder, query, "ARGMAX")
    got = read_words(cpu, tiny_builder.labels["OUTBUF"], tiny_model.output_size)
    x = np.concatenate([libinfer.trigram_encode(query), libinfer.context_encode(" " * 8)])
    np.testing.assert_array_equal(got, libinfer.forward(tiny_model, x))


@pytest.mark.parametrize("query", ["HELLO", "WHAT IS THIS"])
def test_output_logits_with_odd_layer_widths(odd_builder, odd_model, query):
    """Packed weights must stay in step when a layer width isn't a multiple of 4."""
    cpu = run_until(odd_builder, query, "ARGMAX")
    got = read_words(cpu, odd_builder.labels["OUTBUF"], odd_model.output_size)
    x = np.concatenate([libinfer.trigram_encode(query), libinfer.context_encode(" " * 8)])
    np.testing.assert_array_equal(got, libinfer.forward(odd_model, x))


def test_context_updates_after_each_character(tiny_builder, tiny_model):
    """After one emitted character the context buckets must reflect it."""
    builder = buildz80com.build_autoreg_path = tiny_builder
    cpu = run_until(builder, "HELLO", "UPDATE_CTX")
    # Step through UPDATE_CTX and stop at the next ARGMAX.
    cpu.run(max_cycles=400_000_000, stop_pc=builder.labels["ARGMAX"])
    first = libinfer.generate(tiny_model, "HELLO", 1)
    expected = libinfer.context_encode(" " * 8 + first)
    got = read_words(cpu, builder.labels["INBUF"] + 256, 128)
    np.testing.assert_array_equal(got, expected)
