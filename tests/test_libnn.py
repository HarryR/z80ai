"""Tests for the shared code generator's own abstractions.

The emitted machine code is covered elsewhere -- byte-for-byte in
``test_codegen_stability`` and semantically in ``test_kernels`` and
``test_end_to_end``. What is checked here is the layer planning and the
Platform contract, which decide *which* code gets emitted.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools

import pytest

import libnn
from libz80 import Z80Builder


class FakePlatform(libnn.Platform):
    """Records what it was asked to emit instead of emitting anything."""

    name = "fake"
    buffer = "TESTBUF"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def print_char(self, b: Z80Builder) -> None:
        self.calls.append("print_char")
        b.nop()

    def load_query_length(self, b: Z80Builder) -> None:
        self.calls.append("load_query_length")
        b.nop()

    def load_query_pointer(self, b: Z80Builder) -> None:
        self.calls.append("load_query_pointer")
        b.ld_de_nn(0)


# --- layer planning ----------------------------------------------------------


def test_first_layer_reads_the_input_buffer_and_last_writes_the_output():
    plans = libnn.plan_layers([256, 128, 64, 11], "INBUF")
    assert plans[0].in_buffer == "INBUF"
    assert plans[-1].out_buffer == "OUTBUF"
    assert plans[-1].is_last
    assert not any(p.is_last for p in plans[:-1])


def test_scratch_buffers_ping_pong():
    """Each layer must read what the previous one wrote."""
    plans = libnn.plan_layers([256, 128, 96, 64, 11], "INBUF")
    for previous, current in itertools.pairwise(plans):
        assert current.in_buffer == previous.out_buffer
    scratch = {p.out_buffer for p in plans[:-1]}
    assert scratch == {"BUF_A", "BUF_B"}


def test_plan_carries_the_layer_dimensions():
    plans = libnn.plan_layers([256, 128, 11], "INBUF")
    assert [(p.in_size, p.out_size) for p in plans] == [(256, 128), (128, 11)]


def test_a_single_layer_model_reads_input_and_writes_output():
    plans = libnn.plan_layers([256, 11], "INBUF")
    assert len(plans) == 1
    assert (plans[0].in_buffer, plans[0].out_buffer) == ("INBUF", "OUTBUF")


def test_labels_are_one_based_to_match_the_fc_naming():
    plans = libnn.plan_layers([256, 128, 11], "INBUF")
    assert [p.label for p in plans] == ["LAYER1", "LAYER2"]
    assert [p.weights_label for p in plans] == ["WTS1", "WTS2"]
    assert [p.bias_label for p in plans] == ["BIAS1", "BIAS2"]


def test_plans_are_immutable():
    """Buffer assignment is decided once; nothing downstream may rewrite it."""
    plan = libnn.plan_layers([256, 11], "INBUF")[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.in_size = 1


@pytest.mark.parametrize("size,expected", [(1, 1), (255, 255), (256, 0)])
def test_a_256_wide_layer_is_counted_as_a_zero_djnz_start(size, expected):
    """DJNZ decrements first, so a zero start runs 256 times."""
    assert libnn._byte_count(size) == expected


# --- the Platform contract ---------------------------------------------------


def test_platform_is_abstract():
    with pytest.raises(TypeError):
        libnn.Platform()


def test_a_platform_must_implement_every_hook():
    class Incomplete(libnn.Platform):
        def print_char(self, b: Z80Builder) -> None:
            pass

    with pytest.raises(TypeError):
        Incomplete()


def test_printch_asks_the_platform_to_print():
    plat = FakePlatform()
    b = Z80Builder()
    b.label("CHARTBL")
    libnn.emit_printch(b, plat)
    assert plat.calls == ["print_char"]


def test_the_tokenizer_asks_the_platform_where_the_query_is():
    plat = FakePlatform()
    b = Z80Builder()
    libnn.emit_tokenizer(b, plat)
    assert plat.calls == ["load_query_length", "load_query_pointer"]


def test_routines_address_the_buffer_the_platform_names():
    plat = FakePlatform()
    b = Z80Builder()
    libnn.emit_tokenizer(b, plat)
    referenced = {label for _offset, label, _kind, _addend in b.fixups}
    assert "TESTBUF" in referenced
    assert "INBUF" not in referenced


# --- emitted structure -------------------------------------------------------


def build_engine(plat: libnn.Platform, layer_sizes: list[int]) -> Z80Builder:
    """Emit the full shared engine so labels and fixups can be inspected."""
    plans = libnn.plan_layers(layer_sizes, plat.buffer)
    b = Z80Builder()
    libnn.emit_generate(b, plat, layer_sizes[-1] - 1, 8,
                        libnn.emit_layered_inference(plans))
    libnn.emit_printch(b, plat)
    libnn.emit_update_ctx(b, plat)
    libnn.emit_encode_ctx(b, plat)
    libnn.emit_ctx_hash(b, plat)
    libnn.emit_clear_ctx(b, plat)
    libnn.emit_layer_dispatch(b, plans)
    libnn.emit_layer(b)
    libnn.emit_muladd(b)
    libnn.emit_relu(b, plans)
    libnn.emit_argmax(b, layer_sizes[-1])
    libnn.emit_tokenizer(b, plat)
    libnn.emit_tok_hash(b, plat)
    libnn.emit_charset_table(b, " AB\x00")
    libnn.emit_variables(b)
    libnn.emit_buffers(b, plat, layer_sizes)
    weights = [b"\x00" * 4 for _ in plans]
    biases = [[0] * plan.out_size for plan in plans]
    libnn.emit_weights(b, weights, biases)
    return b


def test_the_engine_resolves_with_no_dangling_labels():
    build_engine(FakePlatform(), [256, 32, 4]).build()


def test_every_layer_and_relu_stub_is_emitted():
    b = build_engine(FakePlatform(), [256, 64, 32, 4])
    for i in (1, 2, 3):
        assert f"LAYER{i}" in b.labels
    for i in (1, 2):  # no ReLU after the output layer
        assert f"RELU{i}" in b.labels
    assert "RELU3" not in b.labels


def test_clear_ctx_can_be_emitted_unrolled_or_as_a_loop():
    unrolled = Z80Builder()
    libnn.emit_clear_ctx(unrolled, FakePlatform(), unrolled=True)
    looped = Z80Builder()
    libnn.emit_clear_ctx(looped, FakePlatform(), unrolled=False)
    assert len(looped.code) < len(unrolled.code)
    assert "CLR_LP" in looped.labels
    assert "CLR_LP" not in unrolled.labels


def test_context_offset_splits_the_buffer_in_half():
    assert libnn.CONTEXT_OFFSET == libnn.NUM_BUCKETS * libnn.ACTIVATION_SIZE


def test_emit_weights_rejects_mismatched_weight_and_bias_lists():
    b = Z80Builder()
    with pytest.raises(ValueError):
        libnn.emit_weights(b, [b"\x00", b"\x00"], [[1]])


# --- typing ------------------------------------------------------------------


@pytest.mark.parametrize("module_name", ["libnn", "libz80", "libinfer", "libez80"])
def test_public_functions_are_annotated(module_name):
    """These are the modules a backend author writes against."""
    import importlib

    module = importlib.import_module(module_name)
    unannotated = []
    for name, obj in vars(module).items():
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        if obj.__module__ != module_name:
            continue
        sig = inspect.signature(obj)
        if sig.return_annotation is inspect.Signature.empty:
            unannotated.append(f"{name} (return)")
        unannotated += [
            f"{name}({p})" for p, spec in sig.parameters.items()
            if spec.annotation is inspect.Signature.empty and p not in ("self", "cls")
        ]
    assert not unannotated, f"missing annotations: {unannotated}"
