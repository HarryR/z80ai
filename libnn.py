"""
Shared Z80 code generation for the character-level inference engine.

The CP/M and ZX Spectrum backends were 85% identical line for line, and the
fast CP/M backend shared about three quarters of that again. Every routine that
does not depend on the host machine lives here now, so a fix lands once rather
than once per target -- the ``MULADD`` borrow bug had to be fixed in two places
because it did not.

What stays in a backend is genuinely platform-specific: where the query text
comes from, how a character reaches the screen, and how the weights are laid
out for the inner loop.

Register and memory conventions shared by every routine below:

===========  ============================================================
``CTXCHARS`` the last :data:`CONTEXT_LEN` characters emitted, lower-cased
``RESULT``   index into the charset of the character just chosen
``GENCNT``   characters left to generate before giving up
``ACC``      16-bit neuron accumulator
===========  ============================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from libz80 import Z80Builder

#: Hash buckets per half of the input vector. The first NUM_BUCKETS encode the
#: query, the second NUM_BUCKETS the characters generated so far.
NUM_BUCKETS = 128
#: Characters of generated output fed back as context.
CONTEXT_LEN = 8
#: Fixed-point scale: one n-gram occurrence adds this much to its bucket.
BUCKET_WEIGHT = 32
#: Bytes per activation.
ACTIVATION_SIZE = 2
#: Byte offset from the start of the activation buffer to the context half.
CONTEXT_OFFSET = NUM_BUCKETS * ACTIVATION_SIZE


class Platform(ABC):
    """A target machine: where its input comes from and how it prints.

    Subclasses emit a handful of instructions each; everything else in this
    module is machine-independent.
    """

    #: Human-readable name, used in build output.
    name: str = "unknown"
    #: Label of the activation buffer holding both halves of the input vector.
    buffer: str = "INBUF"
    #: Which :func:`libinfer.pack_2bit` layout the inner loop expects.
    weight_layout: str = "rotated"

    @abstractmethod
    def print_char(self, b: Z80Builder) -> None:
        """Emit code printing the character currently in ``A``."""

    @abstractmethod
    def load_query_length(self, b: Z80Builder) -> None:
        """Emit code loading the query length into ``A``."""

    @abstractmethod
    def load_query_pointer(self, b: Z80Builder) -> None:
        """Emit code loading the address of the query text into ``DE``."""


@dataclass(frozen=True)
class LayerPlan:
    """Which buffers a layer reads and writes, decided at build time."""

    index: int
    in_size: int
    out_size: int
    in_buffer: str
    out_buffer: str
    is_last: bool

    @property
    def label(self) -> str:
        return f"LAYER{self.index + 1}"

    @property
    def weights_label(self) -> str:
        return f"WTS{self.index + 1}"

    @property
    def bias_label(self) -> str:
        return f"BIAS{self.index + 1}"


def plan_layers(layer_sizes: list[int], input_buffer: str) -> list[LayerPlan]:
    """Assign ping-ponged scratch buffers to each layer.

    The first layer reads the tokenized input and the last writes ``OUTBUF``;
    everything between alternates between two scratch buffers.
    """
    num_layers = len(layer_sizes) - 1
    plans = []
    for i in range(num_layers):
        in_buffer = input_buffer if i == 0 else _scratch(i)
        out_buffer = "OUTBUF" if i == num_layers - 1 else _scratch(i + 1)
        plans.append(
            LayerPlan(
                index=i,
                in_size=layer_sizes[i],
                out_size=layer_sizes[i + 1],
                in_buffer=in_buffer,
                out_buffer=out_buffer,
                is_last=i == num_layers - 1,
            )
        )
    return plans


def _scratch(depth: int) -> str:
    return "BUF_A" if depth % 2 == 1 else "BUF_B"


def _byte_count(size: int) -> int:
    """DJNZ reads a zero start as 256, which is how a 256-wide layer is counted."""
    return size if size <= 255 else 0


# --- generation loop ---------------------------------------------------------


def emit_layered_inference(plans: list[LayerPlan]) -> Callable[[Z80Builder], None]:
    """An inference step that calls each LAYER stub, with RELU between them."""

    def emit(b: Z80Builder) -> None:
        for plan in plans:
            b.call(plan.label)
            if not plan.is_last:
                b.call(f"RELU{plan.index + 1}")

    return emit


def emit_generate(
    b: Z80Builder,
    plat: Platform,
    eos_idx: int,
    max_output_len: int,
    emit_inference: Callable[[Z80Builder], None],
) -> None:
    """Emit GENERATE: infer, argmax, print, feed back, repeat until EOS.

    ``emit_inference`` emits whatever runs one forward pass, leaving the scores
    in OUTBUF. Backends differ there: the packed builds call a stub per layer,
    the index-list build calls a single table-driven INFER.
    """
    b.label("GENERATE")
    b.ld_a_n(max_output_len)
    b.ld_mem_label_a("GENCNT")

    b.label("GENLOOP")
    emit_inference(b)

    b.call("ARGMAX")

    b.ld_a_mem_label("RESULT")
    b.cp_n(eos_idx)
    b.ret_z()

    b.call("PRINTCH")
    b.call("UPDATE_CTX")

    b.ld_a_mem_label("GENCNT")
    b.dec_a()
    b.ld_mem_label_a("GENCNT")
    b.jr_nz("GENLOOP")
    b.ret()


def emit_printch(b: Z80Builder, plat: Platform) -> None:
    """Emit PRINTCH: map RESULT through CHARTBL and print it."""
    b.label("PRINTCH")
    b.ld_a_mem_label("RESULT")
    b.ld_hl_label("CHARTBL")
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()
    b.ld_a_hl()
    plat.print_char(b)
    b.ret()


# --- context encoding --------------------------------------------------------


def emit_update_ctx(b: Z80Builder, plat: Platform) -> None:
    """Emit UPDATE_CTX: shift the context window and append the new character."""
    b.label("UPDATE_CTX")
    b.ld_hl_label("CTXCHARS")
    b.inc_hl()
    b.ld_de_label("CTXCHARS")
    b.ld_bc_nn(CONTEXT_LEN - 1)
    b.ldir()

    b.ld_a_mem_label("RESULT")
    b.ld_hl_label("CHARTBL")
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()
    b.ld_a_hl()

    # Lower-case A-Z only, matching the hashing done at training time.
    b.cp_n(ord("A"))
    b.jr_c("UPD_STORE")
    b.cp_n(ord("Z") + 1)
    b.jr_nc("UPD_STORE")
    b.add_a_n(0x20)

    b.label("UPD_STORE")
    b.ld_hl_label("CTXCHARS")
    b.ld_de_nn(CONTEXT_LEN - 1)
    b.add_hl_de()
    b.ld_hl_a()

    b.call("ENCODE_CTX")
    b.ret()


def emit_encode_ctx(b: Z80Builder, plat: Platform) -> None:
    """Emit ENCODE_CTX: hash 1-, 2- and 3-grams of CTXCHARS into the buckets."""
    b.label("ENCODE_CTX")

    # Clear the context half of the activation buffer.
    b.ld_hl_label(plat.buffer)
    b.ld_de_nn(CONTEXT_OFFSET)
    b.add_hl_de()
    b.ld_d_h()
    b.ld_e_l()
    b.inc_de()
    b.xor_a()
    b.ld_hl_a()
    b.ld_bc_nn(CONTEXT_OFFSET - 1)
    b.ldir()

    b.ld_a_n(0)
    b.ld_mem_label_a("CTXPOS")
    b.ld_a_n(1)
    b.ld_mem_label_a("CTXN")

    b.label("CTX_NLOOP")
    b.xor_a()
    b.ld_mem_label_a("CTXPOS")

    b.label("CTX_PLOOP")
    # Stop this n-gram length once pos reaches CONTEXT_LEN - n + 1.
    b.ld_a_n(CONTEXT_LEN + 1)
    b.ld_hl_label("CTXN")
    b.sub_hl_ind()
    b.ld_b_a()
    b.ld_a_mem_label("CTXPOS")
    b.cp_b()
    b.jr_nc("CTX_NEXT_N")

    b.call("CTX_HASH")

    b.ld_a_mem_label("CTXPOS")
    b.inc_a()
    b.ld_mem_label_a("CTXPOS")
    b.jr("CTX_PLOOP")

    b.label("CTX_NEXT_N")
    b.ld_a_mem_label("CTXN")
    b.inc_a()
    b.ld_mem_label_a("CTXN")
    b.cp_n(4)  # unigrams, bigrams, trigrams
    b.jr_c("CTX_NLOOP")
    b.ret()


def emit_ctx_hash(b: Z80Builder, plat: Platform) -> None:
    """Emit CTX_HASH: hash CTXN characters from CTXPOS, seeded with CTXPOS * 7."""
    b.label("CTX_HASH")

    # hash = pos * 7, as pos * 8 - pos.
    b.ld_a_mem_label("CTXPOS")
    b.ld_l_a()
    b.ld_h_n(0)
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.ld_d_h()
    b.ld_e_l()
    b.ld_a_mem_label("CTXPOS")
    b.ld_l_a()
    b.ld_h_n(0)
    b.ex_de_hl()
    b.or_a()
    b.sbc_hl_de()
    b.push_hl()

    # DE = &CTXCHARS[pos]
    b.ld_hl_label("CTXCHARS")
    b.ld_a_mem_label("CTXPOS")
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()
    b.ex_de_hl()

    b.pop_hl()

    b.ld_a_mem_label("CTXN")
    b.ld_b_a()

    b.label("CTX_HLOOP")
    b.push_bc()
    # hash = hash * 31 + char, as hash * 32 - hash + char.
    b.push_hl()
    for _ in range(5):
        b.add_hl_hl()
    b.pop_bc()
    b.or_a()
    b.sbc_hl_bc()
    b.ld_a_de()
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()
    b.inc_de()
    b.pop_bc()
    b.djnz("CTX_HLOOP")

    b.ld_a_l()
    b.and_n(NUM_BUCKETS - 1)

    # HL = &buffer[CONTEXT_OFFSET + bucket * 2]
    b.ld_l_a()
    b.ld_h_n(0)
    b.add_hl_hl()
    b.ld_de_label(plat.buffer)
    b.push_hl()
    b.ld_hl_nn(CONTEXT_OFFSET)
    b.add_hl_de()
    b.ex_de_hl()
    b.pop_hl()
    b.add_hl_de()

    # *bucket += BUCKET_WEIGHT
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.push_hl()
    b.ld_hl_nn(BUCKET_WEIGHT)
    b.add_hl_de()
    b.ex_de_hl()
    b.pop_hl()
    b.ld_hl_d()
    b.dec_hl()
    b.ld_hl_e()
    b.ret()


def emit_clear_ctx(b: Z80Builder, plat: Platform, unrolled: bool = True) -> None:
    """Emit CLEAR_CTX: fill the context window with spaces and re-encode.

    ``unrolled`` picks between straight-line stores and a DJNZ loop; the two
    differ only in size, and each backend keeps whichever it already emitted.
    """
    b.label("CLEAR_CTX")
    b.ld_hl_label("CTXCHARS")
    if unrolled:
        b.ld_a_n(ord(" "))
        for _ in range(CONTEXT_LEN):
            b.ld_hl_a()
            b.inc_hl()
    else:
        b.ld_b_n(CONTEXT_LEN)
        b.label("CLR_LP")
        b.ld_hl_n(ord(" "))
        b.inc_hl()
        b.djnz("CLR_LP")

    b.jp("ENCODE_CTX")  # tail call: ENCODE_CTX returns for us


# --- inference ---------------------------------------------------------------


def emit_layer_dispatch(b: Z80Builder, plans: list[LayerPlan]) -> None:
    """Emit one stub per layer, loading pointers and sizes then entering LAYER."""
    for plan in plans:
        b.label(plan.label)
        b.ld_hl_label(plan.weights_label)
        b.ld_de_label(plan.bias_label)
        b.ld_ix_label(plan.in_buffer)
        b.ld_iy_label(plan.out_buffer)
        b.ld_b_n(_byte_count(plan.out_size))
        b.ld_c_n(_byte_count(plan.in_size))
        if not plan.is_last:
            b.jp("LAYER")
        # The last stub falls through into LAYER, which follows.


def emit_layer(b: Z80Builder) -> None:
    """Emit LAYER: one fully-connected layer over 2-bit packed weights.

    ``HL`` walks the weight stream, ``DE`` the biases, ``IX`` the inputs and
    ``IY`` the outputs; ``B`` counts neurons and ``C`` weights within a neuron.
    A packed byte is reloaded whenever the per-neuron weight counter reaches a
    multiple of four, which is why every neuron starts on a byte boundary.
    """
    b.label("LAYER")
    b.ld_mem_label_bc("SAVCNT")
    b.ld_mem_label_hl("SAVW")
    b.ld_mem_label_de("SAVB")

    b.label("LNEUR")
    b.push_bc()
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("ACC")
    b.push_ix()
    b.pop_hl()
    b.ld_mem_label_hl("CURIN")
    b.ld_hl_mem_label("SAVW")
    b.ld_a_mem_label("SAVCNT")
    b.ld_b_a()
    b.ld_c_n(0)

    b.label("LWT")
    b.ld_a_c()
    b.and_n(0x03)
    b.jr_nz("LSAME")
    b.ld_hl_mem_label("SAVW")
    b.ld_a_hl()
    b.ld_mem_label_a("PACKED")
    b.inc_hl()
    b.ld_mem_label_hl("SAVW")

    b.label("LSAME")
    b.ld_a_mem_label("PACKED")
    b.rrca()
    b.rrca()
    b.ld_mem_label_a("PACKED")
    b.and_n(0x03)
    b.ld_hl_mem_label("CURIN")
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.inc_hl()
    b.ld_mem_label_hl("CURIN")
    # Code 1 means a zero weight, the commonest case, so let DEC A settle it
    # and skip the call entirely.
    b.dec_a()
    b.call_nz("MULADD")
    b.inc_c()
    b.djnz("LWT")

    # Bias, then scale down to keep the next layer inside 16 bits.
    b.ld_hl_mem_label("SAVB")
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.inc_hl()
    b.ld_mem_label_hl("SAVB")
    b.ld_hl_mem_label("ACC")
    b.add_hl_de()
    b.ld_mem_label_hl("ACC")
    b.sra_h()
    b.rr_l()
    b.sra_h()
    b.rr_l()

    b.ld_iyd_l(0)
    b.ld_iyd_h(1)
    b.inc_iy()
    b.inc_iy()
    b.pop_bc()
    b.djnz("LNEUR")
    b.ret()


def emit_muladd(b: Z80Builder) -> None:
    """Emit MULADD: accumulate ``weight * DE`` for a nonzero weight.

    Entered with ``A`` already decremented once, so it holds FFh, 1 or 2 for
    weights of -2, +1 and -1 respectively.
    """
    b.label("MULADD")
    b.ld_hl_mem_label("ACC")
    b.dec_a()
    b.jr_z("MA_P1")
    b.sbc_hl_de()  # carry is clear on entry, from the AND in LSAME
    b.dec_a()
    b.jr_z("MA_MRET")
    b.or_a()  # clear carry: the SBC above may have borrowed
    b.sbc_hl_de()

    b.label("MA_MRET")
    b.ld_mem_label_hl("ACC")
    b.ret()

    b.label("MA_P1")
    b.add_hl_de()
    b.ld_mem_label_hl("ACC")
    b.ret()


def emit_relu(b: Z80Builder, plans: list[LayerPlan]) -> None:
    """Emit one stub per hidden layer plus the shared RELU loop."""
    for plan in plans[:-1]:
        b.label(f"RELU{plan.index + 1}")
        b.ld_hl_label(plan.out_buffer)
        b.ld_b_n(_byte_count(plan.out_size))
        if plan.index != len(plans) - 2:
            b.jr("RELU")
        # The last stub falls through into RELU, which follows.

    b.label("RELU")
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.bit_7_d()
    b.jr_z("RPOS")
    b.dec_hl()
    b.xor_a()
    b.ld_hl_a()
    b.inc_hl()
    b.ld_hl_a()
    b.label("RPOS")
    b.inc_hl()
    b.djnz("RELU")
    b.ret()


def emit_argmax(b: Z80Builder, output_size: int) -> None:
    """Emit ARGMAX over OUTBUF, leaving the winning index in RESULT.

    Strictly greater-than, so the first of equal scores wins.
    """
    b.label("ARGMAX")
    b.ld_hl_label("OUTBUF")
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.inc_hl()
    b.ld_mem_label_de("MAXV")
    b.xor_a()
    b.ld_mem_label_a("MAXI")
    b.ld_b_n(output_size - 1)
    b.ld_c_n(1)

    b.label("AMLP")
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.inc_hl()
    b.push_hl()
    b.ld_hl_mem_label("MAXV")
    b.push_de()
    b.or_a()
    b.ex_de_hl()
    b.sbc_hl_de()
    b.pop_de()
    b.jp_m("AMSK")
    b.jr_z("AMSK")
    b.ld_mem_label_de("MAXV")
    b.ld_a_c()
    b.ld_mem_label_a("MAXI")

    b.label("AMSK")
    b.pop_hl()
    b.inc_c()
    b.djnz("AMLP")
    b.ld_a_mem_label("MAXI")
    b.ld_mem_label_a("RESULT")
    b.ret()


# --- query tokenization ------------------------------------------------------


def emit_tokenizer(b: Z80Builder, plat: Platform) -> None:
    """Emit TOKENIZE: hash the query's trigrams into the first half of the buffer.

    The query is treated as though padded with a space at each end, so an
    n-character query contributes n trigrams.
    """
    b.label("TOKENIZE")

    # Clear the query half of the activation buffer.
    b.ld_hl_label(plat.buffer)
    b.ld_de_label(plat.buffer)
    b.inc_de()
    b.ld_bc_nn(CONTEXT_OFFSET - 1)
    b.ld_a_n(0)
    b.ld_hl_a()
    b.ldir()

    plat.load_query_length(b)
    b.or_a()
    b.jp_z("TOK_DONE")
    b.ld_mem_label_a("TOKLEN")

    plat.load_query_pointer(b)

    b.label("TOK_SKIP_SPACE")
    b.ld_a_mem_label("TOKLEN")
    b.or_a()
    b.jp_z("TOK_DONE")
    b.ld_a_de()
    b.cp_n(ord(" "))
    b.jr_nz("TOK_START")
    b.inc_de()
    b.ld_a_mem_label("TOKLEN")
    b.dec_a()
    b.ld_mem_label_a("TOKLEN")
    b.jr("TOK_SKIP_SPACE")

    b.label("TOK_START")
    b.ld_a_n(ord(" "))  # the leading pad
    b.ld_mem_label_a("TOKC1")
    b.ld_a_de()
    b.cp_n(ord("A"))
    b.jr_c("TOK_FIRST_LOW")
    b.cp_n(ord("Z") + 1)
    b.jr_nc("TOK_FIRST_LOW")
    b.add_a_n(0x20)
    b.label("TOK_FIRST_LOW")
    b.ld_mem_label_a("TOKC2")
    b.inc_de()
    b.ld_a_mem_label("TOKLEN")
    b.dec_a()
    b.ld_mem_label_a("TOKLEN")

    b.label("TOK_LOOP")
    b.ld_a_mem_label("TOKLEN")
    b.or_a()
    b.jr_z("TOK_TRAIL")
    b.ld_a_de()
    b.cp_n(ord("A"))
    b.jr_c("TOK_LOW1")
    b.cp_n(ord("Z") + 1)
    b.jr_nc("TOK_LOW1")
    b.add_a_n(0x20)
    b.label("TOK_LOW1")
    b.ld_mem_label_a("TOKC3")
    b.call("TOK_HASH")
    b.ld_a_mem_label("TOKC2")
    b.ld_mem_label_a("TOKC1")
    b.ld_a_mem_label("TOKC3")
    b.ld_mem_label_a("TOKC2")
    b.inc_de()
    b.ld_a_mem_label("TOKLEN")
    b.dec_a()
    b.ld_mem_label_a("TOKLEN")
    b.jr("TOK_LOOP")

    b.label("TOK_TRAIL")
    b.ld_a_n(ord(" "))  # the trailing pad
    b.ld_mem_label_a("TOKC3")
    b.call("TOK_HASH")
    b.jr("TOK_DONE")


def emit_tok_hash(b: Z80Builder, plat: Platform) -> None:
    """Emit TOK_HASH: ``((c1 * 31 + c2) * 31 + c3) & 127``, then bump the bucket."""
    b.label("TOK_HASH")
    b.push_de()

    b.ld_a_mem_label("TOKC1")
    b.ld_l_a()
    b.ld_h_n(0)
    b.push_hl()
    for _ in range(5):
        b.add_hl_hl()
    b.pop_de()
    b.or_a()
    b.sbc_hl_de()  # * 31
    b.ld_a_mem_label("TOKC2")
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()

    b.push_hl()
    for _ in range(5):
        b.add_hl_hl()
    b.pop_de()
    b.or_a()
    b.sbc_hl_de()  # * 31
    b.ld_a_mem_label("TOKC3")
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()

    b.ld_a_l()
    b.and_n(NUM_BUCKETS - 1)

    # buffer[bucket] += BUCKET_WEIGHT
    b.ld_l_a()
    b.ld_h_n(0)
    b.add_hl_hl()
    b.push_de()
    b.ld_de_label(plat.buffer)
    b.add_hl_de()
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.ld_bc_nn(BUCKET_WEIGHT)
    b.ex_de_hl()
    b.add_hl_bc()
    b.ex_de_hl()
    b.ld_a_d()
    b.ld_hl_a()
    b.dec_hl()
    b.ld_a_e()
    b.ld_hl_a()
    b.pop_de()
    b.pop_de()
    b.ret()

    b.label("TOK_DONE")
    b.ret()


# --- data --------------------------------------------------------------------


def emit_charset_table(b: Z80Builder, charset: str) -> None:
    """Emit CHARTBL, mapping an output index to the byte to print."""
    b.label("CHARTBL")
    for c in charset:
        b.db(0 if c == "\x00" else ord(c))


def emit_layer_variables(b: Z80Builder) -> None:
    """Emit the scratch the packed LAYER and MULADD need."""
    for name in ("SAVCNT", "SAVW", "SAVB", "CURIN"):
        b.label(name)
        b.dw(0)
    for name in ("PACKED", "WEIGHT"):
        b.label(name)
        b.db(0)
    b.label("ACC")
    b.dw(0)


def emit_engine_variables(b: Z80Builder) -> None:
    """Emit the scratch shared by argmax, generation, tokenizing and context."""
    b.label("MAXV")
    b.dw(0)
    for name in ("MAXI", "RESULT", "GENCNT", "TOKLEN", "TOKC1", "TOKC2", "TOKC3",
                 "CTXPOS", "CTXN"):
        b.label(name)
        b.db(0)
    b.label("CTXCHARS")
    b.ds(CONTEXT_LEN)


def emit_variables(b: Z80Builder) -> None:
    """Emit every shared variable, in the order the packed backends expect."""
    emit_layer_variables(b)
    emit_engine_variables(b)


def emit_buffers(b: Z80Builder, plat: Platform, layer_sizes: Sequence[int]) -> None:
    """Emit the activation buffers: input vector, two scratch, and output."""
    b.label(plat.buffer)
    b.ds(layer_sizes[0] * ACTIVATION_SIZE)

    hidden = layer_sizes[1:-1] or [layer_sizes[-1]]
    max_hidden = max(hidden)
    b.label("BUF_A")
    b.ds(max_hidden * ACTIVATION_SIZE)
    b.label("BUF_B")
    b.ds(max_hidden * ACTIVATION_SIZE)
    b.label("OUTBUF")
    b.ds(layer_sizes[-1] * ACTIVATION_SIZE)


def emit_weights(
    b: Z80Builder, packed_weights: list[bytes], biases: list
) -> None:
    """Emit the packed weight stream and 16-bit biases for every layer."""
    for i, (weights, bias) in enumerate(zip(packed_weights, biases, strict=True), start=1):
        b.label(f"WTS{i}")
        b.blob(weights)
        b.label(f"BIAS{i}")
        for v in bias:
            b.dw(int(v) & 0xFFFF)
