#!/usr/bin/env python3
"""
Build a CP/M .COM for character-by-character text generation.

Weights are packed two bits each, four to a byte, which always fits the
transient program area but makes the inner loop spend most of its time
unpacking. ``buildfastz80com.py`` trades size for roughly nine times the speed;
``build.py --target auto`` picks whichever fits.

Run with a query on the command line for a single answer, or with no arguments
for an interactive chat prompt.
"""

from __future__ import annotations

import argparse

import numpy as np

import libnn
from libinfer import discover_layers, pack_2bit, validate_z80_layers
from libz80 import Z80Builder
from loadmodel import load_model_params

#: BDOS entry point.
BDOS = 0x0005
#: CP/M leaves the command tail here: a length byte followed by the text.
CPM_CMDLINE = 0x0080
#: Maximum characters to generate before giving up on ever seeing an EOS.
MAX_OUTPUT_LEN = 50
#: Size of the chat-mode input line (BDOS function 10 buffer).
CHAT_BUFFER_SIZE = 62

# BDOS function numbers.
BDOS_CONSOLE_OUT = 2
BDOS_PRINT_STRING = 9
BDOS_READ_LINE = 10


class CPMPlatform(libnn.Platform):
    """CP/M: characters go through BDOS, the query arrives in the command tail."""

    name = "CP/M"
    buffer = "INBUF"
    weight_layout = "rotated"

    def print_char(self, b: Z80Builder) -> None:
        b.ld_e_a()
        b.ld_c_n(BDOS_CONSOLE_OUT)
        b.call_addr(BDOS)

    def load_query_length(self, b: Z80Builder) -> None:
        b.ld_hl_nn(CPM_CMDLINE)
        b.ld_a_hl()

    def load_query_pointer(self, b: Z80Builder) -> None:
        b.ld_de_nn(CPM_CMDLINE + 1)


def pack_2bit_weights(weights: np.ndarray) -> bytes:
    """Pack 2-bit weights, four per byte, one output neuron per whole bytes.

    The nibble order is scrambled so MULADD can decide between {-2,-1,0,+1}
    with two DECs, putting the most common weight (zero) on the fastest path.
    See :func:`libinfer.pack_2bit` for the encoding.
    """
    return pack_2bit(weights, layout="rotated")


def build_autoreg(
    model_path: str = "command_model_autoreg.pt",
    max_output_len: int = MAX_OUTPUT_LEN,
) -> Z80Builder:
    """Assemble the inference engine and model into a CP/M .COM image.

    Args:
        model_path: A ``.npz`` or ``.pt`` model.
        max_output_len: Characters to generate before giving up on an EOS.

    Returns:
        The builder, with all labels resolvable.

    Raises:
        ValueError: If a layer is wider than a Z80 neuron loop can count.
    """
    print(f"Loading model from {model_path}...")
    params, _arch, charset = load_model_params(model_path)

    eos_idx = len(charset) - 1
    print(f"Charset ({len(charset)} chars): {charset[:-1]!r} + EOS")

    # Sorted numerically, not lexically: a 10-layer model would otherwise run
    # fc10 straight after fc1.
    layer_names, layer_sizes = discover_layers(params)
    input_size, output_size = layer_sizes[0], layer_sizes[-1]

    print(f"Architecture: {' → '.join(map(str, layer_sizes))}")
    print(f"Input: {input_size} (128 query + 128 context)")
    print(f"Output: {output_size} characters")

    validate_z80_layers(layer_sizes)

    packed_weights = [pack_2bit_weights(params[f"{n}_weight"]) for n in layer_names]
    biases = [params[f"{n}_bias"] for n in layer_names]

    plat = CPMPlatform()
    plans = libnn.plan_layers(layer_sizes, plat.buffer)
    b = Z80Builder()

    # === Entry: a command tail means one query, no arguments means chat ===
    b.label("START")
    b.ld_hl_nn(CPM_CMDLINE)
    b.ld_a_hl()
    b.or_a()
    b.jp_z("CHAT")

    b.call("TOKENIZE")
    b.call("CLEAR_CTX")
    b.call("GENERATE")
    b.rst(0)  # warm boot, back to CP/M

    # === Chat mode ===
    b.label("CHAT")
    b.label("CHAT_LOOP")
    b.ld_de_label("CRLF")
    b.ld_c_n(BDOS_PRINT_STRING)
    b.call_addr(BDOS)
    for ch in "> ":
        b.ld_e_n(ord(ch))
        b.ld_c_n(BDOS_CONSOLE_OUT)
        b.call_addr(BDOS)

    b.ld_de_label("CHATBUF")
    b.ld_c_n(BDOS_READ_LINE)
    b.call_addr(BDOS)

    b.ld_de_label("CRLF")
    b.ld_c_n(BDOS_PRINT_STRING)
    b.call_addr(BDOS)

    b.ld_a_mem_label("CHATLEN")
    b.or_a()
    b.jr_z("CHAT_LOOP")  # empty line, prompt again

    b.ld_a_mem_label("CHATDAT")
    b.cp_n(ord("!"))
    b.jp_z("CHAT_EXIT")

    # Stage the line where TOKENIZE expects to find it.
    b.ld_a_mem_label("CHATLEN")
    b.ld_hl_nn(CPM_CMDLINE)
    b.ld_hl_a()
    b.ld_hl_label("CHATDAT")
    b.ld_de_nn(CPM_CMDLINE + 1)
    b.ld_c_a()
    b.ld_b_n(0)
    b.ldir()

    b.call("TOKENIZE")
    b.call("CLEAR_CTX")
    b.call("GENERATE")
    b.jr("CHAT_LOOP")

    b.label("CHAT_EXIT")
    b.rst(0)

    # === Shared engine ===
    libnn.emit_generate(b, plat, eos_idx, max_output_len,
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
    libnn.emit_argmax(b, output_size)
    libnn.emit_tokenizer(b, plat)
    libnn.emit_tok_hash(b, plat)

    # === Data ===
    libnn.emit_charset_table(b, charset)
    b.label("CRLF")
    b.db(13, 10, ord("$"))
    libnn.emit_variables(b)

    b.label("CHATBUF")
    b.db(CHAT_BUFFER_SIZE)  # capacity, read by BDOS
    b.label("CHATLEN")
    b.db(0)  # length, written by BDOS
    b.label("CHATDAT")
    b.ds(CHAT_BUFFER_SIZE)

    libnn.emit_buffers(b, plat, layer_sizes)
    libnn.emit_weights(b, packed_weights, biases)

    return b


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Z80 autoregressive .COM")
    parser.add_argument("--model", "-m", default="command_model_autoreg.pt",
                        help="Model file to load")
    parser.add_argument("--output", "-o", default="z80/CHAT.COM",
                        help="Output .COM file")
    parser.add_argument("--max-output-len", type=int, default=MAX_OUTPUT_LEN,
                        help="Maximum characters generated per response")
    args = parser.parse_args()

    print("Building autoregressive CHAT.COM...\n")
    b = build_autoreg(args.model, max_output_len=args.max_output_len)

    print("\nKey addresses:")
    for name in ("START", "GENERATE", "LAYER", "ARGMAX", "TOKENIZE",
                 "UPDATE_CTX", "CHARTBL"):
        if name in b.labels:
            print(f"  {name}: {b.labels[name]:04X}h")

    b.save(args.output)
    print(f"\nTotal size: {len(b.code)} bytes ({len(b.code) / 1024:.1f} KB)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
