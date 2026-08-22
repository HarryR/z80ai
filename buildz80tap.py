#!/usr/bin/env python3
"""
Build a ZX Spectrum 48K .TAP for character-by-character text generation.

The engine is shared with the CP/M build (see :mod:`libnn`); what differs is
I/O — characters go out through the ROM print routine and input comes from a
keyboard loop rather than BDOS — and the load address.

RAM ends at FFFFh, so the load address bounds how large a model can be. See
``ORG_ADDR`` below and ZX-SPECTRUM.md.
"""

from __future__ import annotations

import argparse

import numpy as np

import libnn
from libinfer import discover_layers, pack_2bit, validate_z80_layers
from libz80 import Z80Builder
from loadmodel import load_model_params

# ZX Spectrum ROM entry points.
ZX_PRINT_A = 0x0010  # RST 10h - print the character in A
ZX_CLS = 0x0DAF  # clear screen
ZX_CHAN_OPEN = 0x1601  # open a stream channel
ZX_KEY_INPUT = 0x10A8  # wait for a key, return it in A

#: Maximum characters to generate before giving up on ever seeing an EOS.
MAX_OUTPUT_LEN = 50
#: Longest query the input line will accept.
MAX_INPUT_LEN = 62

# Memory layout for ZX Spectrum 48K.
#
# RAM runs to FFFFh, so the load address bounds how large a model can be: at the
# old 8000h only 32,768 bytes were available, which both shipped examples
# exceed. 6000h is the lowest address that is clear of the screen (4000-5AFFh),
# the printer buffer (5B00-5BFFh) and the system variables (5C00-5CCAh), and
# leaves room below it for the BASIC loader once RAMTOP is moved down with
# CLEAR. That gives 40,960 bytes.
ORG_ADDR = 0x6000
ZX_RAM_TOP = 0x10000  # one past the last byte of RAM on a 48K machine

# ZX character codes.
ZX_ENTER = 13
ZX_DELETE = 12
ZX_BACKSPACE = 8
ZX_SPACE = 32


class ZXPlatform(libnn.Platform):
    """ZX Spectrum: characters go through RST 10h, input via a keyboard loop."""

    name = "ZX Spectrum"
    buffer = "TOKBUF"
    weight_layout = "rotated"

    def print_char(self, b: Z80Builder) -> None:
        b.rst(ZX_PRINT_A)

    def load_query_length(self, b: Z80Builder) -> None:
        b.ld_a_mem_label("INPLEN")

    def load_query_pointer(self, b: Z80Builder) -> None:
        b.ld_de_label("INPBUF")


def pack_2bit_weights(weights: np.ndarray) -> bytes:
    """Pack 2-bit weights, four per byte, one output neuron per whole bytes.

    Uses the same scrambled nibble order as the CP/M build so both share one
    inner loop; see :func:`libinfer.pack_2bit`.
    """
    return pack_2bit(weights, layout="rotated")


def build_tap_header(filename: str, start: int, length: int) -> bytes:
    """Build a TAP header block describing a CODE file.

    A TAP block is ``[length:2][flag:1][payload][checksum:1]``; the checksum is
    the XOR of the flag and payload bytes.
    """
    header = bytearray()
    header.append(3)  # file type 3 = CODE
    header.extend(filename[:10].ljust(10).encode("ascii"))
    header.append(length & 0xFF)
    header.append((length >> 8) & 0xFF)
    header.append(start & 0xFF)
    header.append((start >> 8) & 0xFF)
    header.extend((0, 0))  # unused for CODE

    checksum = 0
    for byte in header:
        checksum ^= byte  # the 00h flag byte XORs to nothing

    block = bytearray()
    block.append(19)  # flag + 17 header bytes + checksum
    block.append(0)
    block.append(0x00)  # header block
    block.extend(header)
    block.append(checksum)
    return bytes(block)


def build_tap_data(data: bytes) -> bytes:
    """Build a TAP data block wrapping ``data``."""
    checksum = 0xFF  # seeded with the data block flag
    for byte in data:
        checksum ^= byte

    length = len(data) + 2  # flag + checksum
    block = bytearray()
    block.append(length & 0xFF)
    block.append((length >> 8) & 0xFF)
    block.append(0xFF)  # data block
    block.extend(data)
    block.append(checksum)
    return bytes(block)


def emit_read_input(b: Z80Builder) -> None:
    """Emit READ_INPUT: a keyboard line editor over the ROM key routine."""
    b.label("READ_INPUT")
    b.ld_hl_label("INPBUF")
    b.ld_b_n(MAX_INPUT_LEN)
    b.xor_a()
    b.ld_mem_label_a("INPLEN")

    b.label("RI_LOOP")
    b.call_addr(ZX_KEY_INPUT)

    b.cp_n(ZX_ENTER)
    b.jr_z("RI_DONE")
    b.cp_n(ZX_DELETE)
    b.jr_z("RI_DELETE")
    b.cp_n(ZX_SPACE)
    b.jr_c("RI_LOOP")  # ignore other control codes

    # Buffer full? Stash the character in C rather than on the stack: POP AF
    # would restore the flags from before the CP and the branch below would
    # then test the CP 32 above instead. LD A,C preserves flags.
    b.ld_c_a()
    b.ld_a_mem_label("INPLEN")
    b.cp_b()
    b.ld_a_c()
    b.jr_nc("RI_LOOP")

    b.push_af()
    b.push_hl()
    b.ld_hl_label("INPBUF")
    b.ld_c_a()
    b.ld_a_mem_label("INPLEN")
    b.ld_e_a()
    b.ld_d_n(0)
    b.add_hl_de()
    b.ld_a_c()
    b.ld_hl_a()

    b.ld_a_mem_label("INPLEN")
    b.inc_a()
    b.ld_mem_label_a("INPLEN")

    b.pop_hl()
    b.pop_af()
    b.rst(ZX_PRINT_A)  # echo
    b.jr("RI_LOOP")

    b.label("RI_DELETE")
    b.ld_a_mem_label("INPLEN")
    b.or_a()
    b.jr_z("RI_LOOP")
    b.dec_a()
    b.ld_mem_label_a("INPLEN")
    for code in (ZX_BACKSPACE, ZX_SPACE, ZX_BACKSPACE):
        b.ld_a_n(code)
        b.rst(ZX_PRINT_A)
    b.jr("RI_LOOP")

    b.label("RI_DONE")
    b.ld_a_n(ZX_ENTER)
    b.rst(ZX_PRINT_A)
    b.ret()


def build_autoreg(
    model_path: str = "command_model_autoreg.pt",
    max_output_len: int = MAX_OUTPUT_LEN,
    org: int = ORG_ADDR,
) -> Z80Builder:
    """Assemble the inference engine and model into a Spectrum CODE image.

    Args:
        model_path: A ``.npz`` or ``.pt`` model.
        max_output_len: Characters to generate before giving up on an EOS.
        org: Load address. Bounds the model size, since RAM ends at FFFFh.

    Returns:
        The builder, with all labels resolvable.

    Raises:
        ValueError: If a layer is too wide for a Z80 neuron loop, or the image
            would not fit in RAM at ``org``.
    """
    print(f"Loading model from {model_path}...")
    params, _arch, charset = load_model_params(model_path)

    eos_idx = len(charset) - 1
    print(f"Charset ({len(charset)} chars): {charset[:-1]!r} + EOS")

    layer_names, layer_sizes = discover_layers(params)
    num_layers = len(layer_names)
    input_size, output_size = layer_sizes[0], layer_sizes[-1]

    print(f"Architecture: {' → '.join(map(str, layer_sizes))}")
    print(f"Input: {input_size} (128 query + 128 context)")
    print(f"Output: {output_size} characters")

    validate_z80_layers(layer_sizes)

    packed_weights = [pack_2bit_weights(params[f"{n}_weight"]) for n in layer_names]
    biases = [params[f"{n}_bias"] for n in layer_names]

    plat = ZXPlatform()
    plans = libnn.plan_layers(layer_sizes, plat.buffer)
    b = Z80Builder(org=org)

    # === Entry ===
    b.label("START")
    b.di()
    b.ld_a_n(2)  # channel 2, the upper screen
    b.call_addr(ZX_CHAN_OPEN)
    b.call_addr(ZX_CLS)
    b.ei()
    b.jp("CHAT")

    # === Chat mode ===
    b.label("CHAT")
    b.label("CHAT_LOOP")
    b.ld_a_n(ZX_ENTER)
    b.rst(ZX_PRINT_A)
    for ch in "> ":
        b.ld_a_n(ord(ch))
        b.rst(ZX_PRINT_A)

    b.call("READ_INPUT")

    b.ld_a_mem_label("INPLEN")
    b.or_a()
    b.jr_z("CHAT_LOOP")  # empty line, prompt again

    b.ld_a_mem_label("INPBUF")
    b.cp_n(ord("!"))
    b.jp_z("CHAT_EXIT")

    b.call("TOKENIZE")
    b.call("CLEAR_CTX")
    b.call("GENERATE")
    b.jp("CHAT_LOOP")

    b.label("CHAT_EXIT")
    b.ld_a_n(ZX_ENTER)
    b.rst(ZX_PRINT_A)
    b.ret()  # back to BASIC

    emit_read_input(b)

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
    libnn.emit_variables(b)

    b.label("INPLEN")
    b.db(0)
    b.label("INPBUF")
    b.ds(MAX_INPUT_LEN)

    libnn.emit_buffers(b, plat, layer_sizes)
    libnn.emit_weights(b, packed_weights, biases)

    # A .TAP whose image runs past FFFFh cannot load on any Spectrum, so refuse
    # to emit one rather than shipping a tape that fails halfway through.
    end = b.org + len(b.build())
    if end > ZX_RAM_TOP:
        raise ValueError(
            f"image is {len(b.code):,} bytes and would run to {end:#07x} from "
            f"{b.org:#06x}, past the top of RAM ({ZX_RAM_TOP - 1:#06x}) by "
            f"{end - ZX_RAM_TOP:,} bytes. Lower --org or train a smaller model: "
            f"{ZX_RAM_TOP - b.org:,} bytes are available at {b.org:#06x}."
        )

    return b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Z80 autoregressive .TAP for ZX Spectrum"
    )
    parser.add_argument("--model", "-m", default="command_model_autoreg.pt",
                        help="Model file to load")
    parser.add_argument("--output", "-o", default="CHAT.TAP",
                        help="Output .TAP file")
    parser.add_argument("--max-output-len", type=int, default=MAX_OUTPUT_LEN,
                        help="Maximum characters generated per response")
    parser.add_argument("--org", type=lambda v: int(v, 0), default=ORG_ADDR,
                        help=f"Load address (default {ORG_ADDR:#06x})")
    args = parser.parse_args()

    print("Building ZX Spectrum CHAT.TAP...\n")
    b = build_autoreg(args.model, max_output_len=args.max_output_len, org=args.org)

    print("\nKey addresses:")
    for name in ("START", "GENERATE", "LAYER", "ARGMAX", "TOKENIZE",
                 "UPDATE_CTX", "CHARTBL"):
        if name in b.labels:
            print(f"  {name}: {b.labels[name]:04X}h")

    image = b.build()
    tap_data = build_tap_header("CHAT", b.org, len(image)) + build_tap_data(image)

    with open(args.output, "wb") as fh:
        fh.write(tap_data)

    headroom = ZX_RAM_TOP - (b.org + len(image))
    print(f"\nTotal code size: {len(image)} bytes ({len(image) / 1024:.1f} KB)")
    print(f"TAP file size: {len(tap_data)} bytes")
    print(f"Loads at {b.org:#06x}-{b.org + len(image) - 1:#06x}, "
          f"{headroom:,} bytes of RAM to spare")
    print(f"Saved to {args.output}")
    print("\nIn ZX Spectrum BASIC:")
    print(f"  CLEAR {b.org - 1}")
    print('  LOAD "" CODE')
    print(f"  RANDOMIZE USR {b.org}")


if __name__ == "__main__":
    main()
