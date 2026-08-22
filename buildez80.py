#!/usr/bin/env python3
"""
Build an eZ80 (ADL mode) binary for Agon Light / Console8 and friends.

The Z80 builds spend most of their effort working around 64KB: weights are
squeezed to two bits, activations to 16 bits, and layers are capped at 256
neurons because DJNZ counts in a byte.  An eZ80 in ADL mode has a 24-bit
address space - up to 16MB - so this backend drops those compromises:

  * one byte per weight, so there is no unpacking work in the inner loop
  * 24-bit accumulators, so the 16-bit overflow the QAT loss trains against
    cannot happen at all
  * 24-bit activations, so nothing has to be re-scaled between layers
  * neuron and layer counts terminated by sentinels rather than byte counters,
    so layers may be any width

Inference is otherwise identical to the Z80 version: the same trigram hashing,
the same {-2,-1,0,+1} weights, the same >>2 per layer, the same argmax.  A
model built for CP/M runs here unchanged and produces the same text, except
where the Z80 would have overflowed its accumulator - there the eZ80 is right
and the Z80 is not.

Usage:
    python buildez80.py --model examples/guess/model.npz --output CHAT.bin
"""

from __future__ import annotations

import numpy as np

from libez80 import AGON_LOAD_ADDR, EZ80Builder, agon_header
from libinfer import discover_layers
from loadmodel import load_model_params

MAX_OUTPUT_LEN = 50
MAX_INPUT_LEN = 120
NUM_BUCKETS = 128
CONTEXT_LEN = 8

# MOS entry points.
MOS_API = 0x08  # RST 08h, function number in A
MOS_OUTCHAR = 0x10  # RST 10h, character in A
MOS_GETKEY = 0x00

# Weight stream encoding. Values outside {-2,-1,0,1} act as terminators.
W_END_NEURON = 0x02
W_END_LAYER = 0x03


def encode_weights(weights: np.ndarray) -> bytes:
    """One signed byte per weight, each neuron closed by an end sentinel."""
    w = np.clip(np.asarray(weights), -2, 1).astype(np.int8)
    out = bytearray()
    for row in w:
        out.extend(int(v) & 0xFF for v in row)
        out.append(W_END_NEURON)
    out.append(W_END_LAYER)
    return bytes(out)


def encode_biases(biases: np.ndarray) -> bytes:
    """Sign-extended 24-bit biases, so they add directly to the accumulator."""
    out = bytearray()
    for v in np.asarray(biases).astype(np.int64):
        value = int(v) & 0xFFFFFF
        out.extend((value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF))
    return bytes(out)


def build_autoreg(model_path: str = 'command_model_autoreg.pt',
                  max_output_len: int = MAX_OUTPUT_LEN,
                  org: int = AGON_LOAD_ADDR) -> EZ80Builder:
    """Build the eZ80 autoregressive inference binary."""
    print(f"Loading model from {model_path}...")
    params, _arch, charset = load_model_params(model_path)

    eos_idx = len(charset) - 1
    num_chars = len(charset)
    print(f"Charset ({num_chars} chars): {charset[:-1]!r} + EOS")

    layer_names, layer_sizes = discover_layers(params)
    num_layers = len(layer_names)

    output_size = layer_sizes[-1]
    if output_size < 2:
        raise ValueError("charset must have at least two entries")

    print(f"Architecture: {' → '.join(map(str, layer_sizes))}")

    weight_blobs = [encode_weights(params[f'{n}_weight']) for n in layer_names]
    bias_blob = b''.join(encode_biases(params[f'{n}_bias']) for n in layer_names)

    b = EZ80Builder(org=org)
    agon_header(b, 'START')

    # === Entry ===============================================================
    b.label('START')
    b.jp('CHAT_LOOP')

    b.label('CHAT_LOOP')
    b.call('PRNL')
    b.ld_a_n(ord('>'))
    b.rst(MOS_OUTCHAR)
    b.ld_a_n(ord(' '))
    b.rst(MOS_OUTCHAR)

    b.call('READ_INPUT')

    b.ld_a_mem_label('INPLEN')
    b.or_a()
    b.jr_z('CHAT_LOOP')

    b.ld_a_mem_label('INPBUF')
    b.cp_n(ord('!'))
    b.jp_z('CHAT_EXIT')

    b.call('TOKENIZE')
    b.call('CLEAR_CTX')
    b.call('GENERATE')
    b.jp('CHAT_LOOP')

    b.label('CHAT_EXIT')
    b.call('PRNL')
    b.ret()  # back to MOS

    # === PRNL: newline =======================================================
    b.label('PRNL')
    b.ld_a_n(13)
    b.rst(MOS_OUTCHAR)
    b.ld_a_n(10)
    b.rst(MOS_OUTCHAR)
    b.ret()

    # === READ_INPUT: line editor over mos_getkey ==============================
    b.label('READ_INPUT')
    b.xor_a()
    b.ld_mem_label_a('INPLEN')

    b.label('RI_LOOP')
    b.ld_a_n(MOS_GETKEY)
    b.rst(MOS_API)
    b.or_a()
    b.jr_z('RI_LOOP')  # no key ready
    b.cp_n(13)
    b.jr_z('RI_DONE')
    b.cp_n(8)
    b.jr_z('RI_DELETE')
    b.cp_n(127)
    b.jr_z('RI_DELETE')
    b.cp_n(32)
    b.jr_c('RI_LOOP')  # ignore other control codes

    # Buffer full? Keep the character in C so the compare's flags survive.
    b.ld_c_a()
    b.ld_a_mem_label('INPLEN')
    b.cp_n(MAX_INPUT_LEN)
    b.jr_nc('RI_LOOP')

    # INPBUF[INPLEN++] = C, then echo it. A still holds INPLEN.
    b.ld_hl_label('INPBUF')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_hl_c()
    b.inc_a()
    b.ld_mem_label_a('INPLEN')
    b.ld_a_c()
    b.rst(MOS_OUTCHAR)
    b.jr('RI_LOOP')

    b.label('RI_DELETE')
    b.ld_a_mem_label('INPLEN')
    b.or_a()
    b.jr_z('RI_LOOP')
    b.dec_a()
    b.ld_mem_label_a('INPLEN')
    for code in (8, 32, 8):
        b.ld_a_n(code)
        b.rst(MOS_OUTCHAR)
    b.jr('RI_LOOP')

    b.label('RI_DONE')
    b.call('PRNL')
    b.ret()

    # === GENERATE ============================================================
    b.label('GENERATE')
    b.ld_a_n(max_output_len)
    b.ld_mem_label_a('GENCNT')

    b.label('GENLOOP')
    b.call('INFER')
    b.call('ARGMAX')

    b.ld_a_mem_label('RESULT')
    b.cp_n(eos_idx)
    b.ret_z()

    b.call('PRINTCH')
    b.call('UPDATE_CTX')

    b.ld_a_mem_label('GENCNT')
    b.dec_a()
    b.ld_mem_label_a('GENCNT')
    b.jr_nz('GENLOOP')
    b.ret()

    # === PRINTCH =============================================================
    b.label('PRINTCH')
    b.ld_a_mem_label('RESULT')
    b.ld_hl_label('CHARTBL')
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()
    b.ld_a_hl()
    b.rst(MOS_OUTCHAR)
    b.ret()

    # === INFER: run every layer ==============================================
    # Buffers ping-pong; the assignment is fixed at build time so the layer
    # setup is unrolled rather than table-driven.
    b.label('INFER')
    b.ld_hl_label('BIASES')
    b.ld_mem_label_hl('BIASP')

    for i in range(num_layers):
        in_buf = 'INBUF' if i == 0 else ('BUF_A' if i % 2 == 1 else 'BUF_B')
        out_buf = 'OUTBUF' if i == num_layers - 1 else (
            'BUF_A' if (i + 1) % 2 == 1 else 'BUF_B'
        )
        b.label(f'LAYER{i+1}')
        b.ld_hl_label(in_buf)
        b.ld_mem_label_hl('INBASE')
        b.ld_ix_label(out_buf)
        b.ld_bc_label(f'WTS{i+1}')
        b.ld_a_n(0 if i == num_layers - 1 else 1)
        b.ld_mem_label_a('RELUF')
        b.call('LAYER')
    b.ret()

    # === LAYER ===============================================================
    # BC  weight stream (one signed byte per weight, sentinel terminated)
    # SP  input pointer - POP reads a 24-bit activation and advances in one go
    # HL  24-bit accumulator
    # IX  output pointer
    b.label('LAYER')
    b.ld_mem_label_sp('SPSAV')
    b.di()

    b.label('LNEUR')
    b.ld_sp_mem_label('INBASE')
    b.ld_hl_nn(0)

    b.label('LWT')
    b.pop_de()          # DE = next activation
    b.ld_a_bc()         # A  = next weight
    b.inc_bc()
    b.or_a()
    b.jr_z('LWT')       # weight 0: nothing to add (the common case)
    b.jp_m('LNEG')
    b.dec_a()
    b.jr_nz('LEND')     # sentinel: end of this neuron
    b.add_hl_de()       # weight +1
    b.jr('LWT')

    b.label('LNEG')
    b.or_a()            # clear carry before each SBC
    b.sbc_hl_de()
    b.inc_a()
    b.jr_z('LWT')       # weight -1
    b.or_a()
    b.sbc_hl_de()       # weight -2
    b.jr('LWT')

    b.label('LEND')
    # Bias, read through SP so all 24 bits land in DE at once.
    b.ld_sp_mem_label('BIASP')
    b.pop_de()
    b.ld_mem_label_sp('BIASP')
    b.add_hl_de()

    # Arithmetic shift right by 2 across all three bytes.
    b.ld_mem_label_hl('TMP0')
    for _ in range(2):
        b.ld_hl_label('TMP2')
        b.sra_hl_ind()
        b.dec_hl()
        b.rr_hl_ind()
        b.dec_hl()
        b.rr_hl_ind()

    b.ld_a_mem_label('RELUF')
    b.or_a()
    b.jr_z('NORELU')
    b.ld_a_mem_label('TMP2')
    b.or_a()
    b.jp_p('NORELU')
    b.xor_a()
    b.ld_mem_label_a('TMP0')
    b.ld_mem_label_a('TMP1')
    b.ld_mem_label_a('TMP2')

    b.label('NORELU')
    b.ld_a_mem_label('TMP0')
    b.ld_ixd_a(0)
    b.ld_a_mem_label('TMP1')
    b.ld_ixd_a(1)
    b.ld_a_mem_label('TMP2')
    b.ld_ixd_a(2)
    b.inc_ix()
    b.inc_ix()
    b.inc_ix()

    b.ld_a_bc()
    b.cp_n(W_END_LAYER)
    b.jp_nz('LNEUR')

    b.inc_bc()
    b.ld_sp_mem_label('SPSAV')
    b.ei()
    b.ret()

    # === ARGMAX ==============================================================
    b.label('ARGMAX')
    b.ld_mem_label_sp('SPSAV')
    b.di()
    b.ld_sp_label('OUTBUF')
    b.pop_de()  # running maximum
    b.xor_a()
    b.ld_mem_label_a('MAXI')
    b.ld_b_n(output_size - 1)
    b.ld_c_n(1)

    b.label('AMLP')
    b.pop_hl()
    b.ld_mem_label_hl('TMPV')
    b.or_a()
    b.sbc_hl_de()
    b.jp_m('AMSK')
    b.jr_z('AMSK')
    b.ld_de_mem_label('TMPV')
    b.ld_a_c()
    b.ld_mem_label_a('MAXI')

    b.label('AMSK')
    b.inc_c()
    b.djnz('AMLP')

    b.ld_sp_mem_label('SPSAV')
    b.ei()
    b.ld_a_mem_label('MAXI')
    b.ld_mem_label_a('RESULT')
    b.ret()

    # === LOWER: fold A-Z to lower case, everything else untouched ============
    b.label('LOWER')
    b.cp_n(ord('A'))
    b.ret_c()
    b.cp_n(ord('Z') + 1)
    b.ret_nc()
    b.add_a_n(0x20)
    b.ret()

    # === BUCKET_ADD: (HL + 3*A) += 32 ========================================
    b.label('BUCKET_ADD')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.ex_de_hl()        # HL = index, DE = base
    b.add_hl_hl()       # index * 2
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()       # index * 3
    b.add_hl_de()       # base + index * 3
    b.ld_a_hl()
    b.add_a_n(32)
    b.ld_hl_a()
    b.inc_hl()
    b.ld_a_hl()
    b.adc_a_n(0)
    b.ld_hl_a()
    b.inc_hl()
    b.ld_a_hl()
    b.adc_a_n(0)
    b.ld_hl_a()
    b.ret()

    # === TOKENIZE: query text -> first 128 buckets ===========================
    b.label('TOKENIZE')
    b.ld_hl_label('INBUF')
    b.ld_de_label('INBUF')
    b.inc_de()
    b.ld_bc_nn(NUM_BUCKETS * 3 - 1)
    b.ld_hl_n(0)
    b.ldir()

    b.ld_a_mem_label('INPLEN')
    b.or_a()
    b.jp_z('TOK_DONE')
    b.ld_mem_label_a('TOKLEN')
    b.ld_de_label('INPBUF')

    b.label('TOK_SKIP')
    b.ld_a_mem_label('TOKLEN')
    b.or_a()
    b.jp_z('TOK_DONE')
    b.ld_a_de()
    b.cp_n(ord(' '))
    b.jr_nz('TOK_START')
    b.inc_de()
    b.call('TOK_DECLEN')
    b.jr('TOK_SKIP')

    b.label('TOK_START')
    b.ld_a_n(ord(' '))
    b.ld_mem_label_a('TOKC1')
    b.ld_a_de()
    b.call('LOWER')
    b.ld_mem_label_a('TOKC2')
    b.inc_de()
    b.call('TOK_DECLEN')

    b.label('TOK_LOOP')
    b.ld_a_mem_label('TOKLEN')
    b.or_a()
    b.jr_z('TOK_TRAIL')
    b.ld_a_de()
    b.call('LOWER')
    b.ld_mem_label_a('TOKC3')
    b.call('TOK_HASH')
    b.ld_a_mem_label('TOKC2')
    b.ld_mem_label_a('TOKC1')
    b.ld_a_mem_label('TOKC3')
    b.ld_mem_label_a('TOKC2')
    b.inc_de()
    b.call('TOK_DECLEN')
    b.jr('TOK_LOOP')

    b.label('TOK_TRAIL')
    b.ld_a_n(ord(' '))
    b.ld_mem_label_a('TOKC3')
    b.call('TOK_HASH')

    b.label('TOK_DONE')
    b.ret()

    b.label('TOK_DECLEN')
    b.ld_a_mem_label('TOKLEN')
    b.dec_a()
    b.ld_mem_label_a('TOKLEN')
    b.ret()

    # === TOK_HASH: h = ((c1*31 + c2)*31 + c3), bucket = h & 127 ==============
    b.label('TOK_HASH')
    b.push_de()
    b.ld_hl_nn(0)
    b.ld_a_mem_label('TOKC1')
    b.ld_l_a()
    b.call('HASH_STEP2')
    b.ld_a_mem_label('TOKC2')
    b.call('HASH_ADD')
    b.call('HASH_STEP2')
    b.ld_a_mem_label('TOKC3')
    b.call('HASH_ADD')
    b.ld_a_l()
    b.and_n(NUM_BUCKETS - 1)
    b.ld_hl_label('INBUF')
    b.call('BUCKET_ADD')
    b.pop_de()
    b.ret()

    # HASH_STEP2: HL *= 31.  HASH_ADD: HL += A.
    # Both scratch through BC, never DE: CTX_HASH keeps its character pointer
    # in DE across the whole loop.
    b.label('HASH_STEP2')
    b.push_hl()
    for _ in range(5):
        b.add_hl_hl()
    b.pop_bc()
    b.or_a()
    b.sbc_hl_bc()
    b.ret()

    b.label('HASH_ADD')
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()
    b.ret()

    # === CLEAR_CTX / UPDATE_CTX / ENCODE_CTX =================================
    b.label('CLEAR_CTX')
    b.ld_hl_label('CTXCHARS')
    b.ld_b_n(CONTEXT_LEN)
    b.label('CLR_LP')
    b.ld_hl_n(ord(' '))
    b.inc_hl()
    b.djnz('CLR_LP')
    b.jp('ENCODE_CTX')

    b.label('UPDATE_CTX')
    b.ld_hl_label('CTXCHARS')
    b.inc_hl()
    b.ld_de_label('CTXCHARS')
    b.ld_bc_nn(CONTEXT_LEN - 1)
    b.ldir()

    b.ld_a_mem_label('RESULT')
    b.ld_hl_label('CHARTBL')
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()
    b.ld_a_hl()
    b.call('LOWER')
    b.ld_hl_label('CTXLAST')
    b.ld_hl_a()
    b.jp('ENCODE_CTX')

    b.label('ENCODE_CTX')
    b.ld_hl_label('CTXBUF')
    b.ld_de_label('CTXBUF')
    b.inc_de()
    b.ld_bc_nn(NUM_BUCKETS * 3 - 1)
    b.ld_hl_n(0)
    b.ldir()

    b.ld_a_n(1)
    b.ld_mem_label_a('CTXN')

    b.label('CTX_NLOOP')
    b.xor_a()
    b.ld_mem_label_a('CTXPOS')

    b.label('CTX_PLOOP')
    b.ld_a_n(CONTEXT_LEN + 1)
    b.ld_hl_label('CTXN')
    b.sub_hl_ind()  # A = (context_len + 1) - n, the exclusive position bound
    b.ld_b_a()
    b.ld_a_mem_label('CTXPOS')
    b.cp_b()
    b.jr_nc('CTX_NEXT_N')
    b.call('CTX_HASH')
    b.ld_a_mem_label('CTXPOS')
    b.inc_a()
    b.ld_mem_label_a('CTXPOS')
    b.jr('CTX_PLOOP')

    b.label('CTX_NEXT_N')
    b.ld_a_mem_label('CTXN')
    b.inc_a()
    b.ld_mem_label_a('CTXN')
    b.cp_n(4)
    b.jr_c('CTX_NLOOP')
    b.ret()

    # === CTX_HASH: hash CTXN characters from CTXPOS, seeded with CTXPOS*7 ====
    b.label('CTX_HASH')
    b.ld_hl_nn(0)
    b.ld_a_mem_label('CTXPOS')
    b.ld_l_a()
    b.push_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()  # pos * 8
    b.pop_de()
    b.or_a()
    b.sbc_hl_de()  # pos * 7

    b.push_hl()
    b.ld_hl_label('CTXCHARS')
    b.ld_bc_nn(0)
    b.ld_a_mem_label('CTXPOS')
    b.ld_c_a()
    b.add_hl_bc()
    b.ex_de_hl()  # DE = &CTXCHARS[pos]
    b.pop_hl()

    b.ld_a_mem_label('CTXN')
    b.ld_b_a()

    b.label('CTX_HLOOP')
    b.push_bc()
    b.call('HASH_STEP2')
    b.ld_a_de()
    b.call('HASH_ADD')
    b.inc_de()
    b.pop_bc()
    b.djnz('CTX_HLOOP')

    b.ld_a_l()
    b.and_n(NUM_BUCKETS - 1)
    b.ld_hl_label('CTXBUF')
    b.jp('BUCKET_ADD')

    # === DATA ================================================================
    b.label('CHARTBL')
    for c in charset:
        b.db(0 if c == '\x00' else ord(c))

    for name in ('TOKLEN', 'TOKC1', 'TOKC2', 'TOKC3', 'CTXPOS', 'CTXN',
                 'MAXI', 'RESULT', 'GENCNT', 'RELUF', 'INPLEN'):
        b.label(name)
        b.db(0)

    # A three-byte scratch the layer shifts in place, byte by byte.
    for name in ('TMP0', 'TMP1', 'TMP2'):
        b.label(name)
        b.db(0)

    for name in ('SPSAV', 'INBASE', 'BIASP', 'TMPV'):
        b.label(name)
        b.d24(0)

    b.label('CTXCHARS')
    b.ds(CONTEXT_LEN - 1)
    b.label('CTXLAST')
    b.db(0)

    b.label('INPBUF')
    b.ds(MAX_INPUT_LEN + 1)

    # Activation buffers: 24-bit values, three bytes of slack because the
    # inner loop pops one activation past the last weight.
    b.label('INBUF')
    b.ds(NUM_BUCKETS * 3)
    b.label('CTXBUF')
    b.ds(NUM_BUCKETS * 3)
    b.ds(3)

    hidden = layer_sizes[1:-1] or [layer_sizes[-1]]
    max_hidden = max(hidden)
    b.label('BUF_A')
    b.ds(max_hidden * 3 + 3)
    b.label('BUF_B')
    b.ds(max_hidden * 3 + 3)
    b.label('OUTBUF')
    b.ds(output_size * 3 + 3)

    b.label('BIASES')
    b.blob(bias_blob)

    for i, blob in enumerate(weight_blobs, start=1):
        b.label(f'WTS{i}')
        b.blob(blob)

    return b


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Build eZ80 (Agon) inference binary')
    parser.add_argument('--model', '-m', default='command_model_autoreg.pt',
                        help='Model file to load (.npz or .pt)')
    parser.add_argument('--output', '-o', default='CHAT.bin',
                        help='Output MOS binary')
    parser.add_argument('--max-output-len', type=int, default=MAX_OUTPUT_LEN,
                        help='Maximum characters generated per response')
    args = parser.parse_args()

    print("Building eZ80 CHAT.bin...\n")
    b = build_autoreg(args.model, max_output_len=args.max_output_len)

    print("\nKey addresses:")
    for name in ('START', 'GENERATE', 'LAYER', 'ARGMAX', 'TOKENIZE', 'BIASES', 'WTS1'):
        if name in b.labels:
            print(f"  {name}: {b.labels[name]:06X}h")

    b.save(args.output)
    size = len(b.code)
    print(f"\nTotal size: {size:,} bytes ({size / 1024:.1f} KB)")
    print(f"Loads at {AGON_LOAD_ADDR:06X}h, runs in ADL mode")


if __name__ == '__main__':
    main()
