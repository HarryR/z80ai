#!/usr/bin/env python3
"""
Build Z80 Autoregressive Character Generation .COM file

This generates Z80 machine code for character-by-character text generation:
1. Tokenize query into first 128 buckets (trigram hashing)
2. Initialize context (next 128 buckets) to zero
3. Loop:
   a. Run neural network inference (256 inputs → 64 character outputs)
   b. Argmax to find best character
   c. If EOS (index 63), stop
   d. Print the character via BDOS
   e. Update context encoding with new character
   f. Repeat until EOS or max length

Like buildz80com.py but uses indices for for each non-zero weight.  That costs
a smidge over 1 byte per weight but since close to 75% of the weights are zero
it tends to be only about 5 more KB.  Also, the input and output values are
stored in split form in 256 byte aligned buffers.  Instead of the low byte of
a value being in the next byte, it is 256 bytes away.  Normally loading a 16
bit value pointed to be HL goes like this:
    ld   e,(hl)
    inc  hl
    ld   d,(hl)
    inc  hl
With split values the sequence is:
    ld   e,(hl)
    inc  h
    ld   d,(hl)
    dec  h
    inc  l
Awkward, but worth it when mapping an index to a value is "ld l,c" instead
of "ld b,0; sla c; rl b; add hl,bc"  And by using the stack pointer step
through the weights forces a very beneficial unrolling by 2 of the summation
loop.

Advantages:
    Close to 20 times faster than the packed weights version and over
    twice as fast as the skip list version.

Disadvantages:
    May produce a .com file too large if zero weights are less common.
    Uses the stack pointer with interrupts disabled.
    Packed weight version could be considerably faster.

Would make sense to combine these into a single version which chooses the
fastest version that fits.

"""

import numpy as np
from libz80 import Z80Builder
from loadmodel import load_model_params

# Z80 Constants
BDOS = 0x0005
CPM_CMDLINE = 0x0080
MAX_OUTPUT_LEN = 50  # Maximum characters to generate


def pack_weights_and_biases(weights: np.ndarray, biases: np.ndarray) -> bytes:
    """Convert weights into index lists by weight and append bias after"""
    """each node."""
    wt_bias = []
    for n in range(0, weights.shape[0]):
        flat = np.clip(weights[n], -2, 1).astype(np.int8)
        for w in [ -2, -1, 1 ]:
            indices = []
            for i in range(0, len(flat)):
                if flat[i] == w:
                    indices.append(i)

            wt_bias.append(len(indices))
            wt_bias += indices
        bias_val = int(biases[n]) & 0xFFFF
        wt_bias.append(bias_val & 0xFF)
        wt_bias.append((bias_val >> 8) & 0xFF)

    return bytes(wt_bias)


def sum_wt(b: Z80Builder, w: int):
   if w > 0: b.add_a_hl()
   if w < 0: b.sub_a_hl()


def sum_wt_carry(b: Z80Builder, w: int):
   if w > 0: b.adc_a_hl()
   if w < 0: b.sbc_a_hl()


def build_autoreg(model_path: str = 'command_model_autoreg.pt'):
    """Build the autoregressive inference .COM"""

    # Load model (supports both .pt and .npz formats)
    print(f"Loading model from {model_path}...")
    params, arch, charset = load_model_params(model_path)

    eos_idx = len(charset) - 1
    num_chars = len(charset)
    print(f"Charset ({num_chars} chars): {repr(charset[:-1])} + EOS")

    # Discover layers
    layer_names = sorted(set(k.replace('_weight', '').replace('_bias', '')
                            for k in params.keys()))
    num_layers = len(layer_names)

    # Get layer dimensions
    layer_sizes = []
    for i, name in enumerate(layer_names):
        w = params[f'{name}_weight']
        if i == 0:
            layer_sizes.append(w.shape[1])
        layer_sizes.append(w.shape[0])

    input_size = layer_sizes[0]  # 256 (128 query + 128 context)
    output_size = layer_sizes[-1]  # 64 characters

    print(f"Architecture: {' → '.join(map(str, layer_sizes))}")
    print(f"Input: {input_size} (128 query + 128 context)")
    print(f"Output: {output_size} characters")

    # Pack weights and biases
    weights_biases = []
    for name in layer_names:
        weights_biases.append(pack_weights_and_biases(params[f'{name}_weight'], params[f'{name}_bias']))

    b = Z80Builder()

    # === MAIN ===
    b.label('START')
    # Check if command line is empty - if so, enter chat mode
    b.ld_hl_nn(CPM_CMDLINE)
    b.ld_a_hl()
    b.or_a()
    b.jp_z('CHAT')          # No args = interactive chat mode

    # Single query mode (args on command line)
    b.call('TOKENIZE')      # Tokenize query into first 128 buckets
    b.call('CLEAR_CTX')     # Clear context (last 128 buckets)
    b.call('GENERATE')      # Generation loop
    b.rst(0)                # Return to CP/M

    # === CHAT MODE: Interactive loop with '>' prompt ===
    b.label('CHAT')
    b.label('CHAT_LOOP')
    # Print newline
    b.ld_de_label('CRLF')
    b.ld_c_n(9)             # BDOS print string
    b.call_addr(BDOS)
    # Print prompt
    b.ld_e_n(ord('>'))
    b.ld_c_n(2)             # BDOS console output
    b.call_addr(BDOS)
    b.ld_e_n(ord(' '))
    b.ld_c_n(2)
    b.call_addr(BDOS)

    # Read line (BDOS function 10)
    b.ld_de_label('CHATBUF')
    b.ld_c_n(10)            # BDOS read console buffer
    b.call_addr(BDOS)

    # Print newline after input
    b.ld_de_label('CRLF')
    b.ld_c_n(9)
    b.call_addr(BDOS)

    # Check if empty input (just enter)
    b.ld_a_mem_label('CHATLEN')
    b.or_a()
    b.jr_z('CHAT_LOOP')     # Empty input, prompt again

    # Check for exit command (!)
    b.ld_a_mem_label('CHATDAT')
    b.cp_n(ord('!'))
    b.jp_z('CHAT_EXIT')     # Exit if first char is !

    # Copy input to CPM_CMDLINE format for TOKENIZE
    b.ld_a_mem_label('CHATLEN')
    b.ld_hl_nn(CPM_CMDLINE)
    b.ld_hl_a()             # Store length at 0080h
    b.ld_hl_label('CHATDAT')
    b.ld_de_nn(CPM_CMDLINE + 1)
    b.ld_c_a()
    b.ld_b_n(0)
    b.ldir()                # Copy input text

    # Process and generate response
    b.call('TOKENIZE')
    b.call('CLEAR_CTX')
    b.call('GENERATE')

    # Loop for next input
    b.jr('CHAT_LOOP')

    b.label('CHAT_EXIT')
    b.rst(0)                # Return to CP/M

    # === GENERATE: Main generation loop ===
    b.label('GENERATE')
    b.ld_a_n(MAX_OUTPUT_LEN)
    b.ld_mem_label_a('GENCNT')

    b.label('GENLOOP')

    # Copy INBUF to BUF_A, splitting values as we go

    b.ld_hl_label('INBUF')
    b.ld_de_label('BUF_A')
    b.label('INTOA')
    b.ld_a_hl()
    b.inc_hl()
    b.ld_de_a()
    b.ld_a_hl()
    b.inc_hl()
    b.inc_d()
    b.ld_de_a()
    b.dec_d()
    b.inc_e()
    b.jr_nz('INTOA')

    # Run inference through all layers
    b.ld_hl_label('NETWORK')
    b.call('INFER')

    # Find best character
    b.call('ARGMAX')

    # Check for EOS
    b.ld_a_mem_label('RESULT')
    b.cp_n(eos_idx)
    b.ret_z()  # Return if EOS

    # Print character
    b.call('PRINTCH')

    # Update context with new character
    b.call('UPDATE_CTX')

    # Loop if not done
    b.ld_a_mem_label('GENCNT')
    b.dec_a()
    b.ld_mem_label_a('GENCNT')
    b.jr_nz('GENLOOP')
    b.ret()

    # === PRINTCH: Print character from RESULT ===
    b.label('PRINTCH')
    b.ld_a_mem_label('RESULT')
    # Look up in character table
    b.ld_hl_label('CHARTBL')
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()
    b.ld_a_hl()
    b.ld_e_a()
    b.ld_c_n(2)  # BDOS console output
    b.call_addr(BDOS)
    b.ret()

    # === UPDATE_CTX: Update context encoding with new character ===
    # Context uses n-gram hashing with position info
    # We shift the context buffer and add new character contribution
    b.label('UPDATE_CTX')
    # Shift context characters left
    b.ld_hl_label('CTXCHARS')
    b.inc_hl()  # Point to char 1
    b.ld_de_label('CTXCHARS')  # Point to char 0
    b.ld_bc_nn(7)  # Copy 7 bytes
    b.ldir()

    # Store new character at end
    b.ld_a_mem_label('RESULT')
    b.ld_hl_label('CHARTBL')
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()
    b.ld_a_hl()
    # Convert to lowercase for hashing
    b.cp_n(ord('A'))
    b.jr_c('UPD_STORE')
    b.cp_n(ord('Z') + 1)
    b.jr_nc('UPD_STORE')
    b.add_a_n(0x20)
    b.label('UPD_STORE')
    b.ld_hl_label('CTXCHARS')
    b.ld_de_nn(7)
    b.add_hl_de()
    b.ld_hl_a()

    # Re-encode context into buckets
    b.call('ENCODE_CTX')
    b.ret()

    # === ENCODE_CTX: Encode CTXCHARS into context buckets (INBUF+256) ===
    b.label('ENCODE_CTX')
    # Clear context buckets (last 128 of INBUF)
    b.ld_hl_label('INBUF')
    b.ld_de_nn(256)  # 128 buckets * 2 bytes
    b.add_hl_de()
    b.ld_d_h()
    b.ld_e_l()
    b.inc_de()
    b.xor_a()
    b.ld_hl_a()
    b.ld_bc_nn(255)  # 128*2 - 1
    b.ldir()

    # Hash n-grams (1,2,3-grams with position)
    b.ld_a_n(0)
    b.ld_mem_label_a('CTXPOS')

    # For each n-gram length
    b.ld_a_n(1)
    b.ld_mem_label_a('CTXN')

    b.label('CTX_NLOOP')
    # For each position
    b.xor_a()
    b.ld_mem_label_a('CTXPOS')

    b.label('CTX_PLOOP')
    # Check if we have enough chars for this n-gram
    # max_pos = 8 - n + 1, if pos >= max_pos then done with this n
    b.ld_a_n(9)  # 8 + 1
    b.ld_hl_label('CTXN')
    b.sub_hl_ind()  # A = 9 - (CTXN) = max_pos
    b.ld_b_a()
    b.ld_a_mem_label('CTXPOS')
    b.cp_b()
    b.jr_nc('CTX_NEXT_N')

    # Hash this n-gram
    b.call('CTX_HASH')

    # Next position
    b.ld_a_mem_label('CTXPOS')
    b.inc_a()
    b.ld_mem_label_a('CTXPOS')
    b.jr('CTX_PLOOP')

    b.label('CTX_NEXT_N')
    b.ld_a_mem_label('CTXN')
    b.inc_a()
    b.ld_mem_label_a('CTXN')
    b.cp_n(4)  # n = 1,2,3
    b.jr_c('CTX_NLOOP')
    b.ret()

    # === CTX_HASH: Hash n-gram at position CTXPOS with length CTXN ===
    b.label('CTX_HASH')
    # hash = pos * 7
    b.ld_a_mem_label('CTXPOS')
    b.ld_l_a()
    b.ld_h_n(0)
    b.add_hl_hl()  # *2
    b.add_hl_hl()  # *4
    b.add_hl_hl()  # *8
    b.ld_d_h()
    b.ld_e_l()
    b.ld_a_mem_label('CTXPOS')
    b.ld_l_a()
    b.ld_h_n(0)
    b.ex_de_hl()
    b.or_a()
    b.sbc_hl_de()  # *7
    b.push_hl()  # Save hash

    # Get pointer to chars
    b.ld_hl_label('CTXCHARS')
    b.ld_a_mem_label('CTXPOS')
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()
    b.ex_de_hl()  # DE = char pointer

    b.pop_hl()  # Restore hash

    # For each char in n-gram
    b.ld_a_mem_label('CTXN')
    b.ld_b_a()

    b.label('CTX_HLOOP')
    b.push_bc()
    # hash = hash * 31 + char
    b.push_hl()
    b.add_hl_hl()  # *2
    b.add_hl_hl()  # *4
    b.add_hl_hl()  # *8
    b.add_hl_hl()  # *16
    b.add_hl_hl()  # *32
    b.pop_bc()
    b.or_a()
    b.sbc_hl_bc()  # *31
    b.ld_a_de()
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()  # + char
    b.inc_de()
    b.pop_bc()
    b.djnz('CTX_HLOOP')

    # bucket = (hash & 127) + 128
    b.ld_a_l()
    b.and_n(127)

    # Add to bucket (context is at INBUF + 256)
    b.ld_l_a()
    b.ld_h_n(0)
    b.add_hl_hl()  # *2 for word offset
    b.ld_de_label('INBUF')
    b.push_hl()
    b.ld_hl_nn(256)
    b.add_hl_de()
    b.ex_de_hl()
    b.pop_hl()
    b.add_hl_de()

    # Increment bucket value by 32
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.push_hl()
    b.ld_hl_nn(32)
    b.add_hl_de()
    b.ex_de_hl()
    b.pop_hl()
    b.ld_hl_d()
    b.dec_hl()
    b.ld_hl_e()
    b.ret()

    # === CLEAR_CTX: Initialize context with spaces ===
    b.label('CLEAR_CTX')
    # Set CTXCHARS to 8 spaces
    b.ld_hl_label('CTXCHARS')
    b.ld_b_n(8)
    b.label('CLR_LP')
    b.ld_hl_n(ord(' '))
    b.inc_hl()
    b.djnz('CLR_LP')

    # Encode the initial spaces into context buckets
    b.jp('ENCODE_CTX')  # This will return for us

    # === Inference Evaluation ===
    # HL points to NETWORK:
    #    1 byte   number of layers
    # Followed by the layers which are:
    #    1 byte   number of output values
    #    weight + bias data
    #
    # From this we load:
    #    E   number of layers
    #    D   number of outputs of layer
    #    HL  output buffer (only needs high byte)
    #    HL' input buffer
    # On return:
    #    B   number of outputs of last layer
    #    HL' last output buffer

    b.label('INFER');

    b.ld_e_hl() # number of layers
    b.inc_hl()

    b.ld_mem_label_sp('SPSAV')
    b.di()

    b.ld_sp_hl() # rest of the network data

    b.ld_hl_label('BUF_B') # output buffer
    b.exx()
    b.ld_hl_label('BUF_A') # input buffer
    b.exx()

    b.label('LAYER_LOOP')

    b.dec_sp()
    b.pop_af() # A = number of outputs
    b.ld_d_a() # now in D
    b.ld_b_a() # will be last ouput size for ARGMAX

    # SP=weights + biases, HL'=IN, HL=OUT, D=LEN(OUT), E=# of layers

    b.dec_e() # decrement so easier to test for E=1 in ReLU check

    b.label('LNEUR')

    b.exx()
    b.xor_a()
    b.ld_c_a() # accumulator = 0
    for w in [-2, -1, 1]:
        b.pop_de() # E = number of weight indices, D = first weight
        b.srl_e()
        b.jr_nc(f"even{w+2}") # no carry means even number of weights
        # Calculate the D weight we have
        b.ld_l_d()
        sum_wt(b, w)
        b.ld_d_a()
        b.inc_h()
        b.ld_a_c()
        sum_wt_carry(b, w)
        b.ld_c_a()
        b.ld_a_d()
        b.dec_h()
        b.inc_e()
        b.dec_e()
        b.db(0x16) # ld d,n (to skip adjustment of stack pointer)
        b.label(f"even{w+2}")
        b.dec_sp() # read 1 byte too much, back off
        b.jr_z(f"skip{w+2}")
        b.ld_b_e()
        b.label(f"wt{w+2}")
        b.pop_de()
        b.ld_l_e()
        sum_wt(b, w)
        b.ld_e_a()
        b.inc_h()
        b.ld_a_c()
        sum_wt_carry(b, w)
        b.ld_l_d()
        sum_wt(b, w)
        b.ld_c_a()
        b.dec_h()
        b.ld_a_e()
        sum_wt(b, w)
        b.jr_nc(f"c_ok{w+2}")
        if w > 0: b.inc_c()
        if w < 0: b.dec_c()
        b.label(f"c_ok{w+2}")
        b.djnz(f"wt{w+2}")
        if w == -2:
            b.add_a_a()
            b.rl_c()
        b.label(f"skip{w+2}")

    # Bias follows after weights
    b.pop_de()
    b.add_a_e()
    b.ld_e_a()
    b.ld_a_c()
    b.adc_a_d()
    b.ld_c_a()
    b.ld_a_e()

    # Scale output to keep within range.
    b.sra_c()
    b.rra()
    b.sra_c()
    b.rra()

    # Check if ReLU desired
    b.exx()
    b.inc_e()
    b.dec_e()
    b.exx()
    b.jr_z('NO_RELU')
    b.bit_7_c()
    b.jr_z('NO_RELU')
    b.xor_a()
    b.ld_c_a()
    b.label('NO_RELU')

    # write summation to output
    b.exx()
    b.ld_hl_a()
    b.inc_h()
    b.exx()
    b.ld_a_c()
    b.exx()
    b.ld_hl_a()
    b.dec_h()
    b.inc_l()

    # We're back in the regular registers
    b.dec_d()
    b.jp_nz('LNEUR')

    # Swap input and output buffers.  Could be done with an XOR to each H.
    # Also need to zero L
    # Or, considering only B and and E are live, just EXX and pull them over.
    # And B is just being cute, really.
    b.ld_l_n(0)
    b.ld_a_h() # A=H
    b.exx()
    b.ex_af_af()
    b.ld_l_n(0)
    b.ld_a_h() # A'=H'
    b.ex_af_af()
    b.ld_h_a() # H'=A (=H)
    b.exx()
    b.ex_af_af()
    b.ld_h_a() # H=A' (=H')

    b.inc_e() # was -1 as ReLU flag
    b.dec_e()
    b.jp_nz('LAYER_LOOP')

    b.ld_sp_mem_label('SPSAV')
    b.ei()

    b.ret()

    # === ARGMAX ===
    # HL' = layer values, B = layer size.  Exactly what INFER returns with.
    # Hastily fixed up for split values; code could be much improved.
    # Especially if we work backwards so L is our counter (though beware how
    # that could change things -- we should accept "=" to have same operation)
    b.label('ARGMAX')

    b.ld_a_b()
    b.exx()
    b.ld_b_a()

    b.ld_e_hl()
    b.inc_h()
    b.ld_d_hl()
    b.dec_h()
    b.inc_l()

    b.ld_mem_label_de('MAXV')
    b.xor_a()
    b.ld_mem_label_a('MAXI')
    b.ld_c_n(1)

    b.label('AMLP')
    b.ld_e_hl()
    b.inc_h()
    b.ld_d_hl()
    b.dec_h()
    b.inc_l()

    b.push_hl()
    b.ld_hl_mem_label('MAXV')
    b.push_de()
    b.or_a()
    b.ex_de_hl()
    b.sbc_hl_de()
    b.pop_de()
    b.jp_m('AMSK')
    b.jr_z('AMSK')
    b.ld_mem_label_de('MAXV')
    b.ld_a_c()
    b.ld_mem_label_a('MAXI')

    b.label('AMSK')
    b.pop_hl()
    b.inc_c()
    b.djnz('AMLP')
    b.ld_a_mem_label('MAXI')
    b.ld_mem_label_a('RESULT')
    b.ret()

    # === TOKENIZE (query into first 128 buckets) ===
    b.label('TOKENIZE')
    # Clear first 128 buckets of INBUF
    b.ld_hl_label('INBUF')
    b.ld_de_label('INBUF')
    b.inc_de()
    b.ld_bc_nn(255)  # 128 * 2 - 1
    b.ld_a_n(0)
    b.ld_hl_a()
    b.ldir()

    # Get length
    b.ld_hl_nn(CPM_CMDLINE)
    b.ld_a_hl()
    b.or_a()
    b.jp_z('TOK_DONE')
    b.ld_mem_label_a('TOKLEN')

    b.ld_de_nn(CPM_CMDLINE + 1)

    # Skip leading spaces
    b.label('TOK_SKIP_SPACE')
    b.ld_a_mem_label('TOKLEN')
    b.or_a()
    b.jp_z('TOK_DONE')
    b.ld_a_de()
    b.cp_n(ord(' '))
    b.jr_nz('TOK_START')
    b.inc_de()
    b.ld_a_mem_label('TOKLEN')
    b.dec_a()
    b.ld_mem_label_a('TOKLEN')
    b.jr('TOK_SKIP_SPACE')

    b.label('TOK_START')
    b.ld_a_n(ord(' '))
    b.ld_mem_label_a('TOKC1')
    b.ld_a_de()
    b.cp_n(ord('A'))
    b.jr_c('TOK_FIRST_LOW')
    b.cp_n(ord('Z') + 1)
    b.jr_nc('TOK_FIRST_LOW')
    b.add_a_n(0x20)
    b.label('TOK_FIRST_LOW')
    b.ld_mem_label_a('TOKC2')
    b.inc_de()
    b.ld_a_mem_label('TOKLEN')
    b.dec_a()
    b.ld_mem_label_a('TOKLEN')

    b.label('TOK_LOOP')
    b.ld_a_mem_label('TOKLEN')
    b.or_a()
    b.jr_z('TOK_TRAIL')
    b.ld_a_de()
    b.cp_n(ord('A'))
    b.jr_c('TOK_LOW1')
    b.cp_n(ord('Z') + 1)
    b.jr_nc('TOK_LOW1')
    b.add_a_n(0x20)
    b.label('TOK_LOW1')
    b.ld_mem_label_a('TOKC3')
    b.call('TOK_HASH')
    b.ld_a_mem_label('TOKC2')
    b.ld_mem_label_a('TOKC1')
    b.ld_a_mem_label('TOKC3')
    b.ld_mem_label_a('TOKC2')
    b.inc_de()
    b.ld_a_mem_label('TOKLEN')
    b.dec_a()
    b.ld_mem_label_a('TOKLEN')
    b.jr('TOK_LOOP')

    b.label('TOK_TRAIL')
    b.ld_a_n(ord(' '))
    b.ld_mem_label_a('TOKC3')
    b.call('TOK_HASH')
    b.jr('TOK_DONE')

    # === TOK_HASH ===
    b.label('TOK_HASH')
    b.push_de()
    b.ld_a_mem_label('TOKC1')
    b.ld_l_a()
    b.ld_h_n(0)
    b.push_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.pop_de()
    b.or_a()
    b.sbc_hl_de()
    b.ld_a_mem_label('TOKC2')
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()
    b.push_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.pop_de()
    b.or_a()
    b.sbc_hl_de()
    b.ld_a_mem_label('TOKC3')
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc()

    # bucket = L & 127 (first 128 buckets only)
    b.ld_a_l()
    b.and_n(127)

    # INBUF[bucket] += 32
    b.ld_l_a()
    b.ld_h_n(0)
    b.add_hl_hl()
    b.push_de()
    b.ld_de_label('INBUF')
    b.add_hl_de()
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.ld_bc_nn(32)
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

    b.label('TOK_DONE')
    b.ret()

    # === DATA ===
    # Character table (dynamic size based on charset)
    b.label('CHARTBL')
    for c in charset:
        if c == '\x00':
            b.db(0)  # EOS
        else:
            b.db(ord(c))

    b.label('CRLF')
    b.db(13, 10, ord('$'))

    # Variables
    b.label('SPSAV'); b.dw(0)
    b.label('MAXV'); b.dw(0)
    b.label('MAXI'); b.db(0)
    b.label('RESULT'); b.db(0)
    b.label('GENCNT'); b.db(0)
    b.label('TOKLEN'); b.db(0)
    b.label('TOKC1'); b.db(0)
    b.label('TOKC2'); b.db(0)
    b.label('TOKC3'); b.db(0)
    b.label('CTXPOS'); b.db(0)
    b.label('CTXN'); b.db(0)
    b.label('CTXCHARS'); b.ds(8)  # Last 8 output characters

    # Chat mode buffer (BDOS function 10 format)
    b.label('CHATBUF'); b.db(62)  # Max chars (buffer size - 2)
    b.label('CHATLEN'); b.db(0)   # Actual chars read (filled by BDOS)
    b.label('CHATDAT'); b.ds(62)  # Input text buffer

    b.label('NETWORK')
    b.db(num_layers)
    # Weights and biases
    for i in range(num_layers):
        b.db(layer_sizes[i + 1])
        b.db(*weights_biases[i])

    # Buffers
    b.align(256)
    if input_size > 256:
        raise ValueError(f"Input size {input_size} is too big; limit 256.")
    b.label('INBUF'); b.ds(256 * 2)  # 256 buckets * 2 bytes
    max_hidden = max(layer_sizes[1:-1]) if len(layer_sizes) > 2 else layer_sizes[1]
    if max_hidden > 256:
        raise ValueError(f"Layer size {max_hidden} is too big; limit 256.")
    b.label('BUF_A'); b.ds(256 * 2)
    b.label('BUF_B'); b.ds(256 * 2)

    return b


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build Z80 autoregressive .COM')
    parser.add_argument('--model', '-m', type=str, default='command_model_autoreg.pt',
                        help='Model file to load')
    parser.add_argument('--output', '-o', type=str, default='z80/CHAT.COM',
                        help='Output .COM file')
    args = parser.parse_args()

    print("Building autoregressive CHAT.COM...\n")

    b = build_autoreg(args.model)

    # Show key addresses
    print("\nKey addresses:")
    for name in ['START', 'GENERATE', 'LAYER', 'ARGMAX', 'TOKENIZE', 'UPDATE_CTX', 'CHARTBL']:
        if name in b.labels:
            print(f"  {name}: {b.labels[name]:04X}h")

    b.save(args.output)
    print(f"\nTotal size: {len(b.code)} bytes ({len(b.code)/1024:.1f} KB)")
    print(f"Saved to {args.output}")
