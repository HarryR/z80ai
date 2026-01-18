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
"""

import numpy as np
from libz80 import Z80Builder
from loadmodel import load_model_params

# Z80 Constants
BDOS = 0x0005
CPM_CMDLINE = 0x0080
MAX_OUTPUT_LEN = 50  # Maximum characters to generate


def pack_2bit_weights(weights: np.ndarray) -> bytes:
    """Pack 2-bit weights: 4 per byte, LSB first"""
    flat = weights.flatten()
    mapped = np.clip(flat + 2, 0, 3).astype(np.uint8)

    packed = []
    for i in range(0, len(mapped), 4):
        chunk = mapped[i:i+4]
        if len(chunk) < 4:
            chunk = np.pad(chunk, (0, 4 - len(chunk)), constant_values=2)
        byte = \
            (int(chunk[2]) << 6) | \
            (int(chunk[1]) << 4) | \
            (int(chunk[0]) << 2) | \
            (int(chunk[3])) # chunk 3 as last since in the evaluation there will be first a rotation
        packed.append(byte)

    return bytes(packed)


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
    packed_weights = []
    biases = []
    for name in layer_names:
        packed_weights.append(pack_2bit_weights(params[f'{name}_weight']))
        biases.append(params[f'{name}_bias'])

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
    # Run inference through all layers
    for i in range(num_layers):
        b.call(f'LAYER{i+1}')
        if i < num_layers - 1:
            b.call(f'RELU{i+1}')

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
    b.ld_a_n(ord(' '))
    for _ in range(8):
        b.ld_hl_a()
        b.inc_hl()

    # Encode the initial spaces into context buckets
    b.jp('ENCODE_CTX')  # This will return for us

    # === Generate layer dispatch stubs ===
    for i in range(num_layers):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]

        if i == 0:
            in_buf = 'INBUF'
        else:
            in_buf = 'BUF_A' if (i % 2 == 1) else 'BUF_B'

        if i == num_layers - 1:
            out_buf = 'OUTBUF'
        else:
            out_buf = 'BUF_A' if ((i + 1) % 2 == 1) else 'BUF_B'

        b.label(f'LAYER{i+1}')
        b.ld_hl_label(f'WTS{i+1}')
        b.ld_de_label(f'BIAS{i+1}')
        b.ld_ix_label(in_buf)
        b.ld_iy_label(out_buf)
        b.ld_b_n(out_size if out_size <= 255 else 0)
        b.ld_c_n(in_size if in_size <= 255 else 0)
        if i == num_layers - 1:
            pass  # Fall through
        else:
            b.jp('LAYER')

    # === LAYER (same as build_nn.py) ===
    b.label('LAYER')            # LAYER:                                          
    b.ld_mem_label_bc('SAVCNT') #   ld (SAVCNT), bc                             
    b.ld_mem_label_hl('SAVW')   #   ld (SAVW), hl                           
    b.ld_mem_label_de('SAVB')   #   ld (SAVB), de                           
                                # 
    b.label('LNEUR')            # LNEUR:                    
    b.push_bc()                 #   push bc             
    b.ld_hl_nn(0)               #   ld hl, 0               
    b.ld_mem_label_hl('ACC')    #   ld (ACC), hl                          
    b.push_ix()                 #   push ix             
    b.pop_hl()                  #   pop hl            
    b.ld_mem_label_hl('CURIN')  #   ld (CURIN), hl                              
    b.ld_hl_mem_label('SAVW')   #   ld hl, (SAVW)                           
    b.ld_a_mem_label('SAVCNT')  #   ld a, (SAVCNT)                            
    b.ld_b_a()                  #   ld b, a            
    b.ld_c_n(0)                 #   ld c, 0             
                                # 
    b.label('LWT')              # LWT:                  
    b.ld_a_c()                  #   ld a,c            
    b.and_n(0x03)               #   and 11b               
    b.jr_nz('LSAME')            #   jr nz, LSAME                  
    b.ld_hl_mem_label('SAVW')   #   ld hl, (SAVW)                           
    b.ld_a_hl()                 #   ld a, (hl)             
    b.ld_mem_label_a('PACKED')  #   ld (PACKED), a                            
    b.inc_hl()                  #   inc hl            
    b.ld_mem_label_hl('SAVW')   #   ld (SAVW), hl                           
                                # 
    b.label('LSAME')            # LSAME:
                                #   ; Unpack 2-bit weight from packed byte                    
    b.ld_a_mem_label('PACKED')  #   ld a, (PACKED) ; Get packed weights                           
    b.rrca()                    #   rrca ; Rotate for next weight          
    b.rrca()                    #   rrca          
    b.ld_mem_label_a('PACKED')  #   ld (PACKED),a
    b.and_n(0x03)               #   and 11b        ; Mask bottom 2 bits       
                                #   ; 0,1,2,3 will be considered -2,-1,0,+1 in MULADD  
    b.ld_hl_mem_label('CURIN')  #   ld hl, (CURIN)                            
    b.ld_e_hl()                 #   ld e, (hl)             
    b.inc_hl()                  #   inc hl            
    b.ld_d_hl()                 #   ld d, (hl)             
    b.inc_hl()                  #   inc hl            
    b.ld_mem_label_hl('CURIN')  #   ld (CURIN), hl                            
    b.call('MULADD')            #   call MULADD                  
    b.inc_c()                   #   inc c           
    b.djnz('LWT')               #   djnz LWT               
                                # 
    b.ld_hl_mem_label('SAVB')   #   ld hl, (SAVB)                           
    b.ld_e_hl()                 #   ld e, (hl)             
    b.inc_hl()                  #   inc hl            
    b.ld_d_hl()                 #   ld d, (hl)             
    b.inc_hl()                  #   inc hl            
    b.ld_mem_label_hl('SAVB')   #   ld (SAVB), hl                           
    b.ld_hl_mem_label('ACC')    #   ld hl, (ACC)                          
    b.add_hl_de()               #   add hl, de               
    b.ld_mem_label_hl('ACC')    #   ld (ACC), hl
                                #   ; After each layer, arithmetic right-shift by 2 to prevent overflow                          
    b.sra_h()                   #   sra h          ; Shift right arithmetic (preserves sign)    
    b.rr_l()                    #   rr l          
    b.sra_h()                   #   sra h           
    b.rr_l()                    #   rr l           ; ACC = ACC / 4
    b.ld_iyd_l(0)               #   ld (iy+0), l                
    b.ld_iyd_h(1)               #   ld (iy+1), h              
    b.inc_iy()                  #   inc iy            
    b.inc_iy()                  #   inc iy            
    b.pop_bc()                  #   pop bc            
    b.djnz('LNEUR')             #   dec b : jp nz, LNEUR                  
    b.ret()                     #   ret         
                                # 
    # === MULADD ===            #                     
    # a is 0,1,2,3 that are as -2,-1,0,+1
    b.label('MULADD')           # MULADD:                     
    # TODO: ACC should be put in a register
    b.ld_hl_mem_label('ACC')    #   ld hl, (ACC)                          
    b.dec_a()                   #   dec a          
    b.jr_z('MA_M1')             #   jr z, MA_M1 ; jump if a is -1 equivalent
    b.dec_a()                   #   dec a          
    b.ret_z()                   #   ret z ; a is zero equivalent
    b.dec_a()                   #   dec a          
    b.jr_z('MA_P1')             #   jr z, MA_P1 ; jump if a is +1 equivalent
    b.label('MA_M2')            # MA_M2: ; a is -2 equivalent                    
    b.sbc_hl_de()               #   sbc hl, de              
    b.label('MA_M1')            # MA_M1: ; -1                   
    b.sbc_hl_de()               #   sbc hl, de               
    b.ld_mem_label_hl('ACC')    #   ld (ACC), hl                          
    b.ret()                     #   ret         
    b.label('MA_P1')            # MA_P1: ; a is +1                     
    b.add_hl_de()               #   add hl, de               
    b.ld_mem_label_hl('ACC')    #   ld (ACC), hl                          
    b.ret()                     #   ret         

    # === ReLU stubs ===
    for i in range(num_layers - 1):
        out_size = layer_sizes[i + 1]
        buf_name = 'BUF_A' if ((i + 1) % 2 == 1) else 'BUF_B'

        b.label(f'RELU{i+1}')
        b.ld_hl_label(buf_name)
        b.ld_b_n(out_size if out_size <= 255 else 0)
        if i == num_layers - 2:
            pass
        else:
            b.jr('RELU')

    b.label('RELU')
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.bit_7_d()
    b.jr_z('RPOS')
    b.dec_hl()
    b.xor_a()
    b.ld_hl_a()
    b.inc_hl()
    b.ld_hl_a()
    b.label('RPOS')
    b.inc_hl()
    b.djnz('RELU')
    b.ret()

    # === ARGMAX ===
    b.label('ARGMAX')
    b.ld_hl_label('OUTBUF')
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.inc_hl()
    b.ld_mem_label_de('MAXV')
    b.xor_a()
    b.ld_mem_label_a('MAXI')
    b.ld_b_n(output_size - 1)
    b.ld_c_n(1)

    b.label('AMLP')
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.inc_hl()
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
    b.label('SAVCNT'); b.dw(0)
    b.label('SAVW'); b.dw(0)
    b.label('SAVB'); b.dw(0)
    b.label('CURIN'); b.dw(0)
    b.label('PACKED'); b.db(0)
    b.label('WEIGHT'); b.db(0)
    b.label('ACC'); b.dw(0)
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

    # Buffers
    b.label('INBUF'); b.ds(input_size * 2)  # 256 buckets * 2 bytes
    max_hidden = max(layer_sizes[1:-1]) if len(layer_sizes) > 2 else layer_sizes[1]
    b.label('BUF_A'); b.ds(max_hidden * 2)
    b.label('BUF_B'); b.ds(max_hidden * 2)
    b.label('OUTBUF'); b.ds(output_size * 2)

    # Weights and biases
    for i in range(num_layers):
        b.label(f'WTS{i+1}')
        b.db(*packed_weights[i])

        b.label(f'BIAS{i+1}')
        for v in biases[i]:
            b.dw(int(v) & 0xFFFF)

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
