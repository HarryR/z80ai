#!/usr/bin/env python3
"""
Build Z80 snapshot (.z80) from TAP file for ZX Spectrum 48K

This creates a Z80 snapshot with the code already loaded at the correct address,
ready to run immediately without LOAD "" CODE step.
"""

import sys
import struct


def create_z80_snapshot(code: bytes, start_addr: int = 0x8000, output_file: str = 'CHAT.z80'):
    """
    Create a ZX Spectrum 48K .z80 snapshot file (version 1 format)

    Args:
        code: The machine code to load
        start_addr: Address where code is loaded (default 0x8000)
        output_file: Output filename
    """

    # Initialize 48K RAM (all zeros)
    ram = bytearray(49152)  # 0x4000 to 0xFFFF (48K)

    # Copy code into RAM at the correct offset
    offset = start_addr - 0x4000
    ram[offset:offset + len(code)] = code

    # Build Z80 v1 header (30 bytes)
    header = bytearray(30)

    # Registers (most set to 0)
    header[0] = 0x3F    # A register
    header[1] = 0x58    # F register (flags)
    header[2] = 0x00    # C register
    header[3] = 0x00    # B register
    header[4] = 0x00    # L register
    header[5] = 0x00    # H register

    # PC (Program Counter) - set to start address
    header[6] = start_addr & 0xFF        # PC low
    header[7] = (start_addr >> 8) & 0xFF # PC high

    header[8] = 0x00    # SP low
    header[9] = 0xFF    # SP high (stack at top of memory)

    header[10] = 0x3F   # I register (interrupt vector)
    header[11] = 0x00   # R register (refresh)

    # Byte 12 bit 0: bit 7 of R register
    # Bits 1-3: border color (white = 7)
    # Bit 4: 1=SamRom switched (not used for 48K)
    # Bit 5: 1=compressed data
    header[12] = 0x0E   # Border white, not compressed

    header[13] = 0x00   # E register
    header[14] = 0x00   # D register
    header[15] = 0x00   # C' register
    header[16] = 0x00   # B' register
    header[17] = 0x00   # E' register
    header[18] = 0x00   # D' register
    header[19] = 0x00   # L' register
    header[20] = 0x00   # H' register
    header[21] = 0x00   # A' register
    header[22] = 0x00   # F' register

    header[23] = 0x00   # IY low
    header[24] = 0x5C   # IY high (0x5C3A = ERR_NR system variable)
    header[25] = 0x00   # IX low
    header[26] = 0x00   # IX high

    header[27] = 0x00   # Interrupt enable (0=DI, otherwise EI)
    header[28] = 0x00   # IFF2 (used for interrupt mode)

    # Bits 0-1: interrupt mode (0, 1, or 2)
    # Bit 2: 1=issue 2 emulation
    # Bit 3: 1=double interrupt frequency
    # Bits 4-5: video sync (0=normal)
    # Bits 6-7: joystick (0=none)
    header[29] = 0x01   # Interrupt mode 1

    # Write Z80 file
    with open(output_file, 'wb') as f:
        f.write(header)
        f.write(ram)

    print(f"Created Z80 snapshot: {output_file}")
    print(f"Code loaded at: 0x{start_addr:04X}")
    print(f"Code size: {len(code)} bytes")
    print(f"Total file size: {len(header) + len(ram)} bytes")
    print(f"\nTo run in emulator:")
    print(f"  fuse {output_file}")
    print(f"Or just load it - it will start automatically at 0x{start_addr:04X}")


def extract_code_from_tap(tap_file: str) -> bytes:
    """Extract the code data block from a TAP file"""
    with open(tap_file, 'rb') as f:
        data = f.read()

    pos = 0
    code_data = None

    while pos < len(data):
        # Read block length
        if pos + 2 > len(data):
            break

        block_len = data[pos] | (data[pos + 1] << 8)
        pos += 2

        if pos + block_len > len(data):
            break

        # Read flag byte
        flag = data[pos]

        if flag == 0xFF:  # Data block
            # Skip flag and checksum, get actual data
            code_data = data[pos + 1:pos + block_len - 1]
            break

        pos += block_len

    return code_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build Z80 snapshot from TAP file')
    parser.add_argument('tap_file', help='Input TAP file')
    parser.add_argument('--output', '-o', default='CHAT.z80', help='Output Z80 file')
    parser.add_argument('--start', '-s', type=lambda x: int(x, 0), default=0x8000,
                        help='Start address (default: 0x8000)')

    args = parser.parse_args()

    # Extract code from TAP
    print(f"Extracting code from {args.tap_file}...")
    code = extract_code_from_tap(args.tap_file)

    if code is None:
        print("Error: Could not find code data block in TAP file")
        sys.exit(1)

    # Create Z80 snapshot
    create_z80_snapshot(code, args.start, args.output)
