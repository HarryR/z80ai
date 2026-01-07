"""
Z80 utility functions and code builder.

Provides Z80Builder class for emitting machine code and debug utilities
for Z80 development.
"""

from typing import List, Dict, Tuple


class Z80Builder:
    """Simple Z80 code builder with label support"""

    def __init__(self, org: int = 0x0100):
        self.org = org
        self.code = bytearray()
        self.labels: Dict[str, int] = {}
        self.fixups: List[Tuple[int, str, str]] = []  # (offset, label, type)

    def addr(self) -> int:
        return self.org + len(self.code)

    def label(self, name: str):
        self.labels[name] = self.addr()

    def emit(self, *bytes):
        for b in bytes:
            self.code.append(b & 0xFF)

    def emit_word(self, val: int):
        self.emit(val & 0xFF, (val >> 8) & 0xFF)

    def align(self, boundary: int):
        overage = self.addr() % boundary
        if overage < boundary: self.ds(boundary - overage)

    def fixup_word(self, label: str):
        """Emit placeholder word, record fixup"""
        self.fixups.append((len(self.code), label, 'abs'))
        self.emit(0, 0)

    def fixup_rel(self, label: str):
        """Emit placeholder byte for relative jump"""
        self.fixups.append((len(self.code), label, 'rel'))
        self.emit(0)

    def resolve(self):
        """Apply all fixups"""
        for offset, label, ftype in self.fixups:
            if label not in self.labels:
                raise ValueError(f"Unknown label: {label}")
            target = self.labels[label]

            if ftype == 'abs':
                self.code[offset] = target & 0xFF
                self.code[offset + 1] = (target >> 8) & 0xFF
            elif ftype == 'rel':
                # Relative offset from instruction after the offset byte
                from_addr = self.org + offset + 1
                rel = target - from_addr
                if rel < -128 or rel > 127:
                    raise ValueError(f"Relative jump out of range: {label} = {rel}")
                self.code[offset] = rel & 0xFF

    def save(self, filename: str):
        self.resolve()
        with open(filename, 'wb') as f:
            f.write(self.code)
        print(f"Wrote {len(self.code)} bytes to {filename}")

    # === Z80 Instructions ===

    def nop(self): self.emit(0x00)
    def ret(self): self.emit(0xC9)
    def ret_z(self): self.emit(0xC8)  # RET Z
    def ret_nz(self): self.emit(0xC0)  # RET NZ
    def rst(self, n): self.emit(0xC7 | n)
    def halt(self): self.emit(0x76)
    def di(self): self.emit(0xF3)
    def ei(self): self.emit(0xFB)

    def call(self, label: str):
        self.emit(0xCD)
        self.fixup_word(label)

    def call_addr(self, addr: int):
        self.emit(0xCD, addr & 0xFF, (addr >> 8) & 0xFF)

    def jp(self, label: str):
        self.emit(0xC3)
        self.fixup_word(label)

    def jp_nz(self, label: str):
        self.emit(0xC2)
        self.fixup_word(label)

    def jp_z(self, label: str):
        self.emit(0xCA)
        self.fixup_word(label)

    def jp_m(self, label: str):
        self.emit(0xFA)
        self.fixup_word(label)

    def jr(self, label: str):
        self.emit(0x18)
        self.fixup_rel(label)

    def jr_nz(self, label: str):
        self.emit(0x20)
        self.fixup_rel(label)

    def jr_z(self, label: str):
        self.emit(0x28)
        self.fixup_rel(label)

    def jr_nc(self, label: str):
        self.emit(0x30)
        self.fixup_rel(label)

    def jr_c(self, label: str):
        self.emit(0x38)
        self.fixup_rel(label)

    def djnz(self, label: str):
        self.emit(0x10)
        self.fixup_rel(label)

    # Loads
    def ld_hl_nn(self, val): self.emit(0x21); self.emit_word(val)
    def ld_de_nn(self, val): self.emit(0x11); self.emit_word(val)
    def ld_bc_nn(self, val): self.emit(0x01); self.emit_word(val)
    def ld_ix_nn(self, val): self.emit(0xDD, 0x21); self.emit_word(val)
    def ld_iy_nn(self, val): self.emit(0xFD, 0x21); self.emit_word(val)

    def ld_hl_label(self, label: str):
        self.emit(0x21)
        self.fixup_word(label)

    def ld_bc_label(self, label: str):
        self.emit(0x01)
        self.fixup_word(label)

    def ld_de_label(self, label: str):
        self.emit(0x11)
        self.fixup_word(label)

    def ld_ix_label(self, label: str):
        self.emit(0xDD, 0x21)
        self.fixup_word(label)

    def ld_iy_label(self, label: str):
        self.emit(0xFD, 0x21)
        self.fixup_word(label)

    def ld_a_n(self, val): self.emit(0x3E, val & 0xFF)
    def ld_b_n(self, val): self.emit(0x06, val & 0xFF)
    def ld_c_n(self, val): self.emit(0x0E, val & 0xFF)
    def ld_d_n(self, val): self.emit(0x16, val & 0xFF)
    def ld_e_n(self, val): self.emit(0x1E, val & 0xFF)
    def ld_hl_n(self, val): self.emit(0x36, val & 0xFF)

    def ld_hl_mem_label(self, label: str):
        self.emit(0x2A)
        self.fixup_word(label)

    def ld_mem_label_hl(self, label: str):
        self.emit(0x22)
        self.fixup_word(label)

    def ld_mem_label_de(self, label: str):
        self.emit(0xED, 0x53)
        self.fixup_word(label)

    def ld_mem_label_bc(self, label: str):
        self.emit(0xED, 0x43)
        self.fixup_word(label)

    def ld_bc_mem_label(self, label: str):
        self.emit(0xED, 0x4B)
        self.fixup_word(label)

    def ld_mem_label_sp(self, label: str):
        self.emit(0xED, 0x73)
        self.fixup_word(label)

    def ld_sp_mem_label(self, label: str):
        self.emit(0xED, 0x7B)
        self.fixup_word(label)

    def ld_a_mem_label(self, label: str):
        self.emit(0x3A)
        self.fixup_word(label)

    def ld_mem_label_a(self, label: str):
        self.emit(0x32)
        self.fixup_word(label)

    def ld_a_bc(self): self.emit(0x0A)
    def ld_a_hl(self): self.emit(0x7E)
    def ld_hl_a(self): self.emit(0x77)
    def ld_e_hl(self): self.emit(0x5E)
    def ld_d_hl(self): self.emit(0x56)
    def ld_a_b(self): self.emit(0x78)
    def ld_a_h(self): self.emit(0x7C)
    def ld_a_l(self): self.emit(0x7D)
    def ld_a_d(self): self.emit(0x7A)
    def ld_b_a(self): self.emit(0x47)
    def ld_b_c(self): self.emit(0x41)
    def ld_b_hl(self): self.emit(0x46)
    def ld_c_a(self): self.emit(0x4F)
    def ld_d_a(self): self.emit(0x57)
    def ld_a_c(self): self.emit(0x79)
    def ld_e_a(self): self.emit(0x5F)
    def ld_e_c(self): self.emit(0x59)

    def ld_l_ixd(self, d): self.emit(0xDD, 0x6E, d & 0xFF)  # LD L,(IX+d)
    def ld_h_ixd(self, d): self.emit(0xDD, 0x66, d & 0xFF)  # LD H,(IX+d)
    def ld_ixd_l(self, d): self.emit(0xDD, 0x75, d & 0xFF)  # LD (IX+d),L
    def ld_ixd_h(self, d): self.emit(0xDD, 0x74, d & 0xFF)  # LD (IX+d),H
    def ld_iyd_l(self, d): self.emit(0xFD, 0x75, d & 0xFF)  # LD (IY+d),L
    def ld_iyd_h(self, d): self.emit(0xFD, 0x74, d & 0xFF)  # LD (IY+d),H

    def ld_sp_hl(self): self.emit(0xF9)
    def ld_sp_ix(self): self.emit(0xDD, 0xF9)
    def ld_sp_iy(self): self.emit(0xFD, 0xF9)

    # Arithmetic
    def add_a_n(self, val): self.emit(0xC6, val & 0xFF)
    def sub_n(self, val): self.emit(0xD6, val & 0xFF)
    def and_n(self, val): self.emit(0xE6, val & 0xFF)
    def or_a(self): self.emit(0xB7)
    def xor_a(self): self.emit(0xAF)
    def cp_n(self, val): self.emit(0xFE, val & 0xFF)
    def add_hl_de(self): self.emit(0x19)
    def sbc_hl_de(self): self.emit(0xED, 0x52)
    def sbc_hl_bc(self): self.emit(0xED, 0x42)  # SBC HL,BC
    def inc_bc(self): self.emit(0x03)
    def dec_bc(self): self.emit(0x0B)
    def inc_hl(self): self.emit(0x23)
    def dec_hl(self): self.emit(0x2B)
    def dec_sp(self): self.emit(0x3B)
    def inc_c(self): self.emit(0x0C)
    def dec_c(self): self.emit(0x0D)
    def dec_b(self): self.emit(0x05)
    def inc_d(self): self.emit(0x14)
    def dec_d(self): self.emit(0x15)
    def dec_h(self): self.emit(0x25)
    def inc_e(self): self.emit(0x1C)
    def dec_e(self): self.emit(0x1D)
    def inc_l(self): self.emit(0x2C)
    def inc_ix(self): self.emit(0xDD, 0x23)
    def inc_iy(self): self.emit(0xFD, 0x23)

    # Shifts
    def rrca(self): self.emit(0x0F)
    def rra(self): self.emit(0x1F)
    def rlca(self): self.emit(0x07)
    def sra_c(self): self.emit(0xCB, 0x29)
    def sra_h(self): self.emit(0xCB, 0x2C)
    def rr_l(self): self.emit(0xCB, 0x1D)
    def sla_l(self): self.emit(0xCB, 0x25)
    def rl_c(self): self.emit(0xCB, 0x11)
    def rl_h(self): self.emit(0xCB, 0x14)
    def srl_e(self): self.emit(0xCB, 0x3B)
    def add_hl_hl(self): self.emit(0x29)  # HL = HL * 2
    def add_hl_bc(self): self.emit(0x09)  # HL = HL + BC
    def add_hl_de(self): self.emit(0x19)  # HL = HL + DE
    def add_hl_sp(self): self.emit(0x39)  # HL = HL + SP

    # Bit
    def bit_7_c(self): self.emit(0xCB, 0x79)
    def bit_7_d(self): self.emit(0xCB, 0x7A)
    def bit_7_h(self): self.emit(0xCB, 0x7C)
    def bit_7_a(self): self.emit(0xCB, 0x7F)

    # More arithmetic
    def add_a_l(self): self.emit(0x85)
    def add_a_h(self): self.emit(0x84)
    def add_a_hl(self): self.emit(0x86)
    def adc_a_hl(self): self.emit(0x8E)
    def adc_a_d(self): self.emit(0x8A)
    def add_a_a(self): self.emit(0x87)
    def add_a_e(self): self.emit(0x83)
    def sub_l(self): self.emit(0x95)
    def sub_h(self): self.emit(0x94)
    def sub_a_hl(self): self.emit(0x96)
    def sbc_a_hl(self): self.emit(0x9E)
    def sub_hl_ind(self): self.emit(0x96)  # SUB (HL)
    def cp_hl(self): self.emit(0xBE)  # CP (HL)
    def cp_a(self): self.emit(0xBF)
    def cp_b(self): self.emit(0xB8)  # CP B
    def inc_a(self): self.emit(0x3C)
    def dec_a(self): self.emit(0x3D)
    def inc_de(self): self.emit(0x13)
    def inc_b(self): self.emit(0x04)
    def inc_h(self): self.emit(0x24)
    def or_c(self): self.emit(0xB1)
    def or_l(self): self.emit(0xB5)
    def and_a(self): self.emit(0xA7)
    def ld_l_a(self): self.emit(0x6F)
    def ld_h_a(self): self.emit(0x67)
    def ld_h_n(self, val): self.emit(0x26, val & 0xFF)
    def ld_l_n(self, val): self.emit(0x2E, val & 0xFF)
    def ld_a_de(self): self.emit(0x1A)  # LD A,(DE)
    def ld_de_a(self): self.emit(0x12)  # LD (DE),A
    def ld_a_e(self): self.emit(0x7B)
    def ld_b_e(self): self.emit(0x43)
    def ld_l_d(self): self.emit(0x6A)
    def ld_l_e(self): self.emit(0x6B)
    def ld_d_h(self): self.emit(0x54)  # LD D,H
    def ld_e_l(self): self.emit(0x5D)  # LD E,L
    def ld_hl_d(self): self.emit(0x72)  # LD (HL),D
    def ld_hl_e(self): self.emit(0x73)  # LD (HL),E

    # Stack
    def push_af(self): self.emit(0xF5)
    def push_bc(self): self.emit(0xC5)
    def push_de(self): self.emit(0xD5)
    def push_hl(self): self.emit(0xE5)
    def push_ix(self): self.emit(0xDD, 0xE5)
    def push_iy(self): self.emit(0xFD, 0xE5)
    def pop_af(self): self.emit(0xF1)
    def pop_bc(self): self.emit(0xC1)
    def pop_de(self): self.emit(0xD1)
    def pop_hl(self): self.emit(0xE1)
    def pop_ix(self): self.emit(0xDD, 0xE1)
    def pop_iy(self): self.emit(0xFD, 0xE1)

    # Block
    def ldir(self): self.emit(0xED, 0xB0)

    # Exchange
    def ex_de_hl(self): self.emit(0xEB)
    def ex_sp_hl(self): self.emit(0xE3)
    def ex_sp_ix(self): self.emit(0xDD, 0xE3)
    def ex_sp_iy(self): self.emit(0xFD, 0xE3)
    def ex_af_af(self): self.emit(0x08)
    def exx(self): self.emit(0xD9)

    # Data
    def db(self, *vals):
        for v in vals:
            self.emit(v)

    def dw(self, *vals):
        for v in vals:
            self.emit_word(v & 0xFFFF)

    def ds(self, n):
        for _ in range(n):
            self.emit(0)

    def ascii(self, s):
        for c in s:
            self.emit(ord(c))

def add_debug_utils(b, bdos_addr=0x0005):
    """
    Add debug utility routines to a Z80Builder.

    Provides:
    - PRHEX: Print A register as 2 hex digits (preserves BC, DE, HL)
    - PRNYB: Print single hex nibble in A
    - DBGBUF: Print B words from memory at (HL)
    - PRCRLF: Print CR/LF newline

    Usage:
        from z80_utils import add_debug_utils
        b = Z80Builder()
        # ... your code ...
        add_debug_utils(b)
    """

    # === PRHEX: Print A as 2 hex digits (preserves BC, DE, HL) ===
    b.label('PRHEX')
    b.push_bc()
    b.push_de()
    b.push_hl()
    b.ld_d_a()  # Save A in D

    # High nibble
    b.rrca()
    b.rrca()
    b.rrca()
    b.rrca()
    b.and_n(0x0F)
    b.call('PRNYB')

    # Low nibble
    b.ld_a_d()  # Restore saved value
    b.and_n(0x0F)
    b.call('PRNYB')

    b.pop_hl()
    b.pop_de()
    b.pop_bc()
    b.ret()

    # === PRNYB: Print nibble in A (0-F) ===
    b.label('PRNYB')
    b.cp_n(10)
    b.jr_nc('PRNYB_AF')
    b.add_a_n(0x30)  # 0-9 -> '0'-'9'
    b.jr('PRNYB_OUT')
    b.label('PRNYB_AF')
    b.add_a_n(0x37)  # 10-15 -> 'A'-'F'
    b.label('PRNYB_OUT')
    b.ld_e_a()
    b.ld_c_n(2)
    b.call_addr(bdos_addr)
    b.ret()

    # === DBGBUF: Print B words from (HL) as hex with spaces ===
    b.label('DBGBUF')
    b.label('DBGLP')
    b.push_bc()
    b.push_hl()

    # Print high byte first (big-endian display)
    b.inc_hl()
    b.ld_a_hl()
    b.call('PRHEX')

    # Print low byte
    b.pop_hl()
    b.push_hl()
    b.ld_a_hl()
    b.call('PRHEX')

    # Print space
    b.ld_a_n(0x20)
    b.ld_e_a()
    b.ld_c_n(2)
    b.call_addr(bdos_addr)

    b.pop_hl()
    b.inc_hl()
    b.inc_hl()
    b.pop_bc()
    b.djnz('DBGLP')
    b.ret()

    # === PRCRLF: Print CR/LF ===
    b.label('PRCRLF')
    b.push_bc()
    b.push_de()
    b.ld_a_n(13)
    b.ld_e_a()
    b.ld_c_n(2)
    b.call_addr(bdos_addr)
    b.ld_a_n(10)
    b.ld_e_a()
    b.ld_c_n(2)
    b.call_addr(bdos_addr)
    b.pop_de()
    b.pop_bc()
    b.ret()

    # Print space
    b.label('PRSPC')
    b.ld_a_n(0x20)
    b.label('PRCHAR')
    b.push_bc();
    b.push_de();
    b.ld_e_a()
    b.ld_c_n(2)
    b.call_addr(bdos_addr)
    b.pop_de();
    b.pop_bc();
    b.ret()

def add_print_string(b, bdos_addr=0x0005):
    """
    Add string printing utility.

    PRMSG: Print $-terminated string at (DE)
    """
    b.label('PRMSG')
    b.ld_c_n(9)
    b.call_addr(bdos_addr)
    b.ret()


def add_print_decimal(b, bdos_addr=0x0005):
    """
    Add decimal number printing (for small positive numbers 0-255).

    PRDEC: Print A as decimal (destroys A, uses stack)
    """
    b.label('PRDEC')
    b.ld_c_n(0)  # Digit count

    b.label('PRDEC_100')
    b.cp_n(100)
    b.jr_c('PRDEC_10')
    b.sub_n(100)
    b.inc_c()
    b.jr('PRDEC_100')

    b.label('PRDEC_10')
    # C now has hundreds digit
    b.push_af()
    b.ld_a_c()
    b.or_a()
    b.jr_z('PRDEC_SK100')
    b.add_a_n(0x30)
    b.ld_e_a()
    b.push_bc()
    b.ld_c_n(2)
    b.call_addr(bdos_addr)
    b.pop_bc()
    b.ld_c_n(1)  # Flag that we printed something

    b.label('PRDEC_SK100')
    b.pop_af()
    b.push_bc()
    b.ld_b_n(0)  # Tens digit

    b.label('PRDEC_10LP')
    b.cp_n(10)
    b.jr_c('PRDEC_1')
    b.sub_n(10)
    b.inc_b()
    b.jr('PRDEC_10LP')

    b.label('PRDEC_1')
    # B has tens, A has units
    b.push_af()
    b.ld_a_b()
    b.pop_bc()  # Now B has units (from A), C has printed flag

    b.or_a()
    b.jr_nz('PRDEC_TENS')
    b.ld_a_c()
    b.or_a()
    b.jr_z('PRDEC_UNITS')

    b.label('PRDEC_TENS')
    b.push_bc()
    # A still has tens from earlier
    b.add_a_n(0x30)
    b.ld_e_a()
    b.ld_c_n(2)
    b.call_addr(bdos_addr)
    b.pop_bc()

    b.label('PRDEC_UNITS')
    b.ld_a_b()
    b.add_a_n(0x30)
    b.ld_e_a()
    b.ld_c_n(2)
    b.call_addr(bdos_addr)
    b.pop_bc()  # Restore original BC
    b.ret()
