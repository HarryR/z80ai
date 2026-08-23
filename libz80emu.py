"""
Pure-Python Z80 / eZ80 (ADL mode) CPU emulator.

Exists so the generated machine code can actually be executed in tests and
compared against the NumPy reference model in :mod:`libinfer`.  It has no
third-party dependencies, so it runs anywhere CI runs.

The interpreter decodes opcodes structurally (the classic ``x/y/z/p/q``
decomposition) rather than from a 256-entry jump table, which keeps the whole
base instruction set, the CB/ED/DD/FD prefixes and the eZ80 ADL variants in a
few hundred lines.

Cycle counts are derived from the M-cycle structure (4T opcode fetch, 3T per
memory read/write, plus explicit internal cycles) rather than a hand-entered
table.  They are accurate to within a couple of T-states for the instructions
the builders emit, which is enough to compare optimizations against each other.

Usage::

    cpu = Z80()
    cpu.load(0x0100, com_image)
    cpu.pc = 0x0100
    cpu.run(max_cycles=10_000_000)
"""

from __future__ import annotations

from collections.abc import Callable

# --- Flag bits ---------------------------------------------------------------

FLAG_S = 0x80
FLAG_Z = 0x40
FLAG_5 = 0x20
FLAG_H = 0x10
FLAG_3 = 0x08
FLAG_PV = 0x04
FLAG_N = 0x02
FLAG_C = 0x01


def _parity(v: int) -> int:
    p = 0
    while v:
        p ^= v & 1
        v >>= 1
    return FLAG_PV if p == 0 else 0


PARITY = [_parity(i) for i in range(256)]
SZ53 = [(i & (FLAG_S | FLAG_5 | FLAG_3)) | (FLAG_Z if i == 0 else 0) for i in range(256)]
SZ53P = [SZ53[i] | PARITY[i] for i in range(256)]


class Z80Error(RuntimeError):
    """Raised on an unimplemented opcode or a runaway program."""


class Z80:
    """A Z80 CPU. Set ``adl=True`` for eZ80 ADL (24-bit) mode."""

    def __init__(
        self,
        memory: bytearray | None = None,
        mem_size: int | None = None,
        adl: bool = False,
        io_read: Callable[[int], int] | None = None,
        io_write: Callable[[int, int], None] | None = None,
    ) -> None:
        self.adl = adl
        self.amask = 0xFFFFFF if adl else 0xFFFF
        if mem_size is None:
            mem_size = 0x1000000 if adl else 0x10000
        self.mem = bytearray(mem_size) if memory is None else memory
        self.io_read = io_read or (lambda port: 0xFF)
        self.io_write = io_write or (lambda port, val: None)

        self.a = self.f = 0
        self.b = self.c = self.d = self.e = self.h = self.l = 0
        self.a_ = self.f_ = 0
        self.b_ = self.c_ = self.d_ = self.e_ = self.h_ = self.l_ = 0
        # Upper (bits 16-23) bytes of BC/DE/HL. Always zero outside ADL mode,
        # because every write is masked with ``amask``.
        self.bu = self.du = self.hu = 0
        self.bu_ = self.du_ = self.hu_ = 0
        self.ix = self.iy = 0
        self.sp = self.amask
        self.pc = 0
        self.i = self.r = 0
        self.iff1 = self.iff2 = 0
        self.im = 0
        self.halted = False
        self.tstates = 0
        self.instructions = 0

        # addr -> callback(cpu). Return True to suppress the instruction at that
        # address (the callback is responsible for PC, usually via a RET).
        self.hooks: dict[int, Callable[[Z80], bool | None]] = {}
        # Word size for the instruction currently being decoded (eZ80 suffixes
        # override this per instruction).
        self._wsz = 3 if adl else 2

    # --- register pairs ------------------------------------------------------

    @property
    def bc(self) -> int:
        return (self.bu << 16) | (self.b << 8) | self.c

    @bc.setter
    def bc(self, v: int) -> None:
        self.bu, self.b, self.c = (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF

    @property
    def de(self) -> int:
        return (self.du << 16) | (self.d << 8) | self.e

    @de.setter
    def de(self, v: int) -> None:
        self.du, self.d, self.e = (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF

    @property
    def hl(self) -> int:
        return (self.hu << 16) | (self.h << 8) | self.l

    @hl.setter
    def hl(self, v: int) -> None:
        self.hu, self.h, self.l = (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF

    @property
    def af(self) -> int:
        return (self.a << 8) | self.f

    @af.setter
    def af(self, v: int) -> None:
        self.a, self.f = (v >> 8) & 0xFF, v & 0xFF

    # --- memory --------------------------------------------------------------

    def load(self, addr: int, data: bytes) -> None:
        self.mem[addr : addr + len(data)] = data

    def peek(self, addr: int) -> int:
        return self.mem[addr & self.amask]

    def poke(self, addr: int, val: int) -> None:
        self.mem[addr & self.amask] = val & 0xFF

    def peek_word(self, addr: int, size: int | None = None) -> int:
        size = size or self._wsz
        v = 0
        for k in range(size):
            v |= self.mem[(addr + k) & self.amask] << (8 * k)
        return v

    def _rb(self, addr: int) -> int:
        self.tstates += 3
        return self.mem[addr & self.amask]

    def _wb(self, addr: int, val: int) -> None:
        self.tstates += 3
        self.mem[addr & self.amask] = val & 0xFF

    def _rw(self, addr: int, size: int | None = None) -> int:
        size = size or self._wsz
        v = 0
        for k in range(size):
            v |= self._rb(addr + k) << (8 * k)
        return v

    def _ww(self, addr: int, val: int, size: int | None = None) -> None:
        size = size or self._wsz
        for k in range(size):
            self._wb(addr + k, (val >> (8 * k)) & 0xFF)

    def _fetch(self) -> int:
        self.tstates += 4
        b = self.mem[self.pc & self.amask]
        self.pc = (self.pc + 1) & self.amask
        self.r = (self.r & 0x80) | ((self.r + 1) & 0x7F)
        return b

    def _imm8(self) -> int:
        self.tstates += 3
        b = self.mem[self.pc & self.amask]
        self.pc = (self.pc + 1) & self.amask
        return b

    def _imm_signed(self) -> int:
        b = self._imm8()
        return b - 256 if b > 127 else b

    def _imm_word(self, size: int | None = None) -> int:
        size = size or self._wsz
        v = 0
        for k in range(size):
            v |= self._imm8() << (8 * k)
        return v

    def _push(self, val: int) -> None:
        self.sp = (self.sp - self._wsz) & self.amask
        self._ww(self.sp, val)

    def _pop(self) -> int:
        v = self._rw(self.sp)
        self.sp = (self.sp + self._wsz) & self.amask
        return v

    # --- 8-bit register file access -----------------------------------------

    _R_NAMES = ("b", "c", "d", "e", "h", "l", None, "a")

    def _idx_addr(self, idx: str | None, disp: int) -> int:
        if idx is None:
            return self.hl
        base = self.ix if idx == "ix" else self.iy
        return (base + disp) & self.amask

    def _get_r(self, i: int, idx: str | None = None, disp: int = 0) -> int:
        if i == 6:
            return self._rb(self._idx_addr(idx, disp))
        if idx is not None and i in (4, 5):
            base = self.ix if idx == "ix" else self.iy
            return (base >> 8) & 0xFF if i == 4 else base & 0xFF
        return getattr(self, self._R_NAMES[i])

    def _set_r(self, i: int, val: int, idx: str | None = None, disp: int = 0) -> None:
        val &= 0xFF
        if i == 6:
            self._wb(self._idx_addr(idx, disp), val)
            return
        if idx is not None and i in (4, 5):
            base = self.ix if idx == "ix" else self.iy
            if i == 4:
                base = (base & ~0xFF00 & self.amask) | (val << 8)
            else:
                base = (base & ~0xFF & self.amask) | val
            if idx == "ix":
                self.ix = base
            else:
                self.iy = base
            return
        setattr(self, self._R_NAMES[i], val)

    def _get_rp(self, p: int, idx: str | None, use_af: bool = False) -> int:
        if p == 0:
            return self.bc
        if p == 1:
            return self.de
        if p == 2:
            if idx == "ix":
                return self.ix
            if idx == "iy":
                return self.iy
            return self.hl
        return self.af if use_af else self.sp

    def _set_rp(self, p: int, val: int, idx: str | None, use_af: bool = False) -> None:
        val &= self.amask
        if p == 0:
            self.bc = val
        elif p == 1:
            self.de = val
        elif p == 2:
            if idx == "ix":
                self.ix = val
            elif idx == "iy":
                self.iy = val
            else:
                self.hl = val
        elif use_af:
            self.af = val & 0xFFFF
        else:
            self.sp = val

    # --- condition codes -----------------------------------------------------

    def _cond(self, y: int) -> bool:
        f = self.f
        if y == 0:
            return not (f & FLAG_Z)
        if y == 1:
            return bool(f & FLAG_Z)
        if y == 2:
            return not (f & FLAG_C)
        if y == 3:
            return bool(f & FLAG_C)
        if y == 4:
            return not (f & FLAG_PV)
        if y == 5:
            return bool(f & FLAG_PV)
        if y == 6:
            return not (f & FLAG_S)
        return bool(f & FLAG_S)

    # --- ALU -----------------------------------------------------------------

    def _add_a(self, v: int, carry: int = 0) -> None:
        a = self.a
        res = a + v + carry
        r8 = res & 0xFF
        self.f = (
            SZ53[r8]
            | (FLAG_C if res > 0xFF else 0)
            | (FLAG_H if ((a & 0x0F) + (v & 0x0F) + carry) > 0x0F else 0)
            | (FLAG_PV if (~(a ^ v) & (a ^ r8) & 0x80) else 0)
        )
        self.a = r8

    def _sub_a(self, v: int, carry: int = 0, store: bool = True) -> None:
        a = self.a
        res = a - v - carry
        r8 = res & 0xFF
        f = (
            SZ53[r8]
            | FLAG_N
            | (FLAG_C if res < 0 else 0)
            | (FLAG_H if ((a & 0x0F) - (v & 0x0F) - carry) < 0 else 0)
            | (FLAG_PV if ((a ^ v) & (a ^ r8) & 0x80) else 0)
        )
        if store:
            self.a = r8
            self.f = f
        else:  # CP: undocumented 5/3 flags come from the operand
            self.f = (f & ~(FLAG_5 | FLAG_3)) | (v & (FLAG_5 | FLAG_3))

    def _and_a(self, v: int) -> None:
        self.a &= v
        self.f = SZ53P[self.a] | FLAG_H

    def _xor_a(self, v: int) -> None:
        self.a ^= v
        self.f = SZ53P[self.a]

    def _or_a(self, v: int) -> None:
        self.a |= v
        self.f = SZ53P[self.a]

    def _alu(self, op: int, v: int) -> None:
        if op == 0:
            self._add_a(v)
        elif op == 1:
            self._add_a(v, self.f & FLAG_C)
        elif op == 2:
            self._sub_a(v)
        elif op == 3:
            self._sub_a(v, self.f & FLAG_C)
        elif op == 4:
            self._and_a(v)
        elif op == 5:
            self._xor_a(v)
        elif op == 6:
            self._or_a(v)
        else:
            self._sub_a(v, 0, store=False)

    def _inc8(self, v: int) -> int:
        r = (v + 1) & 0xFF
        self.f = (
            (self.f & FLAG_C)
            | SZ53[r]
            | (FLAG_H if (r & 0x0F) == 0 else 0)
            | (FLAG_PV if r == 0x80 else 0)
        )
        return r

    def _dec8(self, v: int) -> int:
        r = (v - 1) & 0xFF
        self.f = (
            (self.f & FLAG_C)
            | FLAG_N
            | SZ53[r]
            | (FLAG_H if (r & 0x0F) == 0x0F else 0)
            | (FLAG_PV if r == 0x7F else 0)
        )
        return r

    def _add16(self, a: int, b: int) -> int:
        width = 24 if self.adl else 16
        res = a + b
        r = res & self.amask
        half = (a & ((1 << (width - 4)) - 1)) + (b & ((1 << (width - 4)) - 1))
        self.f = (
            (self.f & (FLAG_S | FLAG_Z | FLAG_PV))
            | (FLAG_C if res > self.amask else 0)
            | (FLAG_H if half > ((1 << (width - 4)) - 1) else 0)
            | ((r >> (width - 8)) & (FLAG_5 | FLAG_3))
        )
        return r

    def _adc16(self, a: int, b: int) -> int:
        width = 24 if self.adl else 16
        top = 1 << (width - 1)
        carry = self.f & FLAG_C
        res = a + b + carry
        r = res & self.amask
        half = (a & ((1 << (width - 4)) - 1)) + (b & ((1 << (width - 4)) - 1)) + carry
        self.f = (
            (FLAG_S if r & top else 0)
            | (FLAG_Z if r == 0 else 0)
            | ((r >> (width - 8)) & (FLAG_5 | FLAG_3))
            | (FLAG_C if res > self.amask else 0)
            | (FLAG_H if half > ((1 << (width - 4)) - 1) else 0)
            | (FLAG_PV if (~(a ^ b) & (a ^ r) & top) else 0)
        )
        return r

    def _sbc16(self, a: int, b: int) -> int:
        width = 24 if self.adl else 16
        top = 1 << (width - 1)
        carry = self.f & FLAG_C
        res = a - b - carry
        r = res & self.amask
        half = (a & ((1 << (width - 4)) - 1)) - (b & ((1 << (width - 4)) - 1)) - carry
        self.f = (
            (FLAG_S if r & top else 0)
            | (FLAG_Z if r == 0 else 0)
            | ((r >> (width - 8)) & (FLAG_5 | FLAG_3))
            | FLAG_N
            | (FLAG_C if res < 0 else 0)
            | (FLAG_H if half < 0 else 0)
            | (FLAG_PV if ((a ^ b) & (a ^ r) & top) else 0)
        )
        return r

    # --- rotates / shifts ----------------------------------------------------

    def _rot(self, op: int, v: int) -> int:
        c = self.f & FLAG_C
        if op == 0:  # RLC
            r = ((v << 1) | (v >> 7)) & 0xFF
            cf = v >> 7
        elif op == 1:  # RRC
            r = ((v >> 1) | (v << 7)) & 0xFF
            cf = v & 1
        elif op == 2:  # RL
            r = ((v << 1) | c) & 0xFF
            cf = v >> 7
        elif op == 3:  # RR
            r = ((v >> 1) | (c << 7)) & 0xFF
            cf = v & 1
        elif op == 4:  # SLA
            r = (v << 1) & 0xFF
            cf = v >> 7
        elif op == 5:  # SRA
            r = ((v >> 1) | (v & 0x80)) & 0xFF
            cf = v & 1
        elif op == 6:  # SLL (undocumented)
            r = ((v << 1) | 1) & 0xFF
            cf = v >> 7
        else:  # SRL
            r = (v >> 1) & 0xFF
            cf = v & 1
        self.f = SZ53P[r] | cf
        return r

    def _daa(self) -> None:
        a = self.a
        adjust = 0
        carry = self.f & FLAG_C
        if (self.f & FLAG_H) or (a & 0x0F) > 9:
            adjust |= 0x06
        if carry or a > 0x99:
            adjust |= 0x60
            carry = FLAG_C
        if self.f & FLAG_N:
            r = (a - adjust) & 0xFF
            h = FLAG_H if ((a & 0x0F) - (adjust & 0x0F)) < 0 else 0
        else:
            r = (a + adjust) & 0xFF
            h = FLAG_H if ((a & 0x0F) + (adjust & 0x0F)) > 0x0F else 0
        self.a = r
        self.f = SZ53P[r] | h | carry | (self.f & FLAG_N)

    # --- main loop -----------------------------------------------------------

    def step(self) -> None:
        hook = self.hooks.get(self.pc & self.amask)
        if hook is not None and hook(self):
            self.instructions += 1
            return
        if self.halted:
            self.tstates += 4
            return
        self._wsz = 3 if self.adl else 2
        self.instructions += 1
        self._execute(self._fetch(), None)

    def run(self, max_cycles: int = 200_000_000, stop_pc: int | None = None) -> None:
        """Run until HALT, ``stop_pc``, or the cycle budget is exhausted."""
        limit = self.tstates + max_cycles
        while self.tstates < limit:
            if stop_pc is not None and (self.pc & self.amask) == stop_pc:
                return
            self.step()
            if self.halted:
                return
        raise Z80Error(f"cycle budget exhausted at PC={self.pc:06X}")

    # --- decoder -------------------------------------------------------------

    def _ez80_idx_rp(self, op: int, idx: str) -> None:
        """LD rr,(IX+d) / LD (IX+d),rr for rr in BC, DE, HL, IX/IY."""
        disp = self._imm_signed()
        addr = ((self.ix if idx == "ix" else self.iy) + disp) & self.amask
        pair = op >> 4  # 0=BC 1=DE 2=HL 3=the index register itself
        store = bool(op & 0x08)
        if store:
            if pair == 3:
                val = self.ix if idx == "ix" else self.iy
            else:
                val = (self.bc, self.de, self.hl)[pair]
            self._ww(addr, val)
            return
        val = self._rw(addr)
        if pair == 0:
            self.bc = val
        elif pair == 1:
            self.de = val
        elif pair == 2:
            self.hl = val
        elif idx == "ix":
            self.ix = val
        else:
            self.iy = val

    def _execute(self, op: int, idx: str | None) -> None:
        # eZ80 instruction-mode suffixes (.SIS/.LIS/.SIL/.LIL)
        if self.adl and idx is None and op in (0x40, 0x49, 0x52, 0x5B):
            self._wsz = 3 if op in (0x49, 0x5B) else 2
            self._execute(self._fetch(), None)
            return

        if op == 0xDD:
            self._execute(self._fetch(), "ix")
            return
        if op == 0xFD:
            self._execute(self._fetch(), "iy")
            return
        if op == 0xED:
            self._execute_ed(self._fetch())
            return
        if op == 0xCB:
            self._execute_cb(idx)
            return

        # eZ80 register-pair indexed loads: LD rr,(IX+d) and LD (IX+d),rr for
        # BC/DE/HL/IX, moving a whole 24-bit word in one instruction. These sit
        # at 07/0F/17/1F/27/2F/37/3F under a DD or FD prefix, where a plain Z80
        # would see RLCA/RRCA/... and discard the prefix - so they exist only in
        # ADL mode. Encodings per https://mdfs.net/Docs/Comp/eZ80/eZ80OpList.
        if self.adl and idx is not None and (op & 0x07) == 0x07 and op < 0x40:
            self._ez80_idx_rp(op, idx)
            return

        x, y, z = op >> 6, (op >> 3) & 7, op & 7
        p, q = y >> 1, y & 1

        # (IX+d) displacement, fetched once up front where an operand needs it.
        disp = 0
        if idx is not None:
            if x == 0:
                mem_op = z in (4, 5, 6) and y == 6
            elif x == 1:
                mem_op = (y == 6) != (z == 6)  # y == z == 6 is HALT
            elif x == 2:
                mem_op = z == 6
            else:
                mem_op = False
            if mem_op:
                disp = self._imm_signed()
                self.tstates += 5

        if x == 0:
            self._x0(y, z, p, q, idx, disp)
        elif x == 1:
            if y == 6 and z == 6:
                self.halted = True
            elif y == 6:
                self._wb(self._idx_addr(idx, disp), self._get_r(z, None))
            elif z == 6:
                self._set_r(y, self._rb(self._idx_addr(idx, disp)), None)
            else:
                self._set_r(y, self._get_r(z, idx), idx)
        elif x == 2:
            self._alu(y, self._get_r(z, idx, disp))
        else:
            self._x3(y, z, p, q, idx)

    def _x0(self, y: int, z: int, p: int, q: int, idx: str | None, disp: int) -> None:
        if z == 0:
            if y == 0:
                return  # NOP
            if y == 1:  # EX AF,AF'
                self.a, self.a_ = self.a_, self.a
                self.f, self.f_ = self.f_, self.f
                return
            if y == 2:  # DJNZ
                d = self._imm_signed()
                self.tstates += 1
                self.b = (self.b - 1) & 0xFF
                if self.b:
                    self.tstates += 5
                    self.pc = (self.pc + d) & self.amask
                return
            if y == 3:  # JR d
                d = self._imm_signed()
                self.tstates += 5
                self.pc = (self.pc + d) & self.amask
                return
            d = self._imm_signed()  # JR cc,d
            if self._cond(y - 4):
                self.tstates += 5
                self.pc = (self.pc + d) & self.amask
            return

        if z == 1:
            if q == 0:
                self._set_rp(p, self._imm_word(), idx)
            else:  # ADD HL/IX/IY, rp
                self.tstates += 7
                acc = self._get_rp(2, idx)
                self._set_rp(2, self._add16(acc, self._get_rp(p, idx)), idx)
            return

        if z == 2:
            if q == 0:
                if p == 0:
                    self._wb(self.bc, self.a)
                elif p == 1:
                    self._wb(self.de, self.a)
                elif p == 2:
                    self._ww(self._imm_word(), self._get_rp(2, idx))
                else:
                    self._wb(self._imm_word(), self.a)
            else:
                if p == 0:
                    self.a = self._rb(self.bc)
                elif p == 1:
                    self.a = self._rb(self.de)
                elif p == 2:
                    self._set_rp(2, self._rw(self._imm_word()), idx)
                else:
                    self.a = self._rb(self._imm_word())
            return

        if z == 3:
            self.tstates += 2
            v = self._get_rp(p, idx)
            self._set_rp(p, (v + (1 if q == 0 else -1)) & self.amask, idx)
            return

        if z == 4:
            if y == 6:
                addr = self._idx_addr(idx, disp)
                self.tstates += 1
                self._wb(addr, self._inc8(self._rb(addr)))
            else:
                self._set_r(y, self._inc8(self._get_r(y, idx)), idx)
            return

        if z == 5:
            if y == 6:
                addr = self._idx_addr(idx, disp)
                self.tstates += 1
                self._wb(addr, self._dec8(self._rb(addr)))
            else:
                self._set_r(y, self._dec8(self._get_r(y, idx)), idx)
            return

        if z == 6:
            n = self._imm8()
            if y == 6:
                self._wb(self._idx_addr(idx, disp), n)
            else:
                self._set_r(y, n, idx)
            return

        # z == 7: accumulator / flag ops
        if y == 0:  # RLCA
            self.a = ((self.a << 1) | (self.a >> 7)) & 0xFF
            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV)) | (self.a & (FLAG_5 | FLAG_3 | FLAG_C))
        elif y == 1:  # RRCA
            cf = self.a & 1
            self.a = ((self.a >> 1) | (self.a << 7)) & 0xFF
            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV)) | (self.a & (FLAG_5 | FLAG_3)) | cf
        elif y == 2:  # RLA
            cf = self.a >> 7
            self.a = ((self.a << 1) | (self.f & FLAG_C)) & 0xFF
            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV)) | (self.a & (FLAG_5 | FLAG_3)) | cf
        elif y == 3:  # RRA
            cf = self.a & 1
            self.a = ((self.a >> 1) | ((self.f & FLAG_C) << 7)) & 0xFF
            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV)) | (self.a & (FLAG_5 | FLAG_3)) | cf
        elif y == 4:
            self._daa()
        elif y == 5:  # CPL
            self.a ^= 0xFF
            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV | FLAG_C)) | FLAG_H | FLAG_N | (
                self.a & (FLAG_5 | FLAG_3)
            )
        elif y == 6:  # SCF
            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV)) | (self.a & (FLAG_5 | FLAG_3)) | FLAG_C
        else:  # CCF
            c = self.f & FLAG_C
            self.f = (
                (self.f & (FLAG_S | FLAG_Z | FLAG_PV))
                | (self.a & (FLAG_5 | FLAG_3))
                | (FLAG_H if c else 0)
                | (0 if c else FLAG_C)
            )

    def _x3(self, y: int, z: int, p: int, q: int, idx: str | None) -> None:
        if z == 0:  # RET cc
            self.tstates += 1
            if self._cond(y):
                self.pc = self._pop()
            return

        if z == 1:
            if q == 0:
                self._set_rp(p, self._pop(), idx, use_af=True)
            elif p == 0:
                self.pc = self._pop()
            elif p == 1:  # EXX
                self.b, self.b_ = self.b_, self.b
                self.c, self.c_ = self.c_, self.c
                self.d, self.d_ = self.d_, self.d
                self.e, self.e_ = self.e_, self.e
                self.h, self.h_ = self.h_, self.h
                self.l, self.l_ = self.l_, self.l
                self.bu, self.bu_ = self.bu_, self.bu
                self.du, self.du_ = self.du_, self.du
                self.hu, self.hu_ = self.hu_, self.hu
            elif p == 2:  # JP (HL)
                self.pc = self._get_rp(2, idx)
            else:  # LD SP,HL
                self.tstates += 2
                self.sp = self._get_rp(2, idx)
            return

        if z == 2:  # JP cc,nn
            addr = self._imm_word()
            if self._cond(y):
                self.pc = addr
            return

        if z == 3:
            if y == 0:
                self.pc = self._imm_word()
            elif y == 2:  # OUT (n),A
                self.tstates += 4
                self.io_write(self._imm8() | (self.a << 8), self.a)
            elif y == 3:  # IN A,(n)
                self.tstates += 4
                self.a = self.io_read(self._imm8() | (self.a << 8)) & 0xFF
            elif y == 4:  # EX (SP),HL
                self.tstates += 3
                v = self._rw(self.sp)
                self._ww(self.sp, self._get_rp(2, idx))
                self._set_rp(2, v, idx)
            elif y == 5:  # EX DE,HL
                self.d, self.h = self.h, self.d
                self.e, self.l = self.l, self.e
                self.du, self.hu = self.hu, self.du
            elif y == 6:
                self.iff1 = self.iff2 = 0
            else:
                self.iff1 = self.iff2 = 1
            return

        if z == 4:  # CALL cc,nn
            addr = self._imm_word()
            if self._cond(y):
                self.tstates += 1
                self._push(self.pc)
                self.pc = addr
            return

        if z == 5:
            if q == 0:
                self.tstates += 1
                self._push(self._get_rp(p, idx, use_af=True))
            else:  # CALL nn (p == 0; DD/ED/FD were consumed by _execute)
                addr = self._imm_word()
                self.tstates += 1
                self._push(self.pc)
                self.pc = addr
            return

        if z == 6:
            self._alu(y, self._imm8())
            return

        self.tstates += 1  # RST
        self._push(self.pc)
        self.pc = y * 8

    def _execute_cb(self, idx: str | None) -> None:
        if idx is None:
            op = self._fetch()
            disp = 0
        else:
            disp = self._imm_signed()
            op = self._imm8()
            self.tstates += 3
        x, y, z = op >> 6, (op >> 3) & 7, op & 7

        src = self._get_r(z) if idx is None else self._rb(self._idx_addr(idx, disp))

        if x == 0:
            r = self._rot(y, src)
        elif x == 1:  # BIT
            r = src & (1 << y)
            self.f = (
                (self.f & FLAG_C)
                | FLAG_H
                | (0 if r else (FLAG_Z | FLAG_PV))
                | (FLAG_S if (y == 7 and r) else 0)
                | ((src if idx is None else (self._idx_addr(idx, disp) >> 8)) & (FLAG_5 | FLAG_3))
            )
            return
        elif x == 2:
            r = src & ~(1 << y) & 0xFF
        else:
            r = src | (1 << y)

        if idx is None:
            self._set_r(z, r)
        else:
            self._wb(self._idx_addr(idx, disp), r)
            if z != 6:  # undocumented: also copy to the named register
                self._set_r(z, r)

    def _execute_ed(self, op: int) -> None:
        # eZ80 extensions first: MLT reuses ED 4C/5C/6C/7C, which on the Z80 are
        # undocumented aliases of NEG and would otherwise swallow them.
        if self.adl and self._ez80_ed(op):
            return

        x, y, z = op >> 6, (op >> 3) & 7, op & 7
        p, q = y >> 1, y & 1

        if x == 1:
            if z == 0:  # IN r,(C)
                self.tstates += 4
                v = self.io_read(self.bc) & 0xFF
                if y != 6:
                    self._set_r(y, v)
                self.f = (self.f & FLAG_C) | SZ53P[v]
                return
            if z == 1:  # OUT (C),r
                self.tstates += 4
                self.io_write(self.bc, 0 if y == 6 else self._get_r(y))
                return
            if z == 2:
                self.tstates += 7
                rp = self._get_rp(p, None)
                self.hl = self._sbc16(self.hl, rp) if q == 0 else self._adc16(self.hl, rp)
                return
            if z == 3:
                addr = self._imm_word()
                if q == 0:
                    self._ww(addr, self._get_rp(p, None))
                else:
                    self._set_rp(p, self._rw(addr), None)
                return
            if z == 4:  # NEG
                v = self.a
                self.a = 0
                self._sub_a(v)
                return
            if z == 5:  # RETN / RETI
                self.iff1 = self.iff2
                self.pc = self._pop()
                return
            if z == 6:  # IM
                self.im = (0, 0, 1, 2, 0, 0, 1, 2)[y]
                return
            # z == 7
            if y == 0:
                self.tstates += 1
                self.i = self.a
            elif y == 1:
                self.tstates += 1
                self.r = self.a
            elif y == 2:
                self.tstates += 1
                self.a = self.i
                self.f = (self.f & FLAG_C) | SZ53[self.a] | (FLAG_PV if self.iff2 else 0)
            elif y == 3:
                self.tstates += 1
                self.a = self.r
                self.f = (self.f & FLAG_C) | SZ53[self.a] | (FLAG_PV if self.iff2 else 0)
            elif y == 4:  # RRD
                self.tstates += 4
                m = self._rb(self.hl)
                self._wb(self.hl, ((m >> 4) | (self.a << 4)) & 0xFF)
                self.a = (self.a & 0xF0) | (m & 0x0F)
                self.f = (self.f & FLAG_C) | SZ53P[self.a]
            elif y == 5:  # RLD
                self.tstates += 4
                m = self._rb(self.hl)
                self._wb(self.hl, ((m << 4) | (self.a & 0x0F)) & 0xFF)
                self.a = (self.a & 0xF0) | (m >> 4)
                self.f = (self.f & FLAG_C) | SZ53P[self.a]
            return

        if x == 2 and z <= 3 and y >= 4:
            self._block(y, z)
            return

        raise Z80Error(f"unimplemented ED opcode {op:02X} at PC={self.pc - 2:06X}")

    def _ez80_ed(self, op: int) -> bool:
        # MLT rr : 0x4C/0x5C/0x6C/0x7C  (BC/DE/HL/SP := high * low)
        if op in (0x4C, 0x5C, 0x6C, 0x7C):
            self.tstates += 17
            p = (op >> 4) - 4
            v = self._get_rp(p, None) & 0xFFFF
            self._set_rp(p, ((v >> 8) * (v & 0xFF)) & 0xFFFF, None)
            return True
        # LEA/PEA are not emitted by our backend; fail loudly rather than guess.
        return False

    def _block(self, y: int, z: int) -> None:
        # y: 4=I 5=D 6=IR 7=DR
        rep = y >= 6
        delta = -1 if (y & 1) else 1
        if z == 0:  # LDI/LDD/LDIR/LDDR
            self.tstates += 2
            v = self._rb(self.hl)
            self._wb(self.de, v)
            self.hl = (self.hl + delta) & self.amask
            self.de = (self.de + delta) & self.amask
            self.bc = (self.bc - 1) & self.amask
            n = (v + self.a) & 0xFF
            self.f = (
                (self.f & (FLAG_S | FLAG_Z | FLAG_C))
                | (FLAG_PV if self.bc else 0)
                | (n & FLAG_3)
                | (FLAG_5 if n & 0x02 else 0)
            )
            if rep and self.bc:
                self.tstates += 5
                self.pc = (self.pc - 2) & self.amask
            return
        if z == 1:  # CPI/CPD/CPIR/CPDR
            self.tstates += 5
            v = self._rb(self.hl)
            carry = self.f & FLAG_C
            res = (self.a - v) & 0xFF
            half = FLAG_H if ((self.a & 0x0F) - (v & 0x0F)) < 0 else 0
            self.hl = (self.hl + delta) & self.amask
            self.bc = (self.bc - 1) & self.amask
            n = (res - (1 if half else 0)) & 0xFF
            self.f = (
                carry
                | FLAG_N
                | half
                | (FLAG_Z if res == 0 else 0)
                | (res & FLAG_S)
                | (FLAG_PV if self.bc else 0)
                | (n & FLAG_3)
                | (FLAG_5 if n & 0x02 else 0)
            )
            if rep and self.bc and res:
                self.tstates += 5
                self.pc = (self.pc - 2) & self.amask
            return
        if z == 2:  # INI/IND/INIR/INDR
            self.tstates += 5
            v = self.io_read(self.bc) & 0xFF
            self._wb(self.hl, v)
            self.b = (self.b - 1) & 0xFF
            self.hl = (self.hl + delta) & self.amask
            self.f = SZ53[self.b] | FLAG_N
            if rep and self.b:
                self.tstates += 5
                self.pc = (self.pc - 2) & self.amask
            return
        # z == 3: OUTI/OUTD/OTIR/OTDR
        self.tstates += 5
        v = self._rb(self.hl)
        self.b = (self.b - 1) & 0xFF
        self.io_write(self.bc, v)
        self.hl = (self.hl + delta) & self.amask
        self.f = SZ53[self.b] | FLAG_N
        if rep and self.b:
            self.tstates += 5
            self.pc = (self.pc - 2) & self.amask

    # --- introspection -------------------------------------------------------

    def state(self) -> dict[str, int]:
        return {
            "af": self.af, "bc": self.bc, "de": self.de, "hl": self.hl,
            "ix": self.ix, "iy": self.iy, "sp": self.sp, "pc": self.pc,
            "tstates": self.tstates,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Z80 PC={self.pc:04X} AF={self.af:04X} BC={self.bc:04X} "
            f"DE={self.de:04X} HL={self.hl:04X} SP={self.sp:04X} T={self.tstates}>"
        )
