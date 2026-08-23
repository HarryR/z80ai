"""
eZ80 (ADL mode) extensions to :class:`libz80.Z80Builder`.

In ADL mode every address and every ``LD rr,nn`` immediate is 24 bits wide, and
BC/DE/HL/IX/IY/SP are 24-bit registers.  Setting ``addr_size = 3`` makes the
inherited emitters produce ADL-correct encodings, so most of the Z80 helper
methods work unchanged; this module adds the handful of instructions the eZ80
backend needs on top.

Data stays explicit: :meth:`Z80Builder.emit_word` and ``dw`` remain 16-bit,
``d24``/``ds24`` emit 24-bit values, which is what layer activations and
pointers use.
"""

from __future__ import annotations

from libz80 import Z80Builder, _disp

# Agon MOS loads .bin programs here and enters at the first byte.
AGON_LOAD_ADDR = 0x040000

# ADL mode addresses 16MB, but that is address space, not memory.  A shipping
# Agon has 512KB of SRAM at 0x040000-0x0BFFFF and MOS puts the stack at the top
# of it, so this - not 16MB - is what actually bounds a model's size.
AGON_SRAM_TOP = 0x0C0000
AGON_STACK_MARGIN = 0x1000

#: Largest image that will load and still leave MOS its stack.
AGON_MAX_IMAGE = AGON_SRAM_TOP - AGON_STACK_MARGIN - AGON_LOAD_ADDR


class EZ80Builder(Z80Builder):
    """Emits eZ80 machine code for ADL (24-bit) mode."""

    addr_size = 3

    def __init__(self, org: int = AGON_LOAD_ADDR) -> None:
        super().__init__(org=org)
        #: Which layer kernel produced this image; set by buildez80.build_autoreg.
        self.kernel: str | None = None

    # --- 24-bit data ---------------------------------------------------------

    def d24(self, *vals: int) -> None:
        for v in vals:
            self.emit(v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF)

    def ds24(self, count: int) -> None:
        self.ds(count * 3)

    # --- loads the base class doesn't cover ---------------------------------

    def ld_de_mem_label(self, label: str, addend: int = 0) -> None:
        self.emit(0xED, 0x5B)
        self.fixup_word(label, addend)

    def ld_ix_mem_label(self, label: str, addend: int = 0) -> None:
        self.emit(0xDD, 0x2A)
        self.fixup_word(label, addend)

    def ld_ixd_a(self, d: int) -> None:
        self.emit(0xDD, 0x77, d & 0xFF)

    # --- eZ80 register-pair indexed loads ------------------------------------
    #
    # The eZ80 adds LD rr,(IX+d) and LD (IX+d),rr for BC/DE/HL, which move a
    # whole 24-bit word in one instruction.  Opcodes 07/0F/17/1F/27/2F under a
    # DD or FD prefix; on a plain Z80 those same encodings are RLCA/RRCA/... with
    # the prefix ignored, so they only mean this in ADL mode.
    #
    # Encodings per the eZ80 opcode list at
    # https://mdfs.net/Docs/Comp/eZ80/eZ80OpList (Zilog UM0077 has the same
    # table).  The column-major kernel uses the HL forms as its accumulator
    # read-modify-write.

    def ld_hl_ixd(self, d: int) -> None:
        """LD HL,(IX+d) - load 24 bits."""
        self.emit(0xDD, 0x27, _disp(d))

    def ld_ixd_hl(self, d: int) -> None:
        """LD (IX+d),HL - store 24 bits."""
        self.emit(0xDD, 0x2F, _disp(d))

    def ld_hl_iyd(self, d: int) -> None:
        """LD HL,(IY+d) - load 24 bits."""
        self.emit(0xFD, 0x27, _disp(d))

    def ld_iyd_hl(self, d: int) -> None:
        """LD (IY+d),HL - store 24 bits."""
        self.emit(0xFD, 0x2F, _disp(d))

    def ld_de_ixd(self, d: int) -> None:
        """LD DE,(IX+d) - load 24 bits."""
        self.emit(0xDD, 0x17, _disp(d))

    def ld_iyd_bc(self, d: int) -> None:
        """LD (IY+d),BC - store 24 bits."""
        self.emit(0xFD, 0x0F, _disp(d))

    def ld_bc_iyd(self, d: int) -> None:
        """LD BC,(IY+d) - load 24 bits."""
        self.emit(0xFD, 0x07, _disp(d))

    def ld_ixd_de(self, d: int) -> None:
        """LD (IX+d),DE - store 24 bits."""
        self.emit(0xDD, 0x1F, _disp(d))

    # --- shifts on memory ----------------------------------------------------

    def sra_hl_ind(self) -> None:
        """SRA (HL) - arithmetic shift right of the byte at (HL)."""
        self.emit(0xCB, 0x2E)

    def rr_hl_ind(self) -> None:
        """RR (HL) - rotate right through carry of the byte at (HL)."""
        self.emit(0xCB, 0x1E)

    # --- flow ----------------------------------------------------------------

    def jp_p(self, label: str) -> None:
        """JP P,nn - jump if the sign flag is clear."""
        self.emit(0xF2)
        self.fixup_word(label)

    # --- eZ80-only -----------------------------------------------------------
    #
    # MLT is not used by the {-2,-1,0,+1} kernel - add and subtract are cheaper
    # than a multiply for those - but it is the instruction a wider weight
    # format would be built on, so it is exposed and covered by tests.

    def mlt_hl(self) -> None:
        """MLT HL - HL = H * L (unsigned 8x8 -> 16)."""
        self.emit(0xED, 0x6C)

    def mlt_de(self) -> None:
        self.emit(0xED, 0x5C)

    def mlt_bc(self) -> None:
        self.emit(0xED, 0x4C)


def agon_header(b: EZ80Builder, entry_label: str = "START") -> None:
    """Emit the Agon MOS program header.

    MOS jumps to the load address, so byte 0 is a jump to the real entry point.
    At offset 0x40 it expects the magic "MOS", a header version and the ADL
    flag; without it MOS refuses to run the binary in 24-bit mode.
    """
    b.jp(entry_label)
    while len(b.code) < 0x40:
        b.emit(0x00)
    b.ascii("MOS")
    b.db(0x00)  # header version
    b.db(0x01)  # 1 = ADL mode
