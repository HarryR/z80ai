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

from libz80 import Z80Builder

# Agon MOS loads .bin programs here and enters at the first byte.
AGON_LOAD_ADDR = 0x040000


class EZ80Builder(Z80Builder):
    """Emits eZ80 machine code for ADL (24-bit) mode."""

    addr_size = 3

    def __init__(self, org: int = AGON_LOAD_ADDR):
        super().__init__(org=org)

    # --- 24-bit data ---------------------------------------------------------

    def d24(self, *vals: int) -> None:
        for v in vals:
            self.emit(v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF)

    def ds24(self, count: int) -> None:
        self.ds(count * 3)

    # --- loads the base class doesn't cover ---------------------------------

    def ld_de_mem_label(self, label: str) -> None:
        self.emit(0xED, 0x5B)
        self.fixup_word(label)

    def ld_ix_mem_label(self, label: str) -> None:
        self.emit(0xDD, 0x2A)
        self.fixup_word(label)

    def ld_ixd_a(self, d: int) -> None:
        self.emit(0xDD, 0x77, d & 0xFF)

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
