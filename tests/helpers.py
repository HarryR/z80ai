"""Helpers for reading generated kernels out of emulator memory.

Comparing generated text is a weak signal: with few output classes two
different logit vectors often argmax to the same character.  The tests that
matter run a build up to a label and compare the buffers themselves, which is
what these helpers exist for.
"""

from __future__ import annotations

import numpy as np

from libez80 import AGON_LOAD_ADDR
from libhost import AgonHost, CPMHost
from libz80emu import Z80

CPM_ORG = 0x0100
MAX_CYCLES = 400_000_000


def _signed(v: int, width: int) -> int:
    top = 1 << (8 * width)
    return v - top if v & (top >> 1) else v


def read_words(cpu: Z80, addr: int, count: int) -> np.ndarray:
    """Read ``count`` 16-bit signed activations, as the Z80 targets store them."""
    return np.array(
        [
            _signed(cpu.peek(addr + 2 * i) | (cpu.peek(addr + 2 * i + 1) << 8), 2)
            for i in range(count)
        ],
        dtype=np.int64,
    )


def read24(cpu: Z80, addr: int, count: int) -> np.ndarray:
    """Read ``count`` 24-bit signed activations, as the eZ80 target stores them."""
    return np.array(
        [
            _signed(
                sum(cpu.peek(addr + 3 * i + k) << (8 * k) for k in range(3)), 3
            )
            for i in range(count)
        ],
        dtype=np.int64,
    )


def run_cpm_until(builder, query: str, label: str) -> Z80:
    """Run a freshly built .COM up to ``label`` with ``query`` on the cmdline."""
    host = CPMHost(cmdline=query)
    cpu = host.cpu
    cpu.load(CPM_ORG, builder.build())
    cpu.pc = CPM_ORG
    cpu.run(max_cycles=MAX_CYCLES, stop_pc=builder.labels[label])
    assert cpu.pc == builder.labels[label], "never reached " + label
    return cpu


def run_ez80_until(builder, query: str, label: str) -> Z80:
    """Run a freshly built Agon .bin up to ``label``, typing ``query`` at the prompt."""
    host = AgonHost(stdin=[query, "!"])
    cpu = host.cpu
    cpu.load(AGON_LOAD_ADDR, builder.build())
    cpu.pc = AGON_LOAD_ADDR
    cpu.run(max_cycles=MAX_CYCLES, stop_pc=builder.labels[label])
    assert cpu.pc == builder.labels[label], "never reached " + label
    return cpu


def reference_input(query: str, context: str = " " * 8) -> np.ndarray:
    """The input vector a build should have tokenized ``query`` into."""
    import libinfer

    return np.concatenate(
        [libinfer.trigram_encode(query), libinfer.context_encode(context)]
    )
