#!/usr/bin/env python3
"""
Measure inference cost for each build target by running it in the emulator.

Two numbers are reported per target:

  instructions  instructions retired for one forward pass. Platform-neutral,
                so it stays meaningful when comparing across architectures.
  T-states      Z80 bus cycles, and the wall-clock estimate that follows from
                them, at each platform's stock clock.

Usage:
    python bench.py --model examples/guess/model.npz
    python bench.py --model examples/guess/model.npz --target com fast tap
"""

from __future__ import annotations

import argparse
import importlib

from libhost import CPMHost, ZXHost

# module, description, clock Hz
TARGETS = {
    "com": ("buildz80com", "CP/M, packed 2-bit weights", 4_000_000),
    "fast": ("buildfastz80com", "CP/M, per-value index lists", 4_000_000),
    "tap": ("buildz80tap", "ZX Spectrum, packed weights", 3_500_000),
}


def _host(target: str, query: str):
    if target == "tap":
        return ZXHost(stdin=[query, "!"]), 0x8000
    return CPMHost(cmdline=query), 0x0100


def measure(target: str, model_path: str, query: str = "HELLO") -> dict:
    """Cycle and instruction counts for one forward pass of ``target``."""
    module = importlib.import_module(TARGETS[target][0])
    builder = module.build_autoreg(model_path, max_output_len=1)
    image = builder.build()

    host, org = _host(target, query)
    cpu = host.cpu
    cpu.load(org, image)
    cpu.pc = org

    cpu.run(max_cycles=2_000_000_000, stop_pc=builder.labels["GENLOOP"])
    start_t, start_i = cpu.tstates, cpu.instructions
    cpu.run(max_cycles=2_000_000_000, stop_pc=builder.labels["ARGMAX"])

    return {
        "target": target,
        "bytes": len(image),
        "tstates": cpu.tstates - start_t,
        "instructions": cpu.instructions - start_i,
        "clock": TARGETS[target][2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", "-m", default="examples/guess/model.npz")
    parser.add_argument("--query", "-q", default="HELLO")
    parser.add_argument("--target", "-t", nargs="+", default=["com", "fast"],
                        choices=sorted(TARGETS))
    args = parser.parse_args()

    rows = [measure(t, args.model, args.query) for t in args.target]
    baseline = rows[0]["instructions"]

    print(f"\n{'target':8} {'size':>9} {'instructions':>13} {'speedup':>8} "
          f"{'Z80 T-states':>14} {'sec @ clock':>12}")
    print("-" * 70)
    for row in rows:
        speedup = baseline / row["instructions"]
        cycles = f"{row['tstates']:,}"
        seconds = f"{row['tstates'] / row['clock']:.2f}"
        print(
            f"{row['target']:8} {row['bytes']:9,} {row['instructions']:13,} "
            f"{speedup:7.2f}x {cycles:>14} {seconds:>12}"
        )
    print("\nspeedup is relative to the first target listed, by instruction count.\n")


if __name__ == "__main__":
    main()
