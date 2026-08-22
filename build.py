#!/usr/bin/env python3
"""
One front-end for every build target, with automatic selection.

There are two ways to lay out the weights for a Z80:

  packed  two bits per weight, four to a byte. Always fits, but the inner loop
          spends most of its time unpacking - about 27 million T-states per
          generated character for the shipped examples.
  fast    an index list per weight value. Roughly 75% of trained weights are
          zero and cost nothing at all, so it is around 13x quicker, but the
          index lists are larger and a model with fewer zeros may not fit.

`--target auto` (the default) builds the fast layout and falls back to packed
only if the result would overrun the transient program area. That is the choice
you want almost every time, and it is what buildfastz80com.py's own header
asked for.

Usage:
    python build.py --model examples/guess/model.npz --output GUESS.COM
    python build.py --model model.npz --target zx --output CHAT.TAP
"""

from __future__ import annotations

import argparse
import os

# A stock CP/M 2.2 puts the BDOS at E400h; leave a little room for the stack.
CPM_TPA_TOP = 0xE400
CPM_STACK_MARGIN = 0x0200


def _fits_in_tpa(builder) -> bool:
    return builder.org + len(builder.build()) + CPM_STACK_MARGIN <= CPM_TPA_TOP


def build_cpm(model: str, max_output_len: int, prefer: str = "auto"):
    """Build a CP/M .COM, choosing the fastest weight layout that fits."""
    import buildfastz80com
    import buildz80com

    if prefer == "packed":
        return buildz80com.build_autoreg(model, max_output_len=max_output_len), "packed"

    fast = buildfastz80com.build_autoreg(model, max_output_len=max_output_len)
    if prefer == "fast" or _fits_in_tpa(fast):
        return fast, "fast"

    print(
        f"\nFast layout needs {len(fast.build()):,} bytes and will not fit the "
        f"TPA; falling back to packed weights."
    )
    return buildz80com.build_autoreg(model, max_output_len=max_output_len), "packed"


def build_zx(model: str, max_output_len: int):
    import buildz80tap

    return buildz80tap.build_autoreg(model, max_output_len=max_output_len), "packed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", "-m", default="command_model_autoreg.pt",
                        help="Model file to load (.npz or .pt)")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--target", "-t", default="auto",
                        choices=["auto", "cpm", "cpm-fast", "cpm-packed", "zx"],
                        help="Platform and weight layout (default: auto = cpm)")
    parser.add_argument("--max-output-len", type=int, default=50,
                        help="Maximum characters generated per response")
    args = parser.parse_args()

    target = args.target
    if target in ("auto", "cpm"):
        builder, layout = build_cpm(args.model, args.max_output_len, "auto")
    elif target == "cpm-fast":
        builder, layout = build_cpm(args.model, args.max_output_len, "fast")
    elif target == "cpm-packed":
        builder, layout = build_cpm(args.model, args.max_output_len, "packed")
    else:
        builder, layout = build_zx(args.model, args.max_output_len)

    if target == "zx":
        import buildz80tap

        image = builder.build()
        tap = buildz80tap.build_tap_header(
            os.path.basename(args.output).split(".")[0][:10], builder.org, len(image)
        ) + buildz80tap.build_tap_data(image)
        with open(args.output, "wb") as fh:
            fh.write(tap)
        print(f"\nWrote {len(tap):,} bytes to {args.output}")
    else:
        builder.save(args.output)

    size = len(builder.code)
    print(f"Target: {target}  layout: {layout}  size: {size:,} bytes "
          f"({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
