#!/usr/bin/env python3
"""
Execute release artifacts and check they still answer correctly.

CI used to build binaries and ship them without ever running one. This walks a
dist directory, boots each artifact in the emulator on its own platform, and
compares what it prints against the NumPy reference model. It also re-checks the
container formats and the size limits each platform imposes.

A build that assembles but computes the wrong thing - which is exactly what the
MULADD borrow bug did for years - fails here.

Usage:
    python verify_artifacts.py --dist dist
    python verify_artifacts.py --dist dist --query "IS IT AN ANIMAL"
"""

from __future__ import annotations

import argparse
import os
import sys

import libinfer
from libhost import run_agon, run_cpm, run_zx

CPM_TPA = 0x0100
CPM_TPA_TOP = 0xE400  # where a stock CP/M 2.2 BDOS starts
ZX_RAM_TOP = 0x10000  # one past the last byte of RAM on a 48K machine
AGON_LOAD = 0x040000
EZ80_TOP = 0x1000000

#: artifact name -> (model path, platform)
ARTIFACTS = {
    "GUESS.COM": ("examples/guess/model.npz", "cpm"),
    "GUESS-FAST.COM": ("examples/guess/model.npz", "cpm"),
    "GUESS.TAP": ("examples/guess/model.npz", "zx"),
    "GUESS.bin": ("examples/guess/model.npz", "agon"),
    "CHAT.COM": ("examples/tinychat/model.npz", "cpm"),
    "CHAT-FAST.COM": ("examples/tinychat/model.npz", "cpm"),
    "CHAT.TAP": ("examples/tinychat/model.npz", "zx"),
    "CHAT.bin": ("examples/tinychat/model.npz", "agon"),
}


class VerificationError(Exception):
    """An artifact did not behave the way the reference model says it should."""


def parse_tap(data: bytes) -> tuple[bytes, int]:
    """Unwrap a .TAP container, validating both block checksums.

    A TAP is a sequence of [length:2][flag:1][payload][checksum:1] blocks; a
    CODE file is a 19-byte header block followed by the data block.

    Returns:
        The raw image and the load address declared in the header.
    """
    blocks, pos = [], 0
    while pos < len(data):
        if pos + 2 > len(data):
            raise VerificationError("truncated TAP block length")
        length = data[pos] | (data[pos + 1] << 8)
        body = data[pos + 2 : pos + 2 + length]
        if len(body) != length:
            raise VerificationError("truncated TAP block body")
        checksum = 0
        for byte in body[:-1]:
            checksum ^= byte
        if checksum != body[-1]:
            raise VerificationError(
                f"TAP checksum mismatch: got {body[-1]:02X}, want {checksum:02X}"
            )
        blocks.append(body)
        pos += 2 + length

    if len(blocks) != 2 or blocks[0][0] != 0x00 or blocks[1][0] != 0xFF:
        raise VerificationError("expected one header block and one data block")

    header = blocks[0]
    declared = header[12] | (header[13] << 8)
    start = header[14] | (header[15] << 8)
    image = blocks[1][1:-1]
    if declared != len(image):
        raise VerificationError(
            f"TAP header declares {declared} bytes, data block holds {len(image)}"
        )
    return image, start


def check_fits(org: int, size: int, top: int, where: str) -> None:
    """Refuse an artifact that cannot load on the machine it targets."""
    end = org + size
    if end > top:
        raise VerificationError(
            f"{size:,} bytes loaded at {org:#06x} runs to {end:#07x}, "
            f"past {where} ({top - 1:#06x}) by {end - top:,} bytes"
        )


def run_artifact(path: str, platform: str, query: str) -> str:
    """Boot the artifact on its platform and return what it printed."""
    with open(path, "rb") as fh:
        data = fh.read()

    if platform == "cpm":
        check_fits(CPM_TPA, len(data), CPM_TPA_TOP, "the top of the CP/M TPA")
        out, host = run_cpm(data, cmdline=query)
        if not host.finished:
            raise VerificationError("program never returned to CP/M")
        return out

    if platform == "zx":
        # The load address comes from the TAP header, so this checks the
        # artifact as a Spectrum would actually see it.
        image, org = parse_tap(data)
        check_fits(org, len(image), ZX_RAM_TOP, "the top of 48K RAM")
        out, _host = run_zx(image, stdin=[query, "!"], org=org)
        return out

    if platform == "agon":
        check_fits(AGON_LOAD, len(data), EZ80_TOP, "the top of the 16MB space")
        out, _host = run_agon(data, stdin=[query, "!"])
        return out

    raise VerificationError(f"unknown platform {platform}")


def verify(dist: str, query: str) -> list[tuple[str, str, str, bool]]:
    """Check every artifact present in ``dist``. Returns per-artifact results."""
    results = []
    for name, (model_path, platform) in ARTIFACTS.items():
        path = os.path.join(dist, name)
        if not os.path.exists(path):
            continue

        model = libinfer.Model.load(model_path)
        # The eZ80 accumulates in 24 bits and so never wraps; the Z80 targets
        # wrap at 16. Compare each against the semantics it actually implements.
        bits = 24 if platform == "agon" else 16
        expected = libinfer.generate(model, query, accum_bits=bits)

        printed = run_artifact(path, platform, query)
        # CP/M single-query mode prints only the reply; the chat-driven ZX and
        # Agon builds wrap it in prompt and echo chrome.
        ok = printed == expected if platform == "cpm" else expected in printed
        results.append((name, expected, printed, ok))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dist", "-d", default="dist", help="Directory of built artifacts")
    parser.add_argument("--query", "-q", default="HELLO", help="Query to send")
    parser.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"),
                        help="Append a markdown table here (defaults to the CI summary)")
    args = parser.parse_args()

    try:
        results = verify(args.dist, args.query)
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if not results:
        print(f"No known artifacts found in {args.dist}/", file=sys.stderr)
        return 1

    lines = ["| artifact | expected | got | |", "|---|---|---|---|"]
    failures = 0
    print(f"\nVerifying {len(results)} artifacts with query {args.query!r}\n")
    for name, expected, printed, ok in results:
        mark = "ok" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  {mark:4}  {name:16} expected {expected!r}")
        if not ok:
            print(f"        got {printed!r}")
        lines.append(
            f"| `{name}` | `{expected}` | `{printed.strip()[:24]}` | {'✅' if ok else '❌'} |"
        )

    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write(f"\n### Artifact verification (`{args.query}`)\n\n")
            fh.write("\n".join(lines) + "\n")

    print()
    if failures:
        print(f"{failures} of {len(results)} artifacts did not match the reference model")
        return 1
    print(f"All {len(results)} artifacts match the reference model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
