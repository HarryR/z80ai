"""Pin the exact bytes each backend emits for the shipped models.

Refactoring the code generator should not change the code it generates. These
hashes make that explicit: a refactor that is meant to be behaviour-preserving
keeps them, and a deliberate codegen change updates them in the same commit
that explains why.

To update after an intended change::

    ./build-examples.sh dist
    shasum -a 256 dist/*

The semantic tests elsewhere check that the generated code is *correct*; these
only check that it has not moved.
"""

from __future__ import annotations

import hashlib
import os

import buildez80
import buildfastz80com
import buildz80com
import buildz80tap
import pytest

# artifact -> (module, example, sha256 of the image as written to disk)
GOLDEN = {
    "GUESS.COM": (buildz80com, "guess",
                  "1c26b542228fff147991e4b2b8c1e92652d46b4fb8b9eeafbb0442169b085a5a"),
    "GUESS-FAST.COM": (buildfastz80com, "guess",
                       "cd12b1f6055558a9a2ae2f6b241c6c7e5229f1cc325efacbd61842d00f2ef3ac"),
    "GUESS.bin": (buildez80, "guess",
                  "8556a8f9d323760370ef15992ac38c1115558bb2c49c626f2ab9df05175fe0c1"),
    "CHAT.COM": (buildz80com, "tinychat",
                 "d7acba3d23bd5ed94e5028855f01d82fff918766bd2907c8b9c3ae1499c4cd70"),
    "CHAT-FAST.COM": (buildfastz80com, "tinychat",
                      "788d571fd54bf0aa459732428ecc84c62d48038bc03466d4de3e2309ab106e6e"),
    "CHAT.bin": (buildez80, "tinychat",
                 "502f5d94ee43339079023161ffb3dc1ea92f12767af22797e3f91ceb28fc9207"),
}

# The .TAP hashes cover the container, not the raw image, so they are checked
# through the same header/data blocks the build script writes.
GOLDEN_TAP = {
    "GUESS.TAP": ("guess",
                  "4d3a4f26dcfe286e0a4d3bb7d20d18f4202d791506029f707d1cc0dba3c6eae7"),
    "CHAT.TAP": ("tinychat",
                 "9570ee86681ffac7206d124751336e8f5197dfc9592cf06c34928a0790d4252c"),
}


def model_path(examples_dir: str, example: str) -> str:
    path = os.path.join(examples_dir, example, "model.npz")
    if not os.path.exists(path):
        pytest.skip(f"{example} example model not present")
    return path


@pytest.mark.parametrize("artifact", sorted(GOLDEN))
def test_generated_image_is_unchanged(artifact, examples_dir):
    module, example, expected = GOLDEN[artifact]
    image = module.build_autoreg(model_path(examples_dir, example)).build()
    got = hashlib.sha256(image).hexdigest()
    assert got == expected, (
        f"{artifact} changed: {got}\n"
        f"If that was intended, update GOLDEN in this file and say why in the "
        f"commit message."
    )


@pytest.mark.parametrize("artifact", sorted(GOLDEN_TAP))
def test_generated_tap_is_unchanged(artifact, examples_dir):
    example, expected = GOLDEN_TAP[artifact]
    builder = buildz80tap.build_autoreg(model_path(examples_dir, example))
    image = builder.build()
    tap = buildz80tap.build_tap_header("CHAT", builder.org, len(image))
    tap += buildz80tap.build_tap_data(image)
    got = hashlib.sha256(tap).hexdigest()
    assert got == expected, (
        f"{artifact} changed: {got}\n"
        f"If that was intended, update GOLDEN_TAP in this file and say why in "
        f"the commit message."
    )


def test_every_shipped_artifact_is_covered():
    """build-examples.sh and the release list must not outgrow this file."""
    import verify_artifacts

    assert set(verify_artifacts.ARTIFACTS) == set(GOLDEN) | set(GOLDEN_TAP)
