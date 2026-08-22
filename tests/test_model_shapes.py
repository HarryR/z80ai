"""Layer discovery and the limits a Z80 backend can actually assemble."""

from __future__ import annotations

import numpy as np
import pytest

import buildfastz80com
import buildz80com
import buildz80tap
import libinfer


def test_layers_are_discovered_in_numeric_order():
    """Lexical sorting would run fc10 straight after fc1."""
    params = {f"fc{i}_weight": np.zeros((1, 1)) for i in range(1, 12)}
    params.update({f"fc{i}_bias": np.zeros(1) for i in range(1, 12)})
    names, _sizes = libinfer.discover_layers(params)
    assert names == [f"fc{i}" for i in range(1, 12)]


def test_discover_layers_reports_sizes():
    params = {
        "fc1_weight": np.zeros((8, 256)), "fc1_bias": np.zeros(8),
        "fc2_weight": np.zeros((3, 8)), "fc2_bias": np.zeros(3),
    }
    _names, sizes = libinfer.discover_layers(params)
    assert sizes == [256, 8, 3]


def test_validate_accepts_the_z80_maximum():
    """256 is fine: DJNZ reads a zero start as 256."""
    libinfer.validate_z80_layers([256, 256, 11])


def test_validate_rejects_layers_the_z80_cannot_count():
    with pytest.raises(ValueError, match="exceed the Z80 limit"):
        libinfer.validate_z80_layers([256, 300, 11])


@pytest.mark.parametrize("module", [buildz80com, buildfastz80com, buildz80tap])
def test_builders_reject_oversized_models(module, tmp_path, model_factory):
    """Without this the neuron loop silently counts the wrong number of times."""
    model = model_factory([256, 300, 8], charset=" AB\x00", seed=31)
    path = str(tmp_path / "wide.npz")
    model.save_npz(path)
    with pytest.raises(ValueError, match="exceed the Z80 limit"):
        module.build_autoreg(path)
