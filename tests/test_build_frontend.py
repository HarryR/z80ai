"""Tests for the unified build front-end and its target selection."""

from __future__ import annotations

import build
import buildz80com


def test_auto_picks_the_fast_layout_when_it_fits(tiny_model_path):
    builder, layout = build.build_cpm(tiny_model_path, max_output_len=4)
    assert layout == "fast"
    assert build._fits_in_tpa(builder)


def test_auto_falls_back_to_packed_when_fast_would_not_fit(monkeypatch, tiny_model_path):
    """A model with few zero weights makes the index lists overrun the TPA."""
    monkeypatch.setattr(build, "_fits_in_tpa", lambda builder: False)
    _builder, layout = build.build_cpm(tiny_model_path, max_output_len=4)
    assert layout == "packed"


def test_explicit_layouts_are_honoured(tiny_model_path):
    _, packed = build.build_cpm(tiny_model_path, 4, prefer="packed")
    _, fast = build.build_cpm(tiny_model_path, 4, prefer="fast")
    assert (packed, fast) == ("packed", "fast")


def test_fits_in_tpa_leaves_room_for_the_stack(tiny_model_path):
    builder = buildz80com.build_autoreg(tiny_model_path, max_output_len=4)
    end = builder.org + len(builder.build())
    assert build._fits_in_tpa(builder) == (
        end + build.CPM_STACK_MARGIN <= build.CPM_TPA_TOP
    )


def test_ez80_auto_picks_the_fastest_kernel_when_it_fits(tiny_model_path):
    import buildez80

    _builder, layout = build.build_ez80(tiny_model_path, max_output_len=4)
    assert layout == buildez80.KERNELS[0]


def test_ez80_auto_falls_back_to_compact_when_unrolling_would_not_fit(
    monkeypatch, tiny_model_path
):
    """Shrink the ceiling rather than build a model big enough to overrun it."""
    import buildez80

    monkeypatch.setattr(buildez80, "AGON_MAX_IMAGE", 8 * 1024)
    _builder, layout = build.build_ez80(tiny_model_path, max_output_len=4)
    assert layout == "compact"


def test_ez80_explicit_kernels_are_honoured(tiny_model_path):
    import buildez80

    for kernel in buildez80.KERNELS:
        _, chosen = build.build_ez80(tiny_model_path, 4, kernel=kernel)
        assert chosen == kernel
