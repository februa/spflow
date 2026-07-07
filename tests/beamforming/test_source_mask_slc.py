"""source-mask non-source leakage subtractor SLC の回帰試験。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from spflow.beamforming import (
    SourceMaskNonSourceLeakageSubtractor,
    SourceMaskSlcConfig,
    build_source_sector_mask,
)


def _make_source_leakage_beam_output(n_sample: int = 2048) -> NDArray[np.float64]:
    """source-correlated leakage を含む beam-domain 信号を作る。

    source beam 1 と 3 は観測対象 source とし、non-source beam 0 と 2 へそれぞれ
    source 由来の漏れ込みを入れる。A2 SLC は source beam を変えずに、この相関成分だけを下げる。
    """
    time_axis = np.arange(n_sample, dtype=np.float64)
    source_a = np.cos(2.0 * np.pi * 0.071 * time_axis)
    source_b = np.cos(2.0 * np.pi * 0.193 * time_axis + 0.2)
    independent_floor = 0.05 * np.cos(2.0 * np.pi * 0.311 * time_axis)

    # beam_output shape: [n_beam, n_sample]。
    # axis=0 が scan beam、axis=1 が時間 sample であり、
    # source beam 1/3 は copy-through の対象である。
    return np.stack(
        [
            0.7 * source_a + independent_floor,
            source_a,
            0.5 * source_b + independent_floor,
            source_b,
            independent_floor,
        ],
        axis=0,
    ).astype(np.float64)


def test_source_mask_slc_preserves_source_beams_and_reduces_non_source_leakage() -> None:
    """source beam を維持し、non-source beam の source 相関成分だけを下げる。"""
    beam_output = _make_source_leakage_beam_output()
    source_mask = build_source_sector_mask(
        n_beam=beam_output.shape[0],
        source_beam_indices=np.array([1, 3], dtype=np.int64),
        guard_beam_count=0,
    )
    subtractor = SourceMaskNonSourceLeakageSubtractor(
        SourceMaskSlcConfig(
            eta=1.0,
            loading=1.0e-5,
            tap_len=1,
            min_ref=1,
            sample_per_dof=1.0,
            condition_number_limit=1.0e8,
        )
    )

    result = subtractor.process(beam_output=beam_output, source_sector_mask=source_mask)

    assert result.health.mode == "NORMAL"
    assert result.weights is not None
    np.testing.assert_allclose(result.raw_output[1], beam_output[1], atol=1.0e-12)
    np.testing.assert_allclose(result.raw_output[3], beam_output[3], atol=1.0e-12)

    before_non_source_power = float(np.mean(beam_output[0] ** 2 + beam_output[2] ** 2))
    after_non_source_power = float(
        np.mean(np.abs(result.raw_output[0]) ** 2 + np.abs(result.raw_output[2]) ** 2)
    )
    assert after_non_source_power < before_non_source_power * 0.1
    assert result.health.nan_inf_count == 0


def test_source_mask_slc_eta_zero_matches_fixed_baseline() -> None:
    """eta=0 では係数推定後も出力を固定整相 baseline と一致させる。

    eta sweep の基準点では、方式差ではなく評価系の差だけが出るべきである。
    そのため raw/effective とも baseline と完全一致することを確認する。
    """
    beam_output = _make_source_leakage_beam_output()
    source_mask = build_source_sector_mask(
        n_beam=beam_output.shape[0],
        source_beam_indices=np.array([1, 3], dtype=np.int64),
        guard_beam_count=0,
    )
    subtractor = SourceMaskNonSourceLeakageSubtractor(
        SourceMaskSlcConfig(
            eta=0.0,
            loading=1.0e-5,
            tap_len=1,
            min_ref=1,
            sample_per_dof=1.0,
        )
    )

    result = subtractor.process(beam_output=beam_output, source_sector_mask=source_mask)

    np.testing.assert_allclose(result.raw_output, beam_output, atol=1.0e-12)
    np.testing.assert_allclose(result.effective_output, beam_output, atol=1.0e-12)
    assert result.health.nan_inf_count == 0


def test_source_mask_slc_uses_fixed_fallback_when_reference_is_empty() -> None:
    """source reference が空なら SLC を動かさず fixed fallback を返す。

    参照 beam がない状態で leakage 推定を行うと、抑圧対象の物理的根拠がなくなるため、
    例外ではなく DISABLED として fixed baseline を effective output にする。
    """
    beam_output = _make_source_leakage_beam_output()
    source_mask = build_source_sector_mask(
        n_beam=beam_output.shape[0],
        source_beam_indices=np.array([1, 3], dtype=np.int64),
        guard_beam_count=0,
    )
    subtractor = SourceMaskNonSourceLeakageSubtractor(
        SourceMaskSlcConfig(
            eta=1.0,
            loading=1.0e-5,
            tap_len=1,
            min_ref=1,
            sample_per_dof=1.0,
        )
    )

    result = subtractor.process(
        beam_output=beam_output,
        source_sector_mask=source_mask,
        source_reference_beams=np.array([], dtype=np.int64),
    )

    assert result.health.mode == "DISABLED_REFERENCE_CAPACITY"
    assert result.health.safety_fallback_required
    assert result.weights is None
    np.testing.assert_allclose(result.effective_output, beam_output, atol=1.0e-12)


def test_source_mask_slc_rejects_reference_outside_source_mask() -> None:
    """non-source beam を source reference として誤用しないことを確認する。

    A2 は source mask 内の beam から source-correlated leakage を推定する方式であり、
    non-source beam を reference に入れると抑圧対象を再注入する危険がある。
    """
    beam_output = _make_source_leakage_beam_output()
    source_mask = build_source_sector_mask(
        n_beam=beam_output.shape[0],
        source_beam_indices=np.array([1, 3], dtype=np.int64),
        guard_beam_count=0,
    )
    subtractor = SourceMaskNonSourceLeakageSubtractor(SourceMaskSlcConfig())

    try:
        subtractor.process(
            beam_output=beam_output,
            source_sector_mask=source_mask,
            source_reference_beams=np.array([0], dtype=np.int64),
        )
    except ValueError as exc:
        assert "inside source mask" in str(exc)
    else:
        raise AssertionError("source reference outside source mask must be rejected")
