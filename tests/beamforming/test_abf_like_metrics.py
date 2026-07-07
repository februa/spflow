"""ABF-like non-source sector 評価指標の回帰試験。"""

from __future__ import annotations

import numpy as np

from spflow.beamforming import (
    build_source_sector_mask,
    calculate_abf_like_non_source_metrics,
    detect_source_beam_indices_from_level_peaks,
    judge_abf_like_non_source_metrics,
)


def test_build_source_sector_mask_excludes_source_guards_from_non_source() -> None:
    """source guard が non-source sector から除外されることを確認する。

    ABF-like 評価では target や interferer を観測対象 source として残すため、
    source 中心 beam だけでなく guard 内の mainlobe 近傍も false peak として数えない。
    """
    mask = build_source_sector_mask(
        n_beam=9,
        source_beam_indices=np.array([2, 6], dtype=np.int64),
        guard_beam_count=1,
        mask_type="oracle",
    )

    expected_source_mask = np.array(
        [False, True, True, True, False, True, True, True, False], dtype=np.bool_
    )
    np.testing.assert_array_equal(mask.source_mask, expected_source_mask)
    np.testing.assert_array_equal(mask.non_source_mask, np.logical_not(expected_source_mask))
    assert mask.as_dict()["non_source_true_count"] == 3


def test_abf_like_metrics_judges_non_source_envelope_not_marker_only() -> None:
    """non-source sector の包絡線改善を採否指標として計算する。

    source beam 2 と 6 は観測対象として維持し、それ以外の non-source beam では
    global peak、percentile、integrated level が下がる条件を作る。
    """
    axis_azimuth_deg = np.linspace(0.0, 80.0, 9)
    before_levels_db = np.array([-35.0, -20.0, 0.0, -19.0, -14.0, -18.0, -6.0, -19.5, -33.0])
    after_levels_db = np.array([-38.0, -20.1, -0.1, -19.2, -17.0, -18.5, -6.2, -20.0, -36.0])
    mask = build_source_sector_mask(
        n_beam=9,
        source_beam_indices=np.array([2, 6], dtype=np.int64),
        guard_beam_count=1,
        mask_type="oracle",
    )

    metrics = calculate_abf_like_non_source_metrics(
        axis_azimuth_deg=axis_azimuth_deg,
        before_levels_db=before_levels_db,
        after_levels_db=after_levels_db,
        source_sector_mask=mask,
        level_unit_label="dB re input RMS",
    )
    decision = judge_abf_like_non_source_metrics(
        metrics, realtime_factor=0.4, nan_inf_count=0, condition_number=100.0
    )

    assert metrics.max_abs_source_peak_delta_db <= 0.21
    assert metrics.non_source_global_peak_delta_db <= -1.0
    assert metrics.non_source_p95_level_delta_db <= -1.0
    assert metrics.non_source_integrated_level_delta_db <= -1.0
    assert metrics.false_peak_count_delta <= 0
    assert decision.status == "pass"


def test_gated_local_worsening_ignores_deep_valley_that_remains_invisible() -> None:
    """深い谷だけの変化を visible false peak として過大評価しない。

    before が source peak から 60 dB より低く、after も source peak から 40 dB より低い点は、
    表示上の non-source peak を作っていないため gated worsening の対象外にする。
    """
    axis_azimuth_deg = np.arange(5.0, dtype=np.float64)
    before_levels_db = np.array([-100.0, 0.0, -70.0, -6.0, -100.0])
    after_levels_db = np.array([-50.0, 0.0, -65.0, -6.0, -45.0])
    mask = build_source_sector_mask(
        n_beam=5,
        source_beam_indices=np.array([1, 3], dtype=np.int64),
        guard_beam_count=0,
    )

    metrics = calculate_abf_like_non_source_metrics(
        axis_azimuth_deg=axis_azimuth_deg,
        before_levels_db=before_levels_db,
        after_levels_db=after_levels_db,
        source_sector_mask=mask,
        level_unit_label="dB re input RMS",
    )

    assert metrics.max_local_worsening_db_gated == 0.0
    assert metrics.max_local_worsening_azimuth_deg is None


def test_detect_source_beam_indices_from_level_peaks_uses_guard_suppression() -> None:
    """detected mask 用 peak 検出が同じ mainlobe を重複検出しないことを確認する。

    固定整相 before の BL から source mask を作る場合、最大 peak 周辺の guard を候補から外し、
    source 数を過大に見積もらないようにする。
    """
    levels_db = np.array([-40.0, -3.0, 0.0, -3.5, -30.0, -8.0, -6.0, -8.5, -40.0])

    detected = detect_source_beam_indices_from_level_peaks(
        levels_db=levels_db,
        max_source_count=3,
        guard_beam_count=1,
        threshold_db_below_peak=10.0,
    )

    np.testing.assert_array_equal(detected, np.array([2, 6], dtype=np.int64))
