"""BL/FRAZ/BTR 表示配列と BL 校正特徴量を検証する。"""

from __future__ import annotations

import numpy as np

from spflow.beamforming.evaluation_arrays import (
    build_beam_level_display_arrays,
    calculate_bl_shape_features,
    calculate_btr_relative_level_db,
)


def test_display_arrays_keep_target_and_all_source_frequency_views_separate() -> None:
    """異周波数 source が target-frequency BL だけで失われない配列定義を確認する。"""
    spectrum = np.zeros((3, 4, 2), dtype=np.complex128)
    spectrum[0, 1, :] = 1.0
    spectrum[2, 3, :] = 0.5

    arrays = build_beam_level_display_arrays(
        spectrum,
        target_frequency_index=1,
        source_frequency_indices=np.array([1, 3]),
        reference_rms=1.0,
        level_reference_label="dB re input RMS",
        floor_db=-120.0,
    )

    assert arrays.fraz_level_db.shape == (3, 4)
    np.testing.assert_allclose(arrays.target_frequency_bl_level_db, np.array([0.0, -120.0, -120.0]))
    np.testing.assert_allclose(
        arrays.source_frequency_bl_level_db, np.array([0.0, -120.0, -6.020599913279624])
    )


def test_btr_is_normalized_independently_for_each_time_row() -> None:
    """BTR が時刻ごとに最大 beam を 0 dB とする相対表示になることを確認する。"""
    beam_time_rms = np.array([[1.0, 0.5], [2.0, 1.0]], dtype=np.float64)

    btr = calculate_btr_relative_level_db(beam_time_rms)

    np.testing.assert_allclose(btr[:, 0], np.zeros(2))
    np.testing.assert_allclose(btr[:, 1], np.full(2, -6.020599913279624))


def test_bl_shape_features_are_observations_without_adoption_decision() -> None:
    """mainlobe幅、guard外統計、source間valleyを同じBLから抽出できることを確認する。"""
    azimuth_deg = np.arange(7, dtype=np.float64) * 10.0
    levels_db = np.array([-20.0, 0.0, -2.0, -12.0, -1.0, -3.0, -18.0])
    source_mask = np.array([False, True, True, False, True, True, False])

    features = calculate_bl_shape_features(
        azimuth_deg,
        levels_db,
        source_mask,
        source_beam_indices=np.array([1, 4]),
        level_reference_label="dB re input RMS",
    )

    assert features.peak_azimuth_deg == 10.0
    assert features.peak_width_3db_deg == 10.0
    assert features.guard_outside_peak_level_db == -12.0
    assert features.source_to_guard_peak_margin_db == 12.0
    assert features.source_separation_valley_depth_db == 11.0
