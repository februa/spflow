"""target-only、noise-only、mixed BL評価部品を検証する。"""

from __future__ import annotations

import numpy as np

from spflow.beamforming_evaluation.bl_component_metrics import (
    evaluate_mixed_bl_consistency,
    evaluate_noise_only_bl,
    evaluate_target_only_bl,
)


def test_target_only_bl_extracts_first_nulls_and_first_sidelobes() -> None:
    """mainlobeの左右first nullと、その外側で最初の副極を抽出できることを確認する。"""
    azimuth_deg = np.arange(7, dtype=np.float64) * 10.0
    level_db = np.array([-25.0, -13.0, -40.0, 0.0, -35.0, -14.0, -24.0])

    metrics = evaluate_target_only_bl(
        azimuth_deg,
        level_db,
        source_azimuth_deg=30.0,
        source_level_db=0.0,
        level_reference_label="dB re input RMS",
    )

    assert metrics.peak_azimuth_error_deg == 0.0
    assert metrics.peak_level_error_db == 0.0
    assert metrics.left_first_null_azimuth_deg == 20.0
    assert metrics.right_first_null_azimuth_deg == 40.0
    assert metrics.first_null_width_deg == 20.0
    assert metrics.left_first_sidelobe is not None
    assert metrics.right_first_sidelobe is not None
    assert metrics.left_first_sidelobe.level_db_re_mainlobe_peak == -13.0
    assert metrics.right_first_sidelobe.level_db_re_mainlobe_peak == -14.0
    assert metrics.maximum_sidelobe is not None
    assert metrics.maximum_sidelobe.azimuth_deg == 10.0
    assert metrics.grating_lobe_candidates == ()


def test_noise_only_bl_matches_white_noise_covariance_prediction() -> None:
    """矩形4channel CBFのnoise powerが1/4、array gainが約6.02dBになることを確認する。"""
    channel_count = 4
    beam_count = 3
    input_noise_power = 0.01
    weights = np.full((channel_count, beam_count), 1.0 / channel_count, dtype=np.complex128)
    covariance = input_noise_power * np.eye(channel_count, dtype=np.complex128)
    predicted_output_power = input_noise_power / channel_count
    observed_level_db = np.full(beam_count, 10.0 * np.log10(predicted_output_power))

    metrics = evaluate_noise_only_bl(
        observed_level_db,
        weights,
        covariance,
        input_channel_noise_power=input_noise_power,
        reference_rms=1.0,
        level_reference_label="dB re input RMS",
    )

    np.testing.assert_allclose(metrics.prediction_error_db, np.zeros(beam_count), atol=1.0e-12)
    np.testing.assert_allclose(
        metrics.predicted_array_gain_db,
        np.full(beam_count, 10.0 * np.log10(channel_count)),
    )


def test_mixed_bl_consistency_adds_target_and_noise_in_power() -> None:
    """無相関targetとnoiseのmixed levelがdB値の算術和ではなくpower和になることを確認する。"""
    target_level_db = np.array([0.0, -20.0])
    noise_level_db = np.array([-20.0, -20.0])
    expected_level_db = 10.0 * np.log10(
        10.0 ** (target_level_db / 10.0) + 10.0 ** (noise_level_db / 10.0)
    )

    metrics = evaluate_mixed_bl_consistency(
        target_level_db,
        noise_level_db,
        expected_level_db,
        level_reference_label="dB re input RMS",
    )

    np.testing.assert_allclose(metrics.consistency_error_db, np.zeros(2), atol=1.0e-12)
    assert metrics.maximum_absolute_consistency_error_db < 1.0e-12
