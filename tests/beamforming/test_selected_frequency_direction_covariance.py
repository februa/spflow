"""active subset単一周波数の方位別共分散積分を検証する。"""

import numpy as np

from spflow.beamforming import (
    DirectionMatchedCovarianceAccumulator,
    SelectedFrequencyDirectionCovarianceAccumulator,
    build_two_second_covariance_snapshot_schedule,
)


def test_selected_frequency_accumulator_matches_full_bin_result() -> None:
    """同じsnapshot・係数なら単一bin縮退結果が全bin積分の対象binと一致する。"""

    positions = np.array([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    schedule = build_two_second_covariance_snapshot_schedule(
        positions,
        fs_hz=1024.0,
        sound_speed_m_s=1500.0,
        snapshot_length_samples=128,
        beams_per_half=5,
    )
    generator = np.random.default_rng(8401)
    steering_full = (
        generator.standard_normal((3, 65, 9)) + 1j * generator.standard_normal((3, 65, 9))
    ).astype(np.complex64)
    selected_bin = 16
    full = DirectionMatchedCovarianceAccumulator(
        schedule,
        coef=0.25,
        steering_table=steering_full,
    )
    selected = SelectedFrequencyDirectionCovarianceAccumulator(
        schedule,
        steering_full[:, selected_bin, :],
        np.ones(3, dtype=np.float32),
        frequency_bin_index=selected_bin,
        coef=0.25,
    )
    signal = generator.standard_normal((3, 1024)).astype(np.float32)
    for _ in range(2):
        full.process_one_second(signal)
        selected.process_one_second(signal)

    full_result = full.completed_steering_metrics()
    selected_result = selected.completed_result()
    np.testing.assert_allclose(
        selected_result.direction_covariance,
        full.direction_covariance[:, :, :, selected_bin],
        rtol=1.0e-6,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(selected_result.eta, full_result.eta[:, selected_bin], rtol=2.0e-5, atol=2.0e-5)
    assert selected_result.frequency_hz == 128.0


def test_selected_frequency_accumulator_reports_shading_noise_reference() -> None:
    """連続shadingのN_effとnoise基準を完成結果へ保持する。"""

    schedule = build_two_second_covariance_snapshot_schedule(
        np.zeros((3, 3), dtype=np.float32),
        fs_hz=1024.0,
        sound_speed_m_s=1500.0,
        snapshot_length_samples=128,
        beams_per_half=5,
    )
    accumulator = SelectedFrequencyDirectionCovarianceAccumulator(
        schedule,
        np.ones((3, 9), dtype=np.complex64),
        np.array([1.0, 0.5, 0.0], dtype=np.float32),
        frequency_bin_index=16,
        coef=0.25,
    )
    signal = np.random.default_rng(8402).standard_normal((3, 1024)).astype(np.float32)
    accumulator.process_one_second(signal)
    accumulator.process_one_second(signal)
    result = accumulator.completed_result()

    expected_n_eff = 1.5**2 / 1.25
    np.testing.assert_allclose(result.effective_channel_count, expected_n_eff)
    np.testing.assert_allclose(result.noise_eta_reference, 1.0 / expected_n_eff)
