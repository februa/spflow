"""方式3の直接eta積分、一周期遅延Weight、fallbackを検証する。"""

import numpy as np

from spflow.beamforming import (
    CovarianceFallbackSource,
    DirectionCovarianceSelectionConfig,
    DirectionMatchedCovarianceAccumulator,
    DirectionMatchedCovarianceSelector,
    build_two_second_covariance_snapshot_schedule,
    prepare_steering_power_channel_weighting,
)


def _small_schedule():
    """3ch・5beamの小規模完成周期scheduleを返す。"""

    return build_two_second_covariance_snapshot_schedule(
        np.array([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
        fs_hz=1024.0,
        sound_speed_m_s=1500.0,
        snapshot_length_samples=128,
        beams_per_half=5,
    )


def test_direct_eta_matches_quadratic_form_of_integrated_covariance() -> None:
    """同一snapshot・alphaなら直接power比と`u^H R u/trace(R)`が一致する。"""

    schedule = _small_schedule()
    rng = np.random.default_rng(1234)
    steering = (
        rng.standard_normal((3, 65, 9)) + 1j * rng.standard_normal((3, 65, 9))
    ).astype(np.complex64)
    accumulator = DirectionMatchedCovarianceAccumulator(
        schedule,
        coef=0.25,
        steering_table=steering,
    )
    signal = rng.standard_normal((3, 1024)).astype(np.float32)
    accumulator.process_one_second(signal)
    accumulator.process_one_second(signal)
    completed = accumulator.completed_steering_metrics()

    normalized = steering / np.sqrt(np.sum(np.abs(steering) ** 2, axis=0, keepdims=True))
    covariance_direction_bin = np.moveaxis(accumulator.direction_covariance, 3, 1)
    # steering `[ch,bin,direction]`を共分散の`[direction,bin,ch,ch]`へ合わせる。
    normalized_direction_bin = np.transpose(normalized, (2, 1, 0))
    numerator = np.real(
        np.einsum(
            "...i,...ij,...j->...",
            normalized_direction_bin.conj(),
            covariance_direction_bin,
            normalized_direction_bin,
            optimize=True,
        )
    )
    denominator = np.real(np.trace(covariance_direction_bin, axis1=-2, axis2=-1))
    reference_eta = numerator / denominator
    np.testing.assert_allclose(completed.eta[completed.eta_valid], reference_eta[completed.eta_valid], rtol=2.0e-5, atol=2.0e-5)


def test_shaded_direct_eta_matches_weighted_covariance_quadratic_form() -> None:
    """周波数別shadingでも直接積分etaと重み付き共分散二次形式が一致する。"""

    schedule = _small_schedule()
    generator = np.random.default_rng(1235)
    steering = (
        generator.standard_normal((3, 65, 9)) + 1j * generator.standard_normal((3, 65, 9))
    ).astype(np.complex64)
    weights = np.ones((3, 65), dtype=np.float32)
    # 高域側は3番目channelを無効化し、中域は連続shadingとして0.5を与える。
    # 同じ総channel数でもbinごとにactive countとN_effが変わる実運用条件を模擬する。
    weights[2, 33:] = 0.0
    weights[1, 16:33] = 0.5
    accumulator = DirectionMatchedCovarianceAccumulator(
        schedule,
        coef=0.25,
        steering_table=steering,
        channel_weight_table=weights,
    )
    signal = generator.standard_normal((3, 1024)).astype(np.float32)
    accumulator.process_one_second(signal)
    accumulator.process_one_second(signal)
    completed = accumulator.completed_steering_metrics()
    weighting = prepare_steering_power_channel_weighting(steering, weights)

    covariance_direction_bin = np.moveaxis(accumulator.direction_covariance, 3, 1)
    projection_direction_bin = np.transpose(weighting.projection_table, (2, 1, 0))
    numerator = np.real(
        np.einsum(
            "...i,...ij,...j->...",
            projection_direction_bin.conj(),
            covariance_direction_bin,
            projection_direction_bin,
            optimize=True,
        )
    )
    diagonal_power = np.real(np.diagonal(covariance_direction_bin, axis1=-2, axis2=-1))
    denominator = np.einsum("dki,ik->dk", diagonal_power, weights, optimize=True)
    reference_eta = numerator / denominator

    np.testing.assert_allclose(
        completed.eta[completed.eta_valid],
        reference_eta[completed.eta_valid],
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    np.testing.assert_array_equal(completed.active_channel_count[[0, 20, 40]], [3, 3, 2])
    np.testing.assert_allclose(completed.noise_eta_reference, 1.0 / completed.effective_channel_count)


def test_selector_applies_completed_weight_one_cycle_later() -> None:
    """周期tのWeightを周期t+1へ適用し、初回は方式2へfallbackする。"""

    schedule = _small_schedule()
    steering = np.ones((3, 65, 9), dtype=np.complex64)
    selector = DirectionMatchedCovarianceSelector(
        schedule,
        steering,
        DirectionCovarianceSelectionConfig(
            gamma_off=np.zeros(65, dtype=np.float32),
            gamma_on=np.full(65, 0.5, dtype=np.float32),
        ),
        integration_time_seconds=10.0,
    )
    rng = np.random.default_rng(5678)
    signal = rng.standard_normal((3, 1024)).astype(np.float32)
    method2 = np.repeat(np.eye(3, dtype=np.complex64)[:, :, np.newaxis], 65, axis=2)

    assert selector.process_one_second(signal, input_series_id="scene-a", method2_covariance=method2) is None
    first = selector.process_one_second(signal, input_series_id="scene-a", method2_covariance=method2)
    assert first is not None
    np.testing.assert_array_equal(first.applied_weight, np.zeros_like(first.applied_weight))
    np.testing.assert_array_equal(
        first.fallback_source,
        np.full(65, int(CovarianceFallbackSource.METHOD2), dtype=np.int8),
    )

    assert selector.process_one_second(signal, input_series_id="scene-a", method2_covariance=method2) is None
    second = selector.process_one_second(signal, input_series_id="scene-a", method2_covariance=method2)
    assert second is not None
    np.testing.assert_allclose(second.applied_weight, first.completed_weight)
    assert bool(np.any(second.fallback_source == int(CovarianceFallbackSource.METHOD3_WEIGHTED)))


def test_selector_discards_previous_state_when_input_series_changes() -> None:
    """入力系列切替時に途中周期と前系列Weightを破棄する。"""

    schedule = _small_schedule()
    selector = DirectionMatchedCovarianceSelector(
        schedule,
        np.ones((3, 65, 9), dtype=np.complex64),
        DirectionCovarianceSelectionConfig(
            gamma_off=np.zeros(65, dtype=np.float32),
            gamma_on=np.full(65, 0.5, dtype=np.float32),
        ),
        integration_time_seconds=10.0,
    )
    signal = np.ones((3, 1024), dtype=np.float32)
    assert selector.process_one_second(signal, input_series_id="scene-a") is None
    assert selector.process_one_second(signal, input_series_id="scene-b") is None
    completed = selector.process_one_second(signal, input_series_id="scene-b")
    assert completed is not None
    np.testing.assert_array_equal(completed.applied_weight, np.zeros_like(completed.applied_weight))
