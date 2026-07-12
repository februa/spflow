"""channel数・周波数別shadingを含むsteering power校正を検証する。"""

from pathlib import Path

import numpy as np
import pytest

from evaluations.beamforming.steering_power_threshold_calibration import (
    calculate_steering_power_calibration_signature,
    calibrate_steering_power_thresholds,
)
from spflow.beamforming import prepare_steering_power_channel_weighting
from tools.calibrate_steering_power_thresholds import calibrate_threshold_file


def test_weighting_rejects_negative_and_all_zero_frequency_weights() -> None:
    """負係数と全channel無効binを早期拒否し、無効etaの生成を防ぐ。"""

    steering = np.ones((3, 2, 1), dtype=np.complex64)
    negative = np.ones((3, 2), dtype=np.float32)
    negative[0, 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        prepare_steering_power_channel_weighting(steering, negative)

    all_zero_bin = np.ones((3, 2), dtype=np.float32)
    all_zero_bin[:, 1] = 0.0
    with pytest.raises(ValueError, match="at least one channel"):
        prepare_steering_power_channel_weighting(steering, all_zero_bin)


def test_weighted_eta_preserves_matched_target_and_white_noise_reference() -> None:
    """矩形maskと連続shadingでtarget eta=1、noise平均=1/N_effを確認する。"""

    steering = np.ones((4, 2, 1), dtype=np.complex64)
    weights = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.5],
            [1.0, 0.25],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    weighting = prepare_steering_power_channel_weighting(steering, weights)

    np.testing.assert_array_equal(weighting.active_channel_count, [4, 3])
    expected_n_eff = np.array([4.0, 1.75**2 / (1.0 + 0.25 + 0.0625)], dtype=np.float32)
    np.testing.assert_allclose(weighting.effective_channel_count, expected_n_eff, rtol=1.0e-6)
    np.testing.assert_allclose(weighting.noise_eta_reference, 1.0 / expected_n_eff, rtol=1.0e-6)

    # 整合target X=a*sでは、shading分布が異なっても重み付きsteering powerと
    # 重み付きtotal powerが一致し、各binのetaは1を維持する。
    target_spectrum = np.ones((4, 2), dtype=np.complex64)
    projected = np.einsum(
        "ik,ik->k",
        weighting.projection_table[:, :, 0].conj(),
        target_spectrum,
    )
    target_steering_power = np.abs(projected) ** 2
    target_total_power = np.sum(weights * np.abs(target_spectrum) ** 2, axis=0)
    np.testing.assert_allclose(target_steering_power / target_total_power, [1.0, 1.0], atol=1.0e-6)

    # 十分な独立sampleの複素白色雑音で、etaのpower比平均を理論1/N_effへ近づける。
    generator = np.random.default_rng(5201)
    noise = (
        generator.standard_normal((4, 2, 200_000))
        + 1j * generator.standard_normal((4, 2, 200_000))
    ).astype(np.complex64)
    noise_projection = np.einsum(
        "ik,ikn->kn",
        weighting.projection_table[:, :, 0].conj(),
        noise,
        optimize=True,
    )
    steering_power_mean = np.mean(np.abs(noise_projection) ** 2, axis=1)
    total_power_mean = np.mean(
        np.einsum("ik,ikn->kn", weights, np.abs(noise) ** 2, optimize=True),
        axis=1,
    )
    np.testing.assert_allclose(
        steering_power_mean / total_power_mean,
        weighting.noise_eta_reference,
        rtol=8.0e-3,
    )


def test_threshold_calibration_records_channel_profile_and_signature() -> None:
    """異なるN_effを持つbinから周波数別thresholdと監査signatureを生成する。"""

    generator = np.random.default_rng(5202)
    n_eff = np.array([4.0, 2.0], dtype=np.float32)
    active = np.array([4, 3], dtype=np.int32)
    noise = np.clip(
        generator.normal(loc=(1.0 / n_eff)[None, None, :], scale=0.025, size=(20, 8, 2)),
        0.0,
        1.0,
    ).astype(np.float32)
    target = np.clip(
        generator.normal(loc=np.array([0.85, 0.75])[None, None, :], scale=0.04, size=(20, 2, 2)),
        0.0,
        1.0,
    ).astype(np.float32)
    signature = calculate_steering_power_calibration_signature(
        sensor_positions_m=np.zeros((4, 3), dtype=np.float32),
        frequency_hz=np.array([128.0, 256.0], dtype=np.float32),
        direction_azimuth_deg=np.array([20.0, 40.0], dtype=np.float32),
        channel_weight_table=np.array(
            [[1.0, 1.0], [1.0, 0.5], [1.0, 0.5], [1.0, 0.0]],
            dtype=np.float32,
        ),
        sound_speed_m_s=1500.0,
        snapshot_length_samples=128,
        integration_time_seconds=40.0,
    )
    result = calibrate_steering_power_thresholds(
        noise,
        target,
        effective_channel_count=n_eff,
        active_channel_count=active,
        configuration_signature=signature,
    )

    assert len(result.configuration_signature) == 64
    np.testing.assert_array_equal(result.active_channel_count, active)
    np.testing.assert_allclose(result.noise_eta_reference, 1.0 / n_eff)
    assert bool(np.all(result.gamma_off < result.gamma_on))
    assert bool(np.all(result.roc_auc > 0.99))
    assert bool(np.all(result.calibrated_false_positive_rate <= 0.02))
    # target 10 percentileをgamma_onに使うため、境界を含む検出率は理論どおり90%以上となる。
    assert bool(np.all(result.calibrated_detection_rate >= 0.90))


def test_threshold_file_contains_runtime_tables_and_channel_audit(tmp_path: Path) -> None:
    """運用係数JSONへthreshold、N_eff、signatureを欠落なく保存する。"""

    input_path = tmp_path / "samples.npz"
    output_path = tmp_path / "thresholds.json"
    weights = np.array([[1.0, 1.0], [1.0, 0.5], [1.0, 0.5], [1.0, 0.0]], dtype=np.float32)
    profile = prepare_steering_power_channel_weighting(
        np.ones((4, 2, 1), dtype=np.complex64),
        weights,
    )
    noise = np.array(
        [[[0.24, 0.45], [0.26, 0.55]], [[0.25, 0.50], [0.27, 0.52]]],
        dtype=np.float32,
    )
    target = np.array(
        [[[0.82, 0.76]], [[0.88, 0.81]]],
        dtype=np.float32,
    )
    np.savez_compressed(
        input_path,
        noise_eta=noise,
        target_eta=target,
        sensor_positions_m=np.zeros((4, 3), dtype=np.float32),
        frequency_hz=np.array([128.0, 256.0], dtype=np.float32),
        direction_azimuth_deg=np.array([20.0, 40.0], dtype=np.float32),
        channel_weight_table=weights,
        effective_channel_count=profile.effective_channel_count,
        active_channel_count=profile.active_channel_count,
        sound_speed_m_s=np.float64(1500.0),
        snapshot_length_samples=np.int64(128),
        integration_time_seconds=np.float64(40.0),
    )

    payload = calibrate_threshold_file(input_path, output_path)

    assert output_path.exists()
    assert payload["n_ch"] == 4
    assert payload["active_channel_count"] == [4, 3]
    assert len(payload["gamma_off"]) == 2
    assert len(payload["gamma_on"]) == 2
    assert len(payload["configuration_signature"]) == 64
