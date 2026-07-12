"""総channel数と周波数別shadingを変え、steering power閾値校正の成立性を評価する。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluations.beamforming.steering_power_threshold_calibration import (  # noqa: E402
    calculate_steering_power_calibration_signature,
    calibrate_steering_power_thresholds,
)
from spflow.beamforming import prepare_steering_power_channel_weighting  # noqa: E402


OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "steering_power_channel_sweep"
FREQUENCY_HZ = np.array([128.0, 512.0, 1024.0], dtype=np.float32)
DIRECTION_AZIMUTH_DEG = np.linspace(0.0, 180.0, 16, dtype=np.float32)
N_OBSERVATION = 128
N_EFFECTIVE_SNAPSHOT = 40


def _channel_weight_table(n_ch: int) -> NDArray[np.float32]:
    """矩形、連続Kaiser、半数activeの3種類を周波数binへ割り当てる。"""

    weights = np.ones((n_ch, FREQUENCY_HZ.size), dtype=np.float32)
    # 512 Hzは全channelを使う連続shadingとし、active countとN_effの違いを分離する。
    weights[:, 1] = np.asarray(np.kaiser(n_ch, 6.0), dtype=np.float32)
    # 1024 Hzは中央側の半数だけを矩形選択し、binごとのactive subset変化を模擬する。
    active_count = max(2, n_ch // 2)
    start = (n_ch - active_count) // 2
    weights[:, 2] = 0.0
    weights[start : start + active_count, 2] = 1.0
    return weights


def _steering_table(n_ch: int) -> NDArray[np.complex64]:
    """channel・周波数・方位で異なる決定論的unit magnitude steeringを作る。"""

    channel = np.arange(n_ch, dtype=np.float32)[:, np.newaxis, np.newaxis]
    frequency_scale = (FREQUENCY_HZ / FREQUENCY_HZ[-1])[np.newaxis, :, np.newaxis]
    direction_cosine = np.cos(np.deg2rad(DIRECTION_AZIMUTH_DEG))[np.newaxis, np.newaxis, :]
    # ULAの相対位相に相当する連続位相を与え、noise方向sampleが同一vectorへ退化しないようにする。
    phase = np.pi * channel * frequency_scale * direction_cosine
    return np.asarray(np.exp(-1j * phase), dtype=np.complex64)


def _eta_samples(
    steering: NDArray[np.complex64],
    weights: NDArray[np.float32],
    *,
    seed: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """白色雑音と0 dB target+noiseの積分eta sampleを生成する。"""

    generator = np.random.default_rng(seed)
    weighting = prepare_steering_power_channel_weighting(steering, weights)
    n_ch, n_bin, n_direction = steering.shape
    noise_eta = np.empty((N_OBSERVATION, n_direction, n_bin), dtype=np.float32)
    target_eta = np.empty((N_OBSERVATION, 1, n_bin), dtype=np.float32)
    target_direction_index = n_direction // 2
    target_steering = steering[:, :, target_direction_index]
    target_projection = weighting.projection_table[:, :, target_direction_index]
    for observation_index in range(N_OBSERVATION):
        noise = (
            generator.standard_normal((n_ch, n_bin, N_EFFECTIVE_SNAPSHOT))
            + 1j * generator.standard_normal((n_ch, n_bin, N_EFFECTIVE_SNAPSHOT))
        ).astype(np.complex64)
        noise_total = np.sum(
            np.einsum("ik,iks->iks", weights, np.abs(noise) ** 2, optimize=True),
            axis=(0, 2),
        )
        noise_projection = np.einsum(
            "ikd,iks->dks",
            weighting.projection_table.conj(),
            noise,
            optimize=True,
        )
        noise_steering_power = np.sum(np.abs(noise_projection) ** 2, axis=2)
        noise_eta[observation_index] = np.asarray(
            noise_steering_power / noise_total[np.newaxis, :],
            dtype=np.float32,
        )

        # complex source sampleの分散をchannel noiseと同じにし、0 dB re per-channel noise RMSとする。
        source = (
            generator.standard_normal((n_bin, N_EFFECTIVE_SNAPSHOT))
            + 1j * generator.standard_normal((n_bin, N_EFFECTIVE_SNAPSHOT))
        ).astype(np.complex64)
        target_plus_noise = noise + target_steering[:, :, np.newaxis] * source[np.newaxis, :, :]
        target_total = np.sum(
            np.einsum("ik,iks->iks", weights, np.abs(target_plus_noise) ** 2, optimize=True),
            axis=(0, 2),
        )
        target_projected = np.einsum(
            "ik,iks->ks",
            target_projection.conj(),
            target_plus_noise,
            optimize=True,
        )
        target_eta[observation_index, 0] = np.asarray(
            np.sum(np.abs(target_projected) ** 2, axis=1) / target_total,
            dtype=np.float32,
        )
    return noise_eta, target_eta


def main() -> None:
    """4種類の総channel数で理論noise基準と校正FPR/TPRを比較する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for n_ch in (8, 16, 32, 64):
        steering = _steering_table(n_ch)
        weights = _channel_weight_table(n_ch)
        weighting = prepare_steering_power_channel_weighting(steering, weights)
        noise_eta, target_eta = _eta_samples(steering, weights, seed=7300 + n_ch)
        positions = np.zeros((n_ch, 3), dtype=np.float32)
        positions[:, 0] = np.linspace(-16.0, 16.0, n_ch, dtype=np.float32)
        signature = calculate_steering_power_calibration_signature(
            sensor_positions_m=positions,
            frequency_hz=FREQUENCY_HZ,
            direction_azimuth_deg=DIRECTION_AZIMUTH_DEG,
            channel_weight_table=weights,
            sound_speed_m_s=1500.0,
            snapshot_length_samples=128,
            integration_time_seconds=40.0,
        )
        calibration = calibrate_steering_power_thresholds(
            noise_eta,
            target_eta,
            effective_channel_count=weighting.effective_channel_count,
            active_channel_count=weighting.active_channel_count,
            configuration_signature=signature,
        )
        records.append(
            {
                "n_ch": n_ch,
                "frequency_hz": FREQUENCY_HZ.tolist(),
                "active_channel_count": weighting.active_channel_count.tolist(),
                "effective_channel_count": weighting.effective_channel_count.tolist(),
                "noise_eta_reference": weighting.noise_eta_reference.tolist(),
                "noise_eta_mean": calibration.noise_mean.tolist(),
                "noise_reference_error": calibration.noise_reference_error.tolist(),
                "gamma_off": calibration.gamma_off.tolist(),
                "gamma_on": calibration.gamma_on.tolist(),
                "roc_auc": calibration.roc_auc.tolist(),
                "calibrated_false_positive_rate": calibration.calibrated_false_positive_rate.tolist(),
                "calibrated_detection_rate": calibration.calibrated_detection_rate.tolist(),
                "configuration_signature": signature,
            }
        )
    payload = {
        "evaluation": "channel count and frequency-dependent shading threshold calibration",
        "n_observation": N_OBSERVATION,
        "effective_snapshot_count_per_observation": N_EFFECTIVE_SNAPSHOT,
        "target_snr_db_re_per_channel_noise_rms": 0.0,
        "records": records,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
