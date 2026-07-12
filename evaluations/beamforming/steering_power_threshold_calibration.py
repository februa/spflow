"""steering power etaの実運用向け周波数別thresholdを校正する。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SteeringPowerThresholdCalibration:
    """noise/target eta分布から得た周波数別thresholdと品質指標を保持する。

    このクラスは、異なるchannel数と周波数別shadingに対して、soft Weight用threshold、
    noise理論値との差、ROC AUC、校正FPR/TPRを固定shapeで返す。

    signal生成、eta積分、runtime Weight適用、成果物描画は責務に含めない。
    信号処理上は、runtimeへ渡す係数をevaluation側で再現可能に生成する結果型である。

    Attributes:
        gamma_off: soft Weightが0から増加し始めるeta。shapeは`[n_bin]`。
        gamma_on: soft Weightが1になるeta。shapeは`[n_bin]`。
        noise_mean: noise-only eta平均。shapeは`[n_bin]`。
        noise_standard_deviation: noise-only eta標準偏差。shapeは`[n_bin]`。
        noise_eta_reference: shading理論`1/N_eff`。shapeは`[n_bin]`。
        noise_reference_error: `noise_mean-noise_eta_reference`。shapeは`[n_bin]`。
        effective_channel_count: shadingを含む`N_eff`。shapeは`[n_bin]`。
        active_channel_count: 正のshading係数を持つchannel数。shapeは`[n_bin]`。
        roc_auc: bin別ROC AUC。shapeは`[n_bin]`。
        calibrated_false_positive_rate: `eta>gamma_off`のnoise比率。shapeは`[n_bin]`。
        calibrated_detection_rate: `eta>=gamma_on`のtarget比率。shapeは`[n_bin]`。
        configuration_signature: アレイ・周波数・方位・shading等を結合したSHA-256。
    """

    gamma_off: NDArray[np.float32]
    gamma_on: NDArray[np.float32]
    noise_mean: NDArray[np.float32]
    noise_standard_deviation: NDArray[np.float32]
    noise_eta_reference: NDArray[np.float32]
    noise_reference_error: NDArray[np.float32]
    effective_channel_count: NDArray[np.float32]
    active_channel_count: NDArray[np.int32]
    roc_auc: NDArray[np.float32]
    calibrated_false_positive_rate: NDArray[np.float32]
    calibrated_detection_rate: NDArray[np.float32]
    configuration_signature: str


def calculate_steering_power_calibration_signature(
    *,
    sensor_positions_m: NDArray[Any],
    frequency_hz: NDArray[Any],
    direction_azimuth_deg: NDArray[Any],
    channel_weight_table: NDArray[Any],
    sound_speed_m_s: float,
    snapshot_length_samples: int,
    integration_time_seconds: float,
) -> str:
    """thresholdを使い回せる処理条件をSHA-256 signatureへ固定する。

    Args:
        sensor_positions_m: 受波器座標。shapeは`[n_ch,3]`、単位はm。
        frequency_hz: 周波数軸。shapeは`[n_bin]`、単位はHz。
        direction_azimuth_deg: 方位軸。shapeは`[n_direction]`、単位はdeg。
        channel_weight_table: 周波数別shading。shapeは`[n_ch,n_bin]`。
        sound_speed_m_s: 音速。単位はm/s。
        snapshot_length_samples: FFT snapshot長。単位はsample。
        integration_time_seconds: eta指数積分時間。単位はs。

    Returns:
        配列shape、dtype、値とscalar条件を含む64桁SHA-256文字列。

    Raises:
        ValueError: 配列shape、有限性、scalar範囲が不正な場合。

    境界条件:
        配列値をfloat32へ正規化してhashするため、runtimeと校正側は同じfloat32表を使う。
    """

    positions = np.asarray(sensor_positions_m, dtype=np.float32)
    frequencies = np.asarray(frequency_hz, dtype=np.float32)
    azimuth = np.asarray(direction_azimuth_deg, dtype=np.float32)
    weights = np.asarray(channel_weight_table, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("sensor_positions_m must have shape (n_ch, 3).")
    if frequencies.ndim != 1 or azimuth.ndim != 1:
        raise ValueError("frequency_hz and direction_azimuth_deg must be one-dimensional.")
    if weights.shape != (positions.shape[0], frequencies.size):
        raise ValueError("channel_weight_table must have shape (n_ch, n_bin).")
    if not bool(np.all(np.isfinite(positions))) or not bool(np.all(np.isfinite(frequencies))):
        raise ValueError("positions and frequencies must be finite.")
    if not bool(np.all(np.isfinite(azimuth))) or not bool(np.all(np.isfinite(weights))):
        raise ValueError("azimuth and channel weights must be finite.")
    if float(sound_speed_m_s) <= 0.0 or int(snapshot_length_samples) <= 0 or float(integration_time_seconds) <= 0.0:
        raise ValueError("sound speed, snapshot length, and integration time must be positive.")

    digest = hashlib.sha256()
    for name, array in (
        ("positions", positions),
        ("frequency", frequencies),
        ("azimuth", azimuth),
        ("weights", weights),
    ):
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    digest.update(np.asarray([sound_speed_m_s, integration_time_seconds], dtype=np.float64).tobytes())
    digest.update(np.asarray([snapshot_length_samples], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _roc_auc(noise: NDArray[np.float32], target: NDArray[np.float32]) -> float:
    """全thresholdの経験ROCを台形積分しAUCを返す。"""

    thresholds = np.unique(np.concatenate((noise, target)))[::-1]
    thresholds = np.concatenate((np.array([np.inf]), thresholds, np.array([-np.inf])))
    false_positive_rate = np.asarray([np.mean(noise >= value) for value in thresholds], dtype=np.float64)
    detection_rate = np.asarray([np.mean(target >= value) for value in thresholds], dtype=np.float64)
    return float(np.trapezoid(detection_rate, false_positive_rate))


def calibrate_steering_power_thresholds(
    noise_eta: NDArray[Any],
    target_eta: NDArray[Any],
    *,
    effective_channel_count: NDArray[Any],
    active_channel_count: NDArray[Any],
    configuration_signature: str,
    false_positive_rate_target: float = 0.01,
    target_lower_quantile: float = 0.10,
    minimum_threshold_gap: float = 0.02,
) -> SteeringPowerThresholdCalibration:
    """noise/target eta sampleから周波数別soft Weight thresholdを校正する。

    Args:
        noise_eta: noise-only eta。shapeは`[n_observation,n_direction,n_bin]`。
        target_eta: target位置近傍のeta。shapeは`[n_observation,n_target_direction,n_bin]`。
        effective_channel_count: shadingを含む`N_eff`。shapeは`[n_bin]`。
        active_channel_count: 正のshading係数を持つchannel数。shapeは`[n_bin]`。
        configuration_signature: 校正条件を固定したSHA-256文字列。
        false_positive_rate_target: `gamma_off`を決めるnoise上側確率。範囲は`(0,1)`。
        target_lower_quantile: `gamma_on`候補に使うtarget下側quantile。範囲は`(0,1)`。
        minimum_threshold_gap: `gamma_on-gamma_off`の最小値。範囲は`(0,1)`。

    Returns:
        周波数別threshold、noise理論差、ROC、FPR/TPR、channel構成を持つ校正結果。

    Raises:
        ValueError: sample shape、範囲、有限性、channel profile、quantile設定が不正な場合。

    境界条件:
        `gamma_off+gap`が1を超えるbinではgamma_onを1、gamma_offを`1-gap`へ下げ、
        runtimeが要求する厳密な`gamma_off<gamma_on`を維持する。
    """

    noise = np.asarray(noise_eta, dtype=np.float32)
    target = np.asarray(target_eta, dtype=np.float32)
    n_eff = np.asarray(effective_channel_count, dtype=np.float32)
    active = np.asarray(active_channel_count, dtype=np.int32)
    if noise.ndim != 3 or target.ndim != 3 or noise.shape[2] != target.shape[2]:
        raise ValueError("noise_eta and target_eta must have shape (n_observation, n_direction, n_bin).")
    n_bin = int(noise.shape[2])
    if noise.shape[0] == 0 or noise.shape[1] == 0 or target.shape[0] == 0 or target.shape[1] == 0:
        raise ValueError("noise_eta and target_eta must contain observations and directions.")
    if n_eff.shape != (n_bin,) or active.shape != (n_bin,):
        raise ValueError("channel profiles must have shape (n_bin,).")
    if not bool(np.all(np.isfinite(noise))) or not bool(np.all(np.isfinite(target))):
        raise ValueError("eta samples must be finite.")
    if not bool(np.all((noise >= 0.0) & (noise <= 1.0))) or not bool(np.all((target >= 0.0) & (target <= 1.0))):
        raise ValueError("eta samples must be in [0, 1].")
    if not bool(np.all(np.isfinite(n_eff))) or not bool(np.all(n_eff > 0.0)) or not bool(np.all(active > 0)):
        raise ValueError("effective and active channel counts must be positive.")
    fpr_target = float(false_positive_rate_target)
    target_quantile = float(target_lower_quantile)
    gap = float(minimum_threshold_gap)
    if not 0.0 < fpr_target < 1.0 or not 0.0 < target_quantile < 1.0 or not 0.0 < gap < 1.0:
        raise ValueError("calibration probabilities and minimum gap must be in (0, 1).")
    if not configuration_signature:
        raise ValueError("configuration_signature must not be empty.")

    # observationとdirectionを同じsample軸へ畳み、周波数binごとに分布を校正する。
    noise_by_bin = noise.reshape((-1, n_bin))
    target_by_bin = target.reshape((-1, n_bin))
    gamma_off = np.quantile(noise_by_bin, 1.0 - fpr_target, axis=0).astype(np.float32)
    target_floor = np.quantile(target_by_bin, target_quantile, axis=0).astype(np.float32)
    gamma_off = np.minimum(gamma_off, np.float32(1.0 - gap))
    gamma_on = np.minimum(
        np.maximum(target_floor, gamma_off + np.float32(gap)),
        np.float32(1.0),
    ).astype(np.float32)
    noise_mean = np.mean(noise_by_bin, axis=0, dtype=np.float64).astype(np.float32)
    noise_standard_deviation = np.std(noise_by_bin, axis=0, dtype=np.float64).astype(np.float32)
    noise_reference = np.asarray(1.0 / n_eff, dtype=np.float32)
    roc_auc = np.asarray(
        [_roc_auc(noise_by_bin[:, index], target_by_bin[:, index]) for index in range(n_bin)],
        dtype=np.float32,
    )
    calibrated_fpr = np.mean(noise_by_bin > gamma_off[np.newaxis, :], axis=0).astype(np.float32)
    calibrated_tpr = np.mean(target_by_bin >= gamma_on[np.newaxis, :], axis=0).astype(np.float32)
    return SteeringPowerThresholdCalibration(
        gamma_off=gamma_off,
        gamma_on=gamma_on,
        noise_mean=noise_mean,
        noise_standard_deviation=noise_standard_deviation,
        noise_eta_reference=noise_reference,
        noise_reference_error=np.asarray(noise_mean - noise_reference, dtype=np.float32),
        effective_channel_count=n_eff.copy(),
        active_channel_count=active.copy(),
        roc_auc=roc_auc,
        calibrated_false_positive_rate=calibrated_fpr,
        calibrated_detection_rate=calibrated_tpr,
        configuration_signature=str(configuration_signature),
    )
