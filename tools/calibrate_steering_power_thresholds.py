"""記録済みeta sampleから実運用用steering power threshold JSONを生成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluations.beamforming.steering_power_threshold_calibration import (  # noqa: E402
    calculate_steering_power_calibration_signature,
    calibrate_steering_power_thresholds,
)


def _required_array(archive: Any, name: str) -> np.ndarray:
    """NPZから必須配列を読込み、欠落名が分かる例外を返す。"""

    if name not in archive.files:
        raise ValueError(f"input NPZ is missing required array: {name}")
    return np.asarray(archive[name])


def calibrate_threshold_file(
    input_npz_path: Path,
    output_json_path: Path,
    *,
    false_positive_rate_target: float = 0.01,
    target_lower_quantile: float = 0.10,
    minimum_threshold_gap: float = 0.02,
) -> dict[str, Any]:
    """校正sample NPZからruntimeへ渡す周波数別threshold JSONを生成する。

    Args:
        input_npz_path: 校正入力。必須配列はnoise/target eta、受波器座標、周波数軸、
            方位軸、shading、`N_eff`、active channel数、および処理scalar。
        output_json_path: 校正結果JSONの保存先。
        false_positive_rate_target: noise上側quantileの目標誤検出率。
        target_lower_quantile: `gamma_on`に使うtarget下側quantile。
        minimum_threshold_gap: `gamma_on-gamma_off`の最小値。

    Returns:
        JSONへ保存したpayload。threshold配列shapeは`[n_bin]`。

    Raises:
        ValueError: 必須配列欠落、shape、範囲、有限性、処理条件が不正な場合。

    境界条件:
        output親directoryは存在しない場合に作成する。既存fileは同じ入力条件の再校正を
        再現できるよう全体を書き換え、部分的な旧thresholdを残さない。
    """

    input_path = Path(input_npz_path)
    if not input_path.exists():
        raise ValueError("input_npz_path must exist.")
    with np.load(input_path, allow_pickle=False) as archive:
        noise_eta = _required_array(archive, "noise_eta")
        target_eta = _required_array(archive, "target_eta")
        sensor_positions_m = _required_array(archive, "sensor_positions_m")
        frequency_hz = _required_array(archive, "frequency_hz")
        direction_azimuth_deg = _required_array(archive, "direction_azimuth_deg")
        channel_weight_table = _required_array(archive, "channel_weight_table")
        effective_channel_count = _required_array(archive, "effective_channel_count")
        active_channel_count = _required_array(archive, "active_channel_count")
        sound_speed_m_s = float(_required_array(archive, "sound_speed_m_s").reshape(()))
        snapshot_length_samples = int(_required_array(archive, "snapshot_length_samples").reshape(()))
        integration_time_seconds = float(_required_array(archive, "integration_time_seconds").reshape(()))

    signature = calculate_steering_power_calibration_signature(
        sensor_positions_m=sensor_positions_m,
        frequency_hz=frequency_hz,
        direction_azimuth_deg=direction_azimuth_deg,
        channel_weight_table=channel_weight_table,
        sound_speed_m_s=sound_speed_m_s,
        snapshot_length_samples=snapshot_length_samples,
        integration_time_seconds=integration_time_seconds,
    )
    result = calibrate_steering_power_thresholds(
        noise_eta,
        target_eta,
        effective_channel_count=effective_channel_count,
        active_channel_count=active_channel_count,
        configuration_signature=signature,
        false_positive_rate_target=false_positive_rate_target,
        target_lower_quantile=target_lower_quantile,
        minimum_threshold_gap=minimum_threshold_gap,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "configuration_signature": result.configuration_signature,
        "source_npz_path": str(input_path.resolve()),
        "n_ch": int(sensor_positions_m.shape[0]),
        "n_bin": int(frequency_hz.size),
        "sound_speed_m_s": sound_speed_m_s,
        "snapshot_length_samples": snapshot_length_samples,
        "integration_time_seconds": integration_time_seconds,
        "false_positive_rate_target": float(false_positive_rate_target),
        "target_lower_quantile": float(target_lower_quantile),
        "minimum_threshold_gap": float(minimum_threshold_gap),
        "frequency_hz": np.asarray(frequency_hz, dtype=np.float32).tolist(),
        "active_channel_count": result.active_channel_count.tolist(),
        "effective_channel_count": result.effective_channel_count.tolist(),
        "noise_eta_reference": result.noise_eta_reference.tolist(),
        "noise_mean": result.noise_mean.tolist(),
        "noise_standard_deviation": result.noise_standard_deviation.tolist(),
        "noise_reference_error": result.noise_reference_error.tolist(),
        "gamma_off": result.gamma_off.tolist(),
        "gamma_on": result.gamma_on.tolist(),
        "roc_auc": result.roc_auc.tolist(),
        "calibrated_false_positive_rate": result.calibrated_false_positive_rate.tolist(),
        "calibrated_detection_rate": result.calibrated_detection_rate.tolist(),
    }
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    """CLI引数を読み、steering power threshold JSONを生成する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--false-positive-rate", type=float, default=0.01)
    parser.add_argument("--target-lower-quantile", type=float, default=0.10)
    parser.add_argument("--minimum-threshold-gap", type=float, default=0.02)
    args = parser.parse_args()
    calibrate_threshold_file(
        args.input_npz,
        args.output_json,
        false_positive_rate_target=args.false_positive_rate,
        target_lower_quantile=args.target_lower_quantile,
        minimum_threshold_gap=args.minimum_threshold_gap,
    )


if __name__ == "__main__":
    main()
