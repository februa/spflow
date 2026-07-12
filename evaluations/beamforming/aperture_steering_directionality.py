"""開口長に対する広帯域共通信号のsteering方位選択性を評価する。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from spflow.beamforming import (
    DirectionMatchedCovarianceAccumulator,
    build_two_second_covariance_snapshot_schedule,
    relative_arrival_delay,
    steering_from_relative_delay,
)
from spflow.beamforming.diagnostic_plotting import require_matplotlib

from evaluations.beamforming.method3_direction_selection import (
    FS_HZ,
    NFFT,
    SOUND_SPEED_M_S,
    SOURCE_HIGH_HZ,
    SOURCE_LOW_HZ,
    _render_segment,
)
from evaluations.beamforming.method3_sparse_64ch_correlation import ARRAY_POSITIONS_M


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "aperture_steering_directionality"
APERTURE_M = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=np.float32)
SOURCE_AZIMUTH_DEG = 40.0
PROCESS_DURATION_S = 10
INTEGRATION_TIME_S = 10.0


def _geometry(positions_m: NDArray[np.float32]):
    """開口長ごとの中心表と物理steeringを生成する。"""

    schedule = build_two_second_covariance_snapshot_schedule(
        positions_m,
        fs_hz=FS_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        snapshot_length_samples=NFFT,
        beams_per_half=159,
    )
    frequency_hz = np.asarray(np.fft.rfftfreq(NFFT, d=1.0 / FS_HZ), dtype=np.float32)
    azimuth_rad = np.deg2rad(schedule.global_direction_azimuth_deg.astype(np.float64))
    # 方位vector shapeは`[n_direction,3]`。x-y平面の水平方位を表す。
    directions = np.stack(
        (np.cos(azimuth_rad), np.sin(azimuth_rad), np.zeros_like(azimuth_rad)),
        axis=1,
    )
    # delay shapeは`[n_ch,n_direction]`、単位はs。exp(j2πfτ)で物理steeringへ変換する。
    delay_s = relative_arrival_delay(
        positions_m,
        directions,
        sound_speed_m_per_s=SOUND_SPEED_M_S,
    )
    steering = np.transpose(
        steering_from_relative_delay(delay_s, frequency_hz),
        (0, 2, 1),
    ).astype(np.complex64)
    return schedule, frequency_hz, steering


def _write_overlay(
    azimuth_deg: NDArray[np.float32],
    curves: dict[float, NDArray[np.float32]],
) -> None:
    """開口長ごとの帯域平均etaを同一軸で保存する。"""

    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    for aperture_m, curve in curves.items():
        axis.plot(azimuth_deg, curve, label=f"{aperture_m:g} m")
    axis.axvline(SOURCE_AZIMUTH_DEG, color="black", linestyle="--", linewidth=1.0)
    axis.set(
        xlabel="Azimuth [deg]",
        ylabel="Steering consistency eta",
        title="Aperture sweep: 128--1024 Hz broadband common signal",
        xlim=(0.0, 180.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    figure.savefig(OUTPUT_DIR / "aperture_eta_overlay.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """スパース配置の相対形状を保って開口長sweepを実行する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_aperture_m = float(np.ptp(ARRAY_POSITIONS_M[:, 0]))
    records: list[dict[str, float]] = []
    curves: dict[float, NDArray[np.float32]] = {}
    # scheduleのglobal方位軸は開口長によらず0--180度の159点で固定である。
    azimuth_axis = np.linspace(0.0, 180.0, 159, dtype=np.float32)
    for aperture_value in APERTURE_M:
        aperture_m = float(aperture_value)
        positions = np.asarray(ARRAY_POSITIONS_M * np.float32(aperture_m / base_aperture_m), dtype=np.float32)
        schedule, frequency_hz, steering = _geometry(positions)
        signal = _render_segment(
            PROCESS_DURATION_S,
            (SOURCE_AZIMUTH_DEG,),
            noise=False,
            seed=9300,
            array_positions_m=positions,
        )
        accumulator = DirectionMatchedCovarianceAccumulator(
            schedule,
            integration_time_seconds=INTEGRATION_TIME_S,
            steering_table=steering,
        )
        for second in range(PROCESS_DURATION_S):
            accumulator.process_one_second(
                signal[:, second * int(FS_HZ) : (second + 1) * int(FS_HZ)]
            )
        eta = accumulator.completed_steering_metrics().eta
        band = (frequency_hz >= SOURCE_LOW_HZ) & (frequency_hz <= SOURCE_HIGH_HZ)
        curve = np.asarray(np.mean(eta[:, band], axis=1), dtype=np.float32)
        curves[aperture_m] = curve
        azimuth = schedule.global_direction_azimuth_deg
        source_mask = np.abs(azimuth - SOURCE_AZIMUTH_DEG) <= 2.0
        far_mask = np.abs(azimuth - SOURCE_AZIMUTH_DEG) >= 20.0
        peak_index = int(np.argmax(curve))
        # 広帯域信号の自己相関時間の目安は1/B。端間遅延差がこれを超えるかを
        # 当該開口が方位差を観測できる物理条件の指標とする。
        coherence_time_s = 1.0 / (SOURCE_HIGH_HZ - SOURCE_LOW_HZ)
        source_end_to_end_delay_s = aperture_m * abs(np.cos(np.deg2rad(SOURCE_AZIMUTH_DEG))) / SOUND_SPEED_M_S
        mismatch_90_delay_s = aperture_m * abs(
            np.cos(np.deg2rad(SOURCE_AZIMUTH_DEG)) - np.cos(np.deg2rad(90.0))
        ) / SOUND_SPEED_M_S
        records.append(
            {
                "aperture_m": aperture_m,
                "source_end_to_end_delay_ms": 1.0e3 * source_end_to_end_delay_s,
                "source_to_90deg_mismatch_delay_ms": 1.0e3 * mismatch_90_delay_s,
                "coherence_time_estimate_ms": 1.0e3 * coherence_time_s,
                "mismatch_to_coherence_ratio": mismatch_90_delay_s / coherence_time_s,
                "source_neighborhood_eta": float(np.mean(curve[source_mask])),
                "far_20deg_eta": float(np.mean(curve[far_mask])),
                "source_far_margin": float(np.mean(curve[source_mask]) - np.mean(curve[far_mask])),
                "minimum_eta_over_direction": float(np.min(curve)),
                "maximum_eta_over_direction": float(np.max(curve)),
                "peak_azimuth_deg": float(azimuth[peak_index]),
                "peak_error_deg": abs(float(azimuth[peak_index]) - SOURCE_AZIMUTH_DEG),
            }
        )
    _write_overlay(azimuth_axis, curves)
    payload = {
        "common_signal_definition": "same broadband waveform across channels after compensating physical arrival delay",
        "source_azimuth_deg": SOURCE_AZIMUTH_DEG,
        "source_band_hz": [SOURCE_LOW_HZ, SOURCE_HIGH_HZ],
        "snapshot_length_samples": NFFT,
        "snapshot_duration_ms": 1.0e3 * NFFT / FS_HZ,
        "integration_time_s": INTEGRATION_TIME_S,
        "n_channel": int(ARRAY_POSITIONS_M.shape[0]),
        "records": records,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
