"""開口長に対する広帯域共通信号のsteering方位選択性を評価する。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from spflow.beamforming import (
    DirectionMatchedCovarianceAccumulator,
    build_two_second_covariance_snapshot_schedule,
    calculate_covariance_subspace_metrics,
    calculate_sparse_array_spatial_correlation_statistics,
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
from evaluations.beamforming.method3_sparse_64ch_correlation import (
    ARRAY_POSITIONS_M,
    CENTRAL_CHANNEL_MASK,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "aperture_steering_directionality"
APERTURE_M = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 320.0], dtype=np.float32)
# 320 mでは0.1度程度の方位量子化誤差も大きな端間位相誤差になる。
# 開口長の効果だけを比較するため、159方位表上の40度最近点を信号真値とする。
SOURCE_AZIMUTH_DEG = float(np.linspace(0.0, 180.0, 159, dtype=np.float32)[35])
PROCESS_DURATION_S = 10
INTEGRATION_TIME_S = 10.0
PLANE_WAVE_SOURCE_DISTANCE_M = 1.0e6


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


def _direction_summary(
    table: NDArray[np.float32],
    azimuth_deg: NDArray[np.float32],
    band: NDArray[np.bool_],
    *,
    lower_is_better: bool = False,
) -> dict[str, float]:
    """方位・周波数tableから正解近傍と遠方の統計を求める。"""

    curve = np.asarray(np.mean(table[:, band], axis=1), dtype=np.float32)
    source_mask = np.abs(azimuth_deg - SOURCE_AZIMUTH_DEG) <= 2.0
    far_mask = np.abs(azimuth_deg - SOURCE_AZIMUTH_DEG) >= 20.0
    score = -curve if lower_is_better else curve
    peak_index = int(np.argmax(score))
    near = float(np.mean(curve[source_mask]))
    far = float(np.mean(curve[far_mask]))
    return {
        "source_neighborhood_mean": near,
        "far_20deg_mean": far,
        "source_far_margin": far - near if lower_is_better else near - far,
        "peak_azimuth_deg": float(azimuth_deg[peak_index]),
        "peak_error_deg": abs(float(azimuth_deg[peak_index]) - SOURCE_AZIMUTH_DEG),
        "far_extreme": float(np.min(curve[far_mask]) if lower_is_better else np.max(curve[far_mask])),
    }


def _covariance_diagnostics(
    covariance: NDArray[np.complex64],
    steering: NDArray[np.complex64],
    positions_m: NDArray[np.float32],
    azimuth_deg: NDArray[np.float32],
    frequency_hz: NDArray[np.float32],
    band: NDArray[np.bool_],
) -> tuple[dict[str, object], dict[str, NDArray[np.float32]]]:
    """32 mと320 mの方位別共分散を同じ指標で評価する。"""

    subspace = calculate_covariance_subspace_metrics(
        covariance,
        np.transpose(steering, (0, 2, 1)),
    )
    aperture_m = float(np.ptp(positions_m[:, 0]))
    correlation = calculate_sparse_array_spatial_correlation_statistics(
        covariance,
        positions_m,
        frequency_hz,
        CENTRAL_CHANNEL_MASK,
        sound_speed_m_s=SOUND_SPEED_M_S,
        physical_baseline_edges_m=np.linspace(0.0, aperture_m + aperture_m / 64.0, 66, dtype=np.float32),
        wavelength_normalized_edges=np.linspace(
            0.0,
            aperture_m * float(frequency_hz[-1]) / SOUND_SPEED_M_S + 1.0,
            65,
            dtype=np.float32,
        ),
    )
    # covariance shapeは`[direction,ch,ch,bin]`。bandだけを`[direction,bin,ch,ch]`へ移し、
    # Hermitian化後の最小固有値でPSD性を評価する。
    covariance_band = np.moveaxis(covariance[:, :, :, band], 3, 1)
    covariance_hermitian = np.asarray(
        0.5 * (covariance_band + np.swapaxes(covariance_band.conj(), -1, -2)),
        dtype=np.complex64,
    )
    eigenvalues = np.linalg.eigvalsh(covariance_hermitian)
    hermitian_error = np.linalg.norm(
        covariance_band - np.swapaxes(covariance_band.conj(), -1, -2)
    ) / max(float(np.linalg.norm(covariance_band)), np.finfo(np.float32).tiny)
    tables = {
        "trace": subspace.trace_power,
        "correlation_mean": correlation.global_statistics.mean,
        "correlation_median": correlation.global_statistics.median,
        "correlation_percentile_95": correlation.global_statistics.percentile_95,
        "steering_eta": subspace.steering_power_fraction,
        "principal_eigenvalue_fraction": subspace.principal_eigenvalue_fraction,
        "principal_eigenvalue_gap_fraction": subspace.principal_eigenvalue_gap_fraction,
        "steering_rank_one_residual": subspace.steering_rank_one_residual,
    }
    summary: dict[str, object] = {
        "hermitian_relative_error": float(hermitian_error),
        "minimum_eigenvalue_in_band": float(np.min(eigenvalues)),
        "maximum_eigenvalue_in_band": float(np.max(eigenvalues)),
        "metrics": {
            name: _direction_summary(
                table,
                azimuth_deg,
                band,
                lower_is_better=name == "steering_rank_one_residual",
            )
            for name, table in tables.items()
        },
    }
    return summary, tables


def main() -> None:
    """スパース配置の相対形状を保って開口長sweepを実行する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_aperture_m = float(np.ptp(ARRAY_POSITIONS_M[:, 0]))
    records: list[dict[str, object]] = []
    curves: dict[float, NDArray[np.float32]] = {}
    covariance_evaluation: dict[str, dict[str, object]] = {}
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
            source_distance_m=PLANE_WAVE_SOURCE_DISTANCE_M,
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
        source_direction_index = int(
            np.argmin(np.abs(schedule.global_direction_azimuth_deg - SOURCE_AZIMUTH_DEG))
        )
        source_local_indices = np.flatnonzero(
            schedule.direction_match_indices[0] == source_direction_index
        )
        # source方位の2 snapshotについて、中心表が与える両端channel間時間差を求める。
        # 理論到来遅延との差は、中心表のscaleが同一波面を保っているかを示す。
        actual_end_delay_s = np.asarray(
            [
                (
                    schedule.channel_center_samples[0, -1, local_index]
                    - schedule.channel_center_samples[0, 0, local_index]
                )
                / FS_HZ
                for local_index in source_local_indices
            ],
            dtype=np.float64,
        )
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
                "center_table_end_to_end_delay_ms": (1.0e3 * actual_end_delay_s).tolist(),
                "center_table_delay_error_ms": (
                    1.0e3 * (np.abs(actual_end_delay_s) - source_end_to_end_delay_s)
                ).tolist(),
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
        if aperture_m in (32.0, 320.0):
            covariance_summary, covariance_tables = _covariance_diagnostics(
                accumulator.direction_covariance,
                steering,
                positions,
                azimuth,
                frequency_hz,
                band,
            )
            covariance_evaluation[str(int(aperture_m))] = covariance_summary
            np.savez_compressed(  # pyright: ignore[reportArgumentType]
                OUTPUT_DIR / f"covariance_{int(aperture_m)}m.npz",
                azimuth_deg=azimuth,
                frequency_hz=frequency_hz,
                sensor_positions_m=positions,
                direction_covariance=accumulator.direction_covariance,
                **covariance_tables,  # pyright: ignore[reportArgumentType]
            )
    _write_overlay(azimuth_axis, curves)
    payload = {
        "common_signal_definition": "same broadband waveform across channels after compensating physical arrival delay",
        "source_azimuth_deg": SOURCE_AZIMUTH_DEG,
        "source_band_hz": [SOURCE_LOW_HZ, SOURCE_HIGH_HZ],
        "snapshot_length_samples": NFFT,
        "snapshot_duration_ms": 1.0e3 * NFFT / FS_HZ,
        "integration_time_s": INTEGRATION_TIME_S,
        "source_distance_m": PLANE_WAVE_SOURCE_DISTANCE_M,
        "n_channel": int(ARRAY_POSITIONS_M.shape[0]),
        "records": records,
        "covariance_evaluation_32m_vs_320m": covariance_evaluation,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
