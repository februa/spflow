"""100 Hz帯エンドファイア信号を長大等間隔ULAで共分散評価する。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from scene_renderer import (  # noqa: E402
    AcousticSource,
    AmbientField,
    BandLimitedNoiseSpectrum,
    ConstantEnvelope,
    FreeField,
    Receiver,
    Scene,
    SceneRenderer,
    SourceComponent,
    StaticPose,
)
from spflow.beamforming import (  # noqa: E402
    DirectionMatchedCovarianceAccumulator,
    build_two_second_covariance_snapshot_schedule,
    calculate_covariance_subspace_metrics,
    calculate_sparse_array_spatial_correlation_statistics,
    relative_arrival_delay,
    steering_from_relative_delay,
)
from spflow.beamforming.diagnostic_plotting import require_matplotlib  # noqa: E402
from evaluations.beamforming.method3_sparse_64ch_correlation import CoordinateArray  # noqa: E402


OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "low_frequency_long_ula_covariance"
FS_HZ = 1024.0
NFFT = 128
SOURCE_LOW_HZ = 80.0
SOURCE_HIGH_HZ = 120.0
SOURCE_AZIMUTH_DEG = 0.0
SOUND_SPEED_M_S = 1500.0
N_CHANNEL = 64
# 120 Hzの半波長6.25 mに合わせ、80--120 Hz帯全体で空間aliasを避ける。
SPACING_M = SOUND_SPEED_M_S / (2.0 * SOURCE_HIGH_HZ)
APERTURE_M = SPACING_M * (N_CHANNEL - 1)
PROCESS_DURATION_S = 10
INTEGRATION_TIME_S = 10.0
NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ = -32.0


def _positions() -> NDArray[np.float32]:
    """64ch等間隔ULA座標`[n_ch,3]`をm単位で返す。"""

    positions = np.zeros((N_CHANNEL, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(-APERTURE_M / 2.0, APERTURE_M / 2.0, N_CHANNEL, dtype=np.float32)
    return positions


def _render(
    positions_m: NDArray[np.float32],
    *,
    noise: bool,
    seed: int,
    source_azimuth_deg: float = SOURCE_AZIMUTH_DEG,
) -> NDArray[np.float32]:
    """指定方位の80--120 Hz広帯域信号を生成する。

    Args:
        positions_m: 受波器位置。shapeは`[n_ch,3]`、単位はm。
        noise: 空間白色雑音を重畳する場合はTrue。
        seed: 信号と雑音の乱数seed。
        source_azimuth_deg: 信号到来方位。単位はdegree、範囲は0--180度。

    Returns:
        受波信号。shapeは`[n_ch,n_sample]`、dtypeはfloat32。

    Raises:
        ValueError: source方位が評価対象の0--180度外の場合。
    """

    if not 0.0 <= source_azimuth_deg <= 180.0:
        raise ValueError("source_azimuth_deg must be in [0, 180].")

    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=CoordinateArray(positions_m),
    )
    component = SourceComponent(
        spectrum=BandLimitedNoiseSpectrum(SOURCE_LOW_HZ, SOURCE_HIGH_HZ),
        envelope=ConstantEnvelope(),
        amplitude=None,
        level_db=0.0,
        noise_seed=seed,
        noise_filter_length=513,
    )
    source = AcousticSource.from_relative_bearing(
        bearing_deg=source_azimuth_deg,
        distance=1.0e6,
        receiver_pose=receiver.trajectory.pose(0.0),
        components=[component],
        elevation_deg=0.0,
    )
    ambient = []
    if noise:
        ambient.append(
            AmbientField.from_asd_level_db(
                BandLimitedNoiseSpectrum(0.0, FS_HZ / 2.0),
                NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ,
                covariance=np.eye(N_CHANNEL, dtype=np.float32),
                noise_seed=seed + 9000,
                noise_filter_length=513,
            )
        )
    scene = Scene(sources=[source], ambient_fields=ambient, environment=FreeField(c=SOUND_SPEED_M_S))
    time_s = np.arange(int(FS_HZ * PROCESS_DURATION_S), dtype=np.float64) / FS_HZ
    return np.asarray(np.real(SceneRenderer().render(scene, receiver, time_s)), dtype=np.float32)


def _summary(
    table: NDArray[np.float32],
    azimuth_deg: NDArray[np.float32],
    band: NDArray[np.bool_],
    *,
    lower_is_better: bool = False,
) -> dict[str, float]:
    """正解エンドファイアと20度以上遠方の指標を比較する。"""

    curve = np.asarray(np.mean(table[:, band], axis=1), dtype=np.float32)
    near = azimuth_deg <= 2.0
    far = azimuth_deg >= 20.0
    score = -curve if lower_is_better else curve
    peak_index = int(np.argmax(score))
    near_value = float(np.mean(curve[near]))
    far_value = float(np.mean(curve[far]))
    return {
        "source_neighborhood_mean": near_value,
        "far_20deg_mean": far_value,
        "source_far_margin": far_value - near_value if lower_is_better else near_value - far_value,
        "peak_azimuth_deg": float(azimuth_deg[peak_index]),
        "peak_error_deg": float(azimuth_deg[peak_index]),
        "far_extreme": float(np.min(curve[far]) if lower_is_better else np.max(curve[far])),
    }


def _write_curves(
    azimuth_deg: NDArray[np.float32],
    tables: dict[str, NDArray[np.float32]],
    band: NDArray[np.bool_],
) -> None:
    """共分散指標の帯域平均方位curveを同一図へ保存する。"""

    plt = require_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    for axis, name in zip(
        axes.flat,
        ("steering_eta", "correlation_mean", "principal_eigenvalue_fraction", "steering_rank_one_residual"),
        strict=True,
    ):
        axis.plot(azimuth_deg, np.mean(tables[name][:, band], axis=1))
        axis.axvline(SOURCE_AZIMUTH_DEG, color="tab:red", linestyle="--", linewidth=1.0)
        axis.set(title=name, xlabel="Azimuth [deg]", ylabel="Ratio", xlim=(0.0, 180.0), ylim=(0.0, 1.0))
        axis.grid(True, alpha=0.25)
    figure.savefig(OUTPUT_DIR / "covariance_direction_curves.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """target-onlyとtarget+noiseを10秒積分し、長大ULA共分散を評価する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    positions = _positions()
    schedule = build_two_second_covariance_snapshot_schedule(
        positions,
        fs_hz=FS_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        snapshot_length_samples=NFFT,
        beams_per_half=159,
    )
    frequency_hz = np.asarray(np.fft.rfftfreq(NFFT, d=1.0 / FS_HZ), dtype=np.float32)
    azimuth_deg = schedule.global_direction_azimuth_deg
    azimuth_rad = np.deg2rad(azimuth_deg.astype(np.float64))
    directions = np.stack((np.cos(azimuth_rad), np.sin(azimuth_rad), np.zeros_like(azimuth_rad)), axis=1)
    delays_s = relative_arrival_delay(positions, directions, sound_speed_m_per_s=SOUND_SPEED_M_S)
    steering = np.transpose(steering_from_relative_delay(delays_s, frequency_hz), (0, 2, 1)).astype(np.complex64)
    band = (frequency_hz >= SOURCE_LOW_HZ) & (frequency_hz <= SOURCE_HIGH_HZ)
    central_mask = np.zeros(N_CHANNEL, dtype=np.bool_)
    central_mask[N_CHANNEL // 4 : 3 * N_CHANNEL // 4] = True
    scenes: dict[str, NDArray[np.float32]] = {
        "target_only": _render(positions, noise=False, seed=9700),
        "target_plus_noise": _render(positions, noise=True, seed=9800),
    }
    results: dict[str, dict[str, Any]] = {}
    for scene_name, signal in scenes.items():
        accumulator = DirectionMatchedCovarianceAccumulator(
            schedule,
            integration_time_seconds=INTEGRATION_TIME_S,
            steering_table=steering,
        )
        elapsed: list[float] = []
        for second in range(PROCESS_DURATION_S):
            start = time.perf_counter()
            accumulator.process_one_second(signal[:, second * int(FS_HZ) : (second + 1) * int(FS_HZ)])
            elapsed.append(time.perf_counter() - start)
        covariance = accumulator.direction_covariance
        subspace = calculate_covariance_subspace_metrics(covariance, np.transpose(steering, (0, 2, 1)))
        correlation = calculate_sparse_array_spatial_correlation_statistics(
            covariance,
            positions,
            frequency_hz,
            central_mask,
            sound_speed_m_s=SOUND_SPEED_M_S,
            physical_baseline_edges_m=np.linspace(0.0, APERTURE_M + SPACING_M, 65, dtype=np.float32),
            wavelength_normalized_edges=np.linspace(
                0.0,
                APERTURE_M * float(frequency_hz[-1]) / SOUND_SPEED_M_S + 1.0,
                65,
                dtype=np.float32,
            ),
        )
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
        covariance_band = np.moveaxis(covariance[:, :, :, band], 3, 1)
        covariance_hermitian = 0.5 * (covariance_band + np.swapaxes(covariance_band.conj(), -1, -2))
        eigenvalues = np.linalg.eigvalsh(covariance_hermitian)
        hermitian_error = np.linalg.norm(
            covariance_band - np.swapaxes(covariance_band.conj(), -1, -2)
        ) / max(float(np.linalg.norm(covariance_band)), np.finfo(np.float32).tiny)
        results[scene_name] = {
            "processing_seconds_per_input_second": float(np.mean(elapsed)),
            "hermitian_relative_error": float(hermitian_error),
            "minimum_eigenvalue_in_band": float(np.min(eigenvalues)),
            "maximum_eigenvalue_in_band": float(np.max(eigenvalues)),
            "metrics": {
                name: _summary(
                    table,
                    azimuth_deg,
                    band,
                    lower_is_better=name == "steering_rank_one_residual",
                )
                for name, table in tables.items()
            },
        }
        np.savez_compressed(  # pyright: ignore[reportArgumentType]
            OUTPUT_DIR / f"{scene_name}.npz",
            azimuth_deg=azimuth_deg,
            frequency_hz=frequency_hz,
            sensor_positions_m=positions,
            direction_covariance=covariance,
            **tables,  # pyright: ignore[reportArgumentType]
        )
        if scene_name == "target_only":
            _write_curves(azimuth_deg, tables, band)

    endfire_local_indices = np.flatnonzero(schedule.direction_match_indices[0] == 0)
    theoretical_end_delay_ms = float(1.0e3 * (delays_s[-1, 0] - delays_s[0, 0]))
    center_end_delay_ms = [
        float(
            1.0e3
            * (
                schedule.channel_center_samples[0, -1, local_index]
                - schedule.channel_center_samples[0, 0, local_index]
            )
            / FS_HZ
        )
        for local_index in endfire_local_indices
    ]
    payload = {
        "evaluation_pattern": "sparse_array_design / covariance directionality",
        "array": {
            "type": "uniform_linear_array",
            "n_channel": N_CHANNEL,
            "spacing_m": SPACING_M,
            "aperture_m": APERTURE_M,
            "positions_m": positions.tolist(),
        },
        "signal": {
            "source_azimuth_deg": SOURCE_AZIMUTH_DEG,
            "source_band_hz": [SOURCE_LOW_HZ, SOURCE_HIGH_HZ],
            "source_level_db_re_input_rms": 0.0,
            "noise_level_db_re_input_rms_per_sqrt_hz": NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ,
            "sample_rate_hz": FS_HZ,
            "nfft": NFFT,
            "block_duration_s": NFFT / FS_HZ,
            "integration_time_s": INTEGRATION_TIME_S,
        },
        "delay_contract": {
            "theoretical_endfire_end_to_end_delay_ms": theoretical_end_delay_ms,
            "center_table_end_to_end_delay_ms": center_end_delay_ms,
            "maximum_absolute_error_samples": float(
                np.max(
                    np.abs(
                        np.asarray(center_end_delay_ms) * FS_HZ / 1.0e3
                        - theoretical_end_delay_ms * FS_HZ / 1.0e3
                    )
                )
            ),
        },
        "scene_results": results,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
