"""方式3の方位別共分散が持つ方位選択性を、Weight決定前に切り分ける。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from spflow.beamforming import (  # noqa: E402
    DirectionMatchedCovarianceAccumulator,
    calculate_covariance_subspace_metrics,
    calculate_sparse_array_spatial_correlation_statistics,
)
from spflow.beamforming.diagnostic_plotting import centers_to_edges, require_matplotlib  # noqa: E402
from evaluations.beamforming.method3_direction_selection import (  # noqa: E402
    FS_HZ,
    NFFT,
    NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ,
    SOUND_SPEED_M_S,
    SOURCE_HIGH_HZ,
    SOURCE_LOW_HZ,
    _geometry,
    _render_segment,
)
from evaluations.beamforming.method3_sparse_64ch_correlation import (  # noqa: E402
    ADJACENT_SPACING_M,
    ARRAY_APERTURE_M,
    ARRAY_POSITIONS_M,
    CENTRAL_CHANNEL_MASK,
    MAXIMUM_SPACING_M,
    MINIMUM_SPACING_M,
    PHYSICAL_BASELINE_EDGES_M,
    WAVELENGTH_NORMALIZED_EDGES,
)


OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "method3_covariance_directionality"
SOURCE_BEARING_DEG = 40.0
INTEGRATION_TIME_S = 10.0
PROCESS_DURATION_S = 10


def _write_heatmap(path: Path, azimuth: NDArray[np.float32], frequency: NDArray[np.float32], table: NDArray[np.float32], title: str, *, value_range: tuple[float, float]) -> None:
    """方位・周波数metricを全scene共通軸・共通色範囲で保存する。"""

    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    image = axis.pcolormesh(
        centers_to_edges(azimuth.astype(np.float64)),
        centers_to_edges(frequency.astype(np.float64)),
        table.T,
        shading="flat",
        vmin=value_range[0],
        vmax=value_range[1],
        cmap="viridis",
    )
    axis.axvline(SOURCE_BEARING_DEG, color="tab:red", linestyle="--", linewidth=1.0)
    axis.set(xlabel="Azimuth [deg]", ylabel="Frequency [Hz]", title=title, ylim=(0.0, 1500.0))
    figure.colorbar(image, ax=axis, label="Ratio")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_baseline_heatmap(path: Path, azimuth: NDArray[np.float32], baseline: NDArray[np.float32], table: NDArray[np.float32], title: str, baseline_label: str) -> None:
    """信号帯域平均した方位・基線別相関を保存する。"""

    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    image = axis.pcolormesh(
        centers_to_edges(azimuth.astype(np.float64)),
        centers_to_edges(baseline.astype(np.float64)),
        table.T,
        shading="flat",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axis.axvline(SOURCE_BEARING_DEG, color="tab:red", linestyle="--", linewidth=1.0)
    axis.set(xlabel="Azimuth [deg]", ylabel=baseline_label, title=title)
    figure.colorbar(image, ax=axis, label="Mean normalized correlation")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _direction_summary(table: NDArray[np.float32], azimuth: NDArray[np.float32], band: NDArray[np.bool_], *, lower_is_better: bool = False) -> dict[str, float]:
    """正解近傍、遠方、peak誤差、幅を同じ規約で数値化する。"""

    curve = np.nanmean(table[:, band], axis=1)
    score = -curve if lower_is_better else curve
    source_mask = np.abs(azimuth - SOURCE_BEARING_DEG) <= 2.0
    far_mask = np.abs(azimuth - SOURCE_BEARING_DEG) >= 20.0
    peak_index = int(np.nanargmax(score))
    source_value = float(np.interp(SOURCE_BEARING_DEG, azimuth, curve))
    near_value = float(np.nanmean(curve[source_mask]))
    far_value = float(np.nanmean(curve[far_mask]))
    margin = far_value - near_value if lower_is_better else near_value - far_value
    outside_peak = float(np.nanmin(curve[far_mask]) if lower_is_better else np.nanmax(curve[far_mask]))
    # ratio指標ではpeakからbackgroundまでの半値を幅境界とし、dB powerでない指標を-3 dBと誤記しない。
    background = float(np.nanmean(score[far_mask]))
    half_height = background + 0.5 * (float(score[peak_index]) - background)
    selected = score >= half_height
    left = peak_index
    right = peak_index
    while left > 0 and bool(selected[left - 1]):
        left -= 1
    while right + 1 < selected.size and bool(selected[right + 1]):
        right += 1
    return {
        "source_value": source_value,
        "source_plus_minus_2deg_mean": near_value,
        "far_20deg_mean": far_value,
        "source_far_margin": margin,
        "peak_azimuth_deg": float(azimuth[peak_index]),
        "peak_error_deg": float(abs(float(azimuth[peak_index]) - SOURCE_BEARING_DEG),),
        "half_height_width_deg": float(azimuth[right] - azimuth[left]),
        "far_extreme": outside_peak,
    }


def _benchmark_equivalent_quadratic_forms() -> dict[str, float | int]:
    """完成共分散二次形式とsnapshot直接射影の同値性・実測時間を比較する。"""

    generator = np.random.default_rng(9150)
    n_direction, n_bin, n_ch = 159, 65, 64
    spectrum = (generator.standard_normal((n_direction, n_bin, n_ch)) + 1j * generator.standard_normal((n_direction, n_bin, n_ch))).astype(np.complex64)
    steering = (generator.standard_normal((n_direction, n_bin, n_ch)) + 1j * generator.standard_normal((n_direction, n_bin, n_ch))).astype(np.complex64)
    steering /= np.sqrt(np.sum(np.abs(steering) ** 2, axis=-1, keepdims=True)).astype(np.float32)
    covariance = np.einsum("...i,...j->...ij", spectrum, spectrum.conj(), optimize=True).astype(np.complex64)

    def measure_covariance() -> tuple[float, NDArray[np.float32]]:
        start = time.perf_counter()
        value = np.real(np.einsum("...i,...ij,...j->...", steering.conj(), covariance, steering, optimize=True)).astype(np.float32)
        return time.perf_counter() - start, value

    def measure_direct() -> tuple[float, NDArray[np.float32]]:
        start = time.perf_counter()
        value = np.abs(np.einsum("...i,...i->...", steering.conj(), spectrum, optimize=True)) ** 2
        return time.perf_counter() - start, value.astype(np.float32)

    covariance_trials = [measure_covariance() for _ in range(5)]
    direct_trials = [measure_direct() for _ in range(5)]
    np.testing.assert_allclose(covariance_trials[-1][1], direct_trials[-1][1], rtol=2.0e-4, atol=2.0e-3)
    return {
        "covariance_quadratic_median_s": float(np.median([item[0] for item in covariance_trials])),
        "direct_projection_median_s": float(np.median([item[0] for item in direct_trials])),
        "covariance_quadratic_max_s": float(np.max([item[0] for item in covariance_trials])),
        "direct_projection_max_s": float(np.max([item[0] for item in direct_trials])),
        "theoretical_covariance_complex_mac": n_direction * n_bin * n_ch * n_ch,
        "theoretical_direct_complex_mac": n_direction * n_bin * n_ch,
        "temporary_covariance_bytes": int(covariance.nbytes),
        "temporary_direct_projection_bytes": int(n_direction * n_bin * np.dtype(np.complex64).itemsize),
    }


def main() -> None:
    """6 sceneを分離処理し、Weight確定前の観測量と成果物を生成する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    schedule, frequency_hz, steering = _geometry()
    azimuth = schedule.global_direction_azimuth_deg.astype(np.float32)
    band = (frequency_hz >= SOURCE_LOW_HZ) & (frequency_hz <= SOURCE_HIGH_HZ)
    # channel index距離ではなく物理的に近いchannelほど強く相関するToeplitz雑音を別sceneに用いる。
    channel_index = np.arange(64, dtype=np.int32)
    correlated_noise_covariance = np.power(0.85, np.abs(channel_index[:, None] - channel_index[None, :])).astype(np.float32)
    scenes = {
        "noise_only": (_render_segment(PROCESS_DURATION_S, (), noise=True, seed=1100), False),
        "tone_target_only": (_render_segment(PROCESS_DURATION_S, (40.0,), noise=False, seed=2100, tone_frequency_hz=512.0), True),
        "broadband_target_only": (_render_segment(PROCESS_DURATION_S, (40.0,), noise=False, seed=3100), True),
        "broadband_white_noise": (_render_segment(PROCESS_DURATION_S, (40.0,), noise=True, seed=4100), True),
        "broadband_correlated_noise": (_render_segment(PROCESS_DURATION_S, (40.0,), noise=True, seed=5100, ambient_covariance=correlated_noise_covariance), True),
        "multiple_signals": (_render_segment(PROCESS_DURATION_S, (40.0, 75.0), noise=True, seed=6100), True),
    }
    scene_results: dict[str, dict[str, Any]] = {}
    trace_maximum = 0.0
    cached: dict[str, tuple[Any, Any, NDArray[np.bool_]]] = {}
    for scene_name, (signal, has_source) in scenes.items():
        accumulator = DirectionMatchedCovarianceAccumulator(
            schedule,
            integration_time_seconds=INTEGRATION_TIME_S,
            steering_table=steering,
        )
        elapsed: list[float] = []
        eta_history: list[NDArray[np.float32]] = []
        for second in range(PROCESS_DURATION_S):
            start = time.perf_counter()
            accumulator.process_one_second(signal[:, second * int(FS_HZ) : (second + 1) * int(FS_HZ)])
            elapsed.append(time.perf_counter() - start)
            if (second + 1) % 2 == 0:
                eta_history.append(accumulator.completed_steering_metrics().eta.copy())
        covariance = accumulator.direction_covariance.copy()
        subspace = calculate_covariance_subspace_metrics(covariance, np.transpose(steering, (0, 2, 1)))
        correlation = calculate_sparse_array_spatial_correlation_statistics(
            covariance,
            ARRAY_POSITIONS_M,
            frequency_hz,
            CENTRAL_CHANNEL_MASK,
            sound_speed_m_s=SOUND_SPEED_M_S,
            physical_baseline_edges_m=PHYSICAL_BASELINE_EDGES_M,
            wavelength_normalized_edges=WAVELENGTH_NORMALIZED_EDGES,
        )
        # toneは信号が存在する512 Hz binだけで方位選択性を評価し、無信号binで希釈しない。
        evaluation_band = (
            np.isclose(frequency_hz, np.float32(512.0))
            if scene_name == "tone_target_only"
            else band
        )
        cached[scene_name] = (subspace, correlation, evaluation_band)
        trace_maximum = max(trace_maximum, float(np.max(subspace.trace_power)))
        metric_tables = {
            "trace": subspace.trace_power,
            "correlation_mean": correlation.global_statistics.mean,
            "correlation_median": correlation.global_statistics.median,
            "correlation_percentile_95": correlation.global_statistics.percentile_95,
            "correlation_maximum": correlation.global_statistics.maximum,
            "steering_eta_covariance": subspace.steering_power_fraction,
            "steering_eta_direct": eta_history[-1],
            "principal_eigenvalue_fraction": subspace.principal_eigenvalue_fraction,
            "principal_eigenvalue_gap_fraction": subspace.principal_eigenvalue_gap_fraction,
            "steering_rank_one_residual": subspace.steering_rank_one_residual,
        }
        np.savez_compressed(  # pyright: ignore[reportArgumentType]
            OUTPUT_DIR / f"{scene_name}.npz",
            azimuth_deg=azimuth,
            frequency_hz=frequency_hz,
            eta_by_completed_cycle=np.stack(eta_history),
            **metric_tables,  # pyright: ignore[reportArgumentType]
            physical_baseline_mean=correlation.physical_baseline.mean,
            physical_baseline_median=correlation.physical_baseline.median,
            physical_baseline_percentile_95=correlation.physical_baseline.percentile_95,
            physical_baseline_pair_count=correlation.physical_baseline.pair_count,
            physical_baseline_minimum=correlation.physical_baseline.value_minimum,
            physical_baseline_maximum=correlation.physical_baseline.value_maximum,
            physical_baseline_representative=correlation.physical_baseline.value_representative,
            physical_baseline_standard_deviation=correlation.physical_baseline.standard_deviation,
            physical_baseline_interquartile_range=correlation.physical_baseline.interquartile_range,
            wavelength_normalized_mean=correlation.wavelength_normalized_baseline.mean,
            pair_composition_mean=correlation.pair_composition.mean,
            pair_composition_median=correlation.pair_composition.median,
            pair_composition_percentile_95=correlation.pair_composition.percentile_95,
        )
        scene_results[scene_name] = {
            "has_source": has_source,
            "processing_mean_s_per_input_second": float(np.mean(elapsed)),
            "processing_max_s_per_input_second": float(np.max(elapsed)),
            "real_time_ratio": float(np.sum(elapsed) / PROCESS_DURATION_S),
            "eta_direct_covariance_max_abs_error": float(np.max(np.abs(eta_history[-1] - subspace.steering_power_fraction))),
            "metrics": {
                name: _direction_summary(
                    table,
                    azimuth,
                    evaluation_band,
                    lower_is_better=name == "steering_rank_one_residual",
                )
                for name, table in metric_tables.items()
            },
        }

    # traceだけは全sceneで得た絶対最大値を共通上限とし、scene間のpower差を視覚的に保持する。
    for scene_name, (subspace, correlation, evaluation_band) in cached.items():
        tables = {
            "trace": subspace.trace_power / np.float32(max(trace_maximum, np.finfo(np.float32).tiny)),
            "correlation_mean": correlation.global_statistics.mean,
            "correlation_median": correlation.global_statistics.median,
            "correlation_percentile_95": correlation.global_statistics.percentile_95,
            "steering_eta": subspace.steering_power_fraction,
            "principal_eigenvalue_fraction": subspace.principal_eigenvalue_fraction,
            "principal_eigenvalue_gap_fraction": subspace.principal_eigenvalue_gap_fraction,
            "steering_rank_one_residual": subspace.steering_rank_one_residual,
        }
        for name, table in tables.items():
            _write_heatmap(OUTPUT_DIR / f"{scene_name}_{name}.png", azimuth, frequency_hz, table, f"{scene_name}: {name}", value_range=(0.0, 1.0))
        physical_group_valid = np.any(
            correlation.physical_baseline.pair_count[evaluation_band] > 0,
            axis=0,
        )
        physical = np.asarray(
            np.mean(
                correlation.physical_baseline.mean[:, evaluation_band, :][:, :, physical_group_valid],
                axis=1,
            ),
            dtype=np.float32,
        )
        # 代表値は周波数ごとのpair構成で揺れるため、画像軸には定義済みbin境界の中点を使う。
        physical_axis = np.asarray(
            0.5
            * (
                correlation.physical_baseline.group_edges[:-1]
                + correlation.physical_baseline.group_edges[1:]
            )[physical_group_valid],
            dtype=np.float32,
        )
        _write_baseline_heatmap(OUTPUT_DIR / f"{scene_name}_physical_baseline.png", azimuth, physical_axis, physical, f"{scene_name}: physical baseline", "Physical baseline [m]")
        normalized_group_valid = np.any(
            correlation.wavelength_normalized_baseline.pair_count[evaluation_band] > 0,
            axis=0,
        )
        normalized = np.asarray(
            np.mean(
                correlation.wavelength_normalized_baseline.mean[:, evaluation_band, :][
                    :, :, normalized_group_valid
                ],
                axis=1,
            ),
            dtype=np.float32,
        )
        normalized_axis = np.asarray(
            0.5
            * (
                correlation.wavelength_normalized_baseline.group_edges[:-1]
                + correlation.wavelength_normalized_baseline.group_edges[1:]
            )[normalized_group_valid],
            dtype=np.float32,
        )
        _write_baseline_heatmap(OUTPUT_DIR / f"{scene_name}_wavelength_baseline.png", azimuth, normalized_axis, normalized, f"{scene_name}: d/lambda", "Normalized baseline d/lambda")

    configuration = {
        "sample_rate_hz": FS_HZ,
        "snapshot_length_samples": NFFT,
        "snapshot_duration_s": NFFT / FS_HZ,
        "integration_time_s": INTEGRATION_TIME_S,
        "process_duration_s": PROCESS_DURATION_S,
        "source_bearing_deg": SOURCE_BEARING_DEG,
        "source_band_hz": [SOURCE_LOW_HZ, SOURCE_HIGH_HZ],
        "source_level_db_re_input_rms": 0.0,
        "noise_level_db_re_input_rms_per_sqrt_hz": NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ,
        "sound_speed_m_s": SOUND_SPEED_M_S,
        "random_seeds": [1100, 2100, 3100, 4100, 5100, 6100],
        "array_positions_m": ARRAY_POSITIONS_M.tolist(),
        "n_channel": 64,
        "aperture_m": ARRAY_APERTURE_M,
        "minimum_adjacent_spacing_m": MINIMUM_SPACING_M,
        "maximum_adjacent_spacing_m": MAXIMUM_SPACING_M,
        "adjacent_spacing_m": ADJACENT_SPACING_M.tolist(),
        "normal_direction_update_rate_hz": 0.5,
        "effective_snapshots_at_10s": 5.0,
    }
    payload = {
        "configuration": configuration,
        "quadratic_form_benchmark": _benchmark_equivalent_quadratic_forms(),
        "scene_results": scene_results,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
