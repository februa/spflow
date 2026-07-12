"""方式3の直接eta積分、soft Weight、fallbackを64chスパースsceneで評価する。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
    ToneSpectrum,
)
from spflow.beamforming import (  # noqa: E402
    DirectionCovarianceSelectionConfig,
    DirectionMatchedCovarianceAccumulator,
    DirectionMatchedCovarianceSelector,
    build_two_second_covariance_snapshot_schedule,
    relative_arrival_delay,
    steering_from_relative_delay,
)
from spflow.beamforming.diagnostic_plotting import centers_to_edges, require_matplotlib  # noqa: E402
from evaluations.beamforming.method3_sparse_64ch_correlation import (  # noqa: E402
    ARRAY_POSITIONS_M,
    CoordinateArray,
)


FS_HZ = 8192.0
SOUND_SPEED_M_S = 1500.0
NFFT = 128
N_BEAM_PER_HALF = 159
SOURCE_LOW_HZ = 128.0
SOURCE_HIGH_HZ = 1024.0
NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ = -32.0
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "method3_direction_selection"


def _geometry():
    """固定schedule、周波数軸、事前計算steering tableを返す。"""

    schedule = build_two_second_covariance_snapshot_schedule(
        ARRAY_POSITIONS_M,
        fs_hz=FS_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        snapshot_length_samples=NFFT,
        beams_per_half=N_BEAM_PER_HALF,
    )
    frequency_hz = np.asarray(np.fft.rfftfreq(NFFT, d=1.0 / FS_HZ), dtype=np.float32)
    azimuth_rad = np.deg2rad(schedule.global_direction_azimuth_deg.astype(np.float64))
    directions = np.stack((np.cos(azimuth_rad), np.sin(azimuth_rad), np.zeros_like(azimuth_rad)), axis=1)
    delay_s = relative_arrival_delay(ARRAY_POSITIONS_M, directions, sound_speed_m_per_s=SOUND_SPEED_M_S)
    steering_ch_direction_bin = steering_from_relative_delay(delay_s, frequency_hz)
    steering_ch_bin_direction = np.transpose(steering_ch_direction_bin, (0, 2, 1)).astype(np.complex64)
    return schedule, frequency_hz, steering_ch_bin_direction


def _render_segment(
    duration_s: int,
    bearings_deg: tuple[float, ...],
    *,
    noise: bool,
    seed: int,
    tone_frequency_hz: float | None = None,
    ambient_covariance: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """scene_rendererで指定方位sourceと独立channel雑音を生成する。"""

    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=CoordinateArray(ARRAY_POSITIONS_M),
    )
    sources = []
    for source_index, bearing_deg in enumerate(bearings_deg):
        component = SourceComponent(
            spectrum=(
                BandLimitedNoiseSpectrum(SOURCE_LOW_HZ, SOURCE_HIGH_HZ)
                if tone_frequency_hz is None
                else ToneSpectrum(tone_frequency_hz)
            ),
            envelope=ConstantEnvelope(),
            amplitude=None,
            level_db=0.0,
            noise_seed=seed + 100 * source_index,
            noise_filter_length=513,
        )
        sources.append(
            AcousticSource.from_relative_bearing(
                bearing_deg=bearing_deg,
                distance=1000.0,
                receiver_pose=receiver.trajectory.pose(0.0),
                components=[component],
                elevation_deg=0.0,
            )
        )
    ambient_fields = []
    if noise:
        ambient_fields.append(
            AmbientField.from_asd_level_db(
                BandLimitedNoiseSpectrum(0.0, FS_HZ / 2.0),
                NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ,
                covariance=(
                    np.eye(64, dtype=np.float32)
                    if ambient_covariance is None
                    else np.asarray(ambient_covariance, dtype=np.float32)
                ),
                noise_seed=seed + 9000,
                noise_filter_length=513,
            )
        )
    scene = Scene(sources=sources, ambient_fields=ambient_fields, environment=FreeField(c=SOUND_SPEED_M_S))
    axis_t = np.arange(int(FS_HZ * duration_s), dtype=np.float64) / FS_HZ
    return np.asarray(np.real(SceneRenderer().render(scene, receiver, axis_t)), dtype=np.float32)


def _collect_completed_eta(signal: NDArray[np.float32], integration_time_s: float):
    """閾値なしAccumulatorから2秒周期の完成etaを収集する。"""

    schedule, _, steering = _geometry()
    accumulator = DirectionMatchedCovarianceAccumulator(
        schedule,
        integration_time_seconds=integration_time_s,
        steering_table=steering,
    )
    eta: list[NDArray[np.float32]] = []
    elapsed: list[float] = []
    for second in range(signal.shape[1] // int(FS_HZ)):
        start = time.perf_counter()
        accumulator.process_one_second(signal[:, second * int(FS_HZ) : (second + 1) * int(FS_HZ)])
        elapsed.append(time.perf_counter() - start)
        if (second + 1) % 2 == 0:
            eta.append(accumulator.completed_steering_metrics().eta)
    return np.stack(eta), np.asarray(elapsed), accumulator.steering_state_bytes


def _write_heatmap(path: Path, azimuth: NDArray[np.float32], frequency: NDArray[np.float32], table: NDArray[np.float32], title: str) -> None:
    """etaまたはsoft Weightを共通0--1表示で保存する。"""

    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    image = axis.pcolormesh(
        centers_to_edges(azimuth.astype(np.float64)),
        centers_to_edges(frequency.astype(np.float64)),
        table.T,
        shading="flat",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axis.set_ylim(0.0, 1500.0)
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel("Frequency [Hz]")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Ratio")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    """校正scene、未使用scene、積分時間、処理時間比較を実行する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    schedule, frequency_hz, steering = _geometry()
    calibration_noise = _render_segment(10, (), noise=True, seed=1000)
    calibration_target = _render_segment(10, (40.0,), noise=False, seed=2000)
    noise_eta, enabled_benchmark_times, steering_state_bytes = _collect_completed_eta(calibration_noise, 10.0)
    target_eta, _, _ = _collect_completed_eta(calibration_target, 10.0)
    band = (frequency_hz >= SOURCE_LOW_HZ) & (frequency_hz <= SOURCE_HIGH_HZ)
    source_mask = np.abs(schedule.global_direction_azimuth_deg - 40.0) <= 2.0
    gamma_off = np.quantile(noise_eta, 0.99, axis=(0, 1)).astype(np.float32)
    target_floor = np.quantile(target_eta[:, source_mask, :], 0.10, axis=(0, 1)).astype(np.float32)
    gamma_on = np.maximum(target_floor, gamma_off + np.float32(0.02)).astype(np.float32)
    gamma_on = np.minimum(gamma_on, np.float32(1.0))
    gamma_off = np.minimum(gamma_off, gamma_on - np.float32(1.0e-4))

    scenes = {
        "noise_only": calibration_noise,
        "target_only": calibration_target,
        "target_plus_noise": _render_segment(10, (40.0,), noise=True, seed=3000),
        "heldout_direction": _render_segment(10, (72.0,), noise=True, seed=4000),
        "multiple_signals": _render_segment(10, (25.0, 78.0), noise=True, seed=5000),
        "noise_to_target": np.concatenate(
            (_render_segment(4, (), noise=True, seed=6000), _render_segment(6, (40.0,), noise=True, seed=6100)), axis=1
        ),
        "direction_switch": np.concatenate(
            (_render_segment(4, (25.0,), noise=True, seed=7000), _render_segment(6, (78.0,), noise=True, seed=7100)), axis=1
        ),
    }
    config = DirectionCovarianceSelectionConfig(
        gamma_off=gamma_off,
        gamma_on=gamma_on,
        minimum_weight_sum=1.0,
        minimum_effective_direction_count=1.0,
    )
    scene_summary = {}
    saved_arrays = {}
    final_truth = {
        "noise_only": (),
        "target_only": (40.0,),
        "target_plus_noise": (40.0,),
        "heldout_direction": (72.0,),
        "multiple_signals": (25.0, 78.0),
        "noise_to_target": (40.0,),
        "direction_switch": (78.0,),
    }
    for scene_name, signal in scenes.items():
        selector = DirectionMatchedCovarianceSelector(
            schedule,
            steering,
            config,
            integration_time_seconds=10.0,
        )
        results = []
        elapsed = []
        for second in range(signal.shape[1] // int(FS_HZ)):
            start = time.perf_counter()
            result = selector.process_one_second(
                signal[:, second * int(FS_HZ) : (second + 1) * int(FS_HZ)],
                input_series_id=scene_name,
            )
            elapsed.append(time.perf_counter() - start)
            if result is not None:
                results.append(result)
        final = results[-1]
        eta_band_by_direction = np.mean(final.eta[:, band], axis=1)
        peak_index = int(np.argmax(eta_band_by_direction))
        peak_azimuth = float(schedule.global_direction_azimuth_deg[peak_index])
        truths = final_truth[scene_name]
        truth_error = None if not truths else min(abs(peak_azimuth - truth) for truth in truths)
        outside_mask = np.ones(schedule.global_direction_azimuth_deg.size, dtype=np.bool_)
        for truth in truths:
            outside_mask &= np.abs(schedule.global_direction_azimuth_deg - truth) > 10.0
        outside_peak = float(np.max(eta_band_by_direction[outside_mask])) if bool(np.any(outside_mask)) else 0.0
        half_level = float(eta_band_by_direction[peak_index]) * 0.5
        left = peak_index
        right = peak_index
        while left > 0 and float(eta_band_by_direction[left - 1]) >= half_level:
            left -= 1
        while right + 1 < eta_band_by_direction.size and float(eta_band_by_direction[right + 1]) >= half_level:
            right += 1
        mainlobe_width_deg = float(
            schedule.global_direction_azimuth_deg[right] - schedule.global_direction_azimuth_deg[left]
        )
        valid_covariance = final.weighted_covariance[:, :, final.covariance_valid]
        hermitian_errors = []
        minimum_eigenvalues = []
        traces = []
        for covariance in np.moveaxis(valid_covariance, 2, 0):
            norm = float(np.linalg.norm(covariance))
            hermitian_errors.append(float(np.linalg.norm(covariance - covariance.conj().T)) / max(norm, 1.0e-20))
            hermitian = 0.5 * (covariance + covariance.conj().T)
            minimum_eigenvalues.append(float(np.min(np.linalg.eigvalsh(hermitian))))
            traces.append(float(np.real(np.trace(hermitian))))
        tracking_seconds = None
        if scene_name in ("noise_to_target", "direction_switch"):
            new_bearing = truths[0]
            for result_index, completed in enumerate(results):
                completed_second = 2 * (result_index + 1)
                if completed_second <= 4:
                    continue
                history_peak = float(
                    schedule.global_direction_azimuth_deg[
                        int(np.argmax(np.mean(completed.eta[:, band], axis=1)))
                    ]
                )
                if abs(history_peak - new_bearing) <= 2.0:
                    tracking_seconds = float(completed_second - 4)
                    break
        fallback_counts = {
            str(source_code): int(np.count_nonzero(final.fallback_source == source_code))
            for source_code in range(5)
        }
        fallback_reason_counts = {
            str(reason_code): int(np.count_nonzero(final.fallback_reason == reason_code))
            for reason_code in range(7)
        }
        scene_summary[scene_name] = {
            "peak_azimuth_deg": peak_azimuth,
            "peak_error_deg": truth_error,
            "peak_value": float(eta_band_by_direction[peak_index]),
            "mainlobe_half_peak_width_deg": mainlobe_width_deg,
            "outside_truth_guard_maximum_peak": outside_peak,
            "peak_to_outside_margin": float(eta_band_by_direction[peak_index]) - outside_peak,
            "weight_sum_band_mean": float(np.mean(final.weight_sum[band])),
            "valid_weight_count_band_mean": float(np.mean(np.sum(final.completed_weight[:, band] > 0.0, axis=0))),
            "fallback_bin_count": int(np.count_nonzero(final.fallback_source != 1)),
            "fallback_source_counts": fallback_counts,
            "fallback_reason_counts": fallback_reason_counts,
            "tracking_seconds_after_change": tracking_seconds,
            "weighted_covariance_hermitian_error_max": 0.0 if not hermitian_errors else max(hermitian_errors),
            "weighted_covariance_minimum_eigenvalue": 0.0 if not minimum_eigenvalues else min(minimum_eigenvalues),
            "weighted_covariance_trace_minimum": 0.0 if not traces else min(traces),
            "processing_median_seconds_per_input_second": float(np.median(elapsed)),
            "processing_max_seconds_per_input_second": float(np.max(elapsed)),
            "processing_realtime_ratio": float(np.sum(elapsed) / len(elapsed)),
        }
        saved_arrays[f"{scene_name}_eta"] = final.eta
        saved_arrays[f"{scene_name}_weight"] = final.completed_weight
        _write_heatmap(OUTPUT_DIR / f"{scene_name}_eta.png", schedule.global_direction_azimuth_deg, frequency_hz, final.eta, f"{scene_name}: eta")
        _write_heatmap(OUTPUT_DIR / f"{scene_name}_soft_weight.png", schedule.global_direction_azimuth_deg, frequency_hz, final.completed_weight, f"{scene_name}: soft weight")

    threshold_axis = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    negative = noise_eta[:, :, band].reshape(-1)
    positive = target_eta[:, source_mask, :][:, :, band].reshape(-1)
    false_positive_rate = np.asarray([np.mean(negative > threshold) for threshold in threshold_axis], dtype=np.float32)
    true_positive_rate = np.asarray([np.mean(positive > threshold) for threshold in threshold_axis], dtype=np.float32)
    roc_order = np.argsort(false_positive_rate)
    roc_auc = float(np.trapezoid(true_positive_rate[roc_order], false_positive_rate[roc_order]))

    integration_summary = {}
    long_signal = _render_segment(128, (40.0,), noise=True, seed=8000)
    for integration_time_s in (10.0, 40.0, 128.0):
        eta_history, elapsed, _ = _collect_completed_eta(long_signal, integration_time_s)
        final_eta = eta_history[-1]
        peak = np.mean(final_eta[:, band], axis=1)
        integration_summary[str(int(integration_time_s))] = {
            "peak_azimuth_deg": float(schedule.global_direction_azimuth_deg[int(np.argmax(peak))]),
            "source_neighborhood_mean": float(np.mean(final_eta[source_mask][:, band])),
            "processing_realtime_ratio": float(np.sum(elapsed) / 128.0),
        }

    # steering無効/有効の処理時間は同じnoise入力で計測し、追加演算だけを比較する。
    disabled_accumulator = DirectionMatchedCovarianceAccumulator(schedule, integration_time_seconds=10.0)
    disabled_benchmark = []
    for second in range(10):
        start = time.perf_counter()
        disabled_accumulator.process_one_second(calibration_noise[:, second * int(FS_HZ) : (second + 1) * int(FS_HZ)])
        disabled_benchmark.append(time.perf_counter() - start)
    benchmark = {
        "disabled_median_seconds": float(np.median(disabled_benchmark)),
        "disabled_max_seconds": float(np.max(disabled_benchmark)),
        "disabled_realtime_ratio": float(np.sum(disabled_benchmark) / 10.0),
        "enabled_median_seconds": float(np.median(enabled_benchmark_times)),
        "enabled_max_seconds": float(np.max(enabled_benchmark_times)),
        "enabled_realtime_ratio": float(np.sum(enabled_benchmark_times) / 10.0),
        "steering_state_bytes": int(steering_state_bytes),
        "projected_chunk_temporary_bytes": 65 * 8 * np.dtype(np.complex64).itemsize,
        "two_instantaneous_power_chunks_bytes": 2 * 8 * 65 * np.dtype(np.float32).itemsize,
        "quadratic_complex_multiply_accumulate_count": 159 * 65 * 64 * 64,
        "direct_complex_multiply_accumulate_count": 159 * 65 * 64,
    }
    np.savez_compressed(
        OUTPUT_DIR / "method3_direction_selection.npz",
        azimuth_deg=schedule.global_direction_azimuth_deg,
        frequency_hz=frequency_hz,
        gamma_off=gamma_off,
        gamma_on=gamma_on,
        roc_threshold=threshold_axis,
        roc_false_positive_rate=false_positive_rate,
        roc_true_positive_rate=true_positive_rate,
        **saved_arrays,
    )
    summary = {
        "eta_noise_theory": 1.0 / 64.0,
        "noise_eta_band_mean": float(np.mean(noise_eta[:, :, band])),
        "scene_summary": scene_summary,
        "integration_time_summary": integration_summary,
        "runtime_benchmark": benchmark,
        "roc_auc_calibration": roc_auc,
        "calibration": "gamma_off=noise q99; gamma_on=max(target-neighborhood q10, gamma_off+0.02)",
        "threshold_status": "calibration_candidate_not_adopted",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
