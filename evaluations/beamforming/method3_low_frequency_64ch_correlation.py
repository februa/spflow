"""方式3の64ch低周波共分散について複数の正規化相関統計を比較する。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from scene_renderer import (  # noqa: E402
    AcousticSource,
    AmbientField,
    BandLimitedNoiseSpectrum,
    ConstantEnvelope,
    FreeField,
    LinearArray,
    Receiver,
    Scene,
    SceneRenderer,
    SourceComponent,
    StaticPose,
)
from spflow.beamforming import (  # noqa: E402
    DirectionMatchedCovarianceAccumulator,
    build_two_second_covariance_snapshot_schedule,
    calculate_spatial_correlation_statistics,
)
from spflow.beamforming.diagnostic_plotting import centers_to_edges, require_matplotlib  # noqa: E402


FS_HZ = 8192.0
SOUND_SPEED_M_S = 1500.0
PROCESS_DURATION_S = 60
INTEGRATION_TIME_S = 10.0
DISPLAY_SNAPSHOT_SECONDS = (1, 2, 5, 10, 20, 40, 60)
N_CH = 64
SPACING_M = 0.5
SNAPSHOT_LENGTH = 128
N_BEAM_PER_HALF = 159
SOURCE_BEARING_DEG = 40.0
SOURCE_BAND_LOW_HZ = 128.0
SOURCE_BAND_HIGH_HZ = 1024.0
SOURCE_LEVEL_DB_RE_INPUT_RMS = 0.0
NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ = -32.0
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "method3_low_frequency_64ch"

# 等間隔ULAの空間alias回避基準`d <= lambda/2`から、基準周波数は`c/(2d)`となる。
# 信号上限1024 Hzをこの基準1500 Hzより低くし、grating lobeではなく低周波広帯域の相関形成を観測する。
SPATIAL_ALIAS_REFERENCE_HZ = SOUND_SPEED_M_S / (2.0 * SPACING_M)
ARRAY_APERTURE_M = (N_CH - 1) * SPACING_M
REPRESENTATIVE_BASELINE_INDICES = (1, 8, 16, 32, 63)
CORRELATION_DISPLAY_RANGE = (0.0, 1.0)


def _render_evaluation_scene(receiver: Receiver) -> np.ndarray:
    """scene_rendererで低周波広帯域sourceとCH無相関背景雑音を生成する。

    Args:
        receiver: 64ch対称ULA受波器。受波器間隔は低周波帯域に合わせて0.5 mとする。

    Returns:
        受信信号。shapeは`[n_ch,PROCESS_DURATION_S*fs]`、dtypeは`float32`。
    """

    source_component = SourceComponent(
        spectrum=BandLimitedNoiseSpectrum(SOURCE_BAND_LOW_HZ, SOURCE_BAND_HIGH_HZ),
        envelope=ConstantEnvelope(),
        amplitude=None,
        level_db=SOURCE_LEVEL_DB_RE_INPUT_RMS,
        noise_seed=400010,
        noise_filter_length=513,
    )
    source = AcousticSource.from_relative_bearing(
        bearing_deg=SOURCE_BEARING_DEG,
        distance=1000.0,
        receiver_pose=receiver.trajectory.pose(0.0),
        components=[source_component],
        elevation_deg=0.0,
    )
    noise_spectrum = BandLimitedNoiseSpectrum(0.0, FS_HZ / 2.0)
    ambient = AmbientField.from_asd_level_db(
        noise_spectrum,
        NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ,
        covariance=np.eye(N_CH, dtype=np.float32),
        noise_seed=320010,
        noise_filter_length=513,
    )
    scene = Scene(
        sources=[source],
        ambient_fields=[ambient],
        environment=FreeField(c=SOUND_SPEED_M_S),
    )
    sample_count = int(round(FS_HZ * PROCESS_DURATION_S))
    axis_t = np.arange(sample_count, dtype=np.float64) / FS_HZ
    rendered = SceneRenderer().render(scene, receiver, axis_t)
    return np.asarray(np.real(rendered), dtype=np.float32)


def _write_correlation_png(
    output_path: Path,
    azimuth_deg: np.ndarray,
    frequency_hz: np.ndarray,
    correlation: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
) -> None:
    """相関`[方位,周波数]`を共通表示条件のimagescとして保存する。

    Args:
        output_path: PNG保存先。
        azimuth_deg: 方位軸。shapeは`[n_direction]`、単位はdeg。
        frequency_hz: 周波数軸。shapeは`[n_bin]`、単位はHz。
        correlation: 正規化相関。shapeは`[n_direction,n_bin]`。
        title: 図題。
        colorbar_label: 相関統計の表示名。
    """

    matplotlib = require_matplotlib()
    azimuth_edges = centers_to_edges(np.asarray(azimuth_deg, dtype=np.float64))
    frequency_edges = centers_to_edges(np.asarray(frequency_hz, dtype=np.float64))
    figure, axis = matplotlib.subplots(figsize=(11.0, 5.5), constrained_layout=True)
    # table shapeは`[n_direction,n_bin]`。pcolormeshではyをfrequencyにするため転置する。
    image = axis.pcolormesh(
        azimuth_edges,
        frequency_edges,
        np.asarray(correlation, dtype=np.float32).T,
        shading="flat",
        vmin=CORRELATION_DISPLAY_RANGE[0],
        vmax=CORRELATION_DISPLAY_RANGE[1],
        cmap="viridis",
    )
    axis.axvline(SOURCE_BEARING_DEG, color="tab:red", linewidth=1.0, linestyle="--", label="source bearing")
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel("Frequency [Hz]")
    axis.set_ylim(0.0, SPATIAL_ALIAS_REFERENCE_HZ)
    axis.set_title(title)
    axis.legend(loc="upper right")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.savefig(output_path, dpi=160)
    matplotlib.close(figure)


def _write_time_evolution_png(
    output_path: Path,
    azimuth_deg: np.ndarray,
    frequency_hz: np.ndarray,
    snapshot_seconds: np.ndarray,
    correlation_by_snapshot: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
) -> None:
    """指定snapshot時刻の相関imagescを共通表示条件で並べる。

    Args:
        output_path: PNG保存先。
        azimuth_deg: 方位軸。shapeは`[n_direction]`、単位はdeg。
        frequency_hz: 周波数軸。shapeは`[n_bin]`、単位はHz。
        snapshot_seconds: 各panelの処理時刻。shapeは`[n_snapshot]`、単位はs。
        correlation_by_snapshot: 相関。shapeは`[n_snapshot,n_direction,n_bin]`。
        title: 図題。
        colorbar_label: 相関統計の表示名。
    """

    matplotlib = require_matplotlib()
    azimuth_edges = centers_to_edges(np.asarray(azimuth_deg, dtype=np.float64))
    frequency_edges = centers_to_edges(np.asarray(frequency_hz, dtype=np.float64))
    figure, axes = matplotlib.subplots(2, 4, figsize=(16.0, 8.0), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for panel_index, axis in enumerate(axes.flat):
        if panel_index >= snapshot_seconds.size:
            axis.set_visible(False)
            continue
        second = int(snapshot_seconds[panel_index])
        image = axis.pcolormesh(
            azimuth_edges,
            frequency_edges,
            correlation_by_snapshot[panel_index].T,
            shading="flat",
            vmin=CORRELATION_DISPLAY_RANGE[0],
            vmax=CORRELATION_DISPLAY_RANGE[1],
            cmap="viridis",
        )
        axis.axvline(SOURCE_BEARING_DEG, color="tab:red", linewidth=0.8, linestyle="--")
        axis.set_title(f"processed {second} s")
        axis.set_ylim(0.0, SPATIAL_ALIAS_REFERENCE_HZ)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Azimuth [deg]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Frequency [Hz]")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, location="right", shrink=0.95)
        colorbar.set_label(colorbar_label)
    figure.suptitle(title)
    figure.savefig(output_path, dpi=150)
    matplotlib.close(figure)


def _write_all_baseline_trend_png(
    output_path: Path,
    azimuth_deg: np.ndarray,
    baseline_index: np.ndarray,
    baseline_mean_by_snapshot: np.ndarray,
    signal_band_mask: np.ndarray,
    snapshot_seconds: np.ndarray,
) -> None:
    """全基線の信号帯域平均を`[方位,基線長]`で時系列表示する。

    Args:
        output_path: PNG保存先。
        azimuth_deg: 方位軸。shapeは`[n_direction]`、単位はdeg。
        baseline_index: 基線index。shapeは`[n_baseline]`。
        baseline_mean_by_snapshot: 基線別相関。shapeは
            `[n_snapshot,n_direction,n_bin,n_baseline]`。
        signal_band_mask: 128--1024 Hzを選択するmask。shapeは`[n_bin]`。
        snapshot_seconds: 各panelの処理時刻。shapeは`[n_snapshot]`、単位はs。

    Notes:
        周波数軸を信号帯域内で平均し、代表基線だけでは見えない全基線方向の傾向を残す。
        基線長は等間隔ULAなので`baseline_index * spacing` [m]である。
    """

    matplotlib = require_matplotlib()
    azimuth_edges = centers_to_edges(np.asarray(azimuth_deg, dtype=np.float64))
    baseline_length_m = np.asarray(baseline_index, dtype=np.float64) * SPACING_M
    baseline_edges_m = centers_to_edges(baseline_length_m)
    figure, axes = matplotlib.subplots(2, 4, figsize=(16.0, 8.0), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for panel_index, axis in enumerate(axes.flat):
        if panel_index >= snapshot_seconds.size:
            axis.set_visible(False)
            continue
        # 周波数axis=2を信号帯域内で平均し、table `[direction,baseline]`を描画用に転置する。
        snapshot_baseline_mean = baseline_mean_by_snapshot[panel_index]
        # boolean advanced indexingを他axisと同時指定するとmask軸が先頭へ移るため、2段階で選択する。
        # snapshot `[direction,bin,baseline]`からbin axisだけを選び、axis=1で帯域平均する。
        band_mean = np.mean(snapshot_baseline_mean[:, signal_band_mask, :], axis=1)
        image = axis.pcolormesh(
            azimuth_edges,
            baseline_edges_m,
            np.asarray(band_mean, dtype=np.float32).T,
            shading="flat",
            vmin=CORRELATION_DISPLAY_RANGE[0],
            vmax=CORRELATION_DISPLAY_RANGE[1],
            cmap="viridis",
        )
        axis.axvline(SOURCE_BEARING_DEG, color="tab:red", linewidth=0.8, linestyle="--")
        axis.set_title(f"processed {int(snapshot_seconds[panel_index])} s")
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Azimuth [deg]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Baseline length [m]")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, location="right", shrink=0.95)
        colorbar.set_label("Signal-band baseline mean correlation [ratio]")
    figure.suptitle("Method 3 all-baseline trend, 128–1024 Hz (integration time = 10 s)")
    figure.savefig(output_path, dpi=150)
    matplotlib.close(figure)


def main() -> None:
    """64ch低周波sceneを方式3へ入力し、相関統計の比較成果物を保存する。"""

    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=N_CH, spacing=SPACING_M, axis=0, centered=True),
    )
    schedule = build_two_second_covariance_snapshot_schedule(
        receiver.array.positions(),
        fs_hz=FS_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        snapshot_length_samples=SNAPSHOT_LENGTH,
        beams_per_half=N_BEAM_PER_HALF,
    )
    accumulator = DirectionMatchedCovarianceAccumulator(
        schedule,
        integration_time_seconds=INTEGRATION_TIME_S,
    )
    render_start = time.perf_counter()
    rendered = _render_evaluation_scene(receiver)
    render_elapsed_seconds = time.perf_counter() - render_start
    samples_per_second = int(round(FS_HZ))
    snapshot_seconds = np.asarray(DISPLAY_SNAPSHOT_SECONDS, dtype=np.int32)
    snapshot_second_set = set(DISPLAY_SNAPSHOT_SECONDS)
    scalar_metric_names = ("maximum", "mean", "median", "percentile_95")
    scalar_metric_snapshots: dict[str, list[np.ndarray]] = {
        metric_name: [] for metric_name in scalar_metric_names
    }
    baseline_mean_snapshots: list[np.ndarray] = []
    processing_elapsed_by_second: list[float] = []
    correlation_elapsed_by_second: list[float] = []
    table_shape = (schedule.global_direction_azimuth_deg.size, SNAPSHOT_LENGTH // 2 + 1)
    current_scalar_metrics = {
        metric_name: np.zeros(table_shape, dtype=np.float32)
        for metric_name in scalar_metric_names
    }
    current_baseline_mean = np.zeros((*table_shape, N_CH - 1), dtype=np.float32)
    baseline_index = np.arange(1, N_CH, dtype=np.int32)
    frequency_hz = np.asarray(
        np.fft.rfftfreq(SNAPSHOT_LENGTH, d=1.0 / FS_HZ),
        dtype=np.float32,
    )
    for second_index in range(PROCESS_DURATION_S):
        frame = rendered[:, second_index * samples_per_second : (second_index + 1) * samples_per_second]
        process_start = time.perf_counter()
        update = accumulator.process_one_second(frame)
        processing_elapsed_by_second.append(time.perf_counter() - process_start)
        correlation_start = time.perf_counter()
        active_statistics = calculate_spatial_correlation_statistics(
            update.active_direction_covariance,
        )
        correlation_elapsed_by_second.append(time.perf_counter() - correlation_start)
        active_scalar_metrics = {
            "maximum": active_statistics.maximum,
            "mean": active_statistics.mean,
            "median": active_statistics.median,
            "percentile_95": active_statistics.percentile_95,
        }
        # 非更新方位は共分散と同じく前回値を保持し、今回の159方位だけを全統計で置換する。
        for metric_name, active_table in active_scalar_metrics.items():
            current_scalar_metrics[metric_name][update.global_direction_indices] = active_table
        current_baseline_mean[update.global_direction_indices] = active_statistics.baseline_mean
        processed_second = second_index + 1
        if processed_second in snapshot_second_set:
            for metric_name in scalar_metric_names:
                scalar_metric_snapshots[metric_name].append(current_scalar_metrics[metric_name].copy())
            baseline_mean_snapshots.append(current_baseline_mean.copy())

    processing_times = np.asarray(processing_elapsed_by_second, dtype=np.float64)
    correlation_times = np.asarray(correlation_elapsed_by_second, dtype=np.float64)
    scalar_metrics_by_snapshot = {
        metric_name: np.stack(metric_snapshots, axis=0).astype(np.float32)
        for metric_name, metric_snapshots in scalar_metric_snapshots.items()
    }
    baseline_mean_by_snapshot = np.stack(baseline_mean_snapshots, axis=0).astype(np.float32)
    signal_band_mask = (frequency_hz >= SOURCE_BAND_LOW_HZ) & (frequency_hz <= SOURCE_BAND_HIGH_HZ)
    comparison_band_mask = (frequency_hz > SOURCE_BAND_HIGH_HZ) & (frequency_hz <= SPATIAL_ALIAS_REFERENCE_HZ)
    # source近傍はbeam間隔より十分広い±2度、遠方はmainlobe混入を避けるため20度以上離れた方位とする。
    # ここでの値は採否閾値ではなく、最大pair相関が方位選択性を持つかを観測する比較領域である。
    source_neighborhood_mask = np.abs(schedule.global_direction_azimuth_deg - SOURCE_BEARING_DEG) <= 2.0
    far_direction_mask = np.abs(schedule.global_direction_azimuth_deg - SOURCE_BEARING_DEG) >= 20.0
    covariance_storage_bytes = int(accumulator.direction_covariance.nbytes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "method3_spatial_correlation_statistics.npz",
        azimuth_deg=schedule.global_direction_azimuth_deg,
        frequency_hz=frequency_hz,
        snapshot_seconds=snapshot_seconds,
        maximum=current_scalar_metrics["maximum"],
        mean=current_scalar_metrics["mean"],
        median=current_scalar_metrics["median"],
        percentile_95=current_scalar_metrics["percentile_95"],
        maximum_by_snapshot=scalar_metrics_by_snapshot["maximum"],
        mean_by_snapshot=scalar_metrics_by_snapshot["mean"],
        median_by_snapshot=scalar_metrics_by_snapshot["median"],
        percentile_95_by_snapshot=scalar_metrics_by_snapshot["percentile_95"],
        baseline_index=baseline_index,
        baseline_length_m=baseline_index.astype(np.float32) * np.float32(SPACING_M),
        baseline_mean=current_baseline_mean,
        baseline_mean_by_snapshot=baseline_mean_by_snapshot,
        direction_update_coef=accumulator.direction_update_coef,
        processing_elapsed_seconds=processing_times,
        correlation_elapsed_seconds=correlation_times,
    )
    metric_display_names = {
        "maximum": "Maximum",
        "mean": "Lower-triangle mean",
        "median": "Lower-triangle median",
        "percentile_95": "Lower-triangle 95 percentile",
    }
    for metric_name in scalar_metric_names:
        display_name = metric_display_names[metric_name]
        _write_correlation_png(
            OUTPUT_DIR / f"method3_{metric_name}_correlation_imagesc.png",
            schedule.global_direction_azimuth_deg,
            frequency_hz,
            current_scalar_metrics[metric_name],
            title=(
                f"Method 3: {display_name} normalized correlation after {PROCESS_DURATION_S} s "
                f"(integration time {INTEGRATION_TIME_S:g} s)"
            ),
            colorbar_label=f"{display_name} normalized correlation [ratio]",
        )
        _write_time_evolution_png(
            OUTPUT_DIR / f"method3_{metric_name}_correlation_time_evolution.png",
            schedule.global_direction_azimuth_deg,
            frequency_hz,
            snapshot_seconds,
            scalar_metrics_by_snapshot[metric_name],
            title=f"Method 3 {display_name} correlation evolution (integration time = 10 s)",
            colorbar_label=f"{display_name} normalized correlation [ratio]",
        )

    for representative_baseline_index in REPRESENTATIVE_BASELINE_INDICES:
        baseline_axis_index = representative_baseline_index - 1
        baseline_length_m = representative_baseline_index * SPACING_M
        _write_correlation_png(
            OUTPUT_DIR / f"method3_baseline_{representative_baseline_index:02d}_mean_imagesc.png",
            schedule.global_direction_azimuth_deg,
            frequency_hz,
            current_baseline_mean[:, :, baseline_axis_index],
            title=(
                f"Method 3: baseline mean, index={representative_baseline_index}, "
                f"length={baseline_length_m:g} m"
            ),
            colorbar_label="Baseline mean normalized correlation [ratio]",
        )
        _write_time_evolution_png(
            OUTPUT_DIR / f"method3_baseline_{representative_baseline_index:02d}_mean_time_evolution.png",
            schedule.global_direction_azimuth_deg,
            frequency_hz,
            snapshot_seconds,
            baseline_mean_by_snapshot[:, :, :, baseline_axis_index],
            title=(
                f"Method 3 baseline mean evolution, index={representative_baseline_index}, "
                f"length={baseline_length_m:g} m"
            ),
            colorbar_label="Baseline mean normalized correlation [ratio]",
        )
    _write_all_baseline_trend_png(
        OUTPUT_DIR / "method3_all_baseline_signal_band_trend.png",
        schedule.global_direction_azimuth_deg,
        baseline_index,
        baseline_mean_by_snapshot,
        signal_band_mask,
        snapshot_seconds,
    )

    region_statistics: dict[str, dict[str, float | list[float]]] = {}
    for metric_name in scalar_metric_names:
        final_table = current_scalar_metrics[metric_name]
        source_value = float(np.mean(final_table[source_neighborhood_mask][:, signal_band_mask]))
        far_value = float(np.mean(final_table[far_direction_mask][:, signal_band_mask]))
        region_statistics[metric_name] = {
            "source_neighborhood_signal_band_mean": source_value,
            "far_direction_signal_band_mean": far_value,
            "source_minus_far": source_value - far_value,
        }
    source_baseline_mean = np.mean(
        current_baseline_mean[source_neighborhood_mask][:, signal_band_mask, :],
        axis=(0, 1),
    ).astype(np.float32)
    far_baseline_mean = np.mean(
        current_baseline_mean[far_direction_mask][:, signal_band_mask, :],
        axis=(0, 1),
    ).astype(np.float32)
    baseline_source_minus_far = source_baseline_mean - far_baseline_mean
    maximum_contrast_baseline_axis_index = int(np.argmax(baseline_source_minus_far))
    region_statistics["baseline_mean"] = {
        "source_neighborhood_signal_band_mean_by_baseline": source_baseline_mean.astype(float).tolist(),
        "far_direction_signal_band_mean_by_baseline": far_baseline_mean.astype(float).tolist(),
        "source_minus_far_by_baseline": baseline_source_minus_far.astype(float).tolist(),
        "maximum_contrast_baseline_index": int(baseline_index[maximum_contrast_baseline_axis_index]),
        "maximum_contrast_baseline_length_m": float(
            baseline_index[maximum_contrast_baseline_axis_index] * SPACING_M
        ),
        "maximum_source_minus_far": float(baseline_source_minus_far[maximum_contrast_baseline_axis_index]),
    }
    summary = {
        "method": 3,
        "integration_time_seconds": INTEGRATION_TIME_S,
        "process_duration_seconds": PROCESS_DURATION_S,
        "display_snapshot_seconds": list(DISPLAY_SNAPSHOT_SECONDS),
        "fs_hz": FS_HZ,
        "snapshot_length_samples": SNAPSHOT_LENGTH,
        "frequency_resolution_hz": FS_HZ / SNAPSHOT_LENGTH,
        "n_ch": N_CH,
        "spacing_m": SPACING_M,
        "array_aperture_m": ARRAY_APERTURE_M,
        "spatial_alias_reference_hz": SPATIAL_ALIAS_REFERENCE_HZ,
        "n_beam_per_half": N_BEAM_PER_HALF,
        "source_bearing_deg": SOURCE_BEARING_DEG,
        "source_band_hz": [SOURCE_BAND_LOW_HZ, SOURCE_BAND_HIGH_HZ],
        "source_level_db_re_input_rms": SOURCE_LEVEL_DB_RE_INPUT_RMS,
        "noise_level_db_re_input_rms_per_sqrt_hz": NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ,
        "ordinary_direction_update_rate_per_second": 0.5,
        "ordinary_direction_effective_snapshot_count": 0.5 * INTEGRATION_TIME_S,
        "ordinary_direction_coef": float(accumulator.direction_update_coef[0]),
        "shared_90deg_update_rate_per_second": 1.0,
        "shared_90deg_effective_snapshot_count": INTEGRATION_TIME_S,
        "shared_90deg_coef": float(accumulator.direction_update_coef[N_BEAM_PER_HALF - 1]),
        "integration_time_policy": "fixed_for_statistic_comparison",
        "correlation_display_range": list(CORRELATION_DISPLAY_RANGE),
        "correlation_formula": "abs(R_ij) / sqrt(R_ii * R_jj), i > j",
        "scalar_correlation_shape": list(current_scalar_metrics["maximum"].shape),
        "baseline_mean_shape": list(current_baseline_mean.shape),
        "baseline_index": baseline_index.tolist(),
        "baseline_length_m": (baseline_index.astype(np.float32) * np.float32(SPACING_M)).tolist(),
        "representative_baseline_index": list(REPRESENTATIVE_BASELINE_INDICES),
        "direction_covariance_storage_bytes": covariance_storage_bytes,
        "final_all_direction_signal_band_mean_by_statistic": {
            metric_name: float(np.mean(current_scalar_metrics[metric_name][:, signal_band_mask]))
            for metric_name in scalar_metric_names
        },
        "final_all_direction_comparison_band_mean_by_statistic": {
            metric_name: float(np.mean(current_scalar_metrics[metric_name][:, comparison_band_mask]))
            for metric_name in scalar_metric_names
        },
        "region_statistics_128_1024_hz": region_statistics,
        "scene_render_elapsed_seconds": render_elapsed_seconds,
        "processing_elapsed_seconds_total": float(np.sum(processing_times)),
        "processing_elapsed_seconds_median_per_input_second": float(np.median(processing_times)),
        "processing_elapsed_seconds_max_per_input_second": float(np.max(processing_times)),
        "processing_realtime_ratio": float(np.sum(processing_times) / PROCESS_DURATION_S),
        "correlation_elapsed_seconds_total": float(np.sum(correlation_times)),
        "correlation_elapsed_seconds_median_per_input_second": float(np.median(correlation_times)),
        "correlation_elapsed_seconds_max_per_input_second": float(np.max(correlation_times)),
    }
    (OUTPUT_DIR / "correlation_statistics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
