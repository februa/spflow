"""中央密・外側疎64chアレイで方式3の正規化相関統計を比較する。"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from scene_renderer import (  # noqa: E402
    AcousticSource,
    AmbientField,
    BandLimitedNoiseSpectrum,
    ConstantEnvelope,
    FreeField,
    ArrayGeometry,
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
from spflow.beamforming.diagnostic_plotting import centers_to_edges, require_matplotlib  # noqa: E402


FS_HZ = 8192.0
SOUND_SPEED_M_S = 1500.0
PROCESS_DURATION_S = 60
INTEGRATION_TIME_S = 10.0
DISPLAY_SNAPSHOT_SECONDS = (1, 2, 5, 10, 20, 40, 60)
SNAPSHOT_LENGTH = 128
N_BEAM_PER_HALF = 159
SOURCE_BEARING_DEG = 40.0
SOURCE_BAND_LOW_HZ = 128.0
SOURCE_BAND_HIGH_HZ = 1024.0
SOURCE_LEVEL_DB_RE_INPUT_RMS = 0.0
NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ = -32.0
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "method3_sparse_64ch"
DISPLAY_FREQUENCY_MAX_HZ = 1500.0
CORRELATION_DISPLAY_RANGE = (0.0, 1.0)


def _build_center_dense_outer_sparse_positions_m() -> NDArray[np.float32]:
    """既存運用設計と同じ段階疎配置思想で64ch左右対称座標を作る。

    Returns:
        受波器座標。shapeは`[64,3]`、単位はm。

    Notes:
        正側32chを中央16ch×0.25 m、中間8ch×0.5 m、外側8ch×1.0 mで構成し、
        負側へ鏡映する。既存`OperationalSparseArrayDesignConfig`の中央密・外側疎という
        段階spacingを64ch・約32 m開口へ縮小して再利用する。
    """

    positive_x_m = np.concatenate(
        (
            np.arange(0.25, 4.0 + 0.125, 0.25, dtype=np.float32),
            np.arange(4.5, 8.0 + 0.25, 0.5, dtype=np.float32),
            np.arange(9.0, 16.0 + 0.5, 1.0, dtype=np.float32),
        )
    )
    x_m = np.concatenate((-positive_x_m[::-1], positive_x_m)).astype(np.float32)
    positions_m = np.zeros((x_m.size, 3), dtype=np.float32)
    positions_m[:, 0] = x_m
    if positions_m.shape != (64, 3):
        raise ValueError("center-dense outer-sparse layout must contain exactly 64 channels.")
    return positions_m


@dataclass(frozen=True)
class CoordinateArray(ArrayGeometry):
    """scene_rendererへ任意受波器座標を渡す評価用アレイ。

    入力と出力は`positions_m [n_ch,3]`、単位mである。座標生成、信号生成、
    ビーム処理は責務に含めず、scene_rendererのArrayGeometry契約への適合だけを担う。
    """

    positions_m: NDArray[np.float32]

    def positions(self) -> NDArray[Any]:
        """保持した受波器座標のcopyを返す。

        Returns:
            受波器座標。shapeは`[n_ch,3]`、axis=1はx/y/z、単位はm。
        """

        return self.positions_m.copy()


ARRAY_POSITIONS_M = _build_center_dense_outer_sparse_positions_m()
N_CH = int(ARRAY_POSITIONS_M.shape[0])
ARRAY_APERTURE_M = float(np.ptp(ARRAY_POSITIONS_M[:, 0]))
ADJACENT_SPACING_M = np.diff(ARRAY_POSITIONS_M[:, 0])
MINIMUM_SPACING_M = float(np.min(ADJACENT_SPACING_M))
MAXIMUM_SPACING_M = float(np.max(ADJACENT_SPACING_M))
CENTRAL_CHANNEL_MASK = np.abs(ARRAY_POSITIONS_M[:, 0]) <= np.float32(4.0)
# 物理基線binは最小spacing 0.25 m幅とし、座標距離をchannel index差へ置き換えない。
PHYSICAL_BASELINE_EDGES_M = np.arange(
    MINIMUM_SPACING_M / 2.0,
    ARRAY_APERTURE_M + MINIMUM_SPACING_M,
    MINIMUM_SPACING_M,
    dtype=np.float32,
)
# 最大rFFT周波数でのd/lambdaを覆う1波長幅binを全周波数で共用する。
MAXIMUM_NORMALIZED_BASELINE = ARRAY_APERTURE_M * (FS_HZ / 2.0) / SOUND_SPEED_M_S
WAVELENGTH_NORMALIZED_EDGES = np.arange(
    0.0,
    np.ceil(MAXIMUM_NORMALIZED_BASELINE) + 2.0,
    1.0,
    dtype=np.float32,
)


def _render_evaluation_scene(receiver: Receiver) -> np.ndarray:
    """scene_rendererで低周波広帯域sourceとCH無相関背景雑音を生成する。

    Args:
        receiver: 中央密・外側疎の64ch左右対称受波器。

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
    axis.set_ylim(0.0, DISPLAY_FREQUENCY_MAX_HZ)
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
        axis.set_ylim(0.0, DISPLAY_FREQUENCY_MAX_HZ)
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


def _write_group_trend_png(
    output_path: Path,
    azimuth_deg: np.ndarray,
    group_representative: np.ndarray,
    correlation_by_snapshot: np.ndarray,
    snapshot_seconds: np.ndarray,
    *,
    group_axis_label: str,
    title: str,
    colorbar_label: str,
) -> None:
    """全groupの信号帯域平均を`[方位,group]`で時系列表示する。

    Args:
        output_path: PNG保存先。
        azimuth_deg: 方位軸。shapeは`[n_direction]`、単位はdeg。
        group_representative: group代表値。shapeは`[n_group]`。
        correlation_by_snapshot: 信号帯域平均相関。shapeは`[n_snapshot,n_direction,n_group]`。
        snapshot_seconds: 各panelの処理時刻。shapeは`[n_snapshot]`、単位はs。
        group_axis_label: group軸の表示名と単位。
        title: 図題。
        colorbar_label: colorbar表示名。

    Notes:
        代表groupだけでは見えない全group方向の傾向を同一色範囲で残す。
    """

    matplotlib = require_matplotlib()
    azimuth_edges = centers_to_edges(np.asarray(azimuth_deg, dtype=np.float64))
    group_edges = centers_to_edges(np.asarray(group_representative, dtype=np.float64))
    figure, axes = matplotlib.subplots(2, 4, figsize=(16.0, 8.0), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for panel_index, axis in enumerate(axes.flat):
        if panel_index >= snapshot_seconds.size:
            axis.set_visible(False)
            continue
        image = axis.pcolormesh(
            azimuth_edges,
            group_edges,
            np.asarray(correlation_by_snapshot[panel_index], dtype=np.float32).T,
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
        axis.set_ylabel(group_axis_label)
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, location="right", shrink=0.95)
        colorbar.set_label(colorbar_label)
    figure.suptitle(title)
    figure.savefig(output_path, dpi=150)
    matplotlib.close(figure)


def _write_signal_frequency_azimuth_overlay(
    output_path: Path,
    azimuth_deg: NDArray[np.float32],
    frequency_hz: NDArray[np.float32],
    signal_band_mask: NDArray[np.bool_],
    correlation_tables: dict[str, NDArray[np.float32]],
    *,
    title: str,
) -> None:
    """信号帯域の各周波数で複数相関方式を方位方向へ重ねて描く。

    Args:
        output_path: PNG保存先。
        azimuth_deg: 方位軸。shapeは`[n_direction]`、単位はdeg。
        frequency_hz: 周波数軸。shapeは`[n_bin]`、単位はHz。
        signal_band_mask: 入力信号帯域を選ぶmask。shapeは`[n_bin]`。
        correlation_tables: 表示名から相関`[n_direction,n_bin]`への対応。
        title: 図全体の題名。

    Notes:
        全方式でx軸0--180 deg、y軸0--1、60秒snapshotを固定し、表示条件差を排除する。
    """

    matplotlib = require_matplotlib()
    selected_frequency_indices = np.flatnonzero(signal_band_mask)
    n_column = 4
    n_row = int(np.ceil(selected_frequency_indices.size / n_column))
    figure, axes = matplotlib.subplots(
        n_row,
        n_column,
        figsize=(16.0, 3.2 * n_row),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for panel_index, axis in enumerate(np.asarray(axes).flat):
        if panel_index >= selected_frequency_indices.size:
            axis.set_visible(False)
            continue
        frequency_index = int(selected_frequency_indices[panel_index])
        for label, table in correlation_tables.items():
            axis.plot(azimuth_deg, table[:, frequency_index], linewidth=1.0, label=label)
        axis.axvline(SOURCE_BEARING_DEG, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"{float(frequency_hz[frequency_index]):g} Hz")
        axis.set_xlim(0.0, 180.0)
        axis.set_ylim(CORRELATION_DISPLAY_RANGE)
        axis.grid(True, alpha=0.2)
    for axis in np.asarray(axes)[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Azimuth [deg]")
    for axis in np.asarray(axes)[:, 0]:
        axis.set_ylabel("Normalized correlation [ratio]")
    visible_axes = [axis for axis in np.asarray(axes).flat if axis.get_visible()]
    if visible_axes:
        visible_axes[0].legend(loc="lower right", fontsize=8)
    figure.suptitle(title)
    figure.savefig(output_path, dpi=150)
    matplotlib.close(figure)


def _finite_frequency_mean(values: NDArray[np.float32], frequency_mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    """NaNの空groupを除外して指定周波数帯域を平均する。"""

    selected = values[:, frequency_mask, :]
    valid_count = np.sum(np.isfinite(selected), axis=1)
    total = np.nansum(selected, axis=1)
    result = np.full(total.shape, np.nan, dtype=np.float32)
    valid = valid_count > 0
    result[valid] = np.asarray(total[valid] / valid_count[valid], dtype=np.float32)
    return result


def _finite_region_mean(
    values: NDArray[np.float32],
    direction_mask: NDArray[np.bool_],
    frequency_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """空groupをNaNのまま保ち、指定方位・周波数領域を平均する。"""

    selected = values[direction_mask][:, frequency_mask, :]
    valid_count = np.sum(np.isfinite(selected), axis=(0, 1))
    total = np.nansum(selected, axis=(0, 1))
    result = np.full(total.shape, np.nan, dtype=np.float32)
    valid = valid_count > 0
    result[valid] = np.asarray(total[valid] / valid_count[valid], dtype=np.float32)
    return result


def main() -> None:
    """中央密・外側疎64ch sceneを方式3へ入力し、基線別相関成果物を保存する。"""

    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=CoordinateArray(ARRAY_POSITIONS_M),
    )
    schedule = build_two_second_covariance_snapshot_schedule(
        ARRAY_POSITIONS_M,
        fs_hz=FS_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        snapshot_length_samples=SNAPSHOT_LENGTH,
        beams_per_half=N_BEAM_PER_HALF,
    )
    accumulator = DirectionMatchedCovarianceAccumulator(schedule, integration_time_seconds=INTEGRATION_TIME_S)
    frequency_hz = np.asarray(np.fft.rfftfreq(SNAPSHOT_LENGTH, d=1.0 / FS_HZ), dtype=np.float32)
    signal_band_mask = (frequency_hz >= SOURCE_BAND_LOW_HZ) & (frequency_hz <= SOURCE_BAND_HIGH_HZ)
    snapshot_seconds = np.asarray(DISPLAY_SNAPSHOT_SECONDS, dtype=np.int32)
    snapshot_second_set = set(DISPLAY_SNAPSHOT_SECONDS)

    render_start = time.perf_counter()
    rendered = _render_evaluation_scene(receiver)
    render_elapsed_seconds = time.perf_counter() - render_start
    processing_elapsed_by_second: list[float] = []
    correlation_elapsed_by_snapshot: list[float] = []
    scalar_snapshots: dict[str, list[NDArray[np.float32]]] = {
        "maximum": [], "mean": [], "median": [], "percentile_95": []
    }
    physical_band_snapshots: dict[str, list[NDArray[np.float32]]] = {
        "mean": [], "median": [], "percentile_95": []
    }
    normalized_band_mean_snapshots: list[NDArray[np.float32]] = []
    composition_snapshots: dict[str, list[NDArray[np.float32]]] = {
        "mean": [], "median": [], "percentile_95": []
    }
    final_statistics = None
    samples_per_second = int(round(FS_HZ))
    for second_index in range(PROCESS_DURATION_S):
        frame = rendered[:, second_index * samples_per_second : (second_index + 1) * samples_per_second]
        process_start = time.perf_counter()
        accumulator.process_one_second(frame)
        processing_elapsed_by_second.append(time.perf_counter() - process_start)
        if second_index + 1 not in snapshot_second_set:
            continue
        correlation_start = time.perf_counter()
        statistics = calculate_sparse_array_spatial_correlation_statistics(
            accumulator.direction_covariance,
            ARRAY_POSITIONS_M,
            frequency_hz,
            CENTRAL_CHANNEL_MASK,
            sound_speed_m_s=SOUND_SPEED_M_S,
            physical_baseline_edges_m=PHYSICAL_BASELINE_EDGES_M,
            wavelength_normalized_edges=WAVELENGTH_NORMALIZED_EDGES,
        )
        correlation_elapsed_by_snapshot.append(time.perf_counter() - correlation_start)
        final_statistics = statistics
        global_tables = {
            "maximum": statistics.global_statistics.maximum,
            "mean": statistics.global_statistics.mean,
            "median": statistics.global_statistics.median,
            "percentile_95": statistics.global_statistics.percentile_95,
        }
        physical_tables = {
            "mean": statistics.physical_baseline.mean,
            "median": statistics.physical_baseline.median,
            "percentile_95": statistics.physical_baseline.percentile_95,
        }
        composition_tables = {
            "mean": statistics.pair_composition.mean,
            "median": statistics.pair_composition.median,
            "percentile_95": statistics.pair_composition.percentile_95,
        }
        for metric_name in scalar_snapshots:
            scalar_snapshots[metric_name].append(global_tables[metric_name].copy())
        for metric_name in physical_band_snapshots:
            physical_band_snapshots[metric_name].append(
                _finite_frequency_mean(physical_tables[metric_name], signal_band_mask)
            )
        for metric_name in composition_snapshots:
            composition_snapshots[metric_name].append(composition_tables[metric_name].copy())
        normalized_band_mean_snapshots.append(
            _finite_frequency_mean(statistics.wavelength_normalized_baseline.mean, signal_band_mask)
        )

    if final_statistics is None:
        raise RuntimeError("no correlation snapshot was evaluated.")
    processing_times = np.asarray(processing_elapsed_by_second, dtype=np.float64)
    correlation_times = np.asarray(correlation_elapsed_by_snapshot, dtype=np.float64)
    scalar_by_snapshot = {
        name: np.stack(values, axis=0).astype(np.float32) for name, values in scalar_snapshots.items()
    }
    physical_band_by_snapshot = {
        name: np.stack(values, axis=0).astype(np.float32) for name, values in physical_band_snapshots.items()
    }
    normalized_band_mean_by_snapshot = np.stack(normalized_band_mean_snapshots, axis=0).astype(np.float32)
    composition_by_snapshot = {
        name: np.stack(values, axis=0).astype(np.float32) for name, values in composition_snapshots.items()
    }

    azimuth_rad = np.deg2rad(schedule.global_direction_azimuth_deg.astype(np.float64))
    candidate_directions = np.stack(
        (np.cos(azimuth_rad), np.sin(azimuth_rad), np.zeros_like(azimuth_rad)),
        axis=1,
    )
    candidate_delay_s = relative_arrival_delay(
        ARRAY_POSITIONS_M,
        candidate_directions,
        sound_speed_m_per_s=SOUND_SPEED_M_S,
    )
    # 時間軸復元後の共分散は同時刻channel spectrumを表すため、物理到来遅延から作る
    # steering `[n_ch,n_direction,n_bin]`と直接比較できる。
    candidate_steering = steering_from_relative_delay(candidate_delay_s, frequency_hz)
    subspace_start = time.perf_counter()
    subspace_metrics = calculate_covariance_subspace_metrics(
        accumulator.direction_covariance,
        candidate_steering,
        direction_chunk_size=8,
    )
    subspace_elapsed_seconds = time.perf_counter() - subspace_start

    physical_metadata = final_statistics.physical_baseline
    normalized_metadata = final_statistics.wavelength_normalized_baseline
    composition = final_statistics.pair_composition
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "method3_sparse_spatial_correlation_statistics.npz",
        sensor_positions_m=ARRAY_POSITIONS_M,
        central_channel_mask=CENTRAL_CHANNEL_MASK,
        azimuth_deg=schedule.global_direction_azimuth_deg,
        frequency_hz=frequency_hz,
        snapshot_seconds=snapshot_seconds,
        global_mean=final_statistics.global_statistics.mean,
        global_maximum=final_statistics.global_statistics.maximum,
        global_median=final_statistics.global_statistics.median,
        global_percentile_95=final_statistics.global_statistics.percentile_95,
        global_mean_by_snapshot=scalar_by_snapshot["mean"],
        global_maximum_by_snapshot=scalar_by_snapshot["maximum"],
        global_median_by_snapshot=scalar_by_snapshot["median"],
        global_percentile_95_by_snapshot=scalar_by_snapshot["percentile_95"],
        physical_baseline_edges_m=physical_metadata.group_edges,
        physical_baseline_mean=physical_metadata.mean,
        physical_baseline_median=physical_metadata.median,
        physical_baseline_percentile_95=physical_metadata.percentile_95,
        physical_baseline_standard_deviation=physical_metadata.standard_deviation,
        physical_baseline_interquartile_range=physical_metadata.interquartile_range,
        physical_baseline_pair_count=physical_metadata.pair_count,
        physical_baseline_minimum_m=physical_metadata.value_minimum,
        physical_baseline_maximum_m=physical_metadata.value_maximum,
        physical_baseline_representative_m=physical_metadata.value_representative,
        physical_baseline_mean_signal_band_by_snapshot=physical_band_by_snapshot["mean"],
        physical_baseline_median_signal_band_by_snapshot=physical_band_by_snapshot["median"],
        physical_baseline_percentile_95_signal_band_by_snapshot=physical_band_by_snapshot["percentile_95"],
        wavelength_normalized_edges=normalized_metadata.group_edges,
        wavelength_normalized_mean=normalized_metadata.mean,
        wavelength_normalized_standard_deviation=normalized_metadata.standard_deviation,
        wavelength_normalized_interquartile_range=normalized_metadata.interquartile_range,
        wavelength_normalized_pair_count=normalized_metadata.pair_count,
        wavelength_normalized_minimum=normalized_metadata.value_minimum,
        wavelength_normalized_maximum=normalized_metadata.value_maximum,
        wavelength_normalized_representative=normalized_metadata.value_representative,
        wavelength_normalized_mean_signal_band_by_snapshot=normalized_band_mean_by_snapshot,
        pair_composition_names=np.asarray(composition.group_names),
        pair_composition_pair_count=composition.pair_count,
        pair_composition_mean=composition.mean,
        pair_composition_median=composition.median,
        pair_composition_percentile_95=composition.percentile_95,
        pair_composition_standard_deviation=composition.standard_deviation,
        pair_composition_interquartile_range=composition.interquartile_range,
        pair_composition_mean_by_snapshot=composition_by_snapshot["mean"],
        pair_composition_median_by_snapshot=composition_by_snapshot["median"],
        pair_composition_percentile_95_by_snapshot=composition_by_snapshot["percentile_95"],
        steering_power_fraction=subspace_metrics.steering_power_fraction,
        principal_eigenvector_alignment=subspace_metrics.principal_eigenvector_alignment,
        principal_eigenvalue_fraction=subspace_metrics.principal_eigenvalue_fraction,
        principal_to_noise_mean_ratio=subspace_metrics.principal_to_noise_mean_ratio,
        processing_elapsed_seconds=processing_times,
        correlation_elapsed_seconds=correlation_times,
    )

    display_names = {
        "maximum": "Maximum",
        "mean": "Mean",
        "median": "Median",
        "percentile_95": "95 percentile",
    }
    final_global_tables = {
        "maximum": final_statistics.global_statistics.maximum,
        "mean": final_statistics.global_statistics.mean,
        "median": final_statistics.global_statistics.median,
        "percentile_95": final_statistics.global_statistics.percentile_95,
    }
    for metric_name, display_name in display_names.items():
        _write_correlation_png(
            OUTPUT_DIR / f"global_{metric_name}_imagesc.png",
            schedule.global_direction_azimuth_deg,
            frequency_hz,
            final_global_tables[metric_name],
            title=f"Sparse 64ch method 3: all-pair {display_name.lower()} after 60 s",
            colorbar_label=f"All-pair {display_name.lower()} correlation [ratio]",
        )
        _write_time_evolution_png(
            OUTPUT_DIR / f"global_{metric_name}_time_evolution.png",
            schedule.global_direction_azimuth_deg,
            frequency_hz,
            snapshot_seconds,
            scalar_by_snapshot[metric_name],
            title=f"Sparse 64ch method 3 all-pair {display_name.lower()} evolution",
            colorbar_label=f"All-pair {display_name.lower()} correlation [ratio]",
        )

    valid_physical_group = physical_metadata.pair_count[0] > 0
    physical_representative_m = physical_metadata.value_representative[0, valid_physical_group]
    for metric_name, display_name in display_names.items():
        if metric_name == "maximum":
            # 基線binでは最大を要求していないため、全pair最大だけを共通比較へ含める。
            continue
        _write_group_trend_png(
            OUTPUT_DIR / f"physical_baseline_{metric_name}_signal_band_trend.png",
            schedule.global_direction_azimuth_deg,
            physical_representative_m,
            physical_band_by_snapshot[metric_name][:, :, valid_physical_group],
            snapshot_seconds,
            group_axis_label="Physical baseline length [m]",
            title=f"Physical-baseline {display_name.lower()}, 128–1024 Hz",
            colorbar_label=f"Baseline-bin {display_name.lower()} correlation [ratio]",
        )

    normalized_representative = 0.5 * (
        normalized_metadata.group_edges[:-1] + normalized_metadata.group_edges[1:]
    )
    valid_normalized_group = np.any(normalized_metadata.pair_count[signal_band_mask] > 0, axis=0)
    _write_group_trend_png(
        OUTPUT_DIR / "wavelength_normalized_baseline_mean_signal_band_trend.png",
        schedule.global_direction_azimuth_deg,
        normalized_representative[valid_normalized_group],
        normalized_band_mean_by_snapshot[:, :, valid_normalized_group],
        snapshot_seconds,
        group_axis_label="Normalized baseline d/λ [ratio]",
        title="Wavelength-normalized baseline mean, 128–1024 Hz",
        colorbar_label="Normalized-baseline mean correlation [ratio]",
    )

    _write_signal_frequency_azimuth_overlay(
        OUTPUT_DIR / "signal_frequency_global_statistics_azimuth_overlay.png",
        np.asarray(schedule.global_direction_azimuth_deg, dtype=np.float32),
        frequency_hz,
        np.asarray(signal_band_mask, dtype=np.bool_),
        {
            "maximum": final_global_tables["maximum"],
            "mean": final_global_tables["mean"],
            "median": final_global_tables["median"],
            "95 percentile": final_global_tables["percentile_95"],
        },
        title="60 s snapshot: all-pair correlation statistics at source-band frequencies",
    )
    _write_signal_frequency_azimuth_overlay(
        OUTPUT_DIR / "signal_frequency_pair_composition_mean_azimuth_overlay.png",
        np.asarray(schedule.global_direction_azimuth_deg, dtype=np.float32),
        frequency_hz,
        np.asarray(signal_band_mask, dtype=np.bool_),
        {
            composition_name: composition.mean[composition_index]
            for composition_index, composition_name in enumerate(composition.group_names)
        },
        title="60 s snapshot: pair-composition mean at source-band frequencies",
    )

    subspace_display_tables = {
        "steering power fraction": subspace_metrics.steering_power_fraction,
        "principal eigenvector alignment": subspace_metrics.principal_eigenvector_alignment,
        "principal eigenvalue fraction": subspace_metrics.principal_eigenvalue_fraction,
    }
    for metric_label, metric_table in subspace_display_tables.items():
        file_label = metric_label.replace(" ", "_")
        _write_correlation_png(
            OUTPUT_DIR / f"{file_label}_imagesc.png",
            schedule.global_direction_azimuth_deg,
            frequency_hz,
            metric_table,
            title=f"Sparse 64ch method 3: {metric_label} after 60 s",
            colorbar_label=f"{metric_label.title()} [ratio]",
        )
    _write_signal_frequency_azimuth_overlay(
        OUTPUT_DIR / "signal_frequency_steering_subspace_azimuth_overlay.png",
        np.asarray(schedule.global_direction_azimuth_deg, dtype=np.float32),
        frequency_hz,
        np.asarray(signal_band_mask, dtype=np.bool_),
        subspace_display_tables,
        title="60 s snapshot: complex steering and eigenspace metrics at source-band frequencies",
    )

    for composition_index, composition_name in enumerate(composition.group_names):
        for metric_name, display_name in display_names.items():
            if metric_name == "maximum":
                continue
            _write_correlation_png(
                OUTPUT_DIR / f"pair_{composition_name}_{metric_name}_imagesc.png",
                schedule.global_direction_azimuth_deg,
                frequency_hz,
                getattr(composition, metric_name)[composition_index],
                title=f"Pair composition {composition_name}: {display_name.lower()}",
                colorbar_label=f"{display_name} correlation [ratio]",
            )
            _write_time_evolution_png(
                OUTPUT_DIR / f"pair_{composition_name}_{metric_name}_time_evolution.png",
                schedule.global_direction_azimuth_deg,
                frequency_hz,
                snapshot_seconds,
                composition_by_snapshot[metric_name][:, composition_index],
                title=f"Pair composition {composition_name}: {display_name.lower()} evolution",
                colorbar_label=f"{display_name} correlation [ratio]",
            )

    source_neighborhood_mask = np.abs(schedule.global_direction_azimuth_deg - SOURCE_BEARING_DEG) <= 2.0
    far_direction_mask = np.abs(schedule.global_direction_azimuth_deg - SOURCE_BEARING_DEG) >= 20.0
    region_statistics: dict[str, dict[str, float]] = {}
    for metric_name, table in final_global_tables.items():
        source_value = float(np.mean(table[source_neighborhood_mask][:, signal_band_mask]))
        far_value = float(np.mean(table[far_direction_mask][:, signal_band_mask]))
        region_statistics[metric_name] = {
            "source_neighborhood": source_value,
            "far_direction": far_value,
            "source_minus_far": source_value - far_value,
        }
    composition_region_statistics: dict[str, dict[str, dict[str, float]]] = {}
    for composition_index, composition_name in enumerate(composition.group_names):
        composition_region_statistics[composition_name] = {}
        for metric_name in ("mean", "median", "percentile_95"):
            table = getattr(composition, metric_name)[composition_index]
            source_value = float(np.mean(table[source_neighborhood_mask][:, signal_band_mask]))
            far_value = float(np.mean(table[far_direction_mask][:, signal_band_mask]))
            composition_region_statistics[composition_name][metric_name] = {
                "source_neighborhood": source_value,
                "far_direction": far_value,
                "source_minus_far": source_value - far_value,
            }
    physical_source_mean = _finite_region_mean(
        physical_metadata.mean, source_neighborhood_mask, signal_band_mask
    )
    physical_far_mean = _finite_region_mean(
        physical_metadata.mean, far_direction_mask, signal_band_mask
    )
    physical_contrast = physical_source_mean - physical_far_mean
    valid_contrast_indices = np.flatnonzero(np.isfinite(physical_contrast))
    maximum_contrast_group_index = int(
        valid_contrast_indices[np.argmax(physical_contrast[valid_contrast_indices])]
    )
    valid_physical_indices = np.flatnonzero(physical_metadata.pair_count[0] > 0)
    subspace_region_statistics: dict[str, dict[str, float]] = {}
    subspace_tables_for_summary = {
        "steering_power_fraction": subspace_metrics.steering_power_fraction,
        "principal_eigenvector_alignment": subspace_metrics.principal_eigenvector_alignment,
        "principal_eigenvalue_fraction": subspace_metrics.principal_eigenvalue_fraction,
        "principal_to_noise_mean_ratio": subspace_metrics.principal_to_noise_mean_ratio,
    }
    for metric_name, table in subspace_tables_for_summary.items():
        source_value = float(np.mean(table[source_neighborhood_mask][:, signal_band_mask]))
        far_value = float(np.mean(table[far_direction_mask][:, signal_band_mask]))
        band_mean_by_direction = np.mean(table[:, signal_band_mask], axis=1)
        peak_direction_index = int(np.argmax(band_mean_by_direction))
        peak_azimuth_deg = float(schedule.global_direction_azimuth_deg[peak_direction_index])
        subspace_region_statistics[metric_name] = {
            "source_neighborhood": source_value,
            "far_direction": far_value,
            "source_minus_far": source_value - far_value,
            "band_mean_peak_azimuth_deg": peak_azimuth_deg,
            "band_mean_peak_error_deg": peak_azimuth_deg - SOURCE_BEARING_DEG,
            "band_mean_peak_value": float(band_mean_by_direction[peak_direction_index]),
        }
    summary = {
        "method": 3,
        "array_layout": "center_dense_outer_sparse_symmetric_64ch",
        "array_layout_source": "64ch reduction of OperationalSparseArrayDesignConfig staged-spacing concept",
        "sensor_positions_m": ARRAY_POSITIONS_M.astype(float).tolist(),
        "n_ch": N_CH,
        "central_channel_count": int(np.count_nonzero(CENTRAL_CHANNEL_MASK)),
        "outer_channel_count": int(np.count_nonzero(~CENTRAL_CHANNEL_MASK)),
        "array_aperture_m": ARRAY_APERTURE_M,
        "minimum_adjacent_spacing_m": MINIMUM_SPACING_M,
        "maximum_adjacent_spacing_m": MAXIMUM_SPACING_M,
        "integration_time_seconds": INTEGRATION_TIME_S,
        "ordinary_direction_effective_snapshot_count": 0.5 * INTEGRATION_TIME_S,
        "process_duration_seconds": PROCESS_DURATION_S,
        "display_snapshot_seconds": list(DISPLAY_SNAPSHOT_SECONDS),
        "correlation_formula": "abs(R_ij) / sqrt(R_ii * R_jj), i > j",
        "time_axis_restoration": {
            "enabled": True,
            "reference": "per-beam mean channel center sample",
            "formula": "exp(-j * 2*pi * f_k * (center_ch_beam-center_ref_beam) / fs)",
            "coefficient_reuse": "precomputed once for two center-table segments",
        },
        "physical_baseline_group_index": valid_physical_indices.tolist(),
        "physical_baseline_pair_count": physical_metadata.pair_count[0, valid_physical_indices].tolist(),
        "physical_baseline_minimum_m": physical_metadata.value_minimum[0, valid_physical_indices].astype(float).tolist(),
        "physical_baseline_maximum_m": physical_metadata.value_maximum[0, valid_physical_indices].astype(float).tolist(),
        "physical_baseline_representative_m": physical_metadata.value_representative[0, valid_physical_indices].astype(float).tolist(),
        "physical_baseline_dispersion_saved": ["standard_deviation", "interquartile_range"],
        "physical_baseline_maximum_mean_contrast": {
            "group_index": maximum_contrast_group_index,
            "representative_m": float(physical_metadata.value_representative[0, maximum_contrast_group_index]),
            "source_minus_far": float(physical_contrast[maximum_contrast_group_index]),
        },
        "wavelength_normalized_pair_count_shape": list(normalized_metadata.pair_count.shape),
        "wavelength_normalized_metadata_saved": ["pair_count", "minimum", "maximum", "representative"],
        "pair_composition_names": list(composition.group_names),
        "pair_composition_pair_count": composition.pair_count.tolist(),
        "region_statistics_128_1024_hz": region_statistics,
        "pair_composition_region_statistics_128_1024_hz": composition_region_statistics,
        "steering_and_subspace_region_statistics_128_1024_hz": subspace_region_statistics,
        "scene_render_elapsed_seconds": render_elapsed_seconds,
        "processing_elapsed_seconds_total": float(np.sum(processing_times)),
        "correlation_elapsed_seconds_total_for_snapshots": float(np.sum(correlation_times)),
        "subspace_elapsed_seconds": subspace_elapsed_seconds,
        "direction_covariance_storage_bytes": int(accumulator.direction_covariance.nbytes),
    }
    (OUTPUT_DIR / "correlation_statistics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    array_definition = {
        "layout": "center_dense_outer_sparse_symmetric_64ch",
        "source_design": "OperationalSparseArrayDesignConfig staged-spacing concept",
        "n_ch": N_CH,
        "positions_m": ARRAY_POSITIONS_M.astype(float).tolist(),
        "central_channel_mask": CENTRAL_CHANNEL_MASK.tolist(),
        "aperture_m": ARRAY_APERTURE_M,
        "minimum_adjacent_spacing_m": MINIMUM_SPACING_M,
        "maximum_adjacent_spacing_m": MAXIMUM_SPACING_M,
    }
    (OUTPUT_DIR / "sparse_array_definition.json").write_text(
        json.dumps(array_definition, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
