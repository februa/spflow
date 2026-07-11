"""方式3を10秒積分し、方位・周波数別の最大正規化相関を画像化する。"""

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
    calculate_maximum_spatial_correlation_table,
)
from spflow.beamforming.diagnostic_plotting import centers_to_edges, require_matplotlib  # noqa: E402

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 実行環境依存。
    plt = None


FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
PROCESS_DURATION_S = 60
INTEGRATION_TIME_S = 10.0
DISPLAY_SNAPSHOT_SECONDS = (1, 2, 5, 10, 20, 40, 60)
N_CH = 9
SPACING_M = 0.25
SNAPSHOT_LENGTH = 128
N_BEAM_PER_HALF = 159
SOURCE_BEARING_DEG = 40.0
SOURCE_BAND_LOW_HZ = 1000.0
SOURCE_BAND_HIGH_HZ = 4000.0
SOURCE_LEVEL_DB_RE_INPUT_RMS = 0.0
NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ = -32.0
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "method3_maximum_correlation"


def _render_evaluation_scene(receiver: Receiver) -> np.ndarray:
    """scene_rendererで広帯域sourceとCH無相関背景雑音を10秒生成する。

    Args:
        receiver: 9ch対称ULA受波器。

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


def _write_maximum_correlation_png(
    output_path: Path,
    azimuth_deg: np.ndarray,
    frequency_hz: np.ndarray,
    maximum_correlation: np.ndarray,
) -> None:
    """最大相関`[方位,周波数]`をimagesc相当の画像として保存する。"""

    matplotlib = require_matplotlib()
    azimuth_edges = centers_to_edges(np.asarray(azimuth_deg, dtype=np.float64))
    frequency_edges = centers_to_edges(np.asarray(frequency_hz, dtype=np.float64))
    figure, axis = matplotlib.subplots(figsize=(11.0, 5.5), constrained_layout=True)
    # table shapeは`[n_direction,n_bin]`。pcolormeshではyをfrequencyにするため転置する。
    image = axis.pcolormesh(
        azimuth_edges,
        frequency_edges,
        np.asarray(maximum_correlation, dtype=np.float32).T,
        shading="flat",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axis.axvline(SOURCE_BEARING_DEG, color="tab:red", linewidth=1.0, linestyle="--", label="source bearing")
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel("Frequency [Hz]")
    axis.set_title(
        f"Method 3: maximum off-diagonal correlation after {PROCESS_DURATION_S} s "
        f"(integration time {INTEGRATION_TIME_S:g} s)"
    )
    axis.legend(loc="upper right")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Maximum normalized correlation [ratio]")
    figure.savefig(output_path, dpi=160)
    matplotlib.close(figure)


def _write_time_evolution_png(
    output_path: Path,
    azimuth_deg: np.ndarray,
    frequency_hz: np.ndarray,
    snapshot_seconds: np.ndarray,
    maximum_correlation_by_time: np.ndarray,
) -> None:
    """複数処理時刻の最大相関imagescを同一色scaleで並べる。"""

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
            maximum_correlation_by_time[second - 1].T,
            shading="flat",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axis.axvline(SOURCE_BEARING_DEG, color="tab:red", linewidth=0.8, linestyle="--")
        axis.set_title(f"processed {second} s")
        axis.set_ylim(0.0, 5000.0)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Azimuth [deg]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Frequency [Hz]")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, location="right", shrink=0.95)
        colorbar.set_label("Maximum normalized correlation [ratio]")
    figure.suptitle("Method 3 maximum correlation evolution (integration time = 10 s)")
    figure.savefig(output_path, dpi=150)
    matplotlib.close(figure)


def main() -> None:
    """長時間sceneを方式3へ入力し、毎秒の最大相関表と時間変化画像を保存する。"""

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
    maximum_correlation_snapshots: list[np.ndarray] = []
    processing_elapsed_by_second: list[float] = []
    correlation_elapsed_by_second: list[float] = []
    current_maximum_correlation = np.zeros(
        (schedule.global_direction_azimuth_deg.size, SNAPSHOT_LENGTH // 2 + 1),
        dtype=np.float32,
    )
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
        time_table = calculate_maximum_spatial_correlation_table(
            update.active_direction_covariance,
            schedule.global_direction_azimuth_deg[update.global_direction_indices],
            fs_hz=FS_HZ,
        )
        correlation_elapsed_by_second.append(time.perf_counter() - correlation_start)
        # 非更新方位は共分散と同様に前回相関値を保持し、今回の159方位だけを置換する。
        current_maximum_correlation[update.global_direction_indices] = time_table.maximum_correlation
        maximum_correlation_snapshots.append(current_maximum_correlation.copy())

    processing_times = np.asarray(processing_elapsed_by_second, dtype=np.float64)
    correlation_times = np.asarray(correlation_elapsed_by_second, dtype=np.float64)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "method3_maximum_correlation_table.npz",
        azimuth_deg=schedule.global_direction_azimuth_deg,
        frequency_hz=frequency_hz,
        maximum_correlation=current_maximum_correlation,
        processed_seconds=np.arange(1, PROCESS_DURATION_S + 1, dtype=np.int32),
        maximum_correlation_by_time=np.stack(maximum_correlation_snapshots, axis=0).astype(np.float32),
        direction_update_coef=accumulator.direction_update_coef,
        processing_elapsed_seconds=processing_times,
        correlation_elapsed_seconds=correlation_times,
    )
    _write_maximum_correlation_png(
        OUTPUT_DIR / "method3_maximum_correlation_imagesc.png",
        schedule.global_direction_azimuth_deg,
        frequency_hz,
        current_maximum_correlation,
    )
    _write_time_evolution_png(
        OUTPUT_DIR / "method3_maximum_correlation_time_evolution.png",
        schedule.global_direction_azimuth_deg,
        frequency_hz,
        np.asarray(DISPLAY_SNAPSHOT_SECONDS, dtype=np.int32),
        np.stack(maximum_correlation_snapshots, axis=0),
    )
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
        "n_beam_per_half": N_BEAM_PER_HALF,
        "source_bearing_deg": SOURCE_BEARING_DEG,
        "source_band_hz": [SOURCE_BAND_LOW_HZ, SOURCE_BAND_HIGH_HZ],
        "source_level_db_re_input_rms": SOURCE_LEVEL_DB_RE_INPUT_RMS,
        "noise_level_db_re_input_rms_per_sqrt_hz": NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ,
        "ordinary_direction_update_rate_per_second": 0.5,
        "ordinary_direction_coef": float(accumulator.direction_update_coef[0]),
        "shared_90deg_update_rate_per_second": 1.0,
        "shared_90deg_coef": float(accumulator.direction_update_coef[N_BEAM_PER_HALF - 1]),
        "maximum_correlation_shape": list(current_maximum_correlation.shape),
        "scene_render_elapsed_seconds": render_elapsed_seconds,
        "processing_elapsed_seconds_total": float(np.sum(processing_times)),
        "processing_elapsed_seconds_median_per_input_second": float(np.median(processing_times)),
        "processing_elapsed_seconds_max_per_input_second": float(np.max(processing_times)),
        "processing_realtime_ratio": float(np.sum(processing_times) / PROCESS_DURATION_S),
        "correlation_elapsed_seconds_total": float(np.sum(correlation_times)),
        "correlation_elapsed_seconds_median_per_input_second": float(np.median(correlation_times)),
        "correlation_elapsed_seconds_max_per_input_second": float(np.max(correlation_times)),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
