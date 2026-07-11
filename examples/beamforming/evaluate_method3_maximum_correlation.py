"""方式3を10秒積分し、方位・周波数別の最大正規化相関を画像化する。"""

from __future__ import annotations

import json
import sys
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
DURATION_S = 10
N_CH = 9
SPACING_M = 0.25
SNAPSHOT_LENGTH = 128
N_BEAM_PER_HALF = 159
SOURCE_BEARING_DEG = 40.0
SOURCE_BAND_LOW_HZ = 1000.0
SOURCE_BAND_HIGH_HZ = 4000.0
SOURCE_LEVEL_DB_RE_INPUT_RMS = 0.0
NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ = -32.0
UPDATE_COEF = 0.25
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "method3_maximum_correlation"


def _render_evaluation_scene(receiver: Receiver) -> np.ndarray:
    """scene_rendererで広帯域sourceとCH無相関背景雑音を10秒生成する。

    Args:
        receiver: 9ch対称ULA受波器。

    Returns:
        受信信号。shapeは`[n_ch,10*fs]`、dtypeは`float32`。
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
    sample_count = int(round(FS_HZ * DURATION_S))
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
    axis.set_title("Method 3: maximum off-diagonal spatial correlation after 10 s")
    axis.legend(loc="upper right")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Maximum normalized correlation [ratio]")
    figure.savefig(output_path, dpi=160)
    matplotlib.close(figure)


def main() -> None:
    """10秒sceneを方式3へ入力し、最大相関表・画像・条件を保存する。"""

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
    accumulator = DirectionMatchedCovarianceAccumulator(schedule, coef=UPDATE_COEF)
    rendered = _render_evaluation_scene(receiver)
    samples_per_second = int(round(FS_HZ))
    for second_index in range(DURATION_S):
        frame = rendered[:, second_index * samples_per_second : (second_index + 1) * samples_per_second]
        accumulator.process_one_second(frame)

    table = calculate_maximum_spatial_correlation_table(
        accumulator.direction_covariance,
        schedule.global_direction_azimuth_deg,
        fs_hz=FS_HZ,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "method3_maximum_correlation_table.npz",
        azimuth_deg=table.azimuth_deg,
        frequency_hz=table.frequency_hz,
        maximum_correlation=table.maximum_correlation,
    )
    _write_maximum_correlation_png(
        OUTPUT_DIR / "method3_maximum_correlation_imagesc.png",
        table.azimuth_deg,
        table.frequency_hz,
        table.maximum_correlation,
    )
    summary = {
        "method": 3,
        "integration_duration_s": DURATION_S,
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
        "coef": UPDATE_COEF,
        "maximum_correlation_shape": list(table.maximum_correlation.shape),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
