"""100 Hz帯長大ULAで正規化相関の集約方式を比較する。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from evaluations.beamforming.low_frequency_long_ula_covariance import (  # noqa: E402
    APERTURE_M,
    FS_HZ,
    INTEGRATION_TIME_S,
    N_CHANNEL,
    NFFT,
    PROCESS_DURATION_S,
    SOUND_SPEED_M_S,
    SOURCE_HIGH_HZ,
    SOURCE_LOW_HZ,
    _positions,
    _render,
)
from spflow.beamforming import (  # noqa: E402
    DirectionMatchedCovarianceAccumulator,
    build_two_second_covariance_snapshot_schedule,
    calculate_sparse_array_spatial_correlation_statistics,
    relative_arrival_delay,
    steering_from_relative_delay,
)
from spflow.beamforming_evaluation.diagnostic_plotting import require_matplotlib  # noqa: E402

SOURCE_AZIMUTH_DEG = 60.0
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "long_ula_correlation_method_comparison_source_60deg"
SNAPSHOT_SECONDS = (2, 4, 6, 8, 10)
# 100 Hzの波長15 mを基準に、短・中・長基線の時間差効果を分離する。
BASELINE_EDGES_M = np.asarray([0.0, 15.0, 120.0, APERTURE_M + 1.0], dtype=np.float32)
BASELINE_LABELS = ("short_lt_1lambda", "middle_1_to_8lambda", "long_ge_8lambda")


def _summarize_curve(
    table: NDArray[np.float32],
    azimuth_deg: NDArray[np.float32],
    frequency_band: NDArray[np.bool_],
) -> dict[str, float]:
    """相関tableの帯域平均方位選択性を要約する。

    Args:
        table: 正規化相関。shapeは`[n_direction,n_bin]`、単位は無次元比。
        azimuth_deg: 方位軸。shapeは`[n_direction]`、単位はdegree。
        frequency_band: 評価周波数bin mask。shapeは`[n_bin]`。

    Returns:
        source近傍、遠方平均、遠方最大、margin、peak誤差を持つ辞書。

    Raises:
        ValueError: 評価帯域または方位領域が空の場合。

    境界条件:
        source近傍を入力方位の±2度、遠方を入力方位から20度以上として比較する。
    """

    if not bool(np.any(frequency_band)):
        raise ValueError("frequency_band must select at least one bin.")
    curve = np.asarray(np.mean(table[:, frequency_band], axis=1), dtype=np.float32)
    direction_error_deg = np.abs(azimuth_deg - np.float32(SOURCE_AZIMUTH_DEG))
    source_mask = direction_error_deg <= 2.0
    far_mask = direction_error_deg >= 20.0
    if not bool(np.any(source_mask)) or not bool(np.any(far_mask)):
        raise ValueError("azimuth axis must contain source and far regions.")
    source_mean = float(np.mean(curve[source_mask]))
    far_mean = float(np.mean(curve[far_mask]))
    far_maximum = float(np.max(curve[far_mask]))
    peak_index = int(np.argmax(curve))
    return {
        "source_neighborhood_mean": source_mean,
        "far_mean": far_mean,
        "far_maximum": far_maximum,
        "source_far_mean_margin": source_mean - far_mean,
        "source_far_peak_margin": source_mean - far_maximum,
        "peak_azimuth_deg": float(azimuth_deg[peak_index]),
        "peak_error_deg": float(abs(azimuth_deg[peak_index] - SOURCE_AZIMUTH_DEG)),
        "all_direction_range": float(np.max(curve) - np.min(curve)),
    }


def _tables_from_covariance(
    covariance: NDArray[np.complex64],
    positions_m: NDArray[np.float32],
    frequency_hz: NDArray[np.float32],
) -> dict[str, NDArray[np.float32]]:
    """全pairおよび物理基線別の相関集約tableを返す。

    Args:
        covariance: 方位別共分散。shapeは`[n_direction,n_ch,n_ch,n_bin]`。
        positions_m: 受波器位置。shapeは`[n_ch,3]`、単位はm。
        frequency_hz: 周波数軸。shapeは`[n_bin]`、単位はHz。

    Returns:
        集約方式名から相関table`[n_direction,n_bin]`への対応。

    Raises:
        ValueError: 共分散と位置・周波数軸のshapeが一致しない場合。

    境界条件:
        等間隔ULAでもchannel index差ではなく座標から得た物理基線binを使用する。
    """

    central_mask = np.zeros(N_CHANNEL, dtype=np.bool_)
    central_mask[N_CHANNEL // 4 : 3 * N_CHANNEL // 4] = True
    statistics = calculate_sparse_array_spatial_correlation_statistics(
        covariance,
        positions_m,
        frequency_hz,
        central_mask,
        sound_speed_m_s=SOUND_SPEED_M_S,
        physical_baseline_edges_m=BASELINE_EDGES_M,
        wavelength_normalized_edges=np.linspace(0.0, 140.0, 71, dtype=np.float32),
    )
    tables: dict[str, NDArray[np.float32]] = {
        "all_pair_maximum": statistics.global_statistics.maximum,
        "all_pair_mean": statistics.global_statistics.mean,
        "all_pair_median": statistics.global_statistics.median,
        "all_pair_percentile_95": statistics.global_statistics.percentile_95,
    }
    for baseline_index, baseline_label in enumerate(BASELINE_LABELS):
        # binned table shapeは`[n_direction,n_bin,n_baseline]`なので、最後の基線軸を選択する。
        tables[f"{baseline_label}_mean"] = statistics.physical_baseline.mean[:, :, baseline_index]
        tables[f"{baseline_label}_median"] = statistics.physical_baseline.median[:, :, baseline_index]
        tables[f"{baseline_label}_percentile_95"] = statistics.physical_baseline.percentile_95[
            :, :, baseline_index
        ]
    return tables


def _plot_final_curves(
    azimuth_deg: NDArray[np.float32],
    frequency_band: NDArray[np.bool_],
    final_tables: dict[str, NDArray[np.float32]],
) -> None:
    """target+noiseの最終方位curveを集約種別ごとに保存する。"""

    plt = require_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    groups = (
        ("All channel pairs", ("all_pair_maximum", "all_pair_mean", "all_pair_median", "all_pair_percentile_95")),
        ("Short baselines: d < 1 wavelength", tuple(name for name in final_tables if name.startswith("short_"))),
        ("Middle baselines: 1 <= d/lambda < 8", tuple(name for name in final_tables if name.startswith("middle_"))),
        ("Long baselines: d/lambda >= 8", tuple(name for name in final_tables if name.startswith("long_"))),
    )
    for axis, (title, names) in zip(axes.flat, groups, strict=True):
        for name in names:
            axis.plot(azimuth_deg, np.mean(final_tables[name][:, frequency_band], axis=1), label=name)
        axis.axvline(SOURCE_AZIMUTH_DEG, color="black", linestyle="--", linewidth=1.0)
        axis.set(title=title, xlabel="Azimuth [deg]", ylabel="Normalized correlation [ratio]", ylim=(0.0, 1.0))
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    figure.savefig(OUTPUT_DIR / "target_plus_noise_final_direction_curves.png", dpi=160)
    plt.close(figure)


def _plot_margin_convergence(scene_output: dict[str, Any]) -> None:
    """target+noiseのsource対遠方平均marginの時間収束を保存する。"""

    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(10.0, 5.5), constrained_layout=True)
    selected_methods = (
        "all_pair_mean",
        "all_pair_median",
        "all_pair_percentile_95",
        "long_ge_8lambda_mean",
        "long_ge_8lambda_median",
        "long_ge_8lambda_percentile_95",
    )
    snapshots = scene_output["snapshots"]
    for method_name in selected_methods:
        margins = [snapshots[str(second)][method_name]["source_far_mean_margin"] for second in SNAPSHOT_SECONDS]
        axis.plot(SNAPSHOT_SECONDS, margins, marker="o", label=method_name)
    axis.set(
        title="Target + noise: correlation direction margin convergence",
        xlabel="Integration elapsed time [s]",
        ylabel="Source - far mean correlation [ratio]",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(OUTPUT_DIR / "target_plus_noise_margin_convergence.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """3 sceneと5 snapshot時刻で相関集約方式を比較し、JSONと画像を保存する。"""

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
    # steering table shapeは`[n_ch,n_bin,n_direction]`。積分器の固定契約へ合わせる。
    steering = np.transpose(steering_from_relative_delay(delays_s, frequency_hz), (0, 2, 1)).astype(np.complex64)
    frequency_band = (frequency_hz >= SOURCE_LOW_HZ) & (frequency_hz <= SOURCE_HIGH_HZ)

    target_only = _render(positions, noise=False, seed=10100, source_azimuth_deg=SOURCE_AZIMUTH_DEG)
    target_plus_noise = _render(positions, noise=True, seed=10200, source_azimuth_deg=SOURCE_AZIMUTH_DEG)
    # 同一seedのtarget成分を差し引き、target+noiseと同じambient realizationだけを分離する。
    noise_only = target_plus_noise - _render(
        positions,
        noise=False,
        seed=10200,
        source_azimuth_deg=SOURCE_AZIMUTH_DEG,
    )
    scenes = {"target_only": target_only, "noise_only": noise_only, "target_plus_noise": target_plus_noise}
    output: dict[str, Any] = {}
    final_target_plus_noise_tables: dict[str, NDArray[np.float32]] | None = None

    for scene_name, signal in scenes.items():
        accumulator = DirectionMatchedCovarianceAccumulator(
            schedule,
            integration_time_seconds=INTEGRATION_TIME_S,
            steering_table=steering,
        )
        snapshot_summaries: dict[str, dict[str, dict[str, float]]] = {}
        final_tables: dict[str, NDArray[np.float32]] = {}
        for second in range(1, PROCESS_DURATION_S + 1):
            # input shapeは`[n_ch,n_sample_per_second]`。1秒境界で完成済み共分散だけを観測する。
            accumulator.process_one_second(signal[:, (second - 1) * int(FS_HZ) : second * int(FS_HZ)])
            if second not in SNAPSHOT_SECONDS:
                continue
            final_tables = _tables_from_covariance(accumulator.direction_covariance, positions, frequency_hz)
            snapshot_summaries[str(second)] = {
                name: _summarize_curve(table, azimuth_deg, frequency_band) for name, table in final_tables.items()
            }
        if scene_name == "target_plus_noise":
            final_target_plus_noise_tables = final_tables

        method_stability: dict[str, dict[str, float]] = {}
        for method_name in final_tables:
            margins = np.asarray(
                [snapshot_summaries[str(second)][method_name]["source_far_mean_margin"] for second in SNAPSHOT_SECONDS],
                dtype=np.float64,
            )
            method_stability[method_name] = {
                "margin_mean": float(np.mean(margins)),
                "margin_standard_deviation": float(np.std(margins)),
            }
        output[scene_name] = {"snapshots": snapshot_summaries, "time_stability": method_stability}

    if final_target_plus_noise_tables is None:
        raise RuntimeError("target_plus_noise result was not completed.")
    _plot_final_curves(azimuth_deg, frequency_band, final_target_plus_noise_tables)
    _plot_margin_convergence(output["target_plus_noise"])

    payload = {
        "array": {"n_channel": N_CHANNEL, "spacing_m": 6.25, "aperture_m": APERTURE_M},
        "signal_band_hz": [SOURCE_LOW_HZ, SOURCE_HIGH_HZ],
        "source_azimuth_deg": SOURCE_AZIMUTH_DEG,
        "snapshot_seconds": list(SNAPSHOT_SECONDS),
        "correlation_formula": "abs(R_ij) / sqrt(R_ii * R_jj), i > j",
        "physical_baseline_edges_m": BASELINE_EDGES_M.tolist(),
        "scenes": output,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
