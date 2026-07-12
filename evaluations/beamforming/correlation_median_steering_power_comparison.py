"""全channel pair相関中央値とsteering powerの方位選択性を比較する。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from spflow.beamforming import (  # noqa: E402
    DirectionMatchedCovarianceAccumulator,
    build_two_second_covariance_snapshot_schedule,
    calculate_spatial_correlation_statistics,
    relative_arrival_delay,
    steering_from_relative_delay,
)
from spflow.beamforming.diagnostic_plotting import require_matplotlib  # noqa: E402
from evaluations.beamforming.long_ula_frequency_correlation_comparison import (  # noqa: E402
    FS_HZ,
    INTEGRATION_TIME_S,
    NFFT,
    SCENARIOS,
    SOURCE_AZIMUTH_DEG,
    FrequencyScenario,
)
from evaluations.beamforming.low_frequency_long_ula_covariance import (  # noqa: E402
    PROCESS_DURATION_S,
    SOUND_SPEED_M_S,
    _positions,
    _render,
)


OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "correlation_median_steering_power_comparison"
SNAPSHOT_SECONDS = (2, 4, 6, 8, 10)
BASE_SEED = 13200


def _frequency_mask(
    scenario: FrequencyScenario,
    frequency_hz: NDArray[np.float32],
) -> NDArray[np.bool_]:
    """sceneの評価周波数bin mask`[n_bin]`を返す。"""

    if scenario.tone_frequency_hz is not None:
        return np.asarray(np.isclose(frequency_hz, scenario.tone_frequency_hz), dtype=np.bool_)
    return np.asarray(
        (frequency_hz >= scenario.evaluation_band_hz[0]) & (frequency_hz <= scenario.evaluation_band_hz[1]),
        dtype=np.bool_,
    )


def _curve_metrics(
    table: NDArray[np.float32],
    azimuth_deg: NDArray[np.float32],
    frequency_mask: NDArray[np.bool_],
) -> dict[str, float]:
    """方位curveのpeak、margin、半高幅を数値化する。

    Args:
        table: 指標table。shapeは`[n_direction,n_bin]`、範囲は0--1。
        azimuth_deg: 方位軸。shapeは`[n_direction]`、単位はdegree。
        frequency_mask: 評価bin mask。shapeは`[n_bin]`。

    Returns:
        source近傍値、遠方値、margin、peak誤差、半高幅を持つ辞書。

    Raises:
        ValueError: 評価周波数binが空の場合。

    境界条件:
        高いnoise floorを持つ相関中央値にも適用できるよう、半高幅は0.5絶対値ではなく
        curve最小値と最大値の中点を閾値とする。
    """

    if not bool(np.any(frequency_mask)):
        raise ValueError("frequency_mask must select at least one bin.")
    curve = np.asarray(np.mean(table[:, frequency_mask], axis=1), dtype=np.float32)
    error_deg = np.abs(azimuth_deg - np.float32(SOURCE_AZIMUTH_DEG))
    near = error_deg <= 2.0
    far = error_deg >= 20.0
    peak_index = int(np.argmax(curve))
    near_mean = float(np.mean(curve[near]))
    far_mean = float(np.mean(curve[far]))
    far_maximum = float(np.max(curve[far]))

    half_height = float(np.min(curve) + 0.5 * (np.max(curve) - np.min(curve)))
    source_index = int(np.argmin(error_deg))
    left_index = source_index
    right_index = source_index
    while left_index > 0 and curve[left_index - 1] >= half_height:
        left_index -= 1
    while right_index + 1 < curve.size and curve[right_index + 1] >= half_height:
        right_index += 1
    return {
        "source_neighborhood_mean": near_mean,
        "far_mean": far_mean,
        "far_maximum": far_maximum,
        "source_far_mean_margin": near_mean - far_mean,
        "source_far_peak_margin": near_mean - far_maximum,
        "peak_azimuth_deg": float(azimuth_deg[peak_index]),
        "peak_error_deg": float(abs(azimuth_deg[peak_index] - SOURCE_AZIMUTH_DEG)),
        "half_prominence_width_deg": float(azimuth_deg[right_index] - azimuth_deg[left_index]),
        "direction_range": float(np.max(curve) - np.min(curve)),
    }


def _snapshot_tables(
    accumulator: DirectionMatchedCovarianceAccumulator,
) -> dict[str, NDArray[np.float32]]:
    """同じ完成snapshotから全pair中央値と直接積分etaを返す。"""

    correlation = calculate_spatial_correlation_statistics(accumulator.direction_covariance)
    steering_metrics = accumulator.completed_steering_metrics()
    # 両tableともshapeは`[n_direction,n_bin]`。同じsnapshot順序と忘却係数で完成している。
    return {
        "all_pair_correlation_median": correlation.median,
        "steering_power_eta": steering_metrics.eta,
    }


def _plot_scenario(
    scenario: FrequencyScenario,
    azimuth_deg: NDArray[np.float32],
    frequency_mask: NDArray[np.bool_],
    final_tables: dict[str, dict[str, NDArray[np.float32]]],
) -> None:
    """3 sceneの最終curveとtarget+noise正規化curveを同一図へ保存する。"""

    plt = require_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    panel_specs = (
        ("target_only", "Target only: raw ratio", False),
        ("target_plus_noise", "Target + noise: raw ratio", False),
        ("target_plus_noise", "Target + noise: normalized by source value", True),
        ("noise_only", "Noise only: raw ratio", False),
    )
    for axis, (scene_name, title, normalize) in zip(axes.flat, panel_specs, strict=True):
        for metric_name, table in final_tables[scene_name].items():
            curve = np.asarray(np.mean(table[:, frequency_mask], axis=1), dtype=np.float32)
            if normalize:
                source_index = int(np.argmin(np.abs(azimuth_deg - np.float32(SOURCE_AZIMUTH_DEG))))
                denominator = max(float(curve[source_index]), float(np.finfo(np.float32).tiny))
                curve = np.asarray(curve / denominator, dtype=np.float32)
            axis.plot(azimuth_deg, curve, label=metric_name)
        axis.axvline(SOURCE_AZIMUTH_DEG, color="black", linestyle="--", linewidth=1.0)
        axis.set(
            title=title,
            xlabel="Azimuth [deg]",
            ylabel="Ratio" if not normalize else "Ratio re source direction",
            xlim=(0.0, 180.0),
            ylim=(0.0, 1.05 if not normalize else 1.25),
        )
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"{scenario.display_name}: all-pair median vs steering power eta")
    figure.savefig(OUTPUT_DIR / f"{scenario.name}_comparison.png", dpi=160)
    plt.close(figure)


def _plot_time_margin(
    scenario: FrequencyScenario,
    scene_results: dict[str, dict[str, Any]],
) -> None:
    """target+noiseの方位margin時間変化を2指標で比較する。"""

    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    snapshots = scene_results["target_plus_noise"]["snapshots"]
    for metric_name in ("all_pair_correlation_median", "steering_power_eta"):
        margins = [snapshots[str(second)][metric_name]["source_far_mean_margin"] for second in SNAPSHOT_SECONDS]
        axis.plot(SNAPSHOT_SECONDS, margins, marker="o", label=metric_name)
    axis.set(
        title=f"{scenario.display_name}: target + noise margin convergence",
        xlabel="Integration elapsed time [s]",
        ylabel="Source - far mean ratio",
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(OUTPUT_DIR / f"{scenario.name}_margin_convergence.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """4周波数条件・3 sceneで全pair中央値とsteering powerを比較する。"""

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
    # steering table shapeを積分器契約`[n_ch,n_bin,n_direction]`へ固定する。
    steering = np.transpose(steering_from_relative_delay(delays_s, frequency_hz), (0, 2, 1)).astype(np.complex64)
    results: dict[str, Any] = {}

    for scenario_index, scenario in enumerate(SCENARIOS):
        seed = BASE_SEED + 100 * scenario_index
        target = _render(
            positions,
            noise=False,
            seed=seed,
            source_azimuth_deg=SOURCE_AZIMUTH_DEG,
            source_band_hz=scenario.evaluation_band_hz,
            tone_frequency_hz=scenario.tone_frequency_hz,
        )
        target_plus_noise = _render(
            positions,
            noise=True,
            seed=seed,
            source_azimuth_deg=SOURCE_AZIMUTH_DEG,
            source_band_hz=scenario.evaluation_band_hz,
            tone_frequency_hz=scenario.tone_frequency_hz,
        )
        # 同じseedのtargetを差し引き、target+noiseと同一realizationのambientだけを得る。
        scenes = {
            "target_only": target,
            "noise_only": np.asarray(target_plus_noise - target, dtype=np.float32),
            "target_plus_noise": target_plus_noise,
        }
        frequency_mask = _frequency_mask(scenario, frequency_hz)
        if not bool(np.any(frequency_mask)):
            raise RuntimeError(f"evaluation frequency bin is empty: {scenario.name}")
        scenario_results: dict[str, dict[str, Any]] = {}
        final_tables: dict[str, dict[str, NDArray[np.float32]]] = {}

        for scene_name, signal in scenes.items():
            accumulator = DirectionMatchedCovarianceAccumulator(
                schedule,
                integration_time_seconds=INTEGRATION_TIME_S,
                steering_table=steering,
            )
            snapshot_results: dict[str, dict[str, dict[str, float]]] = {}
            tables: dict[str, NDArray[np.float32]] = {}
            for second in range(1, PROCESS_DURATION_S + 1):
                accumulator.process_one_second(signal[:, (second - 1) * int(FS_HZ) : second * int(FS_HZ)])
                if second not in SNAPSHOT_SECONDS:
                    continue
                tables = _snapshot_tables(accumulator)
                snapshot_results[str(second)] = {
                    metric_name: _curve_metrics(table, azimuth_deg, frequency_mask)
                    for metric_name, table in tables.items()
                }
            final_tables[scene_name] = tables
            scenario_results[scene_name] = {"snapshots": snapshot_results}

        _plot_scenario(scenario, azimuth_deg, frequency_mask, final_tables)
        _plot_time_margin(scenario, scenario_results)
        results[scenario.name] = {
            "display_name": scenario.display_name,
            "evaluation_frequency_bins_hz": frequency_hz[frequency_mask].tolist(),
            "scenes": scenario_results,
        }

    payload = {
        "comparison": ["all_pair_correlation_median", "steering_power_eta"],
        "source_azimuth_deg": SOURCE_AZIMUTH_DEG,
        "sample_rate_hz": FS_HZ,
        "nfft": NFFT,
        "integration_time_s": INTEGRATION_TIME_S,
        "snapshot_seconds": list(SNAPSHOT_SECONDS),
        "results": results,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
