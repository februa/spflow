"""長大ULAの相関方位曲線を信号周波数・帯域幅ごとに比較する。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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
    N_CHANNEL,
    PROCESS_DURATION_S,
    SOUND_SPEED_M_S,
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

OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "long_ula_frequency_correlation_comparison"
SOURCE_AZIMUTH_DEG = 60.0
NFFT = 256
INTEGRATION_TIME_S = 10.0
NOISE_SEED = 12100


@dataclass(frozen=True)
class FrequencyScenario:
    """相関方位曲線を比較する一つの信号周波数条件を表す。

    広帯域条件ではband_hz、単一tone条件ではtone_frequency_hzを保持する。
    信号生成や共分散積分そのものは責務に含めない。
    """

    name: str
    display_name: str
    evaluation_band_hz: tuple[float, float]
    center_frequency_hz: float
    tone_frequency_hz: float | None = None


SCENARIOS = (
    FrequencyScenario("broadband_40_60_hz", "Broadband 40-60 Hz", (40.0, 60.0), 50.0),
    FrequencyScenario("broadband_80_120_hz", "Broadband 80-120 Hz", (80.0, 120.0), 100.0),
    FrequencyScenario("broadband_160_240_hz", "Broadband 160-240 Hz", (160.0, 240.0), 200.0),
    # 4 Hz bin幅で100 Hzがbin中心へ一致するため、toneのleakageを評価帯域へ持ち込まない。
    FrequencyScenario("tone_100_hz", "Tone 100 Hz", (100.0, 100.0), 100.0, tone_frequency_hz=100.0),
)


def _summarize(
    table: NDArray[np.float32],
    azimuth_deg: NDArray[np.float32],
    frequency_mask: NDArray[np.bool_],
) -> dict[str, float]:
    """60度近傍と20度以上遠方の相関方位選択性を数値化する。"""

    # table shapeは`[n_direction,n_bin]`。周波数binを平均して方位curveへ縮約する。
    curve = np.asarray(np.mean(table[:, frequency_mask], axis=1), dtype=np.float32)
    direction_error_deg = np.abs(azimuth_deg - np.float32(SOURCE_AZIMUTH_DEG))
    near = direction_error_deg <= 2.0
    far = direction_error_deg >= 20.0
    peak_index = int(np.argmax(curve))
    near_mean = float(np.mean(curve[near]))
    far_mean = float(np.mean(curve[far]))
    far_maximum = float(np.max(curve[far]))
    return {
        "source_neighborhood_mean": near_mean,
        "far_mean": far_mean,
        "far_maximum": far_maximum,
        "source_far_mean_margin": near_mean - far_mean,
        "source_far_peak_margin": near_mean - far_maximum,
        "peak_azimuth_deg": float(azimuth_deg[peak_index]),
        "peak_error_deg": float(abs(azimuth_deg[peak_index] - SOURCE_AZIMUTH_DEG)),
    }


def _correlation_tables(
    covariance: NDArray[np.complex64],
    positions_m: NDArray[np.float32],
    frequency_hz: NDArray[np.float32],
    scenario: FrequencyScenario,
) -> tuple[dict[str, NDArray[np.float32]], NDArray[np.float32]]:
    """周波数条件に応じた波長正規化基線別相関tableを計算する。"""

    wavelength_m = SOUND_SPEED_M_S / scenario.center_frequency_hz
    baseline_edges_m = np.asarray([0.0, wavelength_m, 8.0 * wavelength_m, APERTURE_M + 1.0], dtype=np.float32)
    central_mask = np.zeros(N_CHANNEL, dtype=np.bool_)
    central_mask[N_CHANNEL // 4 : 3 * N_CHANNEL // 4] = True
    statistics = calculate_sparse_array_spatial_correlation_statistics(
        covariance,
        positions_m,
        frequency_hz,
        central_mask,
        sound_speed_m_s=SOUND_SPEED_M_S,
        physical_baseline_edges_m=baseline_edges_m,
        wavelength_normalized_edges=np.linspace(0.0, 140.0, 71, dtype=np.float32),
    )
    tables = {
        "all_pair_maximum": statistics.global_statistics.maximum,
        "all_pair_mean": statistics.global_statistics.mean,
        "all_pair_median": statistics.global_statistics.median,
        "all_pair_percentile_95": statistics.global_statistics.percentile_95,
    }
    baseline_labels = ("short_lt_1lambda", "middle_1_to_8lambda", "long_ge_8lambda")
    for baseline_index, label in enumerate(baseline_labels):
        # binned statistics shapeは`[n_direction,n_bin,n_baseline]`。基線軸を選択する。
        tables[f"{label}_mean"] = statistics.physical_baseline.mean[:, :, baseline_index]
        tables[f"{label}_median"] = statistics.physical_baseline.median[:, :, baseline_index]
        tables[f"{label}_percentile_95"] = statistics.physical_baseline.percentile_95[:, :, baseline_index]
    return tables, baseline_edges_m


def _plot(
    scenario: FrequencyScenario,
    azimuth_deg: NDArray[np.float32],
    frequency_mask: NDArray[np.bool_],
    tables: dict[str, NDArray[np.float32]],
) -> None:
    """一つの周波数条件を前図と同じ4 panel表示で保存する。"""

    plt = require_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    groups = (
        ("All channel pairs", ("all_pair_maximum", "all_pair_mean", "all_pair_median", "all_pair_percentile_95")),
        ("Short baselines: d < 1 wavelength", tuple(name for name in tables if name.startswith("short_"))),
        ("Middle baselines: 1 <= d/lambda < 8", tuple(name for name in tables if name.startswith("middle_"))),
        ("Long baselines: d/lambda >= 8", tuple(name for name in tables if name.startswith("long_"))),
    )
    for axis, (title, names) in zip(axes.flat, groups, strict=True):
        for name in names:
            axis.plot(azimuth_deg, np.mean(tables[name][:, frequency_mask], axis=1), label=name)
        axis.axvline(SOURCE_AZIMUTH_DEG, color="black", linestyle="--", linewidth=1.0)
        axis.set(
            title=title,
            xlabel="Azimuth [deg]",
            ylabel="Normalized correlation [ratio]",
            xlim=(0.0, 180.0),
            ylim=(0.0, 1.0),
        )
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle(f"{scenario.display_name}, source azimuth 60 deg, target + noise")
    figure.savefig(OUTPUT_DIR / f"{scenario.name}_direction_curves.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """4周波数条件のtarget+noise相関方位曲線と数値比較を生成する。"""

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
    # steering shapeを積分器契約`[n_ch,n_bin,n_direction]`へ並べ替える。
    steering = np.transpose(steering_from_relative_delay(delays_s, frequency_hz), (0, 2, 1)).astype(np.complex64)
    results: dict[str, Any] = {}

    for scenario_index, scenario in enumerate(SCENARIOS):
        signal = _render(
            positions,
            noise=True,
            seed=NOISE_SEED + scenario_index * 100,
            source_azimuth_deg=SOURCE_AZIMUTH_DEG,
            source_band_hz=scenario.evaluation_band_hz,
            tone_frequency_hz=scenario.tone_frequency_hz,
        )
        accumulator = DirectionMatchedCovarianceAccumulator(
            schedule,
            integration_time_seconds=INTEGRATION_TIME_S,
            steering_table=steering,
        )
        for second in range(PROCESS_DURATION_S):
            accumulator.process_one_second(signal[:, second * int(FS_HZ) : (second + 1) * int(FS_HZ)])

        if scenario.tone_frequency_hz is None:
            frequency_mask = (frequency_hz >= scenario.evaluation_band_hz[0]) & (
                frequency_hz <= scenario.evaluation_band_hz[1]
            )
        else:
            # 100 Hzは4 Hz bin中心に一致するため、tone binだけを評価する。
            frequency_mask = np.isclose(frequency_hz, scenario.tone_frequency_hz)
        if not bool(np.any(frequency_mask)):
            raise RuntimeError(f"evaluation frequency bin is empty: {scenario.name}")

        tables, baseline_edges_m = _correlation_tables(
            accumulator.direction_covariance,
            positions,
            frequency_hz,
            scenario,
        )
        _plot(scenario, azimuth_deg, frequency_mask, tables)
        results[scenario.name] = {
            "display_name": scenario.display_name,
            "evaluation_frequency_bins_hz": frequency_hz[frequency_mask].tolist(),
            "center_frequency_hz": scenario.center_frequency_hz,
            "baseline_edges_m": baseline_edges_m.tolist(),
            "spacing_over_wavelength_at_center": float(6.25 * scenario.center_frequency_hz / SOUND_SPEED_M_S),
            "spatial_alias_free_at_center": bool(6.25 <= SOUND_SPEED_M_S / (2.0 * scenario.center_frequency_hz)),
            "metrics": {name: _summarize(table, azimuth_deg, frequency_mask) for name, table in tables.items()},
        }

    payload = {
        "array": {"n_channel": N_CHANNEL, "spacing_m": 6.25, "aperture_m": APERTURE_M},
        "source_azimuth_deg": SOURCE_AZIMUTH_DEG,
        "sample_rate_hz": FS_HZ,
        "nfft": NFFT,
        "fft_bin_width_hz": FS_HZ / NFFT,
        "integration_time_s": INTEGRATION_TIME_S,
        "results": results,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
