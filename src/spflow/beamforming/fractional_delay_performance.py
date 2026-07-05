"""小数遅延導入後の固定整相性能を周波数・方位で比較評価するモジュール。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float, require_positive_int
from .diagnostic_plotting import plot_bl_comparison, require_matplotlib
from .directions import make_directions
from .time_delay import (
    FractionalDelayAndSumBeamformer,
    FractionalDelayFilterBank,
    IntegerDelayAndSumBeamformer,
)


FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class FractionalDelayPerformanceConfig:
    """小数遅延導入前後の固定整相性能比較条件を保持する。

    このクラスは、評価対象アレイ、保存済み小数遅延 FIR バンク、周波数グリッド、
    評価方位群、走査ビーム本数をまとめて保持する。

    入力はセンサ座標 `[n_ch, 3]`、保存済み `.npz` パス、
    周波数列 `[n_frequency]`、評価方位列 `[n_sector]`、サンプリング周波数、音速であり、
    出力は `run_fractional_delay_performance_report()` が保存する JSON/CSV/PNG 群である。

    実波形の生成や SLC 重み更新は責務に含めない。
    信号処理上は、整数遅延固定整相と小数遅延固定整相の差を
    同一幾何で比較する事前検証条件に位置づく。
    """

    output_dir: Path
    array_positions_m: np.ndarray
    fractional_delay_filter_bank_path: Path
    fs_hz: float = 32768.0
    sound_speed_m_s: float = 1500.0
    frequency_grid_hz: tuple[float, ...] = (512.0, 1024.0, 2048.0, 3072.0, 4096.0, 6144.0, 8192.0, 10000.0)
    evaluation_azimuths_deg: tuple[float, ...] = (60.0, 90.0, 120.0)
    n_beam_az_real: int = 151
    comparison_specs: tuple[tuple[float, float], ...] = ((10000.0, 60.0), (10000.0, 90.0))

    def __post_init__(self) -> None:
        """入力 shape・単位・保存済みフィルタパスを検証する。"""
        positions = np.asarray(self.array_positions_m, dtype=np.float64)
        require(positions.ndim == 2 and positions.shape[1] == 3, "array_positions_m must have shape (n_ch, 3).")
        require(positions.shape[0] > 0, "array_positions_m must not be empty.")
        require(bool(np.all(np.isfinite(positions))), "array_positions_m must contain only finite values.")
        require_positive_float("fs_hz", float(self.fs_hz))
        require_positive_float("sound_speed_m_s", float(self.sound_speed_m_s))
        require_positive_int("n_beam_az_real", int(self.n_beam_az_real))
        require(Path(self.fractional_delay_filter_bank_path).exists(), "fractional_delay_filter_bank_path must exist.")

        frequency_grid_hz = np.asarray(self.frequency_grid_hz, dtype=np.float64)
        require(frequency_grid_hz.ndim == 1 and frequency_grid_hz.size > 0, "frequency_grid_hz must be a non-empty 1-D sequence.")
        require(bool(np.all(np.isfinite(frequency_grid_hz))), "frequency_grid_hz must contain only finite values.")
        require(bool(np.all(frequency_grid_hz > 0.0)), "frequency_grid_hz must contain only positive values.")
        require(bool(np.all(np.diff(frequency_grid_hz) > 0.0)), "frequency_grid_hz must be strictly increasing.")

        evaluation_azimuths_deg = np.asarray(self.evaluation_azimuths_deg, dtype=np.float64)
        require(
            evaluation_azimuths_deg.ndim == 1 and evaluation_azimuths_deg.size > 0,
            "evaluation_azimuths_deg must be a non-empty 1-D sequence.",
        )
        require(
            bool(np.all((0.0 <= evaluation_azimuths_deg) & (evaluation_azimuths_deg <= 180.0))),
            "evaluation_azimuths_deg must lie in [0, 180].",
        )

        for frequency_hz, azimuth_deg in self.comparison_specs:
            require_positive_float("comparison frequency", float(frequency_hz))
            require(0.0 <= float(azimuth_deg) <= 180.0, "comparison azimuth must lie in [0, 180].")


def _direction_from_azimuth_deg(azimuth_deg: float) -> FloatArray:
    """x-y 平面内の方位角から方向余弦ベクトルを返す。"""
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    return np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0], dtype=np.float64)


def _beam_response_db20(
    beamformer: IntegerDelayAndSumBeamformer | FractionalDelayAndSumBeamformer,
    positions_m: FloatArray,
    axis_azimuth_deg: FloatArray,
    frequency_hz: float,
    sound_speed_m_s: float,
    target_azimuth_deg: float,
    channel_weights: FloatArray | None = None,
) -> FloatArray:
    """指定 beamformer の BL を解析式で返す。

    Args:
        beamformer: 固定整相器。steering 応答が `steering_response()` で得られるもの。
        positions_m: センサ座標。shape は `[n_ch, 3]`。
        axis_azimuth_deg: 走査方位軸。shape は `[n_beam]`、単位は deg。
        frequency_hz: 評価周波数。単位は Hz。
        sound_speed_m_s: 音速。単位は m/s。
        target_azimuth_deg: target 方位。単位は deg。
        channel_weights: channel shading 係数。shape は `[n_ch]`。
            None の場合は全 channel を同じ重みで使う。

        Returns:
            BL。shape は `[n_beam]`、単位は dB20。
    """
    steering_response = np.asarray(beamformer.steering_response(float(frequency_hz)), dtype=np.complex128)
    sensor_positions_m = np.asarray(positions_m, dtype=np.float64)
    if channel_weights is None:
        weights = np.ones(sensor_positions_m.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(channel_weights, dtype=np.float64)
    require(sensor_positions_m.ndim == 2 and sensor_positions_m.shape[1] == 3, "positions_m must have shape (n_ch, 3).")
    require(weights.ndim == 1 and weights.shape[0] == sensor_positions_m.shape[0], "channel_weights must have shape (n_ch,).")
    require(bool(np.all(np.isfinite(weights))), "channel_weights must contain finite values.")
    weight_sum = float(np.sum(weights))
    require(weight_sum > 0.0, "channel_weights must contain positive total weight.")

    target_direction = _direction_from_azimuth_deg(float(target_azimuth_deg))

    # target_arrival_delay_sec[ch] = -(r_ch^T u_target) / c。
    # source 到来位相と固定整相器の steering 応答を channel shading 付きで加重平均すると、
    # target 1 本に対する observation beam ごとの複素 array response が得られる。
    # steering_response shape: [n_ch, n_beam]、source_arrival_phase shape: [n_ch]。
    # weights[:, None] の broadcasting により ch 軸だけを重み付けし、Σw で 0 dB ピーク基準を保つ。
    target_arrival_delay_sec = -(sensor_positions_m @ target_direction) / float(sound_speed_m_s)
    source_arrival_phase = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * target_arrival_delay_sec)
    beam_response = np.sum(weights[:, np.newaxis] * steering_response * source_arrival_phase[:, np.newaxis], axis=0) / weight_sum
    return 20.0 * np.log10(np.maximum(np.abs(beam_response), np.finfo(np.float64).tiny))


def _measure_local_peak_margin_db(
    axis_azimuth_deg: np.ndarray,
    beam_levels_db20: np.ndarray,
    target_azimuth_deg: float,
) -> tuple[float, float]:
    """target 近傍 peak と mainlobe 外 peak の差、および peak 方位を返す。"""
    nearest_index = int(np.argmin(np.abs(axis_azimuth_deg - float(target_azimuth_deg))))
    local_start = max(0, nearest_index - 4)
    local_stop = min(axis_azimuth_deg.size, nearest_index + 5)
    peak_index = int(local_start + np.argmax(beam_levels_db20[local_start:local_stop]))
    left_index = peak_index
    right_index = peak_index

    # peak から左右へ単調減少している区間を local mainlobe とみなし、
    # その外側最大値との差で mainlobe-sidelobe 分離量を定義する。
    while left_index > 0 and float(beam_levels_db20[left_index - 1]) <= float(beam_levels_db20[left_index]):
        left_index -= 1
    while right_index < beam_levels_db20.size - 1 and float(beam_levels_db20[right_index + 1]) <= float(beam_levels_db20[right_index]):
        right_index += 1

    outside_mask = np.ones(axis_azimuth_deg.size, dtype=bool)
    outside_mask[left_index : right_index + 1] = False
    outside_peak_level_db20 = float(np.max(beam_levels_db20[outside_mask])) if np.any(outside_mask) else float(-np.inf)
    return float(beam_levels_db20[peak_index] - outside_peak_level_db20), float(axis_azimuth_deg[peak_index])


def _plot_margin_summary(
    output_path: Path,
    frequencies_hz: np.ndarray,
    integer_worst_margin_db: np.ndarray,
    fractional_worst_margin_db: np.ndarray,
) -> None:
    """周波数に対する最悪方位 peak margin を比較保存する。"""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9.5, 4.5))
    axis.plot(frequencies_hz, integer_worst_margin_db, marker="o", linewidth=1.5, color="tab:red", label="Integer delay")
    axis.plot(frequencies_hz, fractional_worst_margin_db, marker="s", linewidth=1.5, color="tab:blue", label="Fractional delay")
    axis.axhline(13.0, color="black", linestyle=":", linewidth=1.0, label="Required margin 13 dB")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Worst Sector Peak Margin [dB]")
    axis.set_title("Integer vs fractional fixed beam margin")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_fractional_delay_performance_report(config: FractionalDelayPerformanceConfig) -> dict[str, object]:
    """整数遅延と小数遅延の固定整相性能比較レポートを保存する。

    Args:
        config: アレイ、保存済み FIR バンク、周波数グリッド、評価方位群を含む比較条件。

    Returns:
        周波数ごとの peak margin、保存先パス、比較 BL 図パスを含む summary 辞書。
    """
    require_matplotlib()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    directions, axis_azimuth_deg, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=int(config.n_beam_az_real),
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    positions_m = np.asarray(config.array_positions_m, dtype=np.float64)
    filter_bank = FractionalDelayFilterBank.load_npz(config.fractional_delay_filter_bank_path)
    integer_beamformer = IntegerDelayAndSumBeamformer.from_geometry(
        array_pos_m=positions_m,
        dir_cos=np.asarray(directions.T, dtype=np.float64),
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
    )
    fractional_beamformer = FractionalDelayAndSumBeamformer.from_geometry(
        array_pos_m=positions_m,
        dir_cos=np.asarray(directions.T, dtype=np.float64),
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
        fractional_filter_bank=filter_bank,
    )

    records: list[dict[str, object]] = []
    integer_worst_margin_db = []
    fractional_worst_margin_db = []

    for frequency_hz in np.asarray(config.frequency_grid_hz, dtype=np.float64):
        integer_margins_db = []
        fractional_margins_db = []
        for azimuth_deg in np.asarray(config.evaluation_azimuths_deg, dtype=np.float64):
            integer_levels_db20 = _beam_response_db20(
                beamformer=integer_beamformer,
                positions_m=positions_m,
                axis_azimuth_deg=axis_azimuth_deg,
                frequency_hz=float(frequency_hz),
                sound_speed_m_s=float(config.sound_speed_m_s),
                target_azimuth_deg=float(azimuth_deg),
            )
            fractional_levels_db20 = _beam_response_db20(
                beamformer=fractional_beamformer,
                positions_m=positions_m,
                axis_azimuth_deg=axis_azimuth_deg,
                frequency_hz=float(frequency_hz),
                sound_speed_m_s=float(config.sound_speed_m_s),
                target_azimuth_deg=float(azimuth_deg),
            )
            integer_margin_db, integer_peak_azimuth_deg = _measure_local_peak_margin_db(
                axis_azimuth_deg=axis_azimuth_deg,
                beam_levels_db20=integer_levels_db20,
                target_azimuth_deg=float(azimuth_deg),
            )
            fractional_margin_db, fractional_peak_azimuth_deg = _measure_local_peak_margin_db(
                axis_azimuth_deg=axis_azimuth_deg,
                beam_levels_db20=fractional_levels_db20,
                target_azimuth_deg=float(azimuth_deg),
            )
            integer_margins_db.append(float(integer_margin_db))
            fractional_margins_db.append(float(fractional_margin_db))

            records.append(
                {
                    "frequency_hz": float(frequency_hz),
                    "target_azimuth_deg": float(azimuth_deg),
                    "integer_peak_margin_db": float(integer_margin_db),
                    "fractional_peak_margin_db": float(fractional_margin_db),
                    "peak_margin_improvement_db": float(fractional_margin_db - integer_margin_db),
                    "integer_peak_azimuth_deg": float(integer_peak_azimuth_deg),
                    "fractional_peak_azimuth_deg": float(fractional_peak_azimuth_deg),
                }
            )

        integer_worst_margin_db.append(float(np.min(integer_margins_db)))
        fractional_worst_margin_db.append(float(np.min(fractional_margins_db)))

    comparison_png_paths: list[str] = []
    for frequency_hz, azimuth_deg in config.comparison_specs:
        integer_levels_db20 = _beam_response_db20(
            beamformer=integer_beamformer,
            positions_m=positions_m,
            axis_azimuth_deg=axis_azimuth_deg,
            frequency_hz=float(frequency_hz),
            sound_speed_m_s=float(config.sound_speed_m_s),
            target_azimuth_deg=float(azimuth_deg),
        )
        fractional_levels_db20 = _beam_response_db20(
            beamformer=fractional_beamformer,
            positions_m=positions_m,
            axis_azimuth_deg=axis_azimuth_deg,
            frequency_hz=float(frequency_hz),
            sound_speed_m_s=float(config.sound_speed_m_s),
            target_azimuth_deg=float(azimuth_deg),
        )
        _, integer_peak_azimuth_deg = _measure_local_peak_margin_db(
            axis_azimuth_deg=axis_azimuth_deg,
            beam_levels_db20=integer_levels_db20,
            target_azimuth_deg=float(azimuth_deg),
        )
        _, fractional_peak_azimuth_deg = _measure_local_peak_margin_db(
            axis_azimuth_deg=axis_azimuth_deg,
            beam_levels_db20=fractional_levels_db20,
            target_azimuth_deg=float(azimuth_deg),
        )

        output_path = output_dir / f"bl_compare_{int(round(float(frequency_hz))):05d}Hz_{int(round(float(azimuth_deg))):03d}deg.png"
        plot_bl_comparison(
            axis_az_deg=axis_azimuth_deg,
            before_levels_db20=integer_levels_db20,
            after_levels_db20=fractional_levels_db20,
            target_azimuth_deg=float(azimuth_deg),
            before_peak_azimuth_deg=float(integer_peak_azimuth_deg),
            after_peak_azimuth_deg=float(fractional_peak_azimuth_deg),
            title=f"BL comparison integer/fractional ({float(frequency_hz):.0f} Hz, {float(azimuth_deg):.0f} deg)",
            caption="blue after = fractional delay, orange before = integer delay. 高域 off-broadside の位相量子化誤差改善を確認する。",
            output_path=output_path,
            before_label="Integer delay",
            after_label="Fractional delay",
        )
        comparison_png_paths.append(str(output_path.resolve()))

    margin_png_path = output_dir / "margin_summary.png"
    _plot_margin_summary(
        output_path=margin_png_path,
        frequencies_hz=np.asarray(config.frequency_grid_hz, dtype=np.float64),
        integer_worst_margin_db=np.asarray(integer_worst_margin_db, dtype=np.float64),
        fractional_worst_margin_db=np.asarray(fractional_worst_margin_db, dtype=np.float64),
    )

    csv_path = output_dir / "performance_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    summary: dict[str, object] = {
        "fs_hz": float(config.fs_hz),
        "sound_speed_m_s": float(config.sound_speed_m_s),
        "array_n_ch": int(positions_m.shape[0]),
        "array_aperture_m": float(np.max(positions_m[:, 0]) - np.min(positions_m[:, 0])),
        "fractional_delay_filter_bank_path": str(Path(config.fractional_delay_filter_bank_path).resolve()),
        "n_frac_filter": int(filter_bank.n_frac_filter),
        "n_tap": int(filter_bank.n_tap),
        "frequency_grid_hz": [float(frequency_hz) for frequency_hz in config.frequency_grid_hz],
        "evaluation_azimuths_deg": [float(azimuth_deg) for azimuth_deg in config.evaluation_azimuths_deg],
        "integer_worst_margin_db": [float(value) for value in integer_worst_margin_db],
        "fractional_worst_margin_db": [float(value) for value in fractional_worst_margin_db],
        "fractional_meets_required_margin_all": bool(np.all(np.asarray(fractional_worst_margin_db, dtype=np.float64) >= 13.0)),
        "margin_summary_png_path": str(margin_png_path.resolve()),
        "comparison_png_paths": comparison_png_paths,
        "performance_table_csv_path": str(csv_path.resolve()),
        "records": records,
    }
    json_path = output_dir / "performance_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["performance_summary_json_path"] = str(json_path.resolve())
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = [
    "FractionalDelayPerformanceConfig",
    "run_fractional_delay_performance_report",
]
