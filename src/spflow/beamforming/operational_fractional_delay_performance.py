"""運用スパースアレイ定義を使った小数遅延固定整相の性能評価モジュール。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._validation import require, require_positive_float, require_positive_int
from .diagnostic_plotting import plot_bl_comparison, require_matplotlib
from .directions import make_directions
from .fractional_delay_performance import (
    _beam_response_db20,
    _measure_local_peak_margin_db,
    _plot_margin_summary,
)
from .operational_sparse_array import OperationalSparseArrayDefinition
from .time_delay import FractionalDelayAndSumBeamformer, FractionalDelayFilterBank, IntegerDelayAndSumBeamformer


@dataclass(frozen=True)
class OperationalArrayFractionalDelayPerformanceConfig:
    """運用アレイ定義ファイルを使う小数遅延固定整相の評価条件を保持する。

    このクラスは、運用スパースアレイ JSON、保存済み小数遅延 FIR バンク、評価周波数、
    評価方位、出力先をまとめて保持する。

    入力はアレイ定義 JSON と FIR バンク `.npz` であり、出力は
    `run_operational_array_fractional_delay_performance_report()` が保存する
    JSON / CSV / PNG 群である。アレイ CH 数は JSON 内の `positions_m.shape[0]` から読み取る。

    アレイ定義ファイルの作成、小数遅延 FIR バンクの設計、SLC 重み更新は責務に含めない。
    信号処理上は、固定整相 + SLC の前段として使う小数遅延固定整相が、
    周波数ごとの active channel 設計上で所定の BL 性能を満たすかを確認する評価条件に位置づく。
    """

    output_dir: Path
    operational_array_definition_path: Path
    fractional_delay_filter_bank_path: Path
    fs_hz: float = 32768.0
    sound_speed_m_s: float = 1500.0
    frequency_grid_hz: tuple[float, ...] = (
        256.0,
        384.0,
        512.0,
        768.0,
        1024.0,
        1500.0,
        2048.0,
        3072.0,
        4096.0,
        6144.0,
        8192.0,
        10000.0,
    )
    evaluation_azimuths_deg: tuple[float, ...] = (60.0, 90.0, 120.0)
    n_beam_az_real: int = 151
    required_peak_margin_db: float = 13.0
    comparison_specs: tuple[tuple[float, float], ...] = ((10000.0, 60.0), (10000.0, 90.0), (256.0, 90.0))

    def __post_init__(self) -> None:
        """入力ファイル、周波数軸、方位軸の妥当性を検証する。"""
        require(Path(self.operational_array_definition_path).exists(), "operational_array_definition_path must exist.")
        require(Path(self.fractional_delay_filter_bank_path).exists(), "fractional_delay_filter_bank_path must exist.")
        require_positive_float("fs_hz", float(self.fs_hz))
        require_positive_float("sound_speed_m_s", float(self.sound_speed_m_s))
        require_positive_int("n_beam_az_real", int(self.n_beam_az_real))
        require_positive_float("required_peak_margin_db", float(self.required_peak_margin_db))

        frequencies_hz = np.asarray(self.frequency_grid_hz, dtype=np.float64)
        require(frequencies_hz.ndim == 1 and frequencies_hz.size > 0, "frequency_grid_hz must be a non-empty 1-D sequence.")
        require(bool(np.all(np.isfinite(frequencies_hz))), "frequency_grid_hz must contain finite values.")
        require(bool(np.all(frequencies_hz > 0.0)), "frequency_grid_hz must contain only positive values.")
        require(bool(np.all(np.diff(frequencies_hz) > 0.0)), "frequency_grid_hz must be strictly increasing.")

        azimuths_deg = np.asarray(self.evaluation_azimuths_deg, dtype=np.float64)
        require(azimuths_deg.ndim == 1 and azimuths_deg.size > 0, "evaluation_azimuths_deg must be a non-empty 1-D sequence.")
        require(bool(np.all((0.0 <= azimuths_deg) & (azimuths_deg <= 180.0))), "evaluation_azimuths_deg must lie in [0, 180].")


def _active_geometry_for_frequency(
    array_definition: OperationalSparseArrayDefinition,
    frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """周波数に対応する active index と active 配置の要約を返す。"""
    active_indices = array_definition.active_channel_indices_for_frequency(float(frequency_hz))
    active_positions_m = np.asarray(array_definition.positions_m, dtype=np.float64)[active_indices]
    active_x_m = np.sort(active_positions_m[:, 0])
    if active_x_m.size <= 1:
        active_aperture_m = 0.0
        active_min_spacing_m = float("inf")
        active_max_spacing_m = float("inf")
    else:
        spacings_m = np.diff(active_x_m)
        active_aperture_m = float(active_x_m[-1] - active_x_m[0])
        active_min_spacing_m = float(np.min(spacings_m))
        active_max_spacing_m = float(np.max(spacings_m))
    return active_indices, active_positions_m, active_aperture_m, active_min_spacing_m, active_max_spacing_m


def _make_beamformers_for_active_geometry(
    active_positions_m: np.ndarray,
    directions: np.ndarray,
    filter_bank: FractionalDelayFilterBank,
    config: OperationalArrayFractionalDelayPerformanceConfig,
) -> tuple[IntegerDelayAndSumBeamformer, FractionalDelayAndSumBeamformer]:
    """active 配置に対する整数遅延・小数遅延固定整相器を作る。"""
    # active_positions_m shape: [n_active_ch, 3]。
    # 周波数ごとに active channel が変わるため、DelayTable も active 配置ごとに作り直す。
    integer_beamformer = IntegerDelayAndSumBeamformer.from_geometry(
        array_pos_m=active_positions_m,
        dir_cos=np.asarray(directions.T, dtype=np.float64),
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
    )
    fractional_beamformer = FractionalDelayAndSumBeamformer.from_geometry(
        array_pos_m=active_positions_m,
        dir_cos=np.asarray(directions.T, dtype=np.float64),
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
        fractional_filter_bank=filter_bank,
    )
    return integer_beamformer, fractional_beamformer


def run_operational_array_fractional_delay_performance_report(
    config: OperationalArrayFractionalDelayPerformanceConfig,
) -> dict[str, object]:
    """運用スパースアレイで小数遅延固定整相の性能評価レポートを保存する。

    Args:
        config: アレイ定義 JSON、FIR バンク、評価周波数、評価方位、保存先を含む条件。

    Returns:
        周波数ごとの active channel、整数遅延 / 小数遅延 margin、保存ファイルパスを含む summary。
    """
    require_matplotlib()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    array_definition = OperationalSparseArrayDefinition.load_json(Path(config.operational_array_definition_path))
    filter_bank = FractionalDelayFilterBank.load_npz(config.fractional_delay_filter_bank_path)
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

    records: list[dict[str, object]] = []
    integer_worst_margin_db: list[float] = []
    fractional_worst_margin_db: list[float] = []
    active_channel_count: list[int] = []
    active_aperture_m: list[float] = []

    for frequency_hz in np.asarray(config.frequency_grid_hz, dtype=np.float64).tolist():
        active_indices, active_positions_m, aperture_m, min_spacing_m, max_spacing_m = _active_geometry_for_frequency(
            array_definition=array_definition,
            frequency_hz=float(frequency_hz),
        )
        integer_beamformer, fractional_beamformer = _make_beamformers_for_active_geometry(
            active_positions_m=active_positions_m,
            directions=np.asarray(directions, dtype=np.float64),
            filter_bank=filter_bank,
            config=config,
        )

        integer_margins_db: list[float] = []
        fractional_margins_db: list[float] = []
        for azimuth_deg in np.asarray(config.evaluation_azimuths_deg, dtype=np.float64).tolist():
            integer_levels_db20 = _beam_response_db20(
                beamformer=integer_beamformer,
                positions_m=active_positions_m,
                axis_azimuth_deg=axis_azimuth_deg,
                frequency_hz=float(frequency_hz),
                sound_speed_m_s=float(config.sound_speed_m_s),
                target_azimuth_deg=float(azimuth_deg),
            )
            fractional_levels_db20 = _beam_response_db20(
                beamformer=fractional_beamformer,
                positions_m=active_positions_m,
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
                    "active_channel_count": int(active_indices.size),
                    "active_aperture_m": float(aperture_m),
                    "active_min_spacing_m": float(min_spacing_m),
                    "active_max_spacing_m": float(max_spacing_m),
                    "integer_peak_margin_db": float(integer_margin_db),
                    "fractional_peak_margin_db": float(fractional_margin_db),
                    "peak_margin_improvement_db": float(fractional_margin_db - integer_margin_db),
                    "integer_peak_azimuth_deg": float(integer_peak_azimuth_deg),
                    "fractional_peak_azimuth_deg": float(fractional_peak_azimuth_deg),
                }
            )

        integer_worst_margin_db.append(float(np.min(integer_margins_db)))
        fractional_worst_margin_db.append(float(np.min(fractional_margins_db)))
        active_channel_count.append(int(active_indices.size))
        active_aperture_m.append(float(aperture_m))

    comparison_png_paths: list[str] = []
    for frequency_hz, azimuth_deg in config.comparison_specs:
        active_indices, active_positions_m, _, _, _ = _active_geometry_for_frequency(
            array_definition=array_definition,
            frequency_hz=float(frequency_hz),
        )
        integer_beamformer, fractional_beamformer = _make_beamformers_for_active_geometry(
            active_positions_m=active_positions_m,
            directions=np.asarray(directions, dtype=np.float64),
            filter_bank=filter_bank,
            config=config,
        )
        integer_levels_db20 = _beam_response_db20(
            beamformer=integer_beamformer,
            positions_m=active_positions_m,
            axis_azimuth_deg=axis_azimuth_deg,
            frequency_hz=float(frequency_hz),
            sound_speed_m_s=float(config.sound_speed_m_s),
            target_azimuth_deg=float(azimuth_deg),
        )
        fractional_levels_db20 = _beam_response_db20(
            beamformer=fractional_beamformer,
            positions_m=active_positions_m,
            axis_azimuth_deg=axis_azimuth_deg,
            frequency_hz=float(frequency_hz),
            sound_speed_m_s=float(config.sound_speed_m_s),
            target_azimuth_deg=float(azimuth_deg),
        )
        _, integer_peak_azimuth_deg = _measure_local_peak_margin_db(axis_azimuth_deg, integer_levels_db20, float(azimuth_deg))
        _, fractional_peak_azimuth_deg = _measure_local_peak_margin_db(axis_azimuth_deg, fractional_levels_db20, float(azimuth_deg))
        output_path = output_dir / f"bl_compare_operational_{int(round(float(frequency_hz))):05d}Hz_{int(round(float(azimuth_deg))):03d}deg.png"
        plot_bl_comparison(
            axis_az_deg=axis_azimuth_deg,
            before_levels_db20=integer_levels_db20,
            after_levels_db20=fractional_levels_db20,
            target_azimuth_deg=float(azimuth_deg),
            before_peak_azimuth_deg=float(integer_peak_azimuth_deg),
            after_peak_azimuth_deg=float(fractional_peak_azimuth_deg),
            title=f"Operational array BL integer/fractional ({float(frequency_hz):.0f} Hz, {float(azimuth_deg):.0f} deg)",
            caption=(
                f"active_ch={int(active_indices.size)}. "
                "blue after=fractional delay, orange before=integer delay."
            ),
            output_path=output_path,
            before_label="Integer delay",
            after_label="Fractional delay",
        )
        comparison_png_paths.append(str(output_path.resolve()))

    margin_png_path = output_dir / "operational_fractional_margin_summary.png"
    _plot_margin_summary(
        output_path=margin_png_path,
        frequencies_hz=np.asarray(config.frequency_grid_hz, dtype=np.float64),
        integer_worst_margin_db=np.asarray(integer_worst_margin_db, dtype=np.float64),
        fractional_worst_margin_db=np.asarray(fractional_worst_margin_db, dtype=np.float64),
    )

    csv_path = output_dir / "operational_fractional_performance_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    fractional_worst_array = np.asarray(fractional_worst_margin_db, dtype=np.float64)
    summary: dict[str, object] = {
        "fs_hz": float(config.fs_hz),
        "sound_speed_m_s": float(config.sound_speed_m_s),
        "operational_array_definition_path": str(Path(config.operational_array_definition_path).resolve()),
        "physical_array_n_ch": int(array_definition.n_ch),
        "physical_array_aperture_m": float(array_definition.aperture_m),
        "fractional_delay_filter_bank_path": str(Path(config.fractional_delay_filter_bank_path).resolve()),
        "n_frac_filter": int(filter_bank.n_frac_filter),
        "n_tap": int(filter_bank.n_tap),
        "frequency_grid_hz": [float(value) for value in config.frequency_grid_hz],
        "active_channel_count": [int(value) for value in active_channel_count],
        "active_aperture_m": [float(value) for value in active_aperture_m],
        "integer_worst_margin_db": [float(value) for value in integer_worst_margin_db],
        "fractional_worst_margin_db": [float(value) for value in fractional_worst_margin_db],
        "fractional_meets_required_margin_all": bool(np.all(fractional_worst_array >= float(config.required_peak_margin_db))),
        "required_peak_margin_db": float(config.required_peak_margin_db),
        "margin_summary_png_path": str(margin_png_path.resolve()),
        "comparison_png_paths": comparison_png_paths,
        "performance_table_csv_path": str(csv_path.resolve()),
        "records": records,
    }
    json_path = output_dir / "operational_fractional_performance_summary.json"
    summary["performance_summary_json_path"] = str(json_path.resolve())
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = [
    "OperationalArrayFractionalDelayPerformanceConfig",
    "run_operational_array_fractional_delay_performance_report",
]
