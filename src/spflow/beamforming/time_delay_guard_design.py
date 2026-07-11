"""整数遅延固定整相の BL から周波数依存 guard を設計するモジュール。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._validation import require, require_non_negative_float, require_positive_float, require_positive_int
from .diagnostic_plotting import plot_bl_response, require_matplotlib
from .time_delay import IntegerDelayAndSumBeamformer
from .time_delay_diagnostics import (
    TimeDelayDiagnosticConfig,
    TimeDelayDiagnosticSource,
    _build_array_positions,
    _build_beam_grid,
    _generate_target_scene,
    _tone_level_db20_rms,
)


@dataclass(frozen=True)
class TimeDelayGuardDesignConfig:
    """周波数依存 guard 設計の評価条件を保持する。

    このクラスは、SLC を掛けない固定整相だけの BL を低周波から高周波まで測定し、
    mainlobe 保護に必要な guard 幅を周波数ごとに決めるための条件を保持する。

    入力は周波数グリッド、片舷アレイ形状、走査ビーム数、target 方位、
    目標とする mainlobe-sidelobe 分離量などであり、出力は
    `run_integer_delay_guard_design()` が保存する JSON/CSV/PNG 群である。

    SLC 自体の更新、eta の最適化、複数同時 source のキャンセル評価は責務に含めない。
    信号処理上は、固定整相後段 SLC の guard 保護幅を事前設計するための測定条件に位置づく。
    """

    output_dir: Path
    fs_hz: float = 32768.0
    duration_s: float = 1.0
    sound_speed_m_s: float = 1500.0
    target_azimuth_deg: float = 20.0
    target_elevation_deg: float = 0.0
    target_level_db20: float = 0.0
    target_phase_deg: float = 0.0
    noise_level_db20: float = -120.0
    random_seed: int = 1234
    array_n_ch: int = 61
    array_sensor_spacing_m: float = 0.05
    sparse_stride_pattern: tuple[int, ...] | None = None
    array_positions_m: np.ndarray | None = None
    az_min_deg: float = 0.0
    az_max_deg: float = 180.0
    n_beam_az_real: int = 151
    n_beam_az_virtual: int = 0
    display_elevation_deg: float = 0.0
    frequency_grid_hz: tuple[float, ...] | None = None
    frequency_start_hz: float = 512.0
    frequency_stop_hz: float = 4096.0
    n_frequency: int = 15
    required_peak_margin_db: float = 13.0
    half_power_drop_db: float = 3.0
    peak_search_half_width_beam: int = 4
    guard_safety_margin_beams: int = 0

    def __post_init__(self) -> None:
        """設計条件の範囲と shape を検証する。"""
        require_positive_float("fs_hz", float(self.fs_hz))
        require_positive_float("duration_s", float(self.duration_s))
        require_positive_float("sound_speed_m_s", float(self.sound_speed_m_s))
        require(bool(np.isfinite(float(self.target_azimuth_deg))), "target_azimuth_deg must be finite.")
        require(bool(np.isfinite(float(self.target_elevation_deg))), "target_elevation_deg must be finite.")
        require(bool(np.isfinite(float(self.target_level_db20))), "target_level_db20 must be finite.")
        require(bool(np.isfinite(float(self.target_phase_deg))), "target_phase_deg must be finite.")
        require_positive_int("array_n_ch", int(self.array_n_ch))
        require_positive_float("array_sensor_spacing_m", float(self.array_sensor_spacing_m))
        require_positive_int("n_beam_az_real", int(self.n_beam_az_real))
        require(self.n_beam_az_virtual >= 0, "n_beam_az_virtual must be non-negative.")
        require_positive_float("frequency_start_hz", float(self.frequency_start_hz))
        require_positive_float("frequency_stop_hz", float(self.frequency_stop_hz))
        require(float(self.frequency_stop_hz) >= float(self.frequency_start_hz), "frequency_stop_hz must be >= frequency_start_hz.")
        require_positive_int("n_frequency", int(self.n_frequency))
        require_positive_float("required_peak_margin_db", float(self.required_peak_margin_db))
        require_positive_float("half_power_drop_db", float(self.half_power_drop_db))
        require_non_negative_float("peak_search_half_width_beam", float(self.peak_search_half_width_beam))
        require_non_negative_float("guard_safety_margin_beams", float(self.guard_safety_margin_beams))

        if self.frequency_grid_hz is not None:
            frequencies_hz = np.asarray(self.frequency_grid_hz, dtype=np.float64)
            require(frequencies_hz.ndim == 1 and frequencies_hz.size > 0, "frequency_grid_hz must be a non-empty 1-D sequence.")
            require(bool(np.all(np.isfinite(frequencies_hz))), "frequency_grid_hz must contain only finite values.")
            require(bool(np.all(frequencies_hz > 0.0)), "frequency_grid_hz must contain only positive values.")
            require(bool(np.all(np.diff(frequencies_hz) > 0.0)), "frequency_grid_hz must be strictly increasing.")

        if self.sparse_stride_pattern is not None:
            stride_pattern = np.asarray(self.sparse_stride_pattern, dtype=np.int64)
            require(stride_pattern.ndim == 1 and stride_pattern.size > 0, "sparse_stride_pattern must be a non-empty 1-D sequence.")
            require(bool(np.all(stride_pattern > 0)), "sparse_stride_pattern must contain only positive integers.")

        if self.array_positions_m is not None:
            positions = np.asarray(self.array_positions_m, dtype=np.float64)
            require(positions.ndim == 2 and positions.shape[1] == 3, "array_positions_m must have shape (n_ch, 3).")
            require(positions.shape[0] > 0, "array_positions_m must not be empty.")
            require(bool(np.all(np.isfinite(positions))), "array_positions_m must contain only finite values.")


def _resolve_frequency_grid_hz(config: TimeDelayGuardDesignConfig) -> np.ndarray:
    """評価に使う周波数グリッドを返す。"""
    if config.frequency_grid_hz is not None:
        return np.asarray(config.frequency_grid_hz, dtype=np.float64)
    return np.linspace(
        float(config.frequency_start_hz),
        float(config.frequency_stop_hz),
        int(config.n_frequency),
        dtype=np.float64,
    )


def _measure_half_power_region(
    beam_levels_db20: np.ndarray,
    peak_beam_index: int,
    half_power_drop_db: float,
) -> tuple[int, int]:
    """peak 近傍の half-power 領域を beam index で返す。"""
    threshold_db20 = float(beam_levels_db20[int(peak_beam_index)]) - float(half_power_drop_db)
    left_index = int(peak_beam_index)
    right_index = int(peak_beam_index)

    # peak から左右へたどり、peak - drop[dB] 以上を保つ連続領域を mainlobe の半値幅とみなす。
    while left_index > 0 and float(beam_levels_db20[left_index - 1]) >= threshold_db20:
        left_index -= 1
    while right_index < beam_levels_db20.size - 1 and float(beam_levels_db20[right_index + 1]) >= threshold_db20:
        right_index += 1
    return left_index, right_index


def _measure_mainlobe_region_from_local_minima(
    beam_levels_db20: np.ndarray,
    peak_beam_index: int,
) -> tuple[int, int]:
    """peak の左右で最初の局所谷までを mainlobe 領域として返す。"""
    left_index = int(peak_beam_index)
    right_index = int(peak_beam_index)

    # peak から離れるにつれてレベルが単調減少している間だけ進み、
    # はじめて増加へ転じる直前の谷を mainlobe 端とみなす。
    while left_index > 0 and float(beam_levels_db20[left_index - 1]) <= float(beam_levels_db20[left_index]):
        left_index -= 1
    while right_index < beam_levels_db20.size - 1 and float(beam_levels_db20[right_index + 1]) <= float(beam_levels_db20[right_index]):
        right_index += 1
    return left_index, right_index


def _design_guard_half_width_beam(
    beam_levels_db20: np.ndarray,
    peak_beam_index: int,
    initial_half_width_beam: int,
    required_peak_margin_db: float,
    guard_safety_margin_beams: int,
) -> tuple[int, float, float]:
    """mainlobe 外ピークが所望 dB だけ下がる最小 guard half-width を返す。"""
    n_beam = int(beam_levels_db20.size)
    peak_level_db20 = float(beam_levels_db20[int(peak_beam_index)])
    start_half_width_beam = int(initial_half_width_beam) + int(guard_safety_margin_beams)
    best_half_width_beam = n_beam - 1
    best_outside_peak_level_db20 = peak_level_db20
    best_margin_db = 0.0

    for guard_half_width_beam in range(start_half_width_beam, n_beam):
        protected_start = max(0, int(peak_beam_index) - int(guard_half_width_beam))
        protected_stop = min(n_beam, int(peak_beam_index) + int(guard_half_width_beam) + 1)
        outside_mask = np.ones(n_beam, dtype=bool)

        # protected 領域外だけで最大ピークを取り、mainlobe と sidelobe のピーク差を設計条件に使う。
        outside_mask[protected_start:protected_stop] = False
        outside_peak_level_db20 = float(np.max(beam_levels_db20[outside_mask])) if np.any(outside_mask) else float(-np.inf)
        margin_db = peak_level_db20 - outside_peak_level_db20 if np.isfinite(outside_peak_level_db20) else float(np.inf)

        best_half_width_beam = int(guard_half_width_beam)
        best_outside_peak_level_db20 = float(outside_peak_level_db20)
        best_margin_db = float(margin_db)
        if margin_db >= float(required_peak_margin_db):
            break

    return best_half_width_beam, best_outside_peak_level_db20, best_margin_db


def _write_guard_table_csv(path: Path, records: list[dict[str, object]]) -> None:
    """guard 設計結果を CSV へ保存する。"""
    fieldnames = [
        "frequency_hz",
        "nearest_beam_azimuth_deg",
        "peak_azimuth_deg",
        "peak_level_db20",
        "half_power_width_beams",
        "half_power_width_deg",
        "mainlobe_width_beams",
        "mainlobe_width_deg",
        "guard_half_width_beams",
        "guard_width_beams",
        "guard_width_deg",
        "outside_peak_level_db20",
        "mainlobe_to_outside_peak_db",
        "meets_required_peak_margin",
        "required_margin_guard_half_width_beams",
        "bl_png_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({fieldname: record[fieldname] for fieldname in fieldnames})


def _require_record_number(value: object, name: str) -> int | float:
    """集計recordから型検証済みの数値を返す。"""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    return value


def _plot_guard_design_summary(
    output_path: Path,
    frequencies_hz: np.ndarray,
    half_power_width_beams: np.ndarray,
    guard_half_width_beams: np.ndarray,
) -> None:
    """周波数に対する half-power 幅と guard を 1 枚へ保存する。"""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9.5, 4.5))
    axis.plot(frequencies_hz, half_power_width_beams, marker="o", linewidth=1.5, label="Half-power width [beam]")
    axis.plot(frequencies_hz, guard_half_width_beams, marker="s", linewidth=1.5, label="Guard half-width [beam]")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Beam Count")
    axis.set_title("Frequency-dependent guard design")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_integer_delay_guard_design(config: TimeDelayGuardDesignConfig) -> dict[str, Any]:
    """固定整相だけの BL から周波数依存 guard 設計表を保存する。

    Args:
        config: 周波数グリッド、アレイ形状、目標方位、必要ピーク差などを含む設計条件。

    Returns:
        周波数ごとの guard 設計結果、保存 JSON/CSV/PNG パスを含む summary 辞書。

    Raises:
        RuntimeError: matplotlib が利用できず BL/summary PNG を保存できない場合。
        ValueError: 設計条件が不正な場合。
    """
    require_matplotlib()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frequency_grid_hz = _resolve_frequency_grid_hz(config)
    array_positions_m, array_geometry_name, array_is_sparse = _build_array_positions(
        TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            fs_hz=float(config.fs_hz),
            duration_s=float(config.duration_s),
            sound_speed_m_s=float(config.sound_speed_m_s),
            noise_level_db20=float(config.noise_level_db20),
            random_seed=int(config.random_seed),
            array_n_ch=int(config.array_n_ch),
            array_sensor_spacing_m=float(config.array_sensor_spacing_m),
            sparse_stride_pattern=config.sparse_stride_pattern,
            array_positions_m=config.array_positions_m,
            az_min_deg=float(config.az_min_deg),
            az_max_deg=float(config.az_max_deg),
            n_beam_az_real=int(config.n_beam_az_real),
            n_beam_az_virtual=int(config.n_beam_az_virtual),
            display_elevation_deg=float(config.display_elevation_deg),
        )
    )
    beam_grid = _build_beam_grid(
        TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            az_min_deg=float(config.az_min_deg),
            az_max_deg=float(config.az_max_deg),
            n_beam_az_real=int(config.n_beam_az_real),
            n_beam_az_virtual=int(config.n_beam_az_virtual),
            display_elevation_deg=float(config.display_elevation_deg),
        )
    )
    axis_az_deg = np.asarray(beam_grid["axis_az_deg"], dtype=np.float64)
    beamformer = IntegerDelayAndSumBeamformer.from_geometry(
        array_pos_m=array_positions_m,
        dir_cos=np.asarray(beam_grid["directions"], dtype=np.float64),
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
    )

    records: list[dict[str, object]] = []

    for frequency_hz in frequency_grid_hz:
        measurement_config = TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            fs_hz=float(config.fs_hz),
            duration_s=float(config.duration_s),
            sound_speed_m_s=float(config.sound_speed_m_s),
            noise_level_db20=float(config.noise_level_db20),
            random_seed=int(config.random_seed),
            source_specs=(
                TimeDelayDiagnosticSource(
                    azimuth_deg=float(config.target_azimuth_deg),
                    elevation_deg=float(config.target_elevation_deg),
                    frequency_hz=float(frequency_hz),
                    level_db20=float(config.target_level_db20),
                    phase_deg=float(config.target_phase_deg),
                    label=f"{float(frequency_hz):.1f} Hz",
                ),
            ),
        )
        measurement_source_specs = measurement_config.source_specs
        if measurement_source_specs is None:
            # この関数では直前に必ず単一音源を設定する。None は構成契約違反として早期に検出する。
            raise RuntimeError("guard measurement requires one source specification.")
        multichannel_signal, _ = _generate_target_scene(array_positions_m, measurement_source_specs, measurement_config)
        beam_output = beamformer.process(multichannel_signal)
        if isinstance(beam_output, tuple):
            # guard 設計は主ビーム出力だけを評価し、デバッグ用 steered channel は使用しない。
            beam_output = beam_output[0]

        beam_levels_db20 = np.array(
            [
                _tone_level_db20_rms(
                    beam_output[beam_index],
                    frequency_hz=float(frequency_hz),
                    fs_hz=float(config.fs_hz),
                )
                for beam_index in range(beam_output.shape[0])
            ],
            dtype=np.float64,
        )
        nearest_beam_index = int(np.argmin(np.abs(axis_az_deg - float(config.target_azimuth_deg))))
        local_start = max(0, nearest_beam_index - int(config.peak_search_half_width_beam))
        local_stop = min(axis_az_deg.size, nearest_beam_index + int(config.peak_search_half_width_beam) + 1)
        peak_beam_index = int(local_start + np.argmax(beam_levels_db20[local_start:local_stop]))

        half_power_left_beam_index, half_power_right_beam_index = _measure_half_power_region(
            beam_levels_db20=beam_levels_db20,
            peak_beam_index=int(peak_beam_index),
            half_power_drop_db=float(config.half_power_drop_db),
        )
        mainlobe_left_beam_index, mainlobe_right_beam_index = _measure_mainlobe_region_from_local_minima(
            beam_levels_db20=beam_levels_db20,
            peak_beam_index=int(peak_beam_index),
        )
        mainlobe_half_width_beam = max(
            int(peak_beam_index) - int(mainlobe_left_beam_index),
            int(mainlobe_right_beam_index) - int(peak_beam_index),
        )
        guard_half_width_beam = int(mainlobe_half_width_beam) + int(config.guard_safety_margin_beams)
        guard_left_beam_index = max(0, int(peak_beam_index) - int(guard_half_width_beam))
        guard_right_beam_index = min(axis_az_deg.size - 1, int(peak_beam_index) + int(guard_half_width_beam))
        outside_mask = np.ones(axis_az_deg.size, dtype=bool)

        # ここでの outside peak は、mainlobe 幅から決めた guard の外側に残る最大ピークである。
        # この値が 13 dB 条件を満たすかどうかを、eta 設計の前提チェックとして保存する。
        outside_mask[int(guard_left_beam_index) : int(guard_right_beam_index) + 1] = False
        outside_peak_level_db20 = float(np.max(beam_levels_db20[outside_mask])) if np.any(outside_mask) else float(-np.inf)
        achieved_margin_db = float(beam_levels_db20[int(peak_beam_index)] - outside_peak_level_db20) if np.isfinite(outside_peak_level_db20) else float(np.inf)
        required_margin_guard_half_width_beam, _, _ = _design_guard_half_width_beam(
            beam_levels_db20=beam_levels_db20,
            peak_beam_index=int(peak_beam_index),
            initial_half_width_beam=int(guard_half_width_beam),
            required_peak_margin_db=float(config.required_peak_margin_db),
            guard_safety_margin_beams=0,
        )
        meets_required_peak_margin = bool(achieved_margin_db >= float(config.required_peak_margin_db))

        half_power_width_deg = float(axis_az_deg[int(half_power_right_beam_index)] - axis_az_deg[int(half_power_left_beam_index)])
        mainlobe_width_deg = float(axis_az_deg[int(mainlobe_right_beam_index)] - axis_az_deg[int(mainlobe_left_beam_index)])
        guard_width_deg = float(axis_az_deg[int(guard_right_beam_index)] - axis_az_deg[int(guard_left_beam_index)])
        bl_output_path = output_dir / f"guard_bl_{int(round(float(frequency_hz))):05d}Hz.png"

        plot_bl_response(
            axis_az_deg=axis_az_deg,
            beam_levels_db20=beam_levels_db20,
            target_azimuth_deg=float(config.target_azimuth_deg),
            peak_azimuth_deg=float(axis_az_deg[int(peak_beam_index)]),
            title=f"BL for guard design ({float(frequency_hz):.1f} Hz)",
            caption=(
                f"mainlobe width={int(mainlobe_right_beam_index - mainlobe_left_beam_index + 1)} beams, "
                f"guard half-width={int(guard_half_width_beam)} beams, margin={float(achieved_margin_db):.2f} dB, "
                f"required-13dB guard={int(required_margin_guard_half_width_beam)} beams"
            ),
            output_path=bl_output_path,
            response_label="Fixed beam response",
        )

        records.append(
            {
                "frequency_hz": float(frequency_hz),
                "nearest_beam_azimuth_deg": float(axis_az_deg[int(nearest_beam_index)]),
                "peak_azimuth_deg": float(axis_az_deg[int(peak_beam_index)]),
                "peak_level_db20": float(beam_levels_db20[int(peak_beam_index)]),
                "half_power_left_beam_index": int(half_power_left_beam_index),
                "half_power_right_beam_index": int(half_power_right_beam_index),
                "half_power_width_beams": int(half_power_right_beam_index - half_power_left_beam_index + 1),
                "half_power_width_deg": float(half_power_width_deg),
                "mainlobe_left_beam_index": int(mainlobe_left_beam_index),
                "mainlobe_right_beam_index": int(mainlobe_right_beam_index),
                "mainlobe_width_beams": int(mainlobe_right_beam_index - mainlobe_left_beam_index + 1),
                "mainlobe_width_deg": float(mainlobe_width_deg),
                "guard_left_beam_index": int(guard_left_beam_index),
                "guard_right_beam_index": int(guard_right_beam_index),
                "guard_half_width_beams": int(guard_half_width_beam),
                "guard_width_beams": int(2 * int(guard_half_width_beam) + 1),
                "guard_width_deg": float(guard_width_deg),
                "outside_peak_level_db20": float(outside_peak_level_db20),
                "mainlobe_to_outside_peak_db": float(achieved_margin_db),
                "meets_required_peak_margin": bool(meets_required_peak_margin),
                "required_margin_guard_half_width_beams": int(required_margin_guard_half_width_beam),
                "bl_png_path": str(bl_output_path.resolve()),
            }
        )

    json_path = output_dir / "frequency_guard_table.json"
    csv_path = output_dir / "frequency_guard_table.csv"
    plot_path = output_dir / "frequency_guard_table.png"
    half_power_width_beams = np.array(
        [int(_require_record_number(record["half_power_width_beams"], "half_power_width_beams")) for record in records],
        dtype=np.int32,
    )
    guard_half_width_beams = np.array(
        [int(_require_record_number(record["guard_half_width_beams"], "guard_half_width_beams")) for record in records],
        dtype=np.int32,
    )

    _write_guard_table_csv(csv_path, records)
    _plot_guard_design_summary(
        output_path=plot_path,
        frequencies_hz=frequency_grid_hz,
        half_power_width_beams=half_power_width_beams.astype(np.float64),
        guard_half_width_beams=guard_half_width_beams.astype(np.float64),
    )

    sensor_positions_x = np.sort(array_positions_m[:, 0])
    sensor_spacings_m = np.diff(sensor_positions_x)
    summary: dict[str, object] = {
        "fs_hz": float(config.fs_hz),
        "duration_s": float(config.duration_s),
        "sound_speed_m_s": float(config.sound_speed_m_s),
        "target_azimuth_deg": float(config.target_azimuth_deg),
        "target_elevation_deg": float(config.target_elevation_deg),
        "target_level_db20": float(config.target_level_db20),
        "noise_level_db20": float(config.noise_level_db20),
        "required_peak_margin_db": float(config.required_peak_margin_db),
        "half_power_drop_db": float(config.half_power_drop_db),
        "peak_search_half_width_beam": int(config.peak_search_half_width_beam),
        "guard_safety_margin_beams": int(config.guard_safety_margin_beams),
        "array_geometry_name": str(array_geometry_name),
        "array_is_sparse": bool(array_is_sparse),
        "array_n_ch": int(array_positions_m.shape[0]),
        "array_aperture_m": float(sensor_positions_x[-1] - sensor_positions_x[0]),
        "array_min_sensor_spacing_m": float(np.min(sensor_spacings_m)) if sensor_spacings_m.size > 0 else 0.0,
        "array_max_sensor_spacing_m": float(np.max(sensor_spacings_m)) if sensor_spacings_m.size > 0 else 0.0,
        "n_beam": int(axis_az_deg.size),
        "frequency_guard_table_json_path": str(json_path.resolve()),
        "frequency_guard_table_csv_path": str(csv_path.resolve()),
        "frequency_guard_table_png_path": str(plot_path.resolve()),
        "records": records,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = [
    "TimeDelayGuardDesignConfig",
    "run_integer_delay_guard_design",
]
