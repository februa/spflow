"""運用スパースアレイ用の周波数別 Kaiser-Bessel シェーディングを設計するモジュール。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float, require_positive_int
from .diagnostic_plotting import require_matplotlib
from .directions import make_directions
from .operational_sparse_array import OperationalSparseArrayDefinition


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


def _direction_cosines_from_azimuth_deg(azimuth_deg: FloatArray) -> FloatArray:
    """水平面方位角を 1 列片舷アレイの方向余弦へ変換する。"""
    azimuth_rad = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    return np.stack(
        [np.cos(azimuth_rad), np.sin(azimuth_rad), np.zeros_like(azimuth_rad)],
        axis=1,
    )


def _kaiser_bessel_channel_window(n_ch: int, beta: float) -> FloatArray:
    """active channel 数に対応する Kaiser-Bessel 窓を返す。

    Args:
        n_ch: active channel 数。単位は本。
        beta: Kaiser-Bessel 窓の形状パラメータ。0 で矩形窓に一致する。

    Returns:
        channel 方向のシェーディング係数。shape は `[n_ch]`、最大値は 1。

    Raises:
        ValueError: `n_ch` が正でない、または `beta` が負の場合。
    """
    require_positive_int("n_ch", int(n_ch))
    require(float(beta) >= 0.0, "beta must be non-negative.")

    if int(n_ch) == 1:
        return np.ones(1, dtype=np.float64)

    window = np.kaiser(int(n_ch), float(beta)).astype(np.float64)
    max_value = float(np.max(window))
    require(max_value > 0.0, "Kaiser-Bessel window normalization failed.")

    # 係数ファイルでは最大係数を 1 に正規化する。
    # 整相時の絶対利得は sum(weights) で正規化し、窓の端係数低下が方位利得を変えないようにする。
    return window / max_value


def _weighted_beam_levels_db20(
    positions_m: FloatArray,
    channel_weights: FloatArray,
    frequency_hz: float,
    target_azimuth_deg: float,
    scan_azimuths_deg: FloatArray,
    sound_speed_m_s: float,
) -> FloatArray:
    """channel shading を含む exact-delay ビーム応答を dB20 で返す。

    Args:
        positions_m: active センサ座標。shape は `[n_active_ch, 3]`、単位は m。
        channel_weights: active channel のシェーディング係数。shape は `[n_active_ch]`。
        frequency_hz: 評価周波数。単位は Hz。
        target_azimuth_deg: 到来方位。単位は deg。
        scan_azimuths_deg: 待受方位軸。shape は `[n_beam]`、単位は deg。
        sound_speed_m_s: 音速。単位は m/s。

    Returns:
        正規化ビーム応答。shape は `[n_beam]`、単位は dB20。
    """
    sensor_positions_m = np.asarray(positions_m, dtype=np.float64)
    weights = np.asarray(channel_weights, dtype=np.float64)
    scan_axis_deg = np.asarray(scan_azimuths_deg, dtype=np.float64)
    require(sensor_positions_m.ndim == 2 and sensor_positions_m.shape[1] == 3, "positions_m must have shape (n_ch, 3).")
    require(weights.ndim == 1 and weights.shape[0] == sensor_positions_m.shape[0], "channel_weights must have shape (n_ch,).")
    require(scan_axis_deg.ndim == 1 and scan_axis_deg.size > 0, "scan_azimuths_deg must be a non-empty 1-D array.")
    require(bool(np.all(np.isfinite(weights))), "channel_weights must contain finite values.")
    require(float(np.sum(weights)) > 0.0, "at least one channel weight must be positive.")
    require_positive_float("frequency_hz", float(frequency_hz))
    require_positive_float("sound_speed_m_s", float(sound_speed_m_s))

    scan_directions = _direction_cosines_from_azimuth_deg(scan_axis_deg)
    target_direction = _direction_cosines_from_azimuth_deg(np.array([float(target_azimuth_deg)], dtype=np.float64))[0]

    # Δtau[ch, scan] = x_ch * (u_target - u_scan) / c。
    # Kaiser-Bessel シェーディングは ch 方向の加重平均として適用し、
    # response = Σ w_ch exp(-j 2π f Δtau) / Σ w_ch によりピーク利得を 0 dB に正規化する。
    direction_delta = float(target_direction[0]) - scan_directions[:, 0]
    phase = (
        -2.0
        * np.pi
        * float(frequency_hz)
        * sensor_positions_m[:, 0][:, np.newaxis]
        * direction_delta[np.newaxis, :]
        / float(sound_speed_m_s)
    )
    response = np.sum(weights[:, np.newaxis] * np.exp(1j * phase), axis=0) / float(np.sum(weights))
    return 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(np.float64).tiny))


def _weighted_waiting_beam_response_db20(
    positions_m: FloatArray,
    channel_weights: FloatArray,
    frequency_hz: float,
    waiting_azimuth_deg: float,
    signal_azimuths_deg: FloatArray,
    sound_speed_m_s: float,
) -> FloatArray:
    """1 本の待受ビームについて、信号方位掃引に対する応答を返す。

    Args:
        positions_m: active センサ座標。shape は `[n_active_ch, 3]`、単位は m。
        channel_weights: active channel のシェーディング係数。shape は `[n_active_ch]`。
        frequency_hz: 評価周波数。単位は Hz。
        waiting_azimuth_deg: 固定整相で張る待受方位。単位は deg。
        signal_azimuths_deg: 掃引する信号方位軸。shape は `[n_signal_az]`、単位は deg。
        sound_speed_m_s: 音速。単位は m/s。

    Returns:
        待受ビーム出力の正規化応答。shape は `[n_signal_az]`、単位は dB20。
    """
    sensor_positions_m = np.asarray(positions_m, dtype=np.float64)
    weights = np.asarray(channel_weights, dtype=np.float64)
    signal_axis_deg = np.asarray(signal_azimuths_deg, dtype=np.float64)
    require(sensor_positions_m.ndim == 2 and sensor_positions_m.shape[1] == 3, "positions_m must have shape (n_ch, 3).")
    require(weights.ndim == 1 and weights.shape[0] == sensor_positions_m.shape[0], "channel_weights must have shape (n_ch,).")
    require(signal_axis_deg.ndim == 1 and signal_axis_deg.size > 0, "signal_azimuths_deg must be a non-empty 1-D array.")
    require(bool(np.all(np.isfinite(weights))), "channel_weights must contain finite values.")
    require(float(np.sum(weights)) > 0.0, "at least one channel weight must be positive.")

    waiting_direction = _direction_cosines_from_azimuth_deg(np.array([float(waiting_azimuth_deg)], dtype=np.float64))[0]
    signal_directions = _direction_cosines_from_azimuth_deg(signal_axis_deg)

    # Δtau[ch, signal] = x_ch * (u_signal - u_waiting) / c。
    # 待受方位を固定したまま信号方位を 0～180 deg で掃引し、
    # 後段のビーム補間が隣接待受ビームの -3 dB 主ローブ重なりで成立するかを評価する。
    direction_delta = signal_directions[:, 0] - float(waiting_direction[0])
    phase = (
        -2.0
        * np.pi
        * float(frequency_hz)
        * sensor_positions_m[:, 0][:, np.newaxis]
        * direction_delta[np.newaxis, :]
        / float(sound_speed_m_s)
    )
    response = np.sum(weights[:, np.newaxis] * np.exp(1j * phase), axis=0) / float(np.sum(weights))
    return 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(np.float64).tiny))


def _mainlobe_peak_margin_db(
    scan_azimuths_deg: FloatArray,
    beam_levels_db20: FloatArray,
    target_azimuth_deg: float,
) -> float:
    """target 近傍 peak と mainlobe 外 peak の差を返す。"""
    axis_deg = np.asarray(scan_azimuths_deg, dtype=np.float64)
    levels_db20 = np.asarray(beam_levels_db20, dtype=np.float64)
    require(axis_deg.ndim == 1, "scan_azimuths_deg must have shape (n_scan,).")
    require(levels_db20.shape == axis_deg.shape, "beam_levels_db20 must have shape (n_scan,).")

    nearest_index = int(np.argmin(np.abs(axis_deg - float(target_azimuth_deg))))
    search_half_width = max(3, int(round(axis_deg.size / 180.0)))
    search_start = max(0, nearest_index - search_half_width)
    search_stop = min(axis_deg.size, nearest_index + search_half_width + 1)
    peak_index = int(search_start + np.argmax(levels_db20[search_start:search_stop]))

    left_index = peak_index
    right_index = peak_index

    # mainlobe は peak から左右の谷までとし、その外側 peak を sidelobe として測る。
    # シェーディングで主ローブ幅が変化するため、固定 guard 幅ではなく応答形状から境界を決める。
    while left_index > 0 and float(levels_db20[left_index - 1]) <= float(levels_db20[left_index]):
        left_index -= 1
    while right_index < levels_db20.size - 1 and float(levels_db20[right_index + 1]) <= float(levels_db20[right_index]):
        right_index += 1

    outside_mask = np.ones(levels_db20.size, dtype=bool)
    outside_mask[left_index : right_index + 1] = False
    if not bool(np.any(outside_mask)):
        return float("inf")
    return float(levels_db20[peak_index] - np.max(levels_db20[outside_mask]))


def _three_db_mainlobe_interval_deg(
    signal_azimuths_deg: FloatArray,
    beam_levels_db20: FloatArray,
    down_db: float,
) -> tuple[float, float, float, float]:
    """信号方位掃引応答から peak と -down_db 主ローブ範囲を抽出する。

    Args:
        signal_azimuths_deg: 信号方位軸。shape は `[n_signal_az]`、単位は deg。
        beam_levels_db20: 待受ビーム応答。shape は `[n_signal_az]`、単位は dB20。
        down_db: peak から何 dB 下がった範囲までを主ローブとするか。単位は dB。

    Returns:
        `(left_deg, right_deg, peak_azimuth_deg, peak_level_db20)`。
        `left_deg` と `right_deg` は peak 周辺で応答が `peak - down_db` 以上となる連続範囲の端点である。
    """
    signal_axis_deg = np.asarray(signal_azimuths_deg, dtype=np.float64)
    levels_db20 = np.asarray(beam_levels_db20, dtype=np.float64)
    require(signal_axis_deg.ndim == 1 and signal_axis_deg.size > 0, "signal_azimuths_deg must be a non-empty 1-D array.")
    require(levels_db20.shape == signal_axis_deg.shape, "beam_levels_db20 must have shape (n_signal_az,).")
    require_positive_float("down_db", float(down_db))

    peak_index = int(np.argmax(levels_db20))
    peak_level_db20 = float(levels_db20[peak_index])
    threshold_db20 = peak_level_db20 - float(down_db)
    left_index = peak_index
    right_index = peak_index

    # peak から連続して threshold 以上の点だけを主ローブ範囲とする。
    # 離れた grating lobe が -3 dB 以内に入っても、補間に使う主ローブ幅へ混入させないためである。
    while left_index > 0 and float(levels_db20[left_index - 1]) >= threshold_db20:
        left_index -= 1
    while right_index < levels_db20.size - 1 and float(levels_db20[right_index + 1]) >= threshold_db20:
        right_index += 1

    return (
        float(signal_axis_deg[left_index]),
        float(signal_axis_deg[right_index]),
        float(signal_axis_deg[peak_index]),
        float(peak_level_db20),
    )


def _evaluate_three_db_overlap_metrics(
    positions_m: FloatArray,
    channel_weights: FloatArray,
    frequency_hz: float,
    waiting_azimuths_deg: FloatArray,
    signal_azimuths_deg: FloatArray,
    sound_speed_m_s: float,
    down_db: float,
) -> dict[str, float | bool]:
    """隣接待受ビームの -3 dB 主ローブ範囲が重なるかを評価する。

    Returns:
        `minimum_overlap_margin_deg` は、隣接する 2 本の待受ビームについて
        `left beam の右 -3 dB 境界 - right beam の左 -3 dB 境界` の最小値である。
        0 以上なら全隣接ペアで -3 dB 範囲が cross / overlap している。
    """
    waiting_axis_deg = np.asarray(waiting_azimuths_deg, dtype=np.float64)
    signal_axis_deg = np.asarray(signal_azimuths_deg, dtype=np.float64)
    require(waiting_axis_deg.ndim == 1 and waiting_axis_deg.size >= 2, "waiting_azimuths_deg must contain at least two beams.")
    require(signal_axis_deg.ndim == 1 and signal_axis_deg.size > 0, "signal_azimuths_deg must be a non-empty 1-D array.")

    left_edges_deg: list[float] = []
    right_edges_deg: list[float] = []
    peak_azimuths_deg: list[float] = []

    for waiting_azimuth_deg in waiting_axis_deg.tolist():
        levels_db20 = _weighted_waiting_beam_response_db20(
            positions_m=positions_m,
            channel_weights=channel_weights,
            frequency_hz=float(frequency_hz),
            waiting_azimuth_deg=float(waiting_azimuth_deg),
            signal_azimuths_deg=signal_axis_deg,
            sound_speed_m_s=float(sound_speed_m_s),
        )
        left_deg, right_deg, peak_azimuth_deg, _ = _three_db_mainlobe_interval_deg(
            signal_azimuths_deg=signal_axis_deg,
            beam_levels_db20=levels_db20,
            down_db=float(down_db),
        )
        left_edges_deg.append(float(left_deg))
        right_edges_deg.append(float(right_deg))
        peak_azimuths_deg.append(float(peak_azimuth_deg))

    overlap_margins_deg = np.asarray(
        [right_edges_deg[index] - left_edges_deg[index + 1] for index in range(waiting_axis_deg.size - 1)],
        dtype=np.float64,
    )
    widths_deg = np.asarray(
        [right_edges_deg[index] - left_edges_deg[index] for index in range(waiting_axis_deg.size)],
        dtype=np.float64,
    )
    peak_errors_deg = np.abs(np.asarray(peak_azimuths_deg, dtype=np.float64) - waiting_axis_deg)

    return {
        "minimum_overlap_margin_deg": float(np.min(overlap_margins_deg)),
        "minimum_three_db_width_deg": float(np.min(widths_deg)),
        "maximum_three_db_width_deg": float(np.max(widths_deg)),
        "maximum_peak_error_deg": float(np.max(peak_errors_deg)),
        "meets_three_db_overlap": bool(np.min(overlap_margins_deg) >= 0.0),
    }


def _evaluate_peak_margin_db(
    positions_m: FloatArray,
    channel_weights: FloatArray,
    frequency_hz: float,
    axis_azimuth_deg: FloatArray,
    evaluation_azimuths_deg: FloatArray,
    sound_speed_m_s: float,
) -> float:
    """評価方位群に対する最悪 mainlobe-sidelobe margin を返す。"""
    margins_db: list[float] = []
    for target_azimuth_deg in np.asarray(evaluation_azimuths_deg, dtype=np.float64).tolist():
        beam_levels_db20 = _weighted_beam_levels_db20(
            positions_m=positions_m,
            channel_weights=channel_weights,
            frequency_hz=float(frequency_hz),
            target_azimuth_deg=float(target_azimuth_deg),
            scan_azimuths_deg=np.asarray(axis_azimuth_deg, dtype=np.float64),
            sound_speed_m_s=float(sound_speed_m_s),
        )
        margins_db.append(
            _mainlobe_peak_margin_db(
                scan_azimuths_deg=np.asarray(axis_azimuth_deg, dtype=np.float64),
                beam_levels_db20=beam_levels_db20,
                target_azimuth_deg=float(target_azimuth_deg),
            )
        )
    return float(np.min(np.asarray(margins_db, dtype=np.float64)))


def _equivalent_aperture_m(positions_m: FloatArray, channel_weights: FloatArray) -> float:
    """重み付き 2 次モーメントから等価開口長を見積もる。"""
    sensor_x_m = np.asarray(positions_m, dtype=np.float64)[:, 0]
    weights = np.asarray(channel_weights, dtype=np.float64)
    weight_sum = float(np.sum(weights))
    require(weight_sum > 0.0, "channel_weights must contain positive weight.")

    # 一様開口幅 D の分散は D^2 / 12 である。
    # 重み付きセンサ分散をこの式へ戻すことで、シェーディング後の実効開口を比較しやすくする。
    weighted_center_m = float(np.sum(weights * sensor_x_m) / weight_sum)
    weighted_variance_m2 = float(np.sum(weights * (sensor_x_m - weighted_center_m) ** 2) / weight_sum)
    return float(np.sqrt(12.0 * max(weighted_variance_m2, 0.0)))


@dataclass(frozen=True)
class OperationalShadingDesignConfig:
    """周波数別 Kaiser-Bessel シェーディングの設計条件を保持する。

    このクラスは、運用スパースアレイ JSON、評価周波数、Kaiser-Bessel beta 候補、
    待受方位数候補、隣接待受ビームの -3 dB 主ローブ重なり条件、
    sidelobe margin 条件、保存先を保持する。

    入力はアレイ定義ファイルと探索候補であり、出力は
    `run_operational_shading_design()` が保存するシェーディング係数 JSON / CSV / PNG である。
    アレイ CH 数は JSON 内の `positions_m.shape[0]` から読み取る。

    小数遅延 FIR の設計、実波形整相、SLC 重み更新は責務に含めない。
    信号処理上は、固定整相のチャネル加重係数を周波数ごとに事前設計する条件に位置づく。
    """

    output_json_path: Path
    operational_array_definition_path: Path
    output_csv_path: Path | None = None
    output_summary_png_path: Path | None = None
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
    candidate_kaiser_beta: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0)
    candidate_n_beam_az_real: tuple[int, ...] = (151, 181, 241, 303)
    evaluation_azimuths_deg: tuple[float, ...] = (60.0, 90.0, 120.0)
    three_db_down_db: float = 3.0
    required_peak_margin_db: float = 13.0
    signal_azimuth_count: int = 3601

    def __post_init__(self) -> None:
        """設計条件の範囲、単位、候補軸の単調性を検証する。"""
        require(Path(self.operational_array_definition_path).exists(), "operational_array_definition_path must exist.")
        require_positive_float("fs_hz", float(self.fs_hz))
        require_positive_float("sound_speed_m_s", float(self.sound_speed_m_s))
        require_positive_float("three_db_down_db", float(self.three_db_down_db))
        require_positive_float("required_peak_margin_db", float(self.required_peak_margin_db))
        require_positive_int("signal_azimuth_count", int(self.signal_azimuth_count))

        frequencies_hz = np.asarray(self.frequency_grid_hz, dtype=np.float64)
        require(frequencies_hz.ndim == 1 and frequencies_hz.size > 0, "frequency_grid_hz must be a non-empty 1-D sequence.")
        require(bool(np.all(np.isfinite(frequencies_hz))), "frequency_grid_hz must contain finite values.")
        require(bool(np.all(frequencies_hz > 0.0)), "frequency_grid_hz must contain only positive values.")
        require(bool(np.all(np.diff(frequencies_hz) > 0.0)), "frequency_grid_hz must be strictly increasing.")

        beta = np.asarray(self.candidate_kaiser_beta, dtype=np.float64)
        require(beta.ndim == 1 and beta.size > 0, "candidate_kaiser_beta must be a non-empty 1-D sequence.")
        require(bool(np.all(beta >= 0.0)), "candidate_kaiser_beta must contain non-negative values.")
        require(bool(np.all(np.diff(beta) >= 0.0)), "candidate_kaiser_beta must be sorted in ascending order.")

        beam_counts = np.asarray(self.candidate_n_beam_az_real, dtype=np.int64)
        require(beam_counts.ndim == 1 and beam_counts.size > 0, "candidate_n_beam_az_real must be a non-empty 1-D sequence.")
        require(bool(np.all(beam_counts >= 2)), "candidate_n_beam_az_real must contain counts >= 2.")
        require(bool(np.all(np.diff(beam_counts) > 0)), "candidate_n_beam_az_real must be strictly increasing.")

        azimuths_deg = np.asarray(self.evaluation_azimuths_deg, dtype=np.float64)
        require(azimuths_deg.ndim == 1 and azimuths_deg.size > 0, "evaluation_azimuths_deg must be a non-empty 1-D sequence.")
        require(bool(np.all((0.0 <= azimuths_deg) & (azimuths_deg <= 180.0))), "evaluation_azimuths_deg must lie in [0, 180].")


@dataclass(frozen=True)
class OperationalShadingDefinition:
    """保存済み周波数別シェーディング係数を保持する。

    このクラスは、周波数ごとの Kaiser-Bessel beta、待受方位数、full channel 係数、
    active channel index、評価結果を保持し、JSON ファイルとの相互変換を担当する。

    入力は `frequency_grid_hz[n_frequency]`、`shading_coefficients_by_frequency[n_frequency, n_ch]`、
    `active_channel_indices_by_frequency`、設計 record 群であり、出力は固定整相が使用する
    周波数別 channel shading 係数である。

    係数の実適用、FIR 畳み込み、SLC 係数更新は責務に含めない。
    信号処理上は、固定整相時に ch 軸へ掛ける窓係数ファイル表現に位置づく。
    """

    schema_version: int
    fs_hz: float
    sound_speed_m_s: float
    operational_array_definition_path: str
    frequency_grid_hz: FloatArray
    shading_coefficients_by_frequency: FloatArray
    selected_kaiser_beta_by_frequency: FloatArray
    selected_n_beam_az_real_by_frequency: IntArray
    active_channel_indices_by_frequency: tuple[IntArray, ...]
    records: tuple[dict[str, object], ...]
    formula: dict[str, object]

    def __post_init__(self) -> None:
        """保存済み係数の shape、周波数軸、active index 範囲を検証する。"""
        frequencies_hz = np.asarray(self.frequency_grid_hz, dtype=np.float64)
        coefficients = np.asarray(self.shading_coefficients_by_frequency, dtype=np.float64)
        beta = np.asarray(self.selected_kaiser_beta_by_frequency, dtype=np.float64)
        beam_counts = np.asarray(self.selected_n_beam_az_real_by_frequency, dtype=np.int64)
        require(frequencies_hz.ndim == 1 and frequencies_hz.size > 0, "frequency_grid_hz must have shape (n_frequency,).")
        require(coefficients.ndim == 2, "shading_coefficients_by_frequency must have shape (n_frequency, n_ch).")
        require(coefficients.shape[0] == frequencies_hz.size, "coefficient frequency axis mismatch.")
        require(beta.shape == frequencies_hz.shape, "selected beta axis mismatch.")
        require(beam_counts.shape == frequencies_hz.shape, "selected beam count axis mismatch.")
        require(len(self.active_channel_indices_by_frequency) == frequencies_hz.size, "active index axis mismatch.")
        require(bool(np.all(np.isfinite(coefficients))), "shading coefficients must contain finite values.")
        require(bool(np.all(coefficients >= 0.0)), "shading coefficients must be non-negative.")

        normalized_indices: list[IntArray] = []
        for active_indices in self.active_channel_indices_by_frequency:
            index_array = np.asarray(active_indices, dtype=np.int64)
            require(index_array.ndim == 1 and index_array.size > 0, "active channel indices must be non-empty 1-D arrays.")
            require(bool(np.all((0 <= index_array) & (index_array < coefficients.shape[1]))), "active channel index is out of range.")
            normalized_indices.append(index_array)

        object.__setattr__(self, "frequency_grid_hz", frequencies_hz)
        object.__setattr__(self, "shading_coefficients_by_frequency", coefficients)
        object.__setattr__(self, "selected_kaiser_beta_by_frequency", beta)
        object.__setattr__(self, "selected_n_beam_az_real_by_frequency", beam_counts)
        object.__setattr__(self, "active_channel_indices_by_frequency", tuple(normalized_indices))

    @property
    def n_ch(self) -> int:
        """係数ファイルが想定する物理 CH 数を返す。"""
        return int(self.shading_coefficients_by_frequency.shape[1])

    def _frequency_index_for_frequency(self, frequency_hz: float) -> int:
        """任意周波数に対し、保存周波数表の ceiling 側 index を返す。"""
        require_positive_float("frequency_hz", float(frequency_hz))
        index = int(np.searchsorted(self.frequency_grid_hz, float(frequency_hz), side="left"))
        if index >= self.frequency_grid_hz.size:
            index = int(self.frequency_grid_hz.size - 1)
        return index

    def coefficients_for_frequency(self, frequency_hz: float) -> FloatArray:
        """指定周波数で使用する full channel 係数を返す。

        Args:
            frequency_hz: 対象周波数。単位は Hz。

        Returns:
            full channel シェーディング係数。shape は `[n_ch]`。
            active 外チャネルは 0 であり、active 内は Kaiser-Bessel 窓係数である。

        Raises:
            ValueError: `frequency_hz` が正でない場合。
        """
        index = self._frequency_index_for_frequency(float(frequency_hz))
        return np.asarray(self.shading_coefficients_by_frequency[index], dtype=np.float64)

    def active_channel_indices_for_frequency(self, frequency_hz: float) -> IntArray:
        """指定周波数で係数が対応する active channel index を返す。"""
        index = self._frequency_index_for_frequency(float(frequency_hz))
        return np.asarray(self.active_channel_indices_by_frequency[index], dtype=np.int64)

    def to_payload(self) -> dict[str, object]:
        """JSON 保存用 payload へ変換する。"""
        return {
            "schema_version": int(self.schema_version),
            "fs_hz": float(self.fs_hz),
            "sound_speed_m_s": float(self.sound_speed_m_s),
            "operational_array_definition_path": str(self.operational_array_definition_path),
            "frequency_grid_hz": [float(value) for value in self.frequency_grid_hz.tolist()],
            "shading_coefficients_by_frequency": self.shading_coefficients_by_frequency.tolist(),
            "selected_kaiser_beta_by_frequency": [float(value) for value in self.selected_kaiser_beta_by_frequency.tolist()],
            "selected_n_beam_az_real_by_frequency": [int(value) for value in self.selected_n_beam_az_real_by_frequency.tolist()],
            "active_channel_indices_by_frequency": [
                [int(index) for index in active_indices.tolist()] for active_indices in self.active_channel_indices_by_frequency
            ],
            "records": [dict(record) for record in self.records],
            "formula": dict(self.formula),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "OperationalShadingDefinition":
        """JSON payload からシェーディング定義を復元する。"""
        raw_active_indices = payload["active_channel_indices_by_frequency"]
        raw_records = payload["records"]
        raw_formula = payload["formula"]
        if not isinstance(raw_active_indices, list):
            raise TypeError("active_channel_indices_by_frequency must be a list.")
        if not isinstance(raw_records, list):
            raise TypeError("records must be a list.")
        if not isinstance(raw_formula, dict):
            raise TypeError("formula must be a dict.")

        active_index_table: list[IntArray] = []
        for raw_indices in raw_active_indices:
            # JSON 由来の list は実行時に要素型が保証されないため、
            # numpy 配列化と OperationalShadingDefinition.__post_init__ の範囲検証で契約を確定する。
            active_index_table.append(np.asarray(raw_indices, dtype=np.int64))

        records: list[dict[str, object]] = []
        for raw_record in raw_records:
            if not isinstance(raw_record, dict):
                raise TypeError("each record must be a dict.")
            records.append({str(key): value for key, value in raw_record.items()})

        return cls(
            schema_version=int(payload["schema_version"]),
            fs_hz=float(payload["fs_hz"]),
            sound_speed_m_s=float(payload["sound_speed_m_s"]),
            operational_array_definition_path=str(payload["operational_array_definition_path"]),
            frequency_grid_hz=np.asarray(payload["frequency_grid_hz"], dtype=np.float64),
            shading_coefficients_by_frequency=np.asarray(payload["shading_coefficients_by_frequency"], dtype=np.float64),
            selected_kaiser_beta_by_frequency=np.asarray(payload["selected_kaiser_beta_by_frequency"], dtype=np.float64),
            selected_n_beam_az_real_by_frequency=np.asarray(payload["selected_n_beam_az_real_by_frequency"], dtype=np.int64),
            active_channel_indices_by_frequency=tuple(active_index_table),
            records=tuple(records),
            formula={str(key): value for key, value in raw_formula.items()},
        )

    def save_json(self, path: str | Path) -> None:
        """シェーディング定義を JSON ファイルとして保存する。"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_payload(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "OperationalShadingDefinition":
        """保存済み JSON からシェーディング定義を読み込む。"""
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def _evaluate_candidate(
    active_positions_m: FloatArray,
    frequency_hz: float,
    beta: float,
    n_beam_az_real: int,
    config: OperationalShadingDesignConfig,
) -> tuple[FloatArray, dict[str, float | bool], float]:
    """単一候補の係数、-3 dB 重なり指標、peak margin を評価する。"""
    _, axis_azimuth_deg, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=int(n_beam_az_real),
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    signal_azimuths_deg = np.linspace(0.0, 180.0, int(config.signal_azimuth_count), dtype=np.float64)
    weights = _kaiser_bessel_channel_window(int(active_positions_m.shape[0]), float(beta))
    overlap_metrics = _evaluate_three_db_overlap_metrics(
        positions_m=active_positions_m,
        channel_weights=weights,
        frequency_hz=float(frequency_hz),
        waiting_azimuths_deg=np.asarray(axis_azimuth_deg, dtype=np.float64),
        signal_azimuths_deg=signal_azimuths_deg,
        sound_speed_m_s=float(config.sound_speed_m_s),
        down_db=float(config.three_db_down_db),
    )
    peak_margin_db = _evaluate_peak_margin_db(
        positions_m=active_positions_m,
        channel_weights=weights,
        frequency_hz=float(frequency_hz),
        axis_azimuth_deg=np.asarray(axis_azimuth_deg, dtype=np.float64),
        evaluation_azimuths_deg=np.asarray(config.evaluation_azimuths_deg, dtype=np.float64),
        sound_speed_m_s=float(config.sound_speed_m_s),
    )
    return weights, overlap_metrics, float(peak_margin_db)


def _select_shading_for_frequency(
    active_positions_m: FloatArray,
    frequency_hz: float,
    config: OperationalShadingDesignConfig,
) -> tuple[FloatArray, int, float, dict[str, float | bool], float, bool]:
    """-3 dB 重なり条件を満たす最小待受方位数と最小 beta の組を選ぶ。"""
    best_penalty = float("inf")
    best_result: tuple[FloatArray, int, float, dict[str, float | bool], float, bool] | None = None

    for n_beam_az_real in config.candidate_n_beam_az_real:
        for beta in config.candidate_kaiser_beta:
            weights, overlap_metrics, peak_margin_db = _evaluate_candidate(
                active_positions_m=active_positions_m,
                frequency_hz=float(frequency_hz),
                beta=float(beta),
                n_beam_az_real=int(n_beam_az_real),
                config=config,
            )
            minimum_overlap_margin_deg = float(overlap_metrics["minimum_overlap_margin_deg"])
            meets_overlap = bool(overlap_metrics["meets_three_db_overlap"])
            meets_margin = peak_margin_db >= float(config.required_peak_margin_db)
            meets_all = bool(meets_overlap and meets_margin)
            if meets_all:
                return weights, int(n_beam_az_real), float(beta), overlap_metrics, float(peak_margin_db), True

            # 候補が全滅した場合も診断ファイルを残せるよう、条件未達量の合計が最小の候補を保持する。
            # overlap は負値が隣接 -3 dB 範囲の gap を意味するため、負側だけを penalty として扱う。
            penalty = max(0.0, -minimum_overlap_margin_deg) + max(
                0.0,
                float(config.required_peak_margin_db) - peak_margin_db,
            )
            if penalty < best_penalty:
                best_penalty = float(penalty)
                best_result = (
                    weights,
                    int(n_beam_az_real),
                    float(beta),
                    overlap_metrics,
                    float(peak_margin_db),
                    False,
                )

    if best_result is None:
        # 候補軸が空の場合は __post_init__ で拒否されるため通常は到達しない。
        # ここでは将来の条件追加で全候補を skip した場合も、None を返さず明示的に失敗させる。
        raise ValueError("shading candidate search failed.")
    return best_result


def _plot_shading_summary(
    output_path: Path,
    records: list[dict[str, object]],
    config: OperationalShadingDesignConfig,
) -> None:
    """周波数別の beta、待受方位数、-3 dB 重なり量、margin を PNG 保存する。"""
    require_matplotlib()
    import matplotlib.pyplot as plt

    frequencies_hz = np.asarray([float(record["frequency_hz"]) for record in records], dtype=np.float64)
    selected_beta = np.asarray([float(record["selected_kaiser_beta"]) for record in records], dtype=np.float64)
    selected_beams = np.asarray([int(record["selected_n_beam_az_real"]) for record in records], dtype=np.int64)
    overlap_margin = np.asarray([float(record["minimum_three_db_overlap_margin_deg"]) for record in records], dtype=np.float64)
    peak_margin = np.asarray([float(record["worst_peak_margin_db"]) for record in records], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes[0, 0].plot(frequencies_hz, selected_beta, marker="o")
    axes[0, 0].set_ylabel("Kaiser beta")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(frequencies_hz, selected_beams, marker="o", color="tab:orange")
    axes[0, 1].set_ylabel("Waiting beams")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(frequencies_hz, overlap_margin, marker="o", color="tab:green")
    axes[1, 0].axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    axes[1, 0].set_ylabel("3 dB overlap [deg]")
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(frequencies_hz, peak_margin, marker="o", color="tab:red")
    axes[1, 1].axhline(float(config.required_peak_margin_db), linestyle="--", color="black", linewidth=1.0)
    axes[1, 1].set_ylabel("Peak margin [dB]")
    axes[1, 1].set_xlabel("Frequency [Hz]")
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle("Operational Kaiser-Bessel shading design")
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_operational_shading_design(config: OperationalShadingDesignConfig) -> dict[str, object]:
    """運用スパースアレイ用の周波数別 Kaiser-Bessel シェーディングを設計して保存する。

    Args:
        config: アレイ定義 JSON、探索候補、評価条件、保存先を含む設計条件。

    Returns:
        周波数ごとの beta、待受方位数、-3 dB 重なり量、peak margin、保存先を含む summary。

    Raises:
        ValueError: 入力ファイルや探索候補が不正な場合。
    """
    array_definition = OperationalSparseArrayDefinition.load_json(Path(config.operational_array_definition_path))

    full_coefficients: list[FloatArray] = []
    active_index_table: list[IntArray] = []
    selected_beta: list[float] = []
    selected_beam_counts: list[int] = []
    records: list[dict[str, object]] = []

    for frequency_hz in np.asarray(config.frequency_grid_hz, dtype=np.float64).tolist():
        active_indices = array_definition.active_channel_indices_for_frequency(float(frequency_hz))

        # active_positions_m shape: [n_active_ch, 3]。
        # 運用アレイでは周波数ごとに active subset が変わるため、シェーディング係数も active 配置ごとに設計する。
        active_positions_m = np.asarray(array_definition.positions_m, dtype=np.float64)[active_indices]
        weights, n_beam_az_real, beta, overlap_metrics, peak_margin_db, meets_all = _select_shading_for_frequency(
            active_positions_m=active_positions_m,
            frequency_hz=float(frequency_hz),
            config=config,
        )

        full_weight = np.zeros(array_definition.n_ch, dtype=np.float64)
        full_weight[active_indices] = weights
        full_coefficients.append(full_weight)
        active_index_table.append(active_indices)
        selected_beta.append(float(beta))
        selected_beam_counts.append(int(n_beam_az_real))

        active_x_m = active_positions_m[:, 0]
        physical_aperture_m = float(np.max(active_x_m) - np.min(active_x_m)) if active_x_m.size > 1 else 0.0
        equivalent_aperture_m = _equivalent_aperture_m(active_positions_m, weights)

        records.append(
            {
                "frequency_hz": float(frequency_hz),
                "active_channel_count": int(active_indices.size),
                "active_aperture_m": float(physical_aperture_m),
                "equivalent_aperture_m": float(equivalent_aperture_m),
                "selected_kaiser_beta": float(beta),
                "selected_n_beam_az_real": int(n_beam_az_real),
                "minimum_three_db_overlap_margin_deg": float(overlap_metrics["minimum_overlap_margin_deg"]),
                "minimum_three_db_width_deg": float(overlap_metrics["minimum_three_db_width_deg"]),
                "maximum_three_db_width_deg": float(overlap_metrics["maximum_three_db_width_deg"]),
                "maximum_peak_error_deg": float(overlap_metrics["maximum_peak_error_deg"]),
                "worst_peak_margin_db": float(peak_margin_db),
                "meets_three_db_overlap": bool(overlap_metrics["meets_three_db_overlap"]),
                "meets_peak_margin": bool(peak_margin_db >= float(config.required_peak_margin_db)),
                "meets_all": bool(meets_all),
                "active_weight_min": float(np.min(weights)),
                "active_weight_max": float(np.max(weights)),
                "active_weight_sum": float(np.sum(weights)),
            }
        )

    shading_definition = OperationalShadingDefinition(
        schema_version=1,
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
        operational_array_definition_path=str(Path(config.operational_array_definition_path).resolve()),
        frequency_grid_hz=np.asarray(config.frequency_grid_hz, dtype=np.float64),
        shading_coefficients_by_frequency=np.stack(full_coefficients, axis=0),
        selected_kaiser_beta_by_frequency=np.asarray(selected_beta, dtype=np.float64),
        selected_n_beam_az_real_by_frequency=np.asarray(selected_beam_counts, dtype=np.int64),
        active_channel_indices_by_frequency=tuple(active_index_table),
        records=tuple(records),
        formula={
            "window": "Kaiser-Bessel",
            "normalization": "active window max is 1.0; beamforming response must be normalized by sum(weights).",
            "three_db_overlap_condition": "For each waiting beam, sweep signal azimuth from 0 to 180 deg and require adjacent waiting beams' -3 dB mainlobe intervals to overlap.",
            "three_db_down_db": float(config.three_db_down_db),
            "required_peak_margin_db": float(config.required_peak_margin_db),
            "signal_azimuth_count": int(config.signal_azimuth_count),
        },
    )
    shading_definition.save_json(config.output_json_path)

    csv_path = config.output_csv_path
    if csv_path is not None:
        output_csv_path = Path(csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    png_path = config.output_summary_png_path
    if png_path is not None:
        _plot_shading_summary(Path(png_path), records, config)

    meets_all = bool(all(bool(record["meets_all"]) for record in records))
    summary: dict[str, object] = {
        "operational_array_definition_path": str(Path(config.operational_array_definition_path).resolve()),
        "shading_definition_json_path": str(Path(config.output_json_path).resolve()),
        "shading_design_csv_path": None if csv_path is None else str(Path(csv_path).resolve()),
        "shading_summary_png_path": None if png_path is None else str(Path(png_path).resolve()),
        "frequency_grid_hz": [float(value) for value in config.frequency_grid_hz],
        "selected_kaiser_beta_by_frequency": [float(value) for value in selected_beta],
        "selected_n_beam_az_real_by_frequency": [int(value) for value in selected_beam_counts],
        "minimum_three_db_overlap_margin_deg_by_frequency": [float(record["minimum_three_db_overlap_margin_deg"]) for record in records],
        "minimum_three_db_width_deg_by_frequency": [float(record["minimum_three_db_width_deg"]) for record in records],
        "maximum_three_db_width_deg_by_frequency": [float(record["maximum_three_db_width_deg"]) for record in records],
        "maximum_peak_error_deg_by_frequency": [float(record["maximum_peak_error_deg"]) for record in records],
        "worst_peak_margin_db_by_frequency": [float(record["worst_peak_margin_db"]) for record in records],
        "meets_all": bool(meets_all),
        "records": records,
    }
    return summary



@dataclass(frozen=True)
class OperationalFixedBeamShadingDesignConfig:
    """指定ビーム数で 3 dB down 幅に近い Kaiser-Bessel シェーディングを設計する条件を保持する。

    このクラスは、運用スパースアレイ JSON、固定する待受方位数、Kaiser-Bessel beta 候補、
    `-3 dB` 主ローブ overlap の目標値、保存先を保持する。

    入力はアレイ定義ファイルと固定ビーム数であり、出力は
    `run_operational_fixed_beam_shading_design()` が保存するシェーディング係数 JSON / CSV / PNG である。
    CH 数はアレイ JSON 内の `positions_m.shape[0]` から読み取る。

    小数遅延 FIR の設計、実波形整相、SLC 重み更新は責務に含めない。
    信号処理上は、SLC などでビーム数を先に固定した場合に、
    channel shading だけで 3 dB down 幅をどこまで調整できるかを評価する設計条件に位置づく。
    """

    output_json_path: Path
    operational_array_definition_path: Path
    n_beam_az_real: int
    output_csv_path: Path | None = None
    output_summary_png_path: Path | None = None
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
    candidate_kaiser_beta: tuple[float, ...] = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        12.0,
        16.0,
        20.0,
    )
    target_overlap_margin_deg: float = 0.0
    target_overlap_tolerance_deg: float = 0.5
    evaluation_azimuths_deg: tuple[float, ...] = (60.0, 90.0, 120.0)
    three_db_down_db: float = 3.0
    required_peak_margin_db: float = 13.0
    signal_azimuth_count: int = 3601

    def __post_init__(self) -> None:
        """指定ビーム数設計条件の範囲、単位、候補軸を検証する。"""
        require(Path(self.operational_array_definition_path).exists(), "operational_array_definition_path must exist.")
        require_positive_int("n_beam_az_real", int(self.n_beam_az_real))
        require_positive_float("fs_hz", float(self.fs_hz))
        require_positive_float("sound_speed_m_s", float(self.sound_speed_m_s))
        require(float(self.target_overlap_tolerance_deg) >= 0.0, "target_overlap_tolerance_deg must be non-negative.")
        require_positive_float("three_db_down_db", float(self.three_db_down_db))
        require_positive_float("required_peak_margin_db", float(self.required_peak_margin_db))
        require_positive_int("signal_azimuth_count", int(self.signal_azimuth_count))

        frequencies_hz = np.asarray(self.frequency_grid_hz, dtype=np.float64)
        require(frequencies_hz.ndim == 1 and frequencies_hz.size > 0, "frequency_grid_hz must be a non-empty 1-D sequence.")
        require(bool(np.all(np.isfinite(frequencies_hz))), "frequency_grid_hz must contain finite values.")
        require(bool(np.all(frequencies_hz > 0.0)), "frequency_grid_hz must contain only positive values.")
        require(bool(np.all(np.diff(frequencies_hz) > 0.0)), "frequency_grid_hz must be strictly increasing.")

        beta = np.asarray(self.candidate_kaiser_beta, dtype=np.float64)
        require(beta.ndim == 1 and beta.size > 0, "candidate_kaiser_beta must be a non-empty 1-D sequence.")
        require(bool(np.all(beta >= 0.0)), "candidate_kaiser_beta must contain non-negative values.")
        require(bool(np.all(np.diff(beta) >= 0.0)), "candidate_kaiser_beta must be sorted in ascending order.")

        azimuths_deg = np.asarray(self.evaluation_azimuths_deg, dtype=np.float64)
        require(azimuths_deg.ndim == 1 and azimuths_deg.size > 0, "evaluation_azimuths_deg must be a non-empty 1-D sequence.")
        require(bool(np.all((0.0 <= azimuths_deg) & (azimuths_deg <= 180.0))), "evaluation_azimuths_deg must lie in [0, 180].")


def _evaluate_fixed_beam_candidate(
    active_positions_m: FloatArray,
    frequency_hz: float,
    beta: float,
    config: OperationalFixedBeamShadingDesignConfig,
) -> tuple[FloatArray, dict[str, float | bool], float]:
    """指定ビーム数条件で単一 beta 候補の幅一致度と peak margin を評価する。"""
    _, axis_azimuth_deg, _ = make_directions(
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
    signal_azimuths_deg = np.linspace(0.0, 180.0, int(config.signal_azimuth_count), dtype=np.float64)
    weights = _kaiser_bessel_channel_window(int(active_positions_m.shape[0]), float(beta))
    overlap_metrics = _evaluate_three_db_overlap_metrics(
        positions_m=active_positions_m,
        channel_weights=weights,
        frequency_hz=float(frequency_hz),
        waiting_azimuths_deg=np.asarray(axis_azimuth_deg, dtype=np.float64),
        signal_azimuths_deg=signal_azimuths_deg,
        sound_speed_m_s=float(config.sound_speed_m_s),
        down_db=float(config.three_db_down_db),
    )
    peak_margin_db = _evaluate_peak_margin_db(
        positions_m=active_positions_m,
        channel_weights=weights,
        frequency_hz=float(frequency_hz),
        axis_azimuth_deg=np.asarray(axis_azimuth_deg, dtype=np.float64),
        evaluation_azimuths_deg=np.asarray(config.evaluation_azimuths_deg, dtype=np.float64),
        sound_speed_m_s=float(config.sound_speed_m_s),
    )
    return weights, overlap_metrics, float(peak_margin_db)


def _select_fixed_beam_shading_for_frequency(
    active_positions_m: FloatArray,
    frequency_hz: float,
    config: OperationalFixedBeamShadingDesignConfig,
) -> tuple[FloatArray, float, dict[str, float | bool], float, bool, bool]:
    """指定ビーム数で 3 dB overlap 目標に最も近い beta を選ぶ。"""
    best_score = float("inf")
    best_result: tuple[FloatArray, float, dict[str, float | bool], float, bool, bool] | None = None

    for beta in config.candidate_kaiser_beta:
        weights, overlap_metrics, peak_margin_db = _evaluate_fixed_beam_candidate(
            active_positions_m=active_positions_m,
            frequency_hz=float(frequency_hz),
            beta=float(beta),
            config=config,
        )
        overlap_margin_deg = float(overlap_metrics["minimum_overlap_margin_deg"])
        target_error_deg = abs(overlap_margin_deg - float(config.target_overlap_margin_deg))
        meets_width_target = target_error_deg <= float(config.target_overlap_tolerance_deg)
        meets_peak_margin = peak_margin_db >= float(config.required_peak_margin_db)

        # 3 dB 幅一致を主目的としつつ、peak margin 未達は penalty を加える。
        # 指定ビーム数が多すぎる場合、Kaiser 窓では主ローブを狭くできないため beta=0 が最良になり得る。
        score = target_error_deg + max(0.0, float(config.required_peak_margin_db) - peak_margin_db)
        if score < best_score:
            best_score = float(score)
            best_result = (
                weights,
                float(beta),
                overlap_metrics,
                float(peak_margin_db),
                bool(meets_width_target),
                bool(meets_peak_margin),
            )

    if best_result is None:
        raise ValueError("fixed beam shading candidate search failed.")
    return best_result


def run_operational_fixed_beam_shading_design(config: OperationalFixedBeamShadingDesignConfig) -> dict[str, object]:
    """指定ビーム数で 3 dB down 幅に近いシェーディング係数を設計して保存する。

    Args:
        config: アレイ定義 JSON、固定ビーム数、beta 候補、保存先を含む設計条件。

    Returns:
        周波数ごとの beta、3 dB overlap 目標誤差、peak margin、保存先を含む summary。

    Raises:
        ValueError: 入力ファイルや候補条件が不正な場合。
    """
    array_definition = OperationalSparseArrayDefinition.load_json(Path(config.operational_array_definition_path))

    full_coefficients: list[FloatArray] = []
    active_index_table: list[IntArray] = []
    selected_beta: list[float] = []
    selected_beam_counts: list[int] = []
    records: list[dict[str, object]] = []

    for frequency_hz in np.asarray(config.frequency_grid_hz, dtype=np.float64).tolist():
        active_indices = array_definition.active_channel_indices_for_frequency(float(frequency_hz))
        active_positions_m = np.asarray(array_definition.positions_m, dtype=np.float64)[active_indices]
        weights, beta, overlap_metrics, peak_margin_db, meets_width_target, meets_peak_margin = _select_fixed_beam_shading_for_frequency(
            active_positions_m=active_positions_m,
            frequency_hz=float(frequency_hz),
            config=config,
        )

        full_weight = np.zeros(array_definition.n_ch, dtype=np.float64)
        full_weight[active_indices] = weights
        full_coefficients.append(full_weight)
        active_index_table.append(active_indices)
        selected_beta.append(float(beta))
        selected_beam_counts.append(int(config.n_beam_az_real))

        active_x_m = active_positions_m[:, 0]
        physical_aperture_m = float(np.max(active_x_m) - np.min(active_x_m)) if active_x_m.size > 1 else 0.0
        equivalent_aperture_m = _equivalent_aperture_m(active_positions_m, weights)
        overlap_margin_deg = float(overlap_metrics["minimum_overlap_margin_deg"])
        target_error_deg = abs(overlap_margin_deg - float(config.target_overlap_margin_deg))

        records.append(
            {
                "frequency_hz": float(frequency_hz),
                "active_channel_count": int(active_indices.size),
                "active_aperture_m": float(physical_aperture_m),
                "equivalent_aperture_m": float(equivalent_aperture_m),
                "selected_kaiser_beta": float(beta),
                "selected_n_beam_az_real": int(config.n_beam_az_real),
                "minimum_three_db_overlap_margin_deg": float(overlap_margin_deg),
                "target_overlap_margin_deg": float(config.target_overlap_margin_deg),
                "three_db_overlap_target_error_deg": float(target_error_deg),
                "minimum_three_db_width_deg": float(overlap_metrics["minimum_three_db_width_deg"]),
                "maximum_three_db_width_deg": float(overlap_metrics["maximum_three_db_width_deg"]),
                "maximum_peak_error_deg": float(overlap_metrics["maximum_peak_error_deg"]),
                "worst_peak_margin_db": float(peak_margin_db),
                "meets_three_db_width_target": bool(meets_width_target),
                "meets_peak_margin": bool(meets_peak_margin),
                "meets_all": bool(meets_width_target and meets_peak_margin),
                "active_weight_min": float(np.min(weights)),
                "active_weight_max": float(np.max(weights)),
                "active_weight_sum": float(np.sum(weights)),
            }
        )

    shading_definition = OperationalShadingDefinition(
        schema_version=1,
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
        operational_array_definition_path=str(Path(config.operational_array_definition_path).resolve()),
        frequency_grid_hz=np.asarray(config.frequency_grid_hz, dtype=np.float64),
        shading_coefficients_by_frequency=np.stack(full_coefficients, axis=0),
        selected_kaiser_beta_by_frequency=np.asarray(selected_beta, dtype=np.float64),
        selected_n_beam_az_real_by_frequency=np.asarray(selected_beam_counts, dtype=np.int64),
        active_channel_indices_by_frequency=tuple(active_index_table),
        records=tuple(records),
        formula={
            "window": "Kaiser-Bessel",
            "design_mode": "fixed_beam_count_three_db_width_match",
            "normalization": "active window max is 1.0; beamforming response must be normalized by sum(weights).",
            "n_beam_az_real": int(config.n_beam_az_real),
            "target_overlap_margin_deg": float(config.target_overlap_margin_deg),
            "target_overlap_tolerance_deg": float(config.target_overlap_tolerance_deg),
            "three_db_down_db": float(config.three_db_down_db),
            "required_peak_margin_db": float(config.required_peak_margin_db),
            "signal_azimuth_count": int(config.signal_azimuth_count),
            "limitation": "Kaiser-Bessel shading can widen the mainlobe but cannot make it narrower than the unshaded aperture response.",
        },
    )
    shading_definition.save_json(config.output_json_path)

    csv_path = config.output_csv_path
    if csv_path is not None:
        output_csv_path = Path(csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    png_path = config.output_summary_png_path
    if png_path is not None:
        _plot_shading_summary(Path(png_path), records, OperationalShadingDesignConfig(
            output_json_path=config.output_json_path,
            operational_array_definition_path=config.operational_array_definition_path,
            output_csv_path=config.output_csv_path,
            output_summary_png_path=config.output_summary_png_path,
            fs_hz=float(config.fs_hz),
            sound_speed_m_s=float(config.sound_speed_m_s),
            frequency_grid_hz=tuple(float(value) for value in config.frequency_grid_hz),
            candidate_kaiser_beta=tuple(float(value) for value in config.candidate_kaiser_beta),
            candidate_n_beam_az_real=(int(config.n_beam_az_real),),
            evaluation_azimuths_deg=tuple(float(value) for value in config.evaluation_azimuths_deg),
            three_db_down_db=float(config.three_db_down_db),
            required_peak_margin_db=float(config.required_peak_margin_db),
            signal_azimuth_count=int(config.signal_azimuth_count),
        ))

    meets_all = bool(all(bool(record["meets_all"]) for record in records))
    summary: dict[str, object] = {
        "operational_array_definition_path": str(Path(config.operational_array_definition_path).resolve()),
        "shading_definition_json_path": str(Path(config.output_json_path).resolve()),
        "shading_design_csv_path": None if csv_path is None else str(Path(csv_path).resolve()),
        "shading_summary_png_path": None if png_path is None else str(Path(png_path).resolve()),
        "frequency_grid_hz": [float(value) for value in config.frequency_grid_hz],
        "selected_kaiser_beta_by_frequency": [float(value) for value in selected_beta],
        "selected_n_beam_az_real_by_frequency": [int(value) for value in selected_beam_counts],
        "minimum_three_db_overlap_margin_deg_by_frequency": [float(record["minimum_three_db_overlap_margin_deg"]) for record in records],
        "three_db_overlap_target_error_deg_by_frequency": [float(record["three_db_overlap_target_error_deg"]) for record in records],
        "minimum_three_db_width_deg_by_frequency": [float(record["minimum_three_db_width_deg"]) for record in records],
        "maximum_three_db_width_deg_by_frequency": [float(record["maximum_three_db_width_deg"]) for record in records],
        "worst_peak_margin_db_by_frequency": [float(record["worst_peak_margin_db"]) for record in records],
        "meets_all": bool(meets_all),
        "records": records,
    }
    return summary

def load_operational_shading(path: str | Path) -> OperationalShadingDefinition:
    """保存済み JSON から運用シェーディング係数を読み込む。"""
    return OperationalShadingDefinition.load_json(path)


__all__ = [
    "OperationalFixedBeamShadingDesignConfig",
    "OperationalShadingDefinition",
    "OperationalShadingDesignConfig",
    "load_operational_shading",
    "run_operational_fixed_beam_shading_design",
    "run_operational_shading_design",
]
