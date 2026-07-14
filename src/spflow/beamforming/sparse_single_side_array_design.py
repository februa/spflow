"""片舷 1 列スパースアレイの幾何設計と評価レポートを作るモジュール。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._validation import (
    require,
    require_non_negative_float,
    require_positive_float,
    require_positive_int,
)
from ..level_conversion import LevelConverter, level_20log10_rms
from .array_design import BandwiseArrayDesign
from .diagnostic_plotting import plot_bl_comparison, require_matplotlib
from .directions import make_directions
from .time_delay import DelayTable

_UNITY_RESPONSE_LEVEL_CONVERTER = LevelConverter.for_definition(
    level_20log10_rms(reference_rms=1.0, reference_label="mainlobe peak")
)


def _format_sector_field_name(prefix: str, azimuth_deg: float) -> str:
    """sector 方位ごとの CSV/JSON キー名を安定に作る。"""
    rounded_azimuth = int(round(float(azimuth_deg)))
    return f"{prefix}_az{rounded_azimuth:03d}_db"


@dataclass(frozen=True)
class SparseSingleSideArrayDesignConfig:
    """片舷 1 列スパースアレイの設計条件を保持する。

    このクラスは、物理配置の最小格子間隔、中央密配置の長さ、外周スパース配置、
    ならびに 0 Hz から 10 kHz までの設計周波数点をまとめて保持する。

    入力は出力先ディレクトリ、音速、サンプリング周波数、設計周波数グリッド、
    中央密配置チャネル数、外周 positive 側 index 列、および
    exact-delay / integer-delay の評価方位群である。
    出力は `run_sparse_single_side_array_design()` が保存する JSON/CSV/PNG 群である。

    小数遅延 FIR 自体の設計、SLC 重み更新、実波形の生成は責務に含めない。
    信号処理上は、固定整相および将来の小数遅延導入を見据えた
    片舷スパースアレイ幾何の事前設計条件に位置づく。
    """

    output_dir: Path
    fs_hz: float = 32768.0
    sound_speed_m_s: float = 1500.0
    dense_spacing_m: float = 0.05
    dense_center_positive_sensor_count: int = 20
    outer_positive_sensor_indices: tuple[int, ...] = (
        22,
        24,
        27,
        31,
        36,
        42,
        49,
        57,
        66,
        76,
        87,
        99,
        112,
        126,
    )
    design_frequency_grid_hz: tuple[float, ...] = (
        0.0,
        512.0,
        1024.0,
        2048.0,
        3072.0,
        4096.0,
        6144.0,
        8192.0,
        10000.0,
    )
    required_peak_margin_db: float = 13.0
    exact_delay_evaluation_azimuths_deg: tuple[float, ...] = (60.0, 90.0, 120.0)
    integer_delay_evaluation_azimuths_deg: tuple[float, ...] = (60.0, 90.0, 120.0)
    n_exact_delay_scan_azimuth: int = 2401
    n_integer_delay_beam_azimuth: int = 151

    def __post_init__(self) -> None:
        """設計条件の単位系と範囲を検証する。"""
        require_positive_float("fs_hz", float(self.fs_hz))
        require_positive_float("sound_speed_m_s", float(self.sound_speed_m_s))
        require_positive_float("dense_spacing_m", float(self.dense_spacing_m))
        require_positive_int("dense_center_positive_sensor_count", int(self.dense_center_positive_sensor_count))
        require_positive_float("required_peak_margin_db", float(self.required_peak_margin_db))
        require_positive_int("n_exact_delay_scan_azimuth", int(self.n_exact_delay_scan_azimuth))
        require_positive_int("n_integer_delay_beam_azimuth", int(self.n_integer_delay_beam_azimuth))

        outer_indices = np.asarray(self.outer_positive_sensor_indices, dtype=np.int64)
        require(
            outer_indices.ndim == 1,
            "outer_positive_sensor_indices must be a 1-D sequence.",
        )
        if outer_indices.size > 0:
            require(
                bool(np.all(np.diff(outer_indices) > 0)),
                "outer_positive_sensor_indices must be strictly increasing.",
            )
            require(
                int(outer_indices[0]) > int(self.dense_center_positive_sensor_count),
                "outer_positive_sensor_indices must start outside the dense center.",
            )

        design_frequency_grid_hz = np.asarray(self.design_frequency_grid_hz, dtype=np.float64)
        require(
            design_frequency_grid_hz.ndim == 1 and design_frequency_grid_hz.size > 0,
            "design_frequency_grid_hz must be a non-empty 1-D sequence.",
        )
        require(
            bool(np.all(np.isfinite(design_frequency_grid_hz))),
            "design_frequency_grid_hz must contain only finite values.",
        )
        require(
            bool(np.all(np.diff(design_frequency_grid_hz) > 0.0)),
            "design_frequency_grid_hz must be strictly increasing.",
        )
        require_non_negative_float("minimum design frequency", float(design_frequency_grid_hz[0]))

        for azimuth_deg in self.exact_delay_evaluation_azimuths_deg:
            require(
                0.0 <= float(azimuth_deg) <= 180.0,
                "exact_delay_evaluation_azimuths_deg must lie in [0, 180].",
            )
        for azimuth_deg in self.integer_delay_evaluation_azimuths_deg:
            require(
                0.0 <= float(azimuth_deg) <= 180.0,
                "integer_delay_evaluation_azimuths_deg must lie in [0, 180].",
            )


@dataclass(frozen=True)
class SparseSingleSideArrayDesignResult:
    """片舷スパースアレイ設計の計算結果を保持する。

    このクラスは、物理センサ配置、周波数ごとの active channel 選択、
    そのときの exact-delay / integer-delay の sector peak margin を一体で保持する。

    入力は `BandwiseArrayDesign`、設計周波数軸、評価方位軸、
    ならびに周波数ごとの sector margin 行列であり、
    出力は active aperture・alias limit・設計表の問い合わせ結果である。

    図の保存や CSV/JSON の書き出しは責務に含めない。
    信号処理上は、片舷アレイ幾何設計の評価結果オブジェクトに位置づく。
    """

    array_design: BandwiseArrayDesign
    design_frequencies_hz: np.ndarray
    exact_delay_evaluation_azimuths_deg: np.ndarray
    integer_delay_evaluation_azimuths_deg: np.ndarray
    exact_delay_sector_peak_margin_db: np.ndarray
    integer_delay_sector_peak_margin_db: np.ndarray

    def __post_init__(self) -> None:
        """評価表の shape と axis の意味を検証する。"""
        design_frequencies_hz = np.asarray(self.design_frequencies_hz, dtype=np.float64)
        exact_delay_evaluation_azimuths_deg = np.asarray(self.exact_delay_evaluation_azimuths_deg, dtype=np.float64)
        integer_delay_evaluation_azimuths_deg = np.asarray(self.integer_delay_evaluation_azimuths_deg, dtype=np.float64)
        exact_delay_sector_peak_margin_db = np.asarray(self.exact_delay_sector_peak_margin_db, dtype=np.float64)
        integer_delay_sector_peak_margin_db = np.asarray(self.integer_delay_sector_peak_margin_db, dtype=np.float64)

        require(
            design_frequencies_hz.ndim == 1,
            "design_frequencies_hz must have shape (n_frequency,).",
        )
        require(
            exact_delay_evaluation_azimuths_deg.ndim == 1,
            "exact_delay_evaluation_azimuths_deg must have shape (n_sector_exact,).",
        )
        require(
            integer_delay_evaluation_azimuths_deg.ndim == 1,
            "integer_delay_evaluation_azimuths_deg must have shape (n_sector_integer,).",
        )
        require(
            exact_delay_sector_peak_margin_db.shape
            == (design_frequencies_hz.size, exact_delay_evaluation_azimuths_deg.size),
            "exact_delay_sector_peak_margin_db must have shape (n_frequency, n_sector_exact).",
        )
        require(
            integer_delay_sector_peak_margin_db.shape
            == (design_frequencies_hz.size, integer_delay_evaluation_azimuths_deg.size),
            "integer_delay_sector_peak_margin_db must have shape (n_frequency, n_sector_integer).",
        )

        object.__setattr__(self, "design_frequencies_hz", design_frequencies_hz)
        object.__setattr__(
            self,
            "exact_delay_evaluation_azimuths_deg",
            exact_delay_evaluation_azimuths_deg,
        )
        object.__setattr__(
            self,
            "integer_delay_evaluation_azimuths_deg",
            integer_delay_evaluation_azimuths_deg,
        )
        object.__setattr__(
            self,
            "exact_delay_sector_peak_margin_db",
            exact_delay_sector_peak_margin_db,
        )
        object.__setattr__(
            self,
            "integer_delay_sector_peak_margin_db",
            integer_delay_sector_peak_margin_db,
        )

    @property
    def channel_positions_m(self) -> np.ndarray:
        """物理センサ座標を返す。

        Returns:
            物理センサ座標。shape は `[n_ch, 3]`、単位は m。
        """
        return self.array_design.channel_positions_m.copy()

    @property
    def n_frequency(self) -> int:
        """設計周波数点数を返す。"""
        return int(self.design_frequencies_hz.size)

    @property
    def n_ch(self) -> int:
        """物理チャネル数を返す。"""
        return int(self.array_design.n_ch)

    def active_channel_indices(self, frequency_index: int) -> np.ndarray:
        """指定周波数点で使用するチャネル index を返す。

        Args:
            frequency_index: 周波数点 index。`design_frequencies_hz` に対応する。

        Returns:
            使用チャネル index。shape は `[n_active_ch]`。
        """
        return self.array_design.active_channel_indices(int(frequency_index))

    def active_channel_count(self, frequency_index: int) -> int:
        """指定周波数点の使用チャネル数を返す。"""
        return int(self.array_design.active_channel_count(int(frequency_index)))

    def active_aperture_m(self, frequency_index: int) -> float:
        """指定周波数点の実効開口長を返す。"""
        return float(self.array_design.active_aperture_m(int(frequency_index)))

    def minimum_spacing_m(self, frequency_index: int) -> float:
        """指定周波数点の active 配置における最小受波器間隔を返す。"""
        return float(self.array_design.minimum_spacing_m(int(frequency_index)))

    def spatial_alias_limit_hz(self, frequency_index: int, sound_speed_m_s: float) -> float:
        """指定周波数点の active 配置に対応する空間 alias 限界周波数を返す。"""
        return float(self.array_design.spatial_alias_limit_hz(int(frequency_index), float(sound_speed_m_s)))

    def exact_delay_worst_sector_peak_margin_db(self, frequency_index: int) -> float:
        """exact-delay 配置評価での sector 最悪ピーク差を返す。"""
        return float(np.min(self.exact_delay_sector_peak_margin_db[int(frequency_index)]))

    def integer_delay_worst_sector_peak_margin_db(self, frequency_index: int) -> float:
        """integer-delay 実装評価での sector 最悪ピーク差を返す。"""
        return float(np.min(self.integer_delay_sector_peak_margin_db[int(frequency_index)]))


def _build_positive_sensor_indices(config: SparseSingleSideArrayDesignConfig) -> np.ndarray:
    """中央密配置と外周配置から positive 側センサ index 列を作る。"""
    dense_indices = np.arange(1, int(config.dense_center_positive_sensor_count) + 1, dtype=np.int64)
    if len(config.outer_positive_sensor_indices) == 0:
        return dense_indices
    return np.concatenate([dense_indices, np.asarray(config.outer_positive_sensor_indices, dtype=np.int64)])


def _build_positions_from_positive_sensor_indices(
    positive_sensor_indices: np.ndarray,
    dense_spacing_m: float,
) -> np.ndarray:
    """positive 側 index 列から中心対称な 1 列アレイ座標を作る。"""
    positive_indices = np.asarray(positive_sensor_indices, dtype=np.int64)

    # sensor_index shape: [n_ch]
    # 中心 0 を挟んで左右対称に並べることで、片舷走査で使う 1 列アレイの物理配置を表す。
    sensor_index = np.concatenate([-positive_indices[::-1], np.array([0], dtype=np.int64), positive_indices])
    positions = np.zeros((sensor_index.size, 3), dtype=np.float64)
    positions[:, 0] = sensor_index.astype(np.float64) * float(dense_spacing_m)
    return positions


def _sector_direction_cosines(scan_axis_azimuth_deg: np.ndarray) -> np.ndarray:
    """方位角列から x-y 平面上の方向余弦を返す。"""
    azimuth_rad = np.deg2rad(np.asarray(scan_axis_azimuth_deg, dtype=np.float64))
    return np.stack(
        [
            np.cos(azimuth_rad),
            np.sin(azimuth_rad),
            np.zeros_like(azimuth_rad),
        ],
        axis=1,
    )


def _measure_mainlobe_peak_margin_db(
    axis_azimuth_deg: np.ndarray,
    beam_levels_db20: np.ndarray,
    target_azimuth_deg: float,
) -> float:
    """target 近傍 peak とその mainlobe 外最大 peak の差を返す。"""
    nearest_index = int(np.argmin(np.abs(axis_azimuth_deg - float(target_azimuth_deg))))
    search_start = max(0, nearest_index - 12)
    search_stop = min(axis_azimuth_deg.size, nearest_index + 13)
    peak_index = int(search_start + np.argmax(beam_levels_db20[search_start:search_stop]))
    left_index = peak_index
    right_index = peak_index

    # peak から左右へ単調減少している間を mainlobe と見なし、
    # その外側の最大 peak を sidelobe peak としてピーク差を定義する。
    while left_index > 0 and float(beam_levels_db20[left_index - 1]) <= float(beam_levels_db20[left_index]):
        left_index -= 1
    while right_index < beam_levels_db20.size - 1 and float(beam_levels_db20[right_index + 1]) <= float(beam_levels_db20[right_index]):
        right_index += 1

    outside_mask = np.ones(beam_levels_db20.size, dtype=bool)
    outside_mask[left_index : right_index + 1] = False
    if not np.any(outside_mask):
        return float("inf")

    outside_peak_level_db20 = float(np.max(beam_levels_db20[outside_mask]))
    return float(beam_levels_db20[peak_index] - outside_peak_level_db20)


def _exact_delay_beam_levels_db20(
    positions_m: np.ndarray,
    frequency_hz: float,
    target_azimuth_deg: float,
    sound_speed_m_s: float,
    scan_axis_azimuth_deg: np.ndarray,
) -> np.ndarray:
    """幾何だけで決まる exact-delay BL を評価する。

    Args:
        positions_m: active センサ座標。shape は `[n_active_ch, 3]`、単位は m。
        frequency_hz: 評価周波数。単位は Hz。
        target_azimuth_deg: 到来方位。単位は deg。
        sound_speed_m_s: 音速。単位は m/s。
        scan_axis_azimuth_deg: BL を描く走査方位軸。shape は `[n_scan]`、単位は deg。

    Returns:
        exact-delay の BL。shape は `[n_scan]`、単位は dB20。
    """
    x_positions_m = np.asarray(positions_m, dtype=np.float64)[:, 0]
    scan_directions = _sector_direction_cosines(scan_axis_azimuth_deg)
    target_direction = _sector_direction_cosines(np.array([float(target_azimuth_deg)], dtype=np.float64))[0]

    # exact steering では、target と scan の方向余弦差 Δu に対して
    # exp{-j 2π f x Δu / c} の位相ずれだけが残る。
    # x_positions_m[:, None] shape: [n_active_ch, 1]
    # direction_delta[None, :] shape: [1, n_scan]
    # broadcasting により全チャネル・全 scan 方位の array factor を一括評価する。
    direction_delta = float(target_direction[0]) - scan_directions[:, 0]
    phase = (
        -2.0
        * np.pi
        * float(frequency_hz)
        * x_positions_m[:, np.newaxis]
        * direction_delta[np.newaxis, :]
        / float(sound_speed_m_s)
    )
    beam_magnitude = np.abs(np.mean(np.exp(1j * phase), axis=0))
    return np.asarray(
        _UNITY_RESPONSE_LEVEL_CONVERTER.output_rms_to_level(
            beam_magnitude,
            floor_db=_UNITY_RESPONSE_LEVEL_CONVERTER.float64_tiny_level_db,
        ),
        dtype=np.float64,
    )


def _integer_delay_beam_levels_db20(
    positions_m: np.ndarray,
    frequency_hz: float,
    target_azimuth_deg: float,
    config: SparseSingleSideArrayDesignConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """現在の整数遅延固定整相実装に対応する BL を解析式で評価する。

    Args:
        positions_m: active センサ座標。shape は `[n_active_ch, 3]`、単位は m。
        frequency_hz: 評価周波数。単位は Hz。
        target_azimuth_deg: 到来方位。単位は deg。
        config: サンプリング周波数、音速、ビーム本数を含む設計条件。

    Returns:
        `(axis_azimuth_deg, beam_levels_db20)` を返す。
        `axis_azimuth_deg` の shape は `[n_beam]`、単位は deg。
        `beam_levels_db20` の shape は `[n_beam]`、単位は dB20。
    """
    directions, axis_azimuth_deg, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=int(config.n_integer_delay_beam_azimuth),
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    delay_table = DelayTable.from_geometry(
        array_pos_m=np.asarray(positions_m, dtype=np.float64),
        dir_cos=np.asarray(directions.T, dtype=np.float64),
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
    )
    target_direction = _sector_direction_cosines(np.array([float(target_azimuth_deg)], dtype=np.float64))[0]

    # target_arrival_delay_sec[ch] = -(r_ch^T u_target) / c。
    # 現在の整数遅延実装では、各 scan beam の causal integer delay をそのまま使うため、
    # exact-delay 幾何性能とは別に fs 依存の位相量子化誤差が残る。
    target_arrival_delay_sec = -(np.asarray(positions_m, dtype=np.float64) @ target_direction) / float(config.sound_speed_m_s)
    phase = -2.0 * np.pi * float(frequency_hz) * (
        target_arrival_delay_sec[:, np.newaxis] + delay_table.delay_int / float(config.fs_hz)
    )
    beam_magnitude = np.abs(np.mean(np.exp(1j * phase), axis=0))
    beam_levels_db20 = _UNITY_RESPONSE_LEVEL_CONVERTER.output_rms_to_level(
        beam_magnitude,
        floor_db=_UNITY_RESPONSE_LEVEL_CONVERTER.float64_tiny_level_db,
    )
    return axis_azimuth_deg.astype(np.float64), beam_levels_db20.astype(np.float64)


def _evaluate_sector_peak_margins_exact_delay(
    positions_m: np.ndarray,
    frequency_hz: float,
    target_azimuths_deg: np.ndarray,
    sound_speed_m_s: float,
    scan_axis_azimuth_deg: np.ndarray,
) -> np.ndarray:
    """exact-delay で各 target 方位の peak margin を返す。"""
    margins_db = []
    for target_azimuth_deg in np.asarray(target_azimuths_deg, dtype=np.float64):
        beam_levels_db20 = _exact_delay_beam_levels_db20(
            positions_m=positions_m,
            frequency_hz=float(frequency_hz),
            target_azimuth_deg=float(target_azimuth_deg),
            sound_speed_m_s=float(sound_speed_m_s),
            scan_axis_azimuth_deg=scan_axis_azimuth_deg,
        )
        margins_db.append(
            _measure_mainlobe_peak_margin_db(
                axis_azimuth_deg=scan_axis_azimuth_deg,
                beam_levels_db20=beam_levels_db20,
                target_azimuth_deg=float(target_azimuth_deg),
            )
        )
    return np.asarray(margins_db, dtype=np.float64)


def _evaluate_sector_peak_margins_integer_delay(
    positions_m: np.ndarray,
    frequency_hz: float,
    target_azimuths_deg: np.ndarray,
    config: SparseSingleSideArrayDesignConfig,
) -> np.ndarray:
    """integer-delay 実装で各 target 方位の peak margin を返す。"""
    margins_db = []
    for target_azimuth_deg in np.asarray(target_azimuths_deg, dtype=np.float64):
        axis_azimuth_deg, beam_levels_db20 = _integer_delay_beam_levels_db20(
            positions_m=positions_m,
            frequency_hz=float(frequency_hz),
            target_azimuth_deg=float(target_azimuth_deg),
            config=config,
        )
        margins_db.append(
            _measure_mainlobe_peak_margin_db(
                axis_azimuth_deg=axis_azimuth_deg,
                beam_levels_db20=beam_levels_db20,
                target_azimuth_deg=float(target_azimuth_deg),
            )
        )
    return np.asarray(margins_db, dtype=np.float64)


def build_sparse_single_side_array_design(
    config: SparseSingleSideArrayDesignConfig,
) -> SparseSingleSideArrayDesignResult:
    """片舷スパースアレイ設計を計算し、周波数ごとの active 配置を返す。

    Args:
        config: 物理配置、設計周波数、評価方位を含む設計条件。

    Returns:
        物理配置と周波数ごとの exact-delay / integer-delay 評価結果を持つ設計結果。
    """
    positive_sensor_indices = _build_positive_sensor_indices(config)
    positions_m = _build_positions_from_positive_sensor_indices(
        positive_sensor_indices=positive_sensor_indices,
        dense_spacing_m=float(config.dense_spacing_m),
    )
    design_frequencies_hz = np.asarray(config.design_frequency_grid_hz, dtype=np.float64)
    exact_delay_evaluation_azimuths_deg = np.asarray(config.exact_delay_evaluation_azimuths_deg, dtype=np.float64)
    integer_delay_evaluation_azimuths_deg = np.asarray(config.integer_delay_evaluation_azimuths_deg, dtype=np.float64)
    exact_delay_scan_axis_azimuth_deg = np.linspace(
        0.0,
        180.0,
        int(config.n_exact_delay_scan_azimuth),
        dtype=np.float64,
    )

    active_indices_per_frequency: list[np.ndarray] = []
    exact_delay_sector_peak_margin_db = np.zeros(
        (design_frequencies_hz.size, exact_delay_evaluation_azimuths_deg.size),
        dtype=np.float64,
    )
    integer_delay_sector_peak_margin_db = np.zeros(
        (design_frequencies_hz.size, integer_delay_evaluation_azimuths_deg.size),
        dtype=np.float64,
    )

    total_positive_sensor_count = int(positive_sensor_indices.size)
    for frequency_index, frequency_hz in enumerate(design_frequencies_hz.tolist()):
        if float(frequency_hz) <= 0.0:
            best_positive_sensor_count = total_positive_sensor_count
        else:
            best_positive_sensor_count = int(config.dense_center_positive_sensor_count)
            for positive_sensor_count in range(
                int(config.dense_center_positive_sensor_count),
                total_positive_sensor_count + 1,
            ):
                candidate_positions_m = _build_positions_from_positive_sensor_indices(
                    positive_sensor_indices=positive_sensor_indices[:positive_sensor_count],
                    dense_spacing_m=float(config.dense_spacing_m),
                )
                candidate_sector_peak_margin_db = _evaluate_sector_peak_margins_exact_delay(
                    positions_m=candidate_positions_m,
                    frequency_hz=float(frequency_hz),
                    target_azimuths_deg=exact_delay_evaluation_azimuths_deg,
                    sound_speed_m_s=float(config.sound_speed_m_s),
                    scan_axis_azimuth_deg=exact_delay_scan_axis_azimuth_deg,
                )
                if np.min(candidate_sector_peak_margin_db) >= float(config.required_peak_margin_db):
                    best_positive_sensor_count = int(positive_sensor_count)

        active_positive_sensor_indices = positive_sensor_indices[:best_positive_sensor_count]
        active_positions_m = _build_positions_from_positive_sensor_indices(
            positive_sensor_indices=active_positive_sensor_indices,
            dense_spacing_m=float(config.dense_spacing_m),
        )

        # 中心対称 prefix 選択なので、active index は物理チャネル列の中央から外側へ等しく増える。
        active_channel_count = int(active_positions_m.shape[0])
        center_channel_index = int(positions_m.shape[0] // 2)
        active_half_channel_count = int((active_channel_count - 1) // 2)
        active_indices_per_frequency.append(
            np.arange(
                center_channel_index - active_half_channel_count,
                center_channel_index + active_half_channel_count + 1,
                dtype=np.int64,
            )
        )

        if float(frequency_hz) <= 0.0:
            exact_delay_sector_peak_margin_db[frequency_index, :] = np.inf
            integer_delay_sector_peak_margin_db[frequency_index, :] = np.inf
            continue

        exact_delay_sector_peak_margin_db[frequency_index, :] = _evaluate_sector_peak_margins_exact_delay(
            positions_m=active_positions_m,
            frequency_hz=float(frequency_hz),
            target_azimuths_deg=exact_delay_evaluation_azimuths_deg,
            sound_speed_m_s=float(config.sound_speed_m_s),
            scan_axis_azimuth_deg=exact_delay_scan_axis_azimuth_deg,
        )
        integer_delay_sector_peak_margin_db[frequency_index, :] = _evaluate_sector_peak_margins_integer_delay(
            positions_m=active_positions_m,
            frequency_hz=float(frequency_hz),
            target_azimuths_deg=integer_delay_evaluation_azimuths_deg,
            config=config,
        )

    array_design = BandwiseArrayDesign.from_channel_positions_and_active_indices(
        channel_positions_m=positions_m,
        n_band=int(design_frequencies_hz.size),
        active_indices_per_band=active_indices_per_frequency,
    )
    return SparseSingleSideArrayDesignResult(
        array_design=array_design,
        design_frequencies_hz=design_frequencies_hz,
        exact_delay_evaluation_azimuths_deg=exact_delay_evaluation_azimuths_deg,
        integer_delay_evaluation_azimuths_deg=integer_delay_evaluation_azimuths_deg,
        exact_delay_sector_peak_margin_db=exact_delay_sector_peak_margin_db,
        integer_delay_sector_peak_margin_db=integer_delay_sector_peak_margin_db,
    )


def _plot_array_geometry(output_path: Path, result: SparseSingleSideArrayDesignResult) -> None:
    """物理センサ配置を x 座標で可視化する。"""
    require_matplotlib()
    import matplotlib.pyplot as plt

    positions_m = result.channel_positions_m
    sensor_index = np.arange(positions_m.shape[0], dtype=np.int64)

    fig, axis = plt.subplots(figsize=(10, 3.5))
    axis.scatter(positions_m[:, 0], np.zeros_like(sensor_index), color="tab:blue", s=32)
    for channel_index, x_position_m in enumerate(positions_m[:, 0].tolist()):
        axis.text(float(x_position_m), 0.02, str(channel_index), rotation=90, ha="center", va="bottom", fontsize=7)
    axis.set_xlabel("Sensor Position x [m]")
    axis.set_yticks([])
    axis.set_title("Sparse single-side array geometry")
    axis.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_effective_aperture_summary(
    output_path: Path,
    result: SparseSingleSideArrayDesignResult,
) -> None:
    """周波数に対する active aperture と active channel 数を保存する。"""
    require_matplotlib()
    import matplotlib.pyplot as plt

    design_frequencies_hz = result.design_frequencies_hz[1:]
    active_aperture_m = np.array(
        [result.active_aperture_m(frequency_index) for frequency_index in range(1, result.n_frequency)],
        dtype=np.float64,
    )
    active_channel_count = np.array(
        [result.active_channel_count(frequency_index) for frequency_index in range(1, result.n_frequency)],
        dtype=np.int64,
    )

    fig, axis_left = plt.subplots(figsize=(9.5, 4.5))
    axis_left.plot(design_frequencies_hz, active_aperture_m, marker="o", linewidth=1.5, color="tab:blue")
    axis_left.set_xlabel("Frequency [Hz]")
    axis_left.set_ylabel("Effective Aperture [m]", color="tab:blue")
    axis_left.tick_params(axis="y", labelcolor="tab:blue")
    axis_left.grid(True, alpha=0.3)

    axis_right = axis_left.twinx()
    axis_right.plot(design_frequencies_hz, active_channel_count, marker="s", linewidth=1.5, color="tab:orange")
    axis_right.set_ylabel("Active Channel Count", color="tab:orange")
    axis_right.tick_params(axis="y", labelcolor="tab:orange")

    axis_left.set_title("Effective aperture and active channels vs frequency")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sector_margin_summary(
    output_path: Path,
    result: SparseSingleSideArrayDesignResult,
) -> None:
    """exact-delay と integer-delay の sector worst margin を比較保存する。"""
    require_matplotlib()
    import matplotlib.pyplot as plt

    design_frequencies_hz = result.design_frequencies_hz[1:]
    exact_delay_worst_margin_db = np.array(
        [result.exact_delay_worst_sector_peak_margin_db(frequency_index) for frequency_index in range(1, result.n_frequency)],
        dtype=np.float64,
    )
    integer_delay_worst_margin_db = np.array(
        [result.integer_delay_worst_sector_peak_margin_db(frequency_index) for frequency_index in range(1, result.n_frequency)],
        dtype=np.float64,
    )

    fig, axis = plt.subplots(figsize=(9.5, 4.5))
    axis.plot(
        design_frequencies_hz,
        exact_delay_worst_margin_db,
        marker="o",
        linewidth=1.5,
        color="tab:blue",
        label="Exact delay geometry margin",
    )
    axis.plot(
        design_frequencies_hz,
        integer_delay_worst_margin_db,
        marker="s",
        linewidth=1.5,
        color="tab:red",
        label="Integer delay implementation margin",
    )
    axis.axhline(13.0, color="black", linestyle=":", linewidth=1.0, label="Required margin 13 dB")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Worst Sector Peak Margin [dB]")
    axis.set_title("Sector worst peak margin vs frequency")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_bl_comparison_examples(
    output_dir: Path,
    result: SparseSingleSideArrayDesignResult,
    config: SparseSingleSideArrayDesignConfig,
) -> list[str]:
    """設計意図確認用に exact-delay / integer-delay の BL 重ね書きを保存する。"""
    saved_paths: list[str] = []

    comparison_specs = (
        (10000.0, 90.0),
        (10000.0, 60.0),
    )
    _, comparison_axis_azimuth_deg, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=int(config.n_integer_delay_beam_azimuth),
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    axis_azimuth_deg = np.asarray(comparison_axis_azimuth_deg, dtype=np.float64)

    for frequency_hz, target_azimuth_deg in comparison_specs:
        frequency_index = int(np.argmin(np.abs(result.design_frequencies_hz - float(frequency_hz))))
        positions_m = result.array_design.active_positions(frequency_index)
        integer_axis_azimuth_deg, integer_delay_levels_db20 = _integer_delay_beam_levels_db20(
            positions_m=positions_m,
            frequency_hz=float(frequency_hz),
            target_azimuth_deg=float(target_azimuth_deg),
            config=config,
        )
        if not np.allclose(axis_azimuth_deg, integer_axis_azimuth_deg, atol=1e-5):
            axis_azimuth_deg = integer_axis_azimuth_deg.astype(np.float64)
        exact_delay_levels_db20 = _exact_delay_beam_levels_db20(
            positions_m=positions_m,
            frequency_hz=float(frequency_hz),
            target_azimuth_deg=float(target_azimuth_deg),
            sound_speed_m_s=float(config.sound_speed_m_s),
            scan_axis_azimuth_deg=axis_azimuth_deg,
        )
        exact_peak_azimuth_deg = float(axis_azimuth_deg[int(np.argmax(exact_delay_levels_db20))])
        integer_peak_azimuth_deg = float(axis_azimuth_deg[int(np.argmax(integer_delay_levels_db20))])

        output_path = output_dir / f"bl_compare_{int(round(float(frequency_hz))):05d}Hz_{int(round(float(target_azimuth_deg))):03d}deg.png"
        plot_bl_comparison(
            axis_az_deg=axis_azimuth_deg,
            before_levels_db20=exact_delay_levels_db20,
            after_levels_db20=integer_delay_levels_db20,
            target_azimuth_deg=float(target_azimuth_deg),
            before_peak_azimuth_deg=exact_peak_azimuth_deg,
            after_peak_azimuth_deg=integer_peak_azimuth_deg,
            title=f"BL comparison ({float(frequency_hz):.0f} Hz, target {float(target_azimuth_deg):.0f} deg)",
            caption=(
                "blue=exact delay geometry, orange=integer delay implementation. "
                "高域 off-broadside の差は主に delay 量子化誤差に由来する。"
            ),
            output_path=output_path,
            before_label="Exact delay geometry",
            after_label="Integer delay implementation",
        )
        saved_paths.append(str(output_path.resolve()))

    return saved_paths


def _design_notes_markdown() -> str:
    """設計レポートと一緒に保存する注意事項を返す。"""
    return "\n".join(
        [
            "# Sparse Single-Side Array Design Notes",
            "",
            "- 0 Hz では波長が無限大となるため、方位分解能や aperture wavelength 数は定義しない。",
            "- exact-delay geometry margin は、アレイ幾何そのものが作る array factor を評価した値である。",
            "- integer-delay implementation margin は、現在の fs=32768 Hz 整数遅延固定整相に起因する delay 量子化誤差を含む。",
            "- したがって 8 kHz から 10 kHz の off-broadside 劣化は、今回の物理アレイ設計よりも小数遅延未導入の影響が支配的である。",
            "- 将来の小数遅延 FIR 導入時は、この設計で示した exact-delay geometry margin を再現目標とする。",
            "",
        ]
    )



def _safe_optional_finite_float(value: float | None) -> float | None:
    """JSON/CSV へ保存しやすいよう非有限値を `None` へ正規化する。"""
    if value is None:
        return None
    numeric_value = float(value)
    return numeric_value if np.isfinite(numeric_value) else None
def _record_rows(
    result: SparseSingleSideArrayDesignResult,
    config: SparseSingleSideArrayDesignConfig,
) -> list[dict[str, object]]:
    """JSON/CSV 保存用の周波数別設計表を組み立てる。"""
    records: list[dict[str, object]] = []

    for frequency_index, frequency_hz in enumerate(result.design_frequencies_hz.tolist()):
        active_aperture_m = result.active_aperture_m(frequency_index)
        minimum_spacing_m = result.minimum_spacing_m(frequency_index)
        active_channel_count = result.active_channel_count(frequency_index)
        spatial_alias_limit_hz = result.spatial_alias_limit_hz(
            frequency_index=frequency_index,
            sound_speed_m_s=float(config.sound_speed_m_s),
        )
        if float(frequency_hz) <= 0.0:
            active_aperture_wavelengths = None
            estimated_broadside_hpbw_deg = None
        else:
            wavelength_m = float(config.sound_speed_m_s) / float(frequency_hz)
            active_aperture_wavelengths = float(active_aperture_m / wavelength_m) if wavelength_m > 0.0 else None

            # HPBW ≈ 0.886 λ / D [rad] を broadside の目安として記録する。
            # これは exact 値ではないが、低周波から高周波へ行くと開口/波長比がどう変わるかを
            # ひと目で把握するための設計メモとして使う。
            if active_aperture_m > 0.0:
                estimated_broadside_hpbw_deg = float(
                    np.rad2deg(0.886 * wavelength_m / active_aperture_m)
                )
            else:
                estimated_broadside_hpbw_deg = None

        record: dict[str, object] = {
            "frequency_hz": float(frequency_hz),
            "active_channel_count": int(active_channel_count),
            "active_aperture_m": float(active_aperture_m),
            "minimum_spacing_m": float(minimum_spacing_m),
            "spatial_alias_limit_hz": float(spatial_alias_limit_hz),
            "active_aperture_wavelengths": active_aperture_wavelengths,
            "estimated_broadside_hpbw_deg": estimated_broadside_hpbw_deg,
            "exact_delay_worst_sector_peak_margin_db": (
                None if float(frequency_hz) <= 0.0 else _safe_optional_finite_float(result.exact_delay_worst_sector_peak_margin_db(frequency_index))
            ),
            "integer_delay_worst_sector_peak_margin_db": (
                None if float(frequency_hz) <= 0.0 else _safe_optional_finite_float(result.integer_delay_worst_sector_peak_margin_db(frequency_index))
            ),
            "exact_delay_meets_required_peak_margin": (
                None
                if float(frequency_hz) <= 0.0
                else bool(result.exact_delay_worst_sector_peak_margin_db(frequency_index) >= float(config.required_peak_margin_db))
            ),
            "integer_delay_meets_required_peak_margin": (
                None
                if float(frequency_hz) <= 0.0
                else bool(result.integer_delay_worst_sector_peak_margin_db(frequency_index) >= float(config.required_peak_margin_db))
            ),
        }

        for sector_index, azimuth_deg in enumerate(result.exact_delay_evaluation_azimuths_deg.tolist()):
            record[_format_sector_field_name("exact_delay_margin", azimuth_deg)] = (
                None if float(frequency_hz) <= 0.0 else _safe_optional_finite_float(result.exact_delay_sector_peak_margin_db[frequency_index, sector_index])
            )
        for sector_index, azimuth_deg in enumerate(result.integer_delay_evaluation_azimuths_deg.tolist()):
            record[_format_sector_field_name("integer_delay_margin", azimuth_deg)] = (
                None if float(frequency_hz) <= 0.0 else _safe_optional_finite_float(result.integer_delay_sector_peak_margin_db[frequency_index, sector_index])
            )
        records.append(record)

    return records


def _write_record_csv(path: Path, records: list[dict[str, object]]) -> None:
    """設計表を CSV として保存する。"""
    fieldnames = list(records[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def run_sparse_single_side_array_design(
    config: SparseSingleSideArrayDesignConfig,
) -> dict[str, Any]:
    """片舷スパースアレイを設計し、設計表と評価図を保存する。

    Args:
        config: 物理配置、周波数軸、評価条件、保存先を含む設計条件。

    Returns:
        設計表、保存ファイルパス、物理配置要約を含む summary 辞書。
    """
    require_matplotlib()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = build_sparse_single_side_array_design(config)
    records = _record_rows(result=result, config=config)

    json_path = output_dir / "design_summary.json"
    csv_path = output_dir / "design_table.csv"
    geometry_png_path = output_dir / "array_geometry.png"
    aperture_png_path = output_dir / "effective_aperture_summary.png"
    margin_png_path = output_dir / "sector_margin_summary.png"
    notes_path = output_dir / "design_notes.md"
    bl_comparison_paths = _save_bl_comparison_examples(output_dir=output_dir, result=result, config=config)

    _plot_array_geometry(geometry_png_path, result)
    _plot_effective_aperture_summary(aperture_png_path, result)
    _plot_sector_margin_summary(margin_png_path, result)
    _write_record_csv(csv_path, records)
    notes_path.write_text(_design_notes_markdown(), encoding="utf-8")

    physical_sensor_positions_x = np.sort(result.channel_positions_m[:, 0])
    physical_sensor_spacings_m = np.diff(physical_sensor_positions_x)
    summary: dict[str, object] = {
        "fs_hz": float(config.fs_hz),
        "sound_speed_m_s": float(config.sound_speed_m_s),
        "dense_spacing_m": float(config.dense_spacing_m),
        "dense_center_positive_sensor_count": int(config.dense_center_positive_sensor_count),
        "dense_center_channel_count": int(2 * int(config.dense_center_positive_sensor_count) + 1),
        "outer_positive_sensor_indices": [int(index) for index in config.outer_positive_sensor_indices],
        "design_frequency_grid_hz": [float(frequency_hz) for frequency_hz in result.design_frequencies_hz.tolist()],
        "required_peak_margin_db": float(config.required_peak_margin_db),
        "exact_delay_evaluation_azimuths_deg": [
            float(azimuth_deg) for azimuth_deg in result.exact_delay_evaluation_azimuths_deg.tolist()
        ],
        "integer_delay_evaluation_azimuths_deg": [
            float(azimuth_deg) for azimuth_deg in result.integer_delay_evaluation_azimuths_deg.tolist()
        ],
        "array_n_ch": int(result.n_ch),
        "array_aperture_m": float(physical_sensor_positions_x[-1] - physical_sensor_positions_x[0]),
        "array_min_sensor_spacing_m": float(np.min(physical_sensor_spacings_m)),
        "array_max_sensor_spacing_m": float(np.max(physical_sensor_spacings_m)),
        "design_summary_json_path": str(json_path.resolve()),
        "design_table_csv_path": str(csv_path.resolve()),
        "array_geometry_png_path": str(geometry_png_path.resolve()),
        "effective_aperture_summary_png_path": str(aperture_png_path.resolve()),
        "sector_margin_summary_png_path": str(margin_png_path.resolve()),
        "design_notes_path": str(notes_path.resolve()),
        "bl_comparison_png_paths": bl_comparison_paths,
        "records": records,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = [
    "SparseSingleSideArrayDesignConfig",
    "SparseSingleSideArrayDesignResult",
    "build_sparse_single_side_array_design",
    "run_sparse_single_side_array_design",
]


