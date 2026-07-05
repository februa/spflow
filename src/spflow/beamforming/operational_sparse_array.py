"""運用で使用するスパースアレイ定義ファイルを設計・保存・読込するモジュール。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float, require_positive_int
from .array_design import BandwiseArrayDesign


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


def _direction_cosines_from_azimuth_deg(azimuth_deg: FloatArray) -> FloatArray:
    """水平面方位を方向余弦へ変換する。

    Args:
        azimuth_deg: 方位角。shape は `[n_azimuth]`、単位は deg。

    Returns:
        方向余弦。shape は `[n_azimuth, 3]`。
        axis=0 は方位点、axis=1 は x/y/z 成分である。
    """
    azimuth_rad = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))

    # 片舷 1 列アレイでは x 軸方向の方向余弦 cos(theta) が位相差を支配する。
    # y/z 成分も shape 契約を固定するために保持し、固定整相の array_pos [n_ch, 3] と揃える。
    return np.stack(
        [np.cos(azimuth_rad), np.sin(azimuth_rad), np.zeros_like(azimuth_rad)],
        axis=1,
    )


def _beam_levels_db20(
    positions_m: FloatArray,
    frequency_hz: float,
    target_azimuth_deg: float,
    scan_azimuths_deg: FloatArray,
    sound_speed_m_s: float,
) -> FloatArray:
    """exact-delay のビーム応答を dB20 で返す。

    Args:
        positions_m: active センサ位置。shape は `[n_active_ch, 3]`、単位は m。
        frequency_hz: 評価周波数。単位は Hz。
        target_azimuth_deg: 到来方位。単位は deg。
        scan_azimuths_deg: 走査方位。shape は `[n_scan]`、単位は deg。
        sound_speed_m_s: 音速。単位は m/s。

    Returns:
        正規化ビーム応答。shape は `[n_scan]`、単位は dB20。

    Raises:
        ValueError: 配列 shape、周波数、音速が不正な場合。
    """
    sensor_positions_m = np.asarray(positions_m, dtype=np.float64)
    scan_axis_deg = np.asarray(scan_azimuths_deg, dtype=np.float64)
    require(
        sensor_positions_m.ndim == 2 and sensor_positions_m.shape[1] == 3,
        "positions_m must have shape (n_active_ch, 3).",
    )
    require(sensor_positions_m.shape[0] > 0, "positions_m must contain at least one sensor.")
    require(scan_axis_deg.ndim == 1 and scan_axis_deg.size > 0, "scan_azimuths_deg must be a non-empty 1-D array.")
    require_positive_float("frequency_hz", float(frequency_hz))
    require_positive_float("sound_speed_m_s", float(sound_speed_m_s))

    scan_directions = _direction_cosines_from_azimuth_deg(scan_axis_deg)
    target_direction = _direction_cosines_from_azimuth_deg(np.array([float(target_azimuth_deg)], dtype=np.float64))[0]

    # Δtau[ch, scan] = x_ch * (u_target - u_scan) / c。
    # exact-delay 固定整相後に残る位相差は exp(-j 2π f Δtau) であり、
    # axis=0 の ch 平均が conventional delay-and-sum の array factor に対応する。
    direction_delta = float(target_direction[0]) - scan_directions[:, 0]
    phase = (
        -2.0
        * np.pi
        * float(frequency_hz)
        * sensor_positions_m[:, 0][:, np.newaxis]
        * direction_delta[np.newaxis, :]
        / float(sound_speed_m_s)
    )
    response = np.abs(np.mean(np.exp(1j * phase), axis=0))
    return 20.0 * np.log10(np.maximum(response, np.finfo(np.float64).tiny))


def _mainlobe_peak_margin_db(
    scan_azimuths_deg: FloatArray,
    beam_levels_db20: FloatArray,
    target_azimuth_deg: float,
) -> float:
    """target 近傍 peak と mainlobe 外 peak の差を返す。

    Args:
        scan_azimuths_deg: 走査方位。shape は `[n_scan]`、単位は deg。
        beam_levels_db20: ビーム応答。shape は `[n_scan]`、単位は dB20。
        target_azimuth_deg: 評価 target 方位。単位は deg。

    Returns:
        mainlobe peak と mainlobe 外最大 peak の差。単位は dB。
    """
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

    # mainlobe は peak から左右へ単調に落ちる谷までと定義する。
    # 疎配置では sidelobe が高くなりやすいため、固定 guard 幅ではなく応答形状から外側 peak を測る。
    while left_index > 0 and float(levels_db20[left_index - 1]) <= float(levels_db20[left_index]):
        left_index -= 1
    while right_index < levels_db20.size - 1 and float(levels_db20[right_index + 1]) <= float(levels_db20[right_index]):
        right_index += 1

    outside_mask = np.ones(levels_db20.size, dtype=bool)
    outside_mask[left_index : right_index + 1] = False
    if not bool(np.any(outside_mask)):
        return float("inf")

    outside_peak_db20 = float(np.max(levels_db20[outside_mask]))
    return float(levels_db20[peak_index] - outside_peak_db20)


@dataclass(frozen=True)
class OperationalSparseArrayDesignConfig:
    """運用スパースアレイ定義ファイルの設計条件を保持する。

    このクラスは、fs=32768 Hz で 0 Hz から 10 kHz まで確認するための物理配置、
    性能保証下限周波数、ビーム幅目標、sidelobe peak margin 条件、保存先を保持する。

    入力は設計式に使う音速、最大周波数、性能保証下限、目標 HPBW、positive 側の
    段階疎配置セグメントである。出力は `design_operational_sparse_array()` が返す
    `OperationalSparseArrayDefinition` である。

    固定整相処理、小数遅延 FIR 設計、SLC 係数更新は責務に含めない。
    信号処理上は、固定整相 + SLC が参照するアレイ幾何ファイルの事前設計条件に位置づく。
    """

    output_json_path: Path
    output_csv_path: Path | None = None
    fs_hz: float = 32768.0
    sound_speed_m_s: float = 1500.0
    maximum_frequency_hz: float = 10000.0
    valid_frequency_hz_min: float = 200.0
    constant_beamwidth_reference_frequency_hz: float = 200.0
    target_hpbw_deg: float | None = None
    aperture_safety_factors: tuple[float, ...] = (1.00, 1.04, 1.08, 1.12, 1.16, 1.20)
    required_peak_margin_db: float = 13.0
    evaluation_azimuths_deg: tuple[float, ...] = (60.0, 90.0, 120.0)
    design_frequency_grid_hz: tuple[float, ...] = (
        0.0,
        10.0,
        32.0,
        64.0,
        128.0,
        200.0,
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
    positive_layout_segments_m: tuple[tuple[float, float, float], ...] = (
        (0.05, 1.00, 0.05),
        (1.20, 4.80, 0.20),
        (5.30, 12.30, 0.50),
        (13.30, 30.30, 1.00),
        (31.80, 150.30, 1.50),
    )
    scan_azimuth_count: int = 1801
    uniform_subset_segment_counts: tuple[int, ...] = (14, 16, 18, 20, 22, 24)
    gap_alias_safety: float = 0.82

    def __post_init__(self) -> None:
        """設計条件の範囲、単位、周波数軸の単調性を検証する。"""
        require_positive_float("fs_hz", float(self.fs_hz))
        require_positive_float("sound_speed_m_s", float(self.sound_speed_m_s))
        require_positive_float("maximum_frequency_hz", float(self.maximum_frequency_hz))
        require_positive_float("valid_frequency_hz_min", float(self.valid_frequency_hz_min))
        require(float(self.valid_frequency_hz_min) <= float(self.maximum_frequency_hz), "valid_frequency_hz_min must not exceed maximum_frequency_hz.")
        require_positive_float("constant_beamwidth_reference_frequency_hz", float(self.constant_beamwidth_reference_frequency_hz))
        require(
            float(self.valid_frequency_hz_min) == float(self.constant_beamwidth_reference_frequency_hz),
            "valid_frequency_hz_min must match constant_beamwidth_reference_frequency_hz.",
        )
        if self.target_hpbw_deg is not None:
            require_positive_float("target_hpbw_deg", float(self.target_hpbw_deg))
        require_positive_float("required_peak_margin_db", float(self.required_peak_margin_db))
        require_positive_int("scan_azimuth_count", int(self.scan_azimuth_count))
        require_positive_float("gap_alias_safety", float(self.gap_alias_safety))

        frequencies = np.asarray(self.design_frequency_grid_hz, dtype=np.float64)
        require(frequencies.ndim == 1 and frequencies.size > 0, "design_frequency_grid_hz must be a non-empty 1-D sequence.")
        require(bool(np.all(np.isfinite(frequencies))), "design_frequency_grid_hz must contain finite values.")
        require(bool(np.all(np.diff(frequencies) > 0.0)), "design_frequency_grid_hz must be strictly increasing.")
        require(float(frequencies[0]) == 0.0, "design_frequency_grid_hz must start at 0 Hz.")
        require(float(frequencies[-1]) >= float(self.maximum_frequency_hz), "design_frequency_grid_hz must cover maximum_frequency_hz.")

        for start_m, stop_m, spacing_m in self.positive_layout_segments_m:
            require_positive_float("segment start_m", float(start_m))
            require_positive_float("segment stop_m", float(stop_m))
            require_positive_float("segment spacing_m", float(spacing_m))
            require(float(start_m) <= float(stop_m), "layout segment start_m must not exceed stop_m.")

        azimuths = np.asarray(self.evaluation_azimuths_deg, dtype=np.float64)
        require(azimuths.ndim == 1 and azimuths.size > 0, "evaluation_azimuths_deg must be a non-empty 1-D sequence.")
        require(bool(np.all((0.0 <= azimuths) & (azimuths <= 180.0))), "evaluation_azimuths_deg must lie in [0, 180].")


@dataclass(frozen=True)
class OperationalSparseArrayDefinition:
    """保存済み運用スパースアレイ定義を保持する。

    このクラスは、物理センサ座標、設計周波数ごとの active channel index、
    設計式と評価結果を保持し、JSON ファイルとの相互変換を担当する。

    入力は `positions_m[n_ch, 3]`、`active_channel_indices_by_frequency`、設計 record 群であり、
    出力は固定整相・SLC が使用する物理チャネル数、センサ座標、周波数別 active index である。

    BL/FRAZ/BTR の描画、波形生成、SLC 重み更新は責務に含めない。
    信号処理上は、実行時処理がアレイ CH 数をハードコードしないための幾何ファイル表現に位置づく。
    """

    schema_version: int
    fs_hz: float
    sound_speed_m_s: float
    valid_frequency_hz_min: float
    maximum_frequency_hz: float
    positions_m: FloatArray
    design_frequencies_hz: FloatArray
    active_channel_indices_by_frequency: tuple[IntArray, ...]
    records: tuple[dict[str, Any], ...]
    formula: dict[str, Any]

    def __post_init__(self) -> None:
        """保存ファイルから復元した配列 shape と index 範囲を検証する。"""
        positions = np.asarray(self.positions_m, dtype=np.float64)
        frequencies = np.asarray(self.design_frequencies_hz, dtype=np.float64)
        require(positions.ndim == 2 and positions.shape[1] == 3, "positions_m must have shape (n_ch, 3).")
        require(positions.shape[0] > 0, "positions_m must contain at least one channel.")
        require(frequencies.ndim == 1 and frequencies.size > 0, "design_frequencies_hz must have shape (n_frequency,).")
        require(len(self.active_channel_indices_by_frequency) == frequencies.size, "active index table must match n_frequency.")

        normalized_indices: list[IntArray] = []
        for active_indices in self.active_channel_indices_by_frequency:
            index_array = np.asarray(active_indices, dtype=np.int64)
            require(index_array.ndim == 1 and index_array.size > 0, "active channel indices must be non-empty 1-D arrays.")
            require(bool(np.all((0 <= index_array) & (index_array < positions.shape[0]))), "active channel index is out of range.")
            normalized_indices.append(index_array)

        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "design_frequencies_hz", frequencies)
        object.__setattr__(self, "active_channel_indices_by_frequency", tuple(normalized_indices))

    @property
    def n_ch(self) -> int:
        """ファイルに保存された物理チャネル数を返す。"""
        return int(self.positions_m.shape[0])

    @property
    def aperture_m(self) -> float:
        """物理配置全体の開口長を返す。単位は m。"""
        x_positions_m = np.asarray(self.positions_m[:, 0], dtype=np.float64)
        return float(np.max(x_positions_m) - np.min(x_positions_m))

    def active_channel_indices_for_frequency(self, frequency_hz: float) -> IntArray:
        """指定周波数で使う active channel index を返す。

        Args:
            frequency_hz: 使用周波数。単位は Hz。

        Returns:
            active channel index。shape は `[n_active_ch]`。

        Raises:
            ValueError: 周波数が負の場合。

        Notes:
            任意周波数入力では、設計周波数表のうち `frequency_hz` 以上で最も低い
            行を使う。高い周波数では必要開口が狭くなるため、低い周波数の広い
            active set を誤って使うより安全側になる。
        """
        require(float(frequency_hz) >= 0.0, "frequency_hz must be non-negative.")
        if float(frequency_hz) <= 0.0:
            return self.active_channel_indices_by_frequency[0].copy()

        frequency_index = int(np.searchsorted(self.design_frequencies_hz, float(frequency_hz), side="left"))
        if frequency_index >= self.design_frequencies_hz.size:
            frequency_index = int(self.design_frequencies_hz.size - 1)
        return self.active_channel_indices_by_frequency[frequency_index].copy()

    def to_bandwise_array_design(self) -> BandwiseArrayDesign:
        """保存済み active index 表を `BandwiseArrayDesign` へ変換する。

        Returns:
            帯域ごとの active channel を持つ `BandwiseArrayDesign`。
            `channel_positions_m` の shape は `[n_ch, 3]`、`shading_table` の shape は
            `[n_ch, n_frequency]` である。
        """
        return BandwiseArrayDesign.from_channel_positions_and_active_indices(
            channel_positions_m=self.positions_m,
            n_band=int(self.design_frequencies_hz.size),
            active_indices_per_band=[indices.copy() for indices in self.active_channel_indices_by_frequency],
        )

    def to_payload(self) -> dict[str, Any]:
        """JSON 保存用 payload へ変換する。"""
        return {
            "schema_version": int(self.schema_version),
            "fs_hz": float(self.fs_hz),
            "sound_speed_m_s": float(self.sound_speed_m_s),
            "valid_frequency_hz_min": float(self.valid_frequency_hz_min),
            "maximum_frequency_hz": float(self.maximum_frequency_hz),
            "n_ch": int(self.n_ch),
            "aperture_m": float(self.aperture_m),
            "positions_m": np.asarray(self.positions_m, dtype=np.float64).tolist(),
            "design_frequencies_hz": np.asarray(self.design_frequencies_hz, dtype=np.float64).tolist(),
            "active_channel_indices_by_frequency": [
                np.asarray(indices, dtype=np.int64).tolist()
                for indices in self.active_channel_indices_by_frequency
            ],
            "records": list(self.records),
            "formula": dict(self.formula),
        }

    def save_json(self, path: Path) -> None:
        """アレイ定義を JSON ファイルとして保存する。

        Args:
            path: 保存先 JSON パス。

        境界条件:
            親ディレクトリがない場合は作成する。既存ファイルは同じ schema で再生成できる
            設計成果物として上書きする。
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_payload(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OperationalSparseArrayDefinition":
        """JSON payload からアレイ定義を復元する。

        Args:
            payload: `to_payload()` と同じキーを持つ辞書。

        Returns:
            復元した `OperationalSparseArrayDefinition`。

        Raises:
            KeyError: 必須キーが不足する場合。
            ValueError: shape または index 範囲が不正な場合。
        """
        active_table_raw = payload["active_channel_indices_by_frequency"]
        if not isinstance(active_table_raw, list):
            raise ValueError("active_channel_indices_by_frequency must be a list.")

        active_table = tuple(np.asarray(indices, dtype=np.int64) for indices in active_table_raw)
        records_raw = payload.get("records", [])
        if not isinstance(records_raw, list):
            raise ValueError("records must be a list.")
        formula_raw = payload.get("formula", {})
        if not isinstance(formula_raw, dict):
            raise ValueError("formula must be a dict.")

        return cls(
            schema_version=int(payload["schema_version"]),
            fs_hz=float(payload["fs_hz"]),
            sound_speed_m_s=float(payload["sound_speed_m_s"]),
            valid_frequency_hz_min=float(payload["valid_frequency_hz_min"]),
            maximum_frequency_hz=float(payload["maximum_frequency_hz"]),
            positions_m=np.asarray(payload["positions_m"], dtype=np.float64),
            design_frequencies_hz=np.asarray(payload["design_frequencies_hz"], dtype=np.float64),
            active_channel_indices_by_frequency=active_table,
            records=tuple(dict(record) for record in records_raw if isinstance(record, dict)),
            formula=dict(formula_raw),
        )

    @classmethod
    def load_json(cls, path: Path) -> "OperationalSparseArrayDefinition":
        """JSON ファイルからアレイ定義を読み込む。

        Args:
            path: 読み込む JSON パス。

        Returns:
            ファイルに保存されたアレイ定義。

        Raises:
            FileNotFoundError: ファイルが存在しない場合。
            ValueError: JSON の schema または shape が不正な場合。
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("array definition JSON root must be an object.")
        return cls.from_payload(payload)


def _build_physical_positions(config: OperationalSparseArrayDesignConfig) -> FloatArray:
    """positive 側セグメントから中心対称な 1 列アレイ座標を作る。"""
    positive_positions: list[float] = []
    for start_m, stop_m, spacing_m in config.positive_layout_segments_m:
        # stop を含めるため spacing の半分を足す。浮動小数の丸めで重複しないよう最後に unique する。
        segment_positions = np.arange(float(start_m), float(stop_m) + 0.5 * float(spacing_m), float(spacing_m))
        positive_positions.extend(float(np.round(value, 10)) for value in segment_positions.tolist())

    positive_array = np.unique(np.asarray(positive_positions, dtype=np.float64))
    require(positive_array.size > 0, "positive layout must contain at least one sensor.")
    require(bool(np.all(np.diff(positive_array) > 0.0)), "positive sensor positions must be strictly increasing.")

    # positions_m shape: [n_ch, 3]。axis=0 が物理チャネル、axis=1 が x/y/z である。
    # 片舷評価では x 軸上の 1 列配置を使い、0 m を中心に左右対称へ展開する。
    x_positions_m = np.concatenate([-positive_array[::-1], np.array([0.0], dtype=np.float64), positive_array])
    positions_m = np.zeros((x_positions_m.size, 3), dtype=np.float64)
    positions_m[:, 0] = x_positions_m
    return positions_m


def _required_aperture_m(
    frequency_hz: float,
    sound_speed_m_s: float,
    target_hpbw_deg: float,
    safety_factor: float,
    maximum_aperture_m: float,
) -> float:
    """HPBW 目標から必要開口長を返す。

    公式:
        broadside ULA の目安として `HPBW_rad ≈ 0.886 λ / D` を使う。
        よって `D ≈ 0.886 c / (f HPBW_rad)`。

    0 Hz では波長が無限大になるため、本関数は正の周波数だけを受け付ける。
    """
    require_positive_float("frequency_hz", float(frequency_hz))
    target_hpbw_rad = np.deg2rad(float(target_hpbw_deg))
    aperture_m = 0.886 * float(sound_speed_m_s) / (float(frequency_hz) * target_hpbw_rad)
    return float(min(float(maximum_aperture_m), float(safety_factor) * aperture_m))


def _target_hpbw_deg_from_reference_aperture(
    sound_speed_m_s: float,
    reference_frequency_hz: float,
    reference_aperture_m: float,
) -> float:
    """基準周波数と全開口から一定化する HPBW 目標を計算する。

    Args:
        sound_speed_m_s: 音速。単位は m/s。
        reference_frequency_hz: ビーム幅を決める基準周波数。単位は Hz。
        reference_aperture_m: 基準周波数で使う物理全開口。単位は m。

    Returns:
        `HPBW_rad ≈ 0.886 c / (f_ref D_ref)` から求めた目標 HPBW。単位は deg。
    """
    require_positive_float("sound_speed_m_s", float(sound_speed_m_s))
    require_positive_float("reference_frequency_hz", float(reference_frequency_hz))
    require_positive_float("reference_aperture_m", float(reference_aperture_m))

    # 200 Hz 未満は全 CH を使うため、ビーム幅は周波数低下に従って広がる。
    # 200 Hz 以上では、この式で得た HPBW を下限幅として active aperture / shading 側で一定化する。
    target_hpbw_rad = 0.886 * float(sound_speed_m_s) / (
        float(reference_frequency_hz) * float(reference_aperture_m)
    )
    return float(np.rad2deg(target_hpbw_rad))


def _required_constant_beamwidth_aperture_m(
    frequency_hz: float,
    reference_frequency_hz: float,
    reference_aperture_m: float,
    safety_factor: float,
) -> float:
    """200 Hz 基準の一定ビーム幅を保つための active aperture を返す。

    `HPBW ≈ 0.886 c / (f D)` を一定にするには、`f D` を一定に保てばよい。
    したがって `D(f) = D_ref f_ref / f` とする。
    """
    require_positive_float("frequency_hz", float(frequency_hz))
    require_positive_float("reference_frequency_hz", float(reference_frequency_hz))
    require_positive_float("reference_aperture_m", float(reference_aperture_m))
    require_positive_float("safety_factor", float(safety_factor))

    aperture_m = float(reference_aperture_m) * float(reference_frequency_hz) / float(frequency_hz)
    return float(min(float(reference_aperture_m), float(safety_factor) * aperture_m))


def _candidate_all_inside_aperture(positions_m: FloatArray, required_aperture_m: float) -> IntArray:
    """指定開口内にある物理チャネルをすべて使う候補を返す。"""
    x_positions_m = np.asarray(positions_m[:, 0], dtype=np.float64)
    half_aperture_m = 0.5 * float(required_aperture_m)
    return np.flatnonzero(np.abs(x_positions_m) <= half_aperture_m + 1e-9).astype(np.int64)


def _candidate_uniform_subset(
    positions_m: FloatArray,
    frequency_hz: float,
    required_aperture_m: float,
    sound_speed_m_s: float,
    gap_alias_safety: float,
    segment_count: int,
) -> IntArray:
    """開口内から等間隔に近い active subset 候補を作る。

    低域では全センサを使うと、密配置と疎配置の密度差により高い sidelobe が出る場合がある。
    そのため、必要開口を保ったまま、概ね等間隔の sparse subset を作って sidelobe を抑える。
    """
    x_positions_m = np.asarray(positions_m[:, 0], dtype=np.float64)
    half_aperture_m = 0.5 * float(required_aperture_m)
    wavelength_m = float(sound_speed_m_s) / float(frequency_hz)

    # d <= λ/2 は等間隔 ULA の空間 alias 回避条件である。
    # 疎配置では厳密な grating 条件ではないが、候補 grid 間隔の上限として使う。
    alias_limited_step_m = 0.5 * wavelength_m * float(gap_alias_safety)
    aperture_limited_step_m = float(required_aperture_m) / float(segment_count)
    grid_step_m = max(0.05, min(alias_limited_step_m, aperture_limited_step_m))

    positive_targets = np.arange(0.0, half_aperture_m + 0.5 * grid_step_m, grid_step_m)
    target_positions_m = np.concatenate([-positive_targets[:0:-1], positive_targets])
    valid_indices = np.flatnonzero(np.abs(x_positions_m) <= half_aperture_m + 0.05)
    require(valid_indices.size > 0, "candidate aperture does not contain any physical sensor.")

    selected_indices: list[int] = []
    for target_position_m in target_positions_m.tolist():
        nearest_local_index = int(np.argmin(np.abs(x_positions_m[valid_indices] - float(target_position_m))))
        selected_indices.append(int(valid_indices[nearest_local_index]))

    center_index = int(np.argmin(np.abs(x_positions_m)))
    return np.unique(np.asarray([*selected_indices, center_index], dtype=np.int64))


def _active_spacing_summary_m(positions_m: FloatArray, active_indices: IntArray) -> tuple[float, float, float]:
    """active 配置の開口、最小間隔、最大間隔を返す。"""
    active_x_m = np.sort(np.asarray(positions_m, dtype=np.float64)[np.asarray(active_indices, dtype=np.int64), 0])
    if active_x_m.size <= 1:
        return 0.0, float("inf"), float("inf")
    spacings_m = np.diff(active_x_m)
    return (
        float(active_x_m[-1] - active_x_m[0]),
        float(np.min(spacings_m)),
        float(np.max(spacings_m)),
    )


def _evaluate_candidate(
    positions_m: FloatArray,
    active_indices: IntArray,
    frequency_hz: float,
    config: OperationalSparseArrayDesignConfig,
    scan_azimuths_deg: FloatArray,
) -> tuple[float, list[float]]:
    """active 候補の最悪 peak margin と方位別 margin を返す。"""
    active_positions_m = np.asarray(positions_m, dtype=np.float64)[np.asarray(active_indices, dtype=np.int64)]
    margins_db: list[float] = []
    for azimuth_deg in config.evaluation_azimuths_deg:
        beam_levels = _beam_levels_db20(
            positions_m=active_positions_m,
            frequency_hz=float(frequency_hz),
            target_azimuth_deg=float(azimuth_deg),
            scan_azimuths_deg=scan_azimuths_deg,
            sound_speed_m_s=float(config.sound_speed_m_s),
        )
        margins_db.append(
            _mainlobe_peak_margin_db(
                scan_azimuths_deg=scan_azimuths_deg,
                beam_levels_db20=beam_levels,
                target_azimuth_deg=float(azimuth_deg),
            )
        )
    return float(np.min(np.asarray(margins_db, dtype=np.float64))), margins_db


def _choose_active_indices_for_frequency(
    positions_m: FloatArray,
    frequency_hz: float,
    config: OperationalSparseArrayDesignConfig,
    scan_azimuths_deg: FloatArray,
) -> tuple[IntArray, dict[str, Any]]:
    """1 周波数の active channel を設計方針に従って決める。

    200 Hz 未満は低域の波長が長く grating 条件に余裕があるため全 CH を使う。
    200 Hz 以上では `f D` が一定になる active aperture を候補探索し、
    200 Hz 全開口で決まる HPBW より狭くならないようにする。
    """
    x_positions_m = np.asarray(positions_m[:, 0], dtype=np.float64)
    maximum_aperture_m = float(np.max(x_positions_m) - np.min(x_positions_m))
    reference_frequency_hz = float(config.constant_beamwidth_reference_frequency_hz)
    target_hpbw_deg = (
        float(config.target_hpbw_deg)
        if config.target_hpbw_deg is not None
        else _target_hpbw_deg_from_reference_aperture(
            sound_speed_m_s=float(config.sound_speed_m_s),
            reference_frequency_hz=reference_frequency_hz,
            reference_aperture_m=maximum_aperture_m,
        )
    )

    if float(frequency_hz) <= 0.0:
        all_indices = np.arange(positions_m.shape[0], dtype=np.int64)
        aperture_m, min_spacing_m, max_spacing_m = _active_spacing_summary_m(positions_m, all_indices)
        return all_indices, {
            "frequency_hz": float(frequency_hz),
            "active_channel_count": int(all_indices.size),
            "active_aperture_m": float(aperture_m),
            "active_min_spacing_m": float(min_spacing_m),
            "active_max_spacing_m": float(max_spacing_m),
            "active_max_gap_alias_limit_hz": float(config.sound_speed_m_s) / (2.0 * float(max_spacing_m)),
            "required_aperture_m": None,
            "wavelength_m": None,
            "target_hpbw_deg": float(target_hpbw_deg),
            "constant_beamwidth_reference_frequency_hz": float(reference_frequency_hz),
            "worst_peak_margin_db": None,
            "meets_required_peak_margin": None,
            "selection_mode": "all_channels_for_dc",
        }

    if float(frequency_hz) < reference_frequency_hz:
        all_indices = np.arange(positions_m.shape[0], dtype=np.int64)
        aperture_m, min_spacing_m, max_spacing_m = _active_spacing_summary_m(positions_m, all_indices)
        worst_margin_db, margins_db = _evaluate_candidate(
            positions_m=positions_m,
            active_indices=all_indices,
            frequency_hz=float(frequency_hz),
            config=config,
            scan_azimuths_deg=scan_azimuths_deg,
        )
        meets_margin = bool(worst_margin_db >= float(config.required_peak_margin_db))
        record: dict[str, Any] = {
            "frequency_hz": float(frequency_hz),
            "active_channel_count": int(all_indices.size),
            "active_aperture_m": float(aperture_m),
            "active_min_spacing_m": float(min_spacing_m),
            "active_max_spacing_m": float(max_spacing_m),
            "active_max_gap_alias_limit_hz": float(config.sound_speed_m_s) / (2.0 * float(max_spacing_m)),
            "required_aperture_m": float(maximum_aperture_m),
            "wavelength_m": float(config.sound_speed_m_s) / float(frequency_hz),
            "target_hpbw_deg": float(target_hpbw_deg),
            "constant_beamwidth_reference_frequency_hz": float(reference_frequency_hz),
            "worst_peak_margin_db": float(worst_margin_db),
            "meets_required_peak_margin": bool(meets_margin),
            "selection_mode": "all_channels_below_constant_beamwidth_reference",
            "aperture_safety_factor": None,
            "candidate_rank_score": [0 if meets_margin else 1, int(all_indices.size), float(max_spacing_m)],
        }
        for azimuth_deg, margin_db in zip(config.evaluation_azimuths_deg, margins_db, strict=True):
            record[f"peak_margin_az{int(round(float(azimuth_deg))):03d}_db"] = float(margin_db)
        return all_indices, record

    candidates: list[tuple[tuple[int, float, int, float], IntArray, dict[str, Any]]] = []
    wavelength_m = float(config.sound_speed_m_s) / float(frequency_hz)
    for safety_factor in config.aperture_safety_factors:
        if config.target_hpbw_deg is None:
            required_aperture = _required_constant_beamwidth_aperture_m(
                frequency_hz=float(frequency_hz),
                reference_frequency_hz=reference_frequency_hz,
                reference_aperture_m=maximum_aperture_m,
                safety_factor=float(safety_factor),
            )
        else:
            required_aperture = _required_aperture_m(
                frequency_hz=float(frequency_hz),
                sound_speed_m_s=float(config.sound_speed_m_s),
                target_hpbw_deg=float(target_hpbw_deg),
                safety_factor=float(safety_factor),
                maximum_aperture_m=maximum_aperture_m,
            )
        raw_candidate_indices = [
            ("all_inside_aperture", _candidate_all_inside_aperture(positions_m, required_aperture)),
            *(
                (
                    f"uniform_subset_{segment_count}",
                    _candidate_uniform_subset(
                        positions_m=positions_m,
                        frequency_hz=float(frequency_hz),
                        required_aperture_m=required_aperture,
                        sound_speed_m_s=float(config.sound_speed_m_s),
                        gap_alias_safety=float(config.gap_alias_safety),
                        segment_count=int(segment_count),
                    ),
                )
                for segment_count in config.uniform_subset_segment_counts
            ),
        ]
        for selection_mode, active_indices in raw_candidate_indices:
            if active_indices.size < 3:
                continue
            worst_margin_db, margins_db = _evaluate_candidate(
                positions_m=positions_m,
                active_indices=active_indices,
                frequency_hz=float(frequency_hz),
                config=config,
                scan_azimuths_deg=scan_azimuths_deg,
            )
            aperture_m, min_spacing_m, max_spacing_m = _active_spacing_summary_m(positions_m, active_indices)
            meets_margin = bool(worst_margin_db >= float(config.required_peak_margin_db))
            record: dict[str, Any] = {
                "frequency_hz": float(frequency_hz),
                "active_channel_count": int(active_indices.size),
                "active_aperture_m": float(aperture_m),
                "active_min_spacing_m": float(min_spacing_m),
                "active_max_spacing_m": float(max_spacing_m),
                "active_max_gap_alias_limit_hz": float(config.sound_speed_m_s) / (2.0 * float(max_spacing_m)),
                "required_aperture_m": float(required_aperture),
                "wavelength_m": float(wavelength_m),
                "target_hpbw_deg": float(target_hpbw_deg),
                "constant_beamwidth_reference_frequency_hz": float(reference_frequency_hz),
                "worst_peak_margin_db": float(worst_margin_db),
                "meets_required_peak_margin": bool(meets_margin),
                "selection_mode": selection_mode,
                "aperture_safety_factor": float(safety_factor),
            }
            for azimuth_deg, margin_db in zip(config.evaluation_azimuths_deg, margins_db, strict=True):
                record[f"peak_margin_az{int(round(float(azimuth_deg))):03d}_db"] = float(margin_db)

            # pass した候補は channel 数を優先し、同数なら最大 gap が小さい候補を選ぶ。
            # 全候補が未達の場合は、方式の成否を判断する前に最も margin 不足が小さい候補を採用する。
            # これにより、実装上の候補選択が悪くて shading 評価を不利にすることを避ける。
            pass_rank = 0 if meets_margin else 1
            margin_deficit_db = max(0.0, float(config.required_peak_margin_db) - float(worst_margin_db))
            score = (pass_rank, float(margin_deficit_db), int(active_indices.size), float(max_spacing_m))
            candidates.append((score, active_indices.astype(np.int64), record))

    require(len(candidates) > 0, "no active channel candidate was generated.")
    candidates.sort(key=lambda item: item[0])
    best_score, best_indices, best_record = candidates[0]
    best_record["candidate_rank_score"] = [
        int(best_score[0]),
        float(best_score[1]),
        int(best_score[2]),
        float(best_score[3]),
    ]
    return best_indices, best_record

def design_operational_sparse_array(
    config: OperationalSparseArrayDesignConfig,
) -> OperationalSparseArrayDefinition:
    """運用スパースアレイ定義を設計する。

    Args:
        config: fs、音速、周波数範囲、開口式、保存先を含む設計条件。

    Returns:
        物理センサ位置と周波数別 active channel を持つアレイ定義。

    Raises:
        ValueError: 設計候補が生成できない、または shape が不正な場合。
    """
    positions_m = _build_physical_positions(config)
    physical_aperture_m = float(np.max(positions_m[:, 0]) - np.min(positions_m[:, 0]))
    effective_target_hpbw_deg = (
        float(config.target_hpbw_deg)
        if config.target_hpbw_deg is not None
        else _target_hpbw_deg_from_reference_aperture(
            sound_speed_m_s=float(config.sound_speed_m_s),
            reference_frequency_hz=float(config.constant_beamwidth_reference_frequency_hz),
            reference_aperture_m=physical_aperture_m,
        )
    )
    design_frequencies_hz = np.asarray(config.design_frequency_grid_hz, dtype=np.float64)
    scan_azimuths_deg = np.linspace(0.0, 180.0, int(config.scan_azimuth_count), dtype=np.float64)

    active_table: list[IntArray] = []
    records: list[dict[str, Any]] = []
    for frequency_hz in design_frequencies_hz.tolist():
        active_indices, record = _choose_active_indices_for_frequency(
            positions_m=positions_m,
            frequency_hz=float(frequency_hz),
            config=config,
            scan_azimuths_deg=scan_azimuths_deg,
        )
        active_table.append(active_indices)
        records.append(record)

    dense_spacing_limit_m = float(config.sound_speed_m_s) / (2.0 * float(config.maximum_frequency_hz))
    formula = {
        "spacing_alias_condition": "d <= c / (2 f_max)",
        "maximum_alias_free_uniform_spacing_m": dense_spacing_limit_m,
        "hpbw_aperture_formula": "D ~= 0.886 c / (f HPBW_rad)",
        "target_hpbw_deg": float(effective_target_hpbw_deg),
        "constant_beamwidth_reference_frequency_hz": float(config.constant_beamwidth_reference_frequency_hz),
        "constant_beamwidth_reference_aperture_m": float(physical_aperture_m),
        "required_peak_margin_db": float(config.required_peak_margin_db),
        "valid_frequency_hz_min": float(config.valid_frequency_hz_min),
        "note": "0 Hz は波長が無限大のため方位性能保証対象外。性能評価は valid_frequency_hz_min 以上で行う。",
    }

    return OperationalSparseArrayDefinition(
        schema_version=1,
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
        valid_frequency_hz_min=float(config.valid_frequency_hz_min),
        maximum_frequency_hz=float(config.maximum_frequency_hz),
        positions_m=positions_m,
        design_frequencies_hz=design_frequencies_hz,
        active_channel_indices_by_frequency=tuple(active_table),
        records=tuple(records),
        formula=formula,
    )


def _write_records_csv(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    """周波数別設計 record を CSV として保存する。"""
    if len(records) == 0:
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def save_operational_sparse_array(config: OperationalSparseArrayDesignConfig) -> OperationalSparseArrayDefinition:
    """運用スパースアレイを設計し、JSON と任意 CSV を保存する。

    Args:
        config: 保存先と設計条件。

    Returns:
        保存したアレイ定義。使用側は `OperationalSparseArrayDefinition.load_json()` で
        同じ内容を読み込む。
    """
    definition = design_operational_sparse_array(config)
    definition.save_json(Path(config.output_json_path))
    if config.output_csv_path is not None:
        _write_records_csv(Path(config.output_csv_path), definition.records)
    return definition


def load_operational_sparse_array(path: Path) -> OperationalSparseArrayDefinition:
    """運用スパースアレイ定義 JSON を読み込む短縮 API。

    Args:
        path: `save_operational_sparse_array()` が保存した JSON パス。

    Returns:
        アレイ定義。`n_ch` はファイル内の `positions_m` から決まる。
    """
    return OperationalSparseArrayDefinition.load_json(Path(path))


__all__ = [
    "OperationalSparseArrayDesignConfig",
    "OperationalSparseArrayDefinition",
    "design_operational_sparse_array",
    "save_operational_sparse_array",
    "load_operational_sparse_array",
]
