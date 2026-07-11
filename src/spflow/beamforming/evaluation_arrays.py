"""BL/FRAZ/BTR の描画元配列と BL 校正用特徴量を計算する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_index_in_range, require_positive_float
from ..spectral_level import rms_amplitude_to_level_db


@dataclass(frozen=True)
class BeamLevelDisplayArrays:
    """同一 dB reference で表した BL/FRAZ/source-frequency BL を保持する。

    Attributes:
        fraz_level_db: 周波数・方位表示。shape は `[n_beam, n_frequency]`。
        target_frequency_bl_level_db: target 周波数 BL。shape は `[n_beam]`。
        source_frequency_bl_level_db: 各 source 周波数の最大値。shape は `[n_beam]`。
        level_reference_label: `dB re input RMS` などの基準量ラベル。

    このクラスは表示元配列を保持するが、方式比較、採否判定、plot 保存は責務に含めない。
    信号処理上は、同じ配列定義を数値評価と視覚評価で共有する境界に位置づく。
    """

    fraz_level_db: NDArray[np.float64]
    target_frequency_bl_level_db: NDArray[np.float64]
    source_frequency_bl_level_db: NDArray[np.float64]
    level_reference_label: str


@dataclass(frozen=True)
class BlShapeFeatures:
    """人間の BL 視覚評価との校正に使う形状特徴量を保持する。

    Attributes:
        peak_azimuth_deg: 全体 peak 方位。単位は deg。
        peak_level_db: 全体 peak level。単位は `level_reference_label` が示す dB reference。
        peak_width_3db_deg: peak から -3 dB 以上で連結する領域幅。単位は deg。
        guard_outside_peak_level_db: source guard 外の最大 level。
        guard_outside_p95_level_db: source guard 外 level の 95 percentile。
        guard_outside_p99_level_db: source guard 外 level の 99 percentile。
        integrated_guard_outside_level_db: guard 外の線形 power 和を RMS level 化した値。
        source_to_guard_peak_margin_db: source 内 peak と guard 外 peak の差。単位は dB差。
        source_separation_valley_depth_db: 複数 source 間 valley の浅い側 source peak に対する深さ。
            source が 2 個未満の場合は `None`。
        level_reference_label: level 値の dB reference。

    このクラスは観測特徴量であり、方式の合否や scalar score は責務に含めない。
    """

    peak_azimuth_deg: float
    peak_level_db: float
    peak_width_3db_deg: float
    guard_outside_peak_level_db: float
    guard_outside_p95_level_db: float
    guard_outside_p99_level_db: float
    integrated_guard_outside_level_db: float
    source_to_guard_peak_margin_db: float
    source_separation_valley_depth_db: float | None
    level_reference_label: str


def build_beam_level_display_arrays(
    beam_spectrum: NDArray[Any],
    *,
    target_frequency_index: int,
    source_frequency_indices: NDArray[Any],
    reference_rms: float,
    level_reference_label: str,
    frame_axis: int = -1,
    floor_db: float | None = None,
) -> BeamLevelDisplayArrays:
    """beam spectrum から共通定義の FRAZ と BL 配列を作る。

    Args:
        beam_spectrum: 複素 beam spectrum。shape は `[n_beam, n_frequency, n_frame]` を基本とし、
            `frame_axis` だけ任意位置を許可する。残る axis=0 は beam、axis=1 は frequency。
        target_frequency_index: target frequency bin index。
        source_frequency_indices: source frequency bin index。shape は `[n_source_frequency]`。
        reference_rms: 0 dB に対応する RMS amplitude。
        level_reference_label: `dB re input RMS` などの表示基準量。
        frame_axis: snapshot/frame 軸。
        floor_db: 表示用 level floor。`None` なら 0 amplitude は `-inf`。

    Returns:
        FRAZ、target-frequency BL、source-frequency BL の表示元配列。

    Raises:
        ValueError: shape、axis、bin index、reference、label が不正な場合。

    境界条件:
        source-frequency BL は各 source 周波数の FRAZ level の最大値とする。
        異周波数 source を target 周波数だけの BL から消失したと誤解しないためである。
    """
    spectrum = np.asarray(beam_spectrum)
    require(spectrum.ndim == 3, "beam_spectrum must have three axes: beam, frequency, frame.")
    axis = int(frame_axis)
    if axis < 0:
        axis += spectrum.ndim
    require(0 <= axis < spectrum.ndim, "frame_axis is out of bounds.")
    canonical = np.moveaxis(spectrum, axis, -1)
    require(canonical.shape[0] > 0, "beam_spectrum must contain at least one beam.")
    require(canonical.shape[1] > 0, "beam_spectrum must contain at least one frequency.")
    require(canonical.shape[2] > 0, "beam_spectrum must contain at least one frame.")
    require(bool(np.all(np.isfinite(canonical))), "beam_spectrum must be finite.")
    require_positive_float("reference_rms", float(reference_rms))
    require(bool(str(level_reference_label).strip()), "level_reference_label must not be empty.")
    require_index_in_range(
        "target_frequency_index", int(target_frequency_index), canonical.shape[1]
    )
    source_indices = np.asarray(source_frequency_indices, dtype=np.int64)
    require(
        source_indices.ndim == 1 and source_indices.size > 0,
        "source_frequency_indices must have shape (n_source_frequency,).",
    )
    require(
        bool(np.all((0 <= source_indices) & (source_indices < canonical.shape[1]))),
        "source_frequency_indices contain an out-of-range index.",
    )

    # frame 軸で |Y|^2 を平均して RMS amplitude を求める。
    # shape は [n_beam, n_frequency, n_frame] -> [n_beam, n_frequency]。
    fraz_rms = np.sqrt(np.mean(np.abs(canonical) ** 2, axis=-1))
    fraz_level = rms_amplitude_to_level_db(
        fraz_rms,
        reference_rms=float(reference_rms),
        floor_db=floor_db,
    )
    target_bl = np.asarray(fraz_level[:, int(target_frequency_index)], dtype=np.float64)
    # shape [n_beam, n_source_frequency] を source frequency 軸で最大化し、
    # 各既知 source を固有周波数で可視化できる [n_beam] 表示へまとめる。
    source_bl = np.asarray(np.max(fraz_level[:, source_indices], axis=1), dtype=np.float64)
    return BeamLevelDisplayArrays(
        fraz_level_db=np.asarray(fraz_level, dtype=np.float64),
        target_frequency_bl_level_db=target_bl,
        source_frequency_bl_level_db=source_bl,
        level_reference_label=str(level_reference_label),
    )


def calculate_btr_relative_level_db(
    beam_time_rms: NDArray[Any],
    *,
    floor_db: float = -120.0,
) -> NDArray[np.float64]:
    """beam-time RMS amplitude を時刻ごとの最大値基準 BTR へ変換する。

    Args:
        beam_time_rms: beam-time RMS amplitude。shape は `[n_time, n_beam]`。
        floor_db: 各時刻最大値に対する表示下限。単位は `dB re frame max`。

    Returns:
        BTR relative level。shape は `[n_time, n_beam]`、単位は `dB re frame max`。

    Raises:
        ValueError: shape、有限性、非負条件、floor、または全 beam 0 の時刻がある場合。

    境界条件:
        BTR は各時刻を独立に 0 dB 正規化するため、時刻間の絶対 level 比較には使えない。
        全 beam 0 の時刻は reference を定義できないため例外とする。
    """
    rms = np.asarray(beam_time_rms, dtype=np.float64)
    require(rms.ndim == 2, "beam_time_rms must have shape (n_time, n_beam).")
    require(rms.shape[0] > 0 and rms.shape[1] > 0, "beam_time_rms axes must not be empty.")
    require(bool(np.all(np.isfinite(rms))), "beam_time_rms must be finite.")
    require(bool(np.all(rms >= 0.0)), "beam_time_rms must be non-negative.")
    floor = float(floor_db)
    require(bool(np.isfinite(floor)) and floor < 0.0, "floor_db must be a finite negative value.")
    frame_max = np.max(rms, axis=1, keepdims=True)
    require(bool(np.all(frame_max > 0.0)), "each time row must contain a positive beam RMS.")
    relative = rms / frame_max
    return rms_amplitude_to_level_db(relative, reference_rms=1.0, floor_db=floor)


def calculate_bl_shape_features(
    axis_azimuth_deg: NDArray[Any],
    bl_level_db: NDArray[Any],
    source_mask: NDArray[np.bool_],
    *,
    source_beam_indices: NDArray[Any] | None = None,
    level_reference_label: str,
) -> BlShapeFeatures:
    """BL 視覚評価を数値指標へ校正するための形状特徴量を計算する。

    Args:
        axis_azimuth_deg: beam 方位軸。shape は `[n_beam]`、単位は deg、狭義単調増加。
        bl_level_db: BL level。shape は `[n_beam]`。
        source_mask: source mainlobe と guard を示す mask。shape は `[n_beam]`。
        source_beam_indices: source 中心 index。shape は `[n_source]`。2 source 以上なら
            隣接 source 間 valley depth を計算する。省略時は `None`。
        level_reference_label: BL level の dB reference。

    Returns:
        scalar score や合否を含まない BL 形状特徴量。

    Raises:
        ValueError: shape、有限性、軸順序、mask、source index が不正な場合。

    境界条件:
        -3 dB 幅は global peak を含む連結区間だけを使い、離れた sidelobe を mainlobe 幅へ
        混入させない。guard 外が空の場合は評価不能なので例外とする。
    """
    azimuth = np.asarray(axis_azimuth_deg, dtype=np.float64)
    levels = np.asarray(bl_level_db, dtype=np.float64)
    mask = np.asarray(source_mask, dtype=np.bool_)
    require(
        azimuth.ndim == 1 and azimuth.size > 1,
        "axis_azimuth_deg must have shape (n_beam,) with at least two beams.",
    )
    require(levels.shape == azimuth.shape, "bl_level_db shape must match axis_azimuth_deg.")
    require(mask.shape == azimuth.shape, "source_mask shape must match axis_azimuth_deg.")
    require(bool(np.all(np.isfinite(azimuth))), "axis_azimuth_deg must be finite.")
    require(bool(np.all(np.isfinite(levels))), "bl_level_db must be finite.")
    require(bool(np.all(np.diff(azimuth) > 0.0)), "axis_azimuth_deg must be strictly increasing.")
    require(bool(np.any(mask)), "source_mask must include at least one beam.")
    non_source_mask = np.logical_not(mask)
    require(
        bool(np.any(non_source_mask)), "source_mask must leave at least one guard-outside beam."
    )
    require(bool(str(level_reference_label).strip()), "level_reference_label must not be empty.")

    peak_index = int(np.argmax(levels))
    peak_level = float(levels[peak_index])
    above_3db = levels >= peak_level - 3.0
    left_index = peak_index
    while left_index > 0 and bool(above_3db[left_index - 1]):
        left_index -= 1
    right_index = peak_index
    while right_index + 1 < levels.size and bool(above_3db[right_index + 1]):
        right_index += 1
    peak_width_deg = float(azimuth[right_index] - azimuth[left_index])

    outside_levels = levels[non_source_mask]
    source_peak_level = float(np.max(levels[mask]))
    outside_peak_level = float(np.max(outside_levels))
    # dB level を power ratio へ戻して方位 sample 間で和を取り、再び RMS level 表示へ戻す。
    # 非一様方位軸の積分重みは別の特徴量として検討すべきため、ここでは beam sample 和と明記する。
    outside_power_sum = float(np.sum(10.0 ** (outside_levels / 10.0)))
    integrated_level = float(10.0 * np.log10(outside_power_sum))

    valley_depth: float | None = None
    if source_beam_indices is not None:
        source_indices = np.asarray(source_beam_indices, dtype=np.int64)
        require(
            source_indices.ndim == 1 and source_indices.size > 0,
            "source_beam_indices must have shape (n_source,).",
        )
        require(
            bool(np.all((0 <= source_indices) & (source_indices < levels.size))),
            "source_beam_indices contain an out-of-range index.",
        )
        unique_indices = np.unique(source_indices)
        if unique_indices.size >= 2:
            pair_depths: list[float] = []
            for left_source, right_source in zip(
                unique_indices[:-1], unique_indices[1:], strict=True
            ):
                interval = levels[int(left_source) : int(right_source) + 1]
                valley_level = float(np.min(interval))
                shallower_peak = min(
                    float(levels[int(left_source)]), float(levels[int(right_source)])
                )
                pair_depths.append(shallower_peak - valley_level)
            # 複数 source pair がある場合、最も分離しにくい浅い valley を代表値にする。
            valley_depth = float(min(pair_depths))

    return BlShapeFeatures(
        peak_azimuth_deg=float(azimuth[peak_index]),
        peak_level_db=peak_level,
        peak_width_3db_deg=peak_width_deg,
        guard_outside_peak_level_db=outside_peak_level,
        guard_outside_p95_level_db=float(np.percentile(outside_levels, 95.0)),
        guard_outside_p99_level_db=float(np.percentile(outside_levels, 99.0)),
        integrated_guard_outside_level_db=integrated_level,
        source_to_guard_peak_margin_db=source_peak_level - outside_peak_level,
        source_separation_valley_depth_db=valley_depth,
        level_reference_label=str(level_reference_label),
    )
