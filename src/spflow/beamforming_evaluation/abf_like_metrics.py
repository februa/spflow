"""ABF-like 非信号方位抑圧の mask と評価指標を扱うモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_non_negative_float, require_positive_int

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class SourceSectorMask:
    """既知 source 方位と non-source sector を分ける評価 mask を保持する。

    このクラスは、source として保護する beam index、source guard 幅、mask 種別を入力として、
    source mask `[n_beam]` と non-source mask `[n_beam]` を後段評価へ渡す。

    出力は ABF-like 評価や source-mask SLC が共有して使う mask 定義である。
    BL / FRAZ / BTR レベルの計算、source peak 検出、SLC 係数推定は責務に含めない。

    信号処理上は、観測対象 source の mainlobe を評価対象外にし、信号が存在しない方位の
    包絡線抑圧を採否中心に置くための評価領域定義に位置づく。
    """

    source_mask: BoolArray
    source_beam_indices: IntArray
    guard_beam_count: int
    mask_type: str = "oracle"

    def __post_init__(self) -> None:
        """mask と source index の整合を検証する。"""
        source_mask = np.asarray(self.source_mask, dtype=np.bool_)
        source_indices = np.asarray(self.source_beam_indices, dtype=np.int64)
        require(source_mask.ndim == 1, "source_mask must have shape (n_beam,).")
        require(source_mask.size > 0, "source_mask must contain at least one beam.")
        require(source_indices.ndim == 1, "source_beam_indices must have shape (n_source,).")
        require(source_indices.size > 0, "source_beam_indices must contain at least one source.")
        require_non_negative_float("guard_beam_count", float(self.guard_beam_count))
        require(
            bool(np.all((0 <= source_indices) & (source_indices < source_mask.size))),
            "source_beam_indices contain out-of-range index.",
        )
        require(
            bool(np.all(source_mask[source_indices])),
            "source_mask must include every source_beam_indices entry.",
        )

        object.__setattr__(self, "source_mask", source_mask.copy())
        object.__setattr__(self, "source_beam_indices", source_indices.copy())
        object.__setattr__(self, "guard_beam_count", int(self.guard_beam_count))
        object.__setattr__(self, "mask_type", str(self.mask_type))

    @property
    def non_source_mask(self) -> BoolArray:
        """non-source sector mask を返す。

        Returns:
            `~source_mask`。shape は `[n_beam]`。

        境界条件:
            source guard が広すぎて全 beam を覆う場合、non-source は空になる。
            その場合、評価関数または SLC 側で安全側に停止する。
        """
        # source を消さずに source 外だけを評価するため、source mask の反転を
        # non-source sector として一貫して使う。
        return np.logical_not(self.source_mask)

    def source_region_slices(self) -> tuple[slice, ...]:
        """各 source の guard 付き局所領域を返す。

        Returns:
            source ごとの slice。各 slice は `[start, stop)` で、axis=0 の beam 軸に対応する。

        境界条件:
            複数 source の guard が重なる場合でも、source ごとの peak 保護評価では
            それぞれの中心 beam まわりを独立に読む。merged mask は non-source 除外に使う。
        """
        regions: list[slice] = []
        n_beam = int(self.source_mask.size)
        for source_index in self.source_beam_indices.tolist():
            start = max(0, int(source_index) - int(self.guard_beam_count))
            stop = min(n_beam, int(source_index) + int(self.guard_beam_count) + 1)
            regions.append(slice(start, stop))
        return tuple(regions)

    def as_dict(self) -> dict[str, object]:
        """JSON summary に保存しやすい辞書へ変換する。"""
        return {
            "mask_type": self.mask_type,
            "source_count": int(self.source_beam_indices.size),
            "source_beam_indices": [int(index) for index in self.source_beam_indices.tolist()],
            "guard_beam_count": int(self.guard_beam_count),
            "source_mask_true_count": int(np.count_nonzero(self.source_mask)),
            "non_source_true_count": int(np.count_nonzero(self.non_source_mask)),
        }


@dataclass(frozen=True)
class AbfLikeNonSourceMetrics:
    """ABF-like 非信号方位抑圧の採否に使う scalar 指標を保持する。

    このクラスは、固定整相 before と候補方式 after の BL/FRAZ/BTR 等価レベル列から、
    source 保護と non-source sector 抑圧を同時に読むための metric を保持する。

    入力は source mask、before/after レベル、方位軸であり、出力は dB 差分、false peak 数、
    gated local worsening などの summary 値である。

    SLC 重み推定、FIR 設計、BL 描画は責務に含めない。
    信号処理上は、一点 null ではなく source mask 外の包絡線が下がったかを判定する
    ABF-like 評価層に位置づく。
    """

    max_abs_source_peak_delta_db: float
    max_source_azimuth_error_deg: float
    non_source_global_peak_delta_db: float
    non_source_p95_level_delta_db: float
    non_source_p99_level_delta_db: float
    non_source_integrated_level_delta_db: float
    source_to_non_source_margin_delta_db: float
    false_peak_count_before: int
    false_peak_count_after: int
    false_peak_count_delta: int
    max_local_worsening_db_gated: float
    max_local_worsening_azimuth_deg: float | None
    level_unit_label: str
    delta_unit_label: str = "dB re before level"

    def as_dict(self) -> dict[str, float | int | str | None]:
        """JSON summary に保存しやすい辞書へ変換する。"""
        return {
            "max_abs_source_peak_delta_db": float(self.max_abs_source_peak_delta_db),
            "max_source_azimuth_error_deg": float(self.max_source_azimuth_error_deg),
            "non_source_global_peak_delta_db": float(self.non_source_global_peak_delta_db),
            "non_source_p95_level_delta_db": float(self.non_source_p95_level_delta_db),
            "non_source_p99_level_delta_db": float(self.non_source_p99_level_delta_db),
            "non_source_integrated_level_delta_db": float(
                self.non_source_integrated_level_delta_db
            ),
            "source_to_non_source_margin_delta_db": float(
                self.source_to_non_source_margin_delta_db
            ),
            "false_peak_count_before": int(self.false_peak_count_before),
            "false_peak_count_after": int(self.false_peak_count_after),
            "false_peak_count_delta": int(self.false_peak_count_delta),
            "max_local_worsening_db_gated": float(self.max_local_worsening_db_gated),
            "max_local_worsening_azimuth_deg": self.max_local_worsening_azimuth_deg,
            "level_unit_label": self.level_unit_label,
            "delta_unit_label": self.delta_unit_label,
        }


@dataclass(frozen=True)
class AbfLikeMetricDecision:
    """ABF-like 指標の pass/hold/fail 判定を保持する。

    このクラスは、source 保護、non-source 抑圧、false peak、局所悪化の閾値判定をまとめる。
    入力は `AbfLikeNonSourceMetrics` と任意の runtime/健全性指標であり、出力は status と理由である。

    方式の係数更新や fallback 適用は責務に含めない。
    信号処理上は、候補方式を 8 月評価へ入れてよいかを同じ閾値で読む採否層に位置づく。
    """

    status: str
    failure_reasons: tuple[str, ...]
    hold_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """JSON summary に保存しやすい辞書へ変換する。"""
        return {
            "status": self.status,
            "failure_reasons": [str(reason) for reason in self.failure_reasons],
            "hold_reasons": [str(reason) for reason in self.hold_reasons],
        }


def build_source_sector_mask(
    n_beam: int,
    source_beam_indices: NDArray[Any],
    guard_beam_count: int,
    mask_type: str = "oracle",
) -> SourceSectorMask:
    """source beam index と guard 幅から source / non-source mask を作る。

    Args:
        n_beam: beam 数。shape を持たない count。
        source_beam_indices: source 中心 beam index。shape は `[n_source]`。
        guard_beam_count: source 中心の左右に保護する beam 本数。単位は beam。
        mask_type: `oracle` または `detected` など、mask の由来を表す文字列。

    Returns:
        source mask 定義。`source_mask` の shape は `[n_beam]`。

    Raises:
        ValueError: beam 数、source index、guard 幅が不正な場合。

    境界条件:
        source が端 beam にある場合、guard 領域は `[0, n_beam)` にクリップする。
        端で guard が片側だけになるのは、存在しない beam を評価領域へ含めないためである。
    """
    require_positive_int("n_beam", int(n_beam))
    require_non_negative_float("guard_beam_count", float(guard_beam_count))
    source_indices = np.asarray(source_beam_indices, dtype=np.int64)
    require(source_indices.ndim == 1, "source_beam_indices must have shape (n_source,).")
    require(source_indices.size > 0, "source_beam_indices must not be empty.")
    require(
        bool(np.all((0 <= source_indices) & (source_indices < int(n_beam)))),
        "source_beam_indices contain out-of-range index.",
    )

    source_mask = np.zeros(int(n_beam), dtype=np.bool_)
    for source_index in source_indices.tolist():
        start = max(0, int(source_index) - int(guard_beam_count))
        stop = min(int(n_beam), int(source_index) + int(guard_beam_count) + 1)
        # source guard 内は観測対象 source の mainlobe として扱い、
        # sidelobe / false peak / non-source 抑圧の採否から除外する。
        source_mask[start:stop] = True
    return SourceSectorMask(
        source_mask=source_mask,
        source_beam_indices=source_indices,
        guard_beam_count=int(guard_beam_count),
        mask_type=str(mask_type),
    )


def build_source_sector_mask_from_azimuths(
    axis_azimuth_deg: NDArray[Any],
    source_azimuths_deg: NDArray[Any],
    guard_deg: float,
    mask_type: str = "oracle",
) -> SourceSectorMask:
    """source 方位と guard 角から source / non-source mask を作る。

    Args:
        axis_azimuth_deg: beam 方位軸。shape は `[n_beam]`、単位は deg。
        source_azimuths_deg: source 方位。shape は `[n_source]`、単位は deg。
        guard_deg: source 方位から左右に保護する角度幅。単位は deg。
        mask_type: `oracle` または `detected` など、mask の由来を表す文字列。

    Returns:
        source mask 定義。`source_mask` の shape は `[n_beam]`。

    Raises:
        ValueError: 方位軸、source 方位、guard 幅が不正な場合。

    境界条件:
        等 cos 方位軸では beam 間隔が deg 上で非一様になり得る。
        この関数は角度 guard をそのまま使い、最近傍 source index は診断用に保持する。
    """
    azimuths = np.asarray(axis_azimuth_deg, dtype=np.float64)
    source_azimuths = np.asarray(source_azimuths_deg, dtype=np.float64)
    require(azimuths.ndim == 1, "axis_azimuth_deg must have shape (n_beam,).")
    require(azimuths.size > 0, "axis_azimuth_deg must not be empty.")
    require(source_azimuths.ndim == 1, "source_azimuths_deg must have shape (n_source,).")
    require(source_azimuths.size > 0, "source_azimuths_deg must not be empty.")
    require(bool(np.all(np.isfinite(azimuths))), "axis_azimuth_deg must contain finite values.")
    require(
        bool(np.all(np.isfinite(source_azimuths))),
        "source_azimuths_deg must contain finite values.",
    )
    require_non_negative_float("guard_deg", float(guard_deg))

    source_indices = np.zeros(source_azimuths.size, dtype=np.int64)
    source_mask = np.zeros(azimuths.size, dtype=np.bool_)
    for source_number, source_azimuth_deg in enumerate(source_azimuths.tolist()):
        source_indices[source_number] = int(np.argmin(np.abs(azimuths - float(source_azimuth_deg))))
        # 角度 guard は評価上の source mainlobe 保護幅であり、exact marker 一点を
        # source として扱う誤判定を避けるために beam 方位軸上の範囲へ展開する。
        source_mask |= np.abs(azimuths - float(source_azimuth_deg)) <= float(guard_deg)

    if not bool(np.any(source_mask)):
        # guard_deg=0 かつ source 方位が grid 外にある条件では、角度比較だけでは
        # source mask が空になり得る。source を消す評価にしないため最近傍 beam を必ず保護する。
        source_mask[source_indices] = True

    # 角度 guard は source ごとの局所幅が不均一になり得るため、dataclass の
    # guard_beam_count は最近傍 source index だけを確実に含む 0 として記録する。
    return SourceSectorMask(
        source_mask=source_mask,
        source_beam_indices=source_indices,
        guard_beam_count=0,
        mask_type=str(mask_type),
    )


def detect_source_beam_indices_from_level_peaks(
    levels_db: NDArray[Any],
    max_source_count: int,
    guard_beam_count: int,
    threshold_db_below_peak: float,
) -> IntArray:
    """固定整相 before レベルから detected source beam index を選ぶ。

    Args:
        levels_db: beam レベル。shape は `[n_beam]`、単位は呼び出し側の `dB re ...`。
        max_source_count: 検出する source 数の上限。単位は count。
        guard_beam_count: 既に選んだ peak の近傍を次候補から除外する幅。単位は beam。
        threshold_db_below_peak: global peak から何 dB 下まで source と見なすか。単位は dB。

    Returns:
        detected source beam index。shape は `[n_detected_source]`。

    Raises:
        ValueError: レベル列や検出条件が不正な場合。

    境界条件:
        detected mask は実運用に近いが、弱 source を見落とす可能性がある。
        そのため、ここでは候補 peak が閾値未満になった時点で検出を止め、oracle mask と別に評価する。
    """
    levels = np.asarray(levels_db, dtype=np.float64)
    require(levels.ndim == 1, "levels_db must have shape (n_beam,).")
    require(levels.size > 0, "levels_db must not be empty.")
    require(bool(np.all(np.isfinite(levels))), "levels_db must contain finite values.")
    require_positive_int("max_source_count", int(max_source_count))
    require_non_negative_float("guard_beam_count", float(guard_beam_count))
    require_non_negative_float("threshold_db_below_peak", float(threshold_db_below_peak))

    candidate_mask = np.ones(levels.size, dtype=np.bool_)
    detected_indices: list[int] = []
    global_peak_db = float(np.max(levels))
    for _ in range(int(max_source_count)):
        if not bool(np.any(candidate_mask)):
            break
        masked_levels = np.where(candidate_mask, levels, -np.inf)
        peak_index = int(np.argmax(masked_levels))
        peak_level_db = float(masked_levels[peak_index])
        if peak_level_db < global_peak_db - float(threshold_db_below_peak):
            break
        detected_indices.append(peak_index)

        start = max(0, peak_index - int(guard_beam_count))
        stop = min(levels.size, peak_index + int(guard_beam_count) + 1)
        # 同じ source mainlobe を複数 source として数えないよう、
        # 選択済み peak の guard 領域を次の検出候補から外す。
        candidate_mask[start:stop] = False

    return np.asarray(detected_indices, dtype=np.int64)


def calculate_abf_like_non_source_metrics(
    *,
    axis_azimuth_deg: NDArray[Any],
    before_levels_db: NDArray[Any],
    after_levels_db: NDArray[Any],
    source_sector_mask: SourceSectorMask,
    level_unit_label: str,
    false_peak_threshold_db_below_source_peak: float = 13.0,
    worsening_before_gate_db_below_source_peak: float = 60.0,
    worsening_after_gate_db_below_source_peak: float = 40.0,
) -> AbfLikeNonSourceMetrics:
    """source 保護と non-source 抑圧の ABF-like 指標を計算する。

    Args:
        axis_azimuth_deg: beam 方位軸。shape は `[n_beam]`、単位は deg。
        before_levels_db: 固定整相 baseline の beam レベル。shape は `[n_beam]`。
        after_levels_db: 候補方式の beam レベル。shape は `[n_beam]`。
        source_sector_mask: source / non-source mask 定義。
        level_unit_label: レベル値の基準。例: `dB re input RMS`。
        false_peak_threshold_db_below_source_peak: source peak から何 dB 下までを
            false peak 候補として数えるか。単位は dB。
        worsening_before_gate_db_below_source_peak: before が source peak からこの範囲内なら
            local worsening 評価対象に入れる。単位は dB。
        worsening_after_gate_db_below_source_peak: after が source peak からこの範囲内なら
            local worsening 評価対象に入れる。単位は dB。

    Returns:
        ABF-like 採否用 metric。

    Raises:
        ValueError: 入力 shape、mask、単位ラベル、non-source 領域が不正な場合。

    境界条件:
        dB レベルは RMS 振幅の dB20 として扱い、integrated level だけは power 和へ戻してから
        dB10 に戻す。before/after で同じ mask を使うため、和と平均の違いは差分値に影響しない。
        gated worsening では、before が深い谷で after も表示上十分低い点を過大な悪化として数えない。
    """
    azimuths = np.asarray(axis_azimuth_deg, dtype=np.float64)
    before_levels = np.asarray(before_levels_db, dtype=np.float64)
    after_levels = np.asarray(after_levels_db, dtype=np.float64)
    source_mask = np.asarray(source_sector_mask.source_mask, dtype=np.bool_)
    non_source_mask = source_sector_mask.non_source_mask

    require(azimuths.ndim == 1, "axis_azimuth_deg must have shape (n_beam,).")
    require(
        before_levels.shape == azimuths.shape,
        "before_levels_db must match axis_azimuth_deg shape.",
    )
    require(
        after_levels.shape == azimuths.shape,
        "after_levels_db must match axis_azimuth_deg shape.",
    )
    require(
        source_mask.shape == azimuths.shape,
        "source_sector_mask must match axis_azimuth_deg shape.",
    )
    require(bool(np.all(np.isfinite(azimuths))), "axis_azimuth_deg must contain finite values.")
    require(
        bool(np.all(np.isfinite(before_levels))),
        "before_levels_db must contain finite values.",
    )
    require(bool(np.all(np.isfinite(after_levels))), "after_levels_db must contain finite values.")
    require(bool(np.any(non_source_mask)), "non-source sector must contain at least one beam.")
    require(
        str(level_unit_label).startswith("dB re "),
        "level_unit_label must state an explicit dB reference.",
    )
    require_non_negative_float(
        "false_peak_threshold_db_below_source_peak",
        float(false_peak_threshold_db_below_source_peak),
    )
    require_non_negative_float(
        "worsening_before_gate_db_below_source_peak",
        float(worsening_before_gate_db_below_source_peak),
    )
    require_non_negative_float(
        "worsening_after_gate_db_below_source_peak",
        float(worsening_after_gate_db_below_source_peak),
    )

    before_source_peaks: list[float] = []
    after_source_peaks: list[float] = []
    source_azimuth_errors_deg: list[float] = []
    source_regions = source_sector_mask.source_region_slices()
    for source_index, source_region in zip(
        source_sector_mask.source_beam_indices.tolist(),
        source_regions,
    ):
        before_region = before_levels[source_region]
        after_region = after_levels[source_region]
        region_azimuths = azimuths[source_region]

        before_source_peaks.append(float(np.max(before_region)))
        after_source_peaks.append(float(np.max(after_region)))
        after_peak_local_index = int(np.argmax(after_region))
        # source 方位は nearest beam center で保持している。
        # after peak が source guard 内で動いても、中心からのずれを
        # source 保護の方位誤差として読む。
        source_azimuth_error_deg = abs(
            region_azimuths[after_peak_local_index] - azimuths[int(source_index)]
        )
        source_azimuth_errors_deg.append(float(source_azimuth_error_deg))

    before_source_peak_array = np.asarray(before_source_peaks, dtype=np.float64)
    after_source_peak_array = np.asarray(after_source_peaks, dtype=np.float64)
    source_peak_delta = after_source_peak_array - before_source_peak_array
    max_abs_source_peak_delta_db = float(np.max(np.abs(source_peak_delta)))
    source_azimuth_error_array = np.asarray(source_azimuth_errors_deg, dtype=np.float64)
    max_source_azimuth_error_deg = float(np.max(source_azimuth_error_array))

    before_non_source = before_levels[non_source_mask]
    after_non_source = after_levels[non_source_mask]
    non_source_global_peak_delta_db = float(np.max(after_non_source) - np.max(before_non_source))
    before_non_source_p95_db = float(np.percentile(before_non_source, 95.0))
    after_non_source_p95_db = float(np.percentile(after_non_source, 95.0))
    non_source_p95_level_delta_db = after_non_source_p95_db - before_non_source_p95_db
    before_non_source_p99_db = float(np.percentile(before_non_source, 99.0))
    after_non_source_p99_db = float(np.percentile(after_non_source, 99.0))
    non_source_p99_level_delta_db = after_non_source_p99_db - before_non_source_p99_db
    non_source_integrated_level_delta_db = float(
        _integrated_level_db(after_non_source) - _integrated_level_db(before_non_source)
    )

    before_source_global_peak_db = float(np.max(before_source_peak_array))
    after_source_global_peak_db = float(np.max(after_source_peak_array))
    before_margin_db = before_source_global_peak_db - float(np.max(before_non_source))
    after_margin_db = after_source_global_peak_db - float(np.max(after_non_source))
    source_to_non_source_margin_delta_db = float(after_margin_db - before_margin_db)

    false_peak_count_before = _count_non_source_false_peaks(
        levels_db=before_levels,
        source_mask=source_mask,
        source_peak_db=before_source_global_peak_db,
        threshold_db_below_source_peak=float(false_peak_threshold_db_below_source_peak),
    )
    false_peak_count_after = _count_non_source_false_peaks(
        levels_db=after_levels,
        source_mask=source_mask,
        source_peak_db=after_source_global_peak_db,
        threshold_db_below_source_peak=float(false_peak_threshold_db_below_source_peak),
    )

    delta_non_source = after_non_source - before_non_source
    gated_mask = np.logical_or(
        before_non_source
        > before_source_global_peak_db - float(worsening_before_gate_db_below_source_peak),
        after_non_source
        > after_source_global_peak_db - float(worsening_after_gate_db_below_source_peak),
    )
    non_source_indices = np.flatnonzero(non_source_mask)
    if bool(np.any(gated_mask)):
        gated_indices = np.flatnonzero(gated_mask)
        local_index = int(gated_indices[int(np.argmax(delta_non_source[gated_mask]))])
        max_local_worsening_db_gated = float(delta_non_source[local_index])
        max_local_worsening_azimuth_deg: float | None = float(
            azimuths[int(non_source_indices[local_index])]
        )
    else:
        # source peak から十分低い谷だけが変化した場合、表示上の false peak ではないため
        # 最大悪化 0 dB として扱い、方位は該当なしにする。
        max_local_worsening_db_gated = 0.0
        max_local_worsening_azimuth_deg = None

    return AbfLikeNonSourceMetrics(
        max_abs_source_peak_delta_db=max_abs_source_peak_delta_db,
        max_source_azimuth_error_deg=max_source_azimuth_error_deg,
        non_source_global_peak_delta_db=non_source_global_peak_delta_db,
        non_source_p95_level_delta_db=non_source_p95_level_delta_db,
        non_source_p99_level_delta_db=non_source_p99_level_delta_db,
        non_source_integrated_level_delta_db=non_source_integrated_level_delta_db,
        source_to_non_source_margin_delta_db=source_to_non_source_margin_delta_db,
        false_peak_count_before=int(false_peak_count_before),
        false_peak_count_after=int(false_peak_count_after),
        false_peak_count_delta=int(false_peak_count_after - false_peak_count_before),
        max_local_worsening_db_gated=max_local_worsening_db_gated,
        max_local_worsening_azimuth_deg=max_local_worsening_azimuth_deg,
        level_unit_label=str(level_unit_label),
    )


def judge_abf_like_non_source_metrics(
    metrics: AbfLikeNonSourceMetrics,
    *,
    realtime_factor: float | None = None,
    nan_inf_count: int = 0,
    condition_number: float | None = None,
) -> AbfLikeMetricDecision:
    """ABF-like 指標を資料の閾値に沿って pass/hold/fail 判定する。

    Args:
        metrics: source 保護と non-source 抑圧の指標。
        realtime_factor: 実時間係数。`None` の場合は runtime 判定を行わない。
        nan_inf_count: 出力または summary 内の NaN/inf 個数。
        condition_number: SLC 等の loaded covariance 条件数。`None` の場合は判定しない。

    Returns:
        `pass`, `hold`, `fail` の status と理由。

    Raises:
        ValueError: runtime や健全性指標が不正な場合。

    境界条件:
        fail 条件が 1 つでもあれば fail とする。fail がなく hold 条件が残る場合は hold とし、
        すべての主要条件を満たした場合だけ pass とする。
    """
    require_non_negative_float("nan_inf_count", float(nan_inf_count))
    if realtime_factor is not None:
        require_non_negative_float("realtime_factor", float(realtime_factor))
    if condition_number is not None:
        require_non_negative_float("condition_number", float(condition_number))

    failure_reasons: list[str] = []
    hold_reasons: list[str] = []

    if metrics.max_abs_source_peak_delta_db > 1.0:
        failure_reasons.append("source_peak_delta_fail")
    elif metrics.max_abs_source_peak_delta_db > 0.5:
        hold_reasons.append("source_peak_delta_hold")

    if metrics.non_source_global_peak_delta_db > 0.5:
        failure_reasons.append("non_source_global_peak_worse")
    elif metrics.non_source_global_peak_delta_db > -1.0:
        hold_reasons.append("non_source_global_peak_not_pass")

    if metrics.non_source_p95_level_delta_db > 0.0:
        failure_reasons.append("non_source_p95_worse")
    elif metrics.non_source_p95_level_delta_db > -1.0:
        hold_reasons.append("non_source_p95_not_pass")

    if metrics.non_source_integrated_level_delta_db > 0.0:
        failure_reasons.append("non_source_integrated_worse")
    elif metrics.non_source_integrated_level_delta_db > -1.0:
        hold_reasons.append("non_source_integrated_not_pass")

    if metrics.source_to_non_source_margin_delta_db < 0.0:
        failure_reasons.append("source_to_non_source_margin_worse")
    elif metrics.source_to_non_source_margin_delta_db < 0.5:
        hold_reasons.append("source_to_non_source_margin_not_pass")

    if metrics.false_peak_count_delta > 0:
        failure_reasons.append("false_peak_count_increased")

    if metrics.max_local_worsening_db_gated > 6.0:
        failure_reasons.append("max_local_worsening_fail")
    elif metrics.max_local_worsening_db_gated > 3.0:
        hold_reasons.append("max_local_worsening_hold")

    if int(nan_inf_count) > 0:
        failure_reasons.append("nan_inf_detected")

    if realtime_factor is not None:
        if float(realtime_factor) > 1.0:
            failure_reasons.append("runtime_factor_fail")
        elif float(realtime_factor) > 0.7:
            hold_reasons.append("runtime_factor_hold")

    if condition_number is not None:
        if float(condition_number) > 1.0e8:
            failure_reasons.append("condition_number_fail")
        elif float(condition_number) > 1.0e6:
            hold_reasons.append("condition_number_hold")

    status = "fail" if failure_reasons else "hold" if hold_reasons else "pass"
    return AbfLikeMetricDecision(
        status=status,
        failure_reasons=tuple(failure_reasons),
        hold_reasons=tuple(hold_reasons),
    )


def _integrated_level_db(levels_db: FloatArray) -> float:
    """RMS dB20 レベル列を power 和の dB10 へ変換する。"""
    require(levels_db.ndim == 1 and levels_db.size > 0, "levels_db must be a non-empty 1-D array.")
    # level_db が RMS 振幅比 dB20 なので、power 比は 10 ** (level_db / 10) である。
    # sector 全体のエネルギー包絡を読むため power 和を dB10 に戻す。
    power_sum = float(np.sum(np.power(10.0, levels_db / 10.0)))
    return float(10.0 * np.log10(max(power_sum, np.finfo(np.float64).tiny)))


def _count_non_source_false_peaks(
    *,
    levels_db: FloatArray,
    source_mask: BoolArray,
    source_peak_db: float,
    threshold_db_below_source_peak: float,
) -> int:
    """non-source sector の局所 peak 数を数える。"""
    threshold_db = float(source_peak_db) - float(threshold_db_below_source_peak)
    peak_count = 0
    n_beam = int(levels_db.size)
    for beam_index in range(n_beam):
        if bool(source_mask[beam_index]) or bool(levels_db[beam_index] < threshold_db):
            continue

        left_level = -np.inf if beam_index == 0 else float(levels_db[beam_index - 1])
        right_level = -np.inf if beam_index == n_beam - 1 else float(levels_db[beam_index + 1])
        # source mask 外で周囲より高い局所 peak だけを false peak として数える。
        # 端 beam は存在する片側だけで判定し、source guard 内の mainlobe は数えない。
        is_local_peak = (
            float(levels_db[beam_index]) >= left_level
            and float(levels_db[beam_index]) >= right_level
        )
        if is_local_peak:
            peak_count += 1
    return int(peak_count)


__all__ = [
    "AbfLikeMetricDecision",
    "AbfLikeNonSourceMetrics",
    "SourceSectorMask",
    "build_source_sector_mask",
    "build_source_sector_mask_from_azimuths",
    "calculate_abf_like_non_source_metrics",
    "detect_source_beam_indices_from_level_peaks",
    "judge_abf_like_non_source_metrics",
]
