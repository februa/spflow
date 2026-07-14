"""target-only、noise-only、mixed BLを成分別に評価する観測部品を実装する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float
from ..level_conversion import LevelConverter, level_10log10_power


@dataclass(frozen=True)
class BlLocalPeak:
    """BL上の局所peakを保持する。

    Attributes:
        azimuth_deg: peak方位。単位はdeg。
        level_db: 絶対level。単位は`level_reference_label`が示すdB reference。
        level_db_re_mainlobe_peak: mainlobe peakに対する相対level。単位はdB。

    peakの検出結果だけを保持し、sidelobeまたはgrating lobeの採否判定は責務に含めない。
    """

    azimuth_deg: float
    level_db: float
    level_db_re_mainlobe_peak: float

    def as_dict(self) -> dict[str, float]:
        """JSONへ保存可能なPython scalar辞書へ変換する。"""
        return {
            "azimuth_deg": float(self.azimuth_deg),
            "level_db": float(self.level_db),
            "level_db_re_mainlobe_peak": float(self.level_db_re_mainlobe_peak),
        }


@dataclass(frozen=True)
class TargetOnlyBlMetrics:
    """target-only BLのmainlobe、副極、grating-lobe候補を保持する。

    入力SLとsource truthに対する誤差、first-null境界、第一副極を出力する。
    noise floor、mixed source visibility、方式の採否判定は責務に含めない。
    """

    peak_azimuth_deg: float
    peak_azimuth_error_deg: float
    peak_level_db: float
    peak_level_error_db: float
    width_3db_deg: float
    left_first_null_azimuth_deg: float | None
    right_first_null_azimuth_deg: float | None
    first_null_width_deg: float | None
    left_first_sidelobe: BlLocalPeak | None
    right_first_sidelobe: BlLocalPeak | None
    maximum_sidelobe: BlLocalPeak | None
    grating_lobe_candidates: tuple[BlLocalPeak, ...]
    level_reference_label: str

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存可能な辞書へ変換する。"""
        return {
            "peak_azimuth_deg": float(self.peak_azimuth_deg),
            "peak_azimuth_error_deg": float(self.peak_azimuth_error_deg),
            "peak_level_db": float(self.peak_level_db),
            "peak_level_error_db": float(self.peak_level_error_db),
            "width_3db_deg": float(self.width_3db_deg),
            "left_first_null_azimuth_deg": self.left_first_null_azimuth_deg,
            "right_first_null_azimuth_deg": self.right_first_null_azimuth_deg,
            "first_null_width_deg": self.first_null_width_deg,
            "left_first_sidelobe": (
                None if self.left_first_sidelobe is None else self.left_first_sidelobe.as_dict()
            ),
            "right_first_sidelobe": (
                None if self.right_first_sidelobe is None else self.right_first_sidelobe.as_dict()
            ),
            "maximum_sidelobe": (
                None if self.maximum_sidelobe is None else self.maximum_sidelobe.as_dict()
            ),
            "grating_lobe_candidates": [peak.as_dict() for peak in self.grating_lobe_candidates],
            "level_reference_label": self.level_reference_label,
        }


@dataclass(frozen=True)
class NoiseOnlyBlMetrics:
    """noise-only BLと`w^H R_n w`予測値の対応を保持する。

    Attributes:
        observed_level_db: 観測noise-only BL。shapeは`[n_beam]`。
        predicted_level_db: covarianceとweightから予測したBL。shapeは`[n_beam]`。
        prediction_error_db: 観測値−予測値。shapeは`[n_beam]`。
        predicted_array_gain_db: 入力1 channel noise powerに対する予測低減量。
            shapeは`[n_beam]`、正値がSNR改善を表す。
        median_prediction_error_db: 全beam誤差のmedian。
        maximum_absolute_prediction_error_db: 全beamの最大絶対誤差。
        level_reference_label: noise levelのdB reference。

    方式間比較や合否閾値は責務に含めない。
    """

    observed_level_db: NDArray[np.float64]
    predicted_level_db: NDArray[np.float64]
    prediction_error_db: NDArray[np.float64]
    predicted_array_gain_db: NDArray[np.float64]
    median_prediction_error_db: float
    maximum_absolute_prediction_error_db: float
    level_reference_label: str

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存可能な辞書へ変換する。"""
        return {
            "observed_level_db": self.observed_level_db.tolist(),
            "predicted_level_db": self.predicted_level_db.tolist(),
            "prediction_error_db": self.prediction_error_db.tolist(),
            "predicted_array_gain_db": self.predicted_array_gain_db.tolist(),
            "median_prediction_error_db": float(self.median_prediction_error_db),
            "maximum_absolute_prediction_error_db": float(
                self.maximum_absolute_prediction_error_db
            ),
            "level_reference_label": self.level_reference_label,
        }


@dataclass(frozen=True)
class MixedBlConsistency:
    """mixed BLとtarget-only＋noise-only power和の整合を保持する。"""

    expected_mixed_level_db: NDArray[np.float64]
    observed_mixed_level_db: NDArray[np.float64]
    consistency_error_db: NDArray[np.float64]
    median_consistency_error_db: float
    maximum_absolute_consistency_error_db: float
    level_reference_label: str

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存可能な辞書へ変換する。"""
        return {
            "expected_mixed_level_db": self.expected_mixed_level_db.tolist(),
            "observed_mixed_level_db": self.observed_mixed_level_db.tolist(),
            "consistency_error_db": self.consistency_error_db.tolist(),
            "median_consistency_error_db": float(self.median_consistency_error_db),
            "maximum_absolute_consistency_error_db": float(
                self.maximum_absolute_consistency_error_db
            ),
            "level_reference_label": self.level_reference_label,
        }


@dataclass(frozen=True)
class BlComponentEvaluation:
    """target-only、noise-only、mixedのBL評価結果を一つの固定型で保持する。"""

    target_only: TargetOnlyBlMetrics
    noise_only: NoiseOnlyBlMetrics | None = None
    mixed: MixedBlConsistency | None = None

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存可能な成分別辞書へ変換する。"""
        return {
            "target_only": self.target_only.as_dict(),
            "noise_only": None if self.noise_only is None else self.noise_only.as_dict(),
            "mixed": None if self.mixed is None else self.mixed.as_dict(),
        }


def _validate_bl_axis_and_levels(
    axis_azimuth_deg: NDArray[Any],
    level_db: NDArray[Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """BL方位軸とlevel列を検証しfloat64へ正規化する。"""
    azimuth = np.asarray(axis_azimuth_deg, dtype=np.float64)
    levels = np.asarray(level_db, dtype=np.float64)
    require(azimuth.ndim == 1 and azimuth.size >= 3, "axis_azimuth_deg needs >= 3 beams.")
    require(levels.shape == azimuth.shape, "level_db shape must match axis_azimuth_deg.")
    require(bool(np.all(np.isfinite(azimuth))), "axis_azimuth_deg must be finite.")
    require(bool(np.all(np.isfinite(levels))), "level_db must be finite.")
    require(bool(np.all(np.diff(azimuth) > 0.0)), "axis_azimuth_deg must increase.")
    return azimuth, levels


def _local_minimum_indices(levels: NDArray[np.float64]) -> NDArray[np.int64]:
    """両隣以下で少なくとも片側より低い内部sample indexを返す。"""
    center = levels[1:-1]
    mask = (center <= levels[:-2]) & (center <= levels[2:])
    mask &= (center < levels[:-2]) | (center < levels[2:])
    return np.asarray(np.flatnonzero(mask) + 1, dtype=np.int64)


def _local_maximum_indices(levels: NDArray[np.float64]) -> NDArray[np.int64]:
    """両隣以上で少なくとも片側より高い内部sample indexを返す。"""
    center = levels[1:-1]
    mask = (center >= levels[:-2]) & (center >= levels[2:])
    mask &= (center > levels[:-2]) | (center > levels[2:])
    return np.asarray(np.flatnonzero(mask) + 1, dtype=np.int64)


def _peak_observation(
    index: int,
    azimuth: NDArray[np.float64],
    levels: NDArray[np.float64],
    mainlobe_peak_level_db: float,
) -> BlLocalPeak:
    """局所peak indexを公開観測型へ変換する。"""
    return BlLocalPeak(
        azimuth_deg=float(azimuth[index]),
        level_db=float(levels[index]),
        level_db_re_mainlobe_peak=float(levels[index] - mainlobe_peak_level_db),
    )


def evaluate_target_only_bl(
    axis_azimuth_deg: NDArray[Any],
    target_only_level_db: NDArray[Any],
    *,
    source_azimuth_deg: float,
    source_level_db: float,
    level_reference_label: str,
    grating_lobe_candidate_threshold_db_re_peak: float = -3.0,
) -> TargetOnlyBlMetrics:
    """target-only BLからmainlobe、first null、副極、grating候補を抽出する。

    Args:
        axis_azimuth_deg: waiting-beam方位。shapeは`[n_beam]`、単位はdeg。
        target_only_level_db: target-only BL。shapeは`[n_beam]`。
        source_azimuth_deg: source truth方位。単位はdeg。
        source_level_db: 入力source RMS level。単位は`level_reference_label`と同じ。
        level_reference_label: `dB re input RMS`などの基準量。
        grating_lobe_candidate_threshold_db_re_peak: mainlobe外局所peakをgrating候補とする
            相対level下限。単位は`dB re mainlobe peak`。

    Returns:
        target-only BL観測量。合否statusは含まない。

    Raises:
        ValueError: shape、有限性、label、thresholdが不正な場合。

    境界条件:
        first nullはglobal peakの左右で最も近い局所極小とする。端fireや粗いgridで
        局所極小が存在しない側は`None`とし、推測値で補わない。
    """
    azimuth, levels = _validate_bl_axis_and_levels(axis_azimuth_deg, target_only_level_db)
    source_azimuth = float(source_azimuth_deg)
    source_level = float(source_level_db)
    threshold = float(grating_lobe_candidate_threshold_db_re_peak)
    require(bool(np.isfinite(source_azimuth)), "source_azimuth_deg must be finite.")
    require(bool(np.isfinite(source_level)), "source_level_db must be finite.")
    require(bool(np.isfinite(threshold)) and threshold <= 0.0, "grating threshold must be <= 0.")
    require(bool(str(level_reference_label).strip()), "level_reference_label must not be empty.")

    peak_index = int(np.argmax(levels))
    peak_level = float(levels[peak_index])
    minima = _local_minimum_indices(levels)
    left_minima = minima[minima < peak_index]
    right_minima = minima[minima > peak_index]
    left_null_index = None if left_minima.size == 0 else int(left_minima[-1])
    right_null_index = None if right_minima.size == 0 else int(right_minima[0])

    above_3db = levels >= peak_level - 3.0
    left_3db = peak_index
    while left_3db > 0 and bool(above_3db[left_3db - 1]):
        left_3db -= 1
    right_3db = peak_index
    while right_3db + 1 < levels.size and bool(above_3db[right_3db + 1]):
        right_3db += 1

    maxima = _local_maximum_indices(levels)
    left_sidelobe_indices = (
        np.empty(0, dtype=np.int64) if left_null_index is None else maxima[maxima < left_null_index]
    )
    right_sidelobe_indices = (
        np.empty(0, dtype=np.int64)
        if right_null_index is None
        else maxima[maxima > right_null_index]
    )
    left_first_index = None if left_sidelobe_indices.size == 0 else int(left_sidelobe_indices[-1])
    right_first_index = None if right_sidelobe_indices.size == 0 else int(right_sidelobe_indices[0])
    sidelobe_indices = np.concatenate([left_sidelobe_indices, right_sidelobe_indices])
    maximum_sidelobe_index = (
        None
        if sidelobe_indices.size == 0
        else int(sidelobe_indices[int(np.argmax(levels[sidelobe_indices]))])
    )
    grating_indices = sidelobe_indices[levels[sidelobe_indices] - peak_level >= threshold]

    first_null_width = (
        None
        if left_null_index is None or right_null_index is None
        else float(azimuth[right_null_index] - azimuth[left_null_index])
    )
    return TargetOnlyBlMetrics(
        peak_azimuth_deg=float(azimuth[peak_index]),
        peak_azimuth_error_deg=float(azimuth[peak_index] - source_azimuth),
        peak_level_db=peak_level,
        peak_level_error_db=peak_level - source_level,
        width_3db_deg=float(azimuth[right_3db] - azimuth[left_3db]),
        left_first_null_azimuth_deg=(
            None if left_null_index is None else float(azimuth[left_null_index])
        ),
        right_first_null_azimuth_deg=(
            None if right_null_index is None else float(azimuth[right_null_index])
        ),
        first_null_width_deg=first_null_width,
        left_first_sidelobe=(
            None
            if left_first_index is None
            else _peak_observation(left_first_index, azimuth, levels, peak_level)
        ),
        right_first_sidelobe=(
            None
            if right_first_index is None
            else _peak_observation(right_first_index, azimuth, levels, peak_level)
        ),
        maximum_sidelobe=(
            None
            if maximum_sidelobe_index is None
            else _peak_observation(maximum_sidelobe_index, azimuth, levels, peak_level)
        ),
        grating_lobe_candidates=tuple(
            _peak_observation(int(index), azimuth, levels, peak_level)
            for index in grating_indices.tolist()
        ),
        level_reference_label=str(level_reference_label),
    )

def evaluate_noise_only_bl(
    observed_noise_level_db: NDArray[Any],
    weights: NDArray[Any],
    noise_covariance: NDArray[Any],
    *,
    input_channel_noise_power: float,
    reference_rms: float,
    level_reference_label: str,
) -> NoiseOnlyBlMetrics:
    """noise-only BLを`w^H R_n w`予測値と比較する。

    Args:
        observed_noise_level_db: 観測noise-only BL。shapeは`[n_beam]`。
        weights: beamforming重み。shapeは`[n_channel,n_beam]`。
        noise_covariance: 対象帯域のchannel noise covariance。
            shapeは`[n_channel,n_channel]`、単位は振幅二乗。
        input_channel_noise_power: 1 channel入力noise power。array gain基準、単位は振幅二乗。
        reference_rms: 0 dBに対応するRMS amplitude。
        level_reference_label: noise levelのdB reference。

    Returns:
        観測値、理論値、誤差、理論array gain。

    Raises:
        ValueError: shape、有限性、非負power、Hermitian条件が不正な場合。
    """
    observed = np.asarray(observed_noise_level_db, dtype=np.float64)
    beam_weights = np.asarray(weights, dtype=np.complex128)
    covariance = np.asarray(noise_covariance, dtype=np.complex128)
    input_power = float(input_channel_noise_power)
    reference = float(reference_rms)
    require(observed.ndim == 1 and observed.size > 0, "observed noise BL must be 1-D.")
    require(beam_weights.ndim == 2, "weights must have shape (n_channel, n_beam).")
    require(beam_weights.shape[1] == observed.size, "weights beam axis must match observed BL.")
    require(
        covariance.shape == (beam_weights.shape[0], beam_weights.shape[0]),
        "noise_covariance shape must match channel count.",
    )
    require(bool(np.all(np.isfinite(observed))), "observed noise BL must be finite.")
    require(bool(np.all(np.isfinite(beam_weights))), "weights must be finite.")
    require(bool(np.all(np.isfinite(covariance))), "noise_covariance must be finite.")
    require(
        bool(np.allclose(covariance, covariance.conj().T)), "noise_covariance must be Hermitian."
    )
    require_positive_float("input_channel_noise_power", input_power)
    require_positive_float("reference_rms", reference)
    require(bool(str(level_reference_label).strip()), "level_reference_label must not be empty.")

    # predicted_power[beam] = w[:,beam]^H R_n w[:,beam]。
    predicted_power = np.real(
        np.einsum("cb,cd,db->b", np.conjugate(beam_weights), covariance, beam_weights)
    )
    require(bool(np.all(predicted_power > 0.0)), "predicted noise power must be positive.")
    power_definition = level_10log10_power(
        reference_power=reference**2,
        reference_label=str(level_reference_label).removeprefix("dB re "),
        physical_quantity="beam output mean-square",
    )
    power_converter = LevelConverter.for_definition(power_definition)
    predicted_level = power_converter.output_to_level(predicted_power)
    error = observed - predicted_level
    # array gainは同じpower definitionで得た入力levelと出力levelの差なので、
    # reference値に依存せず10log10(P_input/P_output)と一致する。
    input_level = power_converter.output_to_level(input_power)
    array_gain = input_level - predicted_level
    return NoiseOnlyBlMetrics(
        observed_level_db=observed.copy(),
        predicted_level_db=np.asarray(predicted_level, dtype=np.float64),
        prediction_error_db=np.asarray(error, dtype=np.float64),
        predicted_array_gain_db=np.asarray(array_gain, dtype=np.float64),
        median_prediction_error_db=float(np.median(error)),
        maximum_absolute_prediction_error_db=float(np.max(np.abs(error))),
        level_reference_label=str(level_reference_label),
    )


def evaluate_mixed_bl_consistency(
    target_only_level_db: NDArray[Any],
    noise_only_level_db: NDArray[Any],
    observed_mixed_level_db: NDArray[Any],
    *,
    level_reference_label: str,
) -> MixedBlConsistency:
    """mixed BLが無相関target powerとnoise powerの和に一致するか評価する。

    Args:
        target_only_level_db: target-only BL。shapeは`[n_beam]`。
        noise_only_level_db: noise-only BL。shapeは`[n_beam]`。
        observed_mixed_level_db: target+noiseの観測BL。shapeは`[n_beam]`。
        level_reference_label: 3配列で共有するdB reference。

    Returns:
        power和から求めた期待mixed levelと観測誤差。

    Raises:
        ValueError: shape、有限性、labelが不正な場合。
    """
    target = np.asarray(target_only_level_db, dtype=np.float64)
    noise = np.asarray(noise_only_level_db, dtype=np.float64)
    mixed = np.asarray(observed_mixed_level_db, dtype=np.float64)
    require(target.ndim == 1 and target.size > 0, "target-only BL must be 1-D.")
    require(noise.shape == target.shape, "noise-only BL shape must match target-only BL.")
    require(mixed.shape == target.shape, "mixed BL shape must match target-only BL.")
    require(bool(np.all(np.isfinite(target))), "target-only BL must be finite.")
    require(bool(np.all(np.isfinite(noise))), "noise-only BL must be finite.")
    require(bool(np.all(np.isfinite(mixed))), "mixed BL must be finite.")
    require(bool(str(level_reference_label).strip()), "level_reference_label must not be empty.")

    power_definition = level_10log10_power(
        reference_power=1.0,
        reference_label=str(level_reference_label).removeprefix("dB re "),
        physical_quantity="beam output mean-square ratio",
    )
    power_converter = LevelConverter.for_definition(power_definition)
    # targetとnoiseは無相関という評価前提なので、共有referenceへ戻したmean-squareを加算する。
    expected_power = power_converter.input_to_linear(target) + power_converter.input_to_linear(
        noise
    )
    expected_level = power_converter.output_to_level(expected_power)
    error = mixed - expected_level
    return MixedBlConsistency(
        expected_mixed_level_db=np.asarray(expected_level, dtype=np.float64),
        observed_mixed_level_db=mixed.copy(),
        consistency_error_db=np.asarray(error, dtype=np.float64),
        median_consistency_error_db=float(np.median(error)),
        maximum_absolute_consistency_error_db=float(np.max(np.abs(error))),
        level_reference_label=str(level_reference_label),
    )
