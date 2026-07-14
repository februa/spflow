"""軽量 ABF-like 非信号方位抑圧方式の横並び評価を実行する。

このスクリプトは、運用スパースアレイ、保存済み小数遅延 FIR、運用 shading を読み込み、
固定整相 baseline、A2 source-mask SLC、B1 FD/shading candidate を同じ
`ABF_like_non_source_suppression` 指標で比較する。

出力は次の 2 ファイルである。

- artifacts/beamforming/lightweight_abf_like_comparison/comparison_summary.csv
- artifacts/beamforming/lightweight_abf_like_comparison/comparison_report.md
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from spflow.beamforming import (
    SourceMaskNonSourceLeakageSubtractor,
    SourceMaskSlcConfig,
)
from spflow.beamforming.directions import make_directions
from spflow.beamforming.operational_shading import (
    OperationalShadingDefinition,
    _kaiser_bessel_channel_window,
    _sidelobe_distribution_metrics,
)
from spflow.beamforming.operational_sparse_array import OperationalSparseArrayDefinition
from spflow.beamforming.time_delay import (
    FractionalDelayAndSumBeamformer,
    FractionalDelayFilterBank,
)
from spflow.beamforming_evaluation import (
    SourceSectorMask,
    build_source_sector_mask_from_azimuths,
    calculate_abf_like_non_source_metrics,
    detect_source_beam_indices_from_level_peaks,
    judge_abf_like_non_source_metrics,
)

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]


OUTPUT_DIR = Path("artifacts/beamforming/lightweight_abf_like_comparison")
SUMMARY_CSV_PATH = OUTPUT_DIR / "comparison_summary.csv"
REPORT_MD_PATH = OUTPUT_DIR / "comparison_report.md"

ARRAY_DEFINITION_PATH = Path(
    "artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json"
)
SHADING_DEFINITION_PATH = Path(
    "artifacts/beamforming/operational_shading/operational_kaiser_bessel_shading_fs32768.json"
)
FRACTIONAL_DELAY_FILTER_BANK_PATH = Path(
    "artifacts/beamforming/fractional_delay_filter_bank_65x63.npz"
)

N_BEAM_AZ_REAL = 151
N_SAMPLE = 2048
TARGET_AZIMUTH_BASE_DEG = 90.0
TARGET_LEVEL_DB = 0.0
INTERFERER_LEVEL_DB = -6.0
LEVEL_UNIT_LABEL = "dB re input RMS"
DETECTED_SOURCE_THRESHOLD_DB_BELOW_PEAK = 12.0
SAMPLE_PER_DOF_MIN = 5.0
TAP_LEN = 1
CONDITION_NUMBER_LIMIT = 1.0e8


@dataclass(frozen=True)
class EvaluationCase:
    """Tier 条件 1 ケースを保持する。

    入力は target / interferer の方位・周波数・offgrid 条件であり、出力は
    CSV 行の case identifier と source 合成条件である。

    BL/FRAZ/BTR 図の生成や SLC 係数推定は責務に含めない。
    信号処理上は、複数 source 条件を同じ source mask 評価へ渡すための条件表現に位置づく。
    """

    tier: str
    case_id: str
    target_frequency_hz: float
    interferer_frequency_hz: float
    interferer_azimuth_base_deg: float
    offgrid_deg: float

    @property
    def target_azimuth_deg(self) -> float:
        """offgrid を加えた target 方位を返す。単位は deg。"""
        return float(TARGET_AZIMUTH_BASE_DEG + self.offgrid_deg)

    @property
    def interferer_azimuth_deg(self) -> float:
        """offgrid を加えた interferer 方位を返す。単位は deg。"""
        return float(self.interferer_azimuth_base_deg + self.offgrid_deg)


@dataclass(frozen=True)
class SourceGuardVariant:
    """source mask と source reference に使う guard 条件を保持する。"""

    label: str
    guard_half_width_deg: float

    @property
    def source_mask_width_deg(self) -> float:
        """左右 guard を合わせた source mask 幅を返す。単位は deg。"""
        return float(2.0 * self.guard_half_width_deg)


@dataclass(frozen=True)
class B1Candidate:
    """B1 FD/shading selection の候補条件を保持する。"""

    candidate_id: str
    beta: float | None
    uses_operational_shading: bool


@dataclass(frozen=True)
class FrequencyResponseCacheEntry:
    """特定周波数の active 配置と固定整相応答計算器を保持する。"""

    frequency_hz: float
    active_indices: IntArray
    active_positions_m: FloatArray
    beamformer: FractionalDelayAndSumBeamformer


@dataclass(frozen=True)
class SourceSpec:
    """beam-domain 合成に使う source 条件を保持する。"""

    label: str
    frequency_hz: float
    azimuth_deg: float
    level_db: float
    phase_rad: float


@dataclass(frozen=True)
class A2RunResult:
    """A2 source-mask SLC の raw/effective 出力と診断量を保持する。"""

    raw_output: ComplexArray
    effective_output: ComplexArray
    condition_number: float | None
    weight_norm: float | None
    fallback_required: bool
    fallback_reasons: tuple[str, ...]
    n_ref: int
    n_non_source: int
    capacity_block_size: int
    capacity_dof: int
    elapsed_sec: float


def _build_evaluation_cases() -> list[EvaluationCase]:
    """Tier 0 / Tier 1 の評価ケースを作る。

    Returns:
        評価ケース列。Tier 0 は代表条件、Tier 1 は周波数・方位 matrix である。
    """
    cases: list[EvaluationCase] = []
    for offgrid_deg in (0.0, 0.25, 0.5):
        cases.append(
            EvaluationCase(
                tier="tier0",
                case_id=f"tier0_tf10000_if8192_ia060_off{_format_offgrid(offgrid_deg)}",
                target_frequency_hz=10000.0,
                interferer_frequency_hz=8192.0,
                interferer_azimuth_base_deg=60.0,
                offgrid_deg=float(offgrid_deg),
            )
        )

    for target_frequency_hz in (6144.0, 8192.0, 10000.0):
        for interferer_frequency_hz in (6144.0, 8192.0, 10000.0):
            for interferer_azimuth_deg in (45.0, 60.0, 75.0, 105.0, 120.0, 150.0):
                for offgrid_deg in (0.0, 0.25, 0.5):
                    cases.append(
                        EvaluationCase(
                            tier="tier1",
                            case_id=(
                                f"tier1_tf{int(target_frequency_hz)}_"
                                f"if{int(interferer_frequency_hz)}_"
                                f"ia{int(interferer_azimuth_deg):03d}_"
                                f"off{_format_offgrid(offgrid_deg)}"
                            ),
                            target_frequency_hz=float(target_frequency_hz),
                            interferer_frequency_hz=float(interferer_frequency_hz),
                            interferer_azimuth_base_deg=float(interferer_azimuth_deg),
                            offgrid_deg=float(offgrid_deg),
                        )
                    )
    return cases


def _format_offgrid(offgrid_deg: float) -> str:
    """case id 用に offgrid 値を整数文字列へ変換する。"""
    return f"{int(round(float(offgrid_deg) * 100.0)):03d}"


def _rms_levels_db20(beam_output: NDArray[Any]) -> FloatArray:
    """beam ごとの RMS レベルを dB20 で返す。

    Args:
        beam_output: beam-domain 波形。shape は `[n_beam, n_sample]`。
            axis=0 が beam、axis=1 が時間 sample である。

    Returns:
        beam ごとの RMS レベル。shape は `[n_beam]`、単位は `dB re input RMS`。
    """
    signals = np.asarray(beam_output)
    if signals.ndim != 2:
        raise ValueError("beam_output must have shape (n_beam, n_sample).")

    # 複数 source の同時表示を評価するため、各 beam の複素 RMS 包絡線を使う。
    # SLC 係数は複素値になり得るため、abs^2 を power として扱う。
    rms = np.sqrt(np.mean(np.abs(signals) ** 2, axis=1))
    return np.asarray(
        20.0 * np.log10(np.maximum(rms, np.finfo(np.float64).tiny)),
        dtype=np.float64,
    )


def _complex_response_for_source(
    *,
    cache_entry: FrequencyResponseCacheEntry,
    channel_weights: FloatArray,
    source_azimuth_deg: float,
    sound_speed_m_s: float,
) -> ComplexArray:
    """任意 source 方位に対する小数遅延固定整相の複素 beam response を返す。

    Args:
        cache_entry: 評価周波数に対応する active 配置と beamformer。
        channel_weights: active channel shading。shape は `[n_active_ch]`。
        source_azimuth_deg: source 方位。単位は deg。
        sound_speed_m_s: 音速。単位は m/s。

    Returns:
        observation beam ごとの複素応答。shape は `[n_beam]`。
    """
    weights = np.asarray(channel_weights, dtype=np.float64)
    positions_m = np.asarray(cache_entry.active_positions_m, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != positions_m.shape[0]:
        raise ValueError("channel_weights must have shape (n_active_ch,).")
    if not bool(np.all(np.isfinite(weights))) or float(np.sum(weights)) <= 0.0:
        raise ValueError("channel_weights must be finite and have positive sum.")

    azimuth_rad = math.radians(float(source_azimuth_deg))
    source_direction = np.array(
        [math.cos(azimuth_rad), math.sin(azimuth_rad), 0.0],
        dtype=np.float64,
    )
    arrival_delay_sec = -(positions_m @ source_direction) / float(sound_speed_m_s)
    arrival_phase = np.exp(-1j * 2.0 * np.pi * float(cache_entry.frequency_hz) * arrival_delay_sec)
    steering_response = np.asarray(
        cache_entry.beamformer.steering_response(float(cache_entry.frequency_hz)),
        dtype=np.complex128,
    )

    # beam_output[beam] = Σ_ch w_ch * S_ch,beam * exp(-j 2π f tau_ch) / Σ_ch w_ch。
    # S_ch,beam は整数遅延と小数遅延 FIR の周波数応答を含むため、
    # time-domain 固定整相器の beam-domain 出力と同じ正規化で評価できる。
    weighted_arrival_phase = weights * arrival_phase
    return np.asarray(
        steering_response.T @ weighted_arrival_phase / float(np.sum(weights)), dtype=np.complex128
    )


def _synthesize_beam_output(
    *,
    source_specs: tuple[SourceSpec, SourceSpec],
    response_cache: dict[float, FrequencyResponseCacheEntry],
    weights_by_frequency: dict[float, FloatArray],
    fs_hz: float,
    sound_speed_m_s: float,
    n_sample: int,
) -> ComplexArray:
    """周波数別 fixed beam response から beam-domain mixed 波形を合成する。

    Args:
        source_specs: target / interferer の source 条件。
        response_cache: 周波数ごとの固定整相応答計算器。
        weights_by_frequency: 周波数ごとの active channel shading。shape は `[n_active_ch]`。
        fs_hz: サンプリング周波数。単位は Hz。
        sound_speed_m_s: 音速。単位は m/s。
        n_sample: 合成 sample 数。単位は sample。

    Returns:
        beam-domain 複素波形。shape は `[n_beam, n_sample]`。
    """
    if int(n_sample) <= 0:
        raise ValueError("n_sample must be positive.")

    time_axis_s = np.arange(int(n_sample), dtype=np.float64) / float(fs_hz)
    first_entry = next(iter(response_cache.values()))
    n_beam = int(first_entry.beamformer.delay_table.n_beam)
    beam_output = np.zeros((n_beam, int(n_sample)), dtype=np.complex128)

    for source_spec in source_specs:
        cache_entry = response_cache[float(source_spec.frequency_hz)]
        response = _complex_response_for_source(
            cache_entry=cache_entry,
            channel_weights=weights_by_frequency[float(source_spec.frequency_hz)],
            source_azimuth_deg=float(source_spec.azimuth_deg),
            sound_speed_m_s=float(sound_speed_m_s),
        )
        amplitude_rms = float(10.0 ** (float(source_spec.level_db) / 20.0))
        tone = amplitude_rms * np.exp(
            1j
            * (
                2.0 * np.pi * float(source_spec.frequency_hz) * time_axis_s
                + float(source_spec.phase_rad)
            )
        )
        # response[:, None] shape: [n_beam, 1]、tone[None, :] shape: [1, n_sample]。
        # broadcasting で各 source の beam response と時間波形を合成する。
        beam_output += response[:, np.newaxis] * tone[np.newaxis, :]

    return beam_output


def _effective_channel_count(channel_weights: FloatArray) -> float:
    """Kish の有効 channel 数 `N_eff = (sum w)^2 / sum(w^2)` を返す。"""
    weights = np.asarray(channel_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("channel_weights must have shape (n_ch,).")
    weight_sum = float(np.sum(weights))
    weight_power_sum = float(np.sum(weights**2))
    if weight_sum <= 0.0 or weight_power_sum <= 0.0:
        raise ValueError("channel_weights must have positive sum and power.")
    return float((weight_sum * weight_sum) / weight_power_sum)


def _weights_for_candidate(
    *,
    candidate: B1Candidate,
    frequency_hz: float,
    cache_entry: FrequencyResponseCacheEntry,
    shading_definition: OperationalShadingDefinition,
) -> FloatArray:
    """B1 candidate に対応する active channel weight を返す。"""
    active_indices = np.asarray(cache_entry.active_indices, dtype=np.int64)
    if candidate.uses_operational_shading:
        full_weights = shading_definition.coefficients_for_frequency(float(frequency_hz))
        return np.asarray(full_weights[active_indices], dtype=np.float64)
    if candidate.beta is None:
        raise ValueError("beta candidate requires beta value.")
    return np.asarray(
        _kaiser_bessel_channel_window(n_ch=int(active_indices.size), beta=float(candidate.beta)),
        dtype=np.float64,
    )


def _source_mask_for_case(
    *,
    axis_azimuth_deg: FloatArray,
    before_levels_db: FloatArray,
    source_azimuths_deg: FloatArray,
    guard: SourceGuardVariant,
    mask_type: str,
) -> SourceSectorMask:
    """oracle / detected の source mask を作る。"""
    if mask_type == "oracle":
        return build_source_sector_mask_from_azimuths(
            axis_azimuth_deg=axis_azimuth_deg,
            source_azimuths_deg=source_azimuths_deg,
            guard_deg=float(guard.guard_half_width_deg),
            mask_type="oracle",
        )
    if mask_type != "detected":
        raise ValueError("mask_type must be oracle or detected.")

    detected_indices = detect_source_beam_indices_from_level_peaks(
        levels_db=before_levels_db,
        max_source_count=2,
        guard_beam_count=_guard_beam_count_from_deg(
            axis_azimuth_deg, float(guard.guard_half_width_deg)
        ),
        threshold_db_below_peak=DETECTED_SOURCE_THRESHOLD_DB_BELOW_PEAK,
    )
    if detected_indices.size == 0:
        peak_index = int(np.argmax(before_levels_db))
        detected_indices = np.asarray([peak_index], dtype=np.int64)
    detected_azimuths_deg = np.asarray(axis_azimuth_deg[detected_indices], dtype=np.float64)
    return build_source_sector_mask_from_azimuths(
        axis_azimuth_deg=axis_azimuth_deg,
        source_azimuths_deg=detected_azimuths_deg,
        guard_deg=float(guard.guard_half_width_deg),
        mask_type="detected",
    )


def _guard_beam_count_from_deg(axis_azimuth_deg: FloatArray, guard_deg: float) -> int:
    """detected peak 抑制用の角度 guard を beam 本数へ変換する。"""
    axis = np.asarray(axis_azimuth_deg, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        return 1
    minimum_spacing_deg = float(np.min(np.diff(axis)))
    if minimum_spacing_deg <= 0.0:
        return 1
    return int(max(1, math.ceil(float(guard_deg) / minimum_spacing_deg)))


def _relative_loading_power(covariance_matrix: ComplexArray, loading: float) -> float:
    """平均対角 power に対する relative loading を実 power へ変換する。"""
    matrix = np.asarray(covariance_matrix, dtype=np.complex128)
    n_ref = int(matrix.shape[0])
    average_power = float(np.real(np.trace(matrix)) / float(n_ref))
    if not bool(np.isfinite(average_power)) or average_power <= 0.0:
        # 無音に近い reference でも loaded covariance を作れるよう 1.0 を基準にする。
        # ここで 0 にすると、fallback 判定前に solve が特異化する。
        average_power = 1.0
    return float(loading) * average_power


def _run_a2_source_mask_slc(
    *,
    beam_output: ComplexArray,
    source_sector_mask: SourceSectorMask,
    source_reference_beams: IntArray,
    eta: float,
    loading: float,
    sample_per_dof: float,
    train_mode: str,
) -> A2RunResult:
    """A2 source-mask SLC を same-block または one-block-delay で実行する。

    Args:
        beam_output: 固定整相後の mixed beam 波形。shape は `[n_beam, n_sample]`。
        source_sector_mask: source / non-source mask。
        source_reference_beams: source reference beam。shape は `[n_ref]`。
        eta: キャンセル量係数。無次元。
        loading: relative diagonal loading。無次元。
        sample_per_dof: 1 自由度あたりに要求する最小 sample 数。
        train_mode: `same_block` または `one_block_delay`。

    Returns:
        raw/effective 出力と SLC health。

    境界条件:
        `one_block_delay` では前半 block で係数を学習し、後半 block だけを評価する。
        これにより同一 block 学習・評価による過大評価を避ける。
    """
    start_time = time.perf_counter()
    signals = np.asarray(beam_output, dtype=np.complex128)
    if signals.ndim != 2:
        raise ValueError("beam_output must have shape (n_beam, n_sample).")
    if train_mode == "same_block":
        train_signals = signals
        eval_signals = signals
    elif train_mode == "one_block_delay":
        split_index = int(signals.shape[1] // 2)
        if split_index <= 0 or split_index >= signals.shape[1]:
            raise ValueError("one_block_delay requires at least two samples.")
        train_signals = signals[:, :split_index]
        eval_signals = signals[:, split_index:]
    else:
        raise ValueError("train_mode must be same_block or one_block_delay.")

    reference_indices = np.asarray(source_reference_beams, dtype=np.int64)
    non_source_beams = np.flatnonzero(source_sector_mask.non_source_mask).astype(np.int64)
    n_ref = int(reference_indices.size)
    n_non_source = int(non_source_beams.size)
    n_train_sample = int(train_signals.shape[1])
    capacity_dof = int(n_ref * TAP_LEN)
    capacity_is_feasible = n_ref >= 1 and n_train_sample >= float(sample_per_dof) * float(
        capacity_dof
    )

    raw_output = eval_signals.copy()
    effective_output = eval_signals.copy()
    fallback_reasons: list[str] = []
    condition_number: float | None = None
    weight_norm: float | None = None

    if not bool(np.all(np.isfinite(signals))):
        fallback_reasons.append("non_finite_input")
    if n_non_source == 0:
        fallback_reasons.append("non_source_empty")
    if not capacity_is_feasible:
        fallback_reasons.append("reference_capacity_insufficient")

    if len(fallback_reasons) == 0:
        source_reference_train = train_signals[reference_indices, :]
        source_reference_eval = eval_signals[reference_indices, :]
        train_non_source = train_signals[non_source_beams, :]
        eval_non_source = eval_signals[non_source_beams, :]

        # R_ss shape: [n_ref, n_ref]、r_sd shape: [n_non_source, n_ref]。
        # r_sd[b, ref] = Σ_n X_S[ref, n] * conj(d_b[n]) / K であり、
        # source reference から non-source beam b への漏れ込み係数を推定する右辺である。
        covariance_matrix = np.asarray(
            (source_reference_train @ source_reference_train.conj().T) / float(n_train_sample),
            dtype=np.complex128,
        )
        cross_correlations = np.asarray(
            (train_non_source.conj() @ source_reference_train.T) / float(n_train_sample),
            dtype=np.complex128,
        )
        loading_power = _relative_loading_power(covariance_matrix, float(loading))
        loaded_covariance = covariance_matrix + loading_power * np.eye(n_ref, dtype=np.complex128)
        condition_number = float(np.linalg.cond(loaded_covariance))

        try:
            # loaded_covariance @ weights[b].T = r_sd[b].T を全 non-source beam で同時に解く。
            weights = np.asarray(
                np.linalg.solve(loaded_covariance, cross_correlations.T).T,
                dtype=np.complex128,
            )
            cancel_estimate = np.conj(weights) @ source_reference_eval
            raw_output[non_source_beams, :] = eval_non_source - float(eta) * cancel_estimate
            raw_output[source_sector_mask.source_mask, :] = eval_signals[
                source_sector_mask.source_mask,
                :,
            ]
            weight_norm = float(np.linalg.norm(weights))
        except np.linalg.LinAlgError:
            fallback_reasons.append("linear_solve_failed")

    raw_nan_inf_count = int(np.count_nonzero(~np.isfinite(raw_output)))
    if raw_nan_inf_count > 0:
        fallback_reasons.append("non_finite_raw_output")
    if condition_number is not None:
        if not bool(np.isfinite(condition_number)) or condition_number > CONDITION_NUMBER_LIMIT:
            fallback_reasons.append("condition_number_limit_exceeded")

    fallback_required = bool(len(fallback_reasons) > 0)
    if fallback_required:
        effective_output = eval_signals.copy()
    else:
        effective_output = raw_output.copy()

    elapsed_sec = float(time.perf_counter() - start_time)
    return A2RunResult(
        raw_output=np.asarray(raw_output, dtype=np.complex128),
        effective_output=np.asarray(effective_output, dtype=np.complex128),
        condition_number=condition_number,
        weight_norm=weight_norm,
        fallback_required=fallback_required,
        fallback_reasons=tuple(fallback_reasons),
        n_ref=n_ref,
        n_non_source=n_non_source,
        capacity_block_size=n_train_sample,
        capacity_dof=capacity_dof,
        elapsed_sec=elapsed_sec,
    )


def _verify_a2_vectorized_formula(
    *,
    beam_output: ComplexArray,
    source_sector_mask: SourceSectorMask,
    source_reference_beams: IntArray,
) -> None:
    """vectorized A2 実装が公開 A2 実装と同じ same-block 出力になることを確認する。"""
    config = SourceMaskSlcConfig(
        eta=0.5,
        loading=3.0e-2,
        tap_len=1,
        min_ref=1,
        sample_per_dof=SAMPLE_PER_DOF_MIN,
        condition_number_limit=CONDITION_NUMBER_LIMIT,
    )
    reference_result = SourceMaskNonSourceLeakageSubtractor(config).process(
        beam_output=beam_output,
        source_sector_mask=source_sector_mask,
        source_reference_beams=source_reference_beams,
    )
    vectorized_result = _run_a2_source_mask_slc(
        beam_output=beam_output,
        source_sector_mask=source_sector_mask,
        source_reference_beams=source_reference_beams,
        eta=0.5,
        loading=3.0e-2,
        sample_per_dof=SAMPLE_PER_DOF_MIN,
        train_mode="same_block",
    )
    np.testing.assert_allclose(
        vectorized_result.raw_output,
        reference_result.raw_output,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        vectorized_result.effective_output,
        reference_result.effective_output,
        atol=1.0e-10,
    )


def _metrics_row_fields(
    *,
    before_levels_db: FloatArray,
    after_levels_db: FloatArray,
    axis_azimuth_deg: FloatArray,
    source_mask: SourceSectorMask,
    realtime_factor: float | None,
    nan_inf_count: int,
    condition_number: float | None,
) -> dict[str, object]:
    """ABF-like metrics と判定を CSV 行用 dict へ変換する。"""
    metrics = calculate_abf_like_non_source_metrics(
        axis_azimuth_deg=axis_azimuth_deg,
        before_levels_db=before_levels_db,
        after_levels_db=after_levels_db,
        source_sector_mask=source_mask,
        level_unit_label=LEVEL_UNIT_LABEL,
    )
    decision = judge_abf_like_non_source_metrics(
        metrics,
        realtime_factor=realtime_factor,
        nan_inf_count=int(nan_inf_count),
        condition_number=condition_number,
    )
    non_source_mask = source_mask.non_source_mask
    ungated_worsening = float(
        np.max(after_levels_db[non_source_mask] - before_levels_db[non_source_mask])
    )
    row: dict[str, object] = dict(metrics.as_dict())
    row.update(decision.as_dict())
    row["ungated_max_local_worsening_db"] = ungated_worsening
    return row


def _base_row(
    *,
    case: EvaluationCase,
    guard: SourceGuardVariant,
    mask_type: str,
    source_mask: SourceSectorMask,
    method_family: str,
    method_id: str,
    candidate_id: str,
    output_stage: str,
    decision_used: bool,
) -> dict[str, object]:
    """全方式で共通する CSV 行フィールドを作る。"""
    return {
        "tier": case.tier,
        "case_id": case.case_id,
        "method_family": method_family,
        "method_id": method_id,
        "candidate_id": candidate_id,
        "output_stage": output_stage,
        "decision_used": bool(decision_used),
        "target_frequency_hz": float(case.target_frequency_hz),
        "interferer_frequency_hz": float(case.interferer_frequency_hz),
        "target_azimuth_deg": float(case.target_azimuth_deg),
        "interferer_azimuth_deg": float(case.interferer_azimuth_deg),
        "offgrid_deg": float(case.offgrid_deg),
        "mask_type": mask_type,
        "source_guard_label": guard.label,
        "source_mask_width_deg": float(guard.source_mask_width_deg),
        "source_guard_half_width_deg": float(guard.guard_half_width_deg),
        "source_count_in_mask": int(source_mask.source_beam_indices.size),
        "non_source_beam_count": int(np.count_nonzero(source_mask.non_source_mask)),
        "source_beam_indices": "|".join(
            str(int(index)) for index in source_mask.source_beam_indices.tolist()
        ),
        "level_unit_label": LEVEL_UNIT_LABEL,
        "delta_unit_label": "dB re before level",
    }


def _b1_sidelobe_metrics(
    *,
    source_specs: tuple[SourceSpec, SourceSpec],
    response_cache: dict[float, FrequencyResponseCacheEntry],
    weights_by_frequency: dict[float, FloatArray],
    axis_azimuth_deg: FloatArray,
    sound_speed_m_s: float,
) -> dict[str, object]:
    """B1 candidate の source 別 BL sidelobe 指標を最悪値へ集約する。"""
    first_sidelobe_values: list[float] = []
    p95_values: list[float] = []
    p99_values: list[float] = []
    isl_values: list[float] = []
    peak_margin_values: list[float] = []
    for source_spec in source_specs:
        response = _complex_response_for_source(
            cache_entry=response_cache[float(source_spec.frequency_hz)],
            channel_weights=weights_by_frequency[float(source_spec.frequency_hz)],
            source_azimuth_deg=float(source_spec.azimuth_deg),
            sound_speed_m_s=float(sound_speed_m_s),
        )
        source_level_db = float(source_spec.level_db)
        levels_db20 = np.asarray(
            source_level_db
            + 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(np.float64).tiny)),
            dtype=np.float64,
        )
        metrics = _sidelobe_distribution_metrics(
            scan_azimuths_deg=axis_azimuth_deg,
            beam_levels_db20=levels_db20,
            target_azimuth_deg=float(source_spec.azimuth_deg),
        )
        first_sidelobe_values.append(float(metrics["first_sidelobe_level_db_re_mainlobe_peak"]))
        p95_values.append(float(metrics["sidelobe_95_percentile_db_re_mainlobe_peak"]))
        p99_values.append(float(metrics["sidelobe_99_percentile_db_re_mainlobe_peak"]))
        isl_values.append(float(metrics["integrated_sidelobe_level_db_re_mainlobe_peak"]))
        peak_margin_values.append(float(metrics["sidelobe_peak_margin_db"]))

    return {
        "b1_worst_first_sidelobe_level_db_re_mainlobe_peak": float(np.max(first_sidelobe_values)),
        "b1_worst_sidelobe_p95_db_re_mainlobe_peak": float(np.max(p95_values)),
        "b1_worst_sidelobe_p99_db_re_mainlobe_peak": float(np.max(p99_values)),
        "b1_worst_integrated_sidelobe_level_db_re_mainlobe_peak": float(np.max(isl_values)),
        "b1_worst_peak_margin_db": float(np.min(peak_margin_values)),
    }


def _append_fixed_baseline_row(
    *,
    rows: list[dict[str, object]],
    case: EvaluationCase,
    guard: SourceGuardVariant,
    mask_type: str,
    source_mask: SourceSectorMask,
    axis_azimuth_deg: FloatArray,
    before_levels_db: FloatArray,
) -> None:
    """fixed baseline の基準行を追加する。"""
    row = _base_row(
        case=case,
        guard=guard,
        mask_type=mask_type,
        source_mask=source_mask,
        method_family="fixed_baseline",
        method_id="fixed_baseline",
        candidate_id="current_operational_shading",
        output_stage="effective",
        decision_used=True,
    )
    row.update(
        _metrics_row_fields(
            before_levels_db=before_levels_db,
            after_levels_db=before_levels_db,
            axis_azimuth_deg=axis_azimuth_deg,
            source_mask=source_mask,
            realtime_factor=0.0,
            nan_inf_count=0,
            condition_number=None,
        )
    )
    row["status"] = "baseline"
    row["fallback_required"] = False
    row["fallback_reason"] = ""
    row["realtime_factor"] = 0.0
    rows.append(row)


def _append_a2_rows(
    *,
    rows: list[dict[str, object]],
    case: EvaluationCase,
    guard: SourceGuardVariant,
    mask_type: str,
    source_mask: SourceSectorMask,
    source_reference_beams: IntArray,
    axis_azimuth_deg: FloatArray,
    before_output: ComplexArray,
    before_levels_db: FloatArray,
    evaluation_duration_s: float,
) -> None:
    """A2 sweep の raw/effective 行を追加する。"""
    for train_mode in ("same_block", "one_block_delay"):
        for eta in (0.25, 0.5, 0.75, 1.0):
            for loading in (1.0e-3, 1.0e-2, 3.0e-2, 1.0e-1):
                result = _run_a2_source_mask_slc(
                    beam_output=before_output,
                    source_sector_mask=source_mask,
                    source_reference_beams=source_reference_beams,
                    eta=float(eta),
                    loading=float(loading),
                    sample_per_dof=SAMPLE_PER_DOF_MIN,
                    train_mode=train_mode,
                )
                # one-block-delay では後半 block だけを評価するため、before も同じ時間窓へ揃える。
                if train_mode == "one_block_delay":
                    before_window = before_output[:, before_output.shape[1] // 2 :]
                    before_window_levels_db = _rms_levels_db20(before_window)
                    duration_s = float(before_window.shape[1]) / 32768.0
                else:
                    before_window_levels_db = before_levels_db
                    duration_s = float(evaluation_duration_s)
                realtime_factor = float(
                    result.elapsed_sec / max(duration_s, np.finfo(np.float64).eps)
                )

                for output_stage, after_output, decision_used in (
                    ("raw", result.raw_output, False),
                    ("effective", result.effective_output, True),
                ):
                    after_levels_db = _rms_levels_db20(after_output)
                    row = _base_row(
                        case=case,
                        guard=guard,
                        mask_type=mask_type,
                        source_mask=source_mask,
                        method_family="A2_source_mask_slc",
                        method_id="A2_source_mask_slc",
                        candidate_id=(f"eta{eta:g}_loading{loading:g}_{guard.label}_{train_mode}"),
                        output_stage=output_stage,
                        decision_used=decision_used,
                    )
                    row.update(
                        _metrics_row_fields(
                            before_levels_db=before_window_levels_db,
                            after_levels_db=after_levels_db,
                            axis_azimuth_deg=axis_azimuth_deg,
                            source_mask=source_mask,
                            realtime_factor=realtime_factor,
                            nan_inf_count=int(np.count_nonzero(~np.isfinite(after_output))),
                            condition_number=result.condition_number,
                        )
                    )
                    if not decision_used:
                        row["status"] = "diagnostic_raw"
                    row.update(
                        {
                            "eta": float(eta),
                            "loading": float(loading),
                            "train_mode": train_mode,
                            "sample_per_dof_min": float(SAMPLE_PER_DOF_MIN),
                            "capacity_block_size": int(result.capacity_block_size),
                            "capacity_dof": int(result.capacity_dof),
                            "condition_number": result.condition_number,
                            "weight_norm": result.weight_norm,
                            "realtime_factor": realtime_factor,
                            "fallback_required": bool(result.fallback_required),
                            "fallback_reason": "|".join(result.fallback_reasons),
                            "n_ref": int(result.n_ref),
                            "n_non_source": int(result.n_non_source),
                        }
                    )
                    rows.append(row)


def _append_b1_rows(
    *,
    rows: list[dict[str, object]],
    case: EvaluationCase,
    guard: SourceGuardVariant,
    mask_type: str,
    source_mask: SourceSectorMask,
    source_specs: tuple[SourceSpec, SourceSpec],
    response_cache: dict[float, FrequencyResponseCacheEntry],
    shading_definition: OperationalShadingDefinition,
    axis_azimuth_deg: FloatArray,
    before_levels_db: FloatArray,
    fs_hz: float,
    sound_speed_m_s: float,
    n_sample: int,
    b1_candidates: tuple[B1Candidate, ...],
) -> None:
    """B1 FD/shading candidate 行を追加する。"""
    for candidate in b1_candidates:
        weights_by_frequency = {
            float(source.frequency_hz): _weights_for_candidate(
                candidate=candidate,
                frequency_hz=float(source.frequency_hz),
                cache_entry=response_cache[float(source.frequency_hz)],
                shading_definition=shading_definition,
            )
            for source in source_specs
        }
        candidate_output = _synthesize_beam_output(
            source_specs=source_specs,
            response_cache=response_cache,
            weights_by_frequency=weights_by_frequency,
            fs_hz=float(fs_hz),
            sound_speed_m_s=float(sound_speed_m_s),
            n_sample=int(n_sample),
        )
        after_levels_db = _rms_levels_db20(candidate_output)
        row = _base_row(
            case=case,
            guard=guard,
            mask_type=mask_type,
            source_mask=source_mask,
            method_family="B1_fd_shading_selection",
            method_id="B1_fd_shading_selection",
            candidate_id=candidate.candidate_id,
            output_stage="effective",
            decision_used=True,
        )
        row.update(
            _metrics_row_fields(
                before_levels_db=before_levels_db,
                after_levels_db=after_levels_db,
                axis_azimuth_deg=axis_azimuth_deg,
                source_mask=source_mask,
                realtime_factor=0.0,
                nan_inf_count=int(np.count_nonzero(~np.isfinite(candidate_output))),
                condition_number=None,
            )
        )
        n_eff_values: list[float] = []
        snr_loss_values: list[float] = []
        active_counts: list[int] = []
        for source in source_specs:
            weights = weights_by_frequency[float(source.frequency_hz)]
            n_eff = _effective_channel_count(weights)
            n_active = int(weights.size)
            n_eff_values.append(float(n_eff))
            snr_loss_values.append(float(10.0 * np.log10(float(n_active)) - 10.0 * np.log10(n_eff)))
            active_counts.append(n_active)
        row.update(
            _b1_sidelobe_metrics(
                source_specs=source_specs,
                response_cache=response_cache,
                weights_by_frequency=weights_by_frequency,
                axis_azimuth_deg=axis_azimuth_deg,
                sound_speed_m_s=float(sound_speed_m_s),
            )
        )
        row.update(
            {
                "beta": "" if candidate.beta is None else float(candidate.beta),
                "uses_operational_shading": bool(candidate.uses_operational_shading),
                "active_aperture_candidate": "operational_frequency_active_set",
                "target_active_channel_count": int(active_counts[0]),
                "interferer_active_channel_count": int(active_counts[1]),
                "n_eff_target": float(n_eff_values[0]),
                "n_eff_interferer": float(n_eff_values[1]),
                "n_eff_min": float(np.min(np.asarray(n_eff_values, dtype=np.float64))),
                "snr_loss_vs_rectangular_db_max": float(
                    np.max(np.asarray(snr_loss_values, dtype=np.float64))
                ),
                "condition_number": "",
                "weight_norm": "",
                "realtime_factor": 0.0,
                "fallback_required": False,
                "fallback_reason": "",
            }
        )
        rows.append(row)


def _append_b2_final_row(rows: list[dict[str, object]]) -> None:
    """B2 residual delay correction の最終判定行を追加する。"""
    rows.append(
        {
            "tier": "final",
            "case_id": "B2_residual_delay_correction",
            "method_family": "B2_residual_delay_correction",
            "method_id": "B2_residual_delay_correction",
            "candidate_id": "not_evaluated",
            "output_stage": "final_judgment",
            "decision_used": True,
            "status": "not_evaluated",
            "failure_reasons": "residual_delay_correction_candidate_not_defined_in_this_run",
            "hold_reasons": "",
            "fallback_required": "",
            "fallback_reason": "no B2 candidate coefficients were generated",
        }
    )


def _sanitize_csv_value(value: object) -> object:
    """CSV に保存しやすい scalar へ変換する。"""
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, list | tuple):
        return "|".join(str(item) for item in value)
    return value


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """summary CSV を保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _sanitize_csv_value(row.get(key, "")) for key in fieldnames})


def _method_status_counts(rows: list[dict[str, object]], method_family: str) -> dict[str, int]:
    """decision_used=True の row から method ごとの status 件数を集計する。"""
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("method_family") != method_family:
            continue
        if not bool(row.get("decision_used", False)):
            continue
        status = str(row.get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _final_verdict_from_counts(method_family: str, counts: dict[str, int]) -> str:
    """集計 status から方式単位の最終判定を返す。"""
    if method_family == "fixed_baseline":
        return "baseline"
    if method_family == "B2_residual_delay_correction":
        return "not_evaluated"
    if counts.get("pass", 0) > 0 and counts.get("fail", 0) == 0:
        return "pass"
    if counts.get("pass", 0) > 0 or counts.get("hold", 0) > 0:
        return "hold"
    return "fail"


def _require_float_row_value(row: dict[str, object], key: str, default: float = 0.0) -> float:
    """report 用 row から実数値を型検証して取り出す。

    Args:
        row: CSV / Markdown report へ出す 1 行分の辞書。
        key: 取り出す指標名。
        default: key が存在しない場合の既定値。

    Returns:
        Python float に確定した値。

    Raises:
        TypeError: 値が数値ではない場合。

    境界条件:
        CSV 行は方式ごとに列が異なるため、存在しない値は default とする。
        存在する値が bool や文字列の場合は、数値指標として誤読しないよう停止する。
    """
    value = row.get(key)
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be numeric.")
    return float(value)


def _top_rows(
    rows: list[dict[str, object]],
    method_family: str,
    limit: int = 8,
) -> list[dict[str, object]]:
    """non-source p95 改善量を優先して代表 row を抽出する。"""
    candidates = [
        row
        for row in rows
        if row.get("method_family") == method_family
        and bool(row.get("decision_used", False))
        and row.get("output_stage") == "effective"
        and isinstance(row.get("non_source_p95_level_delta_db"), float)
    ]
    return sorted(
        candidates,
        key=lambda row: (
            _require_float_row_value(row, "non_source_p95_level_delta_db"),
            _require_float_row_value(row, "non_source_global_peak_delta_db"),
            _require_float_row_value(row, "max_local_worsening_db_gated"),
        ),
    )[:limit]


def _write_report(rows: list[dict[str, object]], output_path: Path) -> None:
    """comparison_report.md を保存する。"""
    method_families = (
        "fixed_baseline",
        "A2_source_mask_slc",
        "B1_fd_shading_selection",
        "B2_residual_delay_correction",
    )
    lines: list[str] = [
        "# 軽量 ABF-like 非信号方位抑圧 横並び評価",
        "",
        "## 評価条件",
        "",
        "- 評価 role: `ABF_like_non_source_suppression`",
        f"- レベル基準: `{LEVEL_UNIT_LABEL}`",
        f"- 合成 sample 数: `{N_SAMPLE}`",
        "- offgrid は target / interferer の両方に同じ角度を加えた。",
        "- detected mask は fixed baseline の mixed RMS beam level から最大 2 source を検出した。",
        "- A2 raw は診断用であり、pass/hold/fail の方式判定には effective だけを使った。",
        "- A2 one-block-delay は前半 block で学習し、後半 block を評価した。",
        (
            "- B1 は周波数ごとの既存 active aperture 上で "
            "current operational shading と beta 候補を比較した。"
        ),
        "",
        "## 最終判定",
        "",
        "| method | verdict | status counts |",
        "|---|---:|---|",
    ]
    for method_family in method_families:
        counts = _method_status_counts(rows, method_family)
        verdict = _final_verdict_from_counts(method_family, counts)
        count_label = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        lines.append(f"| `{method_family}` | `{verdict}` | {count_label} |")

    lines.extend(
        [
            "",
            "## A2 代表 effective row",
            "",
            (
                "| status | case | candidate | mask | p95 delta | global delta | "
                "gated worsening | fallback |"
            ),
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in _top_rows(rows, "A2_source_mask_slc"):
        lines.append(
            (
                "| {status} | {case} | {candidate} | {mask} | {p95:.3f} | "
                "{global_peak:.3f} | {worsening:.3f} | {fallback} |"
            ).format(
                status=str(row.get("status", "")),
                case=str(row.get("case_id", "")),
                candidate=str(row.get("candidate_id", "")),
                mask=str(row.get("mask_type", "")),
                p95=_require_float_row_value(row, "non_source_p95_level_delta_db"),
                global_peak=_require_float_row_value(row, "non_source_global_peak_delta_db"),
                worsening=_require_float_row_value(row, "max_local_worsening_db_gated"),
                fallback=str(row.get("fallback_required", "")),
            )
        )

    lines.extend(
        [
            "",
            "## B1 代表 effective row",
            "",
            (
                "| status | case | candidate | mask | p95 delta | p99 delta | "
                "N_eff min | SNR loss max |"
            ),
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in _top_rows(rows, "B1_fd_shading_selection"):
        lines.append(
            (
                "| {status} | {case} | {candidate} | {mask} | {p95:.3f} | "
                "{p99:.3f} | {neff:.3f} | {snr_loss:.3f} |"
            ).format(
                status=str(row.get("status", "")),
                case=str(row.get("case_id", "")),
                candidate=str(row.get("candidate_id", "")),
                mask=str(row.get("mask_type", "")),
                p95=_require_float_row_value(row, "non_source_p95_level_delta_db"),
                p99=_require_float_row_value(row, "non_source_p99_level_delta_db"),
                neff=_require_float_row_value(row, "n_eff_min"),
                snr_loss=_require_float_row_value(row, "snr_loss_vs_rectangular_db_max"),
            )
        )

    lines.extend(
        [
            "",
            "## 出力",
            "",
            f"- summary CSV: `{SUMMARY_CSV_PATH.as_posix()}`",
            f"- report: `{REPORT_MD_PATH.as_posix()}`",
            "",
            "## 注意",
            "",
            (
                "この評価は beam-domain 周波数応答から mixed 波形を合成しており、"
                "FRAZ/BTR 図は生成していない。"
            ),
            (
                "ただし source 保護、non-source 包絡線、SLC covariance health、"
                "runtime ratio、shading の N_eff / SNR loss は CSV に保存した。"
            ),
            (
                "B2 residual delay correction は、この実行では補正係数 candidate が"
                "未定義のため未評価として分けた。"
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Tier 0 / Tier 1 の横並び評価を実行し、CSV と Markdown を保存する。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    array_definition = OperationalSparseArrayDefinition.load_json(ARRAY_DEFINITION_PATH)
    shading_definition = OperationalShadingDefinition.load_json(SHADING_DEFINITION_PATH)
    filter_bank = FractionalDelayFilterBank.load_npz(FRACTIONAL_DELAY_FILTER_BANK_PATH)

    directions, axis_azimuth_deg_raw, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=N_BEAM_AZ_REAL,
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    axis_azimuth_deg = np.asarray(axis_azimuth_deg_raw, dtype=np.float64)
    direction_cosines = np.asarray(directions.T, dtype=np.float64)

    frequencies_hz = (6144.0, 8192.0, 10000.0)
    response_cache: dict[float, FrequencyResponseCacheEntry] = {}
    for frequency_hz in frequencies_hz:
        active_indices = array_definition.active_channel_indices_for_frequency(float(frequency_hz))
        active_positions_m = np.asarray(
            array_definition.positions_m[active_indices], dtype=np.float64
        )
        beamformer = FractionalDelayAndSumBeamformer.from_geometry(
            array_pos_m=active_positions_m,
            dir_cos=direction_cosines,
            fs_hz=float(array_definition.fs_hz),
            sound_speed_m_s=float(array_definition.sound_speed_m_s),
            fractional_filter_bank=filter_bank,
        )
        response_cache[float(frequency_hz)] = FrequencyResponseCacheEntry(
            frequency_hz=float(frequency_hz),
            active_indices=np.asarray(active_indices, dtype=np.int64),
            active_positions_m=active_positions_m,
            beamformer=beamformer,
        )

    guards = (
        SourceGuardVariant(label="narrow", guard_half_width_deg=1.0),
        SourceGuardVariant(label="default", guard_half_width_deg=2.0),
        SourceGuardVariant(label="wide", guard_half_width_deg=3.0),
    )
    b1_candidates = (
        B1Candidate(
            candidate_id="current_operational_shading", beta=None, uses_operational_shading=True
        ),
        B1Candidate(candidate_id="kaiser_beta_0.0", beta=0.0, uses_operational_shading=False),
        B1Candidate(candidate_id="kaiser_beta_2.0", beta=2.0, uses_operational_shading=False),
        B1Candidate(candidate_id="kaiser_beta_4.0", beta=4.0, uses_operational_shading=False),
        B1Candidate(candidate_id="kaiser_beta_6.0", beta=6.0, uses_operational_shading=False),
    )

    rows: list[dict[str, object]] = []
    verification_done = False
    evaluation_duration_s = float(N_SAMPLE) / float(array_definition.fs_hz)

    for case in _build_evaluation_cases():
        source_specs = (
            SourceSpec(
                label="target",
                frequency_hz=float(case.target_frequency_hz),
                azimuth_deg=float(case.target_azimuth_deg),
                level_db=float(TARGET_LEVEL_DB),
                phase_rad=0.0,
            ),
            SourceSpec(
                label="interferer",
                frequency_hz=float(case.interferer_frequency_hz),
                azimuth_deg=float(case.interferer_azimuth_deg),
                level_db=float(INTERFERER_LEVEL_DB),
                phase_rad=0.7,
            ),
        )
        current_weights_by_frequency = {
            float(source.frequency_hz): _weights_for_candidate(
                candidate=b1_candidates[0],
                frequency_hz=float(source.frequency_hz),
                cache_entry=response_cache[float(source.frequency_hz)],
                shading_definition=shading_definition,
            )
            for source in source_specs
        }
        before_output = _synthesize_beam_output(
            source_specs=source_specs,
            response_cache=response_cache,
            weights_by_frequency=current_weights_by_frequency,
            fs_hz=float(array_definition.fs_hz),
            sound_speed_m_s=float(array_definition.sound_speed_m_s),
            n_sample=N_SAMPLE,
        )
        before_levels_db = _rms_levels_db20(before_output)
        source_azimuths_deg = np.asarray(
            [case.target_azimuth_deg, case.interferer_azimuth_deg],
            dtype=np.float64,
        )

        for guard in guards:
            for mask_type in ("oracle", "detected"):
                source_mask = _source_mask_for_case(
                    axis_azimuth_deg=axis_azimuth_deg,
                    before_levels_db=before_levels_db,
                    source_azimuths_deg=source_azimuths_deg,
                    guard=guard,
                    mask_type=mask_type,
                )
                source_reference_beams = np.flatnonzero(source_mask.source_mask).astype(np.int64)
                if not verification_done:
                    _verify_a2_vectorized_formula(
                        beam_output=before_output,
                        source_sector_mask=source_mask,
                        source_reference_beams=source_reference_beams,
                    )
                    verification_done = True

                _append_fixed_baseline_row(
                    rows=rows,
                    case=case,
                    guard=guard,
                    mask_type=mask_type,
                    source_mask=source_mask,
                    axis_azimuth_deg=axis_azimuth_deg,
                    before_levels_db=before_levels_db,
                )
                _append_a2_rows(
                    rows=rows,
                    case=case,
                    guard=guard,
                    mask_type=mask_type,
                    source_mask=source_mask,
                    source_reference_beams=source_reference_beams,
                    axis_azimuth_deg=axis_azimuth_deg,
                    before_output=before_output,
                    before_levels_db=before_levels_db,
                    evaluation_duration_s=evaluation_duration_s,
                )
                _append_b1_rows(
                    rows=rows,
                    case=case,
                    guard=guard,
                    mask_type=mask_type,
                    source_mask=source_mask,
                    source_specs=source_specs,
                    response_cache=response_cache,
                    shading_definition=shading_definition,
                    axis_azimuth_deg=axis_azimuth_deg,
                    before_levels_db=before_levels_db,
                    fs_hz=float(array_definition.fs_hz),
                    sound_speed_m_s=float(array_definition.sound_speed_m_s),
                    n_sample=N_SAMPLE,
                    b1_candidates=b1_candidates,
                )

    _append_b2_final_row(rows)
    _write_csv(rows, SUMMARY_CSV_PATH)
    _write_report(rows, REPORT_MD_PATH)
    print(f"saved {len(rows)} rows to {SUMMARY_CSV_PATH}")
    print(f"saved report to {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
