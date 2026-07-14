"""固定遅延+差分補正 MVDR の review_pack を生成する。

このスクリプトは、事前計算 51×128 小数遅延 FIR を使った固定整相を baseline とし、
周波数領域 MVDR、および差分補正 128 tap FIR 化後の重みを比較する。

出力先は `artifacts/beamforming/fixed_delay_diff_mvdr/review_pack/` である。
"""

from __future__ import annotations

import csv
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming import (  # noqa: E402
    DelayTable,
    DifferenceCorrectionFIRDesigner,
    LoadedMVDRWeightDesigner,
    design_fixed_delay_fractional_weights_from_delay_table,
    design_standard_fractional_delay_filter_bank,
    make_directions,
)
from spflow.beamforming_evaluation.diagnostic_plotting import (  # noqa: E402
    centers_to_edges,
    require_matplotlib,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 実行環境依存
    plt = None


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
BoolArray: TypeAlias = NDArray[np.bool_]

REVIEW_PACK_DIR = ROOT / "artifacts" / "beamforming" / "fixed_delay_diff_mvdr" / "review_pack"
FIGURE_DIR = REVIEW_PACK_DIR / "figures"
DATA_DIR = REVIEW_PACK_DIR / "data"
REVIEW_INDEX_PATH = REVIEW_PACK_DIR / "review_index.md"
SCENARIO_SUMMARY_PATH = REVIEW_PACK_DIR / "scenario_summary.csv"
WORST_CASES_PATH = REVIEW_PACK_DIR / "worst_cases.csv"
LEVEL_UNIT_LABEL = "dB re input RMS"
BTR_UNIT_LABEL = "dB re frame max"
SOURCE_MASK_GUARD_DEG = 3.0
MASK_TYPE_LABEL = "source_guard_3deg_non_source_complement"
FALSE_PEAK_MARGIN_DB = 3.0
NEAR_FREQUENCY_OFFSET_HZ = 0.1
METHOD_ORDER = ("fixed_baseline", "mvdr_oracle", "diff_mvdr_fir512")
METHOD_LABELS = {
    "fixed_baseline": "fixed",
    "mvdr_oracle": "MVDR oracle",
    "diff_mvdr_fir512": "diff MVDR FIR512",
}
METHOD_COLORS = {
    "fixed_baseline": "black",
    "mvdr_oracle": "tab:blue",
    "diff_mvdr_fir512": "tab:orange",
}
SUMMARY_COLUMNS = (
    "scenario",
    "method",
    "mask_type",
    "candidate",
    "status",
    "source_peak_delta_db",
    "source_azimuth_error_deg",
    "non_source_global_peak_delta_db",
    "non_source_p95_level_delta_db",
    "non_source_p99_level_delta_db",
    "non_source_integrated_level_delta_db",
    "source_to_non_source_margin_delta_db",
    "false_peak_count_delta",
    "max_local_worsening_db_gated",
    "fallback_required",
    "fallback_reason",
    "runtime_factor",
    "evaluation_pattern",
    "target_azimuth_deg",
    "target_frequency_hz",
    "interferer_azimuth_deg",
    "interferer_frequency_hz",
    "target_mainlobe_delta_db",
    "target_peak_azimuth_error_deg",
    "interferer_leakage_delta_db",
    "interferer_leakage_reduction_db",
    "mixed_target_beam_delta_db",
    "q_blocking_max_db",
    "target_response_error_db",
    "q_reconstruction_rms_error",
    "loaded_condition_number_max",
    "source_count_expected",
    "source_count_detected",
)


@dataclass(frozen=True)
class SourceSpec:
    """方式検証に使う狭帯域平面波 source 条件を表す。

    このクラスは、方位、周波数、RMS 振幅レベル、ラベルを保持する。
    入力は scene 定義であり、出力は steering vector と評価 mask の生成に使う。

    波形合成、MVDR 重み設計、BL/FRAZ/BTR 描画は責務に含めない。
    信号処理上は、固定整相と local leakage canceller 評価の source 真値である。
    """

    label: str
    azimuth_deg: float
    frequency_hz: float
    level_db: float
    phase_deg: float = 0.0


@dataclass(frozen=True)
class ScenarioDefinition:
    """固定遅延+差分補正 MVDR の評価 scenario を表す。

    このクラスは、protected target と任意の interferer 群を保持する。
    入力は source 条件であり、出力は review_pack の 1 scenario に対応する。

    個別 metric の計算、図の保存、採否判定は責務に含めない。
    信号処理上は、Beamforming Evaluation の `fixed_beam_single_source`、
    `slc_target_only`、`slc_same_frequency_interference`、`slc_different_frequency_interference`
    へ対応する評価条件である。
    """

    scenario_id: str
    purpose: str
    evaluation_pattern: str
    target: SourceSpec
    interferers: tuple[SourceSpec, ...]


@dataclass(frozen=True)
class ScenarioReviewData:
    """review_pack の描画前配列と評価結果を保持する。

    このクラスは、BL/FRAZ/BTR の method 別レベル、source/non-source mask、summary row を
    まとめて保存するための中間データである。

    入力は 1 scenario の解析結果、出力は PNG、npz、CSV、Markdown である。

    信号生成や MVDR 設計そのものは責務に含めない。
    信号処理上は、方式検証をレビュー可能なファイル群へ変換する境界に位置づく。
    """

    scenario: ScenarioDefinition
    azimuth_deg: FloatArray
    frequency_hz: FloatArray
    time_sec: FloatArray
    source_mask: BoolArray
    non_source_mask: BoolArray
    bl_levels_db: dict[str, FloatArray]
    source_frequency_bl_levels_db: dict[str, FloatArray]
    fraz_levels_db: dict[str, FloatArray]
    btr_levels_db: dict[str, FloatArray]
    rows: list[dict[str, object]]


def _build_scenarios() -> tuple[ScenarioDefinition, ...]:
    """review_pack に含める scenario 定義を返す。"""
    return (
        ScenarioDefinition(
            scenario_id="target_only_20deg_1536hz",
            purpose="target-only で差分補正枝が protected target を自己消去しないことを確認する。",
            evaluation_pattern="slc_target_only",
            target=SourceSpec("target", 20.0, 1536.0, 0.0),
            interferers=(),
        ),
        ScenarioDefinition(
            scenario_id="near_frequency_interferer_60deg_1536p1hz",
            purpose=(
                "target から 0.1 Hz ずらした near-frequency interferer の "
                "protected target beam への漏れ込み低減を確認する。"
            ),
            evaluation_pattern="slc_different_frequency_interference",
            target=SourceSpec("target", 20.0, 1536.0, 0.0),
            interferers=(
                SourceSpec(
                    "interferer",
                    60.0,
                    1536.0 + NEAR_FREQUENCY_OFFSET_HZ,
                    0.0,
                    phase_deg=70.0,
                ),
            ),
        ),
        ScenarioDefinition(
            scenario_id="different_frequency_interferer_75deg",
            purpose="異周波 interferer 条件で target mainlobe 維持と漏れ込み低減を確認する。",
            evaluation_pattern="slc_different_frequency_interference",
            target=SourceSpec("target", 20.0, 1536.0, 0.0),
            interferers=(SourceSpec("interferer", 75.0, 2304.0, 0.0, phase_deg=40.0),),
        ),
        ScenarioDefinition(
            scenario_id="target_only_20deg_4096hz",
            purpose=(
                "4.096 kHz target-only で差分補正枝が protected target を"
                "自己消去しないことを確認する。"
            ),
            evaluation_pattern="slc_target_only",
            target=SourceSpec("target", 20.0, 4096.0, 0.0),
            interferers=(),
        ),
        ScenarioDefinition(
            scenario_id="near_frequency_interferer_60deg_4096p1hz",
            purpose=(
                "4.096 kHz target から 0.1 Hz ずらした near-frequency interferer の "
                "protected target beam への漏れ込み低減を確認する。"
            ),
            evaluation_pattern="slc_different_frequency_interference",
            target=SourceSpec("target", 20.0, 4096.0, 0.0),
            interferers=(
                SourceSpec(
                    "interferer",
                    60.0,
                    4096.0 + NEAR_FREQUENCY_OFFSET_HZ,
                    0.0,
                    phase_deg=70.0,
                ),
            ),
        ),
        ScenarioDefinition(
            scenario_id="different_frequency_interferer_75deg_4096_5120hz",
            purpose="4.096 kHz target と 5.120 kHz interferer で target 維持と副作用を確認する。",
            evaluation_pattern="slc_different_frequency_interference",
            target=SourceSpec("target", 20.0, 4096.0, 0.0),
            interferers=(SourceSpec("interferer", 75.0, 5120.0, 0.0, phase_deg=40.0),),
        ),
    )


def _build_array_positions(n_ch: int, spacing_m: float) -> FloatArray:
    """中心原点の 1 列 ULA センサ位置を返す。"""
    positions = np.zeros((int(n_ch), 3), dtype=np.float64)
    positions[:, 0] = (np.arange(int(n_ch), dtype=np.float64) - 0.5 * (int(n_ch) - 1)) * float(
        spacing_m
    )
    return positions


def _direction_from_azimuth(azimuth_deg: float) -> FloatArray:
    """方位角から水平面方向余弦を返す。"""
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    return np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0], dtype=np.float64)


def _arrival_steering(
    array_positions_m: FloatArray,
    source: SourceSpec,
    frequencies_hz: FloatArray,
    *,
    sound_speed_m_s: float,
) -> ComplexArray:
    """source の channel steering を周波数軸上で返す。

    Args:
        array_positions_m: センサ位置。shape は `[n_ch, 3]`、単位は m。
        source: source 条件。方位は deg、周波数は Hz。
        frequencies_hz: 評価周波数。shape は `[n_freq]`、単位は Hz。
        sound_speed_m_s: 音速。単位は m/s。

    Returns:
        channel steering。shape は `[n_freq, n_ch]`。

    Notes:
        `arrival_delay_sec[ch] = -(r_ch^T u) / c` とし、
        `exp(-j 2π f arrival_delay_sec[ch])` を観測側の位相として使う。
        既存の time_delay 診断と同じ符号規約である。
    """
    direction = _direction_from_azimuth(float(source.azimuth_deg))
    arrival_delay_sec = -(array_positions_m @ direction) / float(sound_speed_m_s)
    phase = -1j * 2.0 * np.pi * frequencies_hz[:, np.newaxis] * arrival_delay_sec[np.newaxis, :]
    return np.asarray(np.exp(phase), dtype=np.complex128)


def _amplitude_from_db(level_db: float) -> float:
    """RMS 振幅基準の dB 値を線形振幅へ変換する。"""
    return float(10.0 ** (float(level_db) / 20.0))


def _levels_db(values: NDArray[Any]) -> FloatArray:
    """複素振幅を dB20 へ変換する。"""
    return np.asarray(
        20.0 * np.log10(np.maximum(np.abs(values), np.finfo(np.float64).tiny)), dtype=np.float64
    )


def _source_mask(
    axis_azimuth_deg: FloatArray,
    sources: tuple[SourceSpec, ...],
    *,
    guard_deg: float = SOURCE_MASK_GUARD_DEG,
) -> BoolArray:
    """source 方位 guard を含む source mask を作る。"""
    mask = np.zeros(axis_azimuth_deg.shape, dtype=np.bool_)
    for source in sources:
        mask |= np.abs(axis_azimuth_deg - float(source.azimuth_deg)) <= float(guard_deg)
    return mask


def _integrated_level_db(level_db: FloatArray) -> float:
    """dB レベル列を線形 power 平均へ戻して統合レベルを返す。

    non-source sector の総量比較では dB の算術平均を避け、
    `10 log10(mean(10^(L/10)))` により power として積分する。
    """
    power = np.power(10.0, level_db / 10.0)
    return float(10.0 * np.log10(max(float(np.mean(power)), np.finfo(np.float64).tiny)))


def _false_peak_count(non_source_level_db: FloatArray, source_peak_db: float) -> int:
    """source peak から 3 dB 以内の non-source peak 候補数を返す。

    3 dB は主ピークに近い false peak を拾うためのレビュー用閾値であり、
    採否はこの count だけでなく source 維持と non-source 統計を併せて判断する。
    """
    threshold_db = float(source_peak_db) - FALSE_PEAK_MARGIN_DB
    return int(np.count_nonzero(non_source_level_db >= threshold_db))


def _make_covariance(
    steering_by_source: dict[str, ComplexArray],
    sources: tuple[SourceSpec, ...],
    frequency_index: int,
    *,
    noise_power: float,
) -> ComplexArray:
    """source steering から周波数 bin の空間共分散を作る。"""
    n_ch = next(iter(steering_by_source.values())).shape[1]
    covariance = noise_power * np.eye(n_ch, dtype=np.complex128)
    for source in sources:
        steering = steering_by_source[source.label][frequency_index]
        amplitude = _amplitude_from_db(float(source.level_db))
        # R = Σ sigma_s^2 a_s a_s^H + sigma_n^2 I。
        # MVDR はこの空間共分散に対して protected beam の出力 power を最小化する。
        covariance += (amplitude**2) * np.outer(steering, steering.conj())
    return covariance


def _design_method_weights(
    fixed_weights: ComplexArray,
    steering_by_beam: ComplexArray,
    steering_by_source: dict[str, ComplexArray],
    scenario: ScenarioDefinition,
    frequencies_hz: FloatArray,
    target_beam_index: int,
) -> tuple[dict[str, ComplexArray], dict[str, FloatArray | BoolArray]]:
    """fixed / MVDR / diff FIR の周波数別 scan 重みを設計する。

    Returns:
        `(weights_by_method, diagnostics)` を返す。
        各 weight の shape は `[n_freq, n_beam, n_ch]` である。
    """
    n_freq, n_beam, n_ch = fixed_weights.shape
    mvdr_weights = np.zeros_like(fixed_weights)
    diff_final_weights = np.zeros_like(fixed_weights)
    condition_number = np.zeros((n_freq, n_beam), dtype=np.float64)
    fallback_mask = np.zeros((n_freq, n_beam), dtype=np.bool_)
    q_blocking_abs = np.zeros((n_freq, n_beam), dtype=np.float64)
    target_response_error = np.zeros((n_freq, n_beam), dtype=np.float64)
    q_reconstruction_error = np.zeros((n_freq, n_beam), dtype=np.float64)

    all_sources = (scenario.target, *scenario.interferers)
    mvdr_designer = LoadedMVDRWeightDesigner(diagonal_loading_ratio=1.0e-2)
    diff_designer = DifferenceCorrectionFIRDesigner(
        fir_taps=512,
        frequencies_hz=frequencies_hz,
        fs_hz=32768.0,
    )

    for beam_index in range(n_beam):
        covariance = np.stack(
            [
                _make_covariance(
                    steering_by_source,
                    all_sources,
                    frequency_index=freq_index,
                    noise_power=1.0e-4,
                )
                for freq_index in range(n_freq)
            ],
            axis=0,
        )
        # protected target beam では source truth steering を制約に使う。
        # beam center と source truth がずれる高周波条件で target を削ることを避けるためである。
        protected_steering = (
            steering_by_source[scenario.target.label]
            if beam_index == int(target_beam_index)
            else steering_by_beam[:, beam_index, :]
        )
        mvdr_result = mvdr_designer.compute(
            covariance,
            protected_steering,
            fixed_weights[:, beam_index, :],
        )
        diff_result = diff_designer.compute(
            fixed_weights[:, beam_index, :],
            mvdr_result.weights,
            protected_steering,
        )
        mvdr_weights[:, beam_index, :] = mvdr_result.weights
        diff_final_weights[:, beam_index, :] = diff_result.final_weight_freq
        condition_number[:, beam_index] = mvdr_result.loaded_condition_number
        fallback_mask[:, beam_index] = mvdr_result.fallback_mask
        q_blocking_abs[:, beam_index] = np.abs(diff_result.diagnostics.q_blocking_response)
        target_response_error[:, beam_index] = np.abs(
            diff_result.diagnostics.target_response_final
            - diff_result.diagnostics.target_response_w0
        )
        q_reconstruction_error[:, beam_index] = np.sqrt(
            np.mean(np.abs(diff_result.diagnostics.q_reconstruction_error) ** 2, axis=1)
        )

    return (
        {
            "fixed_baseline": fixed_weights,
            "mvdr_oracle": mvdr_weights,
            "diff_mvdr_fir512": diff_final_weights,
        },
        {
            "condition_number": condition_number,
            "fallback_mask": fallback_mask,
            "q_blocking_abs": q_blocking_abs,
            "target_response_error": target_response_error,
            "q_reconstruction_error": q_reconstruction_error,
        },
    )


def _response_cube(
    weights: ComplexArray,
    steering_by_source: dict[str, ComplexArray],
    sources: tuple[SourceSpec, ...],
    frequencies_hz: FloatArray,
) -> ComplexArray:
    """method 重みの source 別応答を `[n_freq, n_beam, n_source]` で返す。"""
    response = np.zeros((weights.shape[0], weights.shape[1], len(sources)), dtype=np.complex128)
    for source_index, source in enumerate(sources):
        source_steering = steering_by_source[source.label]
        source_frequency_index = int(np.argmin(np.abs(frequencies_hz - float(source.frequency_hz))))
        # source 周波数の狭帯域成分だけに応答を持たせる。
        # FRAZ では source 周波数 bin の方位応答が ridge として現れる。
        response[source_frequency_index, :, source_index] = (
            np.einsum(
                "bc,c->b",
                weights[source_frequency_index].conj(),
                source_steering[source_frequency_index],
                optimize=True,
            )
            * _amplitude_from_db(float(source.level_db))
            * np.exp(1j * np.deg2rad(float(source.phase_deg)))
        )
    return response


def _fraz_from_response(response: ComplexArray) -> FloatArray:
    """source 応答 cube から FRAZ レベル `[n_beam, n_freq]` を作る。"""
    # response shape: [n_freq, n_beam, n_source]。
    # source 軸の複素和により、同一周波数 source は干渉を含む mixed 応答として扱う。
    mixed = np.sum(response, axis=2)
    return _levels_db(mixed.T)


def _bl_from_fraz(
    fraz_levels_db: FloatArray, frequencies_hz: FloatArray, target_frequency_hz: float
) -> FloatArray:
    """target 周波数の BL レベルを FRAZ から取り出す。"""
    freq_index = int(np.argmin(np.abs(frequencies_hz - float(target_frequency_hz))))
    return np.asarray(fraz_levels_db[:, freq_index], dtype=np.float64)


def _source_frequency_bl_from_fraz(
    fraz_levels_db: FloatArray, frequencies_hz: FloatArray, sources: tuple[SourceSpec, ...]
) -> FloatArray:
    """source 真値周波数だけを統合した BL レベルを返す。

    Args:
        fraz_levels_db: FRAZ レベル。shape は [n_beam, n_freq]、単位は dB re input RMS。
        frequencies_hz: 評価周波数軸。shape は [n_freq]、単位は Hz。
        sources: scenario に含まれる target と interferer。各 source の周波数を参照する。

    Returns:
        source 真値周波数断面を最大値統合した BL。shape は [n_beam]、単位は dB re input RMS。

    Raises:
        ValueError: sources が空の場合。

    境界条件:
        target-only scenario では target 周波数 1 本だけの BL になる。
        near-frequency scenario では target 周波数だけを見ると interferer が見えなくなるため、
        source-preserving scan の visibility 確認用に source 真値周波数だけを統合する。
    """
    if len(sources) == 0:
        # source がない BL は source visibility 評価として意味を持たないため、
        # 入力定義の誤りとして扱う。
        raise ValueError("sources must not be empty")

    source_frequency_indices: list[int] = []
    for source in sources:
        source_frequency_indices.append(
            int(np.argmin(np.abs(frequencies_hz - float(source.frequency_hz))))
        )

    # selected shape: [n_beam, n_source_frequency]。
    # axis=0 は beam 方位、axis=1 は source 真値に対応する周波数断面を表す。
    # 最大値統合により、各 source が自分の周波数でピークとして残るかを 1 本の BL として表示する。
    selected = fraz_levels_db[:, source_frequency_indices]
    return np.asarray(np.max(selected, axis=1), dtype=np.float64)

def _btr_from_response(
    response_by_source: ComplexArray,
    scenario: ScenarioDefinition,
    frequencies_hz: FloatArray,
    *,
    duration_s: float,
    frame_count: int,
) -> tuple[FloatArray, FloatArray]:
    """source 応答から BTR 相対レベルを作る。

    BTR は frame ごとに最大ビームを 0 dB に正規化し、source track の連続性だけを見る。
    """
    time_sec = np.linspace(
        0.0, float(duration_s), int(frame_count), endpoint=False, dtype=np.float64
    )
    mixed = np.zeros((int(frame_count), response_by_source.shape[1]), dtype=np.complex128)
    sources = (scenario.target, *scenario.interferers)
    for source_index, source in enumerate(sources):
        freq_index = int(np.argmin(np.abs(frequencies_hz - float(source.frequency_hz))))
        modulation = 1.0 + 0.15 * np.cos(2.0 * np.pi * (source_index + 1) * time_sec + source_index)
        phase = np.exp(1j * 2.0 * np.pi * float(source.frequency_hz) * time_sec)
        mixed += (
            modulation[:, np.newaxis]
            * phase[:, np.newaxis]
            * response_by_source[freq_index, :, source_index][
                np.newaxis,
                :,
            ]
        )
    absolute_levels = _levels_db(mixed)
    relative_levels = absolute_levels - np.max(absolute_levels, axis=1, keepdims=True)
    return time_sec, np.asarray(relative_levels, dtype=np.float64)


def _row_float(row: dict[str, object], key: str) -> float:
    """summary row から数値 metric を取り出す。

    `dict[str, object]` は CSV/Markdown 生成で異種 scalar を保持するために使う。
    数値 metric を評価判定へ渡す前にここで型を確定し、Pylance が object を
    float と誤推論しない形にする。不正な値は評価入力の欠落として例外にする。
    """
    value = row.get(key)
    if isinstance(value, bool):
        raise TypeError(f"{key} must be numeric, got bool")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"{key} must be numeric, got {type(value).__name__}")


def _status_from_metrics(row: dict[str, object], method: str) -> str:
    """metric から方式判定を返す。"""
    if method == "fixed_baseline":
        return "fallback_baseline"
    if bool(row["fallback_required"]):
        return "fallback"

    # target mainlobe の 1 dB 超低下は、source 保護の失敗として fail にする。
    # source mask 内 peak だけを見ると、target 以外の source や近傍 peak で低下を隠す可能性がある。
    if _row_float(row, "target_mainlobe_delta_db") < -1.0:
        return "fail_target_loss"

    # non-source 局所悪化が 3 dB を超える場合は、BL/FRAZ 上の局所副作用としてレビュー対象にする。
    if _row_float(row, "max_local_worsening_db_gated") > 3.0:
        return "watch_non_source_worsening"

    interferer_frequency_hz = _row_float(row, "interferer_frequency_hz")
    has_interferer = math.isfinite(interferer_frequency_hz)
    if has_interferer and _row_float(row, "interferer_leakage_reduction_db") < 3.0:
        return "watch_low_leakage_reduction"
    return "pass"


def _evaluate_metrics(
    scenario: ScenarioDefinition,
    axis_azimuth_deg: FloatArray,
    frequencies_hz: FloatArray,
    responses_by_method: dict[str, ComplexArray],
    fraz_by_method: dict[str, FloatArray],
    diagnostics: dict[str, FloatArray | BoolArray],
    runtime_sec: float,
    *,
    duration_s: float,
) -> list[dict[str, object]]:
    """scenario_summary.csv の row を作る。

    BL 系の採否 metric は target 周波数の方位断面 `[n_beam]` で評価する。
    BTR は frame max 正規化なので、ここでは source track 連続性の図示に限定する。
    """
    rows: list[dict[str, object]] = []
    all_sources = (scenario.target, *scenario.interferers)
    target_beam_index = int(
        np.argmin(np.abs(axis_azimuth_deg - float(scenario.target.azimuth_deg)))
    )
    target_freq_index = int(np.argmin(np.abs(frequencies_hz - float(scenario.target.frequency_hz))))
    fixed_target_level = float(
        fraz_by_method["fixed_baseline"][target_beam_index, target_freq_index]
    )
    fixed_mixed_target_level = fixed_target_level

    # source mask は source 方位の guard 領域で、non-source metric はこの補集合だけを見る。
    # target 近傍の主ローブ低下と、非 source 領域の副作用を混同しないために分離する。
    source_mask = _source_mask(axis_azimuth_deg, all_sources)
    non_source_mask = np.logical_not(source_mask)
    if not bool(np.any(source_mask)):
        raise RuntimeError("source mask must include at least one beam")
    if not bool(np.any(non_source_mask)):
        raise RuntimeError("non-source sector must include at least one beam")
    source_count_detected = sum(
        1
        for source in all_sources
        if bool(
            np.any(np.abs(axis_azimuth_deg - float(source.azimuth_deg)) <= SOURCE_MASK_GUARD_DEG)
        )
    )
    source_count_expected = len(all_sources)

    fixed_bl = _bl_from_fraz(
        fraz_by_method["fixed_baseline"], frequencies_hz, float(scenario.target.frequency_hz)
    )
    fixed_source_peak = float(np.max(fixed_bl[source_mask]))
    fixed_non_source = fixed_bl[non_source_mask]
    fixed_non_source_peak = float(np.max(fixed_non_source))
    fixed_non_source_p95 = float(np.percentile(fixed_non_source, 95.0))
    fixed_non_source_p99 = float(np.percentile(fixed_non_source, 99.0))
    fixed_non_source_integrated = _integrated_level_db(fixed_non_source)
    fixed_margin = fixed_source_peak - fixed_non_source_peak
    fixed_false_peak_count = _false_peak_count(fixed_non_source, fixed_source_peak)

    fixed_interferer_leakage = 0.0
    interferer_azimuth = math.nan
    interferer_frequency = math.nan
    if scenario.interferers:
        interferer = scenario.interferers[0]
        interferer_azimuth = float(interferer.azimuth_deg)
        interferer_frequency = float(interferer.frequency_hz)
        interferer_index = 1
        interferer_freq_index = int(
            np.argmin(np.abs(frequencies_hz - float(interferer.frequency_hz)))
        )
        fixed_interferer_leakage = float(
            _levels_db(
                responses_by_method["fixed_baseline"][
                    interferer_freq_index,
                    target_beam_index,
                    interferer_index,
                ]
            )
        )

    condition_number = np.asarray(diagnostics["condition_number"], dtype=np.float64)
    fallback_mask = np.asarray(diagnostics["fallback_mask"], dtype=np.bool_)
    q_blocking_abs = np.asarray(diagnostics["q_blocking_abs"], dtype=np.float64)
    target_response_error = np.asarray(diagnostics["target_response_error"], dtype=np.float64)
    q_reconstruction_error = np.asarray(diagnostics["q_reconstruction_error"], dtype=np.float64)

    for method in METHOD_ORDER:
        bl = _bl_from_fraz(
            fraz_by_method[method], frequencies_hz, float(scenario.target.frequency_hz)
        )
        peak_index = int(np.argmax(bl))
        target_level = float(fraz_by_method[method][target_beam_index, target_freq_index])
        mixed_target_delta = target_level - fixed_mixed_target_level
        source_peak = float(np.max(bl[source_mask]))
        non_source = bl[non_source_mask]
        non_source_peak = float(np.max(non_source))
        non_source_p95 = float(np.percentile(non_source, 95.0))
        non_source_p99 = float(np.percentile(non_source, 99.0))
        non_source_integrated = _integrated_level_db(non_source)
        source_to_non_source_margin = source_peak - non_source_peak
        non_source_delta = non_source - fixed_non_source
        leakage_delta = 0.0
        leakage_reduction = 0.0
        if scenario.interferers:
            interferer = scenario.interferers[0]
            interferer_freq_index = int(
                np.argmin(np.abs(frequencies_hz - float(interferer.frequency_hz)))
            )
            interferer_level = float(
                _levels_db(responses_by_method[method][interferer_freq_index, target_beam_index, 1])
            )
            leakage_delta = interferer_level - fixed_interferer_leakage
            leakage_reduction = -leakage_delta

        row: dict[str, object] = {
            "scenario": scenario.scenario_id,
            "method": method,
            "mask_type": MASK_TYPE_LABEL,
            "candidate": method,
            "evaluation_pattern": scenario.evaluation_pattern,
            "status": "",
            "target_azimuth_deg": float(scenario.target.azimuth_deg),
            "target_frequency_hz": float(scenario.target.frequency_hz),
            "interferer_azimuth_deg": interferer_azimuth,
            "interferer_frequency_hz": interferer_frequency,
            "source_peak_delta_db": source_peak - fixed_source_peak,
            "source_azimuth_error_deg": abs(
                float(axis_azimuth_deg[peak_index]) - float(scenario.target.azimuth_deg)
            ),
            "non_source_global_peak_delta_db": non_source_peak - fixed_non_source_peak,
            "non_source_p95_level_delta_db": non_source_p95 - fixed_non_source_p95,
            "non_source_p99_level_delta_db": non_source_p99 - fixed_non_source_p99,
            "non_source_integrated_level_delta_db": (
                non_source_integrated - fixed_non_source_integrated
            ),
            "source_to_non_source_margin_delta_db": source_to_non_source_margin - fixed_margin,
            "false_peak_count_delta": (
                _false_peak_count(non_source, source_peak) - fixed_false_peak_count
            ),
            "max_local_worsening_db_gated": float(np.max(non_source_delta)),
            "target_mainlobe_delta_db": target_level - fixed_target_level,
            "target_peak_azimuth_error_deg": abs(
                float(axis_azimuth_deg[peak_index]) - float(scenario.target.azimuth_deg)
            ),
            "interferer_leakage_delta_db": leakage_delta,
            "interferer_leakage_reduction_db": leakage_reduction,
            "mixed_target_beam_delta_db": mixed_target_delta,
            "q_blocking_max_db": float(
                20.0 * np.log10(max(float(np.max(q_blocking_abs)), np.finfo(np.float64).tiny))
            ),
            "target_response_error_db": float(
                20.0
                * np.log10(max(float(np.max(target_response_error)), np.finfo(np.float64).tiny))
            ),
            "q_reconstruction_rms_error": float(np.max(q_reconstruction_error)),
            "loaded_condition_number_max": float(np.max(condition_number)),
            "source_count_expected": int(source_count_expected),
            "source_count_detected": int(source_count_detected),
            "fallback_required": bool(np.any(fallback_mask))
            if method != "fixed_baseline"
            else False,
            "fallback_reason": ""
            if method != "fixed_baseline"
            else "baseline retained as fallback",
            "runtime_factor": float(runtime_sec / float(duration_s)),
        }
        row["status"] = _status_from_metrics(row, method)
        rows.append(row)
    return rows


def _evaluate_scenario(
    scenario: ScenarioDefinition,
    array_positions_m: FloatArray,
    axis_azimuth_deg: FloatArray,
    beam_directions: FloatArray,
    frequencies_hz: FloatArray,
) -> ScenarioReviewData:
    """1 scenario を評価し、review 用データを返す。"""
    start_time = time.perf_counter()
    fs_hz = 32768.0
    sound_speed_m_s = 1500.0
    duration_s = 1.0
    filter_bank = design_standard_fractional_delay_filter_bank()
    delay_table = DelayTable.from_geometry(
        array_pos_m=array_positions_m,
        dir_cos=beam_directions,
        fs_hz=fs_hz,
        sound_speed_m_s=sound_speed_m_s,
        fractional_filter_bank=filter_bank,
    )
    fixed_weights = design_fixed_delay_fractional_weights_from_delay_table(
        delay_table,
        filter_bank,
        frequencies_hz,
        fs_hz=fs_hz,
        average_channels=True,
    )

    beam_sources = tuple(
        SourceSpec(
            f"beam_{beam_index:03d}",
            float(axis_azimuth_deg[beam_index]),
            float(frequencies_hz[0]),
            0.0,
        )
        for beam_index in range(axis_azimuth_deg.size)
    )
    steering_by_beam = np.stack(
        [
            _arrival_steering(
                array_positions_m, source, frequencies_hz, sound_speed_m_s=sound_speed_m_s
            )
            for source in beam_sources
        ],
        axis=1,
    )
    all_sources = (scenario.target, *scenario.interferers)
    steering_by_source = {
        source.label: _arrival_steering(
            array_positions_m, source, frequencies_hz, sound_speed_m_s=sound_speed_m_s
        )
        for source in all_sources
    }
    target_beam_index = int(
        np.argmin(np.abs(axis_azimuth_deg - float(scenario.target.azimuth_deg)))
    )
    weights_by_method, diagnostics = _design_method_weights(
        fixed_weights=fixed_weights,
        steering_by_beam=steering_by_beam,
        steering_by_source=steering_by_source,
        scenario=scenario,
        frequencies_hz=frequencies_hz,
        target_beam_index=target_beam_index,
    )

    responses_by_method = {
        method: _response_cube(weights, steering_by_source, all_sources, frequencies_hz)
        for method, weights in weights_by_method.items()
    }
    fraz_by_method = {
        method: _fraz_from_response(response) for method, response in responses_by_method.items()
    }
    bl_by_method = {
        method: _bl_from_fraz(fraz, frequencies_hz, float(scenario.target.frequency_hz))
        for method, fraz in fraz_by_method.items()
    }
    source_frequency_bl_by_method = {
        method: _source_frequency_bl_from_fraz(fraz, frequencies_hz, all_sources)
        for method, fraz in fraz_by_method.items()
    }
    time_sec = np.empty(0, dtype=np.float64)
    btr_by_method: dict[str, FloatArray] = {}
    for method, response in responses_by_method.items():
        time_sec, btr_by_method[method] = _btr_from_response(
            response,
            scenario,
            frequencies_hz,
            duration_s=duration_s,
            frame_count=32,
        )
    runtime_sec = time.perf_counter() - start_time
    rows = _evaluate_metrics(
        scenario,
        axis_azimuth_deg,
        frequencies_hz,
        responses_by_method,
        fraz_by_method,
        diagnostics,
        runtime_sec,
        duration_s=duration_s,
    )
    source_mask = _source_mask(axis_azimuth_deg, all_sources)
    return ScenarioReviewData(
        scenario=scenario,
        azimuth_deg=axis_azimuth_deg,
        frequency_hz=frequencies_hz,
        time_sec=time_sec,
        source_mask=source_mask,
        non_source_mask=np.logical_not(source_mask),
        bl_levels_db=bl_by_method,
        source_frequency_bl_levels_db=source_frequency_bl_by_method,
        fraz_levels_db=fraz_by_method,
        btr_levels_db=btr_by_method,
        rows=rows,
    )


def _plt() -> Any:
    """matplotlib.pyplot module を返す。"""
    require_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is required to build review pack figures.")
    return plt


def _mask_runs(mask: BoolArray) -> list[tuple[int, int]]:
    """bool mask の連続 run を返す。"""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(mask.size)))
    return runs


def _add_mask_spans(axis: Any, azimuth_deg: FloatArray, source_mask: BoolArray) -> None:
    """plot axis へ source mask / non-source sector を描画する。"""
    edges = centers_to_edges(azimuth_deg)
    for run_index, (start, stop) in enumerate(_mask_runs(np.logical_not(source_mask))):
        axis.axvspan(
            float(edges[start]),
            float(edges[stop]),
            color="0.92",
            alpha=0.45,
            linewidth=0.0,
            label="non-source sector" if run_index == 0 else None,
        )
    for run_index, (start, stop) in enumerate(_mask_runs(source_mask)):
        axis.axvspan(
            float(edges[start]),
            float(edges[stop]),
            color="tab:green",
            alpha=0.16,
            linewidth=0.0,
            label="source mask" if run_index == 0 else None,
        )


def _save_figure(fig: Any, path: Path) -> None:
    """figure を PNG 保存して閉じる。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _plot_bl_overlay(data: ScenarioReviewData, output_path: Path) -> None:
    """BL overlay を保存する。"""
    fig, axis = _plt().subplots(figsize=(10.5, 5.0))
    all_levels = np.concatenate([data.bl_levels_db[method] for method in METHOD_ORDER])
    _add_mask_spans(axis, data.azimuth_deg, data.source_mask)
    for method in METHOD_ORDER:
        axis.plot(
            data.azimuth_deg,
            data.bl_levels_db[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_ylim(float(np.min(all_levels) - 1.0), float(np.max(all_levels) + 1.0))
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel(f"RMS Level [{LEVEL_UNIT_LABEL}]")
    axis.set_title(f"{data.scenario.scenario_id}: BL overlay")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, output_path)


def _plot_source_frequency_bl_overlay(data: ScenarioReviewData, output_path: Path) -> None:
    """source 真値周波数集合で統合した BL overlay を保存する。"""
    fig, axis = _plt().subplots(figsize=(10.5, 5.0))
    all_levels = np.concatenate(
        [data.source_frequency_bl_levels_db[method] for method in METHOD_ORDER]
    )
    _add_mask_spans(axis, data.azimuth_deg, data.source_mask)
    for method in METHOD_ORDER:
        axis.plot(
            data.azimuth_deg,
            data.source_frequency_bl_levels_db[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_ylim(float(np.min(all_levels) - 1.0), float(np.max(all_levels) + 1.0))
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel(f"RMS Level [{LEVEL_UNIT_LABEL}]")
    axis.set_title(f"{data.scenario.scenario_id}: source-frequency BL overlay")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, output_path)

def _plot_bl_delta(data: ScenarioReviewData, output_path: Path) -> None:
    """BL delta を保存する。"""
    fig, axis = _plt().subplots(figsize=(10.5, 4.8))
    fixed = data.bl_levels_db["fixed_baseline"]
    deltas = {
        "mvdr_oracle": data.bl_levels_db["mvdr_oracle"] - fixed,
        "diff_mvdr_fir512": data.bl_levels_db["diff_mvdr_fir512"] - fixed,
    }
    max_abs = max(float(np.max(np.abs(delta))) for delta in deltas.values())
    _add_mask_spans(axis, data.azimuth_deg, data.source_mask)
    for method, delta in deltas.items():
        axis.plot(
            data.azimuth_deg,
            delta,
            color=METHOD_COLORS[method],
            label=f"{METHOD_LABELS[method]} - fixed",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylim(-(max_abs + 0.5), max_abs + 0.5)
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel("BL Delta [dB re fixed BL level]")
    axis.set_title(f"{data.scenario.scenario_id}: BL delta")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, output_path)


def _plot_fraz_delta(data: ScenarioReviewData, output_path: Path) -> None:
    """FRAZ delta を保存する。"""
    fixed = data.fraz_levels_db["fixed_baseline"]
    safe_delta = data.fraz_levels_db["mvdr_oracle"] - fixed
    fir_delta = data.fraz_levels_db["diff_mvdr_fir512"] - fixed
    max_abs = max(float(np.max(np.abs(safe_delta))), float(np.max(np.abs(fir_delta))), 1.0)
    az_edges = centers_to_edges(data.azimuth_deg)
    freq_edges = centers_to_edges(data.frequency_hz)
    fig, axes = _plt().subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    image = None
    for axis, method, delta in (
        (axes[0], "mvdr_oracle", safe_delta),
        (axes[1], "diff_mvdr_fir512", fir_delta),
    ):
        image = axis.pcolormesh(
            az_edges,
            freq_edges,
            delta.T,
            shading="flat",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
        _add_mask_spans(axis, data.azimuth_deg, data.source_mask)
        axis.set_title(f"{METHOD_LABELS[method]} - fixed")
        axis.set_xlabel("Azimuth [deg]")
    axes[0].set_ylabel("Frequency [Hz]")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="FRAZ Delta [dB re fixed FRAZ level]")
    fig.suptitle(f"{data.scenario.scenario_id}: FRAZ delta")
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.14, top=0.86, wspace=0.22)
    _save_figure(fig, output_path)


def _plot_btr_panel(data: ScenarioReviewData, output_path: Path) -> None:
    """BTR panel を保存する。"""
    az_edges = centers_to_edges(data.azimuth_deg)
    time_edges = centers_to_edges(data.time_sec)
    fig, axes = _plt().subplots(1, 3, figsize=(14.0, 4.8), sharey=True)
    image = None
    for axis, method in zip(axes, METHOD_ORDER, strict=True):
        image = axis.pcolormesh(
            az_edges,
            time_edges,
            data.btr_levels_db[method],
            shading="flat",
            cmap="viridis",
            vmin=-12.0,
            vmax=0.0,
        )
        _add_mask_spans(axis, data.azimuth_deg, data.source_mask)
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Azimuth [deg]")
    axes[0].set_ylabel("Time [s]")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=f"Relative Level [{BTR_UNIT_LABEL}]")
    fig.suptitle(f"{data.scenario.scenario_id}: BTR source-track continuity")
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.14, top=0.86, wspace=0.22)
    _save_figure(fig, output_path)


def _save_npz(data: ScenarioReviewData, output_path: Path) -> None:
    """BL / FRAZ / BTR の描画前配列を npz 保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        azimuth_deg=data.azimuth_deg,
        frequency_hz=data.frequency_hz,
        time_sec=data.time_sec,
        fixed_level_db=data.bl_levels_db["fixed_baseline"],
        mvdr_oracle_level_db=data.bl_levels_db["mvdr_oracle"],
        diff_mvdr_fir512_level_db=data.bl_levels_db["diff_mvdr_fir512"],
        fixed_source_frequency_level_db=data.source_frequency_bl_levels_db["fixed_baseline"],
        mvdr_oracle_source_frequency_level_db=data.source_frequency_bl_levels_db["mvdr_oracle"],
        diff_mvdr_fir512_source_frequency_level_db=data.source_frequency_bl_levels_db[
            "diff_mvdr_fir512"
        ],
        fixed_fraz_level_db=data.fraz_levels_db["fixed_baseline"],
        mvdr_oracle_fraz_level_db=data.fraz_levels_db["mvdr_oracle"],
        diff_mvdr_fir512_fraz_level_db=data.fraz_levels_db["diff_mvdr_fir512"],
        fixed_btr_level_db=data.btr_levels_db["fixed_baseline"],
        mvdr_oracle_btr_level_db=data.btr_levels_db["mvdr_oracle"],
        diff_mvdr_fir512_btr_level_db=data.btr_levels_db["diff_mvdr_fir512"],
        source_mask=data.source_mask,
        non_source_mask=data.non_source_mask,
    )


def _sanitize_csv_value(value: object) -> object:
    """CSV 保存用 scalar へ整形する。"""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_csv(rows: list[dict[str, object]], path: Path, columns: tuple[str, ...] | None) -> None:
    """CSV を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        columns if columns is not None else tuple(sorted({key for row in rows for key in row}))
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _sanitize_csv_value(row.get(key, "")) for key in fieldnames})


def _build_worst_cases(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """worst_cases.csv の row を作る。"""
    rows: list[dict[str, object]] = []
    metric_specs = (
        ("source_peak_delta_db", False),
        ("source_azimuth_error_deg", True),
        ("non_source_global_peak_delta_db", True),
        ("non_source_p95_level_delta_db", True),
        ("non_source_p99_level_delta_db", True),
        ("non_source_integrated_level_delta_db", True),
        ("source_to_non_source_margin_delta_db", False),
        ("false_peak_count_delta", True),
        ("max_local_worsening_db_gated", True),
        ("target_mainlobe_delta_db", False),
        ("interferer_leakage_reduction_db", False),
        ("q_blocking_max_db", True),
        ("target_response_error_db", True),
        ("q_reconstruction_rms_error", True),
        ("loaded_condition_number_max", True),
        ("runtime_factor", True),
    )
    candidate_rows = [row for row in summary_rows if row["method"] != "fixed_baseline"]
    for metric, descending in metric_specs:
        ranked = sorted(
            candidate_rows,
            key=lambda row: _row_float(row, metric),
            reverse=descending,
        )
        for rank, row in enumerate(ranked[:10], start=1):
            rows.append(
                {
                    "category": "metric_worst_top10",
                    "metric": metric,
                    "rank": rank,
                    "scenario": row["scenario"],
                    "method": row["method"],
                    "value": row[metric],
                    "status": row["status"],
                    "details": "descending" if descending else "ascending",
                }
            )

    # source 数の検出ズレは mask 作成の不備を示すため、方式評価の前提エラーとして分ける。
    for row in summary_rows:
        if _row_float(row, "source_count_expected") != _row_float(row, "source_count_detected"):
            rows.append(
                {
                    "category": "mask_source_count_mismatch",
                    "metric": "source_count_detected",
                    "rank": "",
                    "scenario": row["scenario"],
                    "method": row["method"],
                    "value": row["source_count_detected"],
                    "status": row["status"],
                    "details": f"expected={row['source_count_expected']}",
                }
            )
        if bool(row["fallback_required"]):
            rows.append(
                {
                    "category": "fallback_rows",
                    "metric": "fallback_required",
                    "rank": "",
                    "scenario": row["scenario"],
                    "method": row["method"],
                    "value": True,
                    "status": row["status"],
                    "details": row["fallback_reason"],
                }
            )
        if str(row["status"]).startswith("fail") or str(row["status"]).startswith("watch"):
            rows.append(
                {
                    "category": "negative_or_watch_rows",
                    "metric": "status",
                    "rank": "",
                    "scenario": row["scenario"],
                    "method": row["method"],
                    "value": row["status"],
                    "status": row["status"],
                    "details": "review before adopting method",
                }
            )

    rows_by_key = {(str(row["scenario"]), str(row["method"])): row for row in summary_rows}
    difference_rows: list[dict[str, object]] = []
    diff_metrics = (
        "source_peak_delta_db",
        "non_source_global_peak_delta_db",
        "source_to_non_source_margin_delta_db",
        "interferer_leakage_reduction_db",
        "max_local_worsening_db_gated",
    )
    for scenario in sorted({str(row["scenario"]) for row in summary_rows}):
        mvdr_row = rows_by_key.get((scenario, "mvdr_oracle"))
        diff_row = rows_by_key.get((scenario, "diff_mvdr_fir512"))
        if mvdr_row is None or diff_row is None:
            continue
        for metric in diff_metrics:
            difference_rows.append(
                {
                    "category": "mvdr_vs_diff_large_delta_rows",
                    "metric": metric,
                    "rank": "",
                    "scenario": scenario,
                    "method": "diff_mvdr_fir512 - mvdr_oracle",
                    "value": abs(_row_float(diff_row, metric) - _row_float(mvdr_row, metric)),
                    "status": diff_row["status"],
                    "details": "absolute metric difference",
                }
            )
    difference_rows.sort(key=lambda row: _row_float(row, "value"), reverse=True)
    for rank, row in enumerate(difference_rows[:10], start=1):
        row["rank"] = rank
        rows.append(row)
    return rows


def _relative(path: Path) -> str:
    """review_pack からの相対 path を返す。"""
    return path.relative_to(REVIEW_PACK_DIR).as_posix()


def _write_review_index(data_items: list[ScenarioReviewData]) -> None:
    """review_index.md を保存する。"""
    lines = [
        "# Fixed Delay + Difference MVDR Review Pack",
        "",
        "## 読み方",
        "",
        "- methods: `fixed_baseline`, `mvdr_oracle`, `diff_mvdr_fir512`。",
        "- fixed_baseline は常に fallback として残す。",
        f"- BL / FRAZ は `{LEVEL_UNIT_LABEL}`。delta は fixed に対する相対 dB。",
        f"- BTR は `{BTR_UNIT_LABEL}` であり、抑圧量ではなく source track 連続性確認用。",
        "- source-frequency BL overlay は全 scenario で必ず生成し、source visibility 確認に使う。",
        "- 採否は target-only、interferer-only 相当 metric、mixed target beam を分けて見る。",
        (
            f"- mask type: `{MASK_TYPE_LABEL}`。source 方位±{SOURCE_MASK_GUARD_DEG:.1f} deg "
            "を source mask とする。"
        ),
        "",
        "## ファイル",
        "",
        f"- scenario summary: `{_relative(SCENARIO_SUMMARY_PATH)}`",
        f"- worst cases: `{_relative(WORST_CASES_PATH)}`",
        "- figures: `figures/<scenario>/`",
        "- plot arrays: `data/<scenario>.npz`",
        "",
        "## Scenarios",
        "",
    ]
    for data in data_items:
        lines.extend(
            [
                f"### {data.scenario.scenario_id}",
                "",
                f"- 目的: {data.scenario.purpose}",
                f"- evaluation pattern: `{data.scenario.evaluation_pattern}`",
                f"- mask type: `{MASK_TYPE_LABEL}`",
                (
                    "- target: "
                    f"az={data.scenario.target.azimuth_deg:.1f} deg, "
                    f"f={data.scenario.target.frequency_hz:.1f} Hz"
                ),
                "- interferer: "
                + (
                    ", ".join(
                        [
                            f"az={source.azimuth_deg:.1f} deg, f={source.frequency_hz:.1f} Hz"
                            for source in data.scenario.interferers
                        ]
                    )
                    if data.scenario.interferers
                    else "none"
                ),
                "",
                "| method | status | source peak delta | leakage reduction | runtime factor |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in data.rows:
            lines.append(
                (
                    f"| `{row['method']}` | `{row['status']}` | "
                    f"{_row_float(row, 'source_peak_delta_db'):.3f} | "
                    f"{_row_float(row, 'interferer_leakage_reduction_db'):.3f} | "
                    f"{_row_float(row, 'runtime_factor'):.4f} |"
                )
            )
        scenario_dir = FIGURE_DIR / data.scenario.scenario_id
        lines.extend(
            [
                "",
                "参照:",
                f"- BL overlay: `{_relative(scenario_dir / 'bl_overlay.png')}`",
                (
                    "- source-frequency BL overlay: "
                    f"`{_relative(scenario_dir / 'source_frequency_bl_overlay.png')}`"
                ),
                f"- BL delta: `{_relative(scenario_dir / 'bl_delta.png')}`",
                f"- FRAZ delta: `{_relative(scenario_dir / 'fraz_delta.png')}`",
                f"- BTR panel: `{_relative(scenario_dir / 'btr_panel.png')}`",
                f"- plot arrays: `{_relative(DATA_DIR / f'{data.scenario.scenario_id}.npz')}`",
                "",
            ]
        )
    REVIEW_INDEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_review_pack() -> None:
    """review_pack を生成する。"""
    require_matplotlib()
    if REVIEW_PACK_DIR.exists():
        shutil.rmtree(REVIEW_PACK_DIR)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    array_positions = _build_array_positions(n_ch=32, spacing_m=0.05)
    directions, axis_azimuth_deg, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=121,
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    beam_directions = directions.T.astype(np.float64)
    # 低周波条件と高周波条件を同じ review_pack で比較するため、
    # 4096 Hz / 5120 Hz を含む評価周波数軸にする。
    base_frequencies_hz = np.arange(768.0, 6144.0 + 1.0, 128.0, dtype=np.float64)
    near_frequency_points_hz = np.array(
        [
            1536.0 + NEAR_FREQUENCY_OFFSET_HZ,
            4096.0 + NEAR_FREQUENCY_OFFSET_HZ,
        ],
        dtype=np.float64,
    )
    frequencies_hz = np.unique(np.concatenate([base_frequencies_hz, near_frequency_points_hz]))
    data_items: list[ScenarioReviewData] = []
    for scenario in _build_scenarios():
        data = _evaluate_scenario(
            scenario,
            array_positions,
            axis_azimuth_deg.astype(np.float64),
            beam_directions,
            frequencies_hz,
        )
        data_items.append(data)
        scenario_dir = FIGURE_DIR / scenario.scenario_id
        _plot_bl_overlay(data, scenario_dir / "bl_overlay.png")
        _plot_source_frequency_bl_overlay(data, scenario_dir / "source_frequency_bl_overlay.png")
        _plot_bl_delta(data, scenario_dir / "bl_delta.png")
        _plot_fraz_delta(data, scenario_dir / "fraz_delta.png")
        _plot_btr_panel(data, scenario_dir / "btr_panel.png")
        _save_npz(data, DATA_DIR / f"{scenario.scenario_id}.npz")

    summary_rows = [row for data in data_items for row in data.rows]
    _write_csv(summary_rows, SCENARIO_SUMMARY_PATH, SUMMARY_COLUMNS)
    _write_csv(_build_worst_cases(summary_rows), WORST_CASES_PATH, None)
    _write_review_index(data_items)


def main() -> None:
    """CLI entrypoint。"""
    build_review_pack()
    print(f"saved review index to {REVIEW_INDEX_PATH}")
    print(f"saved scenario summary to {SCENARIO_SUMMARY_PATH}")
    print(f"saved worst cases to {WORST_CASES_PATH}")


if __name__ == "__main__":
    main()
