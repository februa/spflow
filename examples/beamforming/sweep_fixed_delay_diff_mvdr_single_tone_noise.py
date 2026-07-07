"""単一周波数 source + チャネル無相関雑音の方位・周波数 sweep を生成する。"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from examples.beamforming.build_fixed_delay_diff_mvdr_review_pack import (  # noqa: E402
    LEVEL_UNIT_LABEL,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    SourceSpec,
    _arrival_steering,
    _build_array_positions,
    _levels_db,
    _source_mask,
)
from spflow.beamforming import (  # noqa: E402
    DelayTable,
    DifferenceCorrectionFIRDesigner,
    LoadedMVDRWeightDesigner,
    design_fixed_delay_fractional_weights_from_delay_table,
    design_standard_fractional_delay_filter_bank,
    make_directions,
)
from spflow.beamforming.diagnostic_plotting import (  # noqa: E402
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

OUTPUT_DIR = (
    ROOT / "artifacts" / "beamforming" / "fixed_delay_diff_mvdr" / "single_tone_noise_sweep"
)
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
SUMMARY_CSV_PATH = OUTPUT_DIR / "single_tone_noise_sweep.csv"
WORST_CASES_CSV_PATH = OUTPUT_DIR / "worst_cases.csv"
REPORT_MD_PATH = OUTPUT_DIR / "single_tone_noise_sweep_report.md"
ARRAY_NPZ_PATH = DATA_DIR / "single_tone_noise_sweep_arrays.npz"

FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
FIR_TAPS = 512
NOISE_POWER_PER_CHANNEL = 1.0e-2
SOURCE_LEVEL_DB = 0.0
SOURCE_MASK_GUARD_DEG = 3.0
FALSE_PEAK_MARGIN_DB = 3.0
N_CH = 32
SENSOR_SPACING_M = 0.05
FREQUENCY_HZ_VALUES = (
    768.0,
    1024.0,
    1536.0,
    2048.0,
    3072.0,
    4096.0,
    5120.0,
    6144.0,
)
SOURCE_AZIMUTH_DEG_VALUES = (
    10.0,
    20.0,
    30.0,
    45.0,
    60.0,
    75.0,
    90.0,
    105.0,
    120.0,
    135.0,
    150.0,
    170.0,
)
REPRESENTATIVE_FREQUENCY_HZ = 4096.0
REPRESENTATIVE_AZIMUTH_DEG = 60.0
SUMMARY_COLUMNS = (
    "frequency_hz",
    "source_azimuth_deg",
    "nearest_waiting_azimuth_deg",
    "source_to_waiting_azimuth_error_deg",
    "method",
    "status",
    "source_peak_level_db",
    "source_peak_delta_db",
    "source_azimuth_error_deg",
    "snr_input_db",
    "snr_output_db",
    "snr_gain_db",
    "snr_gain_delta_db",
    "noise_output_level_db",
    "sidelobe_margin_db",
    "sidelobe_margin_delta_db",
    "non_source_peak_level_db",
    "false_peak_count",
    "fallback_count",
    "q_reconstruction_rms_error",
    "loaded_condition_number_max",
)

METHOD_DISPLAY_LABELS = {
    **METHOD_LABELS,
    "mvdr_oracle": "MVDR freq ref",
}



@dataclass(frozen=True)
class SingleToneSweepRow:
    """単一 tone + 無相関雑音 sweep の 1 行を表す。

    周波数・方位・method ごとの peak、SNR、sidelobe、差分 FIR 診断を保持する。
    入力は 1 条件の scan 出力、出力は CSV / Markdown / heatmap の scalar metric である。
    波形生成や重み設計は責務に含めない。
    信号処理上は `fixed_beam_single_source` の集計結果である。
    """

    frequency_hz: float
    source_azimuth_deg: float
    nearest_waiting_azimuth_deg: float
    source_to_waiting_azimuth_error_deg: float
    method: str
    status: str
    source_peak_level_db: float
    source_peak_delta_db: float
    source_azimuth_error_deg: float
    snr_input_db: float
    snr_output_db: float
    snr_gain_db: float
    snr_gain_delta_db: float
    noise_output_level_db: float
    sidelobe_margin_db: float
    sidelobe_margin_delta_db: float
    non_source_peak_level_db: float
    false_peak_count: int
    fallback_count: int
    q_reconstruction_rms_error: float
    loaded_condition_number_max: float


@dataclass(frozen=True)
class ConditionResult:
    """1 つの source 条件に対する scan 配列と CSV 行を保持する。"""

    rows: list[SingleToneSweepRow]
    signal_level_db_by_method: dict[str, FloatArray]
    source_mask: BoolArray
    non_source_mask: BoolArray


def _db10(power: NDArray[Any] | float) -> FloatArray:
    """power 比を dB10 へ変換する。"""
    power_array = np.asarray(power, dtype=np.float64)
    return np.asarray(
        10.0 * np.log10(np.maximum(power_array, np.finfo(np.float64).tiny)),
        dtype=np.float64,
    )


def _scalar_db10(power: float) -> float:
    """scalar power 比を dB10 へ変換する。"""
    return float(_db10(float(power)).item())


def _mask_runs(mask: BoolArray) -> list[tuple[int, int]]:
    """boolean mask の連続 run を `[start, stop)` で返す。"""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if bool(value) and run_start is None:
            run_start = int(index)
        elif not bool(value) and run_start is not None:
            runs.append((run_start, int(index)))
            run_start = None
    if run_start is not None:
        runs.append((run_start, int(mask.size)))
    return runs


def _single_source_covariance(source_steering: ComplexArray) -> ComplexArray:
    """単一 source + チャネル無相関雑音の空間共分散を返す。

    Args:
        source_steering: 評価周波数の source steering。shape は [n_ch]。

    Returns:
        空間共分散。shape は [n_ch, n_ch]。

    境界条件:
        この sweep は target 自己抑圧を検出するため、source を共分散へ含める。
        source を除外すると自己抑圧リスクを評価できないため、ここでは行わない。
    """
    n_ch = int(source_steering.shape[0])
    source_power = float(10.0 ** (SOURCE_LEVEL_DB / 10.0))
    # R = sigma_s^2 a a^H + sigma_n^2 I。
    # target source を統計へ含めたうえで、protected target beam の制約で自己抑圧を防げるかを見る。
    return np.asarray(
        source_power * np.outer(source_steering, source_steering.conj())
        + NOISE_POWER_PER_CHANNEL * np.eye(n_ch, dtype=np.complex128),
        dtype=np.complex128,
    )


def _signal_response(
    weights_at_frequency: ComplexArray,
    source_steering_at_frequency: ComplexArray,
) -> ComplexArray:
    """beam ごとの source 複素応答 `w_b^H a_s` を返す。

    Args:
        weights_at_frequency: beamformer 重み。shape は [n_beam, n_ch]。
        source_steering_at_frequency: source steering。shape は [n_ch]。

    Returns:
        source 複素応答。shape は [n_beam]。
    """
    # einsum の b は beam、c は sensor channel を表す。
    return np.asarray(
        np.einsum(
            "bc,c->b",
            weights_at_frequency.conj(),
            source_steering_at_frequency,
            optimize=True,
        ),
        dtype=np.complex128,
    )


def _noise_output_power(weights_at_frequency: ComplexArray) -> FloatArray:
    """白色・チャネル無相関雑音の beam 出力 power を返す。"""
    # sigma_out^2 = sigma_n^2 sum_ch |w_ch|^2。axis=1 は sensor channel。
    return np.asarray(
        NOISE_POWER_PER_CHANNEL * np.sum(np.abs(weights_at_frequency) ** 2, axis=1),
        dtype=np.float64,
    )


def _design_scan_weights(
    fixed_weights: ComplexArray,
    steering_by_beam: ComplexArray,
    source_steering: ComplexArray,
    frequency_hz: float,
) -> tuple[dict[str, ComplexArray], dict[str, FloatArray | BoolArray]]:
    """fixed / MVDR oracle / diff MVDR FIR512 の scan 重みを返す。

    各重みの shape は [n_freq, n_beam, n_ch]。単一周波数評価なので n_freq は 1 である。
    各 beam の待ち受け方位 steering を制約に使い、真値を使わない scan 条件を評価する。
    """
    n_freq, n_beam, _ = fixed_weights.shape
    covariance = np.repeat(
        _single_source_covariance(source_steering[0])[np.newaxis, :, :],
        n_freq,
        axis=0,
    )
    frequencies_hz = np.array([float(frequency_hz)], dtype=np.float64)
    mvdr_weights = np.zeros_like(fixed_weights)
    diff_weights = np.zeros_like(fixed_weights)
    condition_number = np.zeros((n_freq, n_beam), dtype=np.float64)
    fallback_mask = np.zeros((n_freq, n_beam), dtype=np.bool_)
    q_error = np.zeros((n_freq, n_beam), dtype=np.float64)
    mvdr_designer = LoadedMVDRWeightDesigner(diagonal_loading_ratio=1.0e-2)
    diff_designer = DifferenceCorrectionFIRDesigner(
        fir_taps=FIR_TAPS,
        frequencies_hz=frequencies_hz,
        fs_hz=FS_HZ,
    )
    for beam_index in range(n_beam):
        # 実運用では source 真値方位は未知である。
        # そのため MVDR の歪みなし制約は、scan 中の各 beam が待ち受ける方位 θ_beam の
        # steering a(θ_beam, f) に置く。source が beam center からずれる場合は、
        # その steering mismatch も評価結果に含める。
        protected_steering = steering_by_beam[:, beam_index, :]
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
        diff_weights[:, beam_index, :] = diff_result.final_weight_freq
        condition_number[:, beam_index] = mvdr_result.loaded_condition_number
        fallback_mask[:, beam_index] = mvdr_result.fallback_mask
        # q_reconstruction_error shape: [n_freq, n_ch]。channel 軸 RMS を beam 診断値にする。
        q_error[:, beam_index] = np.sqrt(
            np.mean(
                np.abs(diff_result.diagnostics.q_reconstruction_error) ** 2,
                axis=1,
            )
        )
    return (
        {
            "fixed_baseline": fixed_weights,
            "mvdr_oracle": mvdr_weights,
            "diff_mvdr_fir512": diff_weights,
        },
        {
            "condition_number": condition_number,
            "fallback_mask": fallback_mask,
            "q_reconstruction_error": q_error,
        },
    )


def _method_status(row: SingleToneSweepRow) -> str:
    """metric から status を返す。"""
    if row.method == "fixed_baseline":
        return "fallback_baseline"
    if row.fallback_count > 0:
        return "fallback"
    if row.source_peak_delta_db < -1.0:
        return "fail_source_loss"
    if row.source_azimuth_error_deg > SOURCE_MASK_GUARD_DEG:
        return "fail_peak_shift"
    if row.snr_gain_delta_db < -0.5:
        return "watch_snr_loss"
    if row.sidelobe_margin_delta_db < -1.0:
        return "watch_sidelobe_margin_loss"
    return "pass"


def _mainlobe_mask_from_fixed_bl(
    fixed_signal_level_db: FloatArray,
    axis_azimuth_deg: FloatArray,
    source: SourceSpec,
) -> BoolArray:
    """fixed BL の -3 dB 主ローブから source mask を作る。

    Args:
        fixed_signal_level_db: fixed の signal-only BL。shape は [n_beam]。
        axis_azimuth_deg: beam 方位。shape は [n_beam]、単位は deg。
        source: source 真値。方位は deg。

    Returns:
        source mask。shape は [n_beam]。

    境界条件:
        低周波では主ローブが ±3 deg より大きく広がる。
        固定幅 guard だけでは主ローブを non-source と誤判定するため、
        fixed peak を含む -3 dB 連続領域を source mask に含める。
    """
    truth_guard_mask = _source_mask(
        axis_azimuth_deg,
        (source,),
        guard_deg=SOURCE_MASK_GUARD_DEG,
    )
    peak_index = int(np.argmax(fixed_signal_level_db))
    threshold_db = float(fixed_signal_level_db[peak_index]) - 3.0
    above_threshold = fixed_signal_level_db >= threshold_db

    start = peak_index
    while start > 0 and bool(above_threshold[start - 1]):
        start -= 1
    stop = peak_index + 1
    while stop < int(above_threshold.size) and bool(above_threshold[stop]):
        stop += 1

    mainlobe_mask = np.zeros(axis_azimuth_deg.shape, dtype=np.bool_)
    mainlobe_mask[start:stop] = True
    return np.asarray(mainlobe_mask | truth_guard_mask, dtype=np.bool_)


def _evaluate_condition(
    frequency_hz: float,
    source_azimuth_deg: float,
    array_positions_m: FloatArray,
    axis_azimuth_deg: FloatArray,
    beam_directions: FloatArray,
) -> ConditionResult:
    """1 つの周波数・方位条件を評価する。"""
    frequencies_hz = np.array([float(frequency_hz)], dtype=np.float64)
    source = SourceSpec("source", float(source_azimuth_deg), float(frequency_hz), SOURCE_LEVEL_DB)
    filter_bank = design_standard_fractional_delay_filter_bank()
    delay_table = DelayTable.from_geometry(
        array_pos_m=array_positions_m,
        dir_cos=beam_directions,
        fs_hz=FS_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        fractional_filter_bank=filter_bank,
    )
    fixed_weights = design_fixed_delay_fractional_weights_from_delay_table(
        delay_table,
        filter_bank,
        frequencies_hz,
        fs_hz=FS_HZ,
        average_channels=True,
    )
    beam_sources = tuple(
        SourceSpec(f"beam_{idx}", float(az), float(frequency_hz), 0.0)
        for idx, az in enumerate(axis_azimuth_deg.tolist())
    )
    steering_by_beam = np.stack(
        [
            _arrival_steering(
                array_positions_m,
                beam_source,
                frequencies_hz,
                sound_speed_m_s=SOUND_SPEED_M_S,
            )
            for beam_source in beam_sources
        ],
        axis=1,
    )
    source_steering = _arrival_steering(
        array_positions_m,
        source,
        frequencies_hz,
        sound_speed_m_s=SOUND_SPEED_M_S,
    )
    target_beam_index = int(np.argmin(np.abs(axis_azimuth_deg - float(source_azimuth_deg))))
    nearest_waiting_azimuth_deg = float(axis_azimuth_deg[target_beam_index])
    source_to_waiting_azimuth_error_deg = abs(
        nearest_waiting_azimuth_deg - float(source_azimuth_deg)
    )
    weights_by_method, diagnostics = _design_scan_weights(
        fixed_weights,
        steering_by_beam,
        source_steering,
        float(frequency_hz),
    )

    input_snr_db = _scalar_db10(float(10.0 ** (SOURCE_LEVEL_DB / 10.0)) / NOISE_POWER_PER_CHANNEL)
    signal_level_by_method: dict[str, FloatArray] = {}
    snr_output_by_method: dict[str, FloatArray] = {}
    noise_power_by_method: dict[str, FloatArray] = {}
    for method in METHOD_ORDER:
        weights_at_frequency = weights_by_method[method][0]
        response = _signal_response(weights_at_frequency, source_steering[0])
        signal_level_by_method[method] = np.asarray(_levels_db(response), dtype=np.float64)
        signal_power = np.asarray(np.abs(response) ** 2, dtype=np.float64)
        noise_power = _noise_output_power(weights_at_frequency)
        noise_power_by_method[method] = noise_power
        snr_output_by_method[method] = _db10(
            signal_power / np.maximum(noise_power, np.finfo(np.float64).tiny)
        )

    source_mask = _mainlobe_mask_from_fixed_bl(
        signal_level_by_method["fixed_baseline"],
        axis_azimuth_deg,
        source,
    )
    non_source_mask = np.logical_not(source_mask)
    pending_rows: dict[str, SingleToneSweepRow] = {}
    for method in METHOD_ORDER:
        signal_level_db = signal_level_by_method[method]
        snr_output_db = snr_output_by_method[method]
        noise_power = noise_power_by_method[method]
        peak_index = int(np.argmax(signal_level_db))
        source_peak_level_db = float(signal_level_db[peak_index])
        non_source_level_db = signal_level_db[non_source_mask]
        non_source_peak_level_db = float(np.max(non_source_level_db))
        sidelobe_margin_db = source_peak_level_db - non_source_peak_level_db
        threshold_db = source_peak_level_db - FALSE_PEAK_MARGIN_DB
        fallback_count = (
            int(np.count_nonzero(np.asarray(diagnostics["fallback_mask"])))
            if method != "fixed_baseline"
            else 0
        )
        q_error = (
            float(np.max(np.asarray(diagnostics["q_reconstruction_error"])))
            if method == "diff_mvdr_fir512"
            else 0.0
        )
        condition_number = (
            float(np.max(np.asarray(diagnostics["condition_number"])))
            if method != "fixed_baseline"
            else 0.0
        )
        pending_rows[method] = SingleToneSweepRow(
            frequency_hz=float(frequency_hz),
            source_azimuth_deg=float(source_azimuth_deg),
            nearest_waiting_azimuth_deg=nearest_waiting_azimuth_deg,
            source_to_waiting_azimuth_error_deg=source_to_waiting_azimuth_error_deg,
            method=method,
            status="pending",
            source_peak_level_db=source_peak_level_db,
            source_peak_delta_db=0.0,
            source_azimuth_error_deg=abs(
                float(axis_azimuth_deg[peak_index]) - float(source_azimuth_deg)
            ),
            snr_input_db=input_snr_db,
            snr_output_db=float(snr_output_db[peak_index]),
            snr_gain_db=float(snr_output_db[peak_index]) - input_snr_db,
            snr_gain_delta_db=0.0,
            noise_output_level_db=float(_db10(noise_power[peak_index]).item()),
            sidelobe_margin_db=sidelobe_margin_db,
            sidelobe_margin_delta_db=0.0,
            non_source_peak_level_db=non_source_peak_level_db,
            false_peak_count=int(np.count_nonzero(non_source_level_db >= threshold_db)),
            fallback_count=fallback_count,
            q_reconstruction_rms_error=q_error,
            loaded_condition_number_max=condition_number,
        )

    fixed = pending_rows["fixed_baseline"]
    rows: list[SingleToneSweepRow] = []
    for method in METHOD_ORDER:
        row = pending_rows[method]
        adjusted = SingleToneSweepRow(
            frequency_hz=row.frequency_hz,
            source_azimuth_deg=row.source_azimuth_deg,
            nearest_waiting_azimuth_deg=row.nearest_waiting_azimuth_deg,
            source_to_waiting_azimuth_error_deg=row.source_to_waiting_azimuth_error_deg,
            method=row.method,
            status="pending",
            source_peak_level_db=row.source_peak_level_db,
            source_peak_delta_db=row.source_peak_level_db - fixed.source_peak_level_db,
            source_azimuth_error_deg=row.source_azimuth_error_deg,
            snr_input_db=row.snr_input_db,
            snr_output_db=row.snr_output_db,
            snr_gain_db=row.snr_gain_db,
            snr_gain_delta_db=row.snr_gain_db - fixed.snr_gain_db,
            noise_output_level_db=row.noise_output_level_db,
            sidelobe_margin_db=row.sidelobe_margin_db,
            sidelobe_margin_delta_db=row.sidelobe_margin_db - fixed.sidelobe_margin_db,
            non_source_peak_level_db=row.non_source_peak_level_db,
            false_peak_count=row.false_peak_count,
            fallback_count=row.fallback_count,
            q_reconstruction_rms_error=row.q_reconstruction_rms_error,
            loaded_condition_number_max=row.loaded_condition_number_max,
        )
        rows.append(
            SingleToneSweepRow(
                frequency_hz=adjusted.frequency_hz,
                source_azimuth_deg=adjusted.source_azimuth_deg,
                nearest_waiting_azimuth_deg=adjusted.nearest_waiting_azimuth_deg,
                source_to_waiting_azimuth_error_deg=adjusted.source_to_waiting_azimuth_error_deg,
                method=adjusted.method,
                status=_method_status(adjusted),
                source_peak_level_db=adjusted.source_peak_level_db,
                source_peak_delta_db=adjusted.source_peak_delta_db,
                source_azimuth_error_deg=adjusted.source_azimuth_error_deg,
                snr_input_db=adjusted.snr_input_db,
                snr_output_db=adjusted.snr_output_db,
                snr_gain_db=adjusted.snr_gain_db,
                snr_gain_delta_db=adjusted.snr_gain_delta_db,
                noise_output_level_db=adjusted.noise_output_level_db,
                sidelobe_margin_db=adjusted.sidelobe_margin_db,
                sidelobe_margin_delta_db=adjusted.sidelobe_margin_delta_db,
                non_source_peak_level_db=adjusted.non_source_peak_level_db,
                false_peak_count=adjusted.false_peak_count,
                fallback_count=adjusted.fallback_count,
                q_reconstruction_rms_error=adjusted.q_reconstruction_rms_error,
                loaded_condition_number_max=adjusted.loaded_condition_number_max,
            )
        )
    return ConditionResult(
        rows=rows,
        signal_level_db_by_method=signal_level_by_method,
        source_mask=source_mask,
        non_source_mask=non_source_mask,
    )


def _row_dicts(rows: list[SingleToneSweepRow]) -> list[dict[str, object]]:
    """dataclass 行を CSV 用 dict に変換する。"""
    return [{field: getattr(row, field) for field in SUMMARY_COLUMNS} for row in rows]


def _write_csv(rows: list[dict[str, object]], path: Path, columns: tuple[str, ...]) -> None:
    """CSV を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _metric_cube(rows: list[SingleToneSweepRow], metric: str) -> FloatArray:
    """method×frequency×azimuth の metric cube を返す。"""
    cube = np.zeros(
        (len(METHOD_ORDER), len(FREQUENCY_HZ_VALUES), len(SOURCE_AZIMUTH_DEG_VALUES)),
        dtype=np.float64,
    )
    row_by_key = {(row.method, row.frequency_hz, row.source_azimuth_deg): row for row in rows}
    for method_index, method in enumerate(METHOD_ORDER):
        for frequency_index, frequency_hz in enumerate(FREQUENCY_HZ_VALUES):
            for azimuth_index, azimuth_deg in enumerate(SOURCE_AZIMUTH_DEG_VALUES):
                cube[method_index, frequency_index, azimuth_index] = float(
                    getattr(row_by_key[(method, float(frequency_hz), float(azimuth_deg))], metric)
                )
    return cube


def _worst_cases(rows: list[SingleToneSweepRow]) -> list[dict[str, object]]:
    """review 優先度の高い worst case 行を作る。"""
    output: list[dict[str, object]] = []
    for metric, descending in (
        ("source_azimuth_error_deg", True),
        ("snr_gain_db", False),
        ("sidelobe_margin_db", False),
        ("sidelobe_margin_delta_db", False),
        ("q_reconstruction_rms_error", True),
        ("loaded_condition_number_max", True),
        ("source_to_waiting_azimuth_error_deg", True),
    ):
        sorted_rows = sorted(rows, key=lambda row: float(getattr(row, metric)), reverse=descending)
        for rank, row in enumerate(sorted_rows[:10], start=1):
            output.append(
                {
                    "category": "metric_worst_top10",
                    "metric": metric,
                    "rank": rank,
                    "frequency_hz": row.frequency_hz,
                    "source_azimuth_deg": row.source_azimuth_deg,
                    "method": row.method,
                    "value": getattr(row, metric),
                    "status": row.status,
                }
            )
    for row in rows:
        if row.status not in {"pass", "fallback_baseline"} or row.fallback_count > 0:
            output.append(
                {
                    "category": "status_watch_rows",
                    "metric": "status",
                    "rank": "",
                    "frequency_hz": row.frequency_hz,
                    "source_azimuth_deg": row.source_azimuth_deg,
                    "method": row.method,
                    "value": row.status,
                    "status": row.status,
                }
            )
    return output


def _plot_heatmap(
    metric_cube: FloatArray, title: str, colorbar_label: str, output_path: Path, cmap: str
) -> None:
    """method 別 metric heatmap を保存する。"""
    require_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is required to build sweep plots.")
    az_edges = centers_to_edges(np.asarray(SOURCE_AZIMUTH_DEG_VALUES, dtype=np.float64))
    freq_edges = centers_to_edges(np.asarray(FREQUENCY_HZ_VALUES, dtype=np.float64))
    finite_values = metric_cube[np.isfinite(metric_cube)]
    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(15.0, 4.8), sharey=True)
    image = None
    for axis, method_index, method in zip(
        axes, range(len(METHOD_ORDER)), METHOD_ORDER, strict=True
    ):
        image = axis.pcolormesh(
            az_edges,
            freq_edges,
            metric_cube[method_index],
            shading="flat",
            cmap=cmap,
            vmin=float(np.min(finite_values)),
            vmax=float(np.max(finite_values)),
        )
        axis.set_title(METHOD_DISPLAY_LABELS[method])
        axis.set_xlabel("Source azimuth [deg]")
    axes[0].set_ylabel("Frequency [Hz]")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=colorbar_label)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.14, top=0.84, wspace=0.18)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_representative_bl(condition: ConditionResult, axis_azimuth_deg: FloatArray) -> None:
    """代表条件の signal-only BL overlay を保存する。"""
    require_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is required to build representative BL plot.")
    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    all_levels = np.concatenate(
        [condition.signal_level_db_by_method[method] for method in METHOD_ORDER]
    )
    for start, stop in _mask_runs(condition.non_source_mask):
        axis.axvspan(
            float(axis_azimuth_deg[start]),
            float(axis_azimuth_deg[stop - 1]),
            color="0.92",
            alpha=0.65,
            label="non-source sector" if start == 0 else None,
        )
    for start, stop in _mask_runs(condition.source_mask):
        axis.axvspan(
            float(axis_azimuth_deg[start]),
            float(axis_azimuth_deg[stop - 1]),
            color="tab:green",
            alpha=0.16,
            label="source mask" if start == 0 else None,
        )
    for method in METHOD_ORDER:
        axis.plot(
            axis_azimuth_deg,
            condition.signal_level_db_by_method[method],
            color=METHOD_COLORS[method],
            label=METHOD_DISPLAY_LABELS[method],
        )
    axis.set_ylim(float(np.min(all_levels) - 1.0), float(np.max(all_levels) + 1.0))
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel(f"Signal RMS Level [{LEVEL_UNIT_LABEL}]")
    axis.set_title(
        "single tone + uncorrelated noise: "
        f"{REPRESENTATIVE_FREQUENCY_HZ:.0f} Hz, {REPRESENTATIVE_AZIMUTH_DEG:.0f} deg"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "representative_bl_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_npz(
    rows: list[SingleToneSweepRow], representative: ConditionResult, axis_azimuth_deg: FloatArray
) -> None:
    """sweep 図の元配列を npz 保存する。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARRAY_NPZ_PATH,
        frequency_hz=np.asarray(FREQUENCY_HZ_VALUES, dtype=np.float64),
        source_azimuth_deg=np.asarray(SOURCE_AZIMUTH_DEG_VALUES, dtype=np.float64),
        method=np.asarray(METHOD_ORDER, dtype=str),
        snr_gain_db=_metric_cube(rows, "snr_gain_db"),
        snr_gain_delta_db=_metric_cube(rows, "snr_gain_delta_db"),
        source_azimuth_error_deg=_metric_cube(rows, "source_azimuth_error_deg"),
        sidelobe_margin_db=_metric_cube(rows, "sidelobe_margin_db"),
        sidelobe_margin_delta_db=_metric_cube(rows, "sidelobe_margin_delta_db"),
        q_reconstruction_rms_error=_metric_cube(rows, "q_reconstruction_rms_error"),
        representative_azimuth_deg=axis_azimuth_deg,
        representative_fixed_level_db=representative.signal_level_db_by_method["fixed_baseline"],
        representative_mvdr_oracle_level_db=representative.signal_level_db_by_method["mvdr_oracle"],
        representative_diff_mvdr_fir512_level_db=representative.signal_level_db_by_method[
            "diff_mvdr_fir512"
        ],
        representative_source_mask=representative.source_mask,
        representative_non_source_mask=representative.non_source_mask,
    )


def _write_report(rows: list[SingleToneSweepRow]) -> None:
    """日本語 Markdown report を保存する。"""
    pass_rows = [row for row in rows if row.status == "pass"]
    watch_rows = [row for row in rows if row.status not in {"pass", "fallback_baseline"}]
    fixed_rows = [row for row in rows if row.method == "fixed_baseline"]
    diff_rows = [row for row in rows if row.method == "diff_mvdr_fir512"]
    min_fixed_snr_gain = min(row.snr_gain_db for row in fixed_rows)
    min_diff_snr_gain = min(row.snr_gain_db for row in diff_rows)
    max_diff_azimuth_error = max(row.source_azimuth_error_deg for row in diff_rows)
    min_diff_sidelobe_margin = min(row.sidelobe_margin_db for row in diff_rows)
    max_diff_q_error = max(row.q_reconstruction_rms_error for row in diff_rows)
    max_waiting_error = max(row.source_to_waiting_azimuth_error_deg for row in diff_rows)
    # make_directions の beam center に完全一致する source と、grid から外れた source を分ける。
    # 1e-9 deg は float32 由来の方位値を Python float にしたときの丸め差だけを許容する境界である。
    exact_waiting_diff_rows = [
        row for row in diff_rows if row.source_to_waiting_azimuth_error_deg <= 1.0e-9
    ]
    offgrid_diff_rows = [
        row for row in diff_rows if row.source_to_waiting_azimuth_error_deg > 1.0e-9
    ]
    exact_waiting_pass_count = sum(1 for row in exact_waiting_diff_rows if row.status == "pass")
    offgrid_watch_fail_count = sum(1 for row in offgrid_diff_rows if row.status != "pass")
    min_exact_source_peak_delta = min(row.source_peak_delta_db for row in exact_waiting_diff_rows)
    min_offgrid_source_peak_delta = min(row.source_peak_delta_db for row in offgrid_diff_rows)
    lines = [
        "# 単一周波数 + 無相関雑音 sweep レポート",
        "",
        "## 評価条件",
        "",
        "- evaluation pattern: `fixed_beam_single_source`",
        "- source: 単一 tone、入力 RMS level は `0 dB re input RMS`。",
        f"- source frequency: {', '.join(f'{value:.0f} Hz' for value in FREQUENCY_HZ_VALUES)}",
        f"- source azimuth: {', '.join(f'{value:.0f} deg' for value in SOURCE_AZIMUTH_DEG_VALUES)}",
        f"- noise: チャネル無相関白色雑音、power `{NOISE_POWER_PER_CHANNEL:.3g}` per channel。",
        "- MVDR covariance: source を含む `sigma_s^2 a_s a_s^H + sigma_n^2 I`。",
        "- MVDR constraint: 各 scan beam の待ち受け方位 steering `a(theta_beam, f)` を制約に使用。",
        "- source truth steering は共分散生成と評価 metric にのみ使い、MVDR 制約には使わない。",
        f"- input SNR: `{_scalar_db10(1.0 / NOISE_POWER_PER_CHANNEL):.2f} dB`。",
        f"- array: `{N_CH}` ch ULA、spacing `{SENSOR_SPACING_M:.3f} m`。",
        (
            "- method note: `mvdr_oracle` は真値方向 oracle ではなく、"
            "待ち受け方位制約の周波数領域 MVDR 参照値。"
        ),
        f"- methods: {', '.join(f'`{method}`' for method in METHOD_ORDER)}。",
        "",
        "## 成果物の定義",
        "",
        "- `single_tone_noise_sweep.csv`: frequency × source azimuth × method の一次 metric。",
        (
            "- CSV の `nearest_waiting_azimuth_deg` は source 最近傍の待ち受け方位、"
            "`source_to_waiting_azimuth_error_deg` は source 真値との差。"
        ),
        "- `worst_cases.csv`: 方位誤差、SNR gain、sidelobe margin、FIR 誤差などの worst top 10。",
        "- `snr_gain_heatmap.png`: source peak における出力 SNR gain。単位は dB re input SNR。",
        "- `azimuth_error_heatmap.png`: peak 方位と source 真値方位の差。単位は deg。",
        "- `sidelobe_margin_heatmap.png`: source peak と non-source peak の差。単位は dB。",
        (
            "- sidelobe 評価の source mask は fixed BL の -3 dB 主ローブ連続領域と "
            "source 方位 ±3 deg guard の和集合。"
        ),
        "- `sidelobe_margin_delta_heatmap.png`: sidelobe margin の fixed との差。単位は dB。",
        "- `q_reconstruction_error_heatmap.png`: 差分 FIR512 の q 再構成 RMS 誤差。",
        "- `representative_bl_overlay.png`: 代表条件の signal-only BL overlay。図中ラベルは英語。",
        "",
        "## 結論",
        "",
        f"- fixed の最小 SNR gain は `{min_fixed_snr_gain:.2f} dB`。",
        f"- diff MVDR FIR512 の最小 SNR gain は `{min_diff_snr_gain:.2f} dB`。",
        (f"- diff MVDR FIR512 の最大 peak 方位誤差は `{max_diff_azimuth_error:.3f} deg`。"),
        (f"- diff MVDR FIR512 の最小 sidelobe margin は `{min_diff_sidelobe_margin:.2f} dB`。"),
        (f"- diff MVDR FIR512 の最大 q 再構成 RMS 誤差は `{max_diff_q_error:.3e}`。"),
        (f"- source と最近傍待ち受け方位の最大ずれは `{max_waiting_error:.3f} deg`。"),
        f"- pass 行数は {len(pass_rows)}、watch/fail 行数は {len(watch_rows)}。",
        (
            f"- 待ち受け方位一致 rows は diff MVDR FIR512 が {exact_waiting_pass_count} / "
            f"{len(exact_waiting_diff_rows)} pass、最小 source peak delta は "
            f"{min_exact_source_peak_delta:.3f} dB re fixed。"
        ),
        (
            f"- off-grid rows は diff MVDR FIR512 が {offgrid_watch_fail_count} / "
            f"{len(offgrid_diff_rows)} watch/fail、最小 source peak delta は "
            f"{min_offgrid_source_peak_delta:.3f} dB re fixed。"
        ),
        "",
        "## 採否上の注意",
        "",
        (
            "この sweep は単一 source + 無相関雑音であり、"
            "同一周波数干渉や近接周波数干渉の分解性は評価しない。"
        ),
        (
            "source-preserving scan ではなく、単一 source に対する "
            "beam peak、SNR gain、sidelobe margin の基礎確認である。"
        ),
        "BTR は生成していないため、時間方向 track continuity はこの report の対象外である。",
        "",
        "## 参照図",
        "",
        "- `figures/snr_gain_heatmap.png`",
        "- `figures/azimuth_error_heatmap.png`",
        "- `figures/sidelobe_margin_heatmap.png`",
        "- `figures/sidelobe_margin_delta_heatmap.png`",
        "- `figures/q_reconstruction_error_heatmap.png`",
        "- `figures/representative_bl_overlay.png`",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_single_tone_noise_sweep() -> None:
    """単一 tone + 無相関雑音 sweep の成果物を生成する。"""
    require_matplotlib()
    array_positions = _build_array_positions(n_ch=N_CH, spacing_m=SENSOR_SPACING_M)
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
    axis_azimuth = axis_azimuth_deg.astype(np.float64)
    rows: list[SingleToneSweepRow] = []
    representative_result: ConditionResult | None = None
    for frequency_hz in FREQUENCY_HZ_VALUES:
        for source_azimuth_deg in SOURCE_AZIMUTH_DEG_VALUES:
            result = _evaluate_condition(
                float(frequency_hz),
                float(source_azimuth_deg),
                array_positions,
                axis_azimuth,
                beam_directions,
            )
            rows.extend(result.rows)
            if (
                float(frequency_hz) == REPRESENTATIVE_FREQUENCY_HZ
                and float(source_azimuth_deg) == REPRESENTATIVE_AZIMUTH_DEG
            ):
                representative_result = result
    if representative_result is None:
        raise RuntimeError("representative condition was not evaluated")
    _write_csv(_row_dicts(rows), SUMMARY_CSV_PATH, SUMMARY_COLUMNS)
    _write_csv(
        _worst_cases(rows),
        WORST_CASES_CSV_PATH,
        (
            "category",
            "metric",
            "rank",
            "frequency_hz",
            "source_azimuth_deg",
            "method",
            "value",
            "status",
        ),
    )
    _plot_heatmap(
        _metric_cube(rows, "snr_gain_db"),
        "Single tone + uncorrelated noise: SNR gain",
        "SNR Gain [dB re input SNR]",
        FIGURE_DIR / "snr_gain_heatmap.png",
        "viridis",
    )
    _plot_heatmap(
        _metric_cube(rows, "source_azimuth_error_deg"),
        "Single tone + uncorrelated noise: peak azimuth error",
        "Azimuth Error [deg]",
        FIGURE_DIR / "azimuth_error_heatmap.png",
        "magma",
    )
    _plot_heatmap(
        _metric_cube(rows, "sidelobe_margin_db"),
        "Single tone + uncorrelated noise: sidelobe margin",
        "Sidelobe Margin [dB]",
        FIGURE_DIR / "sidelobe_margin_heatmap.png",
        "viridis",
    )
    _plot_heatmap(
        _metric_cube(rows, "sidelobe_margin_delta_db"),
        "Single tone + uncorrelated noise: sidelobe margin delta",
        "Sidelobe Margin Delta [dB re fixed]",
        FIGURE_DIR / "sidelobe_margin_delta_heatmap.png",
        "coolwarm",
    )
    _plot_heatmap(
        _metric_cube(rows, "q_reconstruction_rms_error"),
        "Single tone + uncorrelated noise: q reconstruction error",
        "q Reconstruction RMS Error",
        FIGURE_DIR / "q_reconstruction_error_heatmap.png",
        "magma",
    )
    _plot_representative_bl(representative_result, axis_azimuth)
    _write_npz(rows, representative_result, axis_azimuth)
    _write_report(rows)


def main() -> None:
    """CLI entrypoint。"""
    build_single_tone_noise_sweep()
    print(f"saved summary csv to {SUMMARY_CSV_PATH}")
    print(f"saved worst cases csv to {WORST_CASES_CSV_PATH}")
    print(f"saved report to {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
