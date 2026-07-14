"""時間領域 SLC と MVDR / LCMV / GSC のビーム応答改善量を比較するモジュール。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float, require_positive_int
from ..beamforming_evaluation.fractional_response import (
    calculate_fractional_beam_response_matrix,
)
from ..beamforming_evaluation.level_metrics import (
    calculate_real_tone_response_rms_level_db20,
    calculate_rms_level_db20,
)
from ..beamforming_evaluation.scan_grid import build_beam_scan_grid
from ..level_conversion import LevelConverter, level_20log10_rms
from ..simulation.numerics import SimulationPrecision
from ..simulation.tone_scene import (
    direction_from_azimuth_elevation,
    synthesize_tone_scene,
)
from .diagnostic_plotting import require_matplotlib
from .operational_sparse_array import load_operational_sparse_array
from .operational_time_domain_slc_diagnostics import (
    OperationalTimeDomainSlcDiagnosticConfig,
    _build_source_specs,
    _plot_slc_bl_overlay,
    _protected_target_bl_sidelobe_metrics,
    run_operational_time_domain_slc_leakage_diagnostics,
)
from .slc import SlcConfig
from .time_delay import FractionalDelayAndSumBeamformer
from .time_delay_diagnostics import TimeDelayDiagnosticConfig
from .time_domain_adaptive import (
    apply_time_domain_fir_beamformer,
    build_real_tone_constraint_matrix,
    build_time_domain_tone_constraint_vector,
    build_time_tapped_snapshot_matrix,
    design_time_domain_gsc_weights,
    design_time_domain_lcmv_weights,
    diagnose_time_domain_adaptive_weights,
    estimate_time_domain_covariance,
    evaluate_constraint_response,
)

_INPUT_RMS_LEVEL_CONVERTER = LevelConverter.for_definition(
    level_20log10_rms(reference_rms=1.0, reference_label="input RMS")
)
_UNITY_RESPONSE_LEVEL_CONVERTER = LevelConverter.for_definition(
    level_20log10_rms(reference_rms=1.0, reference_label="unit response")
)


def _calculate_input_rms_level(signal: NDArray[Any]) -> float:
    """共有input RMS契約で波形levelを計算する。"""

    return calculate_rms_level_db20(signal, level_converter=_INPUT_RMS_LEVEL_CONVERTER)


@dataclass(frozen=True)
class OperationalTimeDomainAdaptiveComparisonConfig:
    """運用アレイで時間領域適応方式を SLC baseline と比較する条件を保持する。

    このクラスは、固定整相後の SLC before/after 応答と、channel×tap FIR 型の
    MVDR / LCMV / GSC 応答を、同じ target beam 固定 BL 指標で比較するための条件を保持する。

    入力はアレイ定義ファイル、小数遅延 FIR バンク、target/interferer の方位・周波数・レベル、
    FIR tap 数、対角 loading、出力先である。出力は診断 JSON と before/after BL 画像である。

    音源追跡、MUSIC による null 方位推定、STFT-bin 重み更新は責務に含めない。
    信号処理上は、時間領域 SLC で不足した BL sidelobe 改善量を、時間領域 MVDR / LCMV / GSC と
    同一評価面で比較する方式検討用診断に位置づく。
    """

    output_dir: Path
    operational_array_definition_path: Path
    fractional_delay_filter_bank_path: Path
    processing_frequency_hz: float = 10000.0
    target_azimuth_deg: float = 90.0
    interferer_azimuth_deg: float = 60.0
    interferer_frequency_hz: float = 8192.0
    target_level_db20: float = 0.0
    interferer_level_db20: float = -6.0
    duration_s: float = 5.0
    n_beam_az_real: int = 151
    tap_len: int = 3
    diagonal_loading: float = 3.0e-2
    random_seed: int = 1234
    noise_level_db20: float = -60.0
    btr_block_size: int = 8192

    def __post_init__(self) -> None:
        """診断条件の単位と範囲を検証する。"""
        require_positive_float("processing_frequency_hz", float(self.processing_frequency_hz))
        require_positive_float("interferer_frequency_hz", float(self.interferer_frequency_hz))
        require_positive_float("duration_s", float(self.duration_s))
        require_positive_int("n_beam_az_real", int(self.n_beam_az_real))
        require_positive_int("tap_len", int(self.tap_len))
        require(float(self.diagonal_loading) >= 0.0, "diagonal_loading must be non-negative.")


def _make_time_delay_config(
    *,
    config: OperationalTimeDomainAdaptiveComparisonConfig,
    active_positions_m: NDArray[np.float64],
    fs_hz: float,
    sound_speed_m_s: float,
    include_target: bool,
    include_interferer: bool,
    output_dir: Path,
) -> TimeDelayDiagnosticConfig:
    """channel 信号生成と固定整相で共有する診断 config を作る。"""
    source_specs = _build_source_specs(
        config=OperationalTimeDomainSlcDiagnosticConfig(
            output_dir=output_dir,
            operational_array_definition_path=config.operational_array_definition_path,
            fractional_delay_filter_bank_path=config.fractional_delay_filter_bank_path,
            processing_frequency_hz=float(config.processing_frequency_hz),
            target_azimuth_deg=float(config.target_azimuth_deg),
            interferer_azimuth_deg=float(config.interferer_azimuth_deg),
            interferer_frequency_hz=float(config.interferer_frequency_hz),
            target_level_db20=float(config.target_level_db20),
            interferer_level_db20=float(config.interferer_level_db20),
            duration_s=float(config.duration_s),
            n_beam_az_real=int(config.n_beam_az_real),
            noise_level_db20=float(config.noise_level_db20),
            random_seed=int(config.random_seed),
        ),
        include_target=bool(include_target),
        include_interferer=bool(include_interferer),
    )
    return TimeDelayDiagnosticConfig(
        output_dir=output_dir,
        fs_hz=float(fs_hz),
        duration_s=float(config.duration_s),
        sound_speed_m_s=float(sound_speed_m_s),
        source_specs=source_specs,
        noise_level_db20=float(config.noise_level_db20),
        random_seed=int(config.random_seed),
        array_positions_m=np.asarray(active_positions_m, dtype=np.float64),
        n_beam_az_real=int(config.n_beam_az_real),
        btr_block_size=int(config.btr_block_size),
    )


def _synthesize_configured_scene(
    *,
    active_positions_m: NDArray[np.float64],
    config: TimeDelayDiagnosticConfig,
) -> NDArray[np.floating[Any]]:
    """比較scenarioの診断設定を汎用tone scene生成部品へ写像する。"""

    source_specs = config.source_specs
    if source_specs is None or len(source_specs) == 0:
        # target-only/interferer-onlyを含む比較では、各configへ明示sourceを設定する。
        # 空のままnoiseだけを方式入力へ渡すと成分分解の意味が変わるため早期に停止する。
        raise ValueError("time-domain adaptive comparison requires explicit source_specs.")
    scene = synthesize_tone_scene(
        array_positions_m=active_positions_m,
        sources=source_specs,
        fs_hz=float(config.fs_hz),
        duration_s=float(config.duration_s),
        sound_speed_m_s=float(config.sound_speed_m_s),
        noise_level_db20=float(config.noise_level_db20),
        random_seed=int(config.random_seed),
        precision=SimulationPrecision.SINGLE,
        level_converter=_INPUT_RMS_LEVEL_CONVERTER,
    )
    return scene.signal


def _build_fractional_beamformer(
    *,
    active_positions_m: NDArray[np.float64],
    time_delay_config: TimeDelayDiagnosticConfig,
    fractional_delay_filter_bank_path: Path,
) -> tuple[FractionalDelayAndSumBeamformer, NDArray[np.float64]]:
    """運用 active channel と beam grid から小数遅延固定整相器を作る。"""
    beam_grid = build_beam_scan_grid(
        azimuth_min_deg=float(time_delay_config.az_min_deg),
        azimuth_max_deg=float(time_delay_config.az_max_deg),
        display_elevation_deg=float(time_delay_config.display_elevation_deg),
        n_real_azimuth_beams=int(time_delay_config.n_beam_az_real),
        n_virtual_azimuth_beams=int(time_delay_config.n_beam_az_virtual),
    )
    beamformer = FractionalDelayAndSumBeamformer.from_geometry_and_filter_bank_path(
        array_pos_m=np.asarray(active_positions_m, dtype=np.float64),
        dir_cos=beam_grid.directions,
        fs_hz=float(time_delay_config.fs_hz),
        sound_speed_m_s=float(time_delay_config.sound_speed_m_s),
        fractional_filter_bank_path=fractional_delay_filter_bank_path,
    )
    return beamformer, beam_grid.azimuth_deg


def _steering_vector_for_azimuth(
    *,
    active_positions_m: NDArray[Any],
    azimuth_deg: float,
    frequency_hz: float,
    sound_speed_m_s: float,
) -> NDArray[np.complex128]:
    """指定方位・周波数の channel steering を返す。

    Args:
        active_positions_m: active sensor 位置。shape は `[n_ch, 3]`、単位は m。
        azimuth_deg: source 方位。単位は deg。
        frequency_hz: source 周波数。単位は Hz。
        sound_speed_m_s: 音速。単位は m/s。

    Returns:
        複素 steering。shape は `[n_ch]`。
        `x_ch[n] = a_ch exp(j 2π f n/fs)` の `a_ch` に対応する。
    """
    positions = np.asarray(active_positions_m, dtype=np.float64)
    require(positions.ndim == 2 and positions.shape[1] == 3, "active_positions_m must have shape (n_ch, 3).")
    direction = direction_from_azimuth_elevation(float(azimuth_deg), 0.0)
    # 信号生成と同じ `arrival_delay_sec = -(r^T u) / c` を使う。
    # 複素 tone の channel 係数は exp(-j 2π f arrival_delay) であり、固定整相応答行列の arrival_phase と揃う。
    arrival_delay_sec = -(positions @ direction) / float(sound_speed_m_s)
    return np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * arrival_delay_sec).astype(np.complex128)


def _fixed_response_level_at_azimuth(
    *,
    beamformer: FractionalDelayAndSumBeamformer,
    active_positions_m: NDArray[Any],
    target_beam_index: int,
    source_azimuth_deg: float,
    frequency_hz: float,
    sound_speed_m_s: float,
    source_level_db20: float,
) -> float:
    """固定整相 target beam の任意 source 方位応答を返す。

    BL 表示 grid が真の source 方位を含まない場合、nearest grid だけでは null 深さを誤読する。
    この関数は source 方位を連続値として扱い、設計した制約点そのものの応答を確認する。
    """
    steering_response = np.asarray(beamformer.steering_response(float(frequency_hz)), dtype=np.complex128)
    source_steering = _steering_vector_for_azimuth(
        active_positions_m=active_positions_m,
        azimuth_deg=float(source_azimuth_deg),
        frequency_hz=float(frequency_hz),
        sound_speed_m_s=float(sound_speed_m_s),
    )
    require(0 <= int(target_beam_index) < steering_response.shape[1], "target_beam_index is out of range.")
    source_rms = _INPUT_RMS_LEVEL_CONVERTER.input_to_rms(float(source_level_db20))
    # 固定整相の target beam 応答は、整相器側応答と source 到来 steering の channel 平均である。
    response = np.vdot(steering_response[:, int(target_beam_index)], source_steering) / float(source_steering.shape[0])
    return _INPUT_RMS_LEVEL_CONVERTER.output_rms_to_level(
        float(np.abs(response)) * source_rms,
        floor_db=_INPUT_RMS_LEVEL_CONVERTER.float64_tiny_level_db,
    )


def _adaptive_response_level_at_azimuth(
    *,
    weights: NDArray[Any],
    active_positions_m: NDArray[Any],
    source_azimuth_deg: float,
    frequency_hz: float,
    fs_hz: float,
    sound_speed_m_s: float,
    tap_len: int,
    source_level_db20: float,
) -> float:
    """時間領域適応重みの任意 source 方位応答を返す。"""
    beam_weights = np.asarray(weights, dtype=np.complex128)
    require(beam_weights.ndim == 2 and beam_weights.shape[1] == 1, "weights must have shape (n_dof, 1).")
    steering = _steering_vector_for_azimuth(
        active_positions_m=active_positions_m,
        azimuth_deg=float(source_azimuth_deg),
        frequency_hz=float(frequency_hz),
        sound_speed_m_s=float(sound_speed_m_s),
    )
    positive_constraint = build_time_domain_tone_constraint_vector(
        steering,
        frequency_hz=float(frequency_hz),
        fs_hz=float(fs_hz),
        tap_len=int(tap_len),
    )
    negative_constraint = positive_constraint.conj()
    source_rms = _INPUT_RMS_LEVEL_CONVERTER.input_to_rms(float(source_level_db20))
    # 時間領域適応重みは複素になり得るため、実 tone の BL レベルは正負周波数の RMS 合成で評価する。
    positive_response = np.conj(beam_weights[:, 0]) @ positive_constraint
    negative_response = np.conj(beam_weights[:, 0]) @ negative_constraint
    level = calculate_real_tone_response_rms_level_db20(
        np.array([positive_response], dtype=np.complex128),
        np.array([negative_response], dtype=np.complex128),
        source_rms,
        level_converter=_INPUT_RMS_LEVEL_CONVERTER,
    )
    return float(level[0])

def _adaptive_response_curve(
    *,
    weights: NDArray[Any],
    active_positions_m: NDArray[Any],
    axis_az_deg: NDArray[Any],
    frequency_hz: float,
    fs_hz: float,
    sound_speed_m_s: float,
    tap_len: int,
    source_level_db20: float,
) -> NDArray[np.float64]:
    """時間領域 FIR 適応重みの target beam 固定 BL 応答を計算する。

    Args:
        weights: channel×tap FIR 重み。shape は `[n_ch * L, 1]`。
        active_positions_m: active sensor 位置。shape は `[n_ch, 3]`。
        axis_az_deg: source 方位軸。shape は `[n_look]`、単位は deg。
        frequency_hz: 評価周波数。単位は Hz。
        fs_hz: サンプリング周波数。単位は Hz。
        sound_speed_m_s: 音速。単位は m/s。
        tap_len: FIR tap 数。単位は sample。
        source_level_db20: source RMS 入力レベル。単位は dB re input RMS。

    Returns:
        適応後 target 出力応答。shape は `[n_look]`、単位は dB re input RMS。
    """
    beam_weights = np.asarray(weights, dtype=np.complex128)
    require(beam_weights.ndim == 2 and beam_weights.shape[1] == 1, "weights must have shape (n_dof, 1).")
    azimuths = np.asarray(axis_az_deg, dtype=np.float64)
    source_rms = _INPUT_RMS_LEVEL_CONVERTER.input_to_rms(float(source_level_db20))
    positive_response_values = np.zeros(azimuths.shape[0], dtype=np.complex128)
    negative_response_values = np.zeros(azimuths.shape[0], dtype=np.complex128)
    for look_index, azimuth_deg in enumerate(azimuths.tolist()):
        steering = _steering_vector_for_azimuth(
            active_positions_m=active_positions_m,
            azimuth_deg=float(azimuth_deg),
            frequency_hz=float(frequency_hz),
            sound_speed_m_s=float(sound_speed_m_s),
        )
        positive_constraint = build_time_domain_tone_constraint_vector(
            steering,
            frequency_hz=float(frequency_hz),
            fs_hz=float(fs_hz),
            tap_len=int(tap_len),
        )
        negative_constraint = positive_constraint.conj()
        # target beam 固定 BL なので、各 source 方位の実 tone が同じ FIR 重みへ入ったときの応答を見る。
        # 複素重みでは `H(+f)` と `H(-f)` が非対称になり得るため、両側帯を後で RMS 合成する。
        positive_response_values[look_index] = np.conj(beam_weights[:, 0]) @ positive_constraint
        negative_response_values[look_index] = np.conj(beam_weights[:, 0]) @ negative_constraint
    return calculate_real_tone_response_rms_level_db20(
        positive_response_values,
        negative_response_values,
        source_rms,
        level_converter=_INPUT_RMS_LEVEL_CONVERTER,
    )


def _fixed_target_beam_curve(
    *,
    response_matrix: NDArray[Any],
    target_beam_index: int,
    source_level_db20: float,
) -> NDArray[np.float64]:
    """固定整相 target beam の source 方位応答を dB20 へ変換する。"""
    responses = np.asarray(response_matrix, dtype=np.complex128)
    require(responses.ndim == 2, "response_matrix must have shape (n_beam, n_look).")
    require(0 <= int(target_beam_index) < responses.shape[0], "target_beam_index is out of range.")
    source_rms = _INPUT_RMS_LEVEL_CONVERTER.input_to_rms(float(source_level_db20))
    return np.asarray(
        _INPUT_RMS_LEVEL_CONVERTER.output_rms_to_level(
            np.abs(responses[int(target_beam_index), :]) * source_rms,
            floor_db=_INPUT_RMS_LEVEL_CONVERTER.float64_tiny_level_db,
        ),
        dtype=np.float64,
    )


def _constraint_error_summary(
    *,
    weights: NDArray[Any],
    constraint_matrix: NDArray[Any],
    desired_response: NDArray[Any],
) -> dict[str, float]:
    """設計重みの target/null 制約応答誤差を要約する。"""
    responses = evaluate_constraint_response(weights, constraint_matrix)
    desired = np.asarray(desired_response, dtype=np.complex128)[:, np.newaxis]
    error = responses - desired
    # target 応答は desired=1 の制約誤差を dB20 で見る。
    # null 応答は desired=0 の絶対応答を dB20 で見る。
    target_error = np.abs(error[np.abs(desired[:, 0]) > 0.0, :])
    null_response = np.abs(responses[np.abs(desired[:, 0]) == 0.0, :])
    max_target_error = float(np.max(target_error)) if target_error.size > 0 else 0.0
    max_null_response = float(np.max(null_response)) if null_response.size > 0 else 0.0
    return {
        "max_target_constraint_error_db20": _UNITY_RESPONSE_LEVEL_CONVERTER.output_rms_to_level(
            max_target_error,
            floor_db=_UNITY_RESPONSE_LEVEL_CONVERTER.float64_tiny_level_db,
        ),
        "max_null_constraint_response_db20": _UNITY_RESPONSE_LEVEL_CONVERTER.output_rms_to_level(
            max_null_response,
            floor_db=_UNITY_RESPONSE_LEVEL_CONVERTER.float64_tiny_level_db,
        ),
    }


def _method_summary(
    *,
    method_name: str,
    weights: NDArray[Any],
    covariance: NDArray[Any],
    constraint_matrix: NDArray[Any],
    desired_response: NDArray[Any],
    diagonal_loading: float,
    mixed_channel_signal: NDArray[Any],
    target_channel_signal: NDArray[Any],
    interferer_channel_signal: NDArray[Any],
    fixed_mixed_target: NDArray[Any],
    fixed_target_component: NDArray[Any],
    fixed_interferer_component: NDArray[Any],
    fixed_target_curve_db20: NDArray[np.float64],
    adaptive_target_curve_db20: NDArray[np.float64],
    fixed_interferer_curve_db20: NDArray[np.float64],
    adaptive_interferer_curve_db20: NDArray[np.float64],
    exact_target_before_db20: float,
    exact_target_after_db20: float,
    exact_interferer_before_db20: float,
    exact_interferer_after_db20: float,
    axis_az_deg: NDArray[np.float64],
    target_beam_index: int,
    target_azimuth_deg: float,
    interferer_azimuth_deg: float,
    guard_beam_count: int,
    tap_len: int,
    elapsed_sec: float,
    fs_hz: float,
) -> dict[str, object]:
    """方式ごとの waveform / BL / covariance health 指標をまとめる。"""
    mixed_output = apply_time_domain_fir_beamformer(mixed_channel_signal, weights, tap_len=int(tap_len))[0]
    target_output = apply_time_domain_fir_beamformer(target_channel_signal, weights, tap_len=int(tap_len))[0]
    interferer_output = apply_time_domain_fir_beamformer(interferer_channel_signal, weights, tap_len=int(tap_len))[0]
    diagnostics = diagnose_time_domain_adaptive_weights(
        covariance,
        constraint_matrix,
        weights,
        diagonal_loading=float(diagonal_loading),
    )
    target_source_index = int(np.argmin(np.abs(axis_az_deg - float(target_azimuth_deg))))
    interferer_source_index = int(np.argmin(np.abs(axis_az_deg - float(interferer_azimuth_deg))))
    target_sidelobe_metrics = _protected_target_bl_sidelobe_metrics(
        axis_az_deg=axis_az_deg,
        before_levels_db20=fixed_target_curve_db20,
        after_levels_db20=adaptive_target_curve_db20,
        target_beam_index=int(target_beam_index),
        marker_azimuth_deg=float(target_azimuth_deg),
        guard_beam_count=int(guard_beam_count),
    )
    interferer_sidelobe_metrics = _protected_target_bl_sidelobe_metrics(
        axis_az_deg=axis_az_deg,
        before_levels_db20=fixed_interferer_curve_db20,
        after_levels_db20=adaptive_interferer_curve_db20,
        target_beam_index=int(target_beam_index),
        marker_azimuth_deg=float(interferer_azimuth_deg),
        guard_beam_count=int(guard_beam_count),
    )
    return {
        "method": method_name,
        "level_reference": "dB re input RMS",
        "levels": {
            "mixed_before_db20": _calculate_input_rms_level(fixed_mixed_target),
            "mixed_after_db20": _calculate_input_rms_level(mixed_output),
            "target_before_db20": _calculate_input_rms_level(fixed_target_component),
            "target_after_db20": _calculate_input_rms_level(target_output),
            "interferer_before_db20": _calculate_input_rms_level(fixed_interferer_component),
            "interferer_after_db20": _calculate_input_rms_level(interferer_output),
            "target_power_delta_db": float(
                _calculate_input_rms_level(target_output)
                - _calculate_input_rms_level(fixed_target_component)
            ),
            "interferer_reduction_db": float(
                _calculate_input_rms_level(fixed_interferer_component)
                - _calculate_input_rms_level(interferer_output)
            ),
        },
        "constraint_response": _constraint_error_summary(
            weights=weights,
            constraint_matrix=constraint_matrix,
            desired_response=desired_response,
        ),
        "covariance_health": diagnostics.as_dict(),
        "runtime": {
            "elapsed_sec": float(elapsed_sec),
            "input_duration_sec": float(np.asarray(mixed_channel_signal).shape[1]) / float(fs_hz),
            "realtime_factor": float(elapsed_sec / (float(np.asarray(mixed_channel_signal).shape[1]) / float(fs_hz))),
        },
        "protected_target_bl_summary": {
            "definition": "protected target beam fixed; x-axis is source azimuth, not output beam index",
            "target_frequency_before_at_target_db20": float(fixed_target_curve_db20[target_source_index]),
            "target_frequency_after_at_target_db20": float(adaptive_target_curve_db20[target_source_index]),
            "target_frequency_delta_at_target_db": float(adaptive_target_curve_db20[target_source_index] - fixed_target_curve_db20[target_source_index]),
            "interferer_frequency_before_at_interferer_db20": float(fixed_interferer_curve_db20[interferer_source_index]),
            "interferer_frequency_after_at_interferer_db20": float(adaptive_interferer_curve_db20[interferer_source_index]),
            "interferer_frequency_reduction_at_interferer_db": float(
                fixed_interferer_curve_db20[interferer_source_index] - adaptive_interferer_curve_db20[interferer_source_index]
            ),
            "target_frequency_exact_before_at_target_db20": float(exact_target_before_db20),
            "target_frequency_exact_after_at_target_db20": float(exact_target_after_db20),
            "target_frequency_exact_delta_at_target_db": float(exact_target_after_db20 - exact_target_before_db20),
            "interferer_frequency_exact_before_at_interferer_db20": float(exact_interferer_before_db20),
            "interferer_frequency_exact_after_at_interferer_db20": float(exact_interferer_after_db20),
            "interferer_frequency_exact_reduction_at_interferer_db": float(
                exact_interferer_before_db20 - exact_interferer_after_db20
            ),
            "target_frequency_sidelobe_metrics": target_sidelobe_metrics,
            "interferer_frequency_sidelobe_metrics": interferer_sidelobe_metrics,
        },
    }


def run_operational_time_domain_adaptive_comparison(
    *,
    config: OperationalTimeDomainAdaptiveComparisonConfig,
    slc_config: SlcConfig,
) -> dict[str, object]:
    """SLC baseline と時間領域 MVDR / LCMV / GSC の BL 改善量を比較する。

    Args:
        config: 比較診断条件。
        slc_config: SLC baseline に使う設定。guard と 3 秒忘却積分条件を比較基準に含める。

    Returns:
        SLC baseline と MVDR / LCMV / GSC の成分別レベル、制約応答、BL 改善量、runtime を含む summary。

    Raises:
        ValueError: 入力ファイルや shape が不正な場合。
    """
    require_matplotlib()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    array_definition = load_operational_sparse_array(Path(config.operational_array_definition_path))
    active_indices = array_definition.active_channel_indices_for_frequency(float(config.processing_frequency_hz))
    active_positions_m = np.asarray(array_definition.positions_m[active_indices], dtype=np.float64)
    fs_hz = float(array_definition.fs_hz)
    sound_speed_m_s = float(array_definition.sound_speed_m_s)

    mixed_config = _make_time_delay_config(
        config=config,
        active_positions_m=active_positions_m,
        fs_hz=fs_hz,
        sound_speed_m_s=sound_speed_m_s,
        include_target=True,
        include_interferer=True,
        output_dir=output_dir / "fixed_mixed",
    )
    target_config = _make_time_delay_config(
        config=config,
        active_positions_m=active_positions_m,
        fs_hz=fs_hz,
        sound_speed_m_s=sound_speed_m_s,
        include_target=True,
        include_interferer=False,
        output_dir=output_dir / "fixed_target_only",
    )
    interferer_config = _make_time_delay_config(
        config=config,
        active_positions_m=active_positions_m,
        fs_hz=fs_hz,
        sound_speed_m_s=sound_speed_m_s,
        include_target=False,
        include_interferer=True,
        output_dir=output_dir / "fixed_interferer_only",
    )
    mixed_channel_signal = _synthesize_configured_scene(
        active_positions_m=active_positions_m,
        config=mixed_config,
    )
    target_channel_signal = _synthesize_configured_scene(
        active_positions_m=active_positions_m,
        config=target_config,
    )
    interferer_channel_signal = _synthesize_configured_scene(
        active_positions_m=active_positions_m,
        config=interferer_config,
    )
    beamformer, axis_az_deg = _build_fractional_beamformer(
        active_positions_m=active_positions_m,
        time_delay_config=mixed_config,
        fractional_delay_filter_bank_path=Path(config.fractional_delay_filter_bank_path),
    )
    mixed_beam_output = beamformer.process(mixed_channel_signal)
    target_beam_output = beamformer.process(target_channel_signal)
    interferer_beam_output = beamformer.process(interferer_channel_signal)
    if isinstance(mixed_beam_output, tuple) or isinstance(target_beam_output, tuple) or isinstance(interferer_beam_output, tuple):
        raise TypeError("FractionalDelayAndSumBeamformer.process must return ndarray when return_steered_channels is False.")

    target_beam_index = int(np.argmin(np.abs(axis_az_deg - float(config.target_azimuth_deg))))
    fixed_mixed_target = np.asarray(mixed_beam_output[target_beam_index, :])
    fixed_target_component = np.asarray(target_beam_output[target_beam_index, :])
    fixed_interferer_component = np.asarray(interferer_beam_output[target_beam_index, :])

    target_response_matrix = calculate_fractional_beam_response_matrix(
        beamformer,
        float(config.processing_frequency_hz),
    )
    interferer_response_matrix = calculate_fractional_beam_response_matrix(
        beamformer,
        float(config.interferer_frequency_hz),
    )
    fixed_target_curve_db20 = _fixed_target_beam_curve(
        response_matrix=target_response_matrix,
        target_beam_index=target_beam_index,
        source_level_db20=float(config.target_level_db20),
    )
    fixed_interferer_curve_db20 = _fixed_target_beam_curve(
        response_matrix=interferer_response_matrix,
        target_beam_index=target_beam_index,
        source_level_db20=float(config.interferer_level_db20),
    )

    tapped_snapshots = build_time_tapped_snapshot_matrix(mixed_channel_signal, tap_len=int(config.tap_len))
    covariance = estimate_time_domain_covariance(tapped_snapshots)
    target_steering = _steering_vector_for_azimuth(
        active_positions_m=active_positions_m,
        azimuth_deg=float(config.target_azimuth_deg),
        frequency_hz=float(config.processing_frequency_hz),
        sound_speed_m_s=sound_speed_m_s,
    )
    interferer_steering = _steering_vector_for_azimuth(
        active_positions_m=active_positions_m,
        azimuth_deg=float(config.interferer_azimuth_deg),
        frequency_hz=float(config.interferer_frequency_hz),
        sound_speed_m_s=sound_speed_m_s,
    )
    target_constraints = build_real_tone_constraint_matrix(
        target_steering,
        frequency_hz=float(config.processing_frequency_hz),
        fs_hz=fs_hz,
        tap_len=int(config.tap_len),
    )
    interferer_constraints = build_real_tone_constraint_matrix(
        interferer_steering,
        frequency_hz=float(config.interferer_frequency_hz),
        fs_hz=fs_hz,
        tap_len=int(config.tap_len),
    )
    lcmv_constraints = np.concatenate([target_constraints, interferer_constraints], axis=1)
    mvdr_desired = np.ones(2, dtype=np.complex128)
    lcmv_desired = np.array([1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    fixed_interferer_sidelobe_metrics = _protected_target_bl_sidelobe_metrics(
        axis_az_deg=axis_az_deg,
        before_levels_db20=fixed_interferer_curve_db20,
        after_levels_db20=fixed_interferer_curve_db20,
        target_beam_index=int(target_beam_index),
        marker_azimuth_deg=float(config.interferer_azimuth_deg),
        guard_beam_count=int(slc_config.guard),
    )
    sector_null_azimuths: list[float] = []
    for candidate_azimuth_deg in (
        float(config.interferer_azimuth_deg),
        float(fixed_interferer_sidelobe_metrics["before_first_sidelobe_peak_azimuth_deg"]),
        float(fixed_interferer_sidelobe_metrics["before_guard_outside_peak_azimuth_deg"]),
    ):
        # 近接する同一方位を複数制約に入れると、制約 Gram が悪条件になりやすい。
        # beam grid 上で実質的に同じ制約は 1 本にまとめ、自由度を第一副極抑圧へ使う。
        if all(abs(candidate_azimuth_deg - existing_azimuth_deg) > 0.25 for existing_azimuth_deg in sector_null_azimuths):
            sector_null_azimuths.append(candidate_azimuth_deg)
    sector_null_constraints = np.concatenate(
        [
            build_real_tone_constraint_matrix(
                _steering_vector_for_azimuth(
                    active_positions_m=active_positions_m,
                    azimuth_deg=float(null_azimuth_deg),
                    frequency_hz=float(config.interferer_frequency_hz),
                    sound_speed_m_s=sound_speed_m_s,
                ),
                frequency_hz=float(config.interferer_frequency_hz),
                fs_hz=fs_hz,
                tap_len=int(config.tap_len),
            )
            for null_azimuth_deg in sector_null_azimuths
        ],
        axis=1,
    )
    sector_lcmv_constraints = np.concatenate([target_constraints, sector_null_constraints], axis=1)
    sector_lcmv_desired = np.concatenate(
        [
            np.ones(target_constraints.shape[1], dtype=np.complex128),
            np.zeros(sector_null_constraints.shape[1], dtype=np.complex128),
        ],
        axis=0,
    )

    method_weights: dict[str, tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], float]] = {}
    start_time = time.perf_counter()
    mvdr_weights = design_time_domain_lcmv_weights(
        covariance,
        target_constraints,
        mvdr_desired,
        diagonal_loading=float(config.diagonal_loading),
    )
    method_weights["time_domain_mvdr_real"] = (mvdr_weights, target_constraints, mvdr_desired, time.perf_counter() - start_time)
    start_time = time.perf_counter()
    lcmv_weights = design_time_domain_lcmv_weights(
        covariance,
        lcmv_constraints,
        lcmv_desired,
        diagonal_loading=float(config.diagonal_loading),
    )
    method_weights["time_domain_lcmv_target_interferer_null"] = (
        lcmv_weights,
        lcmv_constraints,
        lcmv_desired,
        time.perf_counter() - start_time,
    )
    start_time = time.perf_counter()
    gsc_weights = design_time_domain_gsc_weights(
        covariance,
        lcmv_constraints,
        lcmv_desired,
        diagonal_loading=float(config.diagonal_loading),
    )
    method_weights["time_domain_gsc_equivalent_lcmv"] = (gsc_weights, lcmv_constraints, lcmv_desired, time.perf_counter() - start_time)
    start_time = time.perf_counter()
    sector_lcmv_weights = design_time_domain_lcmv_weights(
        covariance,
        sector_lcmv_constraints,
        sector_lcmv_desired,
        diagonal_loading=float(config.diagonal_loading),
    )
    method_weights["time_domain_lcmv_sector_first_sidelobe_null"] = (
        sector_lcmv_weights,
        sector_lcmv_constraints,
        sector_lcmv_desired,
        time.perf_counter() - start_time,
    )
    start_time = time.perf_counter()
    sector_gsc_weights = design_time_domain_gsc_weights(
        covariance,
        sector_lcmv_constraints,
        sector_lcmv_desired,
        diagonal_loading=float(config.diagonal_loading),
    )
    method_weights["time_domain_gsc_sector_first_sidelobe_null"] = (
        sector_gsc_weights,
        sector_lcmv_constraints,
        sector_lcmv_desired,
        time.perf_counter() - start_time,
    )

    slc_output_dir = output_dir / "slc_baseline"
    slc_summary = run_operational_time_domain_slc_leakage_diagnostics(
        config=OperationalTimeDomainSlcDiagnosticConfig(
            output_dir=slc_output_dir,
            operational_array_definition_path=Path(config.operational_array_definition_path),
            fractional_delay_filter_bank_path=Path(config.fractional_delay_filter_bank_path),
            processing_frequency_hz=float(config.processing_frequency_hz),
            target_azimuth_deg=float(config.target_azimuth_deg),
            interferer_azimuth_deg=float(config.interferer_azimuth_deg),
            interferer_frequency_hz=float(config.interferer_frequency_hz),
            target_level_db20=float(config.target_level_db20),
            interferer_level_db20=float(config.interferer_level_db20),
            duration_s=float(config.duration_s),
            n_beam_az_real=int(config.n_beam_az_real),
            noise_level_db20=float(config.noise_level_db20),
            random_seed=int(config.random_seed),
        ),
        slc_config=slc_config,
    )

    method_summaries: dict[str, object] = {}
    for method_name, (weights, constraints, desired_response, elapsed_sec) in method_weights.items():
        target_curve = _adaptive_response_curve(
            weights=weights,
            active_positions_m=active_positions_m,
            axis_az_deg=axis_az_deg,
            frequency_hz=float(config.processing_frequency_hz),
            fs_hz=fs_hz,
            sound_speed_m_s=sound_speed_m_s,
            tap_len=int(config.tap_len),
            source_level_db20=float(config.target_level_db20),
        )
        interferer_curve = _adaptive_response_curve(
            weights=weights,
            active_positions_m=active_positions_m,
            axis_az_deg=axis_az_deg,
            frequency_hz=float(config.interferer_frequency_hz),
            fs_hz=fs_hz,
            sound_speed_m_s=sound_speed_m_s,
            tap_len=int(config.tap_len),
            source_level_db20=float(config.interferer_level_db20),
        )
        target_plot_path = output_dir / f"{method_name}_target_frequency_bl_overlay.png"
        interferer_plot_path = output_dir / f"{method_name}_interferer_frequency_bl_overlay.png"
        _plot_slc_bl_overlay(
            output_path=target_plot_path,
            axis_az_deg=axis_az_deg,
            before_levels_db20=fixed_target_curve_db20,
            after_levels_db20=target_curve,
            marker_azimuth_deg=float(config.target_azimuth_deg),
            marker_label="Target source azimuth",
            title=f"{method_name}: protected target-beam response at target frequency",
            caption="固定整相 target beam 応答と時間領域適応後応答の重ね書き。レベルは dB re input RMS。",
            before_label="fixed target beam",
            after_label=method_name,
        )
        _plot_slc_bl_overlay(
            output_path=interferer_plot_path,
            axis_az_deg=axis_az_deg,
            before_levels_db20=fixed_interferer_curve_db20,
            after_levels_db20=interferer_curve,
            marker_azimuth_deg=float(config.interferer_azimuth_deg),
            marker_label="Interferer source azimuth",
            title=f"{method_name}: protected target-beam response at interferer frequency",
            caption="固定整相 target beam へ入る interferer 周波数応答と時間領域適応後応答の重ね書き。レベルは dB re input RMS。",
            before_label="fixed target beam",
            after_label=method_name,
        )
        summary = _method_summary(
            method_name=method_name,
            weights=weights,
            covariance=covariance,
            constraint_matrix=constraints,
            desired_response=desired_response,
            diagonal_loading=float(config.diagonal_loading),
            mixed_channel_signal=mixed_channel_signal,
            target_channel_signal=target_channel_signal,
            interferer_channel_signal=interferer_channel_signal,
            fixed_mixed_target=fixed_mixed_target,
            fixed_target_component=fixed_target_component,
            fixed_interferer_component=fixed_interferer_component,
            fixed_target_curve_db20=fixed_target_curve_db20,
            adaptive_target_curve_db20=target_curve,
            fixed_interferer_curve_db20=fixed_interferer_curve_db20,
            adaptive_interferer_curve_db20=interferer_curve,
            exact_target_before_db20=_fixed_response_level_at_azimuth(
                beamformer=beamformer,
                active_positions_m=active_positions_m,
                target_beam_index=target_beam_index,
                source_azimuth_deg=float(config.target_azimuth_deg),
                frequency_hz=float(config.processing_frequency_hz),
                sound_speed_m_s=sound_speed_m_s,
                source_level_db20=float(config.target_level_db20),
            ),
            exact_target_after_db20=_adaptive_response_level_at_azimuth(
                weights=weights,
                active_positions_m=active_positions_m,
                source_azimuth_deg=float(config.target_azimuth_deg),
                frequency_hz=float(config.processing_frequency_hz),
                fs_hz=fs_hz,
                sound_speed_m_s=sound_speed_m_s,
                tap_len=int(config.tap_len),
                source_level_db20=float(config.target_level_db20),
            ),
            exact_interferer_before_db20=_fixed_response_level_at_azimuth(
                beamformer=beamformer,
                active_positions_m=active_positions_m,
                target_beam_index=target_beam_index,
                source_azimuth_deg=float(config.interferer_azimuth_deg),
                frequency_hz=float(config.interferer_frequency_hz),
                sound_speed_m_s=sound_speed_m_s,
                source_level_db20=float(config.interferer_level_db20),
            ),
            exact_interferer_after_db20=_adaptive_response_level_at_azimuth(
                weights=weights,
                active_positions_m=active_positions_m,
                source_azimuth_deg=float(config.interferer_azimuth_deg),
                frequency_hz=float(config.interferer_frequency_hz),
                fs_hz=fs_hz,
                sound_speed_m_s=sound_speed_m_s,
                tap_len=int(config.tap_len),
                source_level_db20=float(config.interferer_level_db20),
            ),
            axis_az_deg=axis_az_deg,
            target_beam_index=target_beam_index,
            target_azimuth_deg=float(config.target_azimuth_deg),
            interferer_azimuth_deg=float(config.interferer_azimuth_deg),
            guard_beam_count=int(slc_config.guard),
            tap_len=int(config.tap_len),
            elapsed_sec=float(elapsed_sec),
            fs_hz=fs_hz,
        )
        summary["target_frequency_bl_overlay_png_path"] = str(target_plot_path.resolve())
        summary["interferer_frequency_bl_overlay_png_path"] = str(interferer_plot_path.resolve())
        method_summaries[method_name] = summary

    result: dict[str, object] = {
        "evaluation_pattern": "time_domain_adaptive_mvdr_lcmv_gsc",
        "level_reference": "dB re input RMS",
        "array_definition_path": str(Path(config.operational_array_definition_path).resolve()),
        "fractional_delay_filter_bank_path": str(Path(config.fractional_delay_filter_bank_path).resolve()),
        "processing_frequency_hz": float(config.processing_frequency_hz),
        "interferer_frequency_hz": float(config.interferer_frequency_hz),
        "target_azimuth_deg": float(config.target_azimuth_deg),
        "interferer_azimuth_deg": float(config.interferer_azimuth_deg),
        "tap_len": int(config.tap_len),
        "diagonal_loading": float(config.diagonal_loading),
        "target_beam_index": int(target_beam_index),
        "target_beam_azimuth_deg": float(axis_az_deg[target_beam_index]),
        "active_channel_count": int(active_indices.size),
        "sector_null_azimuths_deg": [float(value) for value in sector_null_azimuths],
        "fixed_before_definition": "fixed fractional delay target beam; x-axis is source azimuth",
        "slc_baseline": slc_summary,
        "adaptive_methods": method_summaries,
    }
    summary_path = output_dir / "time_domain_adaptive_comparison_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary_json_path"] = str(summary_path.resolve())
    return result
