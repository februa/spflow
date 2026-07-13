"""整相シミュレーション条件からEBAE/MVDR完成重みを設計する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow.beamforming.ebae import EbaeConfig, design_ebae_weights_band
from spflow.simulation.alignment_config import AlignmentSimulationConfig
from spflow.simulation.alignment_coordinates import ALIGNMENT_METHOD_IDS
from spflow.simulation.alignment_covariance import calculate_alignment_source_covariance
from spflow.simulation.ula_propagation import (
    calculate_frequency_steering,
    calculate_ula_arrival_delays_s,
)

ComplexArray = NDArray[np.complexfloating[Any, Any]]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

ALIGNMENT_ALGORITHM_IDS = ("ebae", "mvdr")


@dataclass(frozen=True)
class AlignmentWeightDesign:
    """EBAE/MVDRの完成重みと座標変換・診断量を保持する。

    重み、整数遅延位相、steeringは shape ``[n_fft,n_beam,n_ch]``、source
    steeringは ``[n_fft,n_ch]``、source maskは ``[n_fft]`` である。この結果型は
    設計済み量を保持するだけで、FIR化、level計算、図表作成を担わない。
    """

    config: AlignmentSimulationConfig
    weights: dict[str, dict[str, ComplexArray]]
    integer_phase: ComplexArray
    steering: ComplexArray
    source_steering: ComplexArray
    source_bin_mask: BoolArray
    ebae_signal_counts: dict[str, IntArray]
    ebae_associated_beams: dict[str, IntArray]


def _design_mvdr_weight(
    config: AlignmentSimulationConfig, covariance: ComplexArray, steering: ComplexArray
) -> ComplexArray:
    """単一bin・単一beamのtrace比例loading付きMVDR重みを返す。"""
    hermitian = np.asarray(
        0.5 * (covariance + covariance.conj().T), dtype=config.precision.complex_dtype
    )
    n_channel = steering.size
    average_power = float(np.real(np.trace(hermitian))) / float(n_channel)
    # trace比例loadingはunitary位相回転で不変であり、座標変換前後の同値性を保つ。
    loaded = hermitian + config.mvdr_diagonal_loading_ratio * average_power * np.eye(
        n_channel, dtype=config.precision.complex_dtype
    )
    # w=R^-1 a/(a^H R^-1 a)により、待受方向の無歪条件w^H a=1を課す。
    solved = np.linalg.solve(loaded, steering)
    return np.asarray(solved / np.vdot(steering, solved), dtype=config.precision.complex_dtype)


def _design_ebae_weight(
    config: AlignmentSimulationConfig,
    covariance: ComplexArray,
    steering_scan: ComplexArray,
    beam_index: int,
) -> tuple[ComplexArray, int, int]:
    """単一bin・単一beamのEBAE重みと信号数・対応方位indexを返す。"""
    n_channel = covariance.shape[0]
    result = design_ebae_weights_band(
        covariance,
        steering_scan,
        snapshot_count=n_channel * n_channel,
        config=EbaeConfig(
            snapshot_rate_hz=float(n_channel * n_channel),
            integration_time_sec=1.0,
            sigmoid_slope=10.0,
            sigmoid_midpoint=0.5,
            diagonal_loading=config.ebae_diagonal_loading,
        ),
    )
    associated = int(result.associated_beam_indices[0]) if result.signal_count > 0 else -1
    return (
        np.asarray(result.weights[:, beam_index], dtype=config.precision.complex_dtype),
        result.signal_count,
        associated,
    )


def _mirror_frequency_weights(values: ComplexArray, fft_size: int) -> ComplexArray:
    """DC～Nyquist重みを共役鏡映してfull DFT重みにする。"""
    full = np.empty((fft_size, values.shape[1], values.shape[2]), dtype=values.dtype)
    full[: fft_size // 2 + 1] = values
    # DC/Nyquistを除いてW[-k]=conj(W[k])とし、full DFTの正負周波数を対応させる。
    full[fft_size // 2 + 1 :] = values[1:-1][::-1].conj()
    return full


def _mirror_integer_diagnostics(values: IntArray, fft_size: int) -> IntArray:
    """DC～Nyquist診断値をfull DFT binへ鏡映する。"""
    full = np.empty((fft_size, values.shape[1]), dtype=np.int64)
    full[: fft_size // 2 + 1] = values
    full[fft_size // 2 + 1 :] = values[1:-1][::-1]
    return full


def design_alignment_weights(config: AlignmentSimulationConfig) -> AlignmentWeightDesign:
    """明示された一つの条件からEBAE/MVDRのS1・S2a・T1・T2a重みを設計する。

    Args:
        config: sampling、ULA位置、方位、source帯域、noise/loadingを含む設計条件。

    Returns:
        full DFT完成重みと座標変換、steering、EBAE診断量。

    Raises:
        ValueError: source帯域に正周波数DFT binが一つも含まれない場合。
        numpy.linalg.LinAlgError: loading後共分散を解けない場合。

    Notes:
        FIR実現、beam level、parameter sweepはこの関数の責務に含めない。
    """
    n_fft = config.fft_size
    n_beam = config.beam_azimuth_deg.size
    n_channel = config.sensor_positions_m.size
    positive_hz = np.asarray(
        np.fft.rfftfreq(n_fft, d=1.0 / config.fs_hz), dtype=config.precision.real_dtype
    )
    full_hz = np.asarray(
        np.fft.fftfreq(n_fft, d=1.0 / config.fs_hz), dtype=config.precision.real_dtype
    )
    beam_delays_s = calculate_ula_arrival_delays_s(
        config.sensor_positions_m, config.beam_azimuth_deg, config.sound_speed_m_per_s
    )
    source_delays_s = calculate_ula_arrival_delays_s(
        config.sensor_positions_m,
        np.asarray([config.target_azimuth_deg], dtype=config.precision.real_dtype),
        config.sound_speed_m_per_s,
    )[0]
    positive_steering = calculate_frequency_steering(beam_delays_s, positive_hz)
    positive_source = calculate_frequency_steering(source_delays_s[None, :], positive_hz)[:, 0, :]
    full_steering = calculate_frequency_steering(beam_delays_s, full_hz)
    full_source = calculate_frequency_steering(source_delays_s[None, :], full_hz)[:, 0, :]
    positive_mask = (positive_hz >= config.target_band_hz[0]) & (
        positive_hz <= config.target_band_hz[1]
    )
    source_bin_count = int(np.count_nonzero(positive_mask))
    if source_bin_count == 0:
        raise ValueError("target_band_hz must contain at least one non-negative DFT bin.")
    full_mask = (np.abs(full_hz) >= config.target_band_hz[0]) & (
        np.abs(full_hz) <= config.target_band_hz[1]
    )
    power_per_bin = config.source_band_rms_power / float(source_bin_count)

    # 待受方位の物理遅延を最寄りsampleへ丸め、残差座標への位相回転Dを作る。
    integer_delays = np.rint(beam_delays_s * config.fs_hz).astype(np.int64)
    positive_phase = np.exp(
        1j * 2.0 * np.pi * positive_hz[:, None, None] * integer_delays[None, :, :] / config.fs_hz
    )
    full_phase = np.exp(
        1j * 2.0 * np.pi * full_hz[:, None, None] * integer_delays[None, :, :] / config.fs_hz
    )
    positive_weights = {
        algorithm: {
            method: np.empty(
                (positive_hz.size, n_beam, n_channel), dtype=config.precision.complex_dtype
            )
            for method in ALIGNMENT_METHOD_IDS
        }
        for algorithm in ALIGNMENT_ALGORITHM_IDS
    }
    counts = {
        method: np.zeros((positive_hz.size, n_beam), dtype=np.int64)
        for method in ALIGNMENT_METHOD_IDS
    }
    associated = {
        method: np.full((positive_hz.size, n_beam), -1, dtype=np.int64)
        for method in ALIGNMENT_METHOD_IDS
    }
    for frequency_index in range(positive_hz.size):
        source_power = power_per_bin if positive_mask[frequency_index] else 0.0
        s_covariance = calculate_alignment_source_covariance(
            source_delays_s,
            positive_source[frequency_index],
            fs_hz=config.fs_hz,
            analysis_width_hz=config.analysis_width_hz,
            noise_power_per_bin_re_input_rms2=config.noise_power_per_bin_re_input_rms2,
            candidate_delay_s=None,
            source_power=source_power,
        )
        steering_scan = np.asarray(
            positive_steering[frequency_index].T, dtype=config.precision.complex_dtype
        )
        for beam_index in range(n_beam):
            constraint = positive_steering[frequency_index, beam_index]
            phase = positive_phase[frequency_index, beam_index]
            rotated_scan = np.asarray(
                phase[:, None] * steering_scan, dtype=config.precision.complex_dtype
            )
            rotated_constraint = np.asarray(
                phase * constraint, dtype=config.precision.complex_dtype
            )
            if source_power == 0.0:
                # 白色雑音だけのbinでは両方式がCBFへ帰着するため、同値な完成重みを直接置く。
                original_cbf = np.asarray(
                    constraint / np.vdot(constraint, constraint),
                    dtype=config.precision.complex_dtype,
                )
                residual_cbf = np.asarray(
                    rotated_constraint / np.vdot(rotated_constraint, rotated_constraint),
                    dtype=config.precision.complex_dtype,
                )
                for algorithm in ALIGNMENT_ALGORITHM_IDS:
                    for method in ("S1", "T1"):
                        positive_weights[algorithm][method][frequency_index, beam_index] = (
                            original_cbf
                        )
                    for method in ("S2a", "T2a"):
                        positive_weights[algorithm][method][frequency_index, beam_index] = (
                            residual_cbf
                        )
                continue
            s_residual = np.asarray(
                phase[:, None] * s_covariance * phase.conj()[None, :],
                dtype=config.precision.complex_dtype,
            )
            t_original = calculate_alignment_source_covariance(
                source_delays_s,
                positive_source[frequency_index],
                fs_hz=config.fs_hz,
                analysis_width_hz=config.analysis_width_hz,
                noise_power_per_bin_re_input_rms2=config.noise_power_per_bin_re_input_rms2,
                candidate_delay_s=beam_delays_s[beam_index],
                source_power=source_power,
            )
            t_residual = np.asarray(
                phase[:, None] * t_original * phase.conj()[None, :],
                dtype=config.precision.complex_dtype,
            )
            for method, covariance, scan, method_constraint in (
                ("S1", s_covariance, steering_scan, constraint),
                ("S2a", s_residual, rotated_scan, rotated_constraint),
                ("T1", t_original, steering_scan, constraint),
                ("T2a", t_residual, rotated_scan, rotated_constraint),
            ):
                positive_weights["mvdr"][method][frequency_index, beam_index] = _design_mvdr_weight(
                    config, covariance, method_constraint
                )
                weight, count, beam = _design_ebae_weight(config, covariance, scan, beam_index)
                positive_weights["ebae"][method][frequency_index, beam_index] = weight
                counts[method][frequency_index, beam_index] = count
                associated[method][frequency_index, beam_index] = beam
    weights = {
        algorithm: {
            method: _mirror_frequency_weights(positive_weights[algorithm][method], n_fft)
            for method in ALIGNMENT_METHOD_IDS
        }
        for algorithm in ALIGNMENT_ALGORITHM_IDS
    }
    return AlignmentWeightDesign(
        config=config,
        weights=weights,
        integer_phase=np.asarray(full_phase, dtype=config.precision.complex_dtype),
        steering=full_steering,
        source_steering=full_source,
        source_bin_mask=np.asarray(full_mask, dtype=np.bool_),
        ebae_signal_counts={
            method: _mirror_integer_diagnostics(counts[method], n_fft)
            for method in ALIGNMENT_METHOD_IDS
        },
        ebae_associated_beams={
            method: _mirror_integer_diagnostics(associated[method], n_fft)
            for method in ALIGNMENT_METHOD_IDS
        },
    )
