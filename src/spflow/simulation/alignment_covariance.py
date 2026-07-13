"""整相方式比較で使う平坦帯域source共分散モデル。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
ComplexArray = NDArray[np.complexfloating[Any, Any]]


def calculate_alignment_source_covariance(
    source_delay_s: FloatArray,
    source_steering: ComplexArray,
    *,
    fs_hz: float,
    analysis_width_hz: float,
    noise_power_per_bin_re_input_rms2: float,
    candidate_delay_s: FloatArray | None,
    source_power: float,
) -> ComplexArray:
    """S共分散または候補方位の整数切出しを含むT共分散を計算する。

    Args:
        source_delay_s: source到来遅延。shape ``[n_ch]``、単位s。
        source_steering: 当該binのsource steering。shape ``[n_ch]``。
        fs_hz: 整数sample切出しのsampling frequency、単位Hz。
        analysis_width_hz: 平坦なbin内積分幅、単位Hz。
        noise_power_per_bin_re_input_rms2: channel別noise power、単位input RMS二乗。
        candidate_delay_s: T共分散の候補方位遅延。shape ``[n_ch]``、単位s。
            ``None``なら同一時刻blockのS共分散を計算する。
        source_power: 当該binのsource power、単位input RMS二乗。

    Returns:
        空間共分散。shape ``[n_ch,n_ch]``、単位input RMS二乗。

    Raises:
        ValueError: shape、dtype、有限性、sampling、幅、またはpowerが不正な場合。
    """
    combined_dtype = np.result_type(source_delay_s, source_steering)
    if combined_dtype == np.dtype(np.complex64):
        real_dtype = np.dtype(np.float32)
        complex_dtype = np.dtype(np.complex64)
    elif combined_dtype == np.dtype(np.complex128):
        real_dtype = np.dtype(np.float64)
        complex_dtype = np.dtype(np.complex128)
    else:
        raise ValueError("source delay and steering must resolve to complex64 or complex128.")
    steering = np.asarray(source_steering, dtype=complex_dtype)
    delays = np.asarray(source_delay_s, dtype=real_dtype)
    if delays.ndim != 1 or steering.ndim != 1 or delays.shape != steering.shape:
        raise ValueError("source delay and steering must have matching shape (n_ch,).")
    if not bool(np.all(np.isfinite(delays))) or not bool(np.all(np.isfinite(steering))):
        raise ValueError("source delay and steering must be finite.")
    if source_power < 0.0 or not np.isfinite(source_power):
        raise ValueError("source_power must be finite and non-negative.")
    if fs_hz <= 0.0 or analysis_width_hz < 0.0:
        raise ValueError("fs_hz must be positive and analysis_width_hz non-negative.")
    if noise_power_per_bin_re_input_rms2 <= 0.0 or not np.isfinite(
        noise_power_per_bin_re_input_rms2
    ):
        raise ValueError("noise power must be finite and positive.")
    residual_delay_s = delays
    if candidate_delay_s is not None:
        candidate = np.asarray(candidate_delay_s, dtype=real_dtype)
        if candidate.shape != delays.shape or not bool(np.all(np.isfinite(candidate))):
            raise ValueError("candidate_delay_s must be finite with shape (n_ch,).")
        # T共分散では候補方位の整数sample時刻差を切出しで除き、残留遅延だけを積分する。
        quantized_candidate_s = np.rint(candidate * fs_hz) / fs_hz
        residual_delay_s = delays - quantized_candidate_s
    pair_delay_s = residual_delay_s[:, None] - residual_delay_s[None, :]
    # 平坦なbin内積分のpair coherenceは sinc(Δf Δtau) に対応する。
    coherence = np.sinc(analysis_width_hz * pair_delay_s)
    outer = steering[:, None] * steering.conj()[None, :]
    return np.asarray(
        source_power * coherence * outer
        + noise_power_per_bin_re_input_rms2 * np.eye(delays.size, dtype=complex_dtype),
        dtype=complex_dtype,
    )
