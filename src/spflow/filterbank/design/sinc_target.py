"""spflow.filterbank.design.sinc_target を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..halfband_stage_candidates import OrthonormalQMFStageCandidate


@dataclass(frozen=True)
class SincTargetResponseMetrics:
    """sinc 目標応答との誤差指標を保持する。"""

    target_window: str
    cutoff: float
    fft_size: int
    fullband_rms_error: float
    passband_rms_error: float
    stopband_rms_error: float
    transition_rms_error: float


def design_windowed_sinc_halfband_target(
    num_taps: int,
    *,
    cutoff: float = 0.25,
    window: str = "blackman",
) -> np.ndarray:
    """Design a real lowpass target used as the desired halfband response."""

    if num_taps <= 0:
        raise ValueError("num_taps must be positive.")
    if cutoff <= 0.0 or cutoff >= 0.5:
        raise ValueError("cutoff must satisfy 0 < cutoff < 0.5.")

    n = np.arange(num_taps, dtype=np.float32)
    midpoint = 0.5 * (num_taps - 1)
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * (n - midpoint))
    taps *= _select_window(window, num_taps)
    taps *= np.sqrt(2.0) / np.sum(taps)
    return taps


def build_halfband_power_target(
    num_taps: int,
    *,
    cutoff: float = 0.25,
    window: str = "blackman",
    fft_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a complemented power target derived from a windowed-sinc lowpass."""

    omega, magnitude = compute_frequency_magnitude(
        design_windowed_sinc_halfband_target(num_taps, cutoff=cutoff, window=window),
        fft_size=fft_size,
    )
    power = magnitude**2
    mirrored = power[::-1]
    target = 2.0 * power / np.maximum(power + mirrored, np.finfo(np.float32).tiny)
    return omega, target


def compute_frequency_magnitude(taps: np.ndarray, *, fft_size: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """FIR 係数の振幅応答を 0..pi の範囲で評価する。"""

    coeffs = np.asarray(taps, dtype=np.float32)
    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("taps must be a non-empty one-dimensional array.")
    if fft_size <= 0:
        raise ValueError("fft_size must be positive.")

    n = np.arange(coeffs.size, dtype=np.float32)
    omega = np.linspace(0.0, np.pi, fft_size // 2 + 1)
    magnitude = np.abs(np.sum(coeffs[np.newaxis, :] * np.exp(-1j * omega[:, np.newaxis] * n[np.newaxis, :]), axis=1))
    return omega, magnitude


def evaluate_candidate_against_sinc_target(
    candidate: OrthonormalQMFStageCandidate,
    *,
    cutoff: float = 0.25,
    window: str = "blackman",
    fft_size: int = 65536,
) -> SincTargetResponseMetrics:
    """候補係数を sinc 目標応答に対して評価する。"""

    target = design_windowed_sinc_halfband_target(
        candidate.analysis_low.size,
        cutoff=cutoff,
        window=window,
    )
    omega, target_magnitude = compute_frequency_magnitude(target, fft_size=fft_size)
    _, candidate_magnitude = compute_frequency_magnitude(candidate.analysis_low, fft_size=fft_size)
    error = candidate_magnitude - target_magnitude

    passband = omega <= 0.22 * np.pi
    stopband = omega >= 0.28 * np.pi
    transition = ~(passband | stopband)

    return SincTargetResponseMetrics(
        target_window=window,
        cutoff=cutoff,
        fft_size=fft_size,
        fullband_rms_error=_rms(error),
        passband_rms_error=_rms(error[passband]),
        stopband_rms_error=_rms(error[stopband]),
        transition_rms_error=_rms(error[transition]),
    )


def _select_window(name: str, length: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(length)
    if name == "hamming":
        return np.hamming(length)
    if name == "blackman":
        return np.blackman(length)
    raise ValueError(f"Unsupported window: {name}")


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr**2)))
