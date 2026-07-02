"""spflow.filterbank.halfband_stage を実装するモジュール。"""

from __future__ import annotations

import numpy as np


class ParaunitaryHalfbandStagePrototype:
    """Minimal 2-channel paraunitary halfband stage prototype.

    This prototype is intentionally small: a 2-point DFT on even/odd samples.
    It is useful as candidate A-0 because it is exactly paraunitary and PR, but
    its transition band is too wide for final practical use.
    """

    def analysis(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x, dtype=np.complex64)
        if arr.shape[-1] % 2 != 0:
            raise ValueError("analysis input length must be even.")
        blocks = arr.reshape(arr.shape[:-1] + (-1, 2))
        spectra = np.fft.fft(blocks, axis=-1) / np.sqrt(2.0)
        return spectra[..., 0], spectra[..., 1]

    def synthesis(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        low_arr = np.asarray(low, dtype=np.complex64)
        high_arr = np.asarray(high, dtype=np.complex64)
        if low_arr.shape != high_arr.shape:
            raise ValueError("low and high branches must have identical shapes.")
        stacked = np.stack([low_arr, high_arr], axis=-1) * np.sqrt(2.0)
        blocks = np.fft.ifft(stacked, axis=-1)
        return blocks.reshape(blocks.shape[:-2] + (-1,))

    def branch_response(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return low/high analysis magnitudes for normalized radian frequency."""

        w = np.asarray(omega, dtype=np.float32)
        low = np.abs((1.0 + np.exp(-1j * w)) / np.sqrt(2.0))
        high = np.abs((1.0 - np.exp(-1j * w)) / np.sqrt(2.0))
        return low, high

    def response_metrics(self, fft_size: int = 16384) -> dict[str, float]:
        if fft_size <= 0:
            raise ValueError("fft_size must be positive.")
        omega = np.linspace(0.0, np.pi, fft_size // 2 + 1)
        low, high = self.branch_response(omega)
        eps = np.finfo(np.float32).tiny

        low_pass = low[omega <= 0.25 * np.pi]
        low_stop = low[omega >= 0.75 * np.pi]
        high_pass = high[omega >= 0.75 * np.pi]
        high_stop = high[omega <= 0.25 * np.pi]

        return {
            "low_passband_ripple_db": _ripple_db(low_pass),
            "high_passband_ripple_db": _ripple_db(high_pass),
            "low_stopband_attenuation_db": float(
                -20.0 * np.log10(max(float(np.max(low_stop)) / max(float(np.max(low_pass)), eps), eps))
            ),
            "high_stopband_attenuation_db": float(
                -20.0 * np.log10(max(float(np.max(high_stop)) / max(float(np.max(high_pass)), eps), eps))
            ),
            "power_complementarity_error": float(np.max(np.abs(low**2 + high**2 - 2.0))),
        }


def _ripple_db(magnitude: np.ndarray) -> float:
    mag = np.asarray(magnitude, dtype=np.float32)
    if mag.size == 0:
        raise ValueError("magnitude must not be empty.")
    eps = np.finfo(np.float32).tiny
    return float(20.0 * np.log10(max(float(np.max(mag)), eps) / max(float(np.min(mag)), eps)))
