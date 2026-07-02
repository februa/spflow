"""spflow.filterbank.modulation を実装するモジュール。"""

from __future__ import annotations

import numpy as np


def dft_analysis(frames: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply real-valued DFT modulation to windowed frames."""

    return np.fft.rfft(frames, axis=axis)


def dft_synthesis(subbands: np.ndarray, fft_size: int, axis: int = -1) -> np.ndarray:
    """Synthesize time-domain frames from positive-frequency subbands."""

    return np.fft.irfft(subbands, n=fft_size, axis=axis)


def full_dft_analysis(frames: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply full complex DFT modulation to windowed frames."""

    return np.fft.fft(frames, axis=axis)


def full_dft_synthesis(subbands: np.ndarray, axis: int = -1) -> np.ndarray:
    """Synthesize time-domain frames from full complex subbands."""

    return np.fft.ifft(subbands, axis=axis)
