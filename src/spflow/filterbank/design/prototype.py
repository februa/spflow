"""spflow.filterbank.design.prototype を実装するモジュール。"""

from __future__ import annotations

import numpy as np


def make_pr_prototype(fft_size: int, kind: str = "sine") -> np.ndarray:
    """Create a prototype window that supports perfect reconstruction."""

    if fft_size <= 0:
        raise ValueError("fft_size must be positive.")
    if fft_size % 2 != 0:
        raise ValueError("fft_size must be even.")

    if kind == "sine":
        n = np.arange(fft_size, dtype=np.float32)
        return np.sin(np.pi * (n + 0.5) / fft_size)

    if kind == "sqrt_hann":
        return np.sqrt(np.hanning(fft_size))

    raise ValueError(f"Unknown prototype kind: {kind}")
