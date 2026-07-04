"""spflow.filterbank.design.prototype を実装するモジュール。"""

from __future__ import annotations

import numpy as np


def make_pr_prototype(fft_size: int, kind: str = "sine") -> np.ndarray:
    """完全再構成を満たす解析合成用プロトタイプ窓を返す。"""

    if fft_size <= 0:
        raise ValueError("fft_size must be positive.")
    if fft_size % 2 != 0:
        raise ValueError("fft_size must be even.")

    if kind == "sine":
        n = np.arange(fft_size, dtype=np.float32)
        # sine 窓は 50% overlap 条件で COLA を満たし、WOLA の PR 基準として使いやすい。
        return np.sin(np.pi * (n + 0.5) / fft_size)

    if kind == "sqrt_hann":
        return np.sqrt(np.hanning(fft_size))

    raise ValueError(f"Unknown prototype kind: {kind}")
