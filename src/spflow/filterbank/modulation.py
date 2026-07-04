"""spflow.filterbank.modulation を実装するモジュール。"""

from __future__ import annotations

import numpy as np


def dft_analysis(frames: np.ndarray, axis: int = -1) -> np.ndarray:
    """実数フレームへ正周波数側 DFT 変調を適用する。

    Args:
        frames: 時間フレーム列。shape は `[..., fft_size, n_frame]` など。
        axis: FFT を取る時間軸。

    Returns:
        正周波数側サブバンド。shape は `[..., fft_size // 2 + 1, n_frame]`。
    """
    # rFFT は実信号のエルミート対称性を利用し、0..Nyquist のみを保持する。
    return np.fft.rfft(frames, axis=axis)


def dft_synthesis(subbands: np.ndarray, fft_size: int, axis: int = -1) -> np.ndarray:
    """正周波数側サブバンドから実数時間フレームを合成する。"""
    # irFFT は省略された負周波数側を共役対称として補完し、実数フレームへ戻す。
    return np.fft.irfft(subbands, n=fft_size, axis=axis)


def full_dft_analysis(frames: np.ndarray, axis: int = -1) -> np.ndarray:
    """複素フレームへ全周波数 DFT 変調を適用する。"""
    return np.fft.fft(frames, axis=axis)


def full_dft_synthesis(subbands: np.ndarray, axis: int = -1) -> np.ndarray:
    """全周波数複素サブバンドから時間フレームを合成する。"""
    return np.fft.ifft(subbands, axis=axis)
