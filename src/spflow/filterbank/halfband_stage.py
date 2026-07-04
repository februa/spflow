"""spflow.filterbank.halfband_stage を実装するモジュール。"""

from __future__ import annotations

import numpy as np


class ParaunitaryHalfbandStagePrototype:
    """最小構成の 2 チャネル paraunitary halfband stage 原型。

    偶数/奇数サンプル 2 点へ DFT を掛けるだけの基準段であり、
    完全再構成とユニタリ性の確認用に用いる。最終実用フィルタのような急峻な遷移帯域は
    責務に含めない。
    """

    def analysis(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """時間列を low/high 2 分岐へ解析する。

        Args:
            x: 入力信号。shape は `[..., n_sample]`。
                末尾軸が時間軸で、`n_sample` は偶数でなければならない。

        Returns:
            `(low, high)`。両者の shape は `[..., n_sample / 2]`。
        """
        arr = np.asarray(x, dtype=np.complex64)
        if arr.shape[-1] % 2 != 0:
            raise ValueError("analysis input length must be even.")
        # reshape 後の shape は [..., n_block, 2]。
        # 2 点 DFT により even/odd サンプル対を低域・高域の 2 分岐へ写す。
        blocks = arr.reshape(arr.shape[:-1] + (-1, 2))
        spectra = np.fft.fft(blocks, axis=-1) / np.sqrt(2.0)
        return spectra[..., 0], spectra[..., 1]

    def synthesis(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """low/high 2 分岐から元の時間列を再合成する。"""
        low_arr = np.asarray(low, dtype=np.complex64)
        high_arr = np.asarray(high, dtype=np.complex64)
        if low_arr.shape != high_arr.shape:
            raise ValueError("low and high branches must have identical shapes.")
        # stack 後の shape は [..., n_block, 2]。
        # 解析時の 1/sqrt(2) を打ち消すため sqrt(2) を掛けてから IFFT する。
        stacked = np.stack([low_arr, high_arr], axis=-1) * np.sqrt(2.0)
        blocks = np.fft.ifft(stacked, axis=-1)
        return blocks.reshape(blocks.shape[:-2] + (-1,))

    def branch_response(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """正規化角周波数に対する low/high 解析振幅を返す。"""
        w = np.asarray(omega, dtype=np.float32)
        low = np.abs((1.0 + np.exp(-1j * w)) / np.sqrt(2.0))
        high = np.abs((1.0 - np.exp(-1j * w)) / np.sqrt(2.0))
        return low, high

    def response_metrics(self, fft_size: int = 16384) -> dict[str, float]:
        """基準段の通過域リップル・阻止域減衰・相補誤差を評価する。"""
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
            # paraunitary 2 分岐では |H0|^2 + |H1|^2 が定数 2 に近いことが相補性指標になる。
            "power_complementarity_error": float(np.max(np.abs(low**2 + high**2 - 2.0))),
        }


def _ripple_db(magnitude: np.ndarray) -> float:
    mag = np.asarray(magnitude, dtype=np.float32)
    if mag.size == 0:
        raise ValueError("magnitude must not be empty.")
    eps = np.finfo(np.float32).tiny
    return float(20.0 * np.log10(max(float(np.max(mag)), eps) / max(float(np.min(mag)), eps)))
