"""spflow.filterbank.causal_analytic_frontend を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalAnalyticResult:
    """因果 analytic front-end の出力と遅延メタデータを保持する。"""

    samples: np.ndarray
    delay_samples_at_root_rate: int
    time_origin_at_root_rate: int = 0


def design_hilbert_fir(num_taps: int, window: str = "hamming") -> np.ndarray:
    """窓付き理想応答から因果 FIR ヒルベルト変換器を設計する。"""
    if num_taps <= 1 or num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd integer greater than 1.")

    center = num_taps // 2
    n = np.arange(num_taps, dtype=np.float32) - center
    taps = np.zeros(num_taps, dtype=np.float32)
    # 理想ヒルベルト変換器 h[n] = 2 / (π n) の奇数サンプルだけを採用する。
    odd = (np.abs(n) > 0.0) & (np.mod(np.abs(n), 2.0) == 1.0)
    taps[odd] = 2.0 / (np.pi * n[odd])

    if window == "hamming":
        win = np.hamming(num_taps)
    elif window == "hann":
        win = np.hanning(num_taps)
    elif window == "rect":
        win = np.ones(num_taps, dtype=np.float32)
    else:
        raise ValueError("window must be 'hamming', 'hann', or 'rect'.")

    return taps * win


class CausalAnalyticFrontend:
    """FIR ヒルベルト変換器ベースの因果 analytic front-end。

    実数入力から遅延付きの複素 analytic 信号を生成し、後段の複素フィルタバンクへ
    渡す。非因果な FFT ベース解析や帯域分割自体は責務に含めない。
    """

    def __init__(self, hilbert_taps: np.ndarray) -> None:
        taps = np.asarray(hilbert_taps, dtype=np.float32)
        if taps.ndim != 1 or taps.size <= 1 or taps.size % 2 == 0:
            raise ValueError("hilbert_taps must be a 1D odd-length array with at least 3 taps.")
        self.hilbert_taps = taps
        self.delay_samples = taps.size // 2

    @classmethod
    def default(cls, num_taps: int = 63, window: str = "hamming") -> "CausalAnalyticFrontend":
        """既定 tap 数と窓で front-end を構築する。"""
        return cls(design_hilbert_fir(num_taps=num_taps, window=window))

    def analyze(self, x: np.ndarray, *, pad_tail: bool = False) -> CausalAnalyticResult:
        """実数入力を因果 analytic 信号へ変換する。

        Args:
            x: 実数入力。shape は `[..., n_sample]`。
            pad_tail: `True` の場合、ヒルベルト FIR の群遅延ぶんだけ末尾をゼロ詰めし、
                最終サンプルの虚部応答も回収する。

        Returns:
            `samples` shape が `[..., n_sample]` または `[..., n_sample + delay]` の
            `CausalAnalyticResult`。
        """
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError("input must have at least one dimension.")

        if pad_tail and self.delay_samples > 0:
            pad_spec = [(0, 0)] * arr.ndim
            pad_spec[-1] = (0, self.delay_samples)
            work = np.pad(arr, pad_spec)
        else:
            work = arr

        delayed_real = self._delay_signal(work)
        imag = self._convolve_last_axis(work, self.hilbert_taps)
        # 実部は群遅延ぶんだけ遅らせた原信号、虚部はヒルベルト変換出力とすることで、
        # causal な analytic 近似 x_d[n] + j h{x}[n] を構成する。
        samples = delayed_real + 1j * imag
        return CausalAnalyticResult(
            samples=samples,
            delay_samples_at_root_rate=self.delay_samples,
            time_origin_at_root_rate=0,
        )

    def recover_real(self, result: CausalAnalyticResult | np.ndarray, *, length: int | None = None) -> np.ndarray:
        """analytic 出力から遅延補償済み実数波形を回復する。"""
        samples = result.samples if isinstance(result, CausalAnalyticResult) else np.asarray(result, dtype=np.complex64)
        start = self.delay_samples
        stop = None if length is None else start + length
        return np.asarray(np.real(samples)[..., start:stop], dtype=np.float32)

    def _delay_signal(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        out = np.zeros_like(arr, dtype=np.float32)
        if self.delay_samples >= arr.shape[-1]:
            return out
        out[..., self.delay_samples :] = arr[..., : arr.shape[-1] - self.delay_samples]
        return out

    @staticmethod
    def _convolve_last_axis(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        filt = np.asarray(taps, dtype=np.float32)
        rows = int(np.prod(arr.shape[:-1])) if arr.ndim > 1 else 1
        # reshape 後の shape は [n_row, n_sample]。
        # 先頭軸をまとめることで、末尾時間軸への FIR を全行へ同じ規約で適用する。
        reshaped = arr.reshape(rows, arr.shape[-1])
        out = np.zeros((rows, arr.shape[-1]), dtype=np.float32)
        for row_idx in range(rows):
            full = np.convolve(reshaped[row_idx], filt, mode="full")
            out[row_idx] = full[: arr.shape[-1]]
        return out.reshape(arr.shape)


class CausalAnalyticFrontendStreamer:
    """`CausalAnalyticFrontend` の逐次処理ラッパー。

    厳密には毎回オフライン結果を再計算して新規出力だけを切り出す参照実装であり、
    計算量最適化は責務に含めない。まずはオフライン一致性を優先する。
    """

    def __init__(self, frontend: CausalAnalyticFrontend) -> None:
        self.frontend = frontend
        self._input: np.ndarray | None = None
        self._emitted = 0

    def process(self, x: np.ndarray) -> CausalAnalyticResult:
        """入力チャンクを蓄積し、新たに確定した analytic サンプルだけ返す。"""
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError("input chunk must have at least one dimension.")
        if arr.shape[-1] == 0:
            return CausalAnalyticResult(
                samples=np.zeros(arr.shape[:-1] + (0,), dtype=np.complex64),
                delay_samples_at_root_rate=self.frontend.delay_samples,
                time_origin_at_root_rate=0,
            )

        if self._input is None:
            self._input = arr.copy()
        else:
            if self._input.shape[:-1] != arr.shape[:-1]:
                raise ValueError("streaming input shape mismatch except along time axis.")
            self._input = np.concatenate([self._input, arr], axis=-1)

        all_out = self.frontend.analyze(self._input, pad_tail=False)
        new = all_out.samples[..., self._emitted :]
        self._emitted = all_out.samples.shape[-1]
        return CausalAnalyticResult(
            samples=new,
            delay_samples_at_root_rate=all_out.delay_samples_at_root_rate,
            time_origin_at_root_rate=all_out.time_origin_at_root_rate,
        )

    def flush(self) -> CausalAnalyticResult:
        """末尾をゼロ詰めして FIR 過渡の残りを回収する。"""
        if self._input is None:
            return CausalAnalyticResult(
                samples=np.zeros((0,), dtype=np.complex64),
                delay_samples_at_root_rate=self.frontend.delay_samples,
                time_origin_at_root_rate=0,
            )
        all_out = self.frontend.analyze(self._input, pad_tail=True)
        tail = all_out.samples[..., self._emitted :]
        self._emitted = all_out.samples.shape[-1]
        return CausalAnalyticResult(
            samples=tail,
            delay_samples_at_root_rate=all_out.delay_samples_at_root_rate,
            time_origin_at_root_rate=all_out.time_origin_at_root_rate,
        )
