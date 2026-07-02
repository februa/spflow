"""spflow.filterbank.causal_analytic_frontend を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalAnalyticResult:
    """Analytic frontend output plus explicit root-rate delay metadata."""

    samples: np.ndarray
    delay_samples_at_root_rate: int
    time_origin_at_root_rate: int = 0


def design_hilbert_fir(num_taps: int, window: str = "hamming") -> np.ndarray:
    """Design a causal FIR Hilbert transformer using a windowed ideal impulse response."""

    if num_taps <= 1 or num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd integer greater than 1.")

    center = num_taps // 2
    n = np.arange(num_taps, dtype=np.float32) - center
    taps = np.zeros(num_taps, dtype=np.float32)
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
    """Causal FIR Hilbert-transformer-based analytic frontend."""

    def __init__(self, hilbert_taps: np.ndarray) -> None:
        taps = np.asarray(hilbert_taps, dtype=np.float32)
        if taps.ndim != 1 or taps.size <= 1 or taps.size % 2 == 0:
            raise ValueError("hilbert_taps must be a 1D odd-length array with at least 3 taps.")
        self.hilbert_taps = taps
        self.delay_samples = taps.size // 2

    @classmethod
    def default(cls, num_taps: int = 63, window: str = "hamming") -> "CausalAnalyticFrontend":
        return cls(design_hilbert_fir(num_taps=num_taps, window=window))

    def analyze(self, x: np.ndarray, *, pad_tail: bool = False) -> CausalAnalyticResult:
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
        samples = delayed_real + 1j * imag
        return CausalAnalyticResult(
            samples=samples,
            delay_samples_at_root_rate=self.delay_samples,
            time_origin_at_root_rate=0,
        )

    def recover_real(self, result: CausalAnalyticResult | np.ndarray, *, length: int | None = None) -> np.ndarray:
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
        reshaped = arr.reshape(rows, arr.shape[-1])
        out = np.zeros((rows, arr.shape[-1]), dtype=np.float32)
        for row_idx in range(rows):
            full = np.convolve(reshaped[row_idx], filt, mode="full")
            out[row_idx] = full[: arr.shape[-1]]
        return out.reshape(arr.shape)


class CausalAnalyticFrontendStreamer:
    """Exact-by-construction streaming wrapper for CausalAnalyticFrontend."""

    def __init__(self, frontend: CausalAnalyticFrontend) -> None:
        self.frontend = frontend
        self._input: np.ndarray | None = None
        self._emitted = 0

    def process(self, x: np.ndarray) -> CausalAnalyticResult:
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
