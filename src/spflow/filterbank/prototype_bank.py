"""spflow.filterbank.prototype_bank を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrototypeFilter:
    """プロトタイプベース DFT フィルタバンク用の FIR 定義。"""

    coefficients: np.ndarray
    n_band: int
    decimation: int

    def __post_init__(self) -> None:
        coeffs = np.asarray(self.coefficients, dtype=np.complex64)
        if coeffs.ndim != 1:
            raise ValueError("coefficients must be a 1-D array.")
        if self.n_band <= 0:
            raise ValueError("n_band must be positive.")
        if self.decimation <= 0:
            raise ValueError("decimation must be positive.")
        if self.n_band != self.decimation:
            raise ValueError("This initial implementation requires n_band == decimation.")
        if coeffs.size == 0:
            raise ValueError("coefficients must not be empty.")
        if coeffs.size % self.decimation != 0:
            raise ValueError("prototype length must be a multiple of decimation.")
        object.__setattr__(self, "coefficients", coeffs)

    @property
    def prototype_length(self) -> int:
        """プロトタイプ tap 長を返す。"""
        return int(self.coefficients.size)

    @property
    def n_phase(self) -> int:
        """ポリフェーズ分解後の相数を返す。"""
        return self.prototype_length // self.decimation

    @classmethod
    def block_dft_baseline(
        cls,
        *,
        n_band: int,
        decimation: int,
        prototype_length: int,
    ) -> "PrototypeFilter":
        """先頭 1 block だけ 1 の block-DFT 基準プロトタイプを作る。"""
        if prototype_length % decimation != 0:
            raise ValueError("prototype_length must be a multiple of decimation.")
        coeffs = np.zeros(prototype_length, dtype=np.complex64)
        coeffs[:decimation] = 1.0
        return cls(coeffs, n_band=n_band, decimation=decimation)

    @classmethod
    def windowed_sinc(
        cls,
        *,
        n_band: int,
        decimation: int,
        prototype_length: int,
        cutoff: float | None = None,
    ) -> "PrototypeFilter":
        """窓付き sinc 低域原型を作る。"""
        if prototype_length % decimation != 0:
            raise ValueError("prototype_length must be a multiple of decimation.")
        if cutoff is None:
            cutoff = 1.0 / decimation
        if cutoff <= 0.0 or cutoff >= 0.5:
            raise ValueError("cutoff must lie in (0, 0.5).")

        n = np.arange(prototype_length, dtype=np.float32)
        center = 0.5 * (prototype_length - 1)
        taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * (n - center))
        taps *= np.hamming(prototype_length)
        taps /= np.sum(taps)
        return cls(taps.astype(np.complex64), n_band=n_band, decimation=decimation)


class PolyphaseDecomposition:
    """プロトタイプ係数をポリフェーズ行列へ並べ替える。"""

    def __init__(self, decimation: int) -> None:
        if decimation <= 0:
            raise ValueError("decimation must be positive.")
        self.decimation = decimation

    def decompose(self, prototype: PrototypeFilter | np.ndarray) -> np.ndarray:
        """1 次元係数列を `[n_phase, decimation]` のポリフェーズ行列へ変換する。"""
        coeffs = prototype.coefficients if isinstance(prototype, PrototypeFilter) else np.asarray(prototype)
        coeffs = np.asarray(coeffs, dtype=np.complex64)
        if coeffs.ndim != 1:
            raise ValueError("prototype must be one-dimensional.")
        if coeffs.size % self.decimation != 0:
            raise ValueError("prototype length must be a multiple of decimation.")
        # reshape 後の axis=0 は polyphase 相、axis=1 は decimation 内サンプル位置を表す。
        return coeffs.reshape(coeffs.size // self.decimation, self.decimation)


class _PrototypeBankBase:
    def __init__(self, *, prototype: PrototypeFilter, band_order: str = "fft", axis: int = -1) -> None:
        if band_order != "fft":
            raise ValueError("Only FFT band order is currently supported.")
        self.prototype = prototype
        self.n_band = prototype.n_band
        self.decimation = prototype.decimation
        self.prototype_length = prototype.prototype_length
        self.band_order = band_order
        self.axis = axis
        self._polyphase = PolyphaseDecomposition(self.decimation).decompose(prototype)

    def _normalize_axis(self, axis: int, ndim: int) -> int:
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis is out of bounds for input.")
        return axis

    def _frame_signal(self, x: np.ndarray) -> np.ndarray:
        n_samples = x.shape[-1]
        if n_samples == 0:
            return np.zeros(x.shape[:-1] + (self.prototype_length, 0), dtype=np.complex64)
        n_frames = int(np.ceil(n_samples / self.decimation))
        padded_length = self.prototype_length + max(0, n_frames - 1) * self.decimation
        pad_width = padded_length - n_samples
        padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad_width)])
        frames = []
        for frame_idx in range(n_frames):
            start = frame_idx * self.decimation
            stop = start + self.prototype_length
            frames.append(padded[..., start:stop])
        return np.stack(frames, axis=-1)


class PrototypeAnalysisDFTFilterBank(_PrototypeBankBase):
    """FFT 順帯域を持つプロトタイプベース複素 DFT 解析バンク。"""

    def __init__(self, *, prototype: PrototypeFilter, band_order: str = "fft", axis: int = -1) -> None:
        super().__init__(prototype=prototype, band_order=band_order, axis=axis)

    def analysis(self, x: np.ndarray) -> np.ndarray:
        """入力複素時間列をプロトタイプベース DFT サブバンドへ解析する。"""
        arr = np.asarray(x, dtype=np.complex64)
        signal_axis = self._normalize_axis(self.axis, arr.ndim)
        moved = np.moveaxis(arr, signal_axis, -1)
        frames = self._frame_signal(moved)

        prefix_shape = frames.shape[:-2]
        n_frame = frames.shape[-1]
        # reshaped shape: [..., n_phase, decimation, n_frame]
        # 1 フレームをポリフェーズ相と相内サンプルへ明示分解する。
        reshaped = frames.reshape(prefix_shape + (self._polyphase.shape[0], self.decimation, n_frame))
        weights = self._polyphase.reshape((1,) * len(prefix_shape) + self._polyphase.shape + (1,))
        weighted = reshaped * weights
        polyphase_sum = np.sum(weighted, axis=-3)
        subbands = np.fft.fft(polyphase_sum, axis=-2)
        return np.moveaxis(subbands, -2, signal_axis)


class PrototypeSynthesisDFTFilterBank(_PrototypeBankBase):
    """手動遅延補償付きプロトタイプベース複素 DFT 合成バンク。"""

    def __init__(
        self,
        *,
        prototype: PrototypeFilter,
        delay_compensation: int = 0,
        band_order: str = "fft",
        axis: int = -1,
    ) -> None:
        super().__init__(prototype=prototype, band_order=band_order, axis=axis)
        self.delay_compensation = delay_compensation

    def synthesis(self, subbands: np.ndarray, length: int | None = None) -> np.ndarray:
        """複素サブバンド列から時間波形を再合成する。"""
        arr = np.asarray(subbands, dtype=np.complex64)
        band_axis = self._normalize_axis(self.axis, arr.ndim - 1)
        if arr.shape[band_axis] != self.n_band:
            raise ValueError("subbands shape does not match the configured number of bands.")

        moved = np.moveaxis(arr, band_axis, -2)
        polyphase_samples = np.fft.ifft(moved, axis=-2)
        prefix_shape = polyphase_samples.shape[:-2]
        n_frame = polyphase_samples.shape[-1]
        weights = self._polyphase.reshape((1,) * len(prefix_shape) + self._polyphase.shape + (1,))
        expanded = polyphase_samples[..., np.newaxis, :, :] * weights
        # expanded shape: [..., n_phase, decimation, n_frame]。
        # 相ごとの重み付き出力を prototype_length 軸へ戻して overlap-add する。
        frames = expanded.reshape(prefix_shape + (self.prototype_length, n_frame))
        reconstructed = self._sum_overlap(frames)
        compensated = self._apply_delay_compensation(reconstructed)
        if length is not None:
            compensated = compensated[..., :length]
        return np.moveaxis(compensated, -1, band_axis)

    def _sum_overlap(self, frames: np.ndarray) -> np.ndarray:
        frame_size = frames.shape[-2]
        n_frame = frames.shape[-1]
        out_length = frame_size + max(0, n_frame - 1) * self.decimation
        output = np.zeros(frames.shape[:-2] + (out_length,), dtype=np.complex64)
        for frame_idx in range(n_frame):
            start = frame_idx * self.decimation
            stop = start + frame_size
            output[..., start:stop] += frames[..., :, frame_idx]
        return output

    def _apply_delay_compensation(self, x: np.ndarray) -> np.ndarray:
        delay = self.delay_compensation
        if delay == 0:
            return x
        if delay > 0:
            if delay >= x.shape[-1]:
                return np.zeros(x.shape[:-1] + (0,), dtype=x.dtype)
            return x[..., delay:]
        pad = np.zeros(x.shape[:-1] + (-delay,), dtype=x.dtype)
        return np.concatenate([pad, x], axis=-1)


class PRChecker:
    """再構成誤差とプロトタイプ応答指標を評価する。"""

    def __init__(self, analysis_bank: PrototypeAnalysisDFTFilterBank, synthesis_bank: PrototypeSynthesisDFTFilterBank) -> None:
        self.analysis_bank = analysis_bank
        self.synthesis_bank = synthesis_bank

    def check_perfect_reconstruction(self, x: np.ndarray, length: int | None = None) -> dict[str, float]:
        """解析後に再合成したときの PR 誤差を返す。"""
        target = np.asarray(x, dtype=np.complex64)
        subbands = self.analysis_bank.analysis(target)
        reconstructed = self.synthesis_bank.synthesis(
            subbands,
            length=length if length is not None else target.shape[-1],
        )
        error = reconstructed - target[..., : reconstructed.shape[-1]]
        return {
            "max_abs_error": float(np.max(np.abs(error))),
            "rms_error": float(np.sqrt(np.mean(np.abs(error) ** 2))),
        }

    def check_subband_closure(self, y: np.ndarray) -> dict[str, float]:
        """任意サブバンド列が合成後の再解析でどれだけ閉じているか評価する。"""
        target = np.asarray(y, dtype=np.complex64)
        time_signal = self.synthesis_bank.synthesis(target, length=target.shape[-1] * self.synthesis_bank.decimation)
        reanalyzed = self.analysis_bank.analysis(time_signal)
        error = reanalyzed - target
        return {
            "max_abs_error": float(np.max(np.abs(error))),
            "rms_error": float(np.sqrt(np.mean(np.abs(error) ** 2))),
        }

    @staticmethod
    def evaluate_prototype_response(prototype: PrototypeFilter, fft_size: int = 16384) -> dict[str, float]:
        """プロトタイプ FIR 単体の通過域・阻止域指標を返す。"""
        response = np.fft.fft(prototype.coefficients, n=fft_size)
        magnitude = np.abs(response[: fft_size // 2 + 1])
        passband_edge = max(1, int(np.floor(fft_size / (2 * prototype.decimation))))
        stopband_start = min(magnitude.size, max(passband_edge + 1, int(np.ceil(3 * fft_size / (2 * prototype.decimation)))))
        passband_peak = float(np.max(magnitude[:passband_edge]))
        stopband_peak = float(np.max(magnitude[stopband_start:])) if stopband_start < magnitude.size else 0.0
        stopband_attenuation_db = float(-20.0 * np.log10(stopband_peak / passband_peak)) if stopband_peak > 0.0 and passband_peak > 0.0 else float("inf")
        return {
            "passband_peak": passband_peak,
            "stopband_peak": stopband_peak,
            "stopband_attenuation_db": stopband_attenuation_db,
        }
