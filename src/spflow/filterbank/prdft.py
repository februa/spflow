"""spflow.filterbank.prdft を実装するモジュール。"""

from __future__ import annotations

import numpy as np

from .design.prototype import make_pr_prototype
from .modulation import dft_analysis, dft_synthesis, full_dft_analysis, full_dft_synthesis
from .polyphase import frame_signal, overlap_add


class PRDFTFilterBank:
    """Real-valued WOLA DFT filter bank with positive-frequency subbands only."""

    def __init__(
        self,
        fft_size: int,
        hop_size: int | None = None,
        prototype: np.ndarray | None = None,
        axis: int = -1,
    ) -> None:
        if fft_size <= 0:
            raise ValueError("fft_size must be positive.")
        if fft_size % 2 != 0:
            raise ValueError("fft_size must be even.")

        self.fft_size = fft_size
        self.hop_size = fft_size // 2 if hop_size is None else hop_size
        self.axis = axis
        self.prototype = (
            make_pr_prototype(fft_size)
            if prototype is None
            else np.asarray(prototype, dtype=np.float32)
        )

        if self.prototype.shape != (fft_size,):
            raise ValueError("prototype must have shape (fft_size,).")
        if self.hop_size <= 0:
            raise ValueError("hop_size must be positive.")

    @property
    def n_bands(self) -> int:
        return self.fft_size // 2 + 1

    def analysis(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        signal_axis = self._normalize_axis(self.axis, arr.ndim)
        moved = np.moveaxis(arr, signal_axis, -1)

        frames, _ = frame_signal(moved, frame_size=self.fft_size, hop_size=self.hop_size)
        windowed = frames * self.prototype[:, np.newaxis]
        subbands = dft_analysis(windowed, axis=-2)

        return np.moveaxis(subbands, -2, signal_axis)

    def synthesis(self, subbands: np.ndarray, length: int | None = None) -> np.ndarray:
        arr = np.asarray(subbands, dtype=np.complex64)
        band_axis = self._normalize_axis(self.axis, arr.ndim - 1)
        if arr.shape[band_axis] != self.n_bands:
            raise ValueError("subbands shape does not match the configured number of bands.")

        moved = np.moveaxis(arr, band_axis, -2)
        time_frames = dft_synthesis(moved, fft_size=self.fft_size, axis=-2)
        reconstructed = overlap_add(
            time_frames,
            window=self.prototype,
            hop_size=self.hop_size,
            length=length,
        )

        return np.moveaxis(reconstructed, -1, band_axis)

    @staticmethod
    def _normalize_axis(axis: int, ndim: int) -> int:
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis is out of bounds for input.")
        return axis


class FullDFTFilterBank(PRDFTFilterBank):
    """Complex full-band WOLA DFT filter bank with explicit negative-frequency bands."""

    @property
    def n_bands(self) -> int:
        return self.fft_size

    def analysis(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.complex64)
        signal_axis = self._normalize_axis(self.axis, arr.ndim)
        moved = np.moveaxis(arr, signal_axis, -1)

        frames, _ = frame_signal(moved, frame_size=self.fft_size, hop_size=self.hop_size)
        windowed = frames * self.prototype[:, np.newaxis]
        subbands = full_dft_analysis(windowed, axis=-2)

        return np.moveaxis(subbands, -2, signal_axis)

    def synthesis(self, subbands: np.ndarray, length: int | None = None) -> np.ndarray:
        arr = np.asarray(subbands, dtype=np.complex64)
        band_axis = self._normalize_axis(self.axis, arr.ndim - 1)
        if arr.shape[band_axis] != self.n_bands:
            raise ValueError("subbands shape does not match the configured number of bands.")

        moved = np.moveaxis(arr, band_axis, -2)
        time_frames = full_dft_synthesis(moved, axis=-2)
        reconstructed = overlap_add(
            time_frames,
            window=self.prototype,
            hop_size=self.hop_size,
            length=length,
        )

        return np.moveaxis(reconstructed, -1, band_axis)


class PolyphaseDFTFilterBank:
    """Initial baseline of the future polyphase DFT filter bank.

    The current implementation is intentionally simple: a critically sampled full-band
    block DFT bank without a prototype FIR. In other words, this is the reversible
    complex block-DFT baseline used to validate arbitrary complex subband processing.

    It is intentionally separate from the existing WOLA/STFT-style filter banks so that
    arbitrary complex subband coefficients satisfy analysis(synthesis(Y)) == Y up to
    numerical precision.
    """

    def __init__(self, fft_size: int, axis: int = -1) -> None:
        if fft_size <= 0:
            raise ValueError("fft_size must be positive.")
        self.fft_size = fft_size
        self.hop_size = fft_size
        self.axis = axis

    @property
    def n_bands(self) -> int:
        return self.fft_size

    def analysis(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.complex64)
        signal_axis = self._normalize_axis(self.axis, arr.ndim)
        moved = np.moveaxis(arr, signal_axis, -1)
        frames, _ = frame_signal(moved, frame_size=self.fft_size, hop_size=self.hop_size)
        subbands = full_dft_analysis(frames, axis=-2)
        return np.moveaxis(subbands, -2, signal_axis)

    def synthesis(self, subbands: np.ndarray, length: int | None = None) -> np.ndarray:
        arr = np.asarray(subbands, dtype=np.complex64)
        band_axis = self._normalize_axis(self.axis, arr.ndim - 1)
        if arr.shape[band_axis] != self.n_bands:
            raise ValueError("subbands shape does not match the configured number of bands.")

        moved = np.moveaxis(arr, band_axis, -2)
        time_frames = full_dft_synthesis(moved, axis=-2)
        reconstructed = np.swapaxes(time_frames, -2, -1).reshape(time_frames.shape[:-2] + (-1,))
        if length is not None:
            return np.moveaxis(reconstructed[..., :length], -1, band_axis)
        return np.moveaxis(reconstructed, -1, band_axis)

    @staticmethod
    def _normalize_axis(axis: int, ndim: int) -> int:
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis is out of bounds for input.")
        return axis


class DFT_FilterBank(PRDFTFilterBank):
    """Compatibility alias for the initial Beamforming-free implementation."""
