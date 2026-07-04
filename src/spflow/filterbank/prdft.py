"""spflow.filterbank.prdft を実装するモジュール。"""

from __future__ import annotations

import numpy as np

from .design.prototype import make_pr_prototype
from .modulation import dft_analysis, dft_synthesis, full_dft_analysis, full_dft_synthesis
from .polyphase import frame_signal, overlap_add


class PRDFTFilterBank:
    """正周波数側のみを保持する実数 WOLA DFT フィルタバンク。

    実数信号向けに PR 用プロトタイプ窓を掛け、`rFFT` ベースで解析合成する。
    複素 full-band 処理は派生クラスへ委譲する。
    """

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
        """保持する正周波数サブバンド数を返す。"""
        return self.fft_size // 2 + 1

    def analysis(self, x: np.ndarray) -> np.ndarray:
        """実数時間列を正周波数サブバンドへ解析する。"""
        arr = np.asarray(x, dtype=np.float32)
        signal_axis = self._normalize_axis(self.axis, arr.ndim)
        # moveaxis 後の shape は [..., n_sample]。
        moved = np.moveaxis(arr, signal_axis, -1)

        frames, _ = frame_signal(moved, frame_size=self.fft_size, hop_size=self.hop_size)
        # frames shape: [..., fft_size, n_frame]
        # prototype[:, None] を掛けて各フレームへ同一解析窓を与える。
        windowed = frames * self.prototype[:, np.newaxis]
        subbands = dft_analysis(windowed, axis=-2)

        return np.moveaxis(subbands, -2, signal_axis)

    def synthesis(self, subbands: np.ndarray, length: int | None = None) -> np.ndarray:
        """正周波数サブバンドから実数時間列を再合成する。"""
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
    """負周波数側も明示保持する複素 full-band WOLA DFT フィルタバンク。"""

    @property
    def n_bands(self) -> int:
        """複素 full-band で保持する帯域数 `fft_size` を返す。"""
        return self.fft_size

    def analysis(self, x: np.ndarray) -> np.ndarray:
        """複素時間列を full-band 複素サブバンドへ解析する。"""
        arr = np.asarray(x, dtype=np.complex64)
        signal_axis = self._normalize_axis(self.axis, arr.ndim)
        moved = np.moveaxis(arr, signal_axis, -1)

        frames, _ = frame_signal(moved, frame_size=self.fft_size, hop_size=self.hop_size)
        windowed = frames * self.prototype[:, np.newaxis]
        subbands = full_dft_analysis(windowed, axis=-2)

        return np.moveaxis(subbands, -2, signal_axis)

    def synthesis(self, subbands: np.ndarray, length: int | None = None) -> np.ndarray:
        """full-band 複素サブバンドから時間列を再合成する。"""
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
    """将来のポリフェーズ DFT バンクへ向けた block-DFT 基準実装。

    現状はプロトタイプ FIR を持たない可逆な複素 block DFT バンクであり、
    任意複素サブバンド処理の閉包性確認を責務とする。
    """

    def __init__(self, fft_size: int, axis: int = -1) -> None:
        if fft_size <= 0:
            raise ValueError("fft_size must be positive.")
        self.fft_size = fft_size
        self.hop_size = fft_size
        self.axis = axis

    @property
    def n_bands(self) -> int:
        """block-DFT の帯域数 `fft_size` を返す。"""
        return self.fft_size

    def analysis(self, x: np.ndarray) -> np.ndarray:
        """複素時間列を block-DFT 複素サブバンドへ解析する。"""
        arr = np.asarray(x, dtype=np.complex64)
        signal_axis = self._normalize_axis(self.axis, arr.ndim)
        moved = np.moveaxis(arr, signal_axis, -1)
        frames, _ = frame_signal(moved, frame_size=self.fft_size, hop_size=self.hop_size)
        subbands = full_dft_analysis(frames, axis=-2)
        return np.moveaxis(subbands, -2, signal_axis)

    def synthesis(self, subbands: np.ndarray, length: int | None = None) -> np.ndarray:
        """block-DFT 複素サブバンドから時間列を再合成する。"""
        arr = np.asarray(subbands, dtype=np.complex64)
        band_axis = self._normalize_axis(self.axis, arr.ndim - 1)
        if arr.shape[band_axis] != self.n_bands:
            raise ValueError("subbands shape does not match the configured number of bands.")

        moved = np.moveaxis(arr, band_axis, -2)
        time_frames = full_dft_synthesis(moved, axis=-2)
        # time_frames shape: [..., fft_size, n_frame]
        # swapaxes 後 [..., n_frame, fft_size] を連結して block 列へ戻す。
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
    """初期実装との互換性のための別名クラス。"""
