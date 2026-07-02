"""spflow.frequency.overlap_save を実装するモジュール。"""

from __future__ import annotations

import numpy as np


class OverlapSaveBuffer:
    """Build overlap-save frames from sequential valid-size input blocks."""

    def __init__(self, frame_size: int, valid_size: int, axis: int = -1) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if valid_size <= 0:
            raise ValueError("valid_size must be positive.")
        if valid_size > frame_size:
            raise ValueError("valid_size must not exceed frame_size.")

        self.frame_size = frame_size
        self.valid_size = valid_size
        self.overlap_size = frame_size - valid_size
        self.axis = axis
        self._pending: np.ndarray | None = None
        self._history: np.ndarray | None = None

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        arr = np.asarray(x)
        work_axis = self._normalize_axis(arr.ndim)
        moved = np.moveaxis(arr, work_axis, -1)

        if self._pending is None:
            self._pending = moved.copy()
        else:
            if self._pending.ndim != moved.ndim or self._pending.shape[:-1] != moved.shape[:-1]:
                raise ValueError("Input shape mismatch except along processing axis.")
            self._pending = np.concatenate([self._pending, moved], axis=-1)

        if self._history is None:
            history_shape = moved.shape[:-1] + (self.overlap_size,)
            self._history = np.zeros(history_shape, dtype=moved.dtype)

        frames: list[np.ndarray] = []
        while self._pending.shape[-1] >= self.valid_size:
            block = self._pending[..., : self.valid_size]
            frame = np.concatenate([self._history, block], axis=-1)
            frames.append(np.moveaxis(frame.copy(), -1, work_axis))
            if self.overlap_size > 0:
                self._history = frame[..., -self.overlap_size :].copy()
            self._pending = self._pending[..., self.valid_size :]
        return frames

    def process(self, x: np.ndarray) -> list[np.ndarray]:
        return self.push(x)

    def flush(self, pad: bool = True, fill_value: float = 0.0) -> list[np.ndarray]:
        if self._pending is None or self._pending.shape[-1] == 0:
            self.reset()
            return []
        if not pad:
            self.reset()
            return []

        pad_width = self.valid_size - self._pending.shape[-1]
        padded = np.pad(
            self._pending,
            [(0, 0)] * (self._pending.ndim - 1) + [(0, pad_width)],
            constant_values=fill_value,
        )
        self._pending = padded
        frames = self.push(np.zeros(padded.shape[:-1] + (0,), dtype=padded.dtype))
        self.reset()
        return frames

    def reset(self) -> None:
        self._pending = None
        self._history = None

    def _normalize_axis(self, ndim: int) -> int:
        axis = self.axis
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis is out of bounds for input.")
        return axis


class ValidRegionExtractor:
    """Extract the valid overlap-save region from a processed frame."""

    def __init__(self, frame_size: int, valid_size: int, axis: int = -1) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if valid_size <= 0:
            raise ValueError("valid_size must be positive.")
        if valid_size > frame_size:
            raise ValueError("valid_size must not exceed frame_size.")
        self.frame_size = frame_size
        self.valid_size = valid_size
        self.axis = axis

    def process(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        work_axis = self._normalize_axis(arr.ndim)
        moved = np.moveaxis(arr, work_axis, -1)
        valid = moved[..., -self.valid_size :]
        return np.moveaxis(valid, -1, work_axis)

    def _normalize_axis(self, ndim: int) -> int:
        axis = self.axis
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis is out of bounds for input.")
        return axis


def make_filter_fft(filters: np.ndarray, frame_size: int, axis: int = -1) -> np.ndarray:
    """Zero-pad time-domain filters on the tail and transform them for overlap-save use."""

    if frame_size <= 0:
        raise ValueError("frame_size must be positive.")

    arr = np.asarray(filters, dtype=np.complex64)
    work_axis = axis
    if work_axis < 0:
        work_axis += arr.ndim
    if work_axis < 0 or work_axis >= arr.ndim:
        raise ValueError("axis is out of bounds for filters.")

    moved = np.moveaxis(arr, work_axis, -1)
    if moved.shape[-1] > frame_size:
        raise ValueError("filter length must not exceed frame_size.")

    padded = np.zeros(moved.shape[:-1] + (frame_size,), dtype=np.complex64)
    padded[..., : moved.shape[-1]] = moved
    spectrum = np.fft.fft(padded, axis=-1)
    return np.moveaxis(spectrum, -1, work_axis)
