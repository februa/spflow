from __future__ import annotations

from typing import Any

import numpy as np


class FrameBuffer:
    """Slice sequential array inputs into overlapped frames."""

    def __init__(self, frame_size: int, hop_size: int, axis: int = -1) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if hop_size <= 0:
            raise ValueError("hop_size must be positive.")
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.axis = axis
        self._buffer: np.ndarray | None = None

    def push(self, x: Any) -> list[np.ndarray]:
        arr = np.asarray(x)
        work_axis = self._normalize_axis(arr.ndim)
        moved = np.moveaxis(arr, work_axis, -1)

        if self._buffer is None:
            self._buffer = moved.copy()
        else:
            if self._buffer.ndim != moved.ndim or self._buffer.shape[:-1] != moved.shape[:-1]:
                raise ValueError("Input shape mismatch except along frame axis.")
            self._buffer = np.concatenate([self._buffer, moved], axis=-1)

        frames: list[np.ndarray] = []
        while self._buffer.shape[-1] >= self.frame_size:
            frame = self._buffer[..., : self.frame_size]
            frames.append(np.moveaxis(frame.copy(), -1, work_axis))
            self._buffer = self._buffer[..., self.hop_size :]
        return frames

    def process(self, x: Any) -> list[np.ndarray]:
        return self.push(x)

    def flush(self, pad: bool = True, fill_value: float = 0.0) -> list[np.ndarray]:
        if self._buffer is None or self._buffer.shape[-1] == 0:
            self.reset()
            return []

        if self._buffer.shape[-1] >= self.frame_size:
            return self._drain_existing_frames()

        if not pad:
            self.reset()
            return []

        padded = np.full(self._buffer.shape[:-1] + (self.frame_size,), fill_value, dtype=self._buffer.dtype)
        padded[..., : self._buffer.shape[-1]] = self._buffer
        axis = self._normalize_axis(padded.ndim)
        result = [np.moveaxis(padded, -1, axis)]
        self.reset()
        return result

    def reset(self) -> None:
        self._buffer = None

    def _drain_existing_frames(self) -> list[np.ndarray]:
        assert self._buffer is not None
        axis = self._normalize_axis(self._buffer.ndim)
        frames: list[np.ndarray] = []
        while self._buffer is not None and self._buffer.shape[-1] >= self.frame_size:
            frame = self._buffer[..., : self.frame_size]
            frames.append(np.moveaxis(frame.copy(), -1, axis))
            self._buffer = self._buffer[..., self.hop_size :]
        return frames

    def _normalize_axis(self, ndim: int) -> int:
        axis = self.axis
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis is out of bounds for input.")
        return axis
