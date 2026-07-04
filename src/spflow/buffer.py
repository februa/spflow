"""spflow.buffer を実装するモジュール。"""

from __future__ import annotations

from typing import Any

import numpy as np


class FrameBuffer:
    """逐次入力を固定長フレームへ切り出す汎用バッファ。

    このクラスは時間軸上で到着する配列を受け取り、`frame_size` 長のフレームを
    `hop_size` 刻みで返す。STFT やブロック FIR の前段で使うフレーミング補助であり、
    窓掛けや FFT 自体は責務に含めない。
    """

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
        """入力を蓄積し、取り出せるフレームを返す。

        Args:
            x: 入力配列。shape は `[..., n_sample]`。
                `axis` が時間軸で、それ以外の軸はチャネルなどの付随次元とする。

        Returns:
            フレーム列。各フレームの shape は `[..., frame_size]`。

        Raises:
            ValueError: 既存バッファと時間軸以外の shape が一致しない場合。
        """
        arr = np.asarray(x)
        work_axis = self._normalize_axis(arr.ndim)
        # moveaxis 後の shape は [..., n_sample]。
        # 末尾軸だけを時間軸として扱うことで、多チャネル入力でも同じ実装を再利用する。
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
            # hop_size だけ前進し、残りは次フレームとの重複区間として保持する。
            self._buffer = self._buffer[..., self.hop_size :]
        return frames

    def process(self, x: Any) -> list[np.ndarray]:
        """`push` の別名として入力をフレーム化する。"""
        return self.push(x)

    def flush(self, pad: bool = True, fill_value: float = 0.0) -> list[np.ndarray]:
        """末尾端数を必要に応じてゼロ詰めして排出する。

        Args:
            pad: `True` の場合、残りサンプルを `frame_size` まで埋めて返す。
                `False` の場合、端数は破棄する。
            fill_value: ゼロ詰め時に使う値。通常は 0.0。

        Returns:
            最終フレーム列。

        Notes:
            末尾端数を返さないと最後の短区間が完全に失われるため、
            周波数解析やブロック処理の末尾評価では `pad=True` を使うのが安全側である。
        """
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
        """内部バッファを空に戻す。"""
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
