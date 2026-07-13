"""系列別の非負整数sample遅延をblock境界をまたいで適用する。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow.simulation.signal_block import SignalBlock

NumericArray = NDArray[Any]
IntArray = NDArray[np.int64]


class StatefulIntegerDelay:
    """系列ごとの非負整数遅延をブロック境界をまたいで適用する。

    入力と出力は shape ``[n_series,n_sample]`` の実数または複素信号であり、遅延の
    単位はsampleである。未到来の履歴は無効として返す。小数遅延、FIR、係数更新、
    系列間の加算は責務に含めない。
    """

    def __init__(self, delays_samples: IntArray) -> None:
        """整数遅延器を構築する。

        Args:
            delays_samples: 系列別遅延。shape ``[n_series]``、単位sample。

        Raises:
            ValueError: 1次元でない、空、または負の遅延を含む場合。

        Notes:
            初回入力より前のsampleは存在しないため、値を0、valid maskをFalseにする。
        """
        delays = np.asarray(delays_samples, dtype=np.int64)
        if delays.ndim != 1 or delays.size == 0:
            raise ValueError("delays_samples must be a non-empty 1-D array.")
        if bool(np.any(delays < 0)):
            raise ValueError("delays_samples must be non-negative.")
        self._delays = delays.copy()
        self._max_delay = int(np.max(delays))
        self._history: NumericArray | None = None
        self._history_valid = np.zeros((delays.size, self._max_delay), dtype=np.bool_)

    @property
    def delays_samples(self) -> IntArray:
        """系列別遅延のコピーをshape ``[n_series]``、単位sampleで返す。"""
        return self._delays.copy()

    def reset(self) -> None:
        """入力系列の切替時に履歴を破棄し、全遅延区間を未完成へ戻す。

        Notes:
            遅延設定と入力dtype契約は維持し、次blockを同じ精度の新系列先頭として扱う。
        """
        if self._history is not None:
            self._history.fill(0.0)
        self._history_valid.fill(False)

    def process(self, data: NumericArray) -> SignalBlock:
        """1ブロックへ系列別整数遅延を適用する。

        Args:
            data: 入力。shape ``[n_series,n_sample]``。

        Returns:
            遅延後の値と完成mask。いずれも入力と同じshape。

        Raises:
            ValueError: 次元、系列数、dtype、または履歴途中の精度が契約と異なる場合。
        """
        block = np.asarray(data)
        if block.ndim != 2 or block.shape[0] != self._delays.size:
            raise ValueError("data must have shape (n_series, n_sample).")
        supported_dtypes = (
            np.dtype(np.float32),
            np.dtype(np.float64),
            np.dtype(np.complex64),
            np.dtype(np.complex128),
        )
        if block.dtype not in supported_dtypes:
            raise ValueError("data dtype must be float32, float64, complex64, or complex128.")
        if self._history is None:
            self._history = np.zeros((self._delays.size, self._max_delay), dtype=block.dtype)
        elif block.dtype != self._history.dtype:
            raise ValueError("data dtype must remain unchanged until reset and reconstruction.")
        n_sample = block.shape[1]
        if self._max_delay == 0:
            return SignalBlock(block.copy(), np.ones(block.shape, dtype=np.bool_))

        # extendedのaxis=1は時間であり、前ブロック末尾の履歴を現在blockの直前へ接続する。
        extended = np.concatenate((self._history, block), axis=1)
        current_valid = np.ones(block.shape, dtype=np.bool_)
        extended_valid = np.concatenate((self._history_valid, current_valid), axis=1)
        output = np.empty_like(block)
        output_valid = np.empty(block.shape, dtype=np.bool_)
        for series_index, delay in enumerate(self._delays):
            start = self._max_delay - int(delay)
            output[series_index] = extended[series_index, start : start + n_sample]
            output_valid[series_index] = extended_valid[series_index, start : start + n_sample]

        # 出力計算が完了してから履歴を公開状態へ進め、例外時の中途更新を避ける。
        self._history[...] = extended[:, -self._max_delay :]
        self._history_valid[...] = extended_valid[:, -self._max_delay :]
        return SignalBlock(output, output_valid)
