"""逐次信号処理のブロック分割と完成状態を検証する支援部品。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

NumericArray = NDArray[Any]
ComplexArray = NDArray[np.complexfloating[Any, Any]]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SignalBlock:
    """逐次処理の値と、その各sampleが完成済みかを保持する。

    ``data`` と ``valid_mask`` はともに shape ``[n_series, n_sample]`` である。
    この結果型は完成状態を運ぶだけで、時刻、単位、処理周期は決めない。
    """

    data: NumericArray
    valid_mask: BoolArray


class StatefulIntegerDelay:
    """系列ごとの非負整数遅延をブロック境界をまたいで適用する。

    入力と出力は shape ``[n_series, n_sample]`` の複素信号であり、遅延の単位は
    sampleである。未到来の履歴は無効として返す。小数遅延、FIR、系列間の加算は
    責務に含めない。
    """

    def __init__(self, delays_samples: IntArray) -> None:
        """整数遅延器を構築する。

        Args:
            delays_samples: 系列別遅延。shape ``[n_series]``、単位はsample。

        Raises:
            ValueError: 1次元でない、空、または負の遅延を含む場合。
        """
        delays = np.asarray(delays_samples, dtype=np.int64)
        if delays.ndim != 1 or delays.size == 0:
            raise ValueError("delays_samples must be a non-empty 1-D array.")
        if bool(np.any(delays < 0)):
            raise ValueError("delays_samples must be non-negative.")
        self._delays = delays.copy()
        self._max_delay = int(np.max(delays))
        self._history: ComplexArray | None = None
        self._history_valid = np.zeros((delays.size, self._max_delay), dtype=np.bool_)

    @property
    def delays_samples(self) -> IntArray:
        """系列別遅延のコピーをshape ``[n_series]``、単位sampleで返す。"""
        return self._delays.copy()

    def reset(self) -> None:
        """入力系列の切替時に履歴を破棄し、全遅延区間を未完成へ戻す。"""
        if self._history is not None:
            self._history.fill(0.0)
        self._history_valid.fill(False)

    def process(self, data: NumericArray) -> SignalBlock:
        """1ブロックへ系列別整数遅延を適用する。

        Args:
            data: 入力。shape ``[n_series, n_sample]``。

        Returns:
            遅延後の値と完成mask。いずれも入力と同じshape。

        Raises:
            ValueError: 次元または系列数が構築時と一致しない場合。
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


class VersionedCausalFIR:
    """版番号付き係数をブロック先頭で切り替える逐次因果FIRである。

    係数はcomplex64またはcomplex128で、信号は対応する実数または複素数、shapeはそれぞれ
    ``[n_series,n_sample]``、``[n_series,n_tap]``、``[n_series,n_sample]``。
    係数版の完成公開とFIR履歴・valid maskを扱うが、係数設計、整数遅延、系列加算は
    責務に含めない。
    """

    def __init__(self, taps: ComplexArray, *, version: int = 0) -> None:
        """完成済み係数でFIRを構築する。

        Args:
            taps: 系列別因果FIR係数。shape ``[n_series,n_tap]``。
            version: 係数の単調増加版番号。

        Raises:
            ValueError: 係数shapeが不正、tapが空、またはversionが負の場合。
        """
        checked = self._validate_taps(taps)
        if version < 0:
            raise ValueError("version must be non-negative.")
        self._active_taps = checked
        self._active_version = version
        self._pending_taps: ComplexArray | None = None
        self._pending_version: int | None = None
        history_length = checked.shape[1] - 1
        self._history = np.zeros((checked.shape[0], history_length), dtype=checked.dtype)
        self._history_valid = np.zeros((checked.shape[0], history_length), dtype=np.bool_)

    @staticmethod
    def _validate_taps(taps: ComplexArray) -> ComplexArray:
        checked = np.asarray(taps)
        if checked.ndim != 2 or checked.shape[0] == 0 or checked.shape[1] == 0:
            raise ValueError("taps must have shape (n_series, n_tap) with non-zero axes.")
        if checked.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError("taps dtype must be complex64 or complex128.")
        if not bool(np.all(np.isfinite(checked))):
            raise ValueError("taps must contain only finite values.")
        return checked.copy()

    @property
    def active_version(self) -> int:
        """現在外部へ公開済みの係数版番号を返す。"""
        return self._active_version

    def request_update(self, taps: ComplexArray, *, version: int) -> None:
        """次の ``process`` 先頭で一括反映する完成済み係数を予約する。

        Args:
            taps: 新係数。shapeは構築時の係数と同じ。
            version: 現在版と予約済み版の双方より大きい版番号。

        Raises:
            ValueError: shape不一致、非有限値、または版番号が単調増加でない場合。
        """
        checked = self._validate_taps(taps)
        if checked.shape != self._active_taps.shape:
            raise ValueError("updated taps must keep (n_series, n_tap) unchanged.")
        if checked.dtype != self._active_taps.dtype:
            # 信号履歴を保持したまま精度を変えると旧精度と新精度の部分結果が混ざる。
            raise ValueError("updated taps dtype must match active taps dtype.")
        latest_version = self._active_version
        if self._pending_version is not None:
            latest_version = self._pending_version
        if version <= latest_version:
            raise ValueError("version must be greater than active and pending versions.")
        self._pending_taps = checked
        self._pending_version = version

    def reset(self) -> None:
        """係数版は保持したまま信号履歴を破棄し、初回tap区間を未完成へ戻す。"""
        self._history.fill(0.0)
        self._history_valid.fill(False)

    def process(self, block: SignalBlock) -> SignalBlock:
        """1ブロックへ、ブロック全体で同一版の因果FIRを適用する。

        Args:
            block: 入力値とvalid mask。shapeは双方 ``[n_series,n_sample]``。

        Returns:
            FIR出力と、全tap入力が完成したsampleだけ真となるmask。

        Raises:
            ValueError: 入力shapeまたは系列数が係数と一致しない場合。
        """
        data = np.asarray(block.data)
        valid = np.asarray(block.valid_mask, dtype=np.bool_)
        if data.ndim != 2 or data.shape != valid.shape:
            raise ValueError("data and valid_mask must have the same 2-D shape.")
        if data.shape[0] != self._active_taps.shape[0]:
            raise ValueError("block n_series must match taps n_series.")
        expected_real_dtype = (
            np.dtype(np.float32)
            if self._active_taps.dtype == np.dtype(np.complex64)
            else np.dtype(np.float64)
        )
        if data.dtype not in (expected_real_dtype, self._active_taps.dtype):
            raise ValueError("block precision must match taps precision.")
        # NumPyと同様に、実入力と複素FIR係数のresult dtypeは係数側の複素dtypeとなる。
        calculation_data = np.asarray(data, dtype=self._active_taps.dtype)

        # 予約係数は局所変数で計算し、全系列の出力完成後にのみactive版として公開する。
        taps = self._pending_taps if self._pending_taps is not None else self._active_taps
        version = (
            self._pending_version if self._pending_version is not None else self._active_version
        )
        extended = np.concatenate((self._history, calculation_data), axis=1)
        extended_valid = np.concatenate((self._history_valid, valid), axis=1)
        output = np.empty_like(calculation_data)
        output_valid = np.empty(valid.shape, dtype=np.bool_)
        n_sample = data.shape[1]
        n_tap = taps.shape[1]
        for series_index in range(data.shape[0]):
            # y[n]=sum_k h[k]x[n-k]。validは同じtap区間の論理積で完成条件を表す。
            output[series_index] = np.convolve(
                extended[series_index], taps[series_index], mode="valid"
            )[:n_sample]
            valid_count = np.convolve(
                extended_valid[series_index].astype(np.int64),
                np.ones(n_tap, dtype=np.int64),
                mode="valid",
            )[:n_sample]
            output_valid[series_index] = valid_count == n_tap

        history_length = n_tap - 1
        if history_length > 0:
            self._history[...] = extended[:, -history_length:]
            self._history_valid[...] = extended_valid[:, -history_length:]
        self._active_taps = taps
        self._active_version = int(version)
        self._pending_taps = None
        self._pending_version = None
        return SignalBlock(output, output_valid)
