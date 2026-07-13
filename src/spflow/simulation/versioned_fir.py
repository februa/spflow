"""完成済み係数をblock先頭で切り替える版管理付き逐次因果FIR。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow.simulation.signal_block import SignalBlock

ComplexArray = NDArray[np.complexfloating[Any, Any]]


class VersionedCausalFIR:
    """版番号付き係数をブロック先頭で切り替える逐次因果FIRである。

    係数はcomplex64またはcomplex128で、信号は対応する実数または複素数、shapeはそれぞれ
    ``[n_series,n_tap]``、``[n_series,n_sample]``。係数版の完成公開、FIR履歴、
    valid maskを扱うが、係数設計、整数遅延、系列加算は責務に含めない。
    """

    def __init__(self, taps: ComplexArray, *, version: int = 0) -> None:
        """完成済み係数でFIRを構築する。

        Args:
            taps: 系列別因果FIR係数。shape ``[n_series,n_tap]``。
            version: 係数の単調増加版番号。

        Raises:
            ValueError: 係数shape、dtype、有限性、またはversionが不正な場合。

        Notes:
            初回tap履歴は存在しないため、対応する出力のvalid maskをFalseにする。
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
            taps: 新係数。shapeとdtypeは構築時の係数と同じ。
            version: 現在版と予約済み版の双方より大きい版番号。

        Raises:
            ValueError: shape、dtype、有限性、または版番号が契約と異なる場合。
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
            ValueError: 入力shape、系列数、または精度が係数と一致しない場合。
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
