"""正式な整数遅延と残差 FIR の実時間評価を支援する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow.beamforming import CausalBlockFIR


@dataclass(frozen=True)
class RuntimeStageResult:
    """評価用の逐次処理結果を保持する。

    `data` と `valid_mask` はともに shape `[n_series, n_sample]` で、axis=0 は
    独立な beam/channel 系列、axis=1 は入力 block と同じ global sample を表す。
    この結果型は信号処理や複数系列の合成を責務に含めない。
    """

    data: NDArray[np.complex128]
    valid_mask: NDArray[np.bool_]


class StatefulIntegerDelayStage:
    """系列別の非負整数遅延を block 境界を跨いで適用する。

    入力と出力は shape `[n_series, n_sample]` であり、系列 `s` について
    `y[s, n] = x[s, n - delays_sample[s]]` を実現する。到来遅延の計算や
    小数遅延 FIR、系列間の加算は責務に含めない。正式 T2a 評価では、有限長
    残差 FIR より前に置く実整数 delay buffer に対応する。
    """

    def __init__(self, delays_sample: NDArray[Any]) -> None:
        """整数遅延段を構成する。

        Args:
            delays_sample: 系列別遅延。shape `[n_series]`、単位 sample。

        Raises:
            ValueError: 1次元でない、空、または負の遅延を含む場合。

        境界条件:
            初回入力より前の sample は存在しないためゼロを返し、その区間の
            `valid_mask` を False にする。最大遅延が block 長を超えても履歴を保持する。
        """
        delays = np.asarray(delays_sample, dtype=np.int64)
        if delays.ndim != 1 or delays.size == 0:
            raise ValueError("delays_sample must have shape (n_series,) and must not be empty.")
        if not bool(np.all(delays >= 0)):
            raise ValueError("delays_sample must contain only non-negative values.")
        self.delays_sample = delays
        self.n_series = int(delays.size)
        self.max_delay_sample = int(np.max(delays))
        self._history = np.zeros((self.n_series, self.max_delay_sample), dtype=np.complex128)
        self._valid_history = np.zeros(
            (self.n_series, self.max_delay_sample), dtype=np.bool_
        )

    def reset(self) -> None:
        """入力系列切替時に波形履歴と有効性履歴をゼロへ戻す。"""
        self._history.fill(0.0)
        self._valid_history.fill(False)

    def process(
        self,
        x_block: NDArray[Any],
        valid_mask: NDArray[Any] | None = None,
    ) -> RuntimeStageResult:
        """1 block に系列別整数遅延を適用する。

        Args:
            x_block: 入力。shape `[n_series, n_sample]`。
            valid_mask: 入力有効性。shape `[n_series, n_sample]`。None は全入力有効。

        Returns:
            遅延後波形と有効性。両方 shape `[n_series, n_sample]`。

        Raises:
            ValueError: 入力または有効性の shape が契約と異なる場合。
        """
        signal = np.asarray(x_block, dtype=np.complex128)
        if signal.ndim != 2 or signal.shape[0] != self.n_series:
            raise ValueError("x_block must have shape (n_series, n_sample).")
        if valid_mask is None:
            input_valid = np.ones(signal.shape, dtype=np.bool_)
        else:
            input_valid = np.asarray(valid_mask, dtype=np.bool_)
            if input_valid.shape != signal.shape:
                raise ValueError("valid_mask must have the same shape as x_block.")

        # extended の axis=1 は global 時間順で、先頭に直前 block の最大遅延分を置く。
        # delay d の出力窓は [max_delay-d : max_delay-d+n_sample] となり、x[n-d] に一致する。
        extended = np.concatenate((self._history, signal), axis=1)
        extended_valid = np.concatenate((self._valid_history, input_valid), axis=1)
        n_sample = int(signal.shape[1])
        output = np.empty_like(signal)
        output_valid = np.empty_like(input_valid)
        for series_index, delay_sample_value in enumerate(self.delays_sample.tolist()):
            delay_sample = int(delay_sample_value)
            start = self.max_delay_sample - delay_sample
            stop = start + n_sample
            output[series_index] = extended[series_index, start:stop]
            output_valid[series_index] = extended_valid[series_index, start:stop]

        if self.max_delay_sample > 0:
            self._history = extended[:, -self.max_delay_sample :].copy()
            self._valid_history = extended_valid[:, -self.max_delay_sample :].copy()
        return RuntimeStageResult(data=output, valid_mask=output_valid)


class ResidualCausalFIRStage:
    """完成周波数重みから得た系列別残差 FIR を逐次適用する。

    入力、出力、valid mask は shape `[n_series, n_sample]` である。FIR は
    `y[s,n] = sum_p taps[s,p] x[s,n-p]` の因果規約であり、通常は完成重み `w` の
    実適用応答 `IFFT(conj(w))` を有限 tap に切り出した係数を渡す。重み設計、
    整数遅延、系列和は責務に含めない。
    """

    def __init__(self, taps: NDArray[Any], *, active_version: int = 0) -> None:
        """残差 FIR 段を構成する。

        Args:
            taps: 系列別係数。shape `[n_series, n_tap]`、tap axis の単位は sample。
            active_version: 初期係数の識別番号。単位なし。

        Raises:
            ValueError: shape が不正、空、または非有限値を含む場合。
        """
        coefficient = np.asarray(taps, dtype=np.complex128)
        if coefficient.ndim != 2 or coefficient.shape[0] == 0 or coefficient.shape[1] == 0:
            raise ValueError("taps must have shape (n_series, n_tap) and must not be empty.")
        if not bool(np.all(np.isfinite(coefficient))):
            raise ValueError("taps must contain only finite values.")
        self.taps = coefficient
        self.n_series = int(coefficient.shape[0])
        self.n_tap = int(coefficient.shape[1])
        self.active_version = int(active_version)
        self._pending_taps: NDArray[np.complex128] | None = None
        self._pending_version: int | None = None
        # 既存 CausalBlockFIR は complex64 運用契約である。評価では戻り値をcomplex128へ
        # 揃えつつ、同じblock履歴規約を使い実装経路との差を直接検出できるようにする。
        self._fir = CausalBlockFIR(n_series=self.n_series, tap_length=self.n_tap)
        self._valid_history = np.zeros((self.n_series, self.n_tap - 1), dtype=np.bool_)

    def request_coefficient_update(self, taps: NDArray[Any], *, version: int) -> None:
        """次回 block 先頭で反映する係数を予約する。

        Args:
            taps: 新しい系列別係数。shape `[n_series, n_tap]`。
            version: 新係数の識別番号。単位なし。

        Raises:
            ValueError: shape が現在の FIR と異なる、非有限値を含む、または
                `version` が現在値以下の場合。

        境界条件:
            呼出時には active 係数を変更しない。次の `process` の先頭で全系列を
            一括更新するため、一つの block 内に旧係数と新係数が混在しない。
            入力履歴は係数更新後も保持し、更新境界で波形 sample を欠落させない。
        """
        coefficient = np.asarray(taps, dtype=np.complex128)
        if coefficient.shape != self.taps.shape:
            raise ValueError("updated taps must have the same shape as active taps.")
        if not bool(np.all(np.isfinite(coefficient))):
            raise ValueError("updated taps must contain only finite values.")
        if int(version) <= self.active_version:
            raise ValueError("version must be greater than active_version.")
        self._pending_taps = coefficient.copy()
        self._pending_version = int(version)

    def reset(self) -> None:
        """入力系列切替時に FIR と有効性の履歴を破棄する。"""
        self._fir.reset()
        self._valid_history.fill(False)

    def process(
        self,
        x_block: NDArray[Any],
        valid_mask: NDArray[Any] | None = None,
    ) -> RuntimeStageResult:
        """1 block に残差 FIR を適用する。

        Args:
            x_block: 入力。shape `[n_series, n_sample]`。
            valid_mask: 入力有効性。shape `[n_series, n_sample]`。None は全入力有効。

        Returns:
            FIR 出力と、全 tap 入力が有効な時刻だけ True の mask。

        Raises:
            ValueError: 入力または有効性の shape が契約と異なる場合。
        """
        signal = np.asarray(x_block, dtype=np.complex128)
        if signal.ndim != 2 or signal.shape[0] != self.n_series:
            raise ValueError("x_block must have shape (n_series, n_sample).")
        if valid_mask is None:
            input_valid = np.ones(signal.shape, dtype=np.bool_)
        else:
            input_valid = np.asarray(valid_mask, dtype=np.bool_)
            if input_valid.shape != signal.shape:
                raise ValueError("valid_mask must have the same shape as x_block.")

        if self._pending_taps is not None:
            # block処理へ入る前に全系列の係数を一括で切り替える。FIR履歴は入力履歴なので
            # 破棄せず、新係数が直前blockの実在sampleにも正しく作用するよう保持する。
            pending_version = self._pending_version
            if pending_version is None:
                raise RuntimeError("pending coefficient version is missing.")
            self.taps = self._pending_taps
            self.active_version = pending_version
            self._pending_taps = None
            self._pending_version = None

        output = np.asarray(self._fir.process(signal, self.taps), dtype=np.complex128)
        extended_valid = np.concatenate((self._valid_history, input_valid), axis=1)
        output_valid = np.empty_like(input_valid)
        for sample_index in range(signal.shape[1]):
            # 因果 FIR の全 tap が実在入力を参照する場合だけ、その出力を評価対象に含める。
            output_valid[:, sample_index] = np.all(
                extended_valid[:, sample_index : sample_index + self.n_tap], axis=1
            )
        if self.n_tap > 1:
            self._valid_history = extended_valid[:, -(self.n_tap - 1) :].copy()
        return RuntimeStageResult(data=output, valid_mask=output_valid)
