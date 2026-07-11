"""spflow.filterbank.complex_halfband_stage を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComplexFIRHalfbandStageFilters:
    """明示的な 2 分岐複素 FIR halfband 段の係数群。"""

    analysis_low: np.ndarray
    analysis_high: np.ndarray
    synthesis_low: np.ndarray
    synthesis_high: np.ndarray
    analysis_phase: int = 0
    synthesis_phase: int = 0
    delay_compensation: int = 0

    def __post_init__(self) -> None:
        analysis_low = np.asarray(self.analysis_low, dtype=np.complex64)
        analysis_high = np.asarray(self.analysis_high, dtype=np.complex64)
        synthesis_low = np.asarray(self.synthesis_low, dtype=np.complex64)
        synthesis_high = np.asarray(self.synthesis_high, dtype=np.complex64)

        if analysis_low.ndim != 1 or analysis_high.ndim != 1 or synthesis_low.ndim != 1 or synthesis_high.ndim != 1:
            raise ValueError("All filters must be one-dimensional.")
        if analysis_low.size == 0 or analysis_high.size == 0 or synthesis_low.size == 0 or synthesis_high.size == 0:
            raise ValueError("Filters must not be empty.")
        if analysis_low.size != analysis_high.size:
            raise ValueError("Analysis filters must have the same length.")
        if synthesis_low.size != synthesis_high.size:
            raise ValueError("Synthesis filters must have the same length.")
        if self.analysis_phase < 0 or self.analysis_phase >= 2:
            raise ValueError("analysis_phase must be 0 or 1.")
        if self.synthesis_phase < 0 or self.synthesis_phase >= 2:
            raise ValueError("synthesis_phase must be 0 or 1.")
        if self.delay_compensation < 0:
            raise ValueError("delay_compensation must be non-negative.")

        object.__setattr__(self, "analysis_low", analysis_low)
        object.__setattr__(self, "analysis_high", analysis_high)
        object.__setattr__(self, "synthesis_low", synthesis_low)
        object.__setattr__(self, "synthesis_high", synthesis_high)

    @classmethod
    def haar_paraunitary(cls) -> "ComplexFIRHalfbandStageFilters":
        """最小構成の Haar paraunitary 段係数を返す。"""
        analysis_low = np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
        analysis_high = np.array([-1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
        synthesis_low = np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
        synthesis_high = np.array([1.0, -1.0], dtype=np.float32) / np.sqrt(2.0)
        return cls(
            analysis_low=analysis_low,
            analysis_high=analysis_high,
            synthesis_low=synthesis_low,
            synthesis_high=synthesis_high,
            analysis_phase=1,
            synthesis_phase=0,
            delay_compensation=0,
        )


class ComplexFIRHalfbandStage:
    """候補 A 系の複素 FIR halfband 段を検証する試作実装。

    明示 FIR 係数・位相規約・遅延補償を固定し、2 分岐の critically sampled 段が
    完全再構成と streaming/offline 一致を満たすかを検証する。
    packet の最終契約や木全体の周波数メタデータ管理は責務に含めない。
    """

    def __init__(self, filters: ComplexFIRHalfbandStageFilters) -> None:
        self.filters = filters

    def analysis(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """入力複素信号を low/high 2 分岐へ解析する。

        Args:
            x: 入力信号。shape は `[..., n_sample]`。末尾軸が時間軸。

        Returns:
            `(low, high)`。両者の shape は `[..., n_out]` で、
            `analysis_phase` に従って 2 サンプルおきに抽出した子系列を返す。
        """
        arr = np.asarray(x, dtype=np.complex64)
        low_full = self._convolve_last_axis(arr, self.filters.analysis_low)
        high_full = self._convolve_last_axis(arr, self.filters.analysis_high)
        phase = self.filters.analysis_phase
        # full 畳み込み後の系列から phase を起点に 2 サンプルおきへ間引き、
        # critically sampled な low/high 子系列を取り出す。
        return low_full[..., phase::2], high_full[..., phase::2]

    def synthesis(self, low: np.ndarray, high: np.ndarray, *, length: int | None = None) -> np.ndarray:
        """low/high 2 分岐から複素時間列を再合成する。

        Args:
            low: 低域子系列。shape は `[..., n_sub]`。
            high: 高域子系列。shape は `[..., n_sub]`。
            length: 必要な出力長。省略時は full 畳み込み長を返す。

        Returns:
            再合成複素時間列。shape は `[..., n_sample]`。
        """
        low_arr = np.asarray(low, dtype=np.complex64)
        high_arr = np.asarray(high, dtype=np.complex64)
        if low_arr.shape != high_arr.shape:
            raise ValueError("low and high branches must have identical shapes.")

        # upsample 後の shape はいずれも [..., 2 * n_sub]。
        # synthesis_phase に従って偶数側または奇数側へ零挿入する。
        up_low = self._upsample_by_two(low_arr, phase=self.filters.synthesis_phase)
        up_high = self._upsample_by_two(high_arr, phase=self.filters.synthesis_phase)
        recon = (
            self._convolve_last_axis(up_low, self.filters.synthesis_low)
            + self._convolve_last_axis(up_high, self.filters.synthesis_high)
        )
        if self.filters.delay_compensation > 0:
            recon = recon[..., self.filters.delay_compensation :]
        if length is not None:
            recon = recon[..., :length]
        return recon

    def stable_analysis_length(self, input_length: int) -> int:
        """flush 前に確定できる解析出力長を返す。"""
        phase = self.filters.analysis_phase
        if input_length <= phase:
            return 0
        return ((input_length - 1 - phase) // 2) + 1

    def full_analysis_length(self, input_length: int) -> int:
        """末尾ゼロ詰めまで含めた full 解析出力長を返す。"""
        full_length = input_length + self.filters.analysis_low.size - 1
        phase = self.filters.analysis_phase
        if full_length <= phase:
            return 0
        return ((full_length - 1 - phase) // 2) + 1

    def stable_synthesis_length(self, subband_length: int) -> int:
        """flush 前に確定できる再合成出力長を返す。"""
        stable = 2 * subband_length + self.filters.synthesis_phase - self.filters.delay_compensation
        return max(0, stable)

    def full_synthesis_length(self, subband_length: int) -> int:
        """末尾過渡まで含めた full 再合成出力長を返す。"""
        full = (
            2 * subband_length
            + self.filters.synthesis_phase
            + self.filters.synthesis_low.size
            - 1
            - self.filters.delay_compensation
        )
        return max(0, full)

    @staticmethod
    def _upsample_by_two(x: np.ndarray, *, phase: int) -> np.ndarray:
        up = np.zeros(x.shape[:-1] + (2 * x.shape[-1],), dtype=np.complex64)
        up[..., phase::2] = x
        return up

    @staticmethod
    def _convolve_last_axis(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.complex64)
        filt = np.asarray(taps, dtype=np.complex64)
        if arr.ndim == 0:
            raise ValueError("input must have at least one dimension.")

        rows = int(np.prod(arr.shape[:-1])) if arr.ndim > 1 else 1
        # reshape 後の shape は [n_row, n_sample]。
        # 末尾時間軸への FIR 畳み込みを全 prefix 行へ同じ規約で適用する。
        reshaped = arr.reshape(rows, arr.shape[-1])
        out = np.zeros((rows, arr.shape[-1] + filt.size - 1), dtype=np.complex64)
        for row_idx in range(rows):
            out[row_idx] = np.convolve(reshaped[row_idx], filt, mode="full")
        return out.reshape(arr.shape[:-1] + (out.shape[-1],))


class ComplexFIRHalfbandStageStreamingAnalyzer:
    """出力サンプルごとに O(L) で動く逐次解析器。"""

    def __init__(self, stage: ComplexFIRHalfbandStage) -> None:
        self.stage = stage
        self._prefix_shape: tuple[int, ...] | None = None
        self._row_count: int | None = None
        tap_count = self.stage.filters.analysis_low.size
        self._history = np.zeros((0, tap_count), dtype=np.complex64)
        self._history_write_pos = 0
        self._analysis_low_reversed = self.stage.filters.analysis_low[::-1]
        self._analysis_high_reversed = self.stage.filters.analysis_high[::-1]
        self._history_orders = tuple(
            np.concatenate(
                [
                    np.arange(offset, tap_count, dtype=np.int64),
                    np.arange(0, offset, dtype=np.int64),
                ]
            )
            for offset in range(tap_count)
        )
        self._sample_count = 0
        self._next_output_full_index = self.stage.filters.analysis_phase
        self._flushed = False

    def process(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """入力チャンクから新たに確定した low/high 子系列だけを返す。"""
        arr = np.asarray(x, dtype=np.complex64)
        if arr.ndim == 0:
            raise ValueError("input chunk must have at least one dimension.")
        if self._flushed:
            raise RuntimeError("Cannot process after flush().")
        if arr.shape[-1] == 0:
            return self._empty_pair(arr.shape[:-1])

        flat = self._reshape_rows(arr)
        low_frames: list[np.ndarray] = []
        high_frames: list[np.ndarray] = []
        for sample_idx in range(flat.shape[1]):
            self._history[:, self._history_write_pos] = flat[:, sample_idx]
            self._history_write_pos = (self._history_write_pos + 1) % self._history.shape[1]
            current_full_index = self._sample_count
            self._sample_count += 1
            if current_full_index != self._next_output_full_index:
                continue

            # ordered_history shape: [n_row, n_tap]。
            # 巡回バッファの書き込み位置に応じて時間順へ並べ替え、
            # reversed taps との内積で FIR 出力 1 サンプルを得る。
            ordered_history = self._history[:, self._history_orders[self._history_write_pos]]
            low_frames.append(ordered_history @ self._analysis_low_reversed)
            high_frames.append(ordered_history @ self._analysis_high_reversed)
            self._next_output_full_index += 2

        return self._stack_outputs(low_frames), self._stack_outputs(high_frames)

    def flush(self) -> tuple[np.ndarray, np.ndarray]:
        """末尾ゼロ詰めにより解析 FIR の残留過渡を回収する。"""
        if self._flushed:
            prefix_shape = () if self._prefix_shape is None else self._prefix_shape
            return self._empty_pair(prefix_shape)
        if self._prefix_shape is None:
            return self._empty_pair(())

        low_frames: list[np.ndarray] = []
        high_frames: list[np.ndarray] = []
        target_sample_count = self._sample_count + self.stage.filters.analysis_low.size - 1
        zero = np.zeros((self._row_count or 0,), dtype=np.complex64)
        while self._sample_count < target_sample_count:
            self._history[:, self._history_write_pos] = zero
            self._history_write_pos = (self._history_write_pos + 1) % self._history.shape[1]
            current_full_index = self._sample_count
            self._sample_count += 1
            if current_full_index != self._next_output_full_index:
                continue

            ordered_history = self._history[:, self._history_orders[self._history_write_pos]]
            low_frames.append(ordered_history @ self._analysis_low_reversed)
            high_frames.append(ordered_history @ self._analysis_high_reversed)
            self._next_output_full_index += 2
        self._flushed = True
        return self._stack_outputs(low_frames), self._stack_outputs(high_frames)

    def _reshape_rows(self, arr: np.ndarray) -> np.ndarray:
        prefix_shape = arr.shape[:-1]
        if self._prefix_shape is None:
            self._prefix_shape = prefix_shape
            self._row_count = int(np.prod(prefix_shape)) if prefix_shape else 1
            self._history = np.zeros((self._row_count, self.stage.filters.analysis_low.size), dtype=np.complex64)
        elif prefix_shape != self._prefix_shape:
            raise ValueError("input chunk shape mismatch except along time axis.")
        assert self._row_count is not None
        return arr.reshape(self._row_count, arr.shape[-1])

    def _stack_outputs(self, frames: list[np.ndarray]) -> np.ndarray:
        prefix_shape = () if self._prefix_shape is None else self._prefix_shape
        if not frames:
            return np.zeros(prefix_shape + (0,), dtype=np.complex64)
        stacked = np.stack(frames, axis=-1)
        return stacked.reshape(prefix_shape + (stacked.shape[-1],))

    @staticmethod
    def _empty_pair(prefix_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        empty = np.zeros(prefix_shape + (0,), dtype=np.complex64)
        return empty, empty.copy()


class ComplexFIRHalfbandStageStreamingSynthesizer:
    """疎な overlap-add 補間で動く逐次再合成器。"""

    def __init__(self, stage: ComplexFIRHalfbandStage) -> None:
        self.stage = stage
        self._prefix_shape: tuple[int, ...] | None = None
        self._row_count: int | None = None
        self._pending = np.zeros((0, 0), dtype=np.complex64)
        self._next_uncropped_index = 0
        self._subband_count = 0
        self._flushed = False

    def process(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """low/high 子チャンクから新たに確定した親系列だけを返す。"""
        low_arr = np.asarray(low, dtype=np.complex64)
        high_arr = np.asarray(high, dtype=np.complex64)
        if low_arr.shape != high_arr.shape:
            raise ValueError("low and high chunks must have identical shapes.")
        if self._flushed:
            raise RuntimeError("Cannot process after flush().")
        if low_arr.shape[-1] == 0:
            return np.zeros(low_arr.shape[:-1] + (0,), dtype=np.complex64)

        low_flat = self._reshape_rows(low_arr)
        high_flat = self._reshape_rows(high_arr)
        taps_len = self.stage.filters.synthesis_low.size
        for sample_idx in range(low_flat.shape[1]):
            inserted_index = 2 * (self._subband_count + sample_idx) + self.stage.filters.synthesis_phase
            relative_index = inserted_index - self._next_uncropped_index
            self._ensure_pending_length(relative_index + taps_len)
            # pending shape: [n_row, n_pending_sample]。
            # 1 個の子サンプルが親系列上で寄与する FIR 区間だけへ加算する。
            self._pending[:, relative_index : relative_index + taps_len] += (
                low_flat[:, sample_idx][:, np.newaxis] * self.stage.filters.synthesis_low[np.newaxis, :]
            )
            self._pending[:, relative_index : relative_index + taps_len] += (
                high_flat[:, sample_idx][:, np.newaxis] * self.stage.filters.synthesis_high[np.newaxis, :]
            )

        self._subband_count += low_flat.shape[1]
        stable_end = 2 * self._subband_count + self.stage.filters.synthesis_phase
        return self._drain_until(stable_end)

    def flush(self) -> np.ndarray:
        """再合成 FIR の残留過渡をすべて排出する。"""
        if self._flushed:
            prefix_shape = () if self._prefix_shape is None else self._prefix_shape
            return np.zeros(prefix_shape + (0,), dtype=np.complex64)
        if self._prefix_shape is None:
            return np.zeros((0,), dtype=np.complex64)
        full_end = 2 * self._subband_count + self.stage.filters.synthesis_phase + self.stage.filters.synthesis_low.size - 1
        self._flushed = True
        return self._drain_until(full_end)

    def _reshape_rows(self, arr: np.ndarray) -> np.ndarray:
        prefix_shape = arr.shape[:-1]
        if self._prefix_shape is None:
            self._prefix_shape = prefix_shape
            self._row_count = int(np.prod(prefix_shape)) if prefix_shape else 1
            self._pending = np.zeros((self._row_count, 0), dtype=np.complex64)
        elif prefix_shape != self._prefix_shape:
            raise ValueError("subband chunk shape mismatch except along time axis.")
        assert self._row_count is not None
        return arr.reshape(self._row_count, arr.shape[-1])

    def _ensure_pending_length(self, required_length: int) -> None:
        if required_length <= self._pending.shape[1]:
            return
        grow = required_length - self._pending.shape[1]
        self._pending = np.concatenate(
            [self._pending, np.zeros((self._pending.shape[0], grow), dtype=np.complex64)],
            axis=1,
        )

    def _drain_until(self, uncropped_end_exclusive: int) -> np.ndarray:
        ready = uncropped_end_exclusive - self._next_uncropped_index
        prefix_shape = () if self._prefix_shape is None else self._prefix_shape
        if ready <= 0:
            return np.zeros(prefix_shape + (0,), dtype=np.complex64)
        self._ensure_pending_length(ready)

        discard_before = self.stage.filters.delay_compensation
        discard = max(0, min(uncropped_end_exclusive, discard_before) - self._next_uncropped_index)
        emitted = self._pending[:, discard:ready].copy()
        if ready >= self._pending.shape[1]:
            self._pending = np.zeros((self._pending.shape[0], 0), dtype=np.complex64)
        else:
            self._pending = self._pending[:, ready:]
        self._next_uncropped_index = uncropped_end_exclusive
        return emitted.reshape(prefix_shape + (emitted.shape[-1],))


class OracleComplexFIRHalfbandStageStreamingAnalyzer:
    """oracle 比較用の逐次解析参照実装。"""

    def __init__(self, stage: ComplexFIRHalfbandStage) -> None:
        self.stage = stage
        self._input: np.ndarray | None = None
        self._emitted = 0

    def process(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """オフライン解析を再計算し、新規確定分だけを返す。"""
        arr = np.asarray(x, dtype=np.complex64)
        if arr.ndim == 0:
            raise ValueError("input chunk must have at least one dimension.")
        if arr.shape[-1] == 0:
            return self._empty_pair(arr.shape[:-1])

        if self._input is None:
            self._input = arr.copy()
        else:
            self._input = np.concatenate([self._input, arr], axis=-1)

        low_all, high_all = self.stage.analysis(self._input)
        stable_len = self.stage.stable_analysis_length(self._input.shape[-1])
        new_low = low_all[..., self._emitted:stable_len]
        new_high = high_all[..., self._emitted:stable_len]
        self._emitted = stable_len
        return new_low, new_high

    def flush(self) -> tuple[np.ndarray, np.ndarray]:
        """オフライン解析の full 長まで末尾過渡を回収する。"""
        if self._input is None:
            return self._empty_pair(())
        low_all, high_all = self.stage.analysis(self._input)
        full_len = self.stage.full_analysis_length(self._input.shape[-1])
        tail_low = low_all[..., self._emitted:full_len]
        tail_high = high_all[..., self._emitted:full_len]
        self._emitted = full_len
        return tail_low, tail_high

    @staticmethod
    def _empty_pair(prefix_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        empty = np.zeros(prefix_shape + (0,), dtype=np.complex64)
        return empty, empty.copy()


class OracleComplexFIRHalfbandStageStreamingSynthesizer:
    """oracle 比較用の逐次再合成参照実装。"""

    def __init__(self, stage: ComplexFIRHalfbandStage) -> None:
        self.stage = stage
        self._low: np.ndarray | None = None
        self._high: np.ndarray | None = None
        self._emitted = 0

    def process(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """オフライン再合成を再計算し、新規確定分だけを返す。"""
        low_arr = np.asarray(low, dtype=np.complex64)
        high_arr = np.asarray(high, dtype=np.complex64)
        if low_arr.shape != high_arr.shape:
            raise ValueError("low and high chunks must have identical shapes.")
        if low_arr.shape[-1] == 0:
            return np.zeros(low_arr.shape[:-1] + (0,), dtype=np.complex64)

        if self._low is None:
            self._low = low_arr.copy()
            self._high = high_arr.copy()
        else:
            previous_low = self._low
            previous_high = self._high
            if previous_high is None:
                # low/high は常に対で更新するため、片側だけ欠ける状態は内部状態破損である。
                raise RuntimeError("streaming synthesis low/high state is inconsistent.")
            self._low = np.concatenate([previous_low, low_arr], axis=-1)
            self._high = np.concatenate([previous_high, high_arr], axis=-1)

        assert self._high is not None
        recon = self.stage.synthesis(self._low, self._high)
        stable_len = self.stage.stable_synthesis_length(self._low.shape[-1])
        new = recon[..., self._emitted:stable_len]
        self._emitted = stable_len
        return new

    def flush(self) -> np.ndarray:
        """オフライン再合成の full 長まで末尾過渡を回収する。"""
        if self._low is None or self._high is None:
            return np.zeros((0,), dtype=np.complex64)
        recon = self.stage.synthesis(self._low, self._high)
        full_len = self.stage.full_synthesis_length(self._low.shape[-1])
        tail = recon[..., self._emitted:full_len]
        self._emitted = full_len
        return tail
