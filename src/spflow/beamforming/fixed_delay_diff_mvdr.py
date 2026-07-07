"""固定遅延+差分補正 MVDR の設計・適用部品を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import (
    require,
    require_non_negative_float,
    require_positive_float,
    require_positive_int,
)
from .time_delay import DelayTable, FractionalDelayFilterBank, design_fractional_delay_filter_bank

STANDARD_FRACTIONAL_DELAY_PATTERN_COUNT = 51
STANDARD_FRACTIONAL_DELAY_TAP_COUNT = 128


@dataclass(frozen=True)
class ShortFFTCovarianceUpdateResult:
    """短 FFT 共分散更新の結果を保持する。

    このクラスは、短 FFT 統計ルートで更新した周波数 bin 別共分散、更新可能フラグ、
    処理済み block 数をまとめて返すための結果型である。

    入力は `ShortFFTCovarianceAccumulator.process` が処理した多チャネル時間信号であり、
    出力は MVDR 重み設計へ渡す共分散 `covariance` と更新タイミング情報である。

    共分散の評価、MVDR 重み設計、時間領域 FIR 適用は責務に含めない。
    信号処理上は、固定遅延+差分補正 MVDR の統計ルート出力に位置づく。
    """

    covariance: NDArray[np.complex128]
    update_ready: bool
    processed_block_count: int
    total_block_count: int


@dataclass(frozen=True)
class LoadedMVDRDesignResult:
    """対角ローディング付き MVDR 重み設計の結果を保持する。

    このクラスは、周波数 bin ごとの MVDR 重み、loaded covariance の条件数、
    fallback が発生した bin を記録する。

    入力は共分散、ステアリングベクトル、fallback 用固定整相重みであり、
    出力は差分補正重み設計へ渡す `weights` である。

    共分散推定、差分 FIR 化、時間領域畳み込みは責務に含めない。
    信号処理上は、周波数 bin ごとの最小分散制約付き重み設計に位置づく。
    """

    weights: NDArray[np.complex128]
    loaded_condition_number: NDArray[np.float64]
    fallback_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class DifferenceCorrectionDiagnostics:
    """差分補正 FIR 化後の周波数応答診断を保持する。

    このクラスは、固定整相、MVDR、128 tap FIR 化後の最終重みについて、
    target 応答、差分補正枝の blocking 応答、周波数応答再構成誤差を記録する。

    入力は `DifferenceCorrectionFIRDesigner` が設計した周波数重みと FIR 係数であり、
    出力は方式検討書や単体試験で確認する診断配列である。

    BL/FRAZ/BTR 描画、時間波形評価、採否判定は責務に含めない。
    信号処理上は、固定整相主経路と差分補正枝の数式対応を確認する検査点に位置づく。
    """

    target_response_w0: NDArray[np.complex128]
    target_response_mvdr: NDArray[np.complex128]
    target_response_final: NDArray[np.complex128]
    q_blocking_response: NDArray[np.complex128]
    q_reconstruction_error: NDArray[np.complex128]


@dataclass(frozen=True)
class DifferenceCorrectionDesignResult:
    """差分補正重みと時間領域 FIR 係数を保持する。

    このクラスは、数式上の差分重み `q_weight_freq`、実際に信号へ掛ける
    `q_apply_taps`、および FIR 化後に再構成した最終重みをまとめる。

    入力は固定整相重み `w0` と MVDR 重みであり、出力は時間領域補正 FIR と
    診断量である。

    MVDR 重み計算、固定整相主経路の時間領域実装、係数更新スケジューリングは
    責務に含めない。信号処理上は、`y = y0 - z` の補正枝係数設計に位置づく。
    """

    q_weight_freq: NDArray[np.complex128]
    q_apply_freq: NDArray[np.complex128]
    q_apply_taps: NDArray[np.complex128]
    reconstructed_q_weight_freq: NDArray[np.complex128]
    final_weight_freq: NDArray[np.complex128]
    diagnostics: DifferenceCorrectionDiagnostics


def design_distortionless_fixed_weights(
    steering_vector: NDArray[Any],
    *,
    denominator_floor: float = 1.0e-12,
) -> NDArray[np.complex128]:
    """ステアリングベクトルから歪みなし固定整相重みを設計する。

    Args:
        steering_vector: 目標方向ステアリング。shape は `[n_bin, n_ch]`。
            axis=0 は統計 FFT の周波数 bin、axis=1 はセンサチャネルである。
        denominator_floor: `a^H a` の最小許容値。無次元の power 比である。

    Returns:
        固定整相重み `w0 = a / (a^H a)`。shape は `[n_bin, n_ch]`。
        各 bin で `w0[k]^H a[k] = 1` を満たす。

    Raises:
        ValueError: `steering_vector` が 2 次元でない、チャネルが空、または
            `a^H a` が `denominator_floor` 以下の bin を含む場合。

    境界条件:
        ステアリングがゼロベクトルに近い bin では歪みなし条件の分母が消える。
        その状態で正規化すると重みが発散するため、明示的に失敗させる。
    """
    steering = np.asarray(steering_vector, dtype=np.complex128)
    require(steering.ndim == 2, "steering_vector must have shape (n_bin, n_ch).")
    require(steering.shape[1] > 0, "steering_vector must contain at least one channel.")
    require_positive_float("denominator_floor", float(denominator_floor))

    # denom[k] = a[k]^H a[k] は target 方向のステアリング power である。
    # ここで正規化することで、固定整相重みは各周波数 bin で `w0^H a = 1` を満たす。
    denominator = np.sum(steering.conj() * steering, axis=1)
    require(
        bool(np.all(np.abs(denominator) > float(denominator_floor))),
        "steering_vector contains a bin with too small a^H a.",
    )
    return np.asarray(steering / denominator[:, np.newaxis], dtype=np.complex128)


def design_standard_fractional_delay_filter_bank() -> FractionalDelayFilterBank:
    """固定遅延+差分補正 MVDR で使う標準小数遅延 FIR バンクを事前設計する。

    Returns:
        小数遅延 FIR バンク。`frac_grid` shape は `[51]`、範囲は
        `[-0.5, 0.5]` sample、`frac_filters` shape は `[51, 128]` である。

    境界条件:
        本方式では実行時に小数遅延 FIR を設計しない。51 本の事前計算フィルタから
        各チャネル・各整相方位の小数遅延に最も近いものを選ぶことで、
        係数生成コストと実行時の係数揺れを固定する。
    """
    return design_fractional_delay_filter_bank(
        n_frac_filter=STANDARD_FRACTIONAL_DELAY_PATTERN_COUNT,
        n_tap=STANDARD_FRACTIONAL_DELAY_TAP_COUNT,
    )


def design_fixed_delay_fractional_weights_from_delay_table(
    delay_table: DelayTable,
    fractional_filter_bank: FractionalDelayFilterBank,
    frequencies_hz: NDArray[Any],
    *,
    fs_hz: float,
    average_channels: bool = True,
) -> NDArray[np.complex128]:
    """整数遅延+小数遅延 FIR 主経路の実応答から固定整相重みを作る。

    Args:
        delay_table: 固定整相用の遅延表。`delay_int` と `frac_filter_index` の shape は
            `[n_ch, n_beam]`。`delay_int` の単位は sample である。
        fractional_filter_bank: 事前計算済み小数遅延 FIR バンク。
            `frac_filters` shape は `[n_frac_filter, n_tap]`。
        frequencies_hz: 評価周波数。shape は `[n_bin]`、単位は Hz。
        fs_hz: サンプリング周波数。単位は Hz。
        average_channels: `True` の場合は delay-and-sum のチャネル平均に合わせて
            `1/n_ch` を掛ける。`False` の場合はチャネル和の重みにする。

    Returns:
        固定整相重み。shape は `[n_bin, n_beam, n_ch]`。
        axis=0 は周波数 bin、axis=1 は整相方位、axis=2 はセンサチャネルである。
        `y[beam,k] = w0[k,beam]^H X[k]` の規約で使う。

    Raises:
        ValueError: `delay_table` が小数遅延フィルタ選択番号を持たない場合、
            周波数軸 shape が不正な場合、または物理パラメータが不正な場合。

    境界条件:
        差分補正 MVDR の `w0` は、理想 steering ではなく実際の時間領域主経路と
        同じ周波数応答にする必要がある。ここで整数遅延、選択済み小数 FIR、
        チャネル平均の係数をすべて含める。
    """
    if delay_table.frac_filter_index is None:
        raise ValueError("delay_table must include frac_filter_index.")
    require_positive_float("fs_hz", float(fs_hz))

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    require(frequencies.ndim == 1, "frequencies_hz must have shape (n_bin,).")
    require(
        bool(np.all(np.isfinite(frequencies))),
        "frequencies_hz must contain only finite values.",
    )

    delay_int = np.asarray(delay_table.delay_int, dtype=np.int64)
    frac_filter_index = np.asarray(delay_table.frac_filter_index, dtype=np.int64)
    require(
        delay_int.shape == frac_filter_index.shape,
        "delay_int and frac_filter_index must agree on shape.",
    )
    require(delay_int.ndim == 2, "delay_int must have shape (n_ch, n_beam).")

    n_ch = int(delay_int.shape[0])
    scale = 1.0 / float(n_ch) if bool(average_channels) else 1.0
    tap_index = np.arange(fractional_filter_bank.n_tap, dtype=np.float64)
    weights = np.empty((frequencies.size, delay_table.n_beam, n_ch), dtype=np.complex128)

    for bin_index, frequency_hz in enumerate(frequencies.tolist()):
        angular_frequency_rad = 2.0 * np.pi * float(frequency_hz) / float(fs_hz)
        # integer_response[ch, beam] = exp(-j omega delay_int[ch, beam])。
        # 時間領域の整数サンプル遅延 x[n-d] が与える位相回転に対応する。
        integer_response = np.exp(-1j * angular_frequency_rad * delay_int.astype(np.float64))
        fractional_response = _selected_fractional_filter_response(
            fractional_filter_bank=fractional_filter_bank,
            frac_filter_index=frac_filter_index,
            angular_frequency_rad=angular_frequency_rad,
            tap_index=tap_index,
        )
        # apply_response[ch, beam] は時間領域主経路が入力 X[ch,k] に掛ける応答である。
        # ビームフォーマ規約は w^H X なので、返す重みは conj(apply_response) 側にする。
        apply_response = scale * integer_response * fractional_response
        weights[bin_index] = np.asarray(apply_response.conj().T, dtype=np.complex128)
    return weights


def _selected_fractional_filter_response(
    *,
    fractional_filter_bank: FractionalDelayFilterBank,
    frac_filter_index: NDArray[np.int64],
    angular_frequency_rad: float,
    tap_index: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """選択済み小数遅延 FIR の周波数応答を `[n_ch, n_beam]` で返す。"""
    filter_taps = np.asarray(fractional_filter_bank.frac_filters, dtype=np.float64)
    require(
        bool(np.all((0 <= frac_filter_index) & (frac_filter_index < filter_taps.shape[0]))),
        "frac_filter_index contains an out-of-range index.",
    )

    unique_filter_indices = np.unique(frac_filter_index)
    unique_responses: dict[int, complex] = {}
    for filter_index in unique_filter_indices.tolist():
        taps = filter_taps[int(filter_index)]
        # H(e^jw) = Σ_l h[l] exp(-j w l)。
        # 実行時の FIR 畳み込みと同じ tap 順で評価し、主経路の実応答を w0 に反映する。
        unique_responses[int(filter_index)] = complex(
            np.sum(taps * np.exp(-1j * float(angular_frequency_rad) * tap_index))
        )

    response = np.empty(frac_filter_index.shape, dtype=np.complex128)
    for filter_index, filter_response in unique_responses.items():
        response[frac_filter_index == int(filter_index)] = filter_response
    return response


class ShortFFTCovarianceAccumulator:
    """短 FFT により周波数 bin 別の多チャネル共分散を逐次更新する。

    このクラスは、入力信号 `[n_ch, n_sample]` を `block_size` sample ごとに分割し、
    `fft_size` 点 FFT で得た `X[k]` から `R[k] = E[X[k]X[k]^H]` を指数平均する。

    入力は時間領域の多チャネル信号、出力は shape `[n_bin, n_ch, n_ch]` の
    複素共分散と、重み更新周期に達したかを表す `update_ready` である。

    MVDR 重み設計、固定整相、差分 FIR 適用は責務に含めない。
    信号処理上は、固定遅延+差分補正 MVDR の統計ルートに位置づく。
    """

    def __init__(
        self,
        *,
        n_ch: int,
        fft_size: int,
        block_size: int,
        fs_hz: float,
        covariance_time_constant_sec: float,
        blocks_per_weight_update: int,
    ) -> None:
        """短 FFT 共分散推定器を構成する。

        Args:
            n_ch: センサチャネル数。単位は count。
            fft_size: 統計推定に使う FFT 点数。単位は sample。
            block_size: 1 回の共分散更新に使う入力 block 長。単位は sample。
            fs_hz: サンプリング周波数。単位は Hz。
            covariance_time_constant_sec: 指数平均の時定数。単位は秒。
            blocks_per_weight_update: MVDR 重み更新 1 回あたりの block 数。単位は count。

        Raises:
            ValueError: 各パラメータが正でない、または `block_size > fft_size` の場合。

        境界条件:
            `process` へ `block_size` 未満の端数だけが入った場合は内部バッファに保持し、
            共分散は更新しない。未来 sample をゼロ詰めして統計を作ると、短時間だけ
            入力 power を過小評価するためである。
        """
        require_positive_int("n_ch", int(n_ch))
        require_positive_int("fft_size", int(fft_size))
        require_positive_int("block_size", int(block_size))
        require_positive_float("fs_hz", float(fs_hz))
        require_positive_float("covariance_time_constant_sec", float(covariance_time_constant_sec))
        require_positive_int("blocks_per_weight_update", int(blocks_per_weight_update))
        require(int(block_size) <= int(fft_size), "block_size must not exceed fft_size.")

        self.n_ch = int(n_ch)
        self.fft_size = int(fft_size)
        self.block_size = int(block_size)
        self.fs_hz = float(fs_hz)
        self.covariance_time_constant_sec = float(covariance_time_constant_sec)
        self.blocks_per_weight_update = int(blocks_per_weight_update)

        block_duration_sec = float(self.block_size) / float(self.fs_hz)
        # R_b = alpha R_{b-1} + (1-alpha) X_b X_b^H。
        # 時定数を秒で指定し、block 単位の指数平均係数へ変換する。
        self.alpha = float(np.exp(-block_duration_sec / float(self.covariance_time_constant_sec)))
        self.covariance = np.zeros((self.fft_size, self.n_ch, self.n_ch), dtype=np.complex128)
        self._input_buffer = np.zeros((self.n_ch, 0), dtype=np.complex128)
        self._total_block_count = 0
        self._blocks_since_weight_update = 0

    def reset(self) -> None:
        """内部共分散と端数バッファを破棄する。

        Returns:
            なし。次回 `process` は初回 block として処理される。

        境界条件:
            入力 scene や target 方向を切り替える場合、古い共分散を残すと
            MVDR が前 scene の干渉統計へ引きずられるため、明示的に初期化する。
        """
        self.covariance = np.zeros((self.fft_size, self.n_ch, self.n_ch), dtype=np.complex128)
        self._input_buffer = np.zeros((self.n_ch, 0), dtype=np.complex128)
        self._total_block_count = 0
        self._blocks_since_weight_update = 0

    def process(self, x: NDArray[Any]) -> ShortFFTCovarianceUpdateResult:
        """入力 chunk を処理し、短 FFT 共分散を更新する。

        Args:
            x: 入力多チャネル信号。shape は `[n_ch, n_sample]`。
                axis=0 はセンサチャネル、axis=1 は時間 sample である。

        Returns:
            更新後の共分散と更新可能フラグ。
            `covariance` の shape は `[fft_size, n_ch, n_ch]` である。

        Raises:
            ValueError: 入力が 2 次元でない、またはチャネル数が一致しない場合。

        境界条件:
            端数 sample は `_input_buffer` に保持し、次 chunk と結合してから FFT する。
            統計ルートの FFT は係数設計用であり、ここで overlap-save は行わない。
        """
        input_signal = np.asarray(x, dtype=np.complex128)
        require(input_signal.ndim == 2, "x must have shape (n_ch, n_sample).")
        require(input_signal.shape[0] == self.n_ch, "x and accumulator must agree on n_ch.")

        # buffered shape: [n_ch, n_buffered_sample + n_sample]。
        # axis=1 の時間 sample だけを block_size ごとに切り出す。
        buffered = np.concatenate([self._input_buffer, input_signal], axis=1)
        processed_block_count = 0
        cursor = 0
        while cursor + self.block_size <= buffered.shape[1]:
            block = buffered[:, cursor : cursor + self.block_size]
            self._update_one_block(block)
            processed_block_count += 1
            cursor += self.block_size

        self._input_buffer = buffered[:, cursor:].copy()
        update_ready = self._blocks_since_weight_update >= self.blocks_per_weight_update
        if update_ready:
            self._blocks_since_weight_update = 0

        return ShortFFTCovarianceUpdateResult(
            covariance=self.covariance.copy(),
            update_ready=bool(update_ready),
            processed_block_count=int(processed_block_count),
            total_block_count=int(self._total_block_count),
        )

    def _update_one_block(self, block: NDArray[np.complex128]) -> None:
        """1 block 分の FFT と共分散指数平均を行う。"""
        # block shape: [n_ch, block_size]。
        # FFT axis=1 は時間 sample 軸であり、出力 X_ch_bin shape は [n_ch, fft_size] になる。
        x_ch_bin = np.fft.fft(block, n=self.fft_size, axis=1)

        # X_bin_ch shape: [n_bin, n_ch]。
        # 共分散 R[k] = X[k] X[k]^H を bin ごとに作るため、周波数 bin を先頭軸へ移す。
        x_bin_ch = np.asarray(np.moveaxis(x_ch_bin, 1, 0), dtype=np.complex128)

        # r_inst[k, ch_i, ch_j] = X[k, ch_i] conj(X[k, ch_j])。
        # 短 FFT の単一 block は 1 snapshot なので、ここでは snapshot 平均は行わず指数平均へ渡す。
        r_inst = np.einsum("kc,kd->kcd", x_bin_ch, x_bin_ch.conj(), optimize=True)
        self.covariance = self.alpha * self.covariance + (1.0 - self.alpha) * r_inst
        self._total_block_count += 1
        self._blocks_since_weight_update += 1


class LoadedMVDRWeightDesigner:
    """周波数 bin ごとに対角ローディング付き MVDR 重みを設計する。

    このクラスは、共分散 `R[k]` と target ステアリング `a[k]` から
    `w[k] = R_load[k]^{-1} a[k] / (a[k]^H R_load[k]^{-1} a[k])` を計算する。

    入力は shape `[n_bin, n_ch, n_ch]` の共分散と `[n_bin, n_ch]` の
    ステアリングであり、出力は `[n_bin, n_ch]` の MVDR 重みである。

    共分散推定、差分補正 FIR 化、波形への重み適用は責務に含めない。
    信号処理上は、短 FFT 統計ルート上の周波数 bin 別 MVDR 設計器に位置づく。
    """

    def __init__(
        self,
        *,
        diagonal_loading_ratio: float,
        denominator_floor: float = 1.0e-12,
    ) -> None:
        """MVDR 設計器を構成する。

        Args:
            diagonal_loading_ratio: 平均対角 power に対する対角ローディング比。無次元。
            denominator_floor: `a^H R^{-1} a` の最小許容値。無次元の応答 power 比。

        Raises:
            ValueError: ローディング比が負、または分母 floor が正でない場合。

        境界条件:
            共分散が特異、分母が小さい、または重みが非有限になった bin は fallback する。
            不安定な MVDR 重みを採用すると補正 FIR が target 自己消去や発振を起こすためである。
        """
        require_non_negative_float("diagonal_loading_ratio", float(diagonal_loading_ratio))
        require_positive_float("denominator_floor", float(denominator_floor))
        self.diagonal_loading_ratio = float(diagonal_loading_ratio)
        self.denominator_floor = float(denominator_floor)
        self._previous_weights: NDArray[np.complex128] | None = None

    def reset(self) -> None:
        """fallback 用の前回 MVDR 重みを破棄する。"""
        self._previous_weights = None

    def compute(
        self,
        covariance: NDArray[Any],
        steering_vector: NDArray[Any],
        fallback_weights: NDArray[Any],
    ) -> LoadedMVDRDesignResult:
        """共分散とステアリングから MVDR 重みを計算する。

        Args:
            covariance: 周波数 bin 別共分散。shape は `[n_bin, n_ch, n_ch]`。
                axis=0 は FFT bin、axis=1/2 はセンサチャネルである。
            steering_vector: target ステアリング。shape は `[n_bin, n_ch]`。
            fallback_weights: fallback 時に使う固定整相重み。shape は `[n_bin, n_ch]`。

        Returns:
            MVDR 重み、loaded covariance 条件数、fallback mask。
            `weights` の shape は `[n_bin, n_ch]` である。

        Raises:
            ValueError: 入力 shape が一致しない場合。

        境界条件:
            `np.linalg.solve` 失敗、分母 floor 未満、非有限重みでは、
            前回の安定重みがあればそれを使い、なければ固定整相重みに戻す。
            これは target を削る不安定更新より、固定整相へ退避する方が安全だからである。
        """
        covariance_array = np.asarray(covariance, dtype=np.complex128)
        steering = np.asarray(steering_vector, dtype=np.complex128)
        fallback = np.asarray(fallback_weights, dtype=np.complex128)
        require(covariance_array.ndim == 3, "covariance must have shape (n_bin, n_ch, n_ch).")
        require(
            covariance_array.shape[1] == covariance_array.shape[2],
            "covariance matrices must be square.",
        )
        require(
            steering.shape == covariance_array.shape[:2],
            "steering_vector must have shape (n_bin, n_ch).",
        )
        require(fallback.shape == steering.shape, "fallback_weights must have shape (n_bin, n_ch).")

        n_bin = int(covariance_array.shape[0])
        n_ch = int(covariance_array.shape[1])
        weights = np.zeros((n_bin, n_ch), dtype=np.complex128)
        condition_number = np.zeros(n_bin, dtype=np.float64)
        fallback_mask = np.zeros(n_bin, dtype=np.bool_)

        # 固定主経路は小数遅延 FIR の群遅延を含むため、target 応答は一般に 1+0j ではない。
        # 差分枝 q が target を通さない条件は `q^H a = w0^H a - w_mvdr^H a = 0` なので、
        # MVDR の distortionless 目標を固定主経路の複素応答に合わせる。
        desired_response = _weight_response(fallback, steering)

        for bin_index in range(n_bin):
            r_loaded = self._make_loaded_covariance(covariance_array[bin_index])
            condition_number[bin_index] = float(np.linalg.cond(r_loaded))
            weight, used_fallback = self._solve_one_bin(
                r_loaded,
                steering[bin_index],
                fallback[bin_index],
                desired_response[bin_index],
                bin_index,
            )
            weights[bin_index] = weight
            fallback_mask[bin_index] = bool(used_fallback)

        self._previous_weights = weights.copy()
        return LoadedMVDRDesignResult(
            weights=weights,
            loaded_condition_number=condition_number,
            fallback_mask=fallback_mask,
        )

    def _make_loaded_covariance(self, covariance: NDArray[np.complex128]) -> NDArray[np.complex128]:
        n_ch = int(covariance.shape[0])
        if self.diagonal_loading_ratio == 0.0:
            return covariance.copy()

        # epsilon = gamma trace(R)/M。
        # trace がゼロの初期状態でも solve が完全特異にならないよう、1.0 を下限 scale として使う。
        average_power = float(np.real(np.trace(covariance)) / float(n_ch))
        loading_power = self.diagonal_loading_ratio * (
            average_power if average_power > 0.0 else 1.0
        )
        return covariance + loading_power * np.eye(n_ch, dtype=np.complex128)

    def _solve_one_bin(
        self,
        loaded_covariance: NDArray[np.complex128],
        steering: NDArray[np.complex128],
        fallback_weight: NDArray[np.complex128],
        desired_response: np.complex128,
        bin_index: int,
    ) -> tuple[NDArray[np.complex128], bool]:
        try:
            # R_load u = a を解く。明示逆行列は条件数悪化を増幅するため作らない。
            response = np.linalg.solve(loaded_covariance, steering)
            denominator = np.vdot(steering, response)
            if abs(denominator) <= self.denominator_floor:
                return self._fallback_weight(fallback_weight, bin_index), True
            # 通常の MVDR は `w^H a = 1` を制約にするが、ここでは固定主経路と同じ
            # 複素 target 応答を保つ。`w^H a = g` には `conj(g)` を重みに掛ける必要がある。
            weight = np.asarray(
                np.conj(desired_response) * response / denominator, dtype=np.complex128
            )
            if not bool(np.all(np.isfinite(weight))):
                return self._fallback_weight(fallback_weight, bin_index), True
            return weight, False
        except np.linalg.LinAlgError:
            return self._fallback_weight(fallback_weight, bin_index), True

    def _fallback_weight(
        self,
        fallback_weight: NDArray[np.complex128],
        bin_index: int,
    ) -> NDArray[np.complex128]:
        if self._previous_weights is not None:
            # 前回値がある場合は係数段差を抑えるため、固定整相へ即時に戻さず前回 MVDR を維持する。
            return np.asarray(self._previous_weights[bin_index].copy(), dtype=np.complex128)
        # 前回値がない初回異常では、target 保護を優先して固定整相重みへ退避する。
        return np.asarray(fallback_weight.copy(), dtype=np.complex128)


class DifferenceCorrectionFIRDesigner:
    """固定整相重みと MVDR 重みの差分を時間領域 FIR 係数へ変換する。

    このクラスは、`q[k] = w0[k] - w_mvdr[k]` を作り、`w^H x` の共役規約に合わせて
    実適用用周波数応答 `conj(q[k])` を任意周波数 least-squares FIR として近似する。

    入力は `[n_bin, n_ch]` の固定整相重みと MVDR 重みであり、出力は
    `[n_ch, fir_taps]` の補正 FIR 係数である。

    MVDR 重み計算、共分散推定、実時間係数切替は責務に含めない。
    信号処理上は、固定整相主経路から引く補正枝の係数設計に位置づく。
    """

    def __init__(
        self,
        *,
        fir_taps: int,
        frequencies_hz: NDArray[Any],
        fs_hz: float,
    ) -> None:
        """差分補正 FIR 設計器を構成する。

        Args:
            fir_taps: 補正 FIR の tap 数。単位は sample。
            frequencies_hz: 設計周波数。shape は `[n_bin]`、単位は Hz。
            fs_hz: サンプリング周波数。単位は Hz。

        Raises:
            ValueError: `fir_taps`、`fs_hz`、または周波数軸が不正な場合。
        """
        require_positive_int("fir_taps", int(fir_taps))
        require_positive_float("fs_hz", float(fs_hz))
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        require(frequencies.ndim == 1, "frequencies_hz must have shape (n_bin,).")
        require(
            bool(np.all(np.isfinite(frequencies))),
            "frequencies_hz must contain only finite values.",
        )
        require(
            bool(np.all((0.0 <= frequencies) & (frequencies < float(fs_hz)))),
            "frequencies_hz must be in [0, fs_hz).",
        )

        self.fir_taps = int(fir_taps)
        self.frequencies_hz = frequencies
        self.fs_hz = float(fs_hz)
        self._frequency_response_matrix = _make_fir_frequency_response_matrix(
            frequencies_hz=frequencies,
            fir_taps=self.fir_taps,
            fs_hz=self.fs_hz,
        )

    def compute(
        self,
        w0: NDArray[Any],
        w_mvdr: NDArray[Any],
        steering_vector: NDArray[Any],
    ) -> DifferenceCorrectionDesignResult:
        """差分補正重みを FIR 化し、周波数応答診断を返す。

        Args:
            w0: 固定整相重み。shape は `[n_bin, n_ch]`。
            w_mvdr: MVDR 重み。shape は `[n_bin, n_ch]`。
            steering_vector: target ステアリング。shape は `[n_bin, n_ch]`。

        Returns:
            差分補正重み、実適用 FIR 係数、FIR 化後の最終重み、診断量。

        Raises:
            ValueError: 入力 shape が一致しない場合。

        境界条件:
            設計周波数数が FIR tap 数より少ない場合は underdetermined になる。
            その場合も最小ノルム least-squares 解を使い、設計周波数上の
            `q_reconstruction_error` を必ず評価して採否を判断する。
        """
        fixed_weight = np.asarray(w0, dtype=np.complex128)
        mvdr_weight = np.asarray(w_mvdr, dtype=np.complex128)
        steering = np.asarray(steering_vector, dtype=np.complex128)
        require(fixed_weight.ndim == 2, "w0 must have shape (n_bin, n_ch).")
        require(mvdr_weight.shape == fixed_weight.shape, "w_mvdr must have shape (n_bin, n_ch).")
        require(
            steering.shape == fixed_weight.shape,
            "steering_vector must have shape (n_bin, n_ch).",
        )
        require(
            fixed_weight.shape[0] == self.frequencies_hz.size,
            "w0 and frequencies_hz must agree on n_bin.",
        )

        # q_weight_freq は数式上の q[k] = w0[k] - w_mvdr[k]。
        # 最終出力は y = w0^H X - q^H X なので、補正枝が target を通さないかを後段で確認する。
        q_weight_freq = fixed_weight - mvdr_weight

        # 実際の畳み込みは Σ h*x で実装するため、周波数応答側には q^H X の共役規約を焼き込む。
        # apply_freq[k, ch] = conj(q_weight_freq[k, ch]) とすると、
        # FFT 上の積和が q[k]^H X[k] になる。
        q_apply_freq = np.conj(q_weight_freq)

        # V[k,l] = exp(-j 2π f_k l / fs)。shape は [n_bin, fir_taps]。
        # 任意の物理周波数 f_k に対し、V @ h が時間領域 FIR の周波数応答になる。
        frequency_response_matrix = self._frequency_response_matrix

        # q_apply_taps.T shape: [fir_taps, n_ch]。
        # 各チャネルについて V h ≈ conj(q) を解き、任意周波数上の補正応答を合わせる。
        q_apply_taps_by_tap, _, _, _ = np.linalg.lstsq(
            frequency_response_matrix,
            q_apply_freq,
            rcond=None,
        )
        q_apply_taps = np.asarray(q_apply_taps_by_tap.T, dtype=np.complex128)

        # 設計に使った同じ物理周波数で FIR 応答を再評価し、近似誤差を診断する。
        reconstructed_apply_freq = np.asarray(
            frequency_response_matrix @ q_apply_taps_by_tap,
            dtype=np.complex128,
        )
        reconstructed_q_weight_freq = np.conj(reconstructed_apply_freq)
        final_weight_freq = fixed_weight - reconstructed_q_weight_freq
        diagnostics = DifferenceCorrectionDiagnostics(
            target_response_w0=_weight_response(fixed_weight, steering),
            target_response_mvdr=_weight_response(mvdr_weight, steering),
            target_response_final=_weight_response(final_weight_freq, steering),
            q_blocking_response=_weight_response(reconstructed_q_weight_freq, steering),
            q_reconstruction_error=q_weight_freq - reconstructed_q_weight_freq,
        )
        return DifferenceCorrectionDesignResult(
            q_weight_freq=q_weight_freq,
            q_apply_freq=q_apply_freq,
            q_apply_taps=q_apply_taps,
            reconstructed_q_weight_freq=reconstructed_q_weight_freq,
            final_weight_freq=final_weight_freq,
            diagnostics=diagnostics,
        )


def _make_fir_frequency_response_matrix(
    *,
    frequencies_hz: NDArray[np.float64],
    fir_taps: int,
    fs_hz: float,
) -> NDArray[np.complex128]:
    """任意周波数における FIR 周波数応答行列を返す。

    Args:
        frequencies_hz: 設計周波数。shape は `[n_bin]`、単位は Hz。
        fir_taps: FIR tap 数。単位は sample。
        fs_hz: サンプリング周波数。単位は Hz。

    Returns:
        応答行列。shape は `[n_bin, fir_taps]`。
        axis=0 は設計周波数、axis=1 は FIR tap index である。
    """
    tap_index = np.arange(int(fir_taps), dtype=np.float64)
    angular_frequency_rad = 2.0 * np.pi * frequencies_hz / float(fs_hz)
    # H(f_k) = Σ_l h[l] exp(-j 2π f_k l / fs)。
    # np.fft.ifft は DFT bin 前提になるため、任意 Hz 配列ではこの Vandermonde 行列を使う。
    response_matrix = np.exp(-1j * angular_frequency_rad[:, np.newaxis] * tap_index[np.newaxis, :])
    return np.asarray(response_matrix, dtype=np.complex128)


class DifferenceCorrectionFIR:
    """差分補正 FIR を時間領域で多チャネル入力へ適用する。

    このクラスは、チャネル別 FIR 係数 `q_apply_taps[ch, tap]` を使い、
    `z[n] = sum_ch sum_l h[ch,l] x[ch,n-l]` を逐次計算する。

    入力は時間領域信号 `[n_ch, n_sample]`、出力は補正枝出力 `[n_sample]` である。

    固定整相主経路、MVDR 重み設計、係数更新スケジューリングは責務に含めない。
    信号処理上は、最終出力 `y[n] = y0[n] - z[n]` の `z[n]` 生成部に位置づく。
    """

    def __init__(self, *, n_ch: int, fir_taps: int) -> None:
        """差分補正 FIR を構成する。

        Args:
            n_ch: センサチャネル数。単位は count。
            fir_taps: FIR tap 数。単位は sample。

        Raises:
            ValueError: `n_ch` または `fir_taps` が正でない場合。

        境界条件:
            初回 chunk では過去 sample が存在しないため、FIR 履歴はゼロで初期化する。
            これは因果 FIR の先頭不足区間をゼロ入力として扱うことに対応する。
        """
        require_positive_int("n_ch", int(n_ch))
        require_positive_int("fir_taps", int(fir_taps))
        self.n_ch = int(n_ch)
        self.fir_taps = int(fir_taps)
        self.taps = np.zeros((self.n_ch, self.fir_taps), dtype=np.complex128)
        self._state = np.zeros((self.n_ch, max(self.fir_taps - 1, 0)), dtype=np.complex128)

    def reset(self) -> None:
        """FIR 履歴をゼロに戻す。

        Returns:
            なし。

        境界条件:
            入力系列を切り替える場合、前系列の末尾 sample を履歴に残すと
            次系列の先頭へ人工的な漏れ込みが出るため、明示的にゼロへ戻す。
        """
        self._state = np.zeros((self.n_ch, max(self.fir_taps - 1, 0)), dtype=np.complex128)

    def update_coefficients(self, taps: NDArray[Any]) -> None:
        """補正 FIR 係数を更新する。

        Args:
            taps: 新しい FIR 係数。shape は `[n_ch, fir_taps]`。
                axis=0 はセンサチャネル、axis=1 は FIR tap である。

        Returns:
            なし。

        Raises:
            ValueError: 係数 shape が一致しない場合、または非有限値を含む場合。

        境界条件:
            NaN/inf 係数は 1 sample で出力全体へ伝搬するため、更新時点で拒否する。
        """
        coefficient = np.asarray(taps, dtype=np.complex128)
        require(
            coefficient.shape == (self.n_ch, self.fir_taps),
            "taps must have shape (n_ch, fir_taps).",
        )
        require(bool(np.all(np.isfinite(coefficient))), "taps must contain only finite values.")
        self.taps = coefficient.copy()

    def process(self, x: NDArray[Any]) -> NDArray[np.complex128]:
        """入力 chunk に差分補正 FIR を適用する。

        Args:
            x: 入力多チャネル信号。shape は `[n_ch, n_sample]`。
                axis=0 はセンサチャネル、axis=1 は時間 sample である。

        Returns:
            補正枝出力 `z`。shape は `[n_sample]`。

        Raises:
            ValueError: 入力 shape が一致しない場合。

        境界条件:
            chunk 分割に依存しない出力にするため、末尾 `fir_taps - 1` sample を
            次回用履歴として保持する。`fir_taps == 1` では履歴は空配列である。
        """
        input_signal = np.asarray(x, dtype=np.complex128)
        require(input_signal.ndim == 2, "x must have shape (n_ch, n_sample).")
        require(input_signal.shape[0] == self.n_ch, "x and FIR must agree on n_ch.")

        # x_ext shape: [n_ch, fir_taps - 1 + n_sample]。
        # 先頭側に過去 sample を置き、各出力時刻で現在 sample までの tap 窓を取り出す。
        x_ext = np.concatenate([self._state, input_signal], axis=1)
        n_sample = int(input_signal.shape[1])
        output = np.zeros(n_sample, dtype=np.complex128)
        reversed_taps = self.taps[:, ::-1]
        for sample_index in range(n_sample):
            segment = x_ext[:, sample_index : sample_index + self.fir_taps]
            # segment は [過去 ... 現在]、reversed_taps は [h[L-1] ... h[0]]。
            # 積和により z[n] = Σ_l h[l] x[n-l] の因果 FIR を実現する。
            output[sample_index] = np.sum(reversed_taps * segment)

        if self.fir_taps > 1:
            self._state = x_ext[:, -(self.fir_taps - 1) :].copy()
        else:
            self._state = np.zeros((self.n_ch, 0), dtype=np.complex128)
        return output


def _weight_response(
    weights: NDArray[np.complex128],
    steering: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """周波数 bin ごとの `w[k]^H a[k]` を計算する。"""
    # weights/steering shape: [n_bin, n_ch]。
    # axis=1 のチャネル内積により、各 bin の target 応答を得る。
    return np.asarray(np.sum(weights.conj() * steering, axis=1), dtype=np.complex128)


__all__ = [
    "DifferenceCorrectionDesignResult",
    "DifferenceCorrectionDiagnostics",
    "DifferenceCorrectionFIR",
    "DifferenceCorrectionFIRDesigner",
    "LoadedMVDRDesignResult",
    "LoadedMVDRWeightDesigner",
    "STANDARD_FRACTIONAL_DELAY_PATTERN_COUNT",
    "STANDARD_FRACTIONAL_DELAY_TAP_COUNT",
    "ShortFFTCovarianceAccumulator",
    "ShortFFTCovarianceUpdateResult",
    "design_distortionless_fixed_weights",
    "design_fixed_delay_fractional_weights_from_delay_table",
    "design_standard_fractional_delay_filter_bank",
]
