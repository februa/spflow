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
class DelayAlignedBeamCovarianceUpdateResult:
    """遅延中心切り出し型の beam 別共分散更新結果を保持する。

    このクラスは、各 beam・各 channel の整数遅延を反映して切り出した
    128 sample snapshot から作った共分散を返すための結果型である。

    入力は `DelayAlignedBeamCovarianceAccumulator.process` が処理した
    `[n_ch, n_sample]` の実時間信号であり、出力は beam 別共分散と
    MVDR 設計へ渡す beam 合算共分散である。

    MVDR 重み設計、固定整相、差分 FIR 化は責務に含めない。
    信号処理上は、差分 MVDR の統計ルートにおける共分散積分方式の
    差し替え候補に位置づく。
    """

    beam_covariance: NDArray[np.complex128]
    summed_covariance_ch_ch_bin: NDArray[np.complex128]
    covariance_for_mvdr: NDArray[np.complex128]
    frequencies_hz: NDArray[np.float64]
    update_ready: bool
    processed_frame_count: int
    total_frame_count: int


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


def extract_delay_centered_snapshots(
    y: NDArray[Any],
    delay_table_sample: NDArray[Any],
    *,
    snapshot_length: int = 128,
    center_sample: int | None = None,
) -> NDArray[np.float64]:
    """整数遅延中心の短時間 snapshot を beam/channel 別に切り出す。

    Args:
        y: 入力実信号。shape は `[n_ch, n_sample]`。
            axis=0 はセンサチャネル、axis=1 は時間 sample である。
        delay_table_sample: 整数遅延 sample 表。shape は `[n_ch, n_beam]`。
            `delay_table_sample[ch, beam]` は基準中心からの中心 sample ずれである。
        snapshot_length: 切り出す sample 数。単位は sample。標準値は 128。
        center_sample: 基準中心 sample。`None` の場合は `n_sample // 2` を使う。
            32768 sample frame では 16384 が基準中心になる。

    Returns:
        遅延中心切り出し snapshot。shape は `[n_beam, n_ch, snapshot_length]`。
        axis=0 は beam、axis=1 は sensor channel、axis=2 は短時間 sample である。

    Raises:
        ValueError: 入力 shape、遅延表 shape、または実信号条件が不正な場合。

    境界条件:
        `center_sample + delay_table_sample[ch, beam]` を中心に偶数長 snapshot を切り出す。
        128 sample では `[center-64, center+64)` を採用する。入力範囲外は、
        存在しない過去・未来 sample を観測していないことを明示するため 0 で埋める。
    """
    raw_signal = np.asarray(y)
    require(raw_signal.ndim == 2, "y must have shape (n_ch, n_sample).")
    if np.iscomplexobj(raw_signal):
        require(
            bool(np.allclose(np.imag(raw_signal), 0.0, rtol=0.0, atol=1.0e-12)),
            "y must be real-valued for 65-bin rFFT covariance estimation.",
        )
    require(bool(np.all(np.isfinite(raw_signal))), "y must contain only finite values.")
    signal = np.asarray(np.real(raw_signal), dtype=np.float64)

    delay_raw = np.asarray(delay_table_sample)
    require(delay_raw.ndim == 2, "delay_table_sample must have shape (n_ch, n_beam).")
    require(delay_raw.shape[0] == signal.shape[0], "delay_table_sample and y must agree on n_ch.")
    require(
        bool(np.all(np.isfinite(delay_raw))),
        "delay_table_sample must contain only finite values.",
    )
    require(
        bool(np.all(delay_raw == np.rint(delay_raw))),
        "delay_table_sample must contain integer sample delays.",
    )
    require_positive_int("snapshot_length", int(snapshot_length))

    n_ch = int(signal.shape[0])
    n_sample = int(signal.shape[1])
    n_beam = int(delay_raw.shape[1])
    if center_sample is None:
        base_center_sample = n_sample // 2
    else:
        base_center_sample = int(center_sample)
    require(0 <= base_center_sample <= n_sample, "center_sample must be inside or at frame edge.")

    delay = np.asarray(delay_raw, dtype=np.int64)
    half_before = int(snapshot_length) // 2
    snapshots = np.zeros((n_beam, n_ch, int(snapshot_length)), dtype=np.float64)
    for beam_index in range(n_beam):
        for ch_index in range(n_ch):
            # center は MATLAB 側の `16384 + delay_table(ch, beam)` に対応する。
            # 偶数長 128 sample では中心 sample を切り出し後 index 64 に置く規約にする。
            center = base_center_sample + int(delay[ch_index, beam_index])
            start = center - half_before
            stop = start + int(snapshot_length)
            source_start = max(start, 0)
            source_stop = min(stop, n_sample)
            if source_start >= source_stop:
                # 遅延中心が入力 frame から完全に外れる場合は、未観測区間として全 0 のままにする。
                continue
            destination_start = source_start - start
            destination_stop = destination_start + (source_stop - source_start)
            snapshots[
                beam_index,
                ch_index,
                destination_start:destination_stop,
            ] = signal[ch_index, source_start:source_stop]
    return snapshots


class DelayAlignedBeamCovarianceAccumulator:
    """遅延中心 snapshot から beam 別の 65-bin 共分散を積分する。

    このクラスは、入力実信号 `[n_ch, n_sample]` に対して、各 beam・各 channel の
    `delay_table_sample[ch, beam]` を中心 sample ずれとして 128 sample を切り出し、
    rFFT の 65 bin で `R_beam[k] = X_beam[k] X_beam[k]^H` を指数平均する。

    入力は frame 単位の多チャネル実信号、出力は beam 別共分散
    `[n_beam, n_ch, n_ch, n_bin]` と、MVDR 用に beam 合算した
    `[n_ch, n_ch, n_bin]` の共分散である。

    MVDR 重み設計、固定整相、差分 FIR 適用は責務に含めない。
    信号処理上は、従来の連続短 FFT 共分散積分と比較できる差し替え可能な統計ルートである。
    """

    def __init__(
        self,
        *,
        delay_table_sample: NDArray[Any],
        fs_hz: float,
        snapshot_length: int = 128,
        frame_size: int = 32768,
        center_sample: int | None = 16384,
        covariance_time_constant_sec: float = 10.0,
        frames_per_weight_update: int = 1,
    ) -> None:
        """遅延中心切り出し型の共分散推定器を構成する。

        Args:
            delay_table_sample: 整数遅延 sample 表。shape は `[n_ch, n_beam]`。
                単位は sample。MATLAB の `int32(tau * fs)` に対応する。
            fs_hz: サンプリング周波数。単位は Hz。
            snapshot_length: 各 beam/channel で切り出す sample 数。標準値は 128 sample。
            frame_size: 1 回の `process` が想定する frame 長。標準値は 32768 sample。
            center_sample: 基準中心 sample。標準値は 16384 sample。
                `None` の場合は入力 frame の `n_sample // 2` を使う。
            covariance_time_constant_sec: frame 単位指数平均の時定数。単位は秒。
            frames_per_weight_update: MVDR 重み更新 1 回あたりの frame 数。単位は count。

        Raises:
            ValueError: 各パラメータ、遅延表、または sample 数が不正な場合。

        境界条件:
            128 sample snapshot の rFFT は実信号を前提にするため、非ゼロ虚部を持つ入力は拒否する。
            複素 baseband 入力を扱う場合は 65 bin ではなく full FFT 共分散を別方式として定義する。
        """
        require_positive_float("fs_hz", float(fs_hz))
        require_positive_int("snapshot_length", int(snapshot_length))
        require_positive_int("frame_size", int(frame_size))
        require_positive_float("covariance_time_constant_sec", float(covariance_time_constant_sec))
        require_positive_int("frames_per_weight_update", int(frames_per_weight_update))
        require(int(snapshot_length) % 2 == 0, "snapshot_length must be even.")

        delay_raw = np.asarray(delay_table_sample)
        require(delay_raw.ndim == 2, "delay_table_sample must have shape (n_ch, n_beam).")
        require(delay_raw.shape[0] > 0 and delay_raw.shape[1] > 0, "delay_table_sample is empty.")
        require(bool(np.all(np.isfinite(delay_raw))), "delay_table_sample must be finite.")
        require(
            bool(np.all(delay_raw == np.rint(delay_raw))),
            "delay_table_sample must contain integer sample delays.",
        )

        self.delay_table_sample = np.asarray(delay_raw, dtype=np.int64)
        self.n_ch = int(self.delay_table_sample.shape[0])
        self.n_beam = int(self.delay_table_sample.shape[1])
        self.fs_hz = float(fs_hz)
        self.snapshot_length = int(snapshot_length)
        self.frame_size = int(frame_size)
        self.center_sample = None if center_sample is None else int(center_sample)
        self.covariance_time_constant_sec = float(covariance_time_constant_sec)
        self.frames_per_weight_update = int(frames_per_weight_update)
        self.n_bin = self.snapshot_length // 2 + 1
        self.frequencies_hz = np.asarray(
            np.fft.rfftfreq(self.snapshot_length, d=1.0 / self.fs_hz),
            dtype=np.float64,
        )

        frame_duration_sec = float(self.frame_size) / self.fs_hz
        # R_t = alpha R_{t-1} + (1-alpha) X_t X_t^H。
        # ここでは 32768 sample frame から 1 個の beam-aligned snapshot 群を作るため、
        # 忘却係数は frame 時間を時定数で割って決める。
        self.alpha = float(np.exp(-frame_duration_sec / self.covariance_time_constant_sec))
        self.beam_covariance = np.zeros(
            (self.n_beam, self.n_ch, self.n_ch, self.n_bin),
            dtype=np.complex128,
        )
        self.summed_covariance_ch_ch_bin = np.zeros(
            (self.n_ch, self.n_ch, self.n_bin),
            dtype=np.complex128,
        )
        self._total_frame_count = 0
        self._frames_since_weight_update = 0

    def reset(self) -> None:
        """内部共分散と frame counter を破棄する。

        Returns:
            なし。次回 `process` は初回 frame として処理される。

        境界条件:
            scene、array geometry、または delay table を切り替える場合、古い beam 別共分散を
            残すと別方位の統計を MVDR に混入させるため、明示的に初期化する。
        """
        self.beam_covariance = np.zeros(
            (self.n_beam, self.n_ch, self.n_ch, self.n_bin),
            dtype=np.complex128,
        )
        self.summed_covariance_ch_ch_bin = np.zeros(
            (self.n_ch, self.n_ch, self.n_bin),
            dtype=np.complex128,
        )
        self._total_frame_count = 0
        self._frames_since_weight_update = 0

    def process(self, y: NDArray[Any]) -> DelayAlignedBeamCovarianceUpdateResult:
        """1 frame の遅延中心 snapshot から beam 別共分散を更新する。

        Args:
            y: 入力実信号。shape は `[n_ch, n_sample]`。
                axis=0 はセンサチャネル、axis=1 は時間 sample である。

        Returns:
            更新後の beam 別共分散、beam 合算共分散、MVDR 用共分散を返す。
            `beam_covariance` の shape は `[n_beam, n_ch, n_ch, 65]`、
            `summed_covariance_ch_ch_bin` の shape は `[n_ch, n_ch, 65]`、
            `covariance_for_mvdr` の shape は `[65, n_ch, n_ch]` である。

        Raises:
            ValueError: 入力 shape、実信号条件、または frame 長が不正な場合。
        """
        raw_signal = np.asarray(y)
        require(raw_signal.ndim == 2, "y must have shape (n_ch, n_sample).")
        require(raw_signal.shape[0] == self.n_ch, "y and delay_table_sample must agree on n_ch.")
        require(raw_signal.shape[1] == self.frame_size, "y must have frame_size samples.")

        snapshots = extract_delay_centered_snapshots(
            raw_signal,
            self.delay_table_sample,
            snapshot_length=self.snapshot_length,
            center_sample=self.center_sample,
        )
        self._update_one_frame(snapshots)
        self._total_frame_count += 1
        self._frames_since_weight_update += 1

        update_ready = self._frames_since_weight_update >= self.frames_per_weight_update
        if update_ready:
            self._frames_since_weight_update = 0

        # summed_covariance_ch_ch_bin shape: [n_ch, n_ch, n_bin]。
        # MVDRWeightDesigner は [n_bin, n_ch, n_ch] 規約なので bin 軸を先頭へ移す。
        covariance_for_mvdr = np.asarray(
            np.moveaxis(self.summed_covariance_ch_ch_bin, 2, 0),
            dtype=np.complex128,
        )
        return DelayAlignedBeamCovarianceUpdateResult(
            beam_covariance=self.beam_covariance.copy(),
            summed_covariance_ch_ch_bin=self.summed_covariance_ch_ch_bin.copy(),
            covariance_for_mvdr=covariance_for_mvdr,
            frequencies_hz=self.frequencies_hz.copy(),
            update_ready=bool(update_ready),
            processed_frame_count=1,
            total_frame_count=int(self._total_frame_count),
        )

    def _update_one_frame(self, snapshots: NDArray[np.float64]) -> None:
        """1 frame から得た beam-aligned snapshot 群で共分散を指数平均する。"""
        # snapshots shape: [n_beam, n_ch, snapshot_length]。
        # axis=2 は 128 sample の短時間軸であり、rFFT 後は 65 個の片側周波数 bin になる。
        x_beam_ch_bin = np.fft.rfft(snapshots, n=self.snapshot_length, axis=2)

        # r_inst[beam, ch_i, ch_j, bin] = X[beam, ch_i, bin] conj(X[beam, ch_j, bin])。
        # beam ごとに遅延中心が異なるため、beam 軸は平均せず保持し、MVDR 直前で合算する。
        r_inst = np.asarray(
            np.einsum("bck,bdk->bcdk", x_beam_ch_bin, x_beam_ch_bin.conj(), optimize=True),
            dtype=np.complex128,
        )
        self.beam_covariance = self.alpha * self.beam_covariance + (1.0 - self.alpha) * r_inst

        # beam 合算 R_sum[ch_i, ch_j, k] = Σ_beam R_beam[ch_i, ch_j, k]。
        # ユーザー指定どおり MVDR 計算時に [n_ch, n_ch, 65] を保持する。
        self.summed_covariance_ch_ch_bin = np.asarray(
            np.sum(self.beam_covariance, axis=0),
            dtype=np.complex128,
        )


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
    """固定整相重みと MVDR 重みの差分を全 FFT bin IFFT で FIR 係数へ変換する。

    このクラスは、全 DFT bin で `q[k] = w0[k] - w_mvdr[k]` を作り、`w^H x` の
    共役規約に合わせた実適用用周波数応答 `conj(q[k])` を IFFT して時間 FIR を得る。
    `fir_taps` が FFT bin 数より短い場合は、IFFT で得たインパルス応答の先頭
    `fir_taps` sample を因果 FIR として採用し、切り捨て後の周波数応答を診断する。

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
            frequencies_hz: 全 DFT bin の設計周波数。shape は `[n_bin]`、単位は Hz。
                `np.fft.fftfreq(n_bin, d=1/fs_hz)` と同じ signed bin 順序である。
            fs_hz: サンプリング周波数。単位は Hz。

        Raises:
            ValueError: `fir_taps`、`fs_hz`、周波数軸、または tap 数と bin 数の関係が不正な場合。
        """
        require_positive_int("fir_taps", int(fir_taps))
        require_positive_float("fs_hz", float(fs_hz))
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        require(frequencies.ndim == 1, "frequencies_hz must have shape (n_bin,).")
        require(
            bool(np.all(np.isfinite(frequencies))),
            "frequencies_hz must contain only finite values.",
        )
        expected_frequencies = _make_full_fft_bin_frequencies(
            fft_size=int(frequencies.size),
            fs_hz=float(fs_hz),
        )
        require(
            bool(np.allclose(frequencies, expected_frequencies, rtol=0.0, atol=1.0e-9)),
            "frequencies_hz must be np.fft.fftfreq(n_bin, d=1/fs_hz) in DFT bin order.",
        )
        require(
            int(fir_taps) <= int(frequencies.size),
            "fir_taps must not exceed the number of full FFT bins.",
        )

        self.fir_taps = int(fir_taps)
        self.frequencies_hz = frequencies
        self.fs_hz = float(fs_hz)
        self.fft_size = int(frequencies.size)

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
            `frequencies_hz` は全 DFT bin なので、`fir_taps == n_bin` では
            IFFT/FFT の丸め誤差だけが残る。`fir_taps < n_bin` では IFFT 後の
            インパルス応答を因果側から切り出すため、切り捨て誤差を
            `q_reconstruction_error` として必ず評価して採否を判断する。
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

        # q_apply_freq shape: [n_bin, n_ch]。axis=0 は DFT bin 順序である。
        # 全 bin 応答を IFFT することで、DFT の定義どおり
        # q_apply_freq[k,ch] = FFT{h[ch,:]}[k] を満たす時間応答を得る。
        q_apply_full_impulse_by_bin_channel = np.fft.ifft(q_apply_freq, axis=0)

        # q_apply_taps shape: [n_ch, fir_taps]。
        # IFFT 結果の axis=0 は時間 sample なので、因果 FIR として先頭 tap を採用し、
        # 時間領域適用部の `[n_ch, fir_taps]` 規約へ transpose する。
        q_apply_taps = np.asarray(
            q_apply_full_impulse_by_bin_channel[: self.fir_taps, :].T,
            dtype=np.complex128,
        )

        # fir_taps < n_bin の場合は、切り出した tap をゼロ詰めして FFT し、
        # 実際に時間領域 FIR として使う応答を全 bin で再評価する。
        q_apply_taps_padded = np.zeros((self.fft_size, fixed_weight.shape[1]), dtype=np.complex128)
        q_apply_taps_padded[: self.fir_taps, :] = q_apply_taps.T
        reconstructed_apply_freq = np.asarray(
            np.fft.fft(q_apply_taps_padded, axis=0),
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


def _make_full_fft_bin_frequencies(
    *,
    fft_size: int,
    fs_hz: float,
) -> NDArray[np.float64]:
    """全 DFT bin の signed 周波数軸を返す。

    Args:
        fft_size: FFT bin 数。単位は sample。
        fs_hz: サンプリング周波数。単位は Hz。

    Returns:
        `np.fft.fftfreq(fft_size, d=1/fs_hz)` と同じ周波数軸。
        shape は `[fft_size]`、単位は Hz。axis=0 は DFT bin 順序である。

    Raises:
        ValueError: `fft_size` または `fs_hz` が正でない場合。

    境界条件:
        差分補正 FIR は全 bin 応答を IFFT して設計するため、正周波数だけの
        rFFT 軸では負周波数側の応答が未定義になる。ここでは FFT/IFFT と同じ
        signed bin 順序を唯一の正式な周波数軸として扱う。
    """
    require_positive_int("fft_size", int(fft_size))
    require_positive_float("fs_hz", float(fs_hz))
    return np.asarray(np.fft.fftfreq(int(fft_size), d=1.0 / float(fs_hz)), dtype=np.float64)


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


@dataclass(frozen=True)
class StreamingBlock:
    """ストリーミング入力 block と同期メタデータを保持する。

    このクラスは、主経路と diff-MVDR 補正枝へ同じ入力サンプル範囲を渡すための
    immutable な block 表現である。

    入力は多チャネル時間波形 `[n_ch, length]` と block 番号・開始 sample であり、
    出力は各経路の `ProcessedBlock` に引き継がれるメタデータである。

    FIR 計算、係数更新、経路合成は責務に含めない。
    信号処理上は、複数経路の sample index を一致させる同期境界に位置づく。
    """

    array_id: str
    block_index: int
    start_sample: int
    length: int
    fs_hz: float
    data: NDArray[Any]
    valid_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class ProcessedBlock:
    """ストリーミング経路の出力 block と同期メタデータを保持する。

    このクラスは、主経路または補正枝が出力した `[n_beam, length]` 波形と、
    合成前に照合すべき block metadata をまとめる。

    入力は各経路の FIR 出力であり、最終加算は `AlignedPathCombiner` が行う。
    FIR 履歴、係数ラッチ、共分散更新は責務に含めない。
    信号処理上は、`y_main[n]` と `y_diff[n]` の sample index 一致を検査する単位である。
    """

    array_id: str
    path_id: str
    block_index: int
    start_sample: int
    length: int
    fs_hz: float
    latency_tag: str
    coeff_version: int
    data: NDArray[np.complex128]
    valid_mask: NDArray[np.bool_]


class CausalBlockFIR:
    """入力履歴付き direct FIR を block 入出力で実行する。

    このクラスは、複数系列の因果 FIR
    `y_s[n] = sum_{p=0}^{M-1} h_s[p] x_s[n-p]` を同一 block 境界で処理する。

    入力は系列別 block `[n_series, length]` と系列別 FIR `[n_series, tap_length]`、
    出力は入力 block と同じ sample index を持つ `[n_series, length]` である。

    ビーム合成、diff-MVDR 重み設計、係数更新スケジューリングは責務に含めない。
    信号処理上は、主経路と補正枝が共有する FIR 切り出し規約そのものである。
    """

    def __init__(self, *, n_series: int, tap_length: int) -> None:
        """因果 block FIR を構成する。

        Args:
            n_series: 独立に FIR を適用する系列数。単位は count。
            tap_length: FIR tap 数。単位は sample。

        Raises:
            ValueError: `n_series` または `tap_length` が正でない場合。

        境界条件:
            初回 block では過去入力がないため、履歴を 0 で初期化する。
            これにより、出力長を削らずに初期過渡を明示的なゼロ過去入力として扱う。
        """
        require_positive_int("n_series", int(n_series))
        require_positive_int("tap_length", int(tap_length))
        self.n_series = int(n_series)
        self.tap_length = int(tap_length)
        self.history_length = self.tap_length - 1
        self._history = np.zeros((self.n_series, self.history_length), dtype=np.complex128)

    def reset(self) -> None:
        """履歴をゼロに戻す。

        Returns:
            なし。

        境界条件:
            入力ストリームを切り替える場合、前ストリームの末尾が次ストリームの先頭へ
            混入しないよう、履歴を明示的に破棄する。
        """
        self._history = np.zeros((self.n_series, self.history_length), dtype=np.complex128)

    def process(self, x_block: NDArray[Any], taps: NDArray[Any]) -> NDArray[np.complex128]:
        """1 block の因果 FIR 出力を返す。

        Args:
            x_block: 入力系列。shape は `[n_series, length]`。
                axis=0 は独立系列、axis=1 は時間 sample である。
            taps: FIR 係数。shape は `[n_series, tap_length]`。
                axis=0 は系列、axis=1 は tap index `p` である。

        Returns:
            FIR 出力。shape は `[n_series, length]`。
            出力 `[:, i]` は入力 block の `[:, i]` と同じ global sample index に対応する。

        Raises:
            ValueError: 入力 shape が不正な場合、または係数に非有限値が含まれる場合。
        """
        signal = np.asarray(x_block, dtype=np.complex128)
        coefficient = np.asarray(taps, dtype=np.complex128)
        require(signal.ndim == 2, "x_block must have shape (n_series, length).")
        require(signal.shape[0] == self.n_series, "x_block and FIR must agree on n_series.")
        require(
            coefficient.shape == (self.n_series, self.tap_length),
            "taps must have shape (n_series, tap_length).",
        )
        require(bool(np.all(np.isfinite(coefficient))), "taps must contain only finite values.")

        block_length = int(signal.shape[1])
        extended = np.concatenate((self._history, signal), axis=1)
        output = np.empty((self.n_series, block_length), dtype=np.complex128)
        for series_index in range(self.n_series):
            # np.convolve の full 出力 index `history_length + i` が、
            # 因果式 y[n0+i] = Σ_p h[p] x[n0+i-p] に対応する。
            convolved = np.convolve(extended[series_index], coefficient[series_index], mode="full")
            output[series_index] = convolved[
                self.history_length : self.history_length + block_length
            ]

        if self.history_length > 0:
            self._history = extended[:, -self.history_length :].copy()
        else:
            self._history = np.zeros((self.n_series, 0), dtype=np.complex128)
        return output


class FractionalDelayMainPath:
    """整数遅延と小数遅延 FIR による主経路を StreamingBlock 単位で処理する。

    このクラスは、`DelayTable` の整数遅延と選択済み小数遅延 FIR を合成した
    チャネル別因果 FIR を作り、固定整相の主経路出力 `[n_beam, length]` を返す。

    入力は `StreamingBlock`、出力は path_id `main_fractional_delay` の `ProcessedBlock` である。

    diff-MVDR 補正、共分散推定、係数更新判断は責務に含めない。
    信号処理上は、`y_main[n]` を定義する固定遅延 + 小数 FIR 経路である。
    """

    def __init__(
        self,
        *,
        delay_table: DelayTable,
        fractional_filter_bank: FractionalDelayFilterBank,
        fs_hz: float,
        array_id: str,
        latency_tag: str,
        coeff_version: int = 0,
        average_channels: bool = True,
    ) -> None:
        """主経路 processor を構成する。

        Args:
            delay_table: 整数遅延と小数遅延フィルタ index。shape は `[n_ch, n_beam]`。
            fractional_filter_bank: 事前計算済み小数遅延 FIR バンク。
            fs_hz: サンプリング周波数。単位は Hz。
            array_id: アレイ識別子。
            latency_tag: 主経路と補正枝で一致させる遅延基準タグ。
            coeff_version: block 境界で latch された係数 version。
            average_channels: `True` の場合は channel 平均、`False` の場合は channel 和。

        Raises:
            ValueError: delay table が小数遅延 index を持たない、または遅延が不正な場合。
        """
        if delay_table.frac_filter_index is None:
            raise ValueError("delay_table must include frac_filter_index.")
        require_positive_float("fs_hz", float(fs_hz))
        delay_int = np.asarray(delay_table.delay_int, dtype=np.int64)
        frac_index = np.asarray(delay_table.frac_filter_index, dtype=np.int64)
        require(delay_int.ndim == 2, "delay_int must have shape (n_ch, n_beam).")
        require(frac_index.shape == delay_int.shape, "frac_filter_index must match delay_int.")
        require(bool(np.all(delay_int >= 0)), "delay_int must be non-negative for causal FIR.")

        self.delay_table = delay_table
        self.fs_hz = float(fs_hz)
        self.array_id = str(array_id)
        self.latency_tag = str(latency_tag)
        self.coeff_version = int(coeff_version)
        self.average_channels = bool(average_channels)
        self.n_ch = int(delay_table.n_ch)
        self.n_beam = int(delay_table.n_beam)
        self._flat_taps = self._build_flat_taps(
            delay_int=delay_int,
            frac_filter_index=frac_index,
            fractional_filter_bank=fractional_filter_bank,
        )
        self._fir = CausalBlockFIR(
            n_series=self.n_beam * self.n_ch,
            tap_length=int(self._flat_taps.shape[1]),
        )

    def _build_flat_taps(
        self,
        *,
        delay_int: NDArray[np.int64],
        frac_filter_index: NDArray[np.int64],
        fractional_filter_bank: FractionalDelayFilterBank,
    ) -> NDArray[np.complex128]:
        filter_taps = np.asarray(fractional_filter_bank.frac_filters, dtype=np.float64)
        require(
            bool(np.all((0 <= frac_filter_index) & (frac_filter_index < filter_taps.shape[0]))),
            "frac_filter_index contains an out-of-range index.",
        )
        tap_length = int(np.max(delay_int)) + int(fractional_filter_bank.n_tap)
        flat_taps = np.zeros((self.n_beam * self.n_ch, tap_length), dtype=np.complex128)
        for beam_index in range(self.n_beam):
            for ch_index in range(self.n_ch):
                row_index = beam_index * self.n_ch + ch_index
                delay_sample = int(delay_int[ch_index, beam_index])
                frac_index = int(frac_filter_index[ch_index, beam_index])
                # combined FIR は h_combined[p] = frac[p - delay_int] であり、
                # 整数遅延後に小数遅延 FIR を掛ける直列処理と同じ因果応答になる。
                flat_taps[
                    row_index,
                    delay_sample : delay_sample + fractional_filter_bank.n_tap,
                ] = filter_taps[frac_index]
        return flat_taps

    def reset(self) -> None:
        """主経路 FIR 履歴をゼロに戻す。"""
        self._fir.reset()

    def process(self, block: StreamingBlock) -> ProcessedBlock:
        """1 block の主経路出力を返す。

        Args:
            block: 入力 block。`data` shape は `[n_ch, length]`。

        Returns:
            主経路出力。`data` shape は `[n_beam, length]`。

        Raises:
            ValueError: block metadata または shape が不正な場合。
        """
        signal = _validate_streaming_block(block, expected_array_id=self.array_id, n_ch=self.n_ch)
        # flat_input shape: [n_beam*n_ch, length]。
        # beam ごとに同じ channel 入力を使い、combined FIR だけを beam/channel ごとに変える。
        flat_input = np.broadcast_to(
            signal[np.newaxis, :, :],
            (self.n_beam, self.n_ch, block.length),
        ).reshape(self.n_beam * self.n_ch, block.length)
        flat_output = self._fir.process(flat_input, self._flat_taps)
        steered = flat_output.reshape(self.n_beam, self.n_ch, block.length)
        if self.average_channels:
            data = np.mean(steered, axis=1, dtype=np.complex128)
        else:
            data = np.sum(steered, axis=1, dtype=np.complex128)
        return ProcessedBlock(
            array_id=block.array_id,
            path_id="main_fractional_delay",
            block_index=block.block_index,
            start_sample=block.start_sample,
            length=block.length,
            fs_hz=block.fs_hz,
            latency_tag=self.latency_tag,
            coeff_version=self.coeff_version,
            data=np.asarray(data, dtype=np.complex128),
            valid_mask=np.asarray(block.valid_mask.copy(), dtype=np.bool_),
        )


class DiffMVDRCorrectionPath:
    """diff-MVDR 補正枝 FIR を StreamingBlock 単位で処理する。

    このクラスは、beam/channel 別の補正 FIR 係数を入力 channel に適用し、
    channel 和で `[n_beam, length]` の補正枝出力を返す。

    入力は `StreamingBlock` と事前に latch 済みの補正 FIR、出力は path_id
    `diff_mvdr_correction` の `ProcessedBlock` である。

    MVDR 重み設計、主経路 FIR、最終加算は責務に含めない。
    信号処理上は、`y_out[n] = y_main[n] + y_diff_mvdr[n]` の `y_diff_mvdr[n]` を生成する。
    """

    def __init__(
        self,
        *,
        correction_taps: NDArray[Any],
        fs_hz: float,
        array_id: str,
        latency_tag: str,
        coeff_version: int = 0,
    ) -> None:
        """補正枝 processor を構成する。

        Args:
            correction_taps: 補正 FIR。shape は `[n_beam, n_ch, n_tap]`。
                `y_out = y_main + y_diff_mvdr` の符号で渡す。
            fs_hz: サンプリング周波数。単位は Hz。
            array_id: アレイ識別子。
            latency_tag: 主経路と一致させる遅延基準タグ。
            coeff_version: block 境界で latch された係数 version。

        Raises:
            ValueError: 係数 shape または値が不正な場合。
        """
        require_positive_float("fs_hz", float(fs_hz))
        taps = np.asarray(correction_taps, dtype=np.complex128)
        require(taps.ndim == 3, "correction_taps must have shape (n_beam, n_ch, n_tap).")
        require(taps.shape[0] > 0 and taps.shape[1] > 0 and taps.shape[2] > 0, "invalid taps.")
        require(bool(np.all(np.isfinite(taps))), "correction_taps must contain only finite values.")
        self.n_beam = int(taps.shape[0])
        self.n_ch = int(taps.shape[1])
        self.n_tap = int(taps.shape[2])
        self.fs_hz = float(fs_hz)
        self.array_id = str(array_id)
        self.latency_tag = str(latency_tag)
        self.coeff_version = int(coeff_version)
        self._flat_taps = taps.reshape(self.n_beam * self.n_ch, self.n_tap)
        self._fir = CausalBlockFIR(n_series=self.n_beam * self.n_ch, tap_length=self.n_tap)

    def reset(self) -> None:
        """補正枝 FIR 履歴をゼロに戻す。"""
        self._fir.reset()

    def process(self, block: StreamingBlock) -> ProcessedBlock:
        """1 block の補正枝出力を返す。

        Args:
            block: 入力 block。`data` shape は `[n_ch, length]`。

        Returns:
            補正枝出力。`data` shape は `[n_beam, length]`。

        Raises:
            ValueError: block metadata または shape が不正な場合。
        """
        signal = _validate_streaming_block(block, expected_array_id=self.array_id, n_ch=self.n_ch)
        flat_input = np.broadcast_to(
            signal[np.newaxis, :, :],
            (self.n_beam, self.n_ch, block.length),
        ).reshape(self.n_beam * self.n_ch, block.length)
        flat_output = self._fir.process(flat_input, self._flat_taps)
        data = np.sum(flat_output.reshape(self.n_beam, self.n_ch, block.length), axis=1)
        return ProcessedBlock(
            array_id=block.array_id,
            path_id="diff_mvdr_correction",
            block_index=block.block_index,
            start_sample=block.start_sample,
            length=block.length,
            fs_hz=block.fs_hz,
            latency_tag=self.latency_tag,
            coeff_version=self.coeff_version,
            data=np.asarray(data, dtype=np.complex128),
            valid_mask=np.asarray(block.valid_mask.copy(), dtype=np.bool_),
        )


class AlignedPathCombiner:
    """主経路と補正枝の block metadata を検証して加算する。

    このクラスは、`start_sample`、`length`、`latency_tag`、`coeff_version` などが
    一致する場合だけ `y_out[n] = y_main[n] + y_diff[n]` を実行する。

    入力は2つの `ProcessedBlock`、出力は path_id `main_plus_diff_mvdr` の `ProcessedBlock` である。

    FIR 処理や係数更新は責務に含めない。
    信号処理上は、暗黙の trim / zero padding / sample shift を防ぐ同期検査点である。
    """

    def add(self, main: ProcessedBlock, diff: ProcessedBlock) -> ProcessedBlock:
        """主経路と補正枝を metadata 検証後に加算する。

        Args:
            main: 主経路出力。`data` shape は `[n_beam, length]`。
            diff: 補正枝出力。`data` shape は `[n_beam, length]`。

        Returns:
            合成後出力。`data` shape は `[n_beam, length]`。

        Raises:
            ValueError: 必須 metadata または data shape が一致しない場合。
        """
        _require_aligned_blocks(main, diff)
        return ProcessedBlock(
            array_id=main.array_id,
            path_id="main_plus_diff_mvdr",
            block_index=main.block_index,
            start_sample=main.start_sample,
            length=main.length,
            fs_hz=main.fs_hz,
            latency_tag=main.latency_tag,
            coeff_version=main.coeff_version,
            data=np.asarray(main.data + diff.data, dtype=np.complex128),
            valid_mask=np.asarray(main.valid_mask & diff.valid_mask, dtype=np.bool_),
        )


class FixedDelayDiffMVDRStreamingProcessor:
    """主経路と diff-MVDR 補正枝を同一 StreamingBlock から処理する。

    このクラスは、`FractionalDelayMainPath` と `DiffMVDRCorrectionPath` を同じ入力 block に
    適用し、`AlignedPathCombiner` で同期検証後に最終出力を返す。

    入力は `StreamingBlock`、出力は最終 beam 出力 `[n_beam, length]` を持つ
    `ProcessedBlock` である。

    共分散更新、係数生成、複数アレイ barrier は責務に含めない。
    信号処理上は、単一アレイ内の主経路・補正枝同期済み streaming 出力を定義する。
    """

    def __init__(
        self,
        *,
        main_path: FractionalDelayMainPath,
        correction_path: DiffMVDRCorrectionPath,
        combiner: AlignedPathCombiner | None = None,
    ) -> None:
        """streaming processor を構成する。

        Args:
            main_path: 小数遅延 FIR 主経路。
            correction_path: diff-MVDR FIR 補正枝。
            combiner: metadata 検証付き合成器。`None` の場合は標準合成器を使う。
        """
        self.main_path = main_path
        self.correction_path = correction_path
        self.combiner = combiner if combiner is not None else AlignedPathCombiner()

    def reset(self) -> None:
        """主経路と補正枝の FIR 履歴をゼロに戻す。"""
        self.main_path.reset()
        self.correction_path.reset()

    def process(self, block: StreamingBlock) -> ProcessedBlock:
        """1 block の最終出力を返す。

        Args:
            block: 主経路と補正枝へ共通に入力する block。

        Returns:
            合成後出力。`data` shape は `[n_beam, length]`。
        """
        main = self.main_path.process(block)
        diff = self.correction_path.process(block)
        return self.combiner.add(main, diff)


def _validate_streaming_block(
    block: StreamingBlock,
    *,
    expected_array_id: str,
    n_ch: int,
) -> NDArray[np.complex128]:
    """StreamingBlock の shape と metadata を検証し、複素入力配列を返す。"""
    require(block.array_id == expected_array_id, "block array_id does not match processor.")
    require_positive_int("block.length", int(block.length))
    require_positive_float("block.fs_hz", float(block.fs_hz))
    signal = np.asarray(block.data, dtype=np.complex128)
    valid_mask = np.asarray(block.valid_mask, dtype=np.bool_)
    require(signal.shape == (int(n_ch), int(block.length)), "block.data shape is invalid.")
    require(valid_mask.shape == (int(block.length),), "block.valid_mask shape is invalid.")
    require(block.start_sample >= 0, "block.start_sample must be non-negative.")
    require(block.block_index >= 0, "block.block_index must be non-negative.")
    return signal


def _require_aligned_blocks(first: ProcessedBlock, second: ProcessedBlock) -> None:
    """2つの ProcessedBlock が加算可能な同期 metadata を持つことを検証する。"""
    require(first.array_id == second.array_id, "array_id mismatch.")
    require(first.block_index == second.block_index, "block_index mismatch.")
    require(first.start_sample == second.start_sample, "start_sample mismatch.")
    require(first.length == second.length, "length mismatch.")
    require(abs(float(first.fs_hz) - float(second.fs_hz)) <= 1.0e-12, "fs_hz mismatch.")
    require(first.latency_tag == second.latency_tag, "latency_tag mismatch.")
    require(first.coeff_version == second.coeff_version, "coeff_version mismatch.")
    require(first.data.shape == second.data.shape, "data shape mismatch.")
    require(first.valid_mask.shape == second.valid_mask.shape, "valid_mask shape mismatch.")


def _weight_response(
    weights: NDArray[np.complex128],
    steering: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """周波数 bin ごとの `w[k]^H a[k]` を計算する。"""
    # weights/steering shape: [n_bin, n_ch]。
    # axis=1 のチャネル内積により、各 bin の target 応答を得る。
    return np.asarray(np.sum(weights.conj() * steering, axis=1), dtype=np.complex128)


__all__ = [
    "AlignedPathCombiner",
    "CausalBlockFIR",
    "extract_delay_centered_snapshots",
    "DelayAlignedBeamCovarianceUpdateResult",
    "DelayAlignedBeamCovarianceAccumulator",
    "DiffMVDRCorrectionPath",
    "FixedDelayDiffMVDRStreamingProcessor",
    "FractionalDelayMainPath",
    "ProcessedBlock",
    "StreamingBlock",
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
