"""方位別遅延整合共分散で再利用する中心サンプル表を構成するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float, require_positive_int
from .geometry import relative_arrival_delay
from .steering_power_weighting import (
    SteeringPowerChannelWeighting,
    prepare_steering_power_channel_weighting,
)


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


@dataclass(frozen=True)
class CovarianceSnapshotCenterSchedule:
    """方式3の中心サンプル表と方位一致表を再利用可能な状態で保持する。

    このクラスは、2個の90度区間についてchannel・beamごとの中心sampleを初期化時に
    1回だけ生成する。frameごとには選択した表を使い、各beam中心から128 sampleを
    1回切り出す。beam間の切り出し区間は重複してよい。

    入力は1秒信号`[n_ch,n_sample]`、出力は`[n_ch,N,n_beam]`のsnapshot表である。
    FFT、共分散計算、指数積分は責務に含めない。

    信号処理上は、方式3で方位ごとに異なるchannel時刻を選ぶ再利用可能なindex設計に
    位置づく。

    Attributes:
        beam_azimuth_deg: 90度区間別beam方位。shapeは`[2,n_beam]`、単位はdeg。
        global_direction_azimuth_deg: 積分先の全方位軸。shapeは`[2*n_beam-1]`、単位はdeg。
        direction_match_indices: local beamから全方位積分先への対応。shapeは`[2,n_beam]`。
        channel_center_samples: 中心sample表。shapeは`[2,n_ch,n_beam]`、単位はsample。
        fs_hz: サンプリング周波数。単位はHz。
        snapshot_length_samples: 各中心から取得する長さ。単位はsample。
    """

    beam_azimuth_deg: FloatArray
    global_direction_azimuth_deg: FloatArray
    direction_match_indices: IntArray
    channel_center_samples: IntArray
    fs_hz: float
    snapshot_length_samples: int

    @property
    def n_ch(self) -> int:
        """channel数を返す。"""

        return int(self.channel_center_samples.shape[1])

    @property
    def n_beam(self) -> int:
        """90度区間当たりのbeam数を返す。"""

        return int(self.channel_center_samples.shape[2])

    def extract_snapshots(self, signal: NDArray[Any], *, azimuth_segment_index: int) -> NDArray[Any]:
        """選択した1秒中心表から全beamのsnapshotを切り出す。

        Args:
            signal: 1秒信号。shapeは`[n_ch,n_sample]`。axis=0はchannel、axis=1はsample。
            azimuth_segment_index: 使用する90度区間。`0`または`1`。

        Returns:
            snapshot表。shapeは`[n_ch,N,n_beam]`。
            axis=0はchannel、axis=1はsnapshot内sample、axis=2はbeamである。

        Raises:
            ValueError: 入力shape、segment index、または中心範囲が不正な場合。

        境界条件:
            偶数長Nでは中心の左を`N//2` sample、中心自身を含む右を`N-N//2`
            sample取得する。異なるbeamのsnapshot区間は重複を許可し、ゼロ詰めしない。

        Notes:
            本メソッドは中心表を再生成しない。同じscheduleをframe間で保持して使う。
        """

        input_signal = np.asarray(signal)
        segment_index = int(azimuth_segment_index)
        samples_per_second = int(round(self.fs_hz))
        require(segment_index in (0, 1), "azimuth_segment_index must be 0 or 1.")
        require(input_signal.shape == (self.n_ch, samples_per_second), "signal must match (n_ch, one_second_samples).")

        snapshot_length = int(self.snapshot_length_samples)
        left_extent = snapshot_length // 2
        offsets = np.arange(snapshot_length, dtype=np.int32) - np.int32(left_extent)
        centers = self.channel_center_samples[segment_index]
        # center `[n_ch,n_beam]`へoffset `[N]`をbroadcastし、index `[n_ch,n_beam,N]`を作る。
        sample_indices = centers[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
        require(bool(np.all(sample_indices >= 0)), "snapshot start must not precede the signal.")
        require(bool(np.all(sample_indices < samples_per_second)), "snapshot stop must not exceed the signal.")
        channel_indices = np.arange(self.n_ch, dtype=np.int32)[:, np.newaxis, np.newaxis]
        snapshots_ch_beam_sample = input_signal[channel_indices, sample_indices]
        # advanced indexing後`[n_ch,n_beam,N]`から、FFT用の`[n_ch,N,n_beam]`へaxisを移す。
        return np.asarray(np.moveaxis(snapshots_ch_beam_sample, 2, 1))

    def extract_snapshot_chunk(
        self,
        signal: NDArray[Any],
        *,
        azimuth_segment_index: int,
        beam_start_index: int,
        beam_stop_index: int,
    ) -> NDArray[Any]:
        """連続するbeam範囲だけをbatched snapshotとして切り出す。

        Args:
            signal: 1秒信号。shapeは`[n_ch,n_sample]`。
            azimuth_segment_index: 使用する90度区間。`0`または`1`。
            beam_start_index: 先頭local beam index。inclusive。
            beam_stop_index: 終端local beam index。exclusive。

        Returns:
            snapshot chunk。shapeは`[n_ch,N,n_chunk_beam]`。

        Raises:
            ValueError: 入力shape、segment、またはbeam範囲が不正な場合。

        境界条件:
            全159 beamの瞬時共分散を一括生成すると`[159,n_ch,n_ch,n_bin]`が巨大になるため、
            共分散更新では本メソッドで作業領域を制限する。
        """

        input_signal = np.asarray(signal)
        segment_index = int(azimuth_segment_index)
        beam_start = int(beam_start_index)
        beam_stop = int(beam_stop_index)
        samples_per_second = int(round(self.fs_hz))
        require(segment_index in (0, 1), "azimuth_segment_index must be 0 or 1.")
        require(0 <= beam_start < beam_stop <= self.n_beam, "beam chunk range is invalid.")
        require(input_signal.shape == (self.n_ch, samples_per_second), "signal must match (n_ch, one_second_samples).")

        snapshot_length = int(self.snapshot_length_samples)
        left_extent = snapshot_length // 2
        offsets = np.arange(snapshot_length, dtype=np.int32) - np.int32(left_extent)
        centers = self.channel_center_samples[segment_index, :, beam_start:beam_stop]
        sample_indices = centers[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
        channel_indices = np.arange(self.n_ch, dtype=np.int32)[:, np.newaxis, np.newaxis]
        snapshots_ch_beam_sample = input_signal[channel_indices, sample_indices]
        return np.asarray(np.moveaxis(snapshots_ch_beam_sample, 2, 1))

    def calculate_time_axis_restoration_phase(self, *, azimuth_segment_index: int) -> NDArray[np.complex64]:
        """channelごとに異なるsnapshot中心を共通時刻へ戻す位相係数を返す。

        Args:
            azimuth_segment_index: 使用する90度区間。`0`または`1`。

        Returns:
            位相復元係数。shapeは`[n_ch,n_bin,n_beam]`、dtypeは`complex64`。
            axis=0はchannel、axis=1はrFFT周波数bin、axis=2はbeamである。

        Raises:
            ValueError: segment indexが`0`または`1`でない場合。

        Notes:
            snapshot中心`center[ch,beam]`はchannelごとに異なるため、そのままFFTすると
            各spectrumは異なる絶対時刻を基準に持つ。beam内のchannel平均中心を共通基準
            `center_ref[beam]`とし、`Delta t=(center-center_ref)/fs`に対して
            `exp(-j 2 pi f Delta t)`を掛け、方式2と同じ共通時間軸へ戻す。
            中心表と同様にframe非依存なので、Accumulator初期化時に1回だけ計算して再利用する。
        """

        segment_index = int(azimuth_segment_index)
        require(segment_index in (0, 1), "azimuth_segment_index must be 0 or 1.")
        centers_sample = np.asarray(
            self.channel_center_samples[segment_index],
            dtype=np.float64,
        )
        # reference_center shapeは`[1,n_beam]`。channel平均により全chを同一beam時刻へ揃える。
        reference_center_sample = np.mean(centers_sample, axis=0, keepdims=True)
        relative_center_time_s = (centers_sample - reference_center_sample) / float(self.fs_hz)
        frequency_hz = np.fft.rfftfreq(
            self.snapshot_length_samples,
            d=1.0 / float(self.fs_hz),
        )
        # phase shapeはbroadcastにより`[n_ch,n_bin,n_beam]`となる。
        phase = np.exp(
            -1j
            * 2.0
            * np.pi
            * relative_center_time_s[:, np.newaxis, :]
            * frequency_hz[np.newaxis, :, np.newaxis]
        )
        return np.asarray(phase, dtype=np.complex64)


@dataclass(frozen=True)
class DirectionMatchedCovarianceUpdate:
    """1秒更新で選択した方位と更新後共分散を保持する。

    Attributes:
        azimuth_segment_index: 今回選択した中心表・方位一致表のindex。
        global_direction_indices: 更新した全方位index。shapeは`[n_beam]`。
        active_direction_covariance: 更新後共分散。shapeは`[n_beam,n_ch,n_ch,n_bin]`。
        processed_second_count: 処理済み1秒frame数。
    """

    azimuth_segment_index: int
    global_direction_indices: IntArray
    active_direction_covariance: NDArray[np.complex64]
    processed_second_count: int


@dataclass(frozen=True)
class CompletedDirectionSteeringMetrics:
    """2秒完成周期で確定したsteering power整合度を保持する。

    Attributes:
        steering_power: `|u^H X|^2`の方位別指数積分。shapeは`[n_direction,n_bin]`。
        total_power: `norm(X)^2`の方位別指数積分。shapeは`[n_direction,n_bin]`。
        eta: `steering_power/total_power`。shapeは`[n_direction,n_bin]`。
        eta_valid: 分母と有限性が有効なmask。shapeは`[n_direction,n_bin]`。
        active_channel_count: 正のshading係数を持つchannel数。shapeは`[n_bin]`。
        effective_channel_count: shadingを含む`N_eff`。shapeは`[n_bin]`。
        noise_eta_reference: 空間白色雑音の`1/N_eff`。shapeは`[n_bin]`。
        completed_cycle_count: 完成した2秒周期数。
    """

    steering_power: NDArray[np.float32]
    total_power: NDArray[np.float32]
    eta: NDArray[np.float32]
    eta_valid: NDArray[np.bool_]
    active_channel_count: NDArray[np.int32]
    effective_channel_count: NDArray[np.float32]
    noise_eta_reference: NDArray[np.float32]
    completed_cycle_count: int


@dataclass(frozen=True)
class MaximumSpatialCorrelationTable:
    """方位・周波数ごとの最大非対角正規化相関を保持する。

    Attributes:
        azimuth_deg: 全方位軸。shapeは`[n_direction]`、単位はdeg。
        frequency_hz: rFFT周波数軸。shapeは`[n_bin]`、単位はHz。
        maximum_correlation: 最大相関。shapeは`[n_direction,n_bin]`、範囲は`[0,1]`。
    """

    azimuth_deg: FloatArray
    frequency_hz: FloatArray
    maximum_correlation: NDArray[np.float32]


def calculate_maximum_spatial_correlation_table(
    direction_covariance: NDArray[Any],
    azimuth_deg: NDArray[Any],
    *,
    fs_hz: float,
    denominator_floor: float = 1.0e-20,
    pair_chunk_size: int = 256,
) -> MaximumSpatialCorrelationTable:
    """方位別共分散から各周波数ビンの最大非対角相関を計算する。

    Args:
        direction_covariance: 方位別共分散。shapeは`[n_direction,n_ch,n_ch,n_bin]`。
        azimuth_deg: 方位軸。shapeは`[n_direction]`、単位はdeg。
        fs_hz: サンプリング周波数。単位はHz。
        denominator_floor: `Rii*Rjj`の最小許容power二乗。これ以下は相関0とする。
        pair_chunk_size: 一括処理する非対角channel pair数。

    Returns:
        `[方位,周波数]`の最大相関テーブルと軸。

    Raises:
        ValueError: shape、Hermitian共分散の対角power、単位、またはchannel数が不正な場合。

    境界条件:
        対角成分は自己相関1になるため必ず除外する。無信号binで分母が小さいpairは、
        数値雑音を高相関と誤認しないよう0とする。
    """

    covariance = np.asarray(direction_covariance, dtype=np.complex64)
    azimuth = np.asarray(azimuth_deg, dtype=np.float32)
    sample_rate = float(fs_hz)
    floor_value = float(denominator_floor)
    chunk_size = int(pair_chunk_size)
    require_positive_float("fs_hz", sample_rate)
    require_positive_float("denominator_floor", floor_value)
    require_positive_int("pair_chunk_size", chunk_size)
    require(covariance.ndim == 4, "direction_covariance must have shape (n_direction, n_ch, n_ch, n_bin).")
    require(covariance.shape[1] == covariance.shape[2], "covariance channel axes must be square.")
    require(covariance.shape[1] >= 2, "maximum off-diagonal correlation requires at least two channels.")
    require(azimuth.shape == (covariance.shape[0],), "azimuth_deg must match n_direction.")

    # diagonal shapeは`[n_direction,n_ch,n_bin]`。Hermitian共分散の対角は実powerなので実部を使う。
    diagonal_power = np.real(np.diagonal(covariance, axis1=1, axis2=2)).transpose(0, 2, 1)
    require(bool(np.all(diagonal_power >= -np.finfo(np.float32).eps)), "covariance diagonal power must be non-negative.")
    diagonal_power = np.maximum(diagonal_power, np.float32(0.0))
    maximum_correlation = np.zeros((covariance.shape[0], covariance.shape[3]), dtype=np.float32)

    first_pair_channels, second_pair_channels = np.triu_indices(covariance.shape[1], k=1)
    # 46,360 pairをPythonで1組ずつ処理せず、chunk内を`[direction,pair,bin]`でベクトル化する。
    # 全pair一括は305chで数GBになるため、pair chunkにより一時配列を制限する。
    for pair_start in range(0, first_pair_channels.size, chunk_size):
        pair_stop = min(pair_start + chunk_size, first_pair_channels.size)
        first_channels = first_pair_channels[pair_start:pair_stop]
        second_channels = second_pair_channels[pair_start:pair_stop]
        denominator_power = (
            diagonal_power[:, first_channels, :] * diagonal_power[:, second_channels, :]
        )
        valid = denominator_power > np.float32(floor_value)
        pair_correlation = np.zeros(denominator_power.shape, dtype=np.float32)
        cross_power = np.abs(covariance[:, first_channels, second_channels, :])
        pair_correlation[valid] = np.asarray(
            cross_power[valid] / np.sqrt(denominator_power[valid]),
            dtype=np.float32,
        )
        maximum_correlation = np.maximum(
            maximum_correlation,
            np.max(pair_correlation, axis=1),
        )

    # 丸め誤差で1を僅かに超える場合だけ物理範囲へ戻す。大幅超過は共分散生成側の異常である。
    require(bool(np.all(maximum_correlation <= 1.0 + 1.0e-4)), "normalized correlation exceeds its physical range.")
    maximum_correlation = np.clip(maximum_correlation, 0.0, 1.0).astype(np.float32, copy=False)
    frequency_hz = np.asarray(
        np.fft.rfftfreq(2 * (covariance.shape[3] - 1), d=1.0 / sample_rate),
        dtype=np.float32,
    )
    return MaximumSpatialCorrelationTable(
        azimuth_deg=azimuth.copy(),
        frequency_hz=frequency_hz,
        maximum_correlation=maximum_correlation,
    )


class DirectionMatchedCovarianceAccumulator:
    """1秒ごとに左右の更新側を切り替え、方位別に指数積分する。

    偶数秒は0–90度、奇数秒は180–90度側の159 snapshotを使う。
    片側79方位を各2回、共有境界90度を1回観測し、globalには159方位だけを保持する。

    入力は1秒信号`[n_ch,fs]`、出力は今回更新した159方位の共分散である。
    相関閾値による採否、beam方向集約、MVDR重み設計は責務に含めない。
    """

    def __init__(
        self,
        schedule: CovarianceSnapshotCenterSchedule,
        *,
        coef: float | None = None,
        integration_time_seconds: float | None = None,
        beam_chunk_size: int = 8,
        steering_table: NDArray[Any] | None = None,
        channel_weight_table: NDArray[Any] | None = None,
        eta_denominator_floor: float = 1.0e-20,
    ) -> None:
        """再利用するscheduleと瞬時共分散の更新係数を設定する。

        Args:
            schedule: 中心表と方位一致表。frame間で同じインスタンスを再利用する。
            coef: 全方位共通の更新係数。`integration_time_seconds`と同時指定不可。
            integration_time_seconds: 等価積分時間。単位はs。方位一致表から各global方位の
                実更新rateを求め、`coef=2/(1+rate*T)`を方位ごとに設定する。
            beam_chunk_size: FFTと瞬時共分散を一括処理するbeam数。
            steering_table: 物理steering。shapeは`[n_ch,n_bin,n_direction]`。
                指定時は瞬時steering/total powerも同時積分する。
            channel_weight_table: 周波数別channel shading。shapeは`[n_ch,n_bin]`、
                非負の線形値。`None`では全channelを係数1で使用する。
            eta_denominator_floor: eta分母の数値下限。

        Raises:
            ValueError: 係数指定が排他的でない、または範囲外の場合。
        """

        require(
            (coef is None) != (integration_time_seconds is None),
            "specify exactly one of coef or integration_time_seconds.",
        )
        self.schedule = schedule
        self.beam_chunk_size = int(beam_chunk_size)
        self.eta_denominator_floor = float(eta_denominator_floor)
        require_positive_int("beam_chunk_size", self.beam_chunk_size)
        require_positive_float("eta_denominator_floor", self.eta_denominator_floor)
        n_direction = int(schedule.global_direction_azimuth_deg.size)
        if integration_time_seconds is not None:
            integration_time = float(integration_time_seconds)
            require_positive_float("integration_time_seconds", integration_time)
            # 2秒周期内に各global方位が何回現れるかを数え、実際の更新rate[update/s]へ変換する。
            update_count_per_cycle = np.bincount(
                schedule.direction_match_indices.reshape(-1),
                minlength=n_direction,
            ).astype(np.float32)
            update_rate_per_second = update_count_per_cycle / np.float32(2.0)
            require(bool(np.all(update_rate_per_second > 0.0)), "every global direction must have an update rate.")
            self.direction_update_coef = np.minimum(
                np.float32(2.0) / (np.float32(1.0) + update_rate_per_second * np.float32(integration_time)),
                np.float32(1.0),
            ).astype(np.float32, copy=False)
        else:
            update_coef = float(coef) if coef is not None else 0.0
            require(0.0 < update_coef <= 1.0, "coef must satisfy 0 < coef <= 1.")
            self.direction_update_coef = np.full(n_direction, update_coef, dtype=np.float32)
        self.n_bin = schedule.snapshot_length_samples // 2 + 1
        self._steering_power_weighting: SteeringPowerChannelWeighting | None = None
        self.steering_power: NDArray[np.float32] | None = None
        self.total_power: NDArray[np.float32] | None = None
        require(
            steering_table is not None or channel_weight_table is None,
            "channel_weight_table requires steering_table.",
        )
        if steering_table is not None:
            steering = np.asarray(steering_table, dtype=np.complex64)
            require(
                steering.shape == (schedule.n_ch, self.n_bin, n_direction),
                "steering_table must have shape (n_ch, n_bin, n_direction).",
            )
            self._steering_power_weighting = prepare_steering_power_channel_weighting(
                steering,
                channel_weight_table,
            )
            self.steering_power = np.zeros((n_direction, self.n_bin), dtype=np.float32)
            self.total_power = np.zeros((n_direction, self.n_bin), dtype=np.float32)
        # 中心sample表はframe間で不変なので、2個のsegmentの時間軸復元位相も初期化時に固定する。
        # shapeは`[2,n_ch,n_bin,n_beam]`で、毎秒のexp計算を避けて同じ係数を再利用する。
        self._time_axis_restoration_phase = np.stack(
            [
                schedule.calculate_time_axis_restoration_phase(azimuth_segment_index=segment_index)
                for segment_index in (0, 1)
            ],
            axis=0,
        ).astype(np.complex64, copy=False)
        self.direction_covariance = np.zeros(
            (
                schedule.global_direction_azimuth_deg.size,
                schedule.n_ch,
                schedule.n_ch,
                self.n_bin,
            ),
            dtype=np.complex64,
        )
        self._processed_second_count = 0

    def reset(self) -> None:
        """全方位共分散と秒counterをゼロへ戻す。"""

        self.direction_covariance.fill(np.complex64(0.0 + 0.0j))
        if self.steering_power is not None:
            self.steering_power.fill(np.float32(0.0))
        if self.total_power is not None:
            self.total_power.fill(np.float32(0.0))
        self._processed_second_count = 0

    def process_one_second(self, signal: NDArray[Any]) -> DirectionMatchedCovarianceUpdate:
        """現在秒の159 snapshotを対応するglobal方位へ順次積分する。

        Args:
            signal: 1秒実信号。shapeは`[n_ch,fs]`。

        Returns:
            選択segment、更新した全方位index、更新後共分散。

        Raises:
            ValueError: 入力が複素数、またはshapeが不正な場合。

        境界条件:
            非選択方位は減衰させず保持する。異なる方位の統計を混ぜないためである。
        """

        input_signal = np.asarray(signal)
        require(not np.iscomplexobj(input_signal), "signal must be real-valued.")
        segment_index = self._processed_second_count % 2
        direction_indices = self.schedule.direction_match_indices[segment_index]
        for beam_start in range(0, self.schedule.n_beam, self.beam_chunk_size):
            beam_stop = min(beam_start + self.beam_chunk_size, self.schedule.n_beam)
            snapshots = self.schedule.extract_snapshot_chunk(
                input_signal,
                azimuth_segment_index=segment_index,
                beam_start_index=beam_start,
                beam_stop_index=beam_stop,
            )
            # chunk `[n_ch,N,n_chunk]`をaxis=1でrFFTし、`X[ch,bin,n_chunk]`へ変換する。
            spectrum = np.asarray(
                np.fft.rfft(snapshots, n=self.schedule.snapshot_length_samples, axis=1),
                dtype=np.complex64,
            )
            # snapshotはchannelごとに異なる中心時刻から切り出されている。
            # `exp(-j2πfΔt)`でbeam内の共通中心時刻へ戻してから、X X^Hを形成する。
            # phase/spectrum shapeはいずれも`[n_ch,n_bin,n_chunk]`である。
            spectrum *= self._time_axis_restoration_phase[
                segment_index,
                :,
                :,
                beam_start:beam_stop,
            ]
            chunk_direction_indices = direction_indices[beam_start:beam_stop]
            steering_power_instantaneous: NDArray[np.float32] | None = None
            total_power_instantaneous: NDArray[np.float32] | None = None
            if self._steering_power_weighting is not None:
                if self.steering_power is None or self.total_power is None:
                    raise RuntimeError("steering power state must exist when steering_table is configured.")
                projection_table = self._steering_power_weighting.projection_table[
                    :, :, chunk_direction_indices
                ]
                # projection^H Xをchannel axis=0で内積する。projectionはshading gを一度含み、
                # total powerにも同じgを使うため、整合targetのeta上限を1に保つ。
                projected = np.einsum(
                    "ikb,ikb->kb",
                    projection_table.conj(),
                    spectrum,
                    optimize=True,
                )
                steering_power_instantaneous = np.asarray(np.abs(projected.T) ** 2, dtype=np.float32)
                # spectrum shapeは`[n_ch,n_bin,n_chunk]`、weight shapeは`[n_ch,n_bin]`。
                # channel axis=0を縮約し、重み付きtotal power `[n_bin,n_chunk]`を得る。
                total_power_instantaneous = np.asarray(
                    np.einsum(
                        "ik,ikb->kb",
                        self._steering_power_weighting.channel_weight_table,
                        np.abs(spectrum) ** 2,
                        optimize=True,
                    ).T,
                    dtype=np.float32,
                )
            instantaneous_covariance = np.asarray(
                np.einsum("ikb,jkb->bijk", spectrum, spectrum.conj(), optimize=True),
                dtype=np.complex64,
            )
            # direction_matchは同一global indexを1秒内に2回含む。advanced indexingの
            # 一括代入では2回目だけが残るため、観測順にR1→R2を更新する。
            for chunk_snapshot_index, global_direction_index_value in enumerate(chunk_direction_indices):
                global_direction_index = int(global_direction_index_value)
                update_coef = self.direction_update_coef[global_direction_index]
                self.direction_covariance[global_direction_index] = np.asarray(
                    (np.float32(1.0) - update_coef) * self.direction_covariance[global_direction_index]
                    + update_coef * instantaneous_covariance[chunk_snapshot_index],
                    dtype=np.complex64,
                )
                if self._steering_power_weighting is not None:
                    # steering/total powerも共分散と同じsnapshot順序と係数で更新し、
                    # 直接積分etaと共分散からのetaの数学的同値性を保つ。
                    steering_power = self.steering_power
                    total_power = self.total_power
                    if (
                        steering_power is None
                        or total_power is None
                        or steering_power_instantaneous is None
                        or total_power_instantaneous is None
                    ):
                        raise RuntimeError("steering power state must exist when steering_table is configured.")
                    steering_power[global_direction_index] = np.asarray(
                        (np.float32(1.0) - update_coef) * steering_power[global_direction_index]
                        + update_coef * steering_power_instantaneous[chunk_snapshot_index],
                        dtype=np.float32,
                    )
                    total_power[global_direction_index] = np.asarray(
                        (np.float32(1.0) - update_coef) * total_power[global_direction_index]
                        + update_coef * total_power_instantaneous[chunk_snapshot_index],
                        dtype=np.float32,
                    )

        self._processed_second_count += 1
        return DirectionMatchedCovarianceUpdate(
            azimuth_segment_index=segment_index,
            global_direction_indices=np.asarray(direction_indices.copy(), dtype=np.int32),
            active_direction_covariance=np.asarray(
                self.direction_covariance[direction_indices].copy(),
                dtype=np.complex64,
            ),
            processed_second_count=self._processed_second_count,
        )

    def completed_steering_metrics(self) -> CompletedDirectionSteeringMetrics:
        """直近2秒周期で完成したetaを返す。

        Returns:
            完成済みsteering power、total power、eta、valid mask、周期数。

        Raises:
            RuntimeError: steering table未設定、初回2秒未完了、または周期途中の場合。

        Notes:
            1秒目の片側方位だけを外部公開しないため、処理秒数が偶数の完成境界でのみ返す。
        """

        if self.steering_power is None or self.total_power is None or self._steering_power_weighting is None:
            raise RuntimeError("steering_table was not configured.")
        if self._processed_second_count < 2 or self._processed_second_count % 2 != 0:
            raise RuntimeError("steering metrics are available only at a completed two-second cycle.")
        valid = (
            np.isfinite(self.steering_power)
            & np.isfinite(self.total_power)
            & (self.total_power > np.float32(self.eta_denominator_floor))
        )
        eta = np.zeros(self.total_power.shape, dtype=np.float32)
        eta[valid] = np.asarray(
            self.steering_power[valid] / self.total_power[valid],
            dtype=np.float32,
        )
        # 理論範囲を丸め誤差だけ超える場合はclipし、大幅超過は実装または入力共分散異常とする。
        if bool(np.any(eta[valid] > 1.0 + 1.0e-4)):
            raise RuntimeError("eta exceeds its physical range.")
        eta = np.clip(eta, 0.0, 1.0).astype(np.float32, copy=False)
        return CompletedDirectionSteeringMetrics(
            steering_power=self.steering_power.copy(),
            total_power=self.total_power.copy(),
            eta=eta,
            eta_valid=np.asarray(valid, dtype=np.bool_),
            active_channel_count=self._steering_power_weighting.active_channel_count.copy(),
            effective_channel_count=self._steering_power_weighting.effective_channel_count.copy(),
            noise_eta_reference=self._steering_power_weighting.noise_eta_reference.copy(),
            completed_cycle_count=self._processed_second_count // 2,
        )

    @property
    def steering_state_bytes(self) -> int:
        """重み付きprojection tableと2個のpower積分状態の常駐byte数を返す。"""

        if self._steering_power_weighting is None:
            steering_bytes = 0
        else:
            # 常駐量にはprojectionだけでなく、runtime監査に公開するshadingと
            # active/N_eff/noise基準表も含め、追加memoryを過小報告しない。
            steering_bytes = int(
                self._steering_power_weighting.projection_table.nbytes
                + self._steering_power_weighting.channel_weight_table.nbytes
                + self._steering_power_weighting.active_channel_count.nbytes
                + self._steering_power_weighting.effective_channel_count.nbytes
                + self._steering_power_weighting.noise_eta_reference.nbytes
            )
        power_bytes = 0 if self.steering_power is None else int(self.steering_power.nbytes)
        power_bytes += 0 if self.total_power is None else int(self.total_power.nbytes)
        return steering_bytes + power_bytes


def build_two_second_covariance_snapshot_schedule(
    sensor_positions_m: NDArray[Any],
    *,
    fs_hz: float,
    sound_speed_m_s: float,
    snapshot_length_samples: int = 128,
    beams_per_half: int = 159,
) -> CovarianceSnapshotCenterSchedule:
    """1秒ごとに切り替える2個の90度中心表と方位一致表を一度だけ作る。

    Args:
        sensor_positions_m: センサ位置。shapeは`[n_ch,3]`、単位はm。
        fs_hz: サンプリング周波数。単位はHz。1秒が整数sampleである必要がある。
        sound_speed_m_s: 音速。単位はm/s。
        snapshot_length_samples: 各beam中心から取得する長さ。単位はsample。
        beams_per_half: 1秒当たりのsnapshot数。片側方位を2回ずつと
            90度を1回含むため、奇数である必要がある。

    Returns:
        `channel_center_samples` shapeが`[2,n_ch,n_beam]`のschedule。

    Raises:
        ValueError: shape、単位、block長、アレイ対称性、またはscale条件が不正な場合。

    境界条件:
        中心の左右にsnapshot半長を確保する。159個のsnapshot区間は互いに重複してよい。
    """

    positions = np.asarray(sensor_positions_m, dtype=np.float32)
    sample_rate = float(fs_hz)
    sound_speed = float(sound_speed_m_s)
    snapshot_length = int(snapshot_length_samples)
    beam_count = int(beams_per_half)
    require_positive_float("fs_hz", sample_rate)
    require_positive_float("sound_speed_m_s", sound_speed)
    require_positive_int("snapshot_length_samples", snapshot_length)
    require_positive_int("beams_per_half", beam_count)
    require(beam_count % 2 == 1, "beams_per_half must be odd so one boundary snapshot remains.")
    require(positions.ndim == 2 and positions.shape[1] == 3, "sensor_positions_m must have shape (n_ch, 3).")
    require(positions.shape[0] > 0, "sensor_positions_m must contain at least one channel.")
    require(bool(np.all(np.isfinite(positions))), "sensor_positions_m must be finite.")
    samples_per_second = int(round(sample_rate))
    require(bool(np.isclose(sample_rate, float(samples_per_second))), "fs_hz must be an integer number of samples per second.")
    require(snapshot_length <= samples_per_second, "snapshot length must fit inside one second.")
    require(
        bool(np.allclose(positions, -np.flip(positions, axis=0), rtol=0.0, atol=1.0e-6)),
        "sensor_positions_m must be centrosymmetric in channel order.",
    )

    # 1秒の159 snapshotは79方位を各2回と90度を1回観測する。
    # local snapshot数とglobal方位数は同じ159だが、対応は一対一ではない。
    direction_count_per_side = (beam_count - 1) // 2
    global_direction_azimuth_deg = np.linspace(0.0, 180.0, beam_count, dtype=np.float32)
    # 同一方位の2観測を隣接させると、到来遅延と時間配置の組合せが台形になる。
    # 秒頭の90度からendfireへ進み、endfireを境に90度へ戻る往復順とする。
    # 次秒頭の90度を閉端として見ると、0.5秒で2個の半台形が接続して長方形になる。
    first_direction_match = np.concatenate(
        (
            np.array([direction_count_per_side], dtype=np.int32),
            np.arange(direction_count_per_side - 1, -1, -1, dtype=np.int32),
            np.arange(direction_count_per_side, dtype=np.int32),
        )
    )
    # centrosymmetric ULAでは反対側方位indexが`158-index`なので、方位表も中心表と
    # 同じ上下反転関係を保つ。90度index 79は両segmentで不変である。
    second_direction_match = np.asarray(beam_count - 1 - first_direction_match, dtype=np.int32)
    first_azimuth_deg = global_direction_azimuth_deg[first_direction_match]
    second_azimuth_deg = global_direction_azimuth_deg[second_direction_match]
    beam_azimuth_deg = np.stack((first_azimuth_deg, second_azimuth_deg), axis=0).astype(np.float32, copy=False)
    direction_match_indices = np.stack((first_direction_match, second_direction_match), axis=0)

    azimuth_rad = np.deg2rad(first_azimuth_deg.astype(np.float64))
    directions = np.stack((np.cos(azimuth_rad), np.sin(azimuth_rad), np.zeros_like(azimuth_rad)), axis=1)
    arrival_delay_s = relative_arrival_delay(positions, directions, sound_speed_m_per_s=sound_speed)
    left_extent = snapshot_length // 2
    right_extent = snapshot_length - left_extent
    # 全channel共通の基準中心だけを1秒内に配置し、到来遅延tauはscaleせず加える。
    # channelごとに行を正規化すると、center[i]-center[j]がfs*(tau[i]-tau[j])
    # と一致せず、長大開口で同一波面の時間区間を切り出せない。
    minimum_delay_s = np.min(arrival_delay_s, axis=0)
    maximum_delay_s = np.max(arrival_delay_s, axis=0)
    feasible_lower_s = left_extent / sample_rate - minimum_delay_s
    feasible_upper_s = 1.0 - right_extent / sample_rate - maximum_delay_s
    require(
        bool(np.all(feasible_upper_s >= feasible_lower_s)),
        "array delay aperture and snapshot length must fit inside one second without scaling delays.",
    )
    # q/159は次秒頭q=159を閉端とする半開区間の時間比である。各候補方位固有の
    # 可行下端・上端をこの比率で補間するため、物理遅延を保ったまま0--1秒を充填できる。
    time_fraction = np.arange(beam_count, dtype=np.float64) / float(beam_count)
    base_center_s = (
        (1.0 - time_fraction) * feasible_lower_s
        + time_fraction * feasible_upper_s
    )
    first_centers = np.rint(sample_rate * (base_center_s[np.newaxis, :] + arrival_delay_s)).astype(
        np.int32
    )
    second_centers = np.flip(first_centers, axis=0)
    channel_center_samples = np.stack((first_centers, second_centers), axis=0).astype(np.int32, copy=False)

    snapshot_start = channel_center_samples - np.int32(left_extent)
    snapshot_stop = channel_center_samples + np.int32(right_extent)
    require(bool(np.all(snapshot_start >= 0)), "all snapshot starts must remain inside one second.")
    require(bool(np.all(snapshot_stop <= samples_per_second)), "all snapshot stops must remain inside one second.")

    return CovarianceSnapshotCenterSchedule(
        beam_azimuth_deg=beam_azimuth_deg,
        global_direction_azimuth_deg=global_direction_azimuth_deg,
        direction_match_indices=direction_match_indices,
        channel_center_samples=channel_center_samples,
        fs_hz=sample_rate,
        snapshot_length_samples=snapshot_length,
    )
