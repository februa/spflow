"""active channel subsetの単一周波数で方位別共分散とsteering powerを積分する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float, require_positive_int
from .covariance_snapshot_schedule import CovarianceSnapshotCenterSchedule
from .steering_power_weighting import prepare_steering_power_channel_weighting


@dataclass(frozen=True)
class SelectedFrequencyDirectionCovarianceResult:
    """単一周波数・全方位の完成共分散とsteering整合量を保持する。

    Attributes:
        frequency_bin_index: 元のrFFTにおける対象bin index。
        frequency_hz: 対象周波数。単位はHz。
        direction_covariance: 方位別共分散。shapeは`[n_direction,n_active_ch,n_active_ch]`。
        steering_power: 方位別`|u_g^H X|^2`指数積分。shapeは`[n_direction]`。
        total_power: 方位別`sum(g|X|^2)`指数積分。shapeは`[n_direction]`。
        eta: steering power比。shapeは`[n_direction]`、範囲は`[0,1]`。
        eta_valid: eta分母が有効なmask。shapeは`[n_direction]`。
        effective_channel_count: shadingを含む`N_eff`。
        noise_eta_reference: 空間白色雑音の`1/N_eff`。
        completed_cycle_count: 完成した2秒周期数。
    """

    frequency_bin_index: int
    frequency_hz: float
    direction_covariance: NDArray[np.complex64]
    steering_power: NDArray[np.float32]
    total_power: NDArray[np.float32]
    eta: NDArray[np.float32]
    eta_valid: NDArray[np.bool_]
    effective_channel_count: float
    noise_eta_reference: float
    completed_cycle_count: int


class SelectedFrequencyDirectionCovarianceAccumulator:
    """active subsetの対象1binだけを保持し、方位別共分散を2秒周期で完成させる。

    入力は1秒実信号`[n_active_ch,fs]`、出力は対象周波数の全方位共分散とetaである。
    対象外bin、threshold校正、方位Weight合成、MVDR設計は責務に含めない。

    信号処理上は、周波数別shadingで未使用のchannelとbinを保持せず、305ch運用アレイの
    全bin共分散が要求する巨大memoryを避ける厳密なactive subset表現である。
    """

    def __init__(
        self,
        schedule: CovarianceSnapshotCenterSchedule,
        steering_table: NDArray[Any],
        channel_weights: NDArray[Any],
        *,
        frequency_bin_index: int,
        coef: float | None = None,
        integration_time_seconds: float | None = None,
        beam_chunk_size: int = 8,
        denominator_floor: float = 1.0e-20,
    ) -> None:
        """対象bin、steering、shading、指数積分係数を設定する。

        Args:
            schedule: active channel座標から生成した2秒中心sample表。
            steering_table: 対象周波数の物理steering。shapeは`[n_active_ch,n_direction]`。
            channel_weights: active channelの非負shading。shapeは`[n_active_ch]`。
            frequency_bin_index: `NFFT` rFFTの対象bin index。
            coef: 全方位共通更新係数。積分時間との同時指定不可。
            integration_time_seconds: 方位更新rateを含む等価積分時間。単位はs。
            beam_chunk_size: 一括処理するsegment内beam数。
            denominator_floor: eta分母の数値下限。

        Raises:
            ValueError: shape、bin、係数、shading、steeringが不正な場合。

        境界条件:
            2秒完成前は結果を公開しない。非選択方位は減衰させず、次の同一方位更新まで保持する。
        """

        require((coef is None) != (integration_time_seconds is None), "specify exactly one integration coefficient.")
        self.schedule = schedule
        self.frequency_bin_index = int(frequency_bin_index)
        self.beam_chunk_size = int(beam_chunk_size)
        self.denominator_floor = float(denominator_floor)
        require_positive_int("beam_chunk_size", self.beam_chunk_size)
        require_positive_float("denominator_floor", self.denominator_floor)
        n_bin_full = schedule.snapshot_length_samples // 2 + 1
        require(0 <= self.frequency_bin_index < n_bin_full, "frequency_bin_index is outside rFFT bins.")
        self.frequency_hz = float(self.frequency_bin_index * schedule.fs_hz / schedule.snapshot_length_samples)
        n_direction = int(schedule.global_direction_azimuth_deg.size)
        steering = np.asarray(steering_table, dtype=np.complex64)
        weights = np.asarray(channel_weights, dtype=np.float32)
        require(steering.shape == (schedule.n_ch, n_direction), "steering_table must have shape (n_ch, n_direction).")
        require(weights.shape == (schedule.n_ch,), "channel_weights must have shape (n_ch,).")
        weighting = prepare_steering_power_channel_weighting(
            steering[:, np.newaxis, :],
            weights[:, np.newaxis],
        )
        self._projection = weighting.projection_table[:, 0, :]
        self._channel_weights = weighting.channel_weight_table[:, 0]
        self.effective_channel_count = float(weighting.effective_channel_count[0])
        self.noise_eta_reference = float(weighting.noise_eta_reference[0])

        if integration_time_seconds is not None:
            integration_time = float(integration_time_seconds)
            require_positive_float("integration_time_seconds", integration_time)
            update_count = np.bincount(
                schedule.direction_match_indices.reshape(-1),
                minlength=n_direction,
            ).astype(np.float32)
            update_rate_hz = update_count / np.float32(2.0)
            self._coef = np.minimum(
                np.float32(2.0) / (np.float32(1.0) + update_rate_hz * np.float32(integration_time)),
                np.float32(1.0),
            ).astype(np.float32)
        else:
            update_coef = float(coef) if coef is not None else 0.0
            require(0.0 < update_coef <= 1.0, "coef must satisfy 0 < coef <= 1.")
            self._coef = np.full(n_direction, update_coef, dtype=np.float32)

        # full restoration phase `[2,ch,bin,beam]`から対象binだけを初期化時に固定する。
        self._restoration_phase = np.stack(
            [
                schedule.calculate_time_axis_restoration_phase(azimuth_segment_index=segment)[:, self.frequency_bin_index, :]
                for segment in (0, 1)
            ],
            axis=0,
        ).astype(np.complex64)
        self._covariance = np.zeros((n_direction, schedule.n_ch, schedule.n_ch), dtype=np.complex64)
        self._steering_power = np.zeros(n_direction, dtype=np.float32)
        self._total_power = np.zeros(n_direction, dtype=np.float32)
        self._processed_second_count = 0

    def process_one_second(self, signal: NDArray[Any]) -> None:
        """1秒のactive channel信号から対象binの159方位を更新する。

        Args:
            signal: 1秒実信号。shapeは`[n_active_ch,fs]`、axis 0はchannel、axis 1はsample。

        Returns:
            戻り値なし。2秒完成境界では`completed_result`から結果を取得する。

        Raises:
            ValueError: 入力が複素数、shape不一致、非有限の場合。

        境界条件:
            例外時は呼出側がresetまたはインスタンス破棄する。途中結果を完成値として公開しない。
        """

        input_signal = np.asarray(signal)
        require(not np.iscomplexobj(input_signal), "signal must be real-valued.")
        require(input_signal.shape == (self.schedule.n_ch, int(round(self.schedule.fs_hz))), "signal must contain one second.")
        require(bool(np.all(np.isfinite(input_signal))), "signal must be finite.")
        segment = self._processed_second_count % 2
        direction_indices = self.schedule.direction_match_indices[segment]
        for beam_start in range(0, self.schedule.n_beam, self.beam_chunk_size):
            beam_stop = min(beam_start + self.beam_chunk_size, self.schedule.n_beam)
            snapshots = self.schedule.extract_snapshot_chunk(
                input_signal,
                azimuth_segment_index=segment,
                beam_start_index=beam_start,
                beam_stop_index=beam_stop,
            )
            # snapshot `[ch,time,beam]`をrFFTし、対象binだけの`X[ch,beam]`を保持する。
            spectrum = np.asarray(
                np.fft.rfft(snapshots, n=self.schedule.snapshot_length_samples, axis=1)[
                    :, self.frequency_bin_index, :
                ],
                dtype=np.complex64,
            )
            spectrum *= self._restoration_phase[segment, :, beam_start:beam_stop]
            active_directions = direction_indices[beam_start:beam_stop]
            projection = self._projection[:, active_directions]
            projected = np.einsum("ib,ib->b", projection.conj(), spectrum, optimize=True)
            steering_instantaneous = np.asarray(np.abs(projected) ** 2, dtype=np.float32)
            total_instantaneous = np.asarray(
                np.einsum("i,ib->b", self._channel_weights, np.abs(spectrum) ** 2, optimize=True),
                dtype=np.float32,
            )
            covariance_instantaneous = np.asarray(
                np.einsum("ib,jb->bij", spectrum, spectrum.conj(), optimize=True),
                dtype=np.complex64,
            )
            # 1秒内に同一global方位が2回現れる。advanced indexingでは
            # 2回更新にならないため、snapshot順にEMAを適用する。
            for chunk_snapshot_index, global_direction_index_value in enumerate(active_directions):
                global_direction_index = int(global_direction_index_value)
                update_coef = self._coef[global_direction_index]
                self._steering_power[global_direction_index] = np.asarray(
                    (np.float32(1.0) - update_coef) * self._steering_power[global_direction_index]
                    + update_coef * steering_instantaneous[chunk_snapshot_index],
                    dtype=np.float32,
                )
                self._total_power[global_direction_index] = np.asarray(
                    (np.float32(1.0) - update_coef) * self._total_power[global_direction_index]
                    + update_coef * total_instantaneous[chunk_snapshot_index],
                    dtype=np.float32,
                )
                self._covariance[global_direction_index] = np.asarray(
                    (np.float32(1.0) - update_coef) * self._covariance[global_direction_index]
                    + update_coef * covariance_instantaneous[chunk_snapshot_index],
                    dtype=np.complex64,
                )
        self._processed_second_count += 1

    def completed_result(self) -> SelectedFrequencyDirectionCovarianceResult:
        """直近2秒周期で完成した全方位結果を返す。

        Returns:
            対象周波数の方位別共分散、power、eta、`N_eff`を持つcopy。

        Raises:
            RuntimeError: 初回2秒未完了、または1秒segment途中の場合。

        境界条件:
            total powerがfloor以下の方位は`eta_valid=False`とし、eta値0を検出値として扱わない。
        """

        if self._processed_second_count < 2 or self._processed_second_count % 2 != 0:
            raise RuntimeError("result is available only at a completed two-second cycle.")
        valid = np.isfinite(self._steering_power) & np.isfinite(self._total_power) & (
            self._total_power > np.float32(self.denominator_floor)
        )
        eta = np.zeros(self._total_power.shape, dtype=np.float32)
        eta[valid] = np.asarray(self._steering_power[valid] / self._total_power[valid], dtype=np.float32)
        if bool(np.any(eta[valid] > 1.0 + 1.0e-4)):
            raise RuntimeError("eta exceeds its physical range.")
        return SelectedFrequencyDirectionCovarianceResult(
            frequency_bin_index=self.frequency_bin_index,
            frequency_hz=self.frequency_hz,
            direction_covariance=self._covariance.copy(),
            steering_power=self._steering_power.copy(),
            total_power=self._total_power.copy(),
            eta=np.clip(eta, 0.0, 1.0).astype(np.float32),
            eta_valid=np.asarray(valid, dtype=np.bool_),
            effective_channel_count=self.effective_channel_count,
            noise_eta_reference=self.noise_eta_reference,
            completed_cycle_count=self._processed_second_count // 2,
        )
