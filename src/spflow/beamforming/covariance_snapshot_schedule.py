"""方位別遅延整合共分散で再利用する中心サンプル表を構成するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float, require_positive_int
from .geometry import relative_arrival_delay


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


class DirectionMatchedCovarianceAccumulator:
    """1秒ごとに中心表と方位一致表を切り替え、同一方位だけを指数積分する。

    偶数秒は0–90度、奇数秒は180–90度の159 snapshotを使う。各beamは1秒につき
    1 snapshotだけを持ち、同じglobal方位slotへ到来する次回snapshotと指数積分する。

    入力は1秒信号`[n_ch,fs]`、出力は今回更新した159方位の共分散である。
    相関閾値による採否、beam方向集約、MVDR重み設計は責務に含めない。
    """

    def __init__(self, schedule: CovarianceSnapshotCenterSchedule, *, coef: float) -> None:
        """再利用するscheduleと瞬時共分散の更新係数を設定する。

        Args:
            schedule: 中心表と方位一致表。frame間で同じインスタンスを再利用する。
            coef: `R_next=(1-coef)R_previous+coef*R_instantaneous`の更新係数。

        Raises:
            ValueError: `coef`が`(0,1]`外の場合。
        """

        update_coef = float(coef)
        require(0.0 < update_coef <= 1.0, "coef must satisfy 0 < coef <= 1.")
        self.schedule = schedule
        self.coef = update_coef
        self.n_bin = schedule.snapshot_length_samples // 2 + 1
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
        self._processed_second_count = 0

    def process_one_second(self, signal: NDArray[Any]) -> DirectionMatchedCovarianceUpdate:
        """現在秒の159方位について、各1 snapshotで共分散を更新する。

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
        snapshots = self.schedule.extract_snapshots(
            input_signal,
            azimuth_segment_index=segment_index,
        )
        # snapshots `[n_ch,N,n_beam]`をaxis=1でrFFTし、`X[ch,bin,beam]`へ変換する。
        spectrum = np.asarray(
            np.fft.rfft(snapshots, n=self.schedule.snapshot_length_samples, axis=1),
            dtype=np.complex64,
        )
        # beamごとの瞬時共分散`R[beam,i,j,k]=X[i,k,beam]conj(X[j,k,beam])`を作る。
        instantaneous_covariance = np.asarray(
            np.einsum("ikb,jkb->bijk", spectrum, spectrum.conj(), optimize=True),
            dtype=np.complex64,
        )
        previous_covariance = self.direction_covariance[direction_indices]
        updated_covariance = np.asarray(
            (1.0 - self.coef) * previous_covariance + self.coef * instantaneous_covariance,
            dtype=np.complex64,
        )
        self.direction_covariance[direction_indices] = updated_covariance

        self._processed_second_count += 1
        return DirectionMatchedCovarianceUpdate(
            azimuth_segment_index=segment_index,
            global_direction_indices=np.asarray(direction_indices.copy(), dtype=np.int32),
            active_direction_covariance=updated_covariance.copy(),
            processed_second_count=self._processed_second_count,
        )


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
        beams_per_half: 90度当たりのbeam数。

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

    first_azimuth_deg = np.linspace(0.0, 90.0, beam_count, dtype=np.float32)
    second_azimuth_deg = 180.0 - first_azimuth_deg
    beam_azimuth_deg = np.stack((first_azimuth_deg, second_azimuth_deg), axis=0).astype(np.float32, copy=False)
    global_direction_azimuth_deg = np.linspace(0.0, 180.0, 2 * beam_count - 1, dtype=np.float32)
    first_direction_match = np.arange(beam_count, dtype=np.int32)
    second_direction_match = np.arange(2 * beam_count - 2, beam_count - 2, -1, dtype=np.int32)
    direction_match_indices = np.stack((first_direction_match, second_direction_match), axis=0)

    azimuth_rad = np.deg2rad(first_azimuth_deg.astype(np.float64))
    directions = np.stack((np.cos(azimuth_rad), np.sin(azimuth_rad), np.zeros_like(azimuth_rad)), axis=1)
    arrival_delay_s = relative_arrival_delay(positions, directions, sound_speed_m_per_s=sound_speed)
    progress_s = np.linspace(0.0, 1.0, beam_count, dtype=np.float64)
    raw_first_centers_s = progress_s[np.newaxis, :] + arrival_delay_s
    raw_start_s = raw_first_centers_s[:, :1]
    raw_span_s = raw_first_centers_s[:, -1:] - raw_start_s
    require(bool(np.all(raw_span_s > 0.0)), "raw center rows must increase from 0 to 90 degrees.")

    left_extent = snapshot_length // 2
    right_extent = snapshot_length - left_extent
    usable_samples = samples_per_second - left_extent - right_extent
    normalized_progress = (raw_first_centers_s - raw_start_s) / raw_span_s
    first_centers = np.rint(left_extent + usable_samples * normalized_progress).astype(np.int32)
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
