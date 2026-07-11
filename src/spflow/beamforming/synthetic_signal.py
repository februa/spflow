"""幾何遅延と level 規約を明示した決定論的なアレイ信号生成を実装する。"""

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
from ..spectral_level import tone_rms_level_db_to_peak_amplitude
from .geometry import relative_arrival_delay


@dataclass(frozen=True)
class PlaneWaveTone:
    """単一平面波 tone の生成結果を保持する。

    Attributes:
        signal: 実数 channel 波形。shape は `[n_channel, n_sample]`。
        time_s: 基準点の時刻軸。shape は `[n_sample]`、単位は s。
        relative_delay_s: 基準点に対する相対到達遅延。shape は `[n_channel]`、単位は s。
        frequency_hz: tone 周波数。単位は Hz。
        level_db_re_rms: 各 channel の tone RMS level。単位は `dB re reference RMS`。

    このクラスは生成済み信号と真値を保持するが、beamforming、雑音付加、評価判定は責務に含めない。
    信号処理上は、アレイ幾何と beamforming の間で位相・level 規約を検証する基準入力である。
    """

    signal: NDArray[np.float64]
    time_s: NDArray[np.float64]
    relative_delay_s: NDArray[np.float64]
    frequency_hz: float
    level_db_re_rms: float


def synthesize_plane_wave_tone(
    sensor_positions_m: NDArray[Any],
    arrival_direction: NDArray[Any],
    *,
    sound_speed_m_per_s: float,
    sampling_frequency_hz: float,
    sample_count: int,
    frequency_hz: float,
    level_db_re_rms: float,
    initial_phase_rad: float = 0.0,
) -> PlaneWaveTone:
    """指定方向から到来する実平面波 tone を channel ごとに生成する。

    Args:
        sensor_positions_m: 基準点相対のセンサ位置。shape は `[n_channel, 3]`、単位は m。
        arrival_direction: receiver から source へ向く単位ベクトル。shape は `[3]`。
        sound_speed_m_per_s: 伝搬速度。単位は m/s。
        sampling_frequency_hz: sampling frequency。単位は Hz。
        sample_count: 生成サンプル数。単位は sample。
        frequency_hz: tone 周波数。単位は Hz。DC は許可せず、Nyquist 未満とする。
        level_db_re_rms: 各 channel の tone RMS level。単位は `dB re reference RMS`。
        initial_phase_rad: 基準点で時刻 0 に与える初期位相。単位は rad。

    Returns:
        波形、時刻軸、相対遅延、tone 条件を持つ `PlaneWaveTone`。

    Raises:
        ValueError: shape、単位、周波数範囲、level、位相が不正な場合。

    境界条件:
        `signal[ch,n] = A_peak cos(2πf(t[n]-tau[ch]) + phase)` とする。
        source に近いセンサの `tau` は負なので波形は早着し、FFT 位相は
        `phase - 2πf tau[ch]` になる。有限長区間の RMS を指定値へ後補正しないため、
        level の厳密確認には整数周期を含む sample_count を選ぶ。
    """
    fs_hz = float(sampling_frequency_hz)
    tone_hz = float(frequency_hz)
    phase_rad = float(initial_phase_rad)
    require_positive_float("sound_speed_m_per_s", float(sound_speed_m_per_s))
    require_positive_float("sampling_frequency_hz", fs_hz)
    require_positive_int("sample_count", int(sample_count))
    require_non_negative_float("frequency_hz", tone_hz)
    require(tone_hz > 0.0, "frequency_hz must be greater than DC.")
    require(tone_hz < 0.5 * fs_hz, "frequency_hz must be below Nyquist.")
    require(bool(np.isfinite(float(level_db_re_rms))), "level_db_re_rms must be finite.")
    require(bool(np.isfinite(phase_rad)), "initial_phase_rad must be finite.")

    relative_delay_s = relative_arrival_delay(
        sensor_positions_m,
        arrival_direction,
        sound_speed_m_per_s=float(sound_speed_m_per_s),
    )
    require(relative_delay_s.ndim == 1, "arrival_direction must describe exactly one direction.")
    time_s = np.arange(int(sample_count), dtype=np.float64) / fs_hz
    peak_amplitude = tone_rms_level_db_to_peak_amplitude(float(level_db_re_rms))
    # phase shape: [n_channel, n_sample]。axis=0 は sensor channel、axis=1 は時刻 sample。
    # t-tau により正の delay は遅着、負の delay は早着として波形位相へ反映する。
    phase = (
        2.0 * np.pi * tone_hz * (time_s[np.newaxis, :] - relative_delay_s[:, np.newaxis])
        + phase_rad
    )
    signal = peak_amplitude * np.cos(phase)
    return PlaneWaveTone(
        signal=np.asarray(signal, dtype=np.float64),
        time_s=time_s,
        relative_delay_s=relative_delay_s,
        frequency_hz=tone_hz,
        level_db_re_rms=float(level_db_re_rms),
    )
