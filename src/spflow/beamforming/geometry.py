"""アレイ座標、到来方向、相対遅延、steering の幾何変換を実装する。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float


def unit_direction_from_positions(
    receiver_position: NDArray[Any],
    source_position: NDArray[Any],
) -> NDArray[np.float64]:
    """receiver から source へ向かう単位方向ベクトルを計算する。

    Args:
        receiver_position: receiver 基準位置。shape は `[3]`、単位は m。
        source_position: source 位置。shape は `[3]`、単位は m。

    Returns:
        receiver から source へ向かう単位ベクトル。shape は `[3]`。

    Raises:
        ValueError: shape、有限性が不正、または二点が同一位置の場合。

    境界条件:
        source と receiver が同一点では到来方向を定義できないため fallback は行わず例外にする。
    """
    receiver = np.asarray(receiver_position, dtype=np.float64)
    source = np.asarray(source_position, dtype=np.float64)
    require(receiver.shape == (3,), "receiver_position must have shape (3,).")
    require(source.shape == (3,), "source_position must have shape (3,).")
    require(bool(np.all(np.isfinite(receiver))), "receiver_position must be finite.")
    require(bool(np.all(np.isfinite(source))), "source_position must be finite.")
    displacement = source - receiver
    distance_m = float(np.linalg.norm(displacement))
    require(distance_m > 0.0, "source_position must differ from receiver_position.")
    return np.asarray(displacement / distance_m, dtype=np.float64)


def relative_arrival_delay(
    sensor_positions_m: NDArray[Any],
    arrival_direction: NDArray[Any],
    *,
    sound_speed_m_per_s: float,
) -> NDArray[np.float64]:
    """センサ位置と到来方向から基準点に対する相対到達遅延を計算する。

    Args:
        sensor_positions_m: 基準点相対のセンサ位置。shape は `[n_channel, 3]`、単位は m。
        arrival_direction: source 方向を向く単位ベクトル。shape は `[3]` または
            `[n_direction, 3]`。各 row のノルムは 1。
        sound_speed_m_per_s: 伝搬速度。単位は m/s。

    Returns:
        相対遅延 `tau = position dot direction / c`。単一方向では shape `[n_channel]`、
        複数方向では `[n_channel, n_direction]`、単位は s。

    Raises:
        ValueError: shape、有限性、単位方向、伝搬速度が不正な場合。

    境界条件:
        原点センサの遅延は 0 s となる。符号は source 方向への位置投影が正のセンサほど
        到来が早いという幾何量を表し、補償位相の符号は steering 関数で明示する。
    """
    positions = np.asarray(sensor_positions_m, dtype=np.float64)
    directions = np.asarray(arrival_direction, dtype=np.float64)
    sound_speed = float(sound_speed_m_per_s)
    require_positive_float("sound_speed_m_per_s", sound_speed)
    require(
        positions.ndim == 2 and positions.shape[1] == 3,
        "sensor_positions_m must have shape (n_channel, 3).",
    )
    require(positions.shape[0] > 0, "sensor_positions_m must contain at least one channel.")
    require(bool(np.all(np.isfinite(positions))), "sensor_positions_m must be finite.")
    require(
        directions.ndim in (1, 2), "arrival_direction must have shape (3,) or (n_direction, 3)."
    )
    require(directions.shape[-1] == 3, "arrival_direction last axis must have length 3.")
    require(bool(np.all(np.isfinite(directions))), "arrival_direction must be finite.")
    norms = np.linalg.norm(directions, axis=-1)
    require(
        bool(np.allclose(norms, 1.0, rtol=1.0e-7, atol=1.0e-9)),
        "arrival_direction rows must be unit vectors.",
    )

    # position・direction の内積は source 方向に沿った距離差 [m] であり、
    # sound speed [m/s] で割ることで基準点に対する相対到達時間差 [s] になる。
    if directions.ndim == 1:
        return np.asarray(positions @ directions / sound_speed, dtype=np.float64)
    return np.asarray(positions @ directions.T / sound_speed, dtype=np.float64)


def steering_from_relative_delay(
    relative_delay_s: NDArray[Any],
    frequency_hz: NDArray[Any],
    *,
    phase_sign: int = -1,
) -> NDArray[np.complex128]:
    """相対遅延を周波数領域の steering 位相へ変換する。

    Args:
        relative_delay_s: 相対到達遅延。shape は `[n_channel]` または
            `[n_channel, n_direction]`、単位は s。
        frequency_hz: 周波数軸。shape は `[n_frequency]`、単位は Hz。
        phase_sign: Fourier convention に対応する符号。`-1` は
            `exp(-j 2 pi f tau)`、`+1` はその共役を生成する。

    Returns:
        steering。入力遅延が `[n_channel]` なら shape `[n_channel, n_frequency]`、
        `[n_channel, n_direction]` なら `[n_channel, n_direction, n_frequency]`。

    Raises:
        ValueError: shape、有限性、周波数、または phase sign が不正な場合。

    境界条件:
        0 Hz または 0 s の位相は必ず `1+0j` になる。phase sign を暗黙に推定せず、
        既存実装の Fourier convention と対応付けて指定する。
    """
    delays = np.asarray(relative_delay_s, dtype=np.float64)
    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    require(
        delays.ndim in (1, 2),
        "relative_delay_s must have shape (n_channel,) or (n_channel, n_direction).",
    )
    require(delays.shape[0] > 0, "relative_delay_s must contain at least one channel.")
    require(
        frequencies.ndim == 1 and frequencies.size > 0,
        "frequency_hz must have shape (n_frequency,).",
    )
    require(bool(np.all(np.isfinite(delays))), "relative_delay_s must be finite.")
    require(bool(np.all(np.isfinite(frequencies))), "frequency_hz must be finite.")
    require(bool(np.all(frequencies >= 0.0)), "frequency_hz must be non-negative.")
    require(int(phase_sign) in (-1, 1), "phase_sign must be -1 or +1.")

    # exp(sign*j*2πfτ) は時間差 τ を各周波数 bin の位相差へ写像する。
    # delay 軸の末尾へ frequency 軸を追加し、channel/direction ごとに broadcast する。
    phase = float(phase_sign) * 2.0 * np.pi * delays[..., np.newaxis] * frequencies
    return np.asarray(np.exp(1j * phase), dtype=np.complex128)
