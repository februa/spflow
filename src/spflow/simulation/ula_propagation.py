"""ULAの幾何遅延と周波数領域ステアリングの計算。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
ComplexArray = NDArray[np.complexfloating[Any, Any]]


def calculate_ula_arrival_delays_s(
    sensor_positions_m: FloatArray,
    azimuth_deg: FloatArray,
    sound_speed_m_per_s: float,
) -> FloatArray:
    """ULA軸位置と方位から基準点に対する到来遅延を計算する。

    Args:
        sensor_positions_m: ULA軸位置。shape ``[n_ch]``、単位m。
        azimuth_deg: 方位。shape ``[n_direction]``、単位deg。0/180 degがendfire。
        sound_speed_m_per_s: 伝搬速度、単位m/s。

    Returns:
        到来遅延。shape ``[n_direction,n_ch]``、単位s。

    Raises:
        ValueError: 入力が1次元でない、非有限、空、または音速が正でない場合。
    """
    input_dtype = np.result_type(sensor_positions_m, azimuth_deg)
    real_dtype = (
        np.dtype(np.float32) if input_dtype == np.dtype(np.float32) else np.dtype(np.float64)
    )
    positions = np.asarray(sensor_positions_m, dtype=real_dtype)
    azimuths = np.asarray(azimuth_deg, dtype=real_dtype)
    if positions.ndim != 1 or positions.size == 0 or azimuths.ndim != 1 or azimuths.size == 0:
        raise ValueError("positions and azimuths must be non-empty 1-D arrays.")
    if not bool(np.all(np.isfinite(positions))) or not bool(np.all(np.isfinite(azimuths))):
        raise ValueError("positions and azimuths must be finite.")
    if sound_speed_m_per_s <= 0.0:
        raise ValueError("sound_speed_m_per_s must be positive.")
    # tau=-r cos(theta)/c。axis=0は方位、axis=1はchannelを表す。
    return np.asarray(
        -np.cos(np.deg2rad(azimuths))[:, None] * positions[None, :] / sound_speed_m_per_s,
        dtype=real_dtype,
    )


def calculate_frequency_steering(delays_s: FloatArray, frequencies_hz: FloatArray) -> ComplexArray:
    """到来遅延を周波数領域の未正規化steeringへ変換する。

    Args:
        delays_s: 到来遅延。shape ``[n_direction,n_ch]``、単位s。
        frequencies_hz: DFT周波数。shape ``[n_frequency]``、単位Hz。

    Returns:
        steering。shape ``[n_frequency,n_direction,n_ch]``。

    Raises:
        ValueError: delaysが2次元でない、周波数が1次元でない、または非有限の場合。
    """
    delays = np.asarray(delays_s)
    frequencies = np.asarray(frequencies_hz)
    if delays.ndim != 2 or frequencies.ndim != 1 or 0 in delays.shape or frequencies.size == 0:
        raise ValueError("delays_s and frequencies_hz must be 2-D and 1-D respectively.")
    if not bool(np.all(np.isfinite(delays))) or not bool(np.all(np.isfinite(frequencies))):
        raise ValueError("delays_s and frequencies_hz must be finite.")
    combined_dtype = np.result_type(delays, frequencies)
    complex_dtype = (
        np.dtype(np.complex64)
        if combined_dtype == np.dtype(np.float32)
        else np.dtype(np.complex128)
    )
    # a(f,theta,ch)=exp(-j2πf tau)。axisは周波数、方位、channelの順である。
    return np.asarray(
        np.exp(-1j * 2.0 * np.pi * frequencies[:, None, None] * delays[None, :, :]),
        dtype=complex_dtype,
    )
