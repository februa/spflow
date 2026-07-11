"""到来方向、相対遅延、steering の共通幾何変換を検証する。"""

from __future__ import annotations

import numpy as np

from spflow.beamforming.geometry import (
    relative_arrival_delay,
    steering_from_relative_delay,
    unit_direction_from_positions,
)


def test_direction_delay_and_steering_follow_declared_fourier_convention() -> None:
    """位置差から `exp(-j2πfτ)` までの符号、shape、単位対応を固定する。"""
    direction = unit_direction_from_positions(
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 4.0, 0.0]),
    )
    positions_m = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0]], dtype=np.float64)
    delays_s = relative_arrival_delay(
        positions_m,
        direction,
        sound_speed_m_per_s=1.0,
    )
    frequencies_hz = np.array([0.0, 0.5], dtype=np.float64)
    steering = steering_from_relative_delay(delays_s, frequencies_hz, phase_sign=-1)

    np.testing.assert_allclose(direction, np.array([0.6, 0.8, 0.0]))
    np.testing.assert_allclose(delays_s, np.array([0.0, -0.5]))
    assert steering.shape == (2, 2)
    np.testing.assert_allclose(steering[:, 0], np.ones(2, dtype=np.complex128))
    np.testing.assert_allclose(steering[1, 1], 1j, atol=1.0e-15)


def test_multiple_direction_delays_keep_channel_and_direction_axes() -> None:
    """複数方向入力で `[channel, direction, frequency]` の軸順を保つことを確認する。"""
    positions_m = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    delays_s = relative_arrival_delay(
        positions_m,
        directions,
        sound_speed_m_per_s=2.0,
    )

    steering = steering_from_relative_delay(delays_s, np.array([1.0]))

    assert delays_s.shape == (2, 2)
    assert steering.shape == (2, 2, 1)
    np.testing.assert_allclose(delays_s[1], np.array([-0.5, 0.0]))
