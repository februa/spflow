"""整相シミュレーションの責務別部品を単独で検証する。"""

from __future__ import annotations

import numpy as np
import pytest

from spflow.simulation.alignment_coordinates import to_original_input_coordinates
from spflow.simulation.alignment_covariance import calculate_alignment_source_covariance
from spflow.simulation.frequency_weight_fir import approximate_frequency_weights_with_fir
from spflow.simulation.ula_propagation import (
    calculate_frequency_steering,
    calculate_ula_arrival_delays_s,
)


def test_ula_propagation_keeps_geometry_and_frequency_axes_explicit() -> None:
    """ULA遅延とsteeringが方位・channel・周波数軸の契約を保つことを確認する。"""
    positions_m = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    azimuth_deg = np.asarray([0.0, 90.0], dtype=np.float32)
    frequencies_hz = np.asarray([0.0, 100.0], dtype=np.float32)

    delays_s = calculate_ula_arrival_delays_s(positions_m, azimuth_deg, 1000.0)
    steering = calculate_frequency_steering(delays_s, frequencies_hz)

    assert delays_s.shape == (2, 3)
    assert steering.shape == (2, 2, 3)
    assert delays_s.dtype == np.dtype(np.float32)
    assert steering.dtype == np.dtype(np.complex64)
    # broadsideではULA軸への射影が0になるため、全channelの相対遅延も0になる。
    np.testing.assert_allclose(delays_s[1], 0.0, atol=1.0e-9)
    # DCは遅延によらずexp(0)=1であり、位相規約の基準点となる。
    np.testing.assert_allclose(steering[0], 1.0 + 0.0j, atol=0.0)


def test_alignment_covariance_can_be_used_without_weight_design_config() -> None:
    """共分散部品が重み設計configなしでnoise-only共分散を構築できることを確認する。"""
    delays_s = np.asarray([-0.001, 0.0, 0.001], dtype=np.float64)
    steering = np.ones(3, dtype=np.complex128)
    covariance = calculate_alignment_source_covariance(
        delays_s,
        steering,
        fs_hz=1000.0,
        analysis_width_hz=20.0,
        noise_power_per_bin_re_input_rms2=0.25,
        candidate_delay_s=None,
        source_power=0.0,
    )

    # source power 0では空間白色noiseだけが残り、R=σ²Iへ厳密に帰着する。
    np.testing.assert_allclose(covariance, 0.25 * np.eye(3), atol=0.0)
    assert covariance.dtype == np.dtype(np.complex128)


def test_frequency_fir_and_coordinate_transform_are_independently_composable() -> None:
    """FIR近似と座標変換が重み設計器なしで直列利用できることを確認する。"""
    weights = np.ones((8, 2, 3), dtype=np.complex64)
    integer_phase = np.full(weights.shape, 1.0j, dtype=np.complex64)
    approximation = approximate_frequency_weights_with_fir(weights, tap_count=8)
    original = to_original_input_coordinates(
        "T2a", approximation.reconstructed_weights, integer_phase
    )

    np.testing.assert_allclose(approximation.reconstructed_weights, weights, atol=1.0e-6)
    # D=jならD^H=-jであり、元入力座標の等価重みは-j倍になる。
    np.testing.assert_allclose(original, -1.0j * weights, atol=1.0e-6)
    with pytest.raises(ValueError, match="unknown alignment method"):
        to_original_input_coordinates("unknown", weights, integer_phase)
