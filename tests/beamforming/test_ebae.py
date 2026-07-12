"""EBAE の N/E AIC、MUSIC、固有mode除外重みを検証する。"""

from __future__ import annotations

import numpy as np

from spflow.beamforming.ebae import (
    EbaeConfig,
    calculate_music_spectrum,
    design_ebae_weights,
    design_ebae_weights_band,
    estimate_signal_count_ne_aic,
)


def test_ne_aic_selects_no_signal_for_equal_noise_eigenvalues() -> None:
    """白色雑音だけなら、等しい固有値から信号数0を選ぶことを確認する。"""
    signal_count, aic_values = estimate_signal_count_ne_aic(np.ones(4), snapshot_count=16)

    assert signal_count == 0
    assert aic_values.shape == (4,)


def test_music_peak_matches_steering_orthogonal_to_noise_space() -> None:
    """雑音部分空間と直交する方位で MUSIC が最大になることを確認する。"""
    noise_space = np.array([[0.0], [1.0]], dtype=np.complex128)
    steering = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex128)

    spectrum = calculate_music_spectrum(noise_space, steering)

    assert np.isinf(spectrum[0])
    assert spectrum[1] == 1.0


def test_ebae_band_preserves_distortionless_response() -> None:
    """1信号条件で、未正規化steeringに対する完成重みの応答が1になることを確認する。"""
    steering = np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0j, -1.0], [1.0, -1.0, 1.0]],
        dtype=np.complex128,
    )
    source = steering[:, 1] / np.linalg.norm(steering[:, 1])
    # 強いrank-1信号と白色雑音を重ね、N/E AIC が1信号を識別できる固有値差を作る。
    covariance = 100.0 * np.outer(source, source.conj()) + np.eye(3)
    config = EbaeConfig(snapshot_rate_hz=9.0, integration_time_sec=1.0)

    result = design_ebae_weights_band(covariance, steering, snapshot_count=9, config=config)

    assert result.signal_count == 1
    assert int(result.associated_beam_indices[0]) == 1
    response = np.sum(steering.conj() * result.weights, axis=0)
    np.testing.assert_allclose(response, np.ones(3), atol=1.0e-10)
    assert not result.used_fallback


def test_ebae_processes_fft_bins_independently() -> None:
    """binごとに異なるsource方位を与え、対応方位が独立に推定されることを確認する。"""
    steering = np.array(
        [[1.0, 1.0], [1.0, 1.0j], [1.0, -1.0]],
        dtype=np.complex128,
    )
    steering_bands = np.repeat(steering[:, :, np.newaxis], 2, axis=2)
    covariances = []
    for beam_index in range(2):
        source = steering[:, beam_index] / np.linalg.norm(steering[:, beam_index])
        covariances.append(100.0 * np.outer(source, source.conj()) + np.eye(3))
    config = EbaeConfig(snapshot_rate_hz=9.0, integration_time_sec=1.0)

    # stack後のshapeは[n_bin,n_ch,n_ch]。complex dtypeを明示し、bin軸を共分散の先頭へ置く。
    covariance_bands = np.asarray(np.stack(covariances, axis=0), dtype=np.complex128)
    result = design_ebae_weights(covariance_bands, steering_bands, config=config)

    np.testing.assert_array_equal(result.signal_counts, np.array([1, 1]))
    np.testing.assert_array_equal(result.associated_beam_indices[:, 0], np.array([0, 1]))
    assert result.weights.shape == (3, 2, 2)


def test_ebae_requires_m_squared_snapshots() -> None:
    """N/E AIC の運用契約 ``rate*T=M^2`` を満たさない設定を拒否する。"""
    covariance = np.eye(3, dtype=np.complex128)
    steering = np.ones((3, 1), dtype=np.complex128)
    config = EbaeConfig(snapshot_rate_hz=8.0, integration_time_sec=1.0)

    try:
        design_ebae_weights_band(covariance, steering, snapshot_count=9, config=config)
    except ValueError as error:
        assert "snapshot_rate_hz * integration_time_sec" in str(error)
    else:
        raise AssertionError("rate*T != M**2 must be rejected")
