"""複素steering整合量と共分散固有空間指標を検証する。"""

import numpy as np

from spflow.beamforming import calculate_covariance_subspace_metrics


def test_rank_one_covariance_separates_aligned_and_orthogonal_steering() -> None:
    """rank-1信号空間では整合steeringを1、直交steeringを0として識別する。"""

    aligned = np.array([1.0, 1.0, 1.0], dtype=np.complex64)
    orthogonal = np.array(
        [1.0, np.exp(2j * np.pi / 3.0), np.exp(4j * np.pi / 3.0)],
        dtype=np.complex64,
    )
    rank_one_covariance = np.outer(aligned, aligned.conj()).astype(np.complex64)
    covariance = np.stack((rank_one_covariance, rank_one_covariance), axis=0)[:, :, :, np.newaxis]
    steering = np.stack((aligned, orthogonal), axis=1)[:, :, np.newaxis]

    result = calculate_covariance_subspace_metrics(covariance, steering, direction_chunk_size=1)

    np.testing.assert_allclose(result.steering_power_fraction[:, 0], [1.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(result.principal_eigenvector_alignment[:, 0], [1.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(result.principal_eigenvalue_fraction[:, 0], [1.0, 1.0], atol=1.0e-6)
    assert bool(np.all(result.principal_to_noise_mean_ratio[:, 0] > 1.0e6))


def test_white_covariance_has_one_over_channel_steering_power_fraction() -> None:
    """空間白色共分散のsteering方向power占有率が`1/nCh`になる。"""

    covariance = np.eye(4, dtype=np.complex64)[np.newaxis, :, :, np.newaxis]
    steering = np.ones((4, 1, 1), dtype=np.complex64)

    result = calculate_covariance_subspace_metrics(covariance, steering)

    np.testing.assert_allclose(result.steering_power_fraction, [[0.25]], atol=1.0e-6)
    np.testing.assert_allclose(result.principal_eigenvalue_fraction, [[0.25]], atol=1.0e-6)
    np.testing.assert_allclose(result.principal_to_noise_mean_ratio, [[1.0]], atol=1.0e-6)
