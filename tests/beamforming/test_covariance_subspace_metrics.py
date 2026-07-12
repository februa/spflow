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
    np.testing.assert_allclose(result.principal_eigenvalue_gap_fraction[:, 0], [1.0, 1.0], atol=1.0e-6)
    np.testing.assert_allclose(result.steering_rank_one_residual[:, 0], [0.0, 1.0], atol=1.0e-6)
    np.testing.assert_allclose(result.trace_power[:, 0], [3.0, 3.0], atol=1.0e-6)
    assert bool(np.all(result.principal_to_noise_mean_ratio[:, 0] > 1.0e6))


def test_white_covariance_has_one_over_channel_steering_power_fraction() -> None:
    """空間白色共分散のsteering方向power占有率が`1/nCh`になる。"""

    covariance = np.eye(4, dtype=np.complex64)[np.newaxis, :, :, np.newaxis]
    steering = np.ones((4, 1, 1), dtype=np.complex64)

    result = calculate_covariance_subspace_metrics(covariance, steering)

    np.testing.assert_allclose(result.steering_power_fraction, [[0.25]], atol=1.0e-6)
    np.testing.assert_allclose(result.principal_eigenvalue_fraction, [[0.25]], atol=1.0e-6)
    np.testing.assert_allclose(result.principal_to_noise_mean_ratio, [[1.0]], atol=1.0e-6)
    np.testing.assert_allclose(result.principal_eigenvalue_gap_fraction, [[0.0]], atol=1.0e-6)
    np.testing.assert_allclose(result.steering_rank_one_residual, [[np.sqrt(3.0) / 2.0]], atol=1.0e-6)


def test_diagonal_unitary_phase_rotation_preserves_covariance_invariants() -> None:
    """候補処理が対角unitary変換だけならpower・固有値・絶対相関は方位不変と確認する。"""

    # 固定seedの複素行列から正定値共分散を作り、偶然の固有値重複を避ける。
    generator = np.random.default_rng(4102)
    samples = (
        generator.standard_normal((4, 7)) + 1j * generator.standard_normal((4, 7))
    ).astype(np.complex64)
    covariance = np.asarray(samples @ samples.conj().T, dtype=np.complex64)
    phase = np.exp(1j * np.array([0.0, 0.2, -0.7, 1.1], dtype=np.float32)).astype(np.complex64)
    rotated = np.asarray(phase[:, None] * covariance * phase.conj()[None, :], dtype=np.complex64)

    # D R D^Hはunitary similarityなので、trace、固有値、Frobenius normを変えない。
    np.testing.assert_allclose(np.trace(rotated), np.trace(covariance), rtol=1.0e-6, atol=1.0e-5)
    np.testing.assert_allclose(np.linalg.eigvalsh(rotated), np.linalg.eigvalsh(covariance), rtol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(rotated), np.linalg.norm(covariance), rtol=1.0e-6)
    # 対角位相回転は各Rijの位相だけを変えるため、絶対値相関も候補方位を識別できない。
    np.testing.assert_allclose(np.abs(rotated), np.abs(covariance), rtol=1.0e-6, atol=1.0e-5)
