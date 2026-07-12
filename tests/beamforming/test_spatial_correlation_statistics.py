"""非対角正規化空間相関の統計量と基線別集計を検証する。"""

import numpy as np

from spflow.beamforming import calculate_spatial_correlation_statistics


def test_statistics_use_lower_triangle_magnitudes_and_group_baselines() -> None:
    """複素位相を平均せず、下三角pairの絶対正規化相関を集約する。"""

    covariance = np.zeros((1, 3, 3, 2), dtype=np.complex64)
    covariance[0, :, :, 0] = np.eye(3, dtype=np.complex64)
    # 下三角3 pairを0.2、0.4、0.8とし、上三角には共役を設定してHermitianにする。
    covariance[0, 1, 0, 0] = np.complex64(0.0 + 0.2j)
    covariance[0, 0, 1, 0] = np.conjugate(covariance[0, 1, 0, 0])
    covariance[0, 2, 0, 0] = np.complex64(-0.4 + 0.0j)
    covariance[0, 0, 2, 0] = np.conjugate(covariance[0, 2, 0, 0])
    covariance[0, 2, 1, 0] = np.complex64(0.8 + 0.0j)
    covariance[0, 1, 2, 0] = np.conjugate(covariance[0, 2, 1, 0])

    result = calculate_spatial_correlation_statistics(covariance)

    assert result.maximum.shape == (1, 2)
    assert result.baseline_mean.shape == (1, 2, 2)
    np.testing.assert_array_equal(result.baseline_index, [1, 2])
    np.testing.assert_allclose(result.maximum[0], [0.8, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(result.mean[0], [(0.2 + 0.4 + 0.8) / 3.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(result.median[0], [0.4, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(result.percentile_95[0], [0.76, 0.0], atol=1.0e-6)
    # 基線index 1はpair (1,0),(2,1)、index 2はpair (2,0)を集約する。
    np.testing.assert_allclose(result.baseline_mean[0, 0], [0.5, 0.4], atol=1.0e-6)


def test_statistics_normalize_unequal_channel_power() -> None:
    """`|Rij|/sqrt(Rii*Rjj)`がchannel power差を正しく除去する。"""

    covariance = np.zeros((1, 2, 2, 1), dtype=np.complex64)
    covariance[0, 0, 0, 0] = np.complex64(4.0 + 0.0j)
    covariance[0, 1, 1, 0] = np.complex64(9.0 + 0.0j)
    covariance[0, 1, 0, 0] = np.complex64(3.0 + 0.0j)
    covariance[0, 0, 1, 0] = np.complex64(3.0 + 0.0j)

    result = calculate_spatial_correlation_statistics(covariance)

    # sqrt(4*9)=6なので、正規化相関は3/6=0.5になる。
    np.testing.assert_allclose(result.mean, [[0.5]], atol=1.0e-6)
    np.testing.assert_allclose(result.baseline_mean, [[[0.5]]], atol=1.0e-6)
