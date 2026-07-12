"""非対角正規化空間相関の統計量と基線別集計を検証する。"""

import numpy as np

from spflow.beamforming import (
    calculate_sparse_array_spatial_correlation_statistics,
    calculate_spatial_correlation_statistics,
)


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


def test_sparse_statistics_use_coordinate_distance_and_pair_composition() -> None:
    """非等間隔配置を座標距離でbin分けし、中央・外側pair数も保持する。"""

    positions_m = np.array(
        [[-2.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    covariance = np.zeros((1, 4, 4, 2), dtype=np.complex64)
    covariance[0, :, :, 0] = np.eye(4, dtype=np.complex64)
    covariance[0, :, :, 1] = np.eye(4, dtype=np.complex64)
    for first_channel in range(1, 4):
        for second_channel in range(first_channel):
            distance_m = float(np.linalg.norm(positions_m[first_channel] - positions_m[second_channel]))
            correlation = np.complex64(distance_m / 4.0 + 0.0j)
            covariance[0, first_channel, second_channel, :] = correlation
            covariance[0, second_channel, first_channel, :] = np.conjugate(correlation)

    result = calculate_sparse_array_spatial_correlation_statistics(
        covariance,
        positions_m,
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([False, True, True, False]),
        sound_speed_m_s=1.0,
        physical_baseline_edges_m=np.array([0.0, 1.25, 2.75, 4.1], dtype=np.float32),
        wavelength_normalized_edges=np.array([0.0, 1.25, 2.75, 4.1], dtype=np.float32),
    )

    np.testing.assert_allclose(np.sort(result.pair_distance_m), [1.0, 1.5, 1.5, 2.5, 2.5, 4.0])
    np.testing.assert_array_equal(result.physical_baseline.pair_count[0], [1, 4, 1])
    np.testing.assert_allclose(result.physical_baseline.value_minimum[0], [1.0, 1.5, 4.0])
    np.testing.assert_allclose(result.physical_baseline.value_maximum[0], [1.0, 2.5, 4.0])
    np.testing.assert_allclose(result.physical_baseline.value_representative[0], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(result.physical_baseline.mean[0, 0], [0.25, 0.5, 1.0])
    np.testing.assert_allclose(result.physical_baseline.standard_deviation[0, 0], [0.0, 0.125, 0.0])
    np.testing.assert_allclose(result.physical_baseline.interquartile_range[0, 0], [0.0, 0.25, 0.0])
    np.testing.assert_array_equal(result.pair_composition.pair_count, [1, 4, 1])
    assert result.pair_composition.mean.shape == (3, 1, 2)
    assert result.pair_composition.median.shape == (3, 1, 2)
    assert result.pair_composition.percentile_95.shape == (3, 1, 2)
    # DCではd*f/c=0なので全6 pairが正規化基線の先頭binへ入る。
    np.testing.assert_array_equal(result.wavelength_normalized_baseline.pair_count[0], [6, 0, 0])
    # f=1 Hz、c=1 m/sでは波長正規化基線値が物理距離と一致する。
    np.testing.assert_array_equal(result.wavelength_normalized_baseline.pair_count[1], [1, 4, 1])
