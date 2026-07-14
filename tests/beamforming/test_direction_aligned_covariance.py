"""候補方位別の時間切り出しと共通時間軸位相復元を検証する。"""

from __future__ import annotations

import numpy as np
import pytest

from spflow.beamforming import (
    calculate_snapshot_time_axis_restoration_phase,
    extract_direction_aligned_rfft_snapshots,
)


def test_calculate_snapshot_time_axis_restoration_phase_uses_channel_mean_reference() -> None:
    """channel別中心と平均中心の差に対する`exp(-j2πkΔn/N)`を返す。"""
    # 2方位で相対中心の符号を反転させ、channel軸とdirection軸の取り違えも検出する。
    centers = np.asarray([[0.0, 1.0], [2.0, -1.0]], dtype=np.float64)

    phase = calculate_snapshot_time_axis_restoration_phase(centers, fft_size=8)

    relative_centers = centers - np.mean(centers, axis=0, keepdims=True)
    frequency_bin = np.arange(5, dtype=np.float64)
    expected = np.exp(
        -1j
        * 2.0
        * np.pi
        * relative_centers[:, np.newaxis, :]
        * frequency_bin[np.newaxis, :, np.newaxis]
        / 8.0
    )
    assert phase.shape == (2, 5, 2)
    np.testing.assert_allclose(phase, expected, atol=1.0e-15)
    # DCは切り出し時刻に依存しないため、全channel・全方位で位相1となる。
    np.testing.assert_allclose(phase[:, 0, :], 1.0)


@pytest.mark.parametrize(
    ("real_dtype", "expected_complex_dtype"),
    [(np.float32, np.complex64), (np.float64, np.complex128)],
)
def test_extract_direction_aligned_rfft_snapshots_restores_common_time_axis(
    real_dtype: type[np.float32] | type[np.float64],
    expected_complex_dtype: type[np.complex64] | type[np.complex128],
) -> None:
    """異なる開始時刻から切り出した同一toneを同じrFFT位相へ復元する。"""
    # FFT bin 1のtoneを両channelへ同相で与える。delay=[0,2]では生の切り出しFFTに
    # 2 sample分の位相差が出るため、復元符号が正しい場合だけchannel spectrumが一致する。
    sample_index = np.arange(18, dtype=np.float64)
    tone = np.cos(2.0 * np.pi * sample_index / 8.0).astype(real_dtype)
    signal = np.stack((tone, tone), axis=0)

    snapshots = extract_direction_aligned_rfft_snapshots(
        signal,
        np.asarray([0, 2], dtype=np.int64),
        fft_size=8,
        hop_size=8,
    )

    assert snapshots.shape == (2, 5, 2)
    assert snapshots.dtype == np.dtype(expected_complex_dtype)
    np.testing.assert_allclose(snapshots[:, :, 0], snapshots[:, :, 1], atol=2.0e-6)
    # NumPy非正規化rFFTではpeak振幅N/2=4となり、位相復元は振幅校正を変えない。
    np.testing.assert_allclose(np.abs(snapshots[:, 1, :]), 4.0, atol=2.0e-6)


def test_extract_direction_aligned_rfft_snapshots_accepts_real_array_like() -> None:
    """NumPyと同様に実数array-likeをfloat64へ正規化して処理する。"""
    snapshots = extract_direction_aligned_rfft_snapshots(
        [[1, 1, 1, 1], [1, 1, 1, 1]],
        np.zeros(2, dtype=np.int64),
        fft_size=4,
        hop_size=4,
    )

    assert snapshots.dtype == np.dtype(np.complex128)
    np.testing.assert_allclose(snapshots[0, 0], np.asarray([4.0, 4.0]))
    np.testing.assert_allclose(snapshots[0, 1:], 0.0)


@pytest.mark.parametrize(
    ("signal", "delays", "fft_size", "expected_exception"),
    [
        (np.zeros((2, 8), dtype=np.complex128), np.asarray([0, 1]), 4, TypeError),
        (np.zeros((2, 8)), np.asarray([0.0, 1.0]), 4, TypeError),
        (np.zeros((2, 8)), np.asarray([0, -1]), 4, ValueError),
        (np.zeros((2, 8)), np.asarray([0, 7]), 4, ValueError),
    ],
)
def test_extract_direction_aligned_rfft_snapshots_rejects_invalid_physical_boundaries(
    signal: np.ndarray,
    delays: np.ndarray,
    fft_size: int,
    expected_exception: type[Exception],
) -> None:
    """複素入力、非整数・負遅延、完全frame不足という物理境界だけを拒否する。"""
    with pytest.raises(expected_exception):
        extract_direction_aligned_rfft_snapshots(
            signal,
            delays,
            fft_size=fft_size,
            hop_size=2,
        )
