"""候補方位別snapshot抽出と周波数共分散の契約を検証する。"""

from __future__ import annotations

import numpy as np
import pytest

from spflow.beamforming import (
    estimate_direction_aligned_frequency_covariance,
    extract_direction_aligned_rfft_snapshots,
)


@pytest.mark.parametrize(
    ("real_dtype", "expected_complex_dtype"),
    [(np.float32, np.complex64), (np.float64, np.complex128)],
)
def test_extract_direction_aligned_rfft_snapshots_aligns_wavefront_and_precision(
    real_dtype: type[np.float32] | type[np.float64],
    expected_complex_dtype: type[np.complex64] | type[np.complex128],
) -> None:
    """channel別遅延を引いた同一波面区間と32/64 bit精度を維持する。"""
    # ch1をch0より2 sample遅らせる条件にし、整列後の各frameがどの入力区間を
    # 参照すべきかを整数列から直接判別できるようにする。
    signal = np.asarray(
        [np.arange(12), 100 + np.arange(12)],
        dtype=real_dtype,
    )
    delays = np.asarray([0, 2], dtype=np.int64)

    snapshots = extract_direction_aligned_rfft_snapshots(
        signal,
        delays,
        analysis_sample_count=10,
        fft_size=4,
        hop_size=2,
    )

    # start=[2,4,6]に対し、ch0は[start:start+4]、ch1は[start-2:start+2]を使う。
    expected_frames = np.asarray(
        [
            [[2, 3, 4, 5], [100, 101, 102, 103]],
            [[4, 5, 6, 7], [102, 103, 104, 105]],
            [[6, 7, 8, 9], [104, 105, 106, 107]],
        ],
        dtype=real_dtype,
    )
    expected = np.moveaxis(np.fft.rfft(expected_frames, axis=2), 2, 1).astype(
        expected_complex_dtype
    )

    assert snapshots.shape == (3, 3, 2)
    assert snapshots.dtype == np.dtype(expected_complex_dtype)
    np.testing.assert_allclose(snapshots, expected)


def test_extract_direction_aligned_rfft_snapshots_keeps_unnormalized_fft_contract() -> None:
    """矩形窓のDC係数がFFT長となり、暗黙のFFT正規化を加えないことを確認する。"""
    # 定数1の4点非正規化FFTではDC振幅が4になるため、規約を最短の入力で固定できる。
    signal = np.ones((2, 4), dtype=np.float64)
    snapshots = extract_direction_aligned_rfft_snapshots(
        signal,
        np.zeros(2, dtype=np.int64),
        analysis_sample_count=4,
        fft_size=4,
        hop_size=4,
    )

    np.testing.assert_allclose(snapshots[0, 0], np.asarray([4.0, 4.0]))
    np.testing.assert_allclose(snapshots[0, 1:], 0.0)


def test_estimate_direction_aligned_frequency_covariance_selects_declared_axes() -> None:
    """指定bin・active channel・先頭L snapshotだけでR=XX^H/Lを作る。"""
    snapshots = np.zeros((3, 2, 3), dtype=np.complex128)
    snapshots[:, 1, :] = np.asarray(
        [
            [1.0 + 1.0j, 10.0 + 0.0j, 2.0 - 1.0j],
            [2.0 + 0.0j, 20.0 + 0.0j, -1.0 + 2.0j],
            [99.0 + 0.0j, 99.0 + 0.0j, 99.0 + 0.0j],
        ]
    )
    active_indices = np.asarray([2, 0], dtype=np.int64)

    covariance = estimate_direction_aligned_frequency_covariance(
        snapshots,
        frequency_index=1,
        active_channel_indices=active_indices,
        snapshot_count=2,
    )

    # active orderを[2,0]に固定し、3枚目を平均へ含めない期待値を直接構成する。
    active = snapshots[:2, 1, :][:, active_indices]
    expected = active.T @ active.conj() / 2.0
    np.testing.assert_allclose(covariance, expected)
    np.testing.assert_allclose(covariance, covariance.conj().T)


@pytest.mark.parametrize(
    ("delays", "analysis_sample_count", "fft_size", "expected_exception"),
    [
        (np.asarray([0, -1], dtype=np.int64), 8, 4, ValueError),
        (np.asarray([0.0, 1.0]), 8, 4, TypeError),
        (np.asarray([0, 1], dtype=np.int64), 9, 4, ValueError),
        (np.asarray([0, 7], dtype=np.int64), 8, 4, ValueError),
    ],
)
def test_extract_direction_aligned_rfft_snapshots_rejects_invalid_boundaries(
    delays: np.ndarray,
    analysis_sample_count: int,
    fft_size: int,
    expected_exception: type[Exception],
) -> None:
    """負遅延、非整数遅延、範囲外sample数、frame不足を早期エラーにする。"""
    signal = np.zeros((2, 8), dtype=np.float64)

    with pytest.raises(expected_exception):
        extract_direction_aligned_rfft_snapshots(
            signal,
            delays,
            analysis_sample_count=analysis_sample_count,
            fft_size=fft_size,
            hop_size=2,
        )


@pytest.mark.parametrize(
    ("frequency_index", "active_indices", "snapshot_count"),
    [
        (2, np.asarray([0], dtype=np.int64), 1),
        (0, np.asarray([0, 0], dtype=np.int64), 1),
        (0, np.asarray([2], dtype=np.int64), 1),
        (0, np.asarray([0], dtype=np.int64), 3),
    ],
)
def test_estimate_direction_aligned_frequency_covariance_rejects_invalid_selection(
    frequency_index: int,
    active_indices: np.ndarray,
    snapshot_count: int,
) -> None:
    """周波数・channel・snapshot選択が入力軸の契約を外れた場合に拒否する。"""
    snapshots = np.ones((2, 2, 2), dtype=np.complex64)

    with pytest.raises(ValueError):
        estimate_direction_aligned_frequency_covariance(
            snapshots,
            frequency_index=frequency_index,
            active_channel_indices=active_indices,
            snapshot_count=snapshot_count,
        )
