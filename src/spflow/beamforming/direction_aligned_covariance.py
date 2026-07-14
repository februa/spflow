"""候補方位へ時間整列した周波数snapshotと空間共分散を作る。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating[Any, Any]]


def extract_direction_aligned_rfft_snapshots(
    signal: NDArray[Any],
    causal_delays_samples: NDArray[Any],
    *,
    analysis_sample_count: int,
    fft_size: int,
    hop_size: int,
) -> ComplexArray:
    """候補方位の整数遅延に従い、同一波面区間のrFFT snapshotを作る。

    Args:
        signal: 実数channel信号。shapeは`[n_ch,n_sample]`、axis=0はchannel、
            axis=1は時間、単位は呼び出し側の線形振幅単位。dtypeはfloat32またはfloat64。
        causal_delays_samples: 各channelへ適用する非負整数遅延。shapeは`[n_ch]`、
            単位はsample。値`d[c]`は出力時刻`n`に入力`x[c,n-d[c]]`を対応させる。
        analysis_sample_count: `signal[:, :analysis_sample_count]`を解析対象とするsample数。
        fft_size: 各snapshotのFFT長。単位はsample。
        hop_size: 隣接snapshot開始位置の間隔。単位はsample。

    Returns:
        方位整列済みrFFT snapshot。shapeは`[n_frame,n_frequency,n_ch]`、
        `n_frequency=fft_size//2+1`。axis=0は時間snapshot、axis=1はone-sided
        rFFT周波数bin、axis=2はchannel。窓は矩形、FFTはNumPyの非正規化規約であり、
        dtypeはfloat32入力ならcomplex64、float64入力ならcomplex128となる。

    Raises:
        TypeError: signalがfloat32/float64でない場合、または遅延が整数でない場合。
        ValueError: shape、遅延、sample数、FFT長、hop長が不正な場合、または完全な
            snapshotを一つも作れない場合。

    Notes:
        この関数は候補方位に対応する時間切り出しとFFTだけを担う。窓設計、共分散平均、
        MVDR/EBAE重み設計、逐次更新scheduleは責務に含めない。
    """
    samples = np.asarray(signal)
    if samples.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("signal dtype must be float32 or float64.")
    if samples.ndim != 2 or samples.shape[0] <= 0:
        raise ValueError("signal must have shape [n_ch, n_sample] with n_ch > 0.")
    if not bool(np.all(np.isfinite(samples))):
        raise ValueError("signal must contain only finite values.")

    raw_delays = np.asarray(causal_delays_samples)
    if not np.issubdtype(raw_delays.dtype, np.integer):
        raise TypeError("causal_delays_samples must contain integers.")
    delays = np.asarray(raw_delays, dtype=np.int64)
    if delays.ndim != 1 or delays.shape[0] != samples.shape[0]:
        raise ValueError("causal_delays_samples must have shape [n_ch].")
    if bool(np.any(delays < 0)):
        raise ValueError("causal_delays_samples must be non-negative.")
    if analysis_sample_count <= 0 or analysis_sample_count > samples.shape[1]:
        raise ValueError("analysis_sample_count must be in [1, n_sample].")
    if fft_size <= 0:
        raise ValueError("fft_size must be positive.")
    if hop_size <= 0:
        raise ValueError("hop_size must be positive.")

    # 全channelでbegin=n-d[c]>=0となる最初の共通出力時刻はmax(d)である。
    # ここを起点にすることで、先頭ゼロ詰めを共分散へ混ぜず同一波面区間を切り出す。
    maximum_delay = int(np.max(delays))
    starts = np.arange(
        maximum_delay,
        analysis_sample_count - fft_size + 1,
        hop_size,
        dtype=np.int64,
    )
    if starts.size == 0:
        raise ValueError("analysis interval cannot provide one complete direction-aligned frame.")

    # frames変換前shapeは[n_ch,n_sample]、変換後shapeは[n_frame,n_ch,n_fft]。
    # frameごとにchannel別遅延d[c]を引き、同じ整列後時刻へ対応する実波形を並べる。
    frames = np.empty((starts.size, samples.shape[0], fft_size), dtype=samples.dtype)
    for frame_index, start in enumerate(starts):
        for channel_index, delay in enumerate(delays):
            begin = int(start) - int(delay)
            frames[frame_index, channel_index] = samples[
                channel_index, begin : begin + fft_size
            ]

    # FFT axis=2は各channelの時間軸。NumPy rFFTはfloat32もcomplex128へ昇格するため、
    # simulation precision契約を保つ目的でfloat32入力だけcomplex64へ戻す。
    spectra = np.moveaxis(np.fft.rfft(frames, axis=2), 2, 1)
    output_dtype = np.complex64 if samples.dtype == np.dtype(np.float32) else np.complex128
    return np.asarray(spectra, dtype=output_dtype)


def estimate_direction_aligned_frequency_covariance(
    snapshots: NDArray[Any],
    *,
    frequency_index: int,
    active_channel_indices: NDArray[Any],
    snapshot_count: int | None = None,
) -> ComplexArray:
    """方位整列済みsnapshotから指定周波数の空間共分散を推定する。

    Args:
        snapshots: `extract_direction_aligned_rfft_snapshots`の出力。
            shapeは`[n_frame,n_frequency,n_ch]`、FFT振幅単位。
        frequency_index: 共分散を作るone-sided rFFT周波数bin index。
        active_channel_indices: 共分散へ含めるchannel index。shapeは`[n_active_ch]`。
            重複せず、`[0,n_ch)`の範囲にある整数とする。
        snapshot_count: 先頭から平均するsnapshot数。Noneは全snapshotを使用する。

    Returns:
        空間共分散`R=(1/L) sum_l x_l x_l^H`。
        shapeは`[n_active_ch,n_active_ch]`、axis=0/1はいずれもactive channel、
        単位は非正規化FFT振幅の二乗。complex64/complex128の入力精度を維持する。

    Raises:
        TypeError: snapshotsがcomplex64/complex128でない場合、またはchannel indexが
            整数でない場合。
        ValueError: shape、周波数index、channel index、snapshot数が不正な場合。

    Notes:
        DC/Nyquistを含め、ここではone-sided powerのconjugate-pair係数を掛けない。
        本量は同一bin内のchannel間空間共分散であり、全帯域power積分ではない。
    """
    spectra = np.asarray(snapshots)
    if spectra.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError("snapshots dtype must be complex64 or complex128.")
    if spectra.ndim != 3 or any(size <= 0 for size in spectra.shape):
        raise ValueError("snapshots must have non-empty shape [n_frame, n_frequency, n_ch].")
    if not bool(np.all(np.isfinite(spectra))):
        raise ValueError("snapshots must contain only finite values.")
    if frequency_index < 0 or frequency_index >= spectra.shape[1]:
        raise ValueError("frequency_index is outside the snapshot frequency axis.")

    raw_indices = np.asarray(active_channel_indices)
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise TypeError("active_channel_indices must contain integers.")
    indices = np.asarray(raw_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("active_channel_indices must be a non-empty one-dimensional array.")
    if bool(np.any(indices < 0)) or bool(np.any(indices >= spectra.shape[2])):
        raise ValueError("active_channel_indices contains an out-of-range channel index.")
    if np.unique(indices).size != indices.size:
        raise ValueError("active_channel_indices must not contain duplicates.")

    selected_snapshot_count = spectra.shape[0] if snapshot_count is None else snapshot_count
    if selected_snapshot_count <= 0 or selected_snapshot_count > spectra.shape[0]:
        raise ValueError("snapshot_count must be in [1, n_frame].")

    # active_snapshots shape: [L,n_active_ch]。frequency軸を固定し、指定channelだけを
    # 選択する。R[c,d]=(1/L)Σ_l X[l,c]conj(X[l,d])としてframe軸を平均する。
    active_snapshots = spectra[:selected_snapshot_count, frequency_index, :][:, indices]
    denominator_dtype = np.float32 if spectra.dtype == np.dtype(np.complex64) else np.float64
    covariance = np.einsum(
        "lc,ld->cd", active_snapshots, active_snapshots.conj(), optimize=True
    ) / np.asarray(selected_snapshot_count, dtype=denominator_dtype)
    return np.asarray(covariance, dtype=spectra.dtype)


__all__ = [
    "estimate_direction_aligned_frequency_covariance",
    "extract_direction_aligned_rfft_snapshots",
]
