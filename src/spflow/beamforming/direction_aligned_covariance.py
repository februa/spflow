"""候補方位に従う時間切り出しと共通時間軸への位相復元を実装する。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complexfloating[Any, Any]]


def calculate_snapshot_time_axis_restoration_phase(
    snapshot_center_samples: ArrayLike,
    *,
    fft_size: int,
) -> NDArray[np.complex128]:
    """channel別snapshot時刻をchannel平均の共通時刻へ戻すrFFT位相を返す。

    Args:
        snapshot_center_samples: channel別snapshot中心。shapeは`[n_ch,n_direction]`、
            axis=0はchannel、axis=1は候補方位、単位はsample。絶対時刻でも相対時刻でもよい。
        fft_size: rFFTのFFT長。単位はsample。

    Returns:
        位相復元係数。shapeは`[n_ch,n_frequency,n_direction]`、
        `n_frequency=fft_size//2+1`。各候補方位のchannel平均中心を共通時刻とし、
        `exp(-j 2 pi k Delta n / fft_size)`を格納する。

    Raises:
        ValueError: 中心表が空の2次元配列でない場合、またはFFT長が正でない場合。

    Notes:
        この位相はchannelごとに異なる区間をFFTしたことで生じる時刻基準差だけを除く。
        T1からT2aへの整数遅延座標変換や、残留小数遅延steeringは責務に含めない。
    """
    centers = np.asarray(snapshot_center_samples, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] == 0:
        raise ValueError("snapshot_center_samples must have shape [n_ch, n_direction].")
    if fft_size <= 0:
        raise ValueError("fft_size must be positive.")

    # reference_center shape: [1,n_direction]。参照時刻の全channel共通位相は
    # R=XX^Hで相殺されるため、位相量を小さく保てるchannel平均を採用する。
    reference_center = np.mean(centers, axis=0, keepdims=True)
    relative_center_samples = centers - reference_center
    frequency_bin = np.arange(fft_size // 2 + 1, dtype=np.float64)

    # 位相shapeは[ch,frequency,direction]。異なる開始時刻Delta nを持つDFTへ
    # exp(-j 2π k Delta n/N)を掛け、全channelを同じ絶対時間基準へ復元する。
    phase = np.exp(
        -1j
        * 2.0
        * np.pi
        * relative_center_samples[:, np.newaxis, :]
        * frequency_bin[np.newaxis, :, np.newaxis]
        / float(fft_size)
    )
    return np.asarray(phase, dtype=np.complex128)


def extract_direction_aligned_rfft_snapshots(
    signal: ArrayLike,
    causal_delays_samples: ArrayLike,
    *,
    fft_size: int,
    hop_size: int,
) -> ComplexArray:
    """候補方位の同一波面区間を切り出し、共通時間軸のrFFT snapshotを作る。

    Args:
        signal: 解析対象の実数channel信号。shapeは`[n_ch,n_sample]`、axis=0は
            channel、axis=1は共通収録時間、単位は呼び出し側の線形振幅単位。
        causal_delays_samples: 候補方位に対応する非負整数遅延。shapeは`[n_ch]`、
            単位はsample。整列後時刻`n`には入力`x[c,n-d[c]]`を対応させる。
        fft_size: 矩形snapshotのFFT長。単位はsample。
        hop_size: 整列後時間軸における隣接snapshot間隔。単位はsample。

    Returns:
        方位整列済みrFFT snapshot。shapeは`[n_frame,n_frequency,n_ch]`。
        axis=0は時間snapshot、axis=1はone-sided rFFT周波数bin、axis=2はchannel。
        channel別切り出し時刻の位相はchannel平均時刻へ復元済みで、FFTは非正規化。
        float32入力はcomplex64、それ以外の実数入力はcomplex128で返す。

    Raises:
        TypeError: signalが複素数の場合、または遅延が整数でない場合。
        ValueError: shape、遅延、FFT長、hop長が不正な場合、または完全なsnapshotを
            一つも切り出せない場合。

    Notes:
        本関数の固有責務は、候補方位別の時間切り出しと共通時間軸への位相復元である。
        active channel選択、snapshot数選択、共分散平均、重み設計は呼び出し側で組み合わせる。
    """
    raw_signal = np.asarray(signal)
    if np.iscomplexobj(raw_signal):
        raise TypeError("signal must be real-valued.")
    if raw_signal.ndim != 2 or raw_signal.shape[0] == 0:
        raise ValueError("signal must have shape [n_ch, n_sample] with n_ch > 0.")
    # NumPyと同様に実数array-likeを受け、明示されたfloat32だけ低精度を維持する。
    samples = (
        np.asarray(raw_signal, dtype=np.float32)
        if raw_signal.dtype == np.dtype(np.float32)
        else np.asarray(raw_signal, dtype=np.float64)
    )

    raw_delays = np.asarray(causal_delays_samples)
    if not np.issubdtype(raw_delays.dtype, np.integer):
        raise TypeError("causal_delays_samples must contain integers.")
    delays = np.asarray(raw_delays, dtype=np.int64)
    if delays.shape != (samples.shape[0],):
        raise ValueError("causal_delays_samples must have shape [n_ch].")
    if bool(np.any(delays < 0)):
        raise ValueError("causal_delays_samples must be non-negative.")
    if fft_size <= 0 or hop_size <= 0:
        raise ValueError("fft_size and hop_size must be positive.")

    # 全channelでbegin=n-d[c]>=0となる最初の整列後時刻はmax(d)である。
    # ここを起点にし、先頭ゼロ詰めを共分散用snapshotへ混ぜない。
    starts = np.arange(
        int(np.max(delays)),
        samples.shape[1] - fft_size + 1,
        hop_size,
        dtype=np.int64,
    )
    if starts.size == 0:
        raise ValueError("signal cannot provide one complete direction-aligned frame.")

    # [n_ch,n_sample]から[n_frame,n_ch,n_fft]へ切り出す。channel cの開始時刻は
    # start-d[c]であり、候補方位で同じ波面に対応するsample列をframe内で揃える。
    frames = np.empty((starts.size, samples.shape[0], fft_size), dtype=samples.dtype)
    for frame_index, start in enumerate(starts):
        for channel_index, delay in enumerate(delays):
            begin = int(start) - int(delay)
            frames[frame_index, channel_index] = samples[
                channel_index, begin : begin + fft_size
            ]

    # raw_spectra shape: [n_frame,n_frequency,n_ch]。切り出し中心の共通項
    # start+N/2は全channel同じなので、相対中心-d[c]だけで復元位相を計算できる。
    raw_spectra = np.moveaxis(np.fft.rfft(frames, axis=2), 2, 1)
    phase_ch_frequency = calculate_snapshot_time_axis_restoration_phase(
        -delays[:, np.newaxis],
        fft_size=fft_size,
    )[:, :, 0]
    output_dtype = np.complex64 if samples.dtype == np.dtype(np.float32) else np.complex128
    # phase.T shapeは[n_frequency,n_ch]で、frame軸へbroadcastする。
    return np.asarray(raw_spectra * phase_ch_frequency.T[np.newaxis, :, :], dtype=output_dtype)


__all__ = [
    "calculate_snapshot_time_axis_restoration_phase",
    "extract_direction_aligned_rfft_snapshots",
]
