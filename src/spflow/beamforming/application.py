"""設計済みビームフォーミング重みを信号へ適用する共通部品を提供する。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_int


def apply_beamformer(snapshots: NDArray[Any], weights: NDArray[Any]) -> NDArray[np.complex64]:
    """単一帯域の複素重みを観測snapshotへ適用する。

    Args:
        snapshots: 観測snapshot。shapeは`[n_ch, n_frame]`。axis=0はsensor channel、
            axis=1は独立な時間frameまたはsnapshotである。
        weights: 設計済み重み。shapeは`[n_ch]`または`[n_ch, n_beam]`。

    Returns:
        ビーム出力。shapeは`[n_beam, n_frame]`、dtypeは`complex64`。

    Raises:
        ValueError: 入力が2次元でない、またはchannel数が一致しない場合。

    境界条件:
        この関数は`y=w^H x`の適用だけを担い、重み設計、FFT、状態保持は行わない。
    """

    values = np.asarray(snapshots, dtype=np.complex64)
    beam_weights = np.asarray(weights, dtype=np.complex64)
    require(values.ndim == 2, "snapshots must have shape (n_ch, n_frame).")
    if beam_weights.ndim == 1:
        beam_weights = beam_weights[:, np.newaxis]
    require(beam_weights.ndim == 2, "weights must have shape (n_ch,) or (n_ch, n_beam).")
    require(values.shape[0] == beam_weights.shape[0], "snapshots and weights must agree on n_ch.")

    # values shape: [n_ch, n_frame]、beam_weights shape: [n_ch, n_beam]。
    # channel軸だけをw^H xとして畳み込み、beam軸とframe軸を保持する。
    return np.asarray(
        np.einsum("cf,cb->bf", values, beam_weights.conj(), optimize=True),
        dtype=np.complex64,
    )


def apply_beamformer_bands(
    snapshots: NDArray[Any],
    weights: NDArray[Any],
) -> NDArray[np.complex64]:
    """帯域別の複素重みを観測snapshotへ適用する。

    Args:
        snapshots: shape`[n_ch, n_band]`または`[n_ch, n_band, n_frame]`の観測。
            axis=0はsensor channel、axis=1は周波数帯域、axis=2は時間frameである。
        weights: 帯域別重み。shapeは`[n_ch, n_beam, n_band]`。

    Returns:
        2次元入力ではshape`[n_beam, n_band]`、3次元入力では
        shape`[n_beam, n_band, n_frame]`のビーム出力。dtypeは`complex64`。

    Raises:
        ValueError: shape、channel数、band数が一致しない場合。

    境界条件:
        各bandは独立な複素snapshotとして扱い、band間の補間や合成は行わない。
    """

    values = np.asarray(snapshots, dtype=np.complex64)
    beam_weights = np.asarray(weights, dtype=np.complex64)
    require(
        beam_weights.ndim == 3,
        "weights must have shape (n_ch, n_beam, n_band).",
    )
    require(values.ndim in {2, 3}, "snapshots must have 2 or 3 dimensions.")
    require(
        values.shape[0] == beam_weights.shape[0] and values.shape[1] == beam_weights.shape[2],
        "snapshots and weights must agree on n_ch and n_band.",
    )

    if values.ndim == 2:
        # values shape: [n_ch, n_band]、weights shape: [n_ch, n_beam, n_band]。
        return np.asarray(
            np.einsum("cb,cdb->db", values, beam_weights.conj(), optimize=True),
            dtype=np.complex64,
        )

    # values shape: [n_ch, n_band, n_frame]。
    # channel軸だけを内積とし、beam、band、frameの各軸を保持する。
    return np.asarray(
        np.einsum("cbf,cdb->dbf", values, beam_weights.conj(), optimize=True),
        dtype=np.complex64,
    )


def apply_beamformer_filter_fft(
    input_spectrum: NDArray[Any],
    filter_spectrum: NDArray[Any],
) -> NDArray[np.complex64]:
    """overlap-save用filter FFTをmulti-channel spectrumへ適用する。

    Args:
        input_spectrum: 入力FFT。shapeは`[n_ch, n_freq]`。axis=1はFFT周波数bin。
        filter_spectrum: filter FFT。shapeは`[n_ch, n_beam, n_freq]`。
            `w^H x`の共役はfilter側へ事前に含める。

    Returns:
        出力spectrum。shapeは`[n_beam, n_freq]`、dtypeは`complex64`。

    Raises:
        ValueError: shape、channel数、frequency bin数が一致しない場合。

    境界条件:
        FFT/IFFTとoverlap-saveの有効区間抽出は呼び出し側の責務とする。
    """

    spectra = np.asarray(input_spectrum, dtype=np.complex64)
    filters = np.asarray(filter_spectrum, dtype=np.complex64)
    require(spectra.ndim == 2, "input_spectrum must have shape (n_ch, n_freq).")
    require(
        filters.ndim == 3,
        "filter_spectrum must have shape (n_ch, n_beam, n_freq).",
    )
    require(
        spectra.shape[0] == filters.shape[0] and spectra.shape[1] == filters.shape[2],
        "input_spectrum and filter_spectrum must agree on n_ch and n_freq.",
    )

    # 周波数binごとにΣ_ch X[ch,k] H[ch,beam,k]を計算する。
    # filter側へconj(w)を焼き込む規約なので、ここでは追加の共役を取らない。
    return np.asarray(
        np.einsum("cf,cbf->bf", spectra, filters, optimize=True),
        dtype=np.complex64,
    )


def build_time_tapped_snapshot_matrix(
    channel_signals: NDArray[Any],
    tap_len: int,
) -> NDArray[np.complex128]:
    """channel信号を時間領域FIR適用用のchannel×tap行列へ展開する。

    Args:
        channel_signals: 入力信号。shapeは`[n_ch, n_sample]`。axis=1は時間sample。
        tap_len: FIR tap数`L`。単位はsample。

    Returns:
        snapshot行列。shapeは`[n_ch * L, n_sample - L + 1]`。
        rowはlag-majorで、各lag内にchannelを並べる。

    Raises:
        ValueError: shapeが不正、tap_lenが正でない、またはsample数が不足する場合。

    境界条件:
        full tapが揃わない先頭`L-1` sampleは返さない。公開出力での扱いは適用関数が決める。
    """

    signals = np.asarray(channel_signals, dtype=np.complex128)
    require(signals.ndim == 2, "channel_signals must have shape (n_ch, n_sample).")
    require_positive_int("tap_len", int(tap_len))
    require(
        signals.shape[1] >= int(tap_len),
        "channel_signals must contain at least tap_len samples.",
    )

    n_ch = int(signals.shape[0])
    n_valid_sample = int(signals.shape[1]) - int(tap_len) + 1
    tapped = np.zeros((n_ch * int(tap_len), n_valid_sample), dtype=np.complex128)
    for lag_index in range(int(tap_len)):
        row_start = lag_index * n_ch
        row_stop = row_start + n_ch
        sample_start = int(tap_len) - 1 - lag_index
        sample_stop = sample_start + n_valid_sample
        # X_tap[lag,ch,n]=x[ch,n+L-1-lag]。lag=0は現在sample、以降は過去sample。
        tapped[row_start:row_stop, :] = signals[:, sample_start:sample_stop]
    return tapped


def apply_time_domain_fir_beamformer(
    channel_signals: NDArray[Any],
    weights: NDArray[Any],
    *,
    tap_len: int,
) -> NDArray[np.complex128]:
    """channel×tap FIR重みを時間波形へ適用する。

    Args:
        channel_signals: 入力信号。shapeは`[n_ch, n_sample]`。
        weights: FIR重み。shapeは`[n_ch * tap_len]`または
            `[n_ch * tap_len, n_output]`。
        tap_len: FIR tap数`L`。単位はsample。

    Returns:
        出力信号。shapeは`[n_output, n_sample]`、dtypeは`complex128`。
        full tapが揃わない先頭`L-1` sampleは0とする。

    Raises:
        ValueError: 入力shapeまたはchannel×tap自由度が一致しない場合。

    境界条件:
        先頭を0とすることで、未完成のFIR出力を完成値として公開しない。
        重み設計、共分散推定、streaming状態保持は責務に含めない。
    """

    tapped = build_time_tapped_snapshot_matrix(channel_signals, tap_len=int(tap_len))
    beam_weights = np.asarray(weights, dtype=np.complex128)
    if beam_weights.ndim == 1:
        beam_weights = beam_weights[:, np.newaxis]
    require(
        beam_weights.ndim == 2,
        "weights must have shape (n_dof,) or (n_dof, n_output).",
    )
    require(
        beam_weights.shape[0] == tapped.shape[0],
        "weights and channel_signals must agree on n_ch * tap_len.",
    )

    n_sample = int(np.asarray(channel_signals).shape[1])
    output = np.zeros((beam_weights.shape[1], n_sample), dtype=np.complex128)
    # y[out,n]=Σ_dof conj(w[dof,out]) X_tap[dof,n]。
    # 時間領域でも周波数領域と同じw^H x規約を使用する。
    output[:, int(tap_len) - 1 :] = beam_weights.conj().T @ tapped
    return output


__all__ = [
    "apply_beamformer",
    "apply_beamformer_bands",
    "apply_beamformer_filter_fft",
    "apply_time_domain_fir_beamformer",
    "build_time_tapped_snapshot_matrix",
]
