"""full DFT重みの有限FIR実現と整数遅延座標変換。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
ComplexArray = NDArray[np.complexfloating[Any, Any]]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class FrequencyWeightFirApproximation:
    """周波数重みを共通tap窓で有限FIR化した結果を保持する。

    再構成重みは ``[n_fft,n_beam,n_ch]``、energy比と窓先頭は ``[n_beam]``。
    窓先頭の単位はsampleである。この結果型は重み設計やlevel計算を担わない。
    """

    reconstructed_weights: ComplexArray
    energy_ratio: FloatArray
    window_start_samples: IntArray


def approximate_frequency_weights_with_fir(
    weights: ComplexArray, tap_count: int
) -> FrequencyWeightFirApproximation:
    """beam内の全channelで共有するcircular窓により周波数重みをFIR近似する。

    Args:
        weights: full DFT重み。shape ``[n_fft,n_beam,n_ch]``。
        tap_count: 採用tap数、単位sample。

    Returns:
        再構成重み、beam別energy比、窓先頭sample。

    Raises:
        ValueError: weightsが3次元でない、軸が空、dtypeが不正、またはtap数が範囲外の場合。
    """
    checked = np.asarray(weights)
    if checked.ndim != 3 or 0 in checked.shape:
        raise ValueError("weights must have non-empty shape (n_fft, n_beam, n_ch).")
    if checked.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("weights dtype must be complex64 or complex128.")
    if not bool(np.all(np.isfinite(checked))):
        raise ValueError("weights must contain only finite values.")
    n_fft, n_beam, _ = checked.shape
    if not 0 < tap_count <= n_fft:
        raise ValueError("tap_count must be in [1, n_fft].")
    reconstructed = np.empty_like(checked)
    real_dtype = (
        np.dtype(np.float32) if checked.dtype == np.dtype(np.complex64) else np.dtype(np.float64)
    )
    energy_ratio = np.empty(n_beam, dtype=real_dtype)
    starts = np.empty(n_beam, dtype=np.int64)
    for beam_index in range(n_beam):
        # 実適用応答conj(w)をIFFTし、axis=0の時間で全channel合算energyを最大化する。
        impulse = np.asarray(
            np.fft.ifft(checked[:, beam_index, :].conj(), axis=0), dtype=checked.dtype
        )
        energy = np.sum(np.abs(impulse) ** 2, axis=1)
        total = float(np.sum(energy))
        extended = np.concatenate((energy, energy[: tap_count - 1]))
        window_energy = np.convolve(extended, np.ones(tap_count), mode="valid")[:n_fft]
        start = int(np.argmax(window_energy)) if total > 0.0 else 0
        starts[beam_index] = start
        energy_ratio[beam_index] = float(window_energy[start] / total) if total > 0.0 else 1.0
        keep = (start + np.arange(tap_count)) % n_fft
        truncated = np.zeros_like(impulse)
        truncated[keep, :] = impulse[keep, :]
        reconstructed[:, beam_index, :] = np.fft.fft(truncated, axis=0).conj()
    return FrequencyWeightFirApproximation(reconstructed, energy_ratio, starts)
