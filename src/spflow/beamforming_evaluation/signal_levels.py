"""beamforming診断波形からtone・spectrum・block RMS levelを計算する。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow._validation import require, require_positive_float, require_positive_int

FloatArray = NDArray[np.floating[Any]]


def calculate_tone_projection_rms_level_db20(
    signal: FloatArray,
    frequency_hz: float,
    fs_hz: float,
    *,
    reference_rms: float = 1.0,
) -> float:
    """実波形を一つの周波数へ射影しtone RMS levelを計算する。

    Args:
        signal: 一つの実時間波形。shapeは`[n_sample]`、axis=0は時間sample。
        frequency_hz: 射影するtone周波数。単位はHz。
        fs_hz: サンプリング周波数。単位はHz。
        reference_rms: 0 dBに対応するRMS振幅。signalと同じ振幅単位。

    Returns:
        tone RMS level。単位は`dB re reference_rms`。

    Raises:
        ValueError: signalが1次元でない、空、非有限、または周波数・基準値が不正な場合。

    境界条件:
        観測区間内の複素射影を使うため、非整数bin toneでは窓なし有限区間のleakageを含む。
        完全ゼロは有限な下限値へ丸める。
    """

    values = np.asarray(signal, dtype=np.float64)
    require(values.ndim == 1 and values.size > 0, "signal must have shape (n_sample,).")
    require(bool(np.all(np.isfinite(values))), "signal must contain only finite values.")
    require_positive_float("frequency_hz", float(frequency_hz))
    require_positive_float("fs_hz", float(fs_hz))
    require_positive_float("reference_rms", float(reference_rms))
    require(
        float(frequency_hz) <= 0.5 * float(fs_hz),
        "frequency_hz must not exceed the Nyquist frequency.",
    )

    time_axis_s = np.arange(values.shape[0], dtype=np.float64) / float(fs_hz)
    reference = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * time_axis_s)

    # coefficientは実toneの正周波数側peak/2に対応する。
    # peak=2*|coefficient|、RMS=peak/sqrt(2)として入力tone RMSへ戻す。
    coefficient = np.vdot(reference, values.astype(np.complex128)) / values.shape[0]
    rms_amplitude = np.sqrt(2.0) * np.abs(coefficient)
    normalized_rms = float(rms_amplitude) / float(reference_rms)
    return float(20.0 * np.log10(max(normalized_rms, np.finfo(np.float64).tiny)))


def calculate_one_sided_rms_spectrum_db20(
    real_signals: FloatArray,
    fs_hz: float,
    *,
    reference_rms: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """複数実波形のone-sided per-bin RMS spectrum levelを計算する。

    Args:
        real_signals: 実波形。shapeは`[n_series, n_sample]`、axis=0はbeam等の系列、
            axis=1は時間sample。
        fs_hz: サンプリング周波数。単位はHz。
        reference_rms: 0 dBに対応するRMS振幅。入力と同じ振幅単位。

    Returns:
        `(frequency_hz, level_db20)`。周波数軸shapeは`[n_freq]`、単位はHz。
        level shapeは`[n_series, n_freq]`、単位はper-bin `dB re reference_rms`。

    Raises:
        ValueError: 入力shape、有限性、fs_hz、reference_rmsが不正な場合。

    境界条件:
        DCと偶数sample時のNyquistは片側だけなのでsqrt(2)補正しない。
        この出力はper-bin levelであり、band積分RMS levelではない。
    """

    signals = np.asarray(real_signals, dtype=np.float64)
    require(
        signals.ndim == 2 and signals.shape[0] > 0 and signals.shape[1] > 0,
        "real_signals must have shape (n_series, n_sample).",
    )
    require(bool(np.all(np.isfinite(signals))), "real_signals must contain only finite values.")
    require_positive_float("fs_hz", float(fs_hz))
    require_positive_float("reference_rms", float(reference_rms))

    n_sample = signals.shape[1]
    spectrum = np.fft.rfft(signals, axis=1) / np.float64(n_sample)

    # interior binをsqrt(2)倍し、実信号の負周波数側powerを正周波数側へ集約する。
    if spectrum.shape[1] > 1:
        if (n_sample % 2) == 0 and spectrum.shape[1] > 2:
            spectrum[:, 1:-1] *= np.sqrt(2.0)
        else:
            spectrum[:, 1:] *= np.sqrt(2.0)

    frequency_hz = np.fft.rfftfreq(n_sample, d=1.0 / float(fs_hz)).astype(np.float64)
    normalized_rms = np.abs(spectrum) / float(reference_rms)
    level_db20 = 20.0 * np.log10(np.maximum(normalized_rms, np.finfo(np.float64).tiny))
    return frequency_hz, np.asarray(level_db20, dtype=np.float64)


def calculate_block_rms_levels_db20(
    real_signals: FloatArray,
    fs_hz: float,
    block_size: int,
    *,
    reference_rms: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """BTR用に系列ごとの非overlap block RMS levelを計算する。

    Args:
        real_signals: 実波形。shapeは`[n_series, n_sample]`、axis=0はbeam等の系列、
            axis=1は時間sample。
        fs_hz: サンプリング周波数。単位はHz。
        block_size: RMS観測区間。単位はsample。
        reference_rms: 0 dBに対応するRMS振幅。入力と同じ振幅単位。

    Returns:
        `(level_db20, time_s)`。level shapeは`[n_time, n_series]`、axis=0はblock時刻、
        axis=1はbeam等の系列、単位は`dB re reference_rms`。time shapeは`[n_time]`、単位は秒。

    Raises:
        ValueError: 入力shape、有限性、fs_hz、block_size、reference_rmsが不正な場合。

    境界条件:
        最終端のblock_size未満のsampleは、不完全な観測値を公開しないため破棄する。
        BTR表示でframe最大値を引く場合、その後の単位は`dB re frame max`となる。
    """

    signals = np.asarray(real_signals, dtype=np.float64)
    require(
        signals.ndim == 2 and signals.shape[0] > 0 and signals.shape[1] > 0,
        "real_signals must have shape (n_series, n_sample).",
    )
    require(bool(np.all(np.isfinite(signals))), "real_signals must contain only finite values.")
    require_positive_float("fs_hz", float(fs_hz))
    require_positive_int("block_size", int(block_size))
    require_positive_float("reference_rms", float(reference_rms))

    trimmed_length = (signals.shape[1] // int(block_size)) * int(block_size)
    require(trimmed_length > 0, "signal length must be at least one RMS block.")

    # 変換前shapeは[n_series, n_time*block_size]、変換後は
    # [n_series, n_time, block_size]。axis=2で時間block内RMSを取る。
    blocked = signals[:, :trimmed_length].reshape(
        signals.shape[0],
        trimmed_length // int(block_size),
        int(block_size),
    )
    normalized_rms = np.sqrt(np.mean(blocked**2, axis=2)) / float(reference_rms)
    level_db20 = 20.0 * np.log10(np.maximum(normalized_rms, np.finfo(np.float64).tiny))
    time_s = (np.arange(level_db20.shape[1], dtype=np.float64) * int(block_size)) / float(fs_hz)
    return np.asarray(level_db20.T, dtype=np.float64), time_s


__all__ = [
    "calculate_block_rms_levels_db20",
    "calculate_one_sided_rms_spectrum_db20",
    "calculate_tone_projection_rms_level_db20",
]
