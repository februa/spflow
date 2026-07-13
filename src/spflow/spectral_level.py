"""RMS level と one-sided spectrum の変換規約を実装するモジュール。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._validation import require, require_positive_float, require_positive_int
from .level_conversion import (
    LevelConverter,
    level_20log10_onesided_asd,
    level_20log10_rms,
)


def _normalized_rms_converter(reference_rms: float = 1.0) -> LevelConverter:
    """既存RMS変換関数をLevelConverterへ接続する互換converterを生成する。"""

    definition = level_20log10_rms(
        reference_rms=float(reference_rms),
        reference_label="reference RMS",
    )
    return LevelConverter(input_definition=definition, output_definition=definition)


def level_db_to_rms_amplitude(level_db_re_rms: float) -> float:
    """RMS level を線形 RMS amplitude へ変換する。

    Args:
        level_db_re_rms: RMS level。単位は `dB re reference RMS`。

    Returns:
        `10^(level/20)` で得る RMS amplitude。基準量が 1 RMS のとき、0 dB は 1。

    Raises:
        ValueError: level が有限値でない場合。

    境界条件:
        戻り値は RMS amplitude であり、実 cos 波へ直接渡す peak amplitude ではない。
    """
    return _normalized_rms_converter().input_to_rms(float(level_db_re_rms))


def tone_rms_level_db_to_peak_amplitude(level_db_re_rms: float) -> float:
    """実正弦波の RMS level を時間波形の peak amplitude へ変換する。

    Args:
        level_db_re_rms: 正弦波 RMS level。単位は `dB re reference RMS`。

    Returns:
        peak amplitude。単位は基準 RMS と同じ線形振幅単位。

    Raises:
        ValueError: level が有限値でない場合。

    境界条件:
        0 dB は RMS=1、peak=`sqrt(2)` とする。実正弦波の
        `RMS = peak / sqrt(2)` に対応し、複素指数信号には適用しない。
    """
    return _normalized_rms_converter().input_to_real_cosine_peak(float(level_db_re_rms))


def noise_asd_level_db_to_band_rms(
    level_db_re_rms_per_sqrt_hz: float,
    *,
    bandwidth_hz: float,
) -> float:
    """one-sided 白色雑音 ASD level を指定帯域内の RMS amplitude へ変換する。

    Args:
        level_db_re_rms_per_sqrt_hz: one-sided ASD level。
            単位は `dB re reference RMS/sqrt(Hz)`。
        bandwidth_hz: 積分する one-sided 帯域幅。単位は Hz。

    Returns:
        `10^(NL/20) * sqrt(bandwidth_hz)` で得る帯域内 RMS amplitude。

    Raises:
        ValueError: level が有限値でない、または帯域幅が正でない場合。

    境界条件:
        bandwidth は FFT bin 数ではなく Hz で与える。1 bin の RMS が必要な場合は、
        `bandwidth_hz=frequency_resolution_hz` とする。
    """
    bandwidth = float(bandwidth_hz)
    require_positive_float("bandwidth_hz", bandwidth)
    asd_definition = level_20log10_onesided_asd(
        reference_asd=1.0,
        reference_label="reference RMS/sqrt(Hz)",
    )
    asd_converter = LevelConverter(
        input_definition=asd_definition,
        output_definition=asd_definition,
    )
    linear_asd = asd_converter.input_to_linear(float(level_db_re_rms_per_sqrt_hz))

    # ASDは振幅/√Hzなので、一定密度を明示された帯域Bでpower積分すると
    # A_band=sqrt(ASD²*B)=ASD*sqrt(B)となる。Converterは帯域積分を担わない。
    return float(linear_asd * np.sqrt(bandwidth))


def noise_asd_level_db_to_sample_rms(
    level_db_re_rms_per_sqrt_hz: float,
    *,
    sampling_frequency_hz: float,
) -> float:
    """one-sided 白色雑音 ASD level を実時間波形の sample RMS へ変換する。

    Args:
        level_db_re_rms_per_sqrt_hz: one-sided amplitude spectral density level。
            単位は `dB re reference RMS/sqrt(Hz)`。
        sampling_frequency_hz: sampling frequency。単位は Hz。

    Returns:
        実白色雑音の sample RMS。単位は基準 RMS と同じ線形振幅単位。

    Raises:
        ValueError: level が有限値でない、または sampling frequency が正でない場合。

    境界条件:
        実信号の one-sided bandwidth を `fs/2` Hz とし、ASD の二乗を帯域積分する。
        したがって sample RMS は `ASD * sqrt(fs/2)` となる。
    """
    fs_hz = float(sampling_frequency_hz)
    require_positive_float("sampling_frequency_hz", fs_hz)
    return noise_asd_level_db_to_band_rms(
        level_db_re_rms_per_sqrt_hz,
        bandwidth_hz=fs_hz / 2.0,
    )


def one_sided_rfft_bin_rms_power(
    spectrum: NDArray[Any],
    *,
    sample_count: int,
    frequency_axis: int = -1,
) -> NDArray[np.float64]:
    """非正規化 rFFT の各 bin を one-sided RMS power へ変換する。

    Args:
        spectrum: `np.fft.rfft` と同じ非正規化複素 spectrum。
            frequency axis の長さは `sample_count // 2 + 1`。
        sample_count: FFT に使用した実時間サンプル数。単位は sample。
        frequency_axis: 周波数 bin 軸。負の axis も許容する。

    Returns:
        one-sided RMS power。shape と axis 配置は `spectrum` と同じ。
        内部正周波数 bin は `2|X[k]|^2/N^2`、DC と偶数長 FFT の
        Nyquist bin は `|X[k]|^2/N^2` とする。

    Raises:
        ValueError: spectrum が空、axis が不正、または bin 数が FFT 長と一致しない場合。

    境界条件:
        奇数長 FFT には Nyquist bin が存在しないため、最後の正周波数 bin も 2 倍する。
        DC と Nyquist を 2 倍しないことで Parseval の RMS power と一致させる。
    """
    require_positive_int("sample_count", int(sample_count))
    values = np.asarray(spectrum)
    require(values.ndim > 0, "spectrum must have at least one axis.")
    axis = int(frequency_axis)
    if axis < 0:
        axis += values.ndim
    require(0 <= axis < values.ndim, "frequency_axis is out of bounds.")
    expected_bin_count = int(sample_count) // 2 + 1
    require(
        values.shape[axis] == expected_bin_count,
        "spectrum frequency-axis length must equal sample_count // 2 + 1.",
    )

    # 非正規化 DFT の Parseval 対応は |X[k]|^2/N^2 である。
    # one-sided 表現では負周波数側を省略するため、共役対を持つ内部 bin だけ 2 倍する。
    power = np.asarray(np.abs(values) ** 2 / float(sample_count) ** 2, dtype=np.float64)
    correction = np.full(expected_bin_count, 2.0, dtype=np.float64)
    correction[0] = 1.0
    if int(sample_count) % 2 == 0:
        correction[-1] = 1.0
    correction_shape = [1] * values.ndim
    correction_shape[axis] = expected_bin_count
    return power * correction.reshape(correction_shape)


def integrate_one_sided_band_rms_power(
    bin_rms_power: NDArray[Any],
    band_mask: NDArray[np.bool_],
    *,
    frequency_axis: int = -1,
) -> NDArray[np.float64]:
    """one-sided bin RMS power を指定帯域で積分する。

    Args:
        bin_rms_power: one-sided RMS power。shape は任意で、周波数軸を一つ持つ。
        band_mask: 積分対象 bin mask。shape は `[n_bin]`。
        frequency_axis: `bin_rms_power` の周波数軸。

    Returns:
        周波数軸を除いた band-integrated RMS power。単位は入力振幅単位の二乗。

    Raises:
        ValueError: shape、axis、mask が不正、または負 power が含まれる場合。

    境界条件:
        空の band は level の意味を持たないため許可しない。narrowband tone も broadband も、
        同じ bin power の和として扱う。
    """
    power = np.asarray(bin_rms_power, dtype=np.float64)
    mask = np.asarray(band_mask, dtype=np.bool_)
    require(power.ndim > 0, "bin_rms_power must have at least one axis.")
    axis = int(frequency_axis)
    if axis < 0:
        axis += power.ndim
    require(0 <= axis < power.ndim, "frequency_axis is out of bounds.")
    require(mask.ndim == 1, "band_mask must have shape (n_bin,).")
    require(mask.size == power.shape[axis], "band_mask length must match the frequency axis.")
    require(bool(np.any(mask)), "band_mask must select at least one bin.")
    require(bool(np.all(np.isfinite(power))), "bin_rms_power must be finite.")
    require(bool(np.all(power >= 0.0)), "bin_rms_power must be non-negative.")
    selected = np.compress(mask, power, axis=axis)
    return np.asarray(np.sum(selected, axis=axis), dtype=np.float64)


def rms_amplitude_to_level_db(
    rms_amplitude: NDArray[Any] | float,
    *,
    reference_rms: float = 1.0,
    floor_db: float | None = None,
) -> NDArray[np.float64]:
    """RMS amplitude を基準量付き dB level へ変換する。

    Args:
        rms_amplitude: 非負 RMS amplitude。shape は任意。
        reference_rms: 0 dB に対応する RMS amplitude。
        floor_db: 表示・保存用の下限 dB。`None` の場合、0 amplitude は `-inf`。

    Returns:
        `20 log10(rms_amplitude/reference_rms)`。shape は入力と同じ。

    Raises:
        ValueError: reference が正でない、振幅が負、または有限でない場合。

    境界条件:
        floor は数値安定化ではなく表示契約である。指定時のみ線形振幅を floor 相当値へ
        clip し、微小な浮動小数残差を可視ピークと誤認しないようにする。
    """
    amplitude = np.asarray(rms_amplitude, dtype=np.float64)
    converter = _normalized_rms_converter(float(reference_rms))
    level = converter.output_rms_to_level(amplitude, floor_db=floor_db)
    return np.asarray(level, dtype=np.float64)
