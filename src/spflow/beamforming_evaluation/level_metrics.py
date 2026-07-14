"""beamforming評価で共有するRMS level計算を実装する。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow._validation import require, require_positive_float
from spflow.level_conversion import LevelConverter, level_20log10_rms


def _rms_level_converter(reference_rms: float) -> LevelConverter:
    """評価RMSを明示referenceへ写像するconverterを生成する。"""

    definition = level_20log10_rms(
        reference_rms=float(reference_rms),
        reference_label="reference RMS",
    )
    return LevelConverter.for_definition(definition)


def _resolve_rms_level_converter(
    *,
    reference_rms: float | None,
    level_converter: LevelConverter | None,
) -> LevelConverter:
    """互換reference引数または共有Converterのどちらか一方を確定する。"""

    if level_converter is not None:
        require(
            reference_rms is None,
            "reference_rms and level_converter must not be specified together.",
        )
        return level_converter
    effective_reference_rms = 1.0 if reference_rms is None else float(reference_rms)
    return _rms_level_converter(effective_reference_rms)


def calculate_rms_level_db20(
    signal: NDArray[Any],
    *,
    reference_rms: float | None = None,
    level_converter: LevelConverter | None = None,
) -> float:
    """実数または複素信号のRMS levelを計算する。

    Args:
        signal: 評価信号。shapeは任意で、全要素を同じ観測区間としてRMS化する。
        reference_rms: 0 dBに対応するRMS振幅。`level_converter`未指定時だけ使い、
            `None`の場合は1 RMSとする。
        level_converter: 入力生成時から共有するlevel変換器。output definitionはRMSとする。

    Returns:
        `20*log10(signal_rms/reference_rms)`。単位は`dB re reference_rms`。

    Raises:
        ValueError: signalが空、非有限値を含む、referenceが不正、または
            reference_rmsとlevel_converterを同時指定した場合。

    境界条件:
        完全なゼロ信号はfloat64の最小正規化数を下限とし、JSONへ保存可能な有限値を返す。
    """

    values = np.asarray(signal)
    require(values.size > 0, "signal must not be empty.")
    require(bool(np.all(np.isfinite(values))), "signal must contain only finite values.")
    rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
    converter = _resolve_rms_level_converter(
        reference_rms=reference_rms,
        level_converter=level_converter,
    )
    return converter.output_rms_to_level(rms, floor_db=converter.float64_tiny_level_db)


def calculate_real_tone_response_rms_level_db20(
    positive_frequency_response: NDArray[Any],
    negative_frequency_response: NDArray[Any],
    source_rms: float,
    *,
    reference_rms: float | None = None,
    level_converter: LevelConverter | None = None,
) -> NDArray[np.float64]:
    """実toneの正負周波数応答から出力RMS levelを計算する。

    Args:
        positive_frequency_response: `+f`側の複素応答。shapeは任意。
        negative_frequency_response: `-f`側の複素応答。shapeは正側応答と同じ。
        source_rms: 入力toneのRMS振幅。信号振幅単位。
        reference_rms: 0 dBに対応するRMS振幅。`level_converter`未指定時だけ使い、
            `None`の場合は1 RMSとする。
        level_converter: 入力生成時から共有するlevel変換器。output definitionはRMSとする。

    Returns:
        入力応答と同じshapeの出力RMS level。単位は`dB re reference_rms`。

    Raises:
        ValueError: 応答shapeが一致しない、非有限値を含む、source_rmsまたは
            referenceが不正、またはreference_rmsとlevel_converterを同時指定した場合。

    境界条件:
        複素適応重みでは`H(-f)=conj(H(+f))`とは限らないため、片側応答だけで
        levelを決めない。正負両側のpowerを平均して実時間波形のRMSへ対応させる。
    """

    positive_response = np.asarray(positive_frequency_response, dtype=np.complex128)
    negative_response = np.asarray(negative_frequency_response, dtype=np.complex128)
    require(
        positive_response.shape == negative_response.shape,
        "positive and negative responses must have the same shape.",
    )
    require(
        bool(np.all(np.isfinite(positive_response)))
        and bool(np.all(np.isfinite(negative_response))),
        "positive and negative responses must contain only finite values.",
    )
    require_positive_float("source_rms", float(source_rms))

    # 実toneは正負周波数へpeak/2ずつ分かれ、時間平均では交差項が消える。
    # output_rms = source_rms*sqrt((|H(+f)|^2+|H(-f)|^2)/2)。
    output_rms = float(source_rms) * np.sqrt(
        (np.abs(positive_response) ** 2 + np.abs(negative_response) ** 2) / 2.0
    )
    converter = _resolve_rms_level_converter(
        reference_rms=reference_rms,
        level_converter=level_converter,
    )
    return np.asarray(
        converter.output_rms_to_level(
            output_rms,
            floor_db=converter.float64_tiny_level_db,
        ),
        dtype=np.float64,
    )


__all__ = [
    "calculate_real_tone_response_rms_level_db20",
    "calculate_rms_level_db20",
]
