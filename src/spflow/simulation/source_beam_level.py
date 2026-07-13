"""完成重みに対するsource成分のband積分beam level計算。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow.simulation.alignment_weight_design import AlignmentWeightDesign

FloatArray = NDArray[np.floating[Any]]
ComplexArray = NDArray[np.complexfloating[Any, Any]]


def calculate_source_beam_level_db(
    weights_original: ComplexArray,
    design: AlignmentWeightDesign,
    *,
    floor_db_re_input_rms: float = -100.0,
) -> FloatArray:
    """設計source帯域を積分したtarget-only beam levelを計算する。

    Args:
        weights_original: 元入力座標の重み。shape ``[n_fft,n_beam,n_ch]``。
        design: source steering、帯域mask、shape契約を含む完成設計。
        floor_db_re_input_rms: 対数表示のpower床、単位dB re input RMS。

    Returns:
        待受beam別level。shape ``[n_beam]``、単位dB re input RMS。

    Raises:
        ValueError: 重みshapeが設計shapeと一致しない、または表示床が非有限の場合。

    Notes:
        BL形状特徴、合否判定、図表作成はこの関数の責務に含めない。
    """
    checked = np.asarray(weights_original)
    if checked.shape != design.steering.shape:
        raise ValueError("weights_original shape must match design steering shape.")
    if checked.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("weights_original dtype must be complex64 or complex128.")
    if not bool(np.all(np.isfinite(checked))):
        raise ValueError("weights_original must contain only finite values.")
    if not np.isfinite(floor_db_re_input_rms):
        raise ValueError("floor_db_re_input_rms must be finite.")
    band_weights = checked[design.source_bin_mask]
    # response[f,beam]=w[f,beam]^H a_source[f]。channel軸だけを内積として畳み込む。
    response = np.einsum(
        "fbc,fc->fb",
        band_weights.conj(),
        design.source_steering[design.source_bin_mask],
        optimize=True,
    )
    # 正負source binに等powerを置く契約では、bin平均powerが帯域積分応答に対応する。
    power = np.mean(np.abs(response) ** 2, axis=0)
    floor_power = 10.0 ** (floor_db_re_input_rms / 10.0)
    real_dtype = (
        np.dtype(np.float32) if checked.dtype == np.dtype(np.complex64) else np.dtype(np.float64)
    )
    return np.asarray(10.0 * np.log10(np.maximum(power, floor_power)), dtype=real_dtype)
