"""整相方式の座標識別子と整数遅延座標変換。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating[Any, Any]]

ALIGNMENT_METHOD_IDS = ("S1", "S2a", "T1", "T2a")


def to_original_input_coordinates(
    method: str, weights: ComplexArray, integer_phase: ComplexArray
) -> ComplexArray:
    """残差座標のS2a/T2a重みを元入力座標の等価重みへ変換する。

    Args:
        method: ``S1``、``S2a``、``T1``、``T2a``のいずれか。
        weights: 当該方式座標の重み。shape ``[n_fft,n_beam,n_ch]``。
        integer_phase: 整数遅延位相D。weightsと同じshape。

    Returns:
        元入力座標の重み。shapeは入力と同じ。

    Raises:
        ValueError: methodまたはshapeが不正な場合。

    Notes:
        重み設計やFIR近似はこの関数の責務に含めない。
    """
    if method not in ALIGNMENT_METHOD_IDS:
        raise ValueError(f"unknown alignment method: {method}")
    checked = np.asarray(weights)
    phase = np.asarray(integer_phase)
    if checked.shape != phase.shape or checked.ndim != 3:
        raise ValueError("weights and integer_phase must have the same 3-D shape.")
    if checked.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("weights dtype must be complex64 or complex128.")
    if phase.dtype != checked.dtype:
        raise ValueError("weights and integer_phase dtype must match.")
    if not bool(np.all(np.isfinite(checked))) or not bool(np.all(np.isfinite(phase))):
        raise ValueError("weights and integer_phase must be finite.")
    if method in ("S2a", "T2a"):
        # y=v^H D xなので、元入力座標の等価weightはD^H v=conj(D)*vである。
        return np.asarray(phase.conj() * checked, dtype=checked.dtype)
    return checked.copy()
