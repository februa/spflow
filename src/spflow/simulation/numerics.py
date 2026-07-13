"""シミュレーション配列へ伝播させる数値精度の契約。"""

from __future__ import annotations

from enum import Enum

import numpy as np


class SimulationPrecision(str, Enum):
    """実数・複素数を対にして選ぶシミュレーション精度。

    ``SINGLE`` は ``float32/complex64``、``DOUBLE`` は
    ``float64/complex128`` に対応する。配列のdtypeを生成元から後段へ伝播させるための
    値であり、変更可能なprocess-global設定は持たない。
    """

    SINGLE = "single"
    DOUBLE = "double"

    @property
    def real_dtype(self) -> np.dtype[np.float32] | np.dtype[np.float64]:
        """対応するNumPy実数dtypeを返す。"""
        if self is SimulationPrecision.SINGLE:
            return np.dtype(np.float32)
        return np.dtype(np.float64)

    @property
    def complex_dtype(self) -> np.dtype[np.complex64] | np.dtype[np.complex128]:
        """対応するNumPy複素dtypeを返す。"""
        if self is SimulationPrecision.SINGLE:
            return np.dtype(np.complex64)
        return np.dtype(np.complex128)
