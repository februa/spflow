"""逐次処理の値と完成状態を同じshapeで運ぶ結果型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

NumericArray = NDArray[Any]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SignalBlock:
    """逐次処理の値と、その各sampleが完成済みかを保持する。

    ``data`` と ``valid_mask`` はともに shape ``[n_series,n_sample]`` である。
    この結果型は完成状態を運ぶだけで、遅延、FIR、時刻、単位、処理周期を決めない。
    """

    data: NumericArray
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        """値と完成maskのshape・dtype契約を検証する。

        Raises:
            ValueError: 配列が2次元でない、shapeが一致しない、またはdtypeが不正な場合。
        """
        data = np.asarray(self.data)
        valid = np.asarray(self.valid_mask)
        if data.ndim != 2 or data.shape != valid.shape:
            raise ValueError("data and valid_mask must have the same 2-D shape.")
        supported_dtypes = (
            np.dtype(np.float32),
            np.dtype(np.float64),
            np.dtype(np.complex64),
            np.dtype(np.complex128),
        )
        if data.dtype not in supported_dtypes:
            raise ValueError("data dtype must be float32, float64, complex64, or complex128.")
        if valid.dtype != np.dtype(np.bool_):
            raise ValueError("valid_mask dtype must be bool.")
