"""BL/FRAZ/BTR評価に使う方位走査gridを固定shapeで構成する。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spflow._validation import require, require_positive_int
from spflow.beamforming.directions import make_directions


@dataclass(frozen=True)
class BeamScanGrid:
    """beam方向余弦と表示軸を対応づけて保持する。

    `directions`はshape `[n_beam, 3]`、`azimuth_deg`と`elevation_deg`は
    shape `[n_axis]`である。入力は`build_beam_scan_grid`が生成した固定shape配列、
    出力はbeamformerとBL/FRAZ/BTR描画へ渡す走査軸である。

    beamforming、信号生成、level計算、描画は責務に含めない。
    信号処理上は方向余弦順序と表示方位軸の対応を失わないための境界結果型である。
    """

    directions: NDArray[np.float64]
    azimuth_deg: NDArray[np.float64]
    elevation_deg: NDArray[np.float64]
    display_elevation_index: int

    def __post_init__(self) -> None:
        """方向余弦と表示軸の固定shapeおよび対応indexを検証する。"""

        require(
            self.directions.ndim == 2 and self.directions.shape[1] == 3,
            "directions must have shape (n_beam, 3).",
        )
        require(self.directions.shape[0] > 0, "directions must not be empty.")
        require(self.azimuth_deg.ndim == 1, "azimuth_deg must have shape (n_azimuth,).")
        require(self.azimuth_deg.shape[0] > 0, "azimuth_deg must not be empty.")
        require(self.elevation_deg.ndim == 1, "elevation_deg must have shape (n_elevation,).")
        require(self.elevation_deg.shape[0] > 0, "elevation_deg must not be empty.")
        require(
            0 <= int(self.display_elevation_index) < self.elevation_deg.shape[0],
            "display_elevation_index must reference elevation_deg.",
        )
        require(
            self.directions.shape[0] == self.azimuth_deg.shape[0] * self.elevation_deg.shape[0],
            "directions count must equal n_azimuth * n_elevation.",
        )
        require(
            bool(np.all(np.isfinite(self.directions))),
            "directions must contain only finite values.",
        )
        require(
            bool(np.all(np.isfinite(self.azimuth_deg))),
            "azimuth_deg must contain only finite values.",
        )
        require(
            bool(np.all(np.isfinite(self.elevation_deg))),
            "elevation_deg must contain only finite values.",
        )


def build_beam_scan_grid(
    *,
    azimuth_min_deg: float,
    azimuth_max_deg: float,
    display_elevation_deg: float,
    n_real_azimuth_beams: int,
    n_virtual_azimuth_beams: int = 0,
) -> BeamScanGrid:
    """右舷array用のBL/FRAZ/BTR方位走査gridを構成する。

    Args:
        azimuth_min_deg: 表示する最小方位角。単位はdeg。
        azimuth_max_deg: 表示する最大方位角。単位はdeg。
        display_elevation_deg: 固定俯仰角。単位はdeg。
        n_real_azimuth_beams: 実beam数。単位は本。
        n_virtual_azimuth_beams: 補間用virtual beam数。単位は本、0以上。

    Returns:
        `directions` shape `[n_beam, 3]`、方位・俯仰表示軸を持つgrid。
        directionsのaxis=0はbeam、axis=1は`x, y, z`方向余弦。

    Raises:
        ValueError: 方位範囲、beam数、俯仰角が不正な場合。

    境界条件:
        俯仰方向は1点に固定し、`display_elevation_index`は常に0となる。
        方位軸は`make_directions`のequal-cos規約を維持し、線形角度補間へ置き換えない。
    """

    require(np.isfinite(float(azimuth_min_deg)), "azimuth_min_deg must be finite.")
    require(np.isfinite(float(azimuth_max_deg)), "azimuth_max_deg must be finite.")
    require(
        float(azimuth_min_deg) < float(azimuth_max_deg),
        "azimuth_min_deg must be smaller than azimuth_max_deg.",
    )
    require(
        np.isfinite(float(display_elevation_deg)),
        "display_elevation_deg must be finite.",
    )
    require_positive_int("n_real_azimuth_beams", int(n_real_azimuth_beams))
    require(
        int(n_virtual_azimuth_beams) >= 0,
        "n_virtual_azimuth_beams must be non-negative.",
    )

    directions, azimuth_deg, elevation_deg = make_directions(
        az_min_deg=float(azimuth_min_deg),
        az_max_deg=float(azimuth_max_deg),
        el_min_deg=float(display_elevation_deg),
        el_max_deg=float(display_elevation_deg),
        n_beam_az_real=int(n_real_azimuth_beams),
        n_beam_az_virtual=int(n_virtual_azimuth_beams),
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[float(display_elevation_deg)],
    )
    return BeamScanGrid(
        # make_directionsはshape `[3, n_beam]`なので、beamformer境界の
        # `[n_beam, 3]`へ転置し、axis=0をbeamとして固定する。
        directions=np.asarray(directions.T, dtype=np.float64),
        azimuth_deg=np.asarray(azimuth_deg, dtype=np.float64),
        elevation_deg=np.asarray(elevation_deg, dtype=np.float64),
        display_elevation_index=0,
    )


__all__ = ["BeamScanGrid", "build_beam_scan_grid"]
