"""方位別共分散から非対角の正規化空間相関統計を計算する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float


FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class SpatialCorrelationStatistics:
    """非対角チャネルペアの正規化相関統計を保持する。

    このクラスは、方位別・周波数別の最大値、平均、中央値、95 percentileと、
    等間隔ULAを想定した基線index別平均を固定shapeで返す。

    入力信号の生成、共分散の積分、画像化、閾値による採否判定は責務に含めない。
    信号処理上は、方式3で生成した共分散の方位選択性を比較する観測量に位置づく。

    Attributes:
        maximum: 全非対角pairの最大値。shapeは`[n_direction,n_bin]`。
        mean: 全非対角pairの平均。shapeは`[n_direction,n_bin]`。
        median: 全非対角pairの中央値。shapeは`[n_direction,n_bin]`。
        percentile_95: 全非対角pairの95 percentile。shapeは`[n_direction,n_bin]`。
        baseline_mean: `abs(i-j)`別平均。shapeは`[n_direction,n_bin,n_ch-1]`。
        baseline_index: 基線index。shapeは`[n_ch-1]`、値は1から`n_ch-1`。
    """

    maximum: NDArray[np.float32]
    mean: NDArray[np.float32]
    median: NDArray[np.float32]
    percentile_95: NDArray[np.float32]
    baseline_mean: NDArray[np.float32]
    baseline_index: NDArray[np.int32]


def calculate_spatial_correlation_statistics(
    direction_covariance: NDArray[Any],
    *,
    denominator_floor: float = 1.0e-20,
) -> SpatialCorrelationStatistics:
    """共分散の下三角全pairから正規化相関統計を計算する。

    Args:
        direction_covariance: 方位別共分散。shapeは`[n_direction,n_ch,n_ch,n_bin]`。
        denominator_floor: `Rii*Rjj`の最小許容power二乗。これ以下は相関0とする。

    Returns:
        最大、平均、中央値、95 percentile、および基線index別平均。
        全統計は無次元比であり、物理範囲は`[0,1]`。

    Raises:
        ValueError: 共分散shape、channel数、対角power、または安定化値が不正な場合。

    境界条件:
        自己相関1を統計へ混入させないため対角成分を除外し、下三角`i>j`だけを使う。
        複素相関を直接平均せず、各pairで`|Rij|/sqrt(Rii*Rjj)`を求めてから集約する。
        分母がfloor以下の無power pairは、数値雑音を高相関と誤認しないよう0とする。
    """

    covariance = np.asarray(direction_covariance, dtype=np.complex64)
    floor_value = float(denominator_floor)
    require_positive_float("denominator_floor", floor_value)
    require(covariance.ndim == 4, "direction_covariance must have shape (n_direction, n_ch, n_ch, n_bin).")
    require(covariance.shape[1] == covariance.shape[2], "covariance channel axes must be square.")
    require(covariance.shape[1] >= 2, "off-diagonal correlation requires at least two channels.")

    # diagonal_power shapeは`[n_direction,n_ch,n_bin]`。Hermitian共分散の対角は実powerである。
    diagonal_power = np.real(np.diagonal(covariance, axis1=1, axis2=2)).transpose(0, 2, 1)
    require(bool(np.all(diagonal_power >= -np.finfo(np.float32).eps)), "covariance diagonal power must be non-negative.")
    diagonal_power = np.maximum(diagonal_power, np.float32(0.0))

    # 下三角pair軸を明示的に作り、correlation shapeを`[n_direction,n_pair,n_bin]`へ揃える。
    first_channels, second_channels = np.tril_indices(covariance.shape[1], k=-1)
    denominator_power = diagonal_power[:, first_channels, :] * diagonal_power[:, second_channels, :]
    valid = denominator_power > np.float32(floor_value)
    pair_correlation = np.zeros(denominator_power.shape, dtype=np.float32)
    cross_magnitude = np.abs(covariance[:, first_channels, second_channels, :])
    pair_correlation[valid] = np.asarray(
        cross_magnitude[valid] / np.sqrt(denominator_power[valid]),
        dtype=np.float32,
    )
    require(bool(np.all(pair_correlation <= 1.0 + 1.0e-4)), "normalized correlation exceeds its physical range.")
    pair_correlation = np.clip(pair_correlation, 0.0, 1.0).astype(np.float32, copy=False)

    maximum = np.asarray(np.max(pair_correlation, axis=1), dtype=np.float32)
    mean = np.asarray(np.mean(pair_correlation, axis=1), dtype=np.float32)
    median = np.asarray(np.median(pair_correlation, axis=1), dtype=np.float32)
    percentile_95 = np.asarray(np.percentile(pair_correlation, 95.0, axis=1), dtype=np.float32)

    baseline_index = np.arange(1, covariance.shape[1], dtype=np.int32)
    baseline_mean = np.empty(
        (covariance.shape[0], covariance.shape[3], baseline_index.size),
        dtype=np.float32,
    )
    pair_baseline_index = first_channels - second_channels
    for output_index, channel_separation in enumerate(baseline_index):
        # 等間隔ULAでは`abs(i-j)`が同じpairは同じ物理基線長を持つため、そのpair軸だけを平均する。
        selected_pairs = pair_baseline_index == channel_separation
        baseline_mean[:, :, output_index] = np.asarray(
            np.mean(pair_correlation[:, selected_pairs, :], axis=1),
            dtype=np.float32,
        )

    return SpatialCorrelationStatistics(
        maximum=maximum,
        mean=mean,
        median=median,
        percentile_95=percentile_95,
        baseline_mean=baseline_mean,
        baseline_index=baseline_index,
    )
