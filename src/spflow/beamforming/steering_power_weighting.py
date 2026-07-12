"""周波数別channel shadingをsteering power整合量へ適用する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require


@dataclass(frozen=True)
class SteeringPowerChannelWeighting:
    """steering power用の周波数別channel重みと理論noise基準を保持する。

    このクラスは物理steeringと非負shading係数から、重み付き内積に使うprojection table、
    active channel数、有効channel数、空間白色雑音のeta理論値を固定shapeで返す。

    snapshot FFT、指数積分、threshold校正、共分散合成は責務に含めない。
    信号処理上は、異なる総channel数と周波数別shadingを同じeta定義へ接続する。

    Attributes:
        channel_weight_table: shading係数。shapeは`[n_ch,n_bin]`、非負の線形値。
        projection_table: `g*a/sqrt(sum(g*|a|^2))`。shapeは`[n_ch,n_bin,n_direction]`。
        active_channel_count: 正の係数を持つchannel数。shapeは`[n_bin]`。
        effective_channel_count: `(sum(g))^2/sum(g^2)`。shapeは`[n_bin]`。
        noise_eta_reference: 空間白色雑音の`E[eta]=1/N_eff`。shapeは`[n_bin]`。
    """

    channel_weight_table: NDArray[np.float32]
    projection_table: NDArray[np.complex64]
    active_channel_count: NDArray[np.int32]
    effective_channel_count: NDArray[np.float32]
    noise_eta_reference: NDArray[np.float32]


def prepare_steering_power_channel_weighting(
    steering_table: NDArray[Any],
    channel_weight_table: NDArray[Any] | None = None,
) -> SteeringPowerChannelWeighting:
    """物理steeringと周波数別shadingから重み付きsteering power係数を作る。

    Args:
        steering_table: 物理steering。shapeは`[n_ch,n_bin,n_direction]`、無次元複素値。
        channel_weight_table: channel shading。shapeは`[n_ch,n_bin]`、非負の線形値。
            `None`では全channelを係数1で使用する。

    Returns:
        projection table、active channel数、`N_eff`、noise eta理論値を持つ固定結果。

    Raises:
        ValueError: steeringが3次元でない、重みshapeが一致しない、非有限・負の係数を含む、
            または全係数0の周波数binが存在する場合。

    境界条件:
        係数0のchannelはそのbinのsteering powerとtotal powerへ寄与しない。
        係数全体のscaleはetaで相殺されるため、入力値を正規化せず保持する。
    """

    steering = np.asarray(steering_table, dtype=np.complex64)
    require(steering.ndim == 3, "steering_table must have shape (n_ch, n_bin, n_direction).")
    require(bool(np.all(np.isfinite(steering))), "steering_table must be finite.")
    n_ch, n_bin, _ = steering.shape
    if channel_weight_table is None:
        weights = np.ones((n_ch, n_bin), dtype=np.float32)
    else:
        weights = np.asarray(channel_weight_table, dtype=np.float32)
        require(weights.shape == (n_ch, n_bin), "channel_weight_table must have shape (n_ch, n_bin).")
        require(bool(np.all(np.isfinite(weights))), "channel_weight_table must be finite.")
        require(bool(np.all(weights >= 0.0)), "channel_weight_table must be non-negative.")

    weight_sum = np.sum(weights, axis=0, dtype=np.float64)
    weight_power_sum = np.sum(weights.astype(np.float64) ** 2, axis=0)
    require(bool(np.all(weight_sum > 0.0)), "every frequency bin must use at least one channel.")
    require(bool(np.all(weight_power_sum > 0.0)), "every frequency bin must have positive weight power.")

    # weighted_norm_squared[bin,direction]はa^H G a。一般の振幅付きsteeringでも
    # target整合時eta=1となるよう、単純なsum(g)ではなく|a|^2を含めて正規化する。
    weighted_norm_squared = np.einsum(
        "ik,ikb->kb",
        weights,
        np.abs(steering) ** 2,
        optimize=True,
    )
    require(bool(np.all(weighted_norm_squared > 0.0)), "weighted steering norm must be positive.")
    # projection=g*a/sqrt(a^H G a)をXへ適用すると、
    # |projection^H X|^2 / sum(g|X|^2)が重み付きetaになる。
    projection_table = np.asarray(
        weights[:, :, np.newaxis] * steering / np.sqrt(weighted_norm_squared)[np.newaxis, :, :],
        dtype=np.complex64,
    )
    effective_channel_count = np.asarray(
        (weight_sum * weight_sum) / weight_power_sum,
        dtype=np.float32,
    )
    noise_eta_reference = np.asarray(1.0 / effective_channel_count, dtype=np.float32)
    active_channel_count = np.asarray(np.count_nonzero(weights > 0.0, axis=0), dtype=np.int32)
    return SteeringPowerChannelWeighting(
        channel_weight_table=weights.copy(),
        projection_table=projection_table,
        active_channel_count=active_channel_count,
        effective_channel_count=effective_channel_count,
        noise_eta_reference=noise_eta_reference,
    )
