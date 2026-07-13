"""小数遅延固定整相器の理論beam-to-beam応答を評価する。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow.beamforming.time_delay import FractionalDelayAndSumBeamformer

FloatArray = NDArray[np.floating[Any]]


def normalize_evaluation_channel_weights(
    channel_weights: FloatArray | None,
    *,
    n_ch: int,
) -> NDArray[np.float64]:
    """評価時のchannel shadingを非負の1次元配列へ正規化する。

    Args:
        channel_weights: channel shading。shapeは`[n_ch]`、無次元。
            Noneの場合は全channelを1.0とする矩形重み。
        n_ch: active channel数。単位は本。

    Returns:
        `float64`重み。shapeは`[n_ch]`。

    Raises:
        ValueError: n_chが正でない、shape不一致、非有限値、負値、係数和0の場合。

    境界条件:
        beam出力と同じ`sum(w*x)/sum(w)`を後段で使うため、ここでは係数和1への
        正規化を行わず、入力shadingの相対比を保持する。
    """

    if int(n_ch) <= 0:
        raise ValueError("n_ch must be positive.")
    if channel_weights is None:
        return np.ones(int(n_ch), dtype=np.float64)

    weights = np.asarray(channel_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != int(n_ch):
        raise ValueError("channel_weights must have shape (n_ch,).")
    if not bool(np.all(np.isfinite(weights))):
        raise ValueError("channel_weights must contain only finite values.")
    if bool(np.any(weights < 0.0)):
        raise ValueError("channel_weights must be non-negative.")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("channel_weights must contain positive total weight.")
    return weights


def calculate_fractional_beam_response_matrix(
    beamformer: FractionalDelayAndSumBeamformer,
    frequency_hz: float,
    channel_weights: FloatArray | None = None,
) -> NDArray[np.complex128]:
    """小数遅延固定整相の理論beam-to-beam応答行列を計算する。

    Args:
        beamformer: 小数遅延固定整相器。delay tableとFIR周波数応答を保持する。
        frequency_hz: 評価周波数。単位はHz。
        channel_weights: active channel shading。shapeは`[n_ch]`、無次元。

    Returns:
        複素応答行列。shapeは`[n_observation_beam, n_look_beam]`。
        axis=0は固定整相の待受beam、axis=1は入力source方向である。

    Raises:
        ValueError: 周波数が正でない、またはchannel重みが不正な場合。

    境界条件:
        channel shadingを含む実beam出力と同じ`sum(w*x)/sum(w)`で正規化し、
        SLC blocking matrixの主ローブ保護条件と評価応答を一致させる。
    """

    if not np.isfinite(float(frequency_hz)) or float(frequency_hz) <= 0.0:
        raise ValueError("frequency_hz must be positive and finite.")

    arrival_delay_s = np.asarray(
        beamformer.delay_table.arrival_delay_sec,
        dtype=np.float64,
    )
    weights = normalize_evaluation_channel_weights(
        channel_weights,
        n_ch=int(arrival_delay_s.shape[0]),
    )
    weight_sum = float(np.sum(weights))

    # steering_response[ch, observation_beam]は整数遅延位相と小数遅延FIR応答を含む。
    # arrival_phase[ch, look_beam]=exp(-j*2*pi*f*tau)はsource到来位相である。
    steering_response = np.asarray(
        beamformer.steering_response(float(frequency_hz)),
        dtype=np.complex128,
    )
    arrival_phase = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * arrival_delay_s)
    # weights[:, None]をbroadcastし、shape `[n_ch, n_look_beam]`の到来位相へ
    # channel shadingを適用してからchannel軸を行列積で縮約する。
    weighted_arrival_phase = weights[:, np.newaxis] * arrival_phase
    return np.asarray(
        (steering_response.T @ weighted_arrival_phase) / weight_sum,
        dtype=np.complex128,
    )


__all__ = [
    "calculate_fractional_beam_response_matrix",
    "normalize_evaluation_channel_weights",
]
