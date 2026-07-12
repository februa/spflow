"""方位別共分散の複素steering整合量と固有空間指標を計算する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float, require_positive_int


@dataclass(frozen=True)
class CovarianceSubspaceMetrics:
    """複素steering整合量と主固有空間の評価値を保持する。

    このクラスは方位・周波数別に、共分散powerのsteering方向集中度、主固有ベクトルと
    steeringの一致度、最大固有値占有率、最大固有値対雑音固有値平均比を保持する。

    共分散積分、steering生成、信号生成、閾値による採否は責務に含めない。
    信号処理上は、絶対値相関では失われる複素位相と信号・雑音部分空間分離を観測する。

    Attributes:
        steering_power_fraction: `Re(a^H R a)/(trace(R)*a^H a)`。shapeは`[n_direction,n_bin]`。
        principal_eigenvector_alignment: `|a^H u1|^2/(a^H a)`。shapeは`[n_direction,n_bin]`。
        principal_eigenvalue_fraction: `lambda1/sum(lambda)`。shapeは`[n_direction,n_bin]`。
        principal_to_noise_mean_ratio: `lambda1/mean(lambda2...)`。shapeは`[n_direction,n_bin]`。
        trace_power: 共分散trace。shapeは`[n_direction,n_bin]`。
        principal_eigenvalue_gap_fraction: `(lambda1-lambda2)/trace(R)`。
        steering_rank_one_residual: 最適steering rank-1 modelのFrobenius相対残差。
    """

    steering_power_fraction: NDArray[np.float32]
    principal_eigenvector_alignment: NDArray[np.float32]
    principal_eigenvalue_fraction: NDArray[np.float32]
    principal_to_noise_mean_ratio: NDArray[np.float32]
    trace_power: NDArray[np.float32]
    principal_eigenvalue_gap_fraction: NDArray[np.float32]
    steering_rank_one_residual: NDArray[np.float32]


def calculate_covariance_subspace_metrics(
    direction_covariance: NDArray[Any],
    steering: NDArray[Any],
    *,
    denominator_floor: float = 1.0e-20,
    direction_chunk_size: int = 8,
) -> CovarianceSubspaceMetrics:
    """方位別共分散と同一方位steeringから複素整合・固有空間指標を計算する。

    Args:
        direction_covariance: 共分散。shapeは`[n_direction,n_ch,n_ch,n_bin]`。
        steering: 候補方位steering。shapeは`[n_ch,n_direction,n_bin]`。
        denominator_floor: traceや雑音固有値平均の安定化下限。
        direction_chunk_size: 一度に固有値分解する方位数。

    Returns:
        4種類の評価値。各shapeは`[n_direction,n_bin]`、dtypeは`float32`。

    Raises:
        ValueError: shape、channel数、有限性、安定化値、chunk数が不正な場合。

    境界条件:
        powerがfloor以下のbinは全指標を0とする。最大固有値が重複する白色共分散では
        主固有ベクトル方向が一意でないため、alignment単独ではなく最大固有値占有率と併記する。
    """

    covariance = np.asarray(direction_covariance, dtype=np.complex64)
    steering_vector = np.asarray(steering, dtype=np.complex64)
    floor_value = float(denominator_floor)
    chunk_size = int(direction_chunk_size)
    require_positive_float("denominator_floor", floor_value)
    require_positive_int("direction_chunk_size", chunk_size)
    require(covariance.ndim == 4, "direction_covariance must have shape (n_direction, n_ch, n_ch, n_bin).")
    require(covariance.shape[1] == covariance.shape[2], "covariance channel axes must be square.")
    require(covariance.shape[1] >= 2, "subspace metrics require at least two channels.")
    require(
        steering_vector.shape == (covariance.shape[1], covariance.shape[0], covariance.shape[3]),
        "steering must have shape (n_ch, n_direction, n_bin).",
    )
    require(bool(np.all(np.isfinite(covariance))), "direction_covariance must be finite.")
    require(bool(np.all(np.isfinite(steering_vector))), "steering must be finite.")

    output_shape = (covariance.shape[0], covariance.shape[3])
    steering_power_fraction = np.zeros(output_shape, dtype=np.float32)
    principal_eigenvector_alignment = np.zeros(output_shape, dtype=np.float32)
    principal_eigenvalue_fraction = np.zeros(output_shape, dtype=np.float32)
    principal_to_noise_mean_ratio = np.zeros(output_shape, dtype=np.float32)
    trace_power_output = np.zeros(output_shape, dtype=np.float32)
    principal_eigenvalue_gap_fraction = np.zeros(output_shape, dtype=np.float32)
    steering_rank_one_residual = np.zeros(output_shape, dtype=np.float32)

    for direction_start in range(0, covariance.shape[0], chunk_size):
        direction_stop = min(direction_start + chunk_size, covariance.shape[0])
        # covariance_chunkは`[direction,bin,ch,ch]`へ移し、NumPyのbatched eighへ渡す。
        covariance_chunk = np.moveaxis(
            covariance[direction_start:direction_stop],
            3,
            1,
        )
        # 有限snapshot誤差で生じる微小な非Hermitian成分を平均し、実固有値分解を保証する。
        covariance_hermitian = np.asarray(
            0.5 * (covariance_chunk + np.swapaxes(covariance_chunk.conj(), -1, -2)),
            dtype=np.complex64,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_hermitian)
        eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=np.float32), np.float32(0.0))
        principal_eigenvector = np.asarray(eigenvectors[..., -1], dtype=np.complex64)

        # steering_chunk shapeは`[direction,bin,ch]`。各候補方位と同じslotのsteeringを使う。
        steering_chunk = np.moveaxis(
            steering_vector[:, direction_start:direction_stop, :],
            0,
            -1,
        )
        steering_norm_squared = np.sum(np.abs(steering_chunk) ** 2, axis=-1)
        trace_power = np.sum(eigenvalues, axis=-1)
        valid_power = trace_power > np.float32(floor_value)

        # a^H R aは複素位相を保持した二次形式であり、|Rij|を先に取る相関統計とは異なる。
        steering_quadratic_power = np.real(
            np.einsum(
                "...i,...ij,...j->...",
                steering_chunk.conj(),
                covariance_hermitian,
                steering_chunk,
                optimize=True,
            )
        )
        steering_denominator = trace_power * steering_norm_squared
        steering_fraction_chunk = np.zeros(trace_power.shape, dtype=np.float32)
        valid_steering = valid_power & (steering_denominator > np.float32(floor_value))
        steering_fraction_chunk[valid_steering] = np.asarray(
            steering_quadratic_power[valid_steering] / steering_denominator[valid_steering],
            dtype=np.float32,
        )

        principal_inner_product = np.sum(steering_chunk.conj() * principal_eigenvector, axis=-1)
        alignment_chunk = np.zeros(trace_power.shape, dtype=np.float32)
        valid_norm = steering_norm_squared > np.float32(floor_value)
        alignment_chunk[valid_norm] = np.asarray(
            np.abs(principal_inner_product[valid_norm]) ** 2 / steering_norm_squared[valid_norm],
            dtype=np.float32,
        )
        eigenvalue_fraction_chunk = np.zeros(trace_power.shape, dtype=np.float32)
        eigenvalue_fraction_chunk[valid_power] = eigenvalues[..., -1][valid_power] / trace_power[valid_power]
        eigenvalue_gap_chunk = np.zeros(trace_power.shape, dtype=np.float32)
        eigenvalue_gap_chunk[valid_power] = (
            eigenvalues[..., -1][valid_power] - eigenvalues[..., -2][valid_power]
        ) / trace_power[valid_power]

        # u=a/||a||はnorm 1なので、最小二乗rank-1 modelのpは
        # u^H R u=(a^H R a)/(a^H a)。未正規化aの二次形式をそのままpにすると、
        # channel数の二乗だけ過大になり、白色共分散でも残差を0へ誤クリップしてしまう。
        # ||R-puu^H||_F^2=||R||_F^2-p^2を使い、巨大なmodel行列を追加生成しない。
        covariance_frobenius_squared = np.sum(np.abs(covariance_hermitian) ** 2, axis=(-2, -1))
        normalized_steering_power = np.zeros(trace_power.shape, dtype=np.float32)
        normalized_steering_power[valid_norm] = np.asarray(
            steering_quadratic_power[valid_norm] / steering_norm_squared[valid_norm],
            dtype=np.float32,
        )
        residual_squared = np.maximum(
            covariance_frobenius_squared - np.maximum(normalized_steering_power, 0.0) ** 2,
            0.0,
        )
        residual_chunk = np.zeros(trace_power.shape, dtype=np.float32)
        valid_frobenius = covariance_frobenius_squared > np.float32(floor_value)
        residual_chunk[valid_frobenius] = np.asarray(
            np.sqrt(residual_squared[valid_frobenius] / covariance_frobenius_squared[valid_frobenius]),
            dtype=np.float32,
        )

        noise_eigenvalue_mean = np.mean(eigenvalues[..., :-1], axis=-1)
        eigenvalue_ratio_chunk = np.zeros(trace_power.shape, dtype=np.float32)
        # rank-1理想共分散ではnoise平均が0になるため、floorを分母に使い大きな分離比として表す。
        noise_denominator = np.maximum(noise_eigenvalue_mean, np.float32(floor_value))
        eigenvalue_ratio_chunk[valid_power] = eigenvalues[..., -1][valid_power] / noise_denominator[valid_power]

        steering_power_fraction[direction_start:direction_stop] = np.clip(
            steering_fraction_chunk, 0.0, 1.0
        )
        principal_eigenvector_alignment[direction_start:direction_stop] = np.clip(
            alignment_chunk, 0.0, 1.0
        )
        principal_eigenvalue_fraction[direction_start:direction_stop] = np.clip(
            eigenvalue_fraction_chunk, 0.0, 1.0
        )
        principal_to_noise_mean_ratio[direction_start:direction_stop] = eigenvalue_ratio_chunk
        trace_power_output[direction_start:direction_stop] = trace_power
        principal_eigenvalue_gap_fraction[direction_start:direction_stop] = np.clip(
            eigenvalue_gap_chunk, 0.0, 1.0
        )
        steering_rank_one_residual[direction_start:direction_stop] = np.clip(
            residual_chunk, 0.0, 1.0
        )

    return CovarianceSubspaceMetrics(
        steering_power_fraction=steering_power_fraction,
        principal_eigenvector_alignment=principal_eigenvector_alignment,
        principal_eigenvalue_fraction=principal_eigenvalue_fraction,
        principal_to_noise_mean_ratio=principal_to_noise_mean_ratio,
        trace_power=trace_power_output,
        principal_eigenvalue_gap_fraction=principal_eigenvalue_gap_fraction,
        steering_rank_one_residual=steering_rank_one_residual,
    )
