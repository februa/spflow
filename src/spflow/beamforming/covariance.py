"""spflow.beamforming.covariance を実装するモジュール。"""

from __future__ import annotations

import math

import numpy as np


def estimate_covariance(X: np.ndarray) -> np.ndarray:
    """サブバンド観測から空間共分散行列を推定する。

    Args:
        X: 観測スナップショット。shape は `[n_ch, n_frame]`。
            axis=0 はセンサチャネル、axis=1 は時間フレームまたは独立スナップショット。

    Returns:
        空間共分散行列 `Rxx = XX^H / n_frame`。shape は `[n_ch, n_ch]`。

    Raises:
        ValueError: 入力 shape が `[n_ch, n_frame]` でない場合。
        ValueError: フレーム数が 0 以下の場合。
    """
    snapshots = np.asarray(X, dtype=np.complex64)
    if snapshots.ndim != 2:
        raise ValueError("X must have shape (n_ch, n_frame).")
    if snapshots.shape[1] <= 0:
        raise ValueError("X must contain at least one frame.")

    # Rxx[ch_i, ch_j] = (1 / n_frame) Σ_t x[ch_i, t] conj(x[ch_j, t])。
    # 行列積 X X^H を使うことで、全フレーム平均の空間共分散を一度に計算する。
    return snapshots @ snapshots.conj().T / snapshots.shape[1]


def estimate_covariance_snapshots(snapshots: np.ndarray, *, normalization: float = 1.0) -> np.ndarray:
    """各スナップショットベクトルから 1 枚ずつ共分散行列を作る。

    Args:
        snapshots: スナップショット列。shape は `[n_snapshot, n_ch]`。
            axis=0 は更新時刻、axis=1 はチャネル。
        normalization: 正規化係数。単位系を揃えるため、各スナップショットを
            `snapshot / normalization` してから外積を作る。

    Returns:
        スナップショットごとの共分散。shape は `[n_snapshot, n_ch, n_ch]`。

    Raises:
        ValueError: 入力 shape が想定と異なる場合。
        ValueError: `normalization <= 0` の場合。
    """
    vectors = np.asarray(snapshots, dtype=np.complex64)
    if vectors.ndim != 2:
        raise ValueError("snapshots must have shape (n_snapshot, n_ch).")
    if vectors.shape[1] <= 0:
        raise ValueError("snapshots must contain at least one channel.")
    if normalization <= 0.0:
        raise ValueError("normalization must be positive.")

    # scaled shape: [n_snapshot, n_ch]
    # einsum("bi,bj->bij") は各時刻 b について outer(x_b, x_b^H) を作る。
    scaled = vectors / normalization
    return np.einsum("bi,bj->bij", scaled, scaled.conj(), optimize=True)


def integration_blocks_from_integration_time(integration_time: float, rate: float) -> int:
    """積分時間と更新レートから必要ブロック数を求める。"""
    if integration_time < 0.0:
        raise ValueError("integration_time must be non-negative.")
    if rate <= 0.0:
        raise ValueError("rate must be positive.")
    return int(math.ceil(integration_time * rate))


def recommended_integration_time_for_independent_samples(
    n_ch: int,
    rate: float,
    independent_samples_per_channel: float = 2.0,
) -> float:
    """独立標本数の目安から積分時間を推奨する。"""
    if n_ch <= 0:
        raise ValueError("n_ch must be positive.")
    if rate <= 0.0:
        raise ValueError("rate must be positive.")
    if independent_samples_per_channel <= 0.0:
        raise ValueError("independent_samples_per_channel must be positive.")
    return float((n_ch * independent_samples_per_channel) / rate)


def forgetting_factor_from_integration_time(integration_time: float, rate: float) -> float:
    """積分時間と更新レートから忘却係数 `alpha` を求める。

    戻り値は逐次更新式

        R[n] = (1 - alpha) R[n-1] + alpha R_current[n]

    における `alpha` であり、`integration_time * rate` が大きいほど
    平均化窓を長く取り、`alpha` は小さくなる。
    """
    if integration_time < 0.0:
        raise ValueError("integration_time must be non-negative.")
    if rate <= 0.0:
        raise ValueError("rate must be positive.")
    return float(min(2.0 / (1.0 + integration_time * rate), 1.0))


class CovarianceEstimator:
    """指数忘却付きの逐次共分散推定器。

    このクラスは各更新で得られる共分散行列を受け取り、必要なら忘却係数により
    時間平滑化した推定値を返す。ビーム重み設計そのものは責務に含めない。
    """

    def __init__(
        self,
        *,
        forgetting_factor: float | None = None,
        smoothing: float | None = None,
    ) -> None:
        if forgetting_factor is not None and not (0.0 < forgetting_factor <= 1.0):
            raise ValueError("forgetting_factor must be in (0.0, 1.0].")
        if smoothing is not None and not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0.0, 1.0).")
        if forgetting_factor is not None and smoothing is not None:
            raise ValueError("Specify either forgetting_factor or smoothing, not both.")

        self.forgetting_factor = forgetting_factor
        self.smoothing = smoothing
        self._prev: np.ndarray | None = None

    @classmethod
    def from_integration_time(cls, integration_time: float, rate: float) -> "CovarianceEstimator":
        """積分時間から忘却係数を解釈して推定器を生成する。"""
        return cls(forgetting_factor=forgetting_factor_from_integration_time(integration_time, rate))

    def process(self, X: np.ndarray) -> np.ndarray:
        """観測行列を 1 回分処理し、更新後の共分散を返す。

        Args:
            X: 観測スナップショット。shape は `[n_ch, n_frame]`。

        Returns:
            現在の共分散推定値。shape は `[n_ch, n_ch]`。
        """
        current = estimate_covariance(X)
        return self._integrate(current)

    def process_snapshot(self, snapshot: np.ndarray, *, normalization: float = 1.0) -> np.ndarray:
        """単一スナップショットベクトルを処理する。

        Args:
            snapshot: 単一スナップショット。shape は `[n_ch]` または `[n_ch, 1]`。
            normalization: 正規化係数。

        Returns:
            更新後の共分散推定値。shape は `[n_ch, n_ch]`。
        """
        if normalization <= 0.0:
            raise ValueError("normalization must be positive.")

        vector = np.asarray(snapshot, dtype=np.complex64)
        if vector.ndim == 1:
            vector = vector[:, np.newaxis]
        elif vector.ndim != 2 or vector.shape[1] != 1:
            raise ValueError("snapshot must have shape (n_ch,) or (n_ch, 1).")

        # 列ベクトル x 1 本に対して x x^H を作り、逐次共分散更新へ渡す。
        current = estimate_covariance(vector / normalization)
        return self._integrate(current)

    def process_snapshots(self, snapshots: np.ndarray, *, normalization: float = 1.0) -> np.ndarray:
        """行ごとのスナップショット列を逐次積分する。

        Args:
            snapshots: スナップショット列。shape は `[n_snapshot, n_ch]`。
            normalization: 正規化係数。

        Returns:
            各更新時点の共分散。shape は `[n_snapshot, n_ch, n_ch]`。
        """
        current = estimate_covariance_snapshots(snapshots, normalization=normalization)
        return self._integrate(current)

    def reset(self) -> None:
        """内部状態を破棄し、初期推定状態へ戻す。"""
        self._prev = None

    def _integrate(self, current: np.ndarray) -> np.ndarray:
        alpha = self._resolve_forgetting_factor()
        if alpha is None:
            return current

        if self._prev is None:
            # 初回更新では過去推定が存在しないため、現在値をそのまま採用する。
            self._prev = current.copy()
        else:
            # 忘却係数 alpha により、新規観測と過去推定の凸結合を取る。
            self._prev = (1.0 - alpha) * self._prev + alpha * current
        return self._prev.copy()

    def _resolve_forgetting_factor(self) -> float | None:
        if self.forgetting_factor is not None:
            return self.forgetting_factor
        if self.smoothing is not None:
            return 1.0 - self.smoothing
        return None


def integrate_band_covariances(
    X: np.ndarray,
    *,
    forgetting_factor: float,
    normalization: float = 1.0,
    n_blocks: int | None = None,
) -> np.ndarray:
    """帯域ごとの逐次スナップショットから空間共分散を積分する。

    Args:
        X: 入力サブバンド。shape は `[n_ch, n_band, n_block]`。
            axis=0 はチャネル、axis=1 は帯域、axis=2 は更新ブロック。
        forgetting_factor: 忘却係数 `alpha`。
        normalization: スナップショット正規化係数。
        n_blocks: 使用するブロック数。省略時は全ブロック。

    Returns:
        最終共分散。shape は `[n_band, n_ch, n_ch]`。
    """
    subbands = np.asarray(X, dtype=np.complex64)
    if subbands.ndim != 3:
        raise ValueError("X must have shape (n_ch, n_band, n_block).")
    if normalization <= 0.0:
        raise ValueError("normalization must be positive.")

    n_ch, n_band, n_frame = subbands.shape
    block_count = n_frame if n_blocks is None else n_blocks
    if block_count <= 0:
        raise ValueError("n_blocks must be positive.")
    if block_count > n_frame:
        raise ValueError("n_blocks exceeds the available number of blocks.")

    estimator = CovarianceEstimator(forgetting_factor=forgetting_factor)
    rxx = np.zeros((n_band, n_ch, n_ch), dtype=np.complex64)
    for block_idx in range(block_count):
        # subbands[:, :, block_idx] shape: [n_ch, n_band]
        # 転置後 [n_band, n_ch] とし、各帯域を 1 スナップショットとして扱う。
        rxx = estimator.process_snapshots(
            subbands[:, :, block_idx].T,
            normalization=normalization,
        )
    return rxx
