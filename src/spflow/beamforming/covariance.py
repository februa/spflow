"""spflow.beamforming.covariance を実装するモジュール。"""

from __future__ import annotations

import math

import numpy as np


def estimate_covariance(X: np.ndarray) -> np.ndarray:
    """Estimate spatial covariance from subband snapshots."""

    snapshots = np.asarray(X, dtype=np.complex64)
    if snapshots.ndim != 2:
        raise ValueError("X must have shape (n_ch, n_frame).")
    if snapshots.shape[1] <= 0:
        raise ValueError("X must contain at least one frame.")

    return snapshots @ snapshots.conj().T / snapshots.shape[1]


def estimate_covariance_snapshots(snapshots: np.ndarray, *, normalization: float = 1.0) -> np.ndarray:
    """Estimate one covariance matrix per snapshot vector.

    snapshots must have shape (n_snapshot, n_ch). Each output covariance is
    ``outer(snapshot / normalization, conj(snapshot / normalization))``.
    """

    vectors = np.asarray(snapshots, dtype=np.complex64)
    if vectors.ndim != 2:
        raise ValueError("snapshots must have shape (n_snapshot, n_ch).")
    if vectors.shape[1] <= 0:
        raise ValueError("snapshots must contain at least one channel.")
    if normalization <= 0.0:
        raise ValueError("normalization must be positive.")

    scaled = vectors / normalization
    return np.einsum("bi,bj->bij", scaled, scaled.conj(), optimize=True)


def integration_blocks_from_integration_time(integration_time: float, rate: float) -> int:
    """Return the number of covariance-update blocks implied by integration_time * rate."""

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
    """Heuristic integration time so that integration_time * rate == n_ch * independent_samples_per_channel."""

    if n_ch <= 0:
        raise ValueError("n_ch must be positive.")
    if rate <= 0.0:
        raise ValueError("rate must be positive.")
    if independent_samples_per_channel <= 0.0:
        raise ValueError("independent_samples_per_channel must be positive.")
    return float((n_ch * independent_samples_per_channel) / rate)


def forgetting_factor_from_integration_time(integration_time: float, rate: float) -> float:
    """Convert integration time and actual covariance-update rate into a forgetting factor.

    The returned factor is the weight applied to the current covariance snapshot:

        R[n] = (1 - alpha) * R[n-1] + alpha * R_current[n]

    where alpha = min(2 / (1 + integration_time * rate), 1).
    """

    if integration_time < 0.0:
        raise ValueError("integration_time must be non-negative.")
    if rate <= 0.0:
        raise ValueError("rate must be positive.")
    return float(min(2.0 / (1.0 + integration_time * rate), 1.0))


class CovarianceEstimator:
    """Stateful covariance estimator with optional exponential forgetting."""

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
        return cls(forgetting_factor=forgetting_factor_from_integration_time(integration_time, rate))

    def process(self, X: np.ndarray) -> np.ndarray:
        current = estimate_covariance(X)
        return self._integrate(current)

    def process_snapshot(self, snapshot: np.ndarray, *, normalization: float = 1.0) -> np.ndarray:
        if normalization <= 0.0:
            raise ValueError("normalization must be positive.")

        vector = np.asarray(snapshot, dtype=np.complex64)
        if vector.ndim == 1:
            vector = vector[:, np.newaxis]
        elif vector.ndim != 2 or vector.shape[1] != 1:
            raise ValueError("snapshot must have shape (n_ch,) or (n_ch, 1).")

        current = estimate_covariance(vector / normalization)
        return self._integrate(current)

    def process_snapshots(self, snapshots: np.ndarray, *, normalization: float = 1.0) -> np.ndarray:
        """Process one snapshot vector per row and return one covariance per row."""

        current = estimate_covariance_snapshots(snapshots, normalization=normalization)
        return self._integrate(current)

    def reset(self) -> None:
        self._prev = None

    def _integrate(self, current: np.ndarray) -> np.ndarray:
        alpha = self._resolve_forgetting_factor()
        if alpha is None:
            return current

        if self._prev is None:
            self._prev = current.copy()
        else:
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
    """Integrate one spatial covariance matrix per band using per-block, per-bin snapshots.

    X must have shape (n_ch, n_band, n_block). Each update uses:

        snapshot = X[:, band, block] / normalization
        cov = snapshot @ snapshot^H
        Rxx[band] <- (1 - alpha) * Rxx[band] + alpha * cov
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
        rxx = estimator.process_snapshots(
            subbands[:, :, block_idx].T,
            normalization=normalization,
        )
    return rxx
