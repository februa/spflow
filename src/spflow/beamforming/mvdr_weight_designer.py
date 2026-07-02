"""spflow.beamforming.mvdr_weight_designer を実装するモジュール。"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..callback import DoubleBufferCallback


def _as_steering_matrix(steering: np.ndarray) -> np.ndarray:
    steering_matrix = np.asarray(steering, dtype=np.complex64)
    if steering_matrix.ndim == 1:
        steering_matrix = steering_matrix[:, np.newaxis]
    if steering_matrix.ndim != 2:
        raise ValueError("steering must have shape (n_ch, n_beam).")
    return steering_matrix


def _validate_bandwise_inputs(covariance: np.ndarray, steering_array: np.ndarray) -> tuple[int, int, int]:
    if covariance.ndim != 3:
        raise ValueError("Rxx must have shape (n_band, n_ch, n_ch).")
    if steering_array.ndim != 3:
        raise ValueError("steering must have shape (n_ch, n_beam, n_band) for bandwise design.")
    if covariance.shape[1] != covariance.shape[2]:
        raise ValueError("Rxx must contain square covariance matrices.")
    if covariance.shape[0] != steering_array.shape[2]:
        raise ValueError("Rxx and steering must agree on n_band.")
    if covariance.shape[1] != steering_array.shape[0]:
        raise ValueError("Rxx and steering must agree on n_ch.")
    return covariance.shape[0], covariance.shape[1], steering_array.shape[1]


def design_mvdr_weights(Rxx: np.ndarray, steering: np.ndarray, diag_load: float = 1e-3) -> np.ndarray:
    """Design MVDR weights for a single band."""

    covariance = np.asarray(Rxx, dtype=np.complex64)
    steering_matrix = _as_steering_matrix(steering)

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Rxx must have shape (n_ch, n_ch).")
    if covariance.shape[0] != steering_matrix.shape[0]:
        raise ValueError("Rxx and steering must agree on n_ch.")
    if diag_load < 0.0:
        raise ValueError("diag_load must be non-negative.")

    loaded = covariance.copy()
    if diag_load > 0.0:
        base = np.real(np.trace(loaded)) / loaded.shape[0]
        load = diag_load * (base if base > 0.0 else 1.0)
        loaded = loaded + load * np.eye(loaded.shape[0], dtype=np.complex64)

    response = np.linalg.solve(loaded, steering_matrix)
    denom = np.sum(steering_matrix.conj() * response, axis=0)
    return response / denom[np.newaxis, :]


def design_mvdr_weights_bands(Rxx: np.ndarray, steering: np.ndarray, diag_load: float = 1e-3) -> np.ndarray:
    """Design MVDR weights for many bands with stacked linear solves."""

    covariance = np.asarray(Rxx, dtype=np.complex64)
    steering_array = np.asarray(steering, dtype=np.complex64)
    if diag_load < 0.0:
        raise ValueError("diag_load must be non-negative.")

    n_band, n_ch, _ = _validate_bandwise_inputs(covariance, steering_array)
    loaded = covariance.copy()
    if diag_load > 0.0:
        base = np.real(np.trace(loaded, axis1=1, axis2=2)) / n_ch
        load = diag_load * np.where(base > 0.0, base, 1.0)
        loaded = loaded + load[:, np.newaxis, np.newaxis] * np.eye(n_ch, dtype=np.complex64)[np.newaxis, :, :]

    steering_batch = np.moveaxis(steering_array, -1, 0)
    response = np.linalg.solve(loaded, steering_batch)
    denom = np.sum(steering_batch.conj() * response, axis=1)
    weights = response / denom[:, np.newaxis, :]
    return np.moveaxis(weights, 0, -1)


def design_mvdr_weights_with_channel_window(
    Rxx: np.ndarray,
    steering: np.ndarray,
    channel_window: np.ndarray,
    diag_load: float = 1e-3,
) -> np.ndarray:
    """Design MVDR weights using only channels whose shading coefficient is non-zero.

    For practical MVDR use, the shading table is treated as a rectangular selector:
    `used = (channel_window != 0)`.
    """

    covariance = np.asarray(Rxx, dtype=np.complex64)
    steering_array = np.asarray(steering, dtype=np.complex64)
    window = np.asarray(channel_window, dtype=np.float32)

    if covariance.ndim == 2:
        steering_matrix = _as_steering_matrix(steering_array)
        if window.ndim == 2:
            if window.shape[1] != 1:
                raise ValueError('channel_window must have one band for 2D covariance input.')
            window = window[:, 0]
        if window.ndim != 1 or window.shape[0] != covariance.shape[0]:
            raise ValueError('channel_window must have shape (n_ch,) for 2D covariance input.')
        used = window != 0.0
        if not np.any(used):
            raise ValueError('channel_window must select at least one channel.')
        reduced = design_mvdr_weights(covariance[np.ix_(used, used)], steering_matrix[used], diag_load=diag_load)
        full = np.zeros_like(steering_matrix)
        full[used] = reduced
        return full

    if covariance.ndim != 3:
        raise ValueError('Rxx must have shape (n_ch, n_ch) or (n_band, n_ch, n_ch).')
    if steering_array.ndim != 3:
        raise ValueError('steering must have shape (n_ch, n_beam, n_band) for bandwise design.')
    if window.ndim == 1:
        if window.shape[0] != covariance.shape[1]:
            raise ValueError('channel_window must agree on n_ch.')
        window = np.repeat(window[:, np.newaxis], covariance.shape[0], axis=1)
    if window.ndim != 2:
        raise ValueError('channel_window must have shape (n_ch,) or (n_ch, n_band).')
    if covariance.shape[0] != steering_array.shape[2]:
        raise ValueError('Rxx and steering must agree on n_band.')
    if covariance.shape[1] != covariance.shape[2]:
        raise ValueError('Rxx must contain square covariance matrices.')
    if covariance.shape[1] != steering_array.shape[0]:
        raise ValueError('Rxx and steering must agree on n_ch.')
    if window.shape != (steering_array.shape[0], steering_array.shape[2]):
        raise ValueError('channel_window must have shape (n_ch, n_band).')

    n_band, n_ch = covariance.shape[0], covariance.shape[1]
    n_beam = steering_array.shape[1]
    weights = np.zeros((n_ch, n_beam, n_band), dtype=np.complex64)
    for band_idx in range(n_band):
        used = window[:, band_idx] != 0.0
        if not np.any(used):
            raise ValueError('Each band must select at least one channel.')
        reduced = design_mvdr_weights(
            covariance[band_idx][np.ix_(used, used)],
            steering_array[used, :, band_idx],
            diag_load=diag_load,
        )
        weights[used, :, band_idx] = reduced
    return weights


class MVDRWeightDesigner:
    """Design MVDR weights for one or many subbands."""

    def __init__(self, diag_load: float = 1e-3) -> None:
        if diag_load < 0.0:
            raise ValueError("diag_load must be non-negative.")
        self.diag_load = diag_load

    def process(self, Rxx: np.ndarray, steering: np.ndarray) -> np.ndarray:
        covariance = np.asarray(Rxx, dtype=np.complex64)
        steering_array = np.asarray(steering, dtype=np.complex64)

        if covariance.ndim == 2:
            return design_mvdr_weights(covariance, steering_array, diag_load=self.diag_load)

        return design_mvdr_weights_bands(covariance, steering_array, diag_load=self.diag_load)


class MVDRWeightCallback(DoubleBufferCallback):
    """StepScheduler callback for bandwise MVDR weight updates."""

    def __init__(self, diag_load: float = 1e-3) -> None:
        super().__init__()
        self.designer = MVDRWeightDesigner(diag_load=diag_load)

    def signature(self, inputs: Any) -> Any:
        if "signature" in inputs:
            return inputs["signature"]
        return (id(inputs["Rxx"]), id(inputs["steering"]))

    def make_initial_output(self, inputs: Any) -> np.ndarray:
        steering = np.asarray(inputs["steering"], dtype=np.complex64)
        if steering.ndim != 3:
            raise ValueError("steering must have shape (n_ch, n_beam, n_band).")
        return np.zeros_like(steering)

    def make_work_buffer(self, inputs: Any) -> np.ndarray:
        return np.zeros_like(self.prev)

    def make_items(self, inputs: Any):
        covariance = np.asarray(inputs["Rxx"], dtype=np.complex64)
        if covariance.ndim != 3:
            raise ValueError("Rxx must have shape (n_band, n_ch, n_ch).")
        return range(covariance.shape[0])

    def update_item(self, item: Any, inputs: Any) -> None:
        band_idx = int(item)
        covariance = np.asarray(inputs["Rxx"], dtype=np.complex64)
        steering = np.asarray(inputs["steering"], dtype=np.complex64)
        self.work[:, :, band_idx] = self.designer.process(
            covariance[band_idx],
            steering[:, :, band_idx],
        )
