"""spflow.beamforming.cbf を実装するモジュール。"""

from __future__ import annotations

import numpy as np

from ..frequency import OverlapSaveBuffer, ValidRegionExtractor, make_filter_fft

from .mvdr_filter import apply_beamformer, apply_beamformer_bands, apply_beamformer_filter_fft


def _as_steering_matrix(steering: np.ndarray) -> np.ndarray:
    steering_matrix = np.asarray(steering, dtype=np.complex64)
    if steering_matrix.ndim == 1:
        steering_matrix = steering_matrix[:, np.newaxis]
    if steering_matrix.ndim != 2:
        raise ValueError("steering must have shape (n_ch, n_beam).")
    return steering_matrix


def apply_channel_window_to_steering(steering: np.ndarray, channel_window: np.ndarray) -> np.ndarray:
    """steering にチャネル別または帯域別の重みを掛ける。"""

    steering_array = np.asarray(steering, dtype=np.complex64)
    window = np.asarray(channel_window, dtype=np.float32)

    if steering_array.ndim == 2:
        if window.ndim == 2:
            if window.shape[1] != 1:
                raise ValueError('channel_window must have one band for 2D steering input.')
            window = window[:, 0]
        if window.ndim != 1 or window.shape[0] != steering_array.shape[0]:
            raise ValueError('channel_window must have shape (n_ch,) for 2D steering input.')
        return steering_array * window[:, np.newaxis]

    if steering_array.ndim != 3:
        raise ValueError('steering must have shape (n_ch, n_beam) or (n_ch, n_beam, n_band).')
    if window.ndim == 1:
        if window.shape[0] != steering_array.shape[0]:
            raise ValueError('channel_window must agree on n_ch.')
        window = np.repeat(window[:, np.newaxis], steering_array.shape[2], axis=1)
    if window.ndim != 2:
        raise ValueError('channel_window must have shape (n_ch,) or (n_ch, n_band).')
    if window.shape[0] != steering_array.shape[0] or window.shape[1] != steering_array.shape[2]:
        raise ValueError('channel_window and steering must agree on n_ch and n_band.')
    return steering_array * window[:, np.newaxis, :]


def design_cbf_weights(steering: np.ndarray) -> np.ndarray:
    """Design conventional beamformer weights from steering vectors."""

    steering_array = np.asarray(steering, dtype=np.complex64)
    if steering_array.ndim == 3:
        weights = np.zeros_like(steering_array)
        for band_idx in range(steering_array.shape[-1]):
            weights[:, :, band_idx] = design_cbf_weights(steering_array[:, :, band_idx])
        return weights

    steering_matrix = _as_steering_matrix(steering_array)
    norm = np.sum(np.abs(steering_matrix) ** 2, axis=0, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("steering vectors must be non-zero.")
    return steering_matrix / norm


def design_cbf_weights_with_channel_window(steering: np.ndarray, channel_window: np.ndarray) -> np.ndarray:
    """Design CBF weights after applying a per-channel, per-band shading window."""

    return design_cbf_weights(apply_channel_window_to_steering(steering, channel_window))


def design_cbf_overlap_save_filters(steering: np.ndarray, frame_size: int) -> np.ndarray:
    """Convert CBF steering weights into overlap-save filter FFTs.

    Conjugation is baked into the time-domain filter so runtime projection uses a plain transpose.
    """

    weights = design_cbf_weights(steering)
    taps = np.conjugate(weights)[..., np.newaxis]
    return make_filter_fft(taps, frame_size=frame_size, axis=-1)


class CBFBeamformer:
    """Apply fixed conventional beamformer weights."""

    def __init__(self, steering: np.ndarray, channel_window: np.ndarray | None = None) -> None:
        self.weights = (
            design_cbf_weights(steering)
            if channel_window is None
            else design_cbf_weights_with_channel_window(steering, channel_window)
        )

    def process(self, X: np.ndarray) -> np.ndarray:
        snapshots = np.asarray(X)
        if snapshots.ndim == 2 and self.weights.ndim == 3:
            return apply_beamformer_bands(snapshots, self.weights)
        return apply_beamformer(snapshots, self.weights)


class CBFOverlapSaveBeamformer:
    """Apply bandwise CBF steering as overlap-save filters on subband time signals."""

    def __init__(
        self,
        steering: np.ndarray,
        frame_size: int = 2048,
        valid_size: int = 1024,
        channel_window: np.ndarray | None = None,
    ) -> None:
        if valid_size <= 0:
            raise ValueError("valid_size must be positive.")
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if valid_size > frame_size:
            raise ValueError("valid_size must not exceed frame_size.")

        steering_array = np.asarray(steering, dtype=np.complex64)
        if steering_array.ndim == 2:
            steering_array = steering_array[:, :, np.newaxis]
        if steering_array.ndim != 3:
            raise ValueError("steering must have shape (n_ch, n_beam, n_band).")
        if channel_window is not None:
            steering_array = apply_channel_window_to_steering(steering_array, channel_window)

        self.frame_size = frame_size
        self.valid_size = valid_size
        self.filter_ffts = design_cbf_overlap_save_filters(steering_array, frame_size=frame_size)
        self.n_band = steering_array.shape[2]
        self.buffers = [
            OverlapSaveBuffer(frame_size=frame_size, valid_size=valid_size, axis=-1)
            for _ in range(self.n_band)
        ]
        self.valid_extractors = [
            ValidRegionExtractor(frame_size=frame_size, valid_size=valid_size, axis=-1)
            for _ in range(self.n_band)
        ]

    def process(self, X: np.ndarray) -> list[tuple[int, np.ndarray]]:
        subbands = np.asarray(X, dtype=np.complex64)
        if subbands.ndim != 3:
            raise ValueError("X must have shape (n_ch, n_band, n_sample).")
        if subbands.shape[1] != self.n_band:
            raise ValueError("X and steering must agree on n_band.")

        outputs: list[tuple[int, np.ndarray]] = []
        for band_idx in range(self.n_band):
            frames = self.buffers[band_idx].process(subbands[:, band_idx, :])
            for frame in frames:
                frame_fft = np.fft.fft(frame, n=self.frame_size, axis=-1)
                filtered_frame = apply_beamformer_filter_fft(
                    frame_fft,
                    self.filter_ffts[:, :, band_idx, :],
                )
                time_frame = np.fft.ifft(filtered_frame, n=self.frame_size, axis=-1)
                valid = self.valid_extractors[band_idx].process(time_frame)
                outputs.append((band_idx, valid))
        return outputs

    def flush(self) -> list[tuple[int, np.ndarray]]:
        outputs: list[tuple[int, np.ndarray]] = []
        for band_idx in range(self.n_band):
            frames = self.buffers[band_idx].flush(pad=True, fill_value=0.0)
            for frame in frames:
                frame_fft = np.fft.fft(frame, n=self.frame_size, axis=-1)
                filtered_frame = apply_beamformer_filter_fft(
                    frame_fft,
                    self.filter_ffts[:, :, band_idx, :],
                )
                time_frame = np.fft.ifft(filtered_frame, n=self.frame_size, axis=-1)
                valid = self.valid_extractors[band_idx].process(time_frame)
                outputs.append((band_idx, valid))
        return outputs
