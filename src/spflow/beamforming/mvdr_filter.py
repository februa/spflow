"""spflow.beamforming.mvdr_filter を実装するモジュール。"""

from __future__ import annotations

import numpy as np

from ..frequency import OverlapSaveBuffer, ValidRegionExtractor, make_filter_fft


def beam_response_rms_db(response: np.ndarray | complex) -> float:
    """Convert complex beam response into an RMS-level dB metric."""

    response_scalar = np.asarray(response, dtype=np.complex64)
    if response_scalar.size != 1:
        raise ValueError("response must be scalar-like.")
    return float(20.0 * np.log10(np.abs(response_scalar.reshape(-1)[0])))


def apply_beamformer(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply beamformer weights to subband snapshots."""

    snapshots = np.asarray(X, dtype=np.complex64)
    beam_weights = np.asarray(weights, dtype=np.complex64)

    if snapshots.ndim != 2:
        raise ValueError("X must have shape (n_ch, n_frame).")
    if beam_weights.ndim == 1:
        beam_weights = beam_weights[:, np.newaxis]
    if beam_weights.ndim != 2:
        raise ValueError("weights must have shape (n_ch, n_beam).")
    if snapshots.shape[0] != beam_weights.shape[0]:
        raise ValueError("X and weights must agree on n_ch.")

    return np.einsum("cf,cb->bf", snapshots, beam_weights.conj(), optimize=True)


def apply_beamformer_bands(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply beamformer weights to all frequency bands."""

    snapshots = np.asarray(X, dtype=np.complex64)
    beam_weights = np.asarray(weights, dtype=np.complex64)

    if beam_weights.ndim != 3:
        raise ValueError("weights must have shape (n_ch, n_beam, n_band).")

    if snapshots.ndim == 2:
        if snapshots.shape[0] != beam_weights.shape[0] or snapshots.shape[1] != beam_weights.shape[2]:
            raise ValueError("X and weights must agree on n_ch and n_band.")
        return np.einsum("cb,cdb->db", snapshots, beam_weights.conj(), optimize=True)

    if snapshots.ndim == 3:
        if snapshots.shape[0] != beam_weights.shape[0] or snapshots.shape[1] != beam_weights.shape[2]:
            raise ValueError("X and weights must agree on n_ch and n_band.")
        return np.einsum("cbf,cdb->dbf", snapshots, beam_weights.conj(), optimize=True)

    raise ValueError("X must have shape (n_ch, n_band) or (n_ch, n_band, n_frame).")


def apply_beamformer_filter_fft(X_fft: np.ndarray, filter_fft: np.ndarray) -> np.ndarray:
    """Apply overlap-save filter FFTs to multichannel frames.

    `filter_fft` is expected to already include the beamformer conjugation.
    Runtime projection therefore uses a plain transpose instead of a conjugate transpose.
    """

    spectra = np.asarray(X_fft, dtype=np.complex64)
    filters = np.asarray(filter_fft, dtype=np.complex64)

    if spectra.ndim != 2:
        raise ValueError("X_fft must have shape (n_ch, n_freq).")
    if filters.ndim != 3:
        raise ValueError("filter_fft must have shape (n_ch, n_beam, n_freq).")
    if spectra.shape[0] != filters.shape[0] or spectra.shape[1] != filters.shape[2]:
        raise ValueError("X_fft and filter_fft must agree on n_ch and n_freq.")

    return np.einsum("cf,cbf->bf", spectra, filters, optimize=True)


def design_mvdr_overlap_save_filters(weights: np.ndarray, frame_size: int) -> np.ndarray:
    """Convert MVDR weights into overlap-save filter FFTs.

    Conjugation is baked into the time-domain filter so runtime projection uses a plain transpose.
    """

    beam_weights = np.asarray(weights, dtype=np.complex64)
    if beam_weights.ndim == 2:
        beam_weights = beam_weights[:, :, np.newaxis]
    if beam_weights.ndim != 3:
        raise ValueError("weights must have shape (n_ch, n_beam) or (n_ch, n_beam, n_band).")

    taps = np.conjugate(beam_weights)[..., np.newaxis]
    return make_filter_fft(taps, frame_size=frame_size, axis=-1)


class MVDRFilter:
    """Apply stored or supplied MVDR weights to subband snapshots."""

    def __init__(self, weights: np.ndarray | None = None) -> None:
        self.weights = None if weights is None else np.asarray(weights, dtype=np.complex64)

    def update_weights(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.complex64)

    def process(self, X: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        active_weights = self.weights if weights is None else np.asarray(weights, dtype=np.complex64)
        if active_weights is None:
            raise ValueError("weights are not set.")
        if np.asarray(active_weights).ndim == 3:
            return apply_beamformer_bands(X, active_weights)
        return apply_beamformer(X, active_weights)


class MVDROverlapSaveBeamformer:
    """Apply bandwise MVDR weights as overlap-save filters on subband time signals."""

    def __init__(
        self,
        weights: np.ndarray,
        frame_size: int = 2048,
        valid_size: int = 1024,
    ) -> None:
        if valid_size <= 0:
            raise ValueError("valid_size must be positive.")
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if valid_size > frame_size:
            raise ValueError("valid_size must not exceed frame_size.")

        beam_weights = np.asarray(weights, dtype=np.complex64)
        if beam_weights.ndim == 2:
            beam_weights = beam_weights[:, :, np.newaxis]
        if beam_weights.ndim != 3:
            raise ValueError("weights must have shape (n_ch, n_beam, n_band).")

        self.frame_size = frame_size
        self.valid_size = valid_size
        self.n_band = beam_weights.shape[2]
        self.n_beam = beam_weights.shape[1]
        self.weights = beam_weights.copy()
        self.filter_ffts = design_mvdr_overlap_save_filters(self.weights, frame_size=frame_size)
        self.buffers = [
            OverlapSaveBuffer(frame_size=frame_size, valid_size=valid_size, axis=-1)
            for _ in range(self.n_band)
        ]
        self.valid_extractors = [
            ValidRegionExtractor(frame_size=frame_size, valid_size=valid_size, axis=-1)
            for _ in range(self.n_band)
        ]

    def update_weights(self, weights: np.ndarray) -> None:
        beam_weights = np.asarray(weights, dtype=np.complex64)
        if beam_weights.ndim == 2:
            beam_weights = beam_weights[:, :, np.newaxis]
        if beam_weights.shape != self.weights.shape:
            raise ValueError("updated weights must match the original shape.")
        self.weights = beam_weights.copy()
        self.filter_ffts = design_mvdr_overlap_save_filters(self.weights, frame_size=self.frame_size)

    def process(self, X: np.ndarray) -> list[tuple[int, np.ndarray]]:
        subbands = np.asarray(X, dtype=np.complex64)
        if subbands.ndim != 3:
            raise ValueError("X must have shape (n_ch, n_band, n_sample).")
        if subbands.shape[1] != self.n_band:
            raise ValueError("X and weights must agree on n_band.")

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
