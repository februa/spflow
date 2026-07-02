"""spflow.filterbank.prdft_polyphase を実装するモジュール。"""

from __future__ import annotations

import numpy as np

from .prototype_bank import PolyphaseDecomposition, PrototypeFilter


class _PolyphasePRBase:
    def __init__(self, *, prototype: PrototypeFilter, band_order: str = 'fft', axis: int = -1) -> None:
        if band_order != 'fft':
            raise ValueError('Only FFT band order is currently supported.')
        self.prototype = prototype
        self.n_band = prototype.n_band
        self.decimation = prototype.decimation
        self.prototype_length = prototype.prototype_length
        self.band_order = band_order
        self.axis = axis
        self._polyphase = PolyphaseDecomposition(self.decimation).decompose(prototype)

    @property
    def n_phase(self) -> int:
        return self._polyphase.shape[0]

    @property
    def transient_length(self) -> int:
        return (self.n_phase - 1) * self.decimation

    def _normalize_axis(self, axis: int, ndim: int) -> int:
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError('axis is out of bounds for input.')
        return axis

    def _frame_blocks(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        n_samples = x.shape[-1]
        if n_samples == 0:
            empty = np.zeros(x.shape[:-1] + (self.n_phase - 1, self.decimation), dtype=np.complex64)
            return empty, 0

        n_frames = int(np.ceil(n_samples / self.decimation))
        front_pad = self.transient_length
        back_pad = n_frames * self.decimation - n_samples
        padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(front_pad, back_pad)])
        blocks = padded.reshape(padded.shape[:-1] + (-1, self.decimation))
        return blocks, n_frames


class PolyphasePRDFTAnalysisBank(_PolyphasePRBase):
    """Critically sampled polyphase DFT analysis bank with explicit edge padding."""

    def analysis(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.complex64)
        signal_axis = self._normalize_axis(self.axis, arr.ndim)
        moved = np.moveaxis(arr, signal_axis, -1)
        blocks, n_frames = self._frame_blocks(moved)
        if n_frames == 0:
            empty = np.zeros(moved.shape[:-1] + (self.n_band, 0), dtype=np.complex64)
            return np.moveaxis(empty, -2, signal_axis)

        phase_samples = np.zeros(moved.shape[:-1] + (self.decimation, n_frames), dtype=np.complex64)
        polyphase_reversed = self._polyphase[::-1]
        for frame_idx in range(n_frames):
            window = blocks[..., frame_idx : frame_idx + self.n_phase, :]
            phase_samples[..., :, frame_idx] = np.sum(window * polyphase_reversed, axis=-2)

        subbands = np.fft.fft(phase_samples, axis=-2)
        return np.moveaxis(subbands, -2, signal_axis)


class PolyphasePRDFTSynthesisBank(_PolyphasePRBase):
    """Critically sampled polyphase DFT synthesis bank with overlap-add in block domain."""

    def __init__(
        self,
        *,
        prototype: PrototypeFilter,
        delay_compensation: int = 0,
        band_order: str = 'fft',
        axis: int = -1,
    ) -> None:
        super().__init__(prototype=prototype, band_order=band_order, axis=axis)
        self.delay_compensation = delay_compensation

    def synthesis(self, subbands: np.ndarray, length: int | None = None) -> np.ndarray:
        arr = np.asarray(subbands, dtype=np.complex64)
        band_axis = self._normalize_axis(self.axis, arr.ndim - 1)
        if arr.shape[band_axis] != self.n_band:
            raise ValueError('subbands shape does not match the configured number of bands.')

        moved = np.moveaxis(arr, band_axis, -2)
        if moved.shape[-1] == 0:
            empty = np.zeros(moved.shape[:-2] + (0,), dtype=np.complex64)
            return np.moveaxis(empty, -1, band_axis)

        phase_samples = np.fft.ifft(moved, axis=-2)
        out_blocks = np.zeros(
            moved.shape[:-2] + (moved.shape[-1] + self.n_phase - 1, self.decimation),
            dtype=np.complex64,
        )
        for phase_idx in range(self.decimation):
            out_blocks[..., :, phase_idx] = self._convolve_last_axis(
                phase_samples[..., phase_idx, :],
                self._polyphase[:, phase_idx],
            )

        reconstructed = out_blocks.reshape(out_blocks.shape[:-2] + (-1,))
        compensated = self._apply_delay_compensation(reconstructed)
        if length is not None:
            compensated = compensated[..., :length]
        return np.moveaxis(compensated, -1, band_axis)

    def _convolve_last_axis(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        out = np.zeros(x.shape[:-1] + (x.shape[-1] + coeffs.size - 1,), dtype=np.complex64)
        for tap_idx, coeff in enumerate(coeffs):
            out[..., tap_idx : tap_idx + x.shape[-1]] += coeff * x
        return out

    def _apply_delay_compensation(self, x: np.ndarray) -> np.ndarray:
        delay = self.delay_compensation
        if delay == 0:
            return x
        if delay > 0:
            if delay >= x.shape[-1]:
                return np.zeros(x.shape[:-1] + (0,), dtype=x.dtype)
            return x[..., delay:]
        pad = np.zeros(x.shape[:-1] + (-delay,), dtype=x.dtype)
        return np.concatenate([pad, x], axis=-1)


class PolyphasePRPairDesigner:
    """Design synthesis prototypes from branch-wise polyphase PR constraints."""

    def __init__(self, n_band: int, decimation: int) -> None:
        if n_band <= 0:
            raise ValueError('n_band must be positive.')
        if decimation <= 0:
            raise ValueError('decimation must be positive.')
        if n_band != decimation:
            raise ValueError('This initial implementation requires n_band == decimation.')
        self.n_band = n_band
        self.decimation = decimation
        self._decomposition = PolyphaseDecomposition(decimation)

    def build_branch_matrices(
        self,
        analysis_prototype: PrototypeFilter,
        *,
        synthesis_prototype_length: int | None = None,
    ) -> np.ndarray:
        self._validate_prototype(analysis_prototype)
        synthesis_prototype_length = (
            analysis_prototype.prototype_length if synthesis_prototype_length is None else synthesis_prototype_length
        )
        if synthesis_prototype_length <= 0 or synthesis_prototype_length % self.decimation != 0:
            raise ValueError('synthesis_prototype_length must be a positive multiple of decimation.')

        analysis_polyphase = self._decomposition.decompose(analysis_prototype)
        synthesis_n_phase = synthesis_prototype_length // self.decimation
        response_length = analysis_polyphase.shape[0] + synthesis_n_phase - 1
        matrices = np.zeros((self.decimation, response_length, synthesis_n_phase), dtype=np.complex64)
        for phase_idx in range(self.decimation):
            branch = analysis_polyphase[:, phase_idx]
            for tap_idx in range(synthesis_n_phase):
                basis = np.zeros(synthesis_n_phase, dtype=np.complex64)
                basis[tap_idx] = 1.0
                matrices[phase_idx, :, tap_idx] = np.convolve(branch, basis)
        return matrices

    def design_synthesis_prototype(
        self,
        analysis_prototype: PrototypeFilter,
        *,
        delay_blocks: int,
        synthesis_prototype_length: int | None = None,
        regularization: float = 0.0,
        branch_matrices: np.ndarray | None = None,
    ) -> PrototypeFilter:
        self._validate_prototype(analysis_prototype)
        matrices = (
            self.build_branch_matrices(
                analysis_prototype,
                synthesis_prototype_length=synthesis_prototype_length,
            )
            if branch_matrices is None
            else np.asarray(branch_matrices, dtype=np.complex64)
        )
        response_length = matrices.shape[1]
        synthesis_n_phase = matrices.shape[2]
        if delay_blocks < 0 or delay_blocks >= response_length:
            raise ValueError('delay_blocks is out of range for the branch response length.')
        if regularization < 0.0:
            raise ValueError('regularization must be non-negative.')

        coeffs = np.zeros((synthesis_n_phase, self.decimation), dtype=np.complex64)
        target = np.zeros(response_length, dtype=np.complex64)
        target[delay_blocks] = 1.0
        for phase_idx in range(self.decimation):
            matrix = matrices[phase_idx]
            if regularization == 0.0:
                solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
            else:
                lhs = matrix.conj().T @ matrix + regularization * np.eye(synthesis_n_phase, dtype=np.complex64)
                rhs = matrix.conj().T @ target
                solution = np.linalg.solve(lhs, rhs)
            coeffs[:, phase_idx] = solution

        return PrototypeFilter(coeffs.reshape(-1), n_band=self.n_band, decimation=self.decimation)

    def evaluate_pair_residual(
        self,
        analysis_prototype: PrototypeFilter,
        synthesis_prototype: PrototypeFilter,
        *,
        delay_blocks: int,
        branch_matrices: np.ndarray | None = None,
    ) -> dict[str, float]:
        self._validate_prototype(analysis_prototype)
        self._validate_prototype(synthesis_prototype)
        matrices = (
            self.build_branch_matrices(
                analysis_prototype,
                synthesis_prototype_length=synthesis_prototype.prototype_length,
            )
            if branch_matrices is None
            else np.asarray(branch_matrices, dtype=np.complex64)
        )
        if matrices.shape[2] != synthesis_prototype.prototype_length // self.decimation:
            raise ValueError('branch_matrices do not match the synthesis prototype length.')
        if delay_blocks < 0 or delay_blocks >= matrices.shape[1]:
            raise ValueError('delay_blocks is out of range for the branch response length.')

        target = np.zeros(matrices.shape[1], dtype=np.complex64)
        target[delay_blocks] = 1.0
        synthesis_polyphase = self._decomposition.decompose(synthesis_prototype)
        errors = []
        for phase_idx in range(self.decimation):
            response = matrices[phase_idx] @ synthesis_polyphase[:, phase_idx]
            errors.append(response - target)
        error = np.stack(errors, axis=0)
        return {
            'max_abs_error': float(np.max(np.abs(error))),
            'rms_error': float(np.sqrt(np.mean(np.abs(error) ** 2))),
        }

    def _validate_prototype(self, prototype: PrototypeFilter) -> None:
        if prototype.n_band != self.n_band or prototype.decimation != self.decimation:
            raise ValueError('prototype does not match the configured n_band/decimation.')
