"""spflow.filterbank.prdft_modulated を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .prototype_bank import PrototypeFilter


class AnalysisBankProtocol(Protocol):
    """有限長PR評価に必要な解析バンク契約。"""

    @property
    def transient_length(self) -> int:
        """解析過渡長を返す。"""
        ...

    def analysis(self, x: np.ndarray) -> np.ndarray:
        """入力のsubband表現を返す。"""
        ...


class SynthesisBankProtocol(Protocol):
    """有限長PR評価に必要な合成バンク契約。"""

    @property
    def transient_length(self) -> int:
        """合成過渡長を返す。"""
        ...

    def synthesis(self, subbands: np.ndarray, *, length: int | None = None) -> np.ndarray:
        """subband表現から時間波形を再構成する。"""
        ...


@dataclass(frozen=True)
class DFTModulatedFilterDesigner:
    """原型 FIR から明示変調型の解析・合成フィルタ群を作る。"""

    n_band: int
    decimation: int

    def __post_init__(self) -> None:
        if self.n_band <= 0:
            raise ValueError("n_band must be positive.")
        if self.decimation <= 0:
            raise ValueError("decimation must be positive.")
        if self.n_band != self.decimation:
            raise ValueError("This initial implementation requires n_band == decimation.")

    def analysis_filters(self, prototype: PrototypeFilter) -> np.ndarray:
        """解析側の DFT 変調 FIR 群を返す。"""
        self._validate_prototype(prototype)
        n = np.arange(prototype.prototype_length, dtype=np.float32)
        k = np.arange(self.n_band, dtype=np.float32)[:, np.newaxis]
        # exp(-j 2π k n / K) により、原型低域 FIR を各帯域中心へ複素変調する。
        modulation = np.exp(-1j * 2.0 * np.pi * k * n[np.newaxis, :] / self.n_band)
        return prototype.coefficients[np.newaxis, :] * modulation

    def synthesis_filters(self, prototype: PrototypeFilter) -> np.ndarray:
        """合成側の DFT 変調 FIR 群を返す。"""
        self._validate_prototype(prototype)
        n = np.arange(prototype.prototype_length, dtype=np.float32)
        k = np.arange(self.n_band, dtype=np.float32)[:, np.newaxis]
        modulation = np.exp(1j * 2.0 * np.pi * k * n[np.newaxis, :] / self.n_band)
        return (prototype.coefficients[np.newaxis, :] * modulation) / self.n_band

    def _validate_prototype(self, prototype: PrototypeFilter) -> None:
        if prototype.n_band != self.n_band or prototype.decimation != self.decimation:
            raise ValueError("prototype does not match the configured n_band/decimation.")


class _ModulatedBankBase:
    def __init__(self, *, prototype: PrototypeFilter, band_order: str = "fft", axis: int = -1) -> None:
        if band_order != "fft":
            raise ValueError("Only FFT band order is currently supported.")
        self.prototype = prototype
        self.n_band = prototype.n_band
        self.decimation = prototype.decimation
        self.prototype_length = prototype.prototype_length
        self.band_order = band_order
        self.axis = axis
        self.designer = DFTModulatedFilterDesigner(self.n_band, self.decimation)

    @property
    def transient_length(self) -> int:
        return self.prototype_length - self.decimation

    def _normalize_axis(self, axis: int, ndim: int) -> int:
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis is out of bounds for input.")
        return axis

    def _frame_signal(self, x: np.ndarray) -> np.ndarray:
        n_samples = x.shape[-1]
        if n_samples == 0:
            return np.zeros(x.shape[:-1] + (self.prototype_length, 0), dtype=np.complex64)
        n_frames = int(np.ceil(n_samples / self.decimation))
        padded_length = self.prototype_length + max(0, n_frames - 1) * self.decimation
        pad_width = padded_length - n_samples
        padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad_width)])
        frames = []
        for frame_idx in range(n_frames):
            start = frame_idx * self.decimation
            stop = start + self.prototype_length
            frames.append(padded[..., start:stop])
        return np.stack(frames, axis=-1)


class PRDFTAnalysisBank(_ModulatedBankBase):
    """明示変調 FIR と間引きを使う解析バンク。"""

    def __init__(self, *, prototype: PrototypeFilter, band_order: str = "fft", axis: int = -1) -> None:
        super().__init__(prototype=prototype, band_order=band_order, axis=axis)
        self.filters = self.designer.analysis_filters(prototype)

    def analysis(self, x: np.ndarray) -> np.ndarray:
        """明示変調 FIR 解析バンクでサブバンド列を得る。"""
        arr = np.asarray(x, dtype=np.complex64)
        signal_axis = self._normalize_axis(self.axis, arr.ndim)
        moved = np.moveaxis(arr, signal_axis, -1)
        frames = self._frame_signal(moved)
        # frames shape: [..., n_tap, n_frame]
        # filters shape: [n_band, n_tap]
        # einsum により各フレームと各帯域 FIR の内積を一括計算する。
        subbands = np.einsum('...nf,kn->...kf', frames, self.filters, optimize=True)
        return np.moveaxis(subbands, -2, signal_axis)


class PRDFTSynthesisBank(_ModulatedBankBase):
    """明示変調 FIR と補間を使う合成バンク。"""

    def __init__(
        self,
        *,
        prototype: PrototypeFilter,
        delay_compensation: int = 0,
        band_order: str = "fft",
        axis: int = -1,
    ) -> None:
        super().__init__(prototype=prototype, band_order=band_order, axis=axis)
        self.delay_compensation = delay_compensation
        self.filters = self.designer.synthesis_filters(prototype)

    def synthesis(self, subbands: np.ndarray, length: int | None = None) -> np.ndarray:
        """明示変調 FIR 合成バンクで時間列を再構成する。"""
        arr = np.asarray(subbands, dtype=np.complex64)
        band_axis = self._normalize_axis(self.axis, arr.ndim - 1)
        if arr.shape[band_axis] != self.n_band:
            raise ValueError("subbands shape does not match the configured number of bands.")

        moved = np.moveaxis(arr, band_axis, -2)
        # moved shape: [..., n_band, n_frame]
        # filters shape: [n_band, n_tap]
        # 帯域ごとの合成 FIR 出力を足し合わせ、各フレームの時間波形を得る。
        frames = np.einsum('...kf,kn->...nf', moved, self.filters, optimize=True)
        reconstructed = self._overlap_add(frames)
        compensated = self._apply_delay_compensation(reconstructed)
        if length is not None:
            compensated = compensated[..., :length]
        return np.moveaxis(compensated, -1, band_axis)

    def _overlap_add(self, frames: np.ndarray) -> np.ndarray:
        frame_size = frames.shape[-2]
        n_frame = frames.shape[-1]
        out_length = frame_size + max(0, n_frame - 1) * self.decimation
        output = np.zeros(frames.shape[:-2] + (out_length,), dtype=np.complex64)
        for frame_idx in range(n_frame):
            start = frame_idx * self.decimation
            stop = start + frame_size
            output[..., start:stop] += frames[..., :, frame_idx]
        return output

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


class FiniteLengthPRChecker:
    """明示 padding/crop/valid-region 規約で有限長 PR を評価する。"""

    def __init__(self, analysis_bank: AnalysisBankProtocol, synthesis_bank: SynthesisBankProtocol) -> None:
        self.analysis_bank = analysis_bank
        self.synthesis_bank = synthesis_bank

    def reconstruct_full(
        self,
        x: np.ndarray,
        *,
        pad_front: int | None = None,
        pad_back: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """有限長入力へ明示 padding を加えた full 再構成結果を返す。"""
        arr = np.asarray(x, dtype=np.complex64)
        front = self.analysis_bank.transient_length if pad_front is None else pad_front
        back = self.synthesis_bank.transient_length if pad_back is None else pad_back
        padded = np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(front, back)])
        subbands = self.analysis_bank.analysis(padded)
        reconstructed = self.synthesis_bank.synthesis(subbands, length=padded.shape[-1])
        return reconstructed, subbands, {
            "pad_front": int(front),
            "pad_back": int(back),
            "input_length": int(arr.shape[-1]),
            "padded_length": int(padded.shape[-1]),
        }

    def reconstruct(
        self,
        x: np.ndarray,
        *,
        pad_front: int | None = None,
        pad_back: int | None = None,
        crop_mode: str = "input_aligned",
        crop_front: int | None = None,
        crop_length: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """full 再構成結果から指定 crop 規約の区間だけを返す。"""
        arr = np.asarray(x, dtype=np.complex64)
        reconstructed, subbands, metadata = self.reconstruct_full(arr, pad_front=pad_front, pad_back=pad_back)
        start, stop = self._resolve_crop_bounds(
            metadata=metadata,
            crop_mode=crop_mode,
            crop_front=crop_front,
            crop_length=crop_length,
        )
        return reconstructed[..., start:stop], subbands

    def check(
        self,
        x: np.ndarray,
        *,
        pad_front: int | None = None,
        pad_back: int | None = None,
        crop_mode: str = "input_aligned",
        crop_front: int | None = None,
        crop_length: int | None = None,
        valid_region_mode: str = "transient",
        valid_margin: int | None = None,
    ) -> dict[str, float]:
        """有限長 PR 誤差と valid region 誤差を返す。"""
        arr = np.asarray(x, dtype=np.complex64)
        reconstructed, _ = self.reconstruct(
            arr,
            pad_front=pad_front,
            pad_back=pad_back,
            crop_mode=crop_mode,
            crop_front=crop_front,
            crop_length=crop_length,
        )
        _, _, metadata = self.reconstruct_full(arr, pad_front=pad_front, pad_back=pad_back)
        reference = self._resolve_reference(
            arr,
            metadata=metadata,
            crop_mode=crop_mode,
            crop_front=crop_front,
            crop_length=crop_length,
        )
        error = reconstructed - reference
        start, stop = self._resolve_valid_region_bounds(
            output_length=error.shape[-1],
            valid_region_mode=valid_region_mode,
            valid_margin=valid_margin,
        )
        valid_error = error[..., start:stop] if stop > start else error
        return {
            "max_abs_error": float(np.max(np.abs(error))),
            "rms_error": float(np.sqrt(np.mean(np.abs(error) ** 2))),
            "valid_max_abs_error": float(np.max(np.abs(valid_error))),
            "valid_rms_error": float(np.sqrt(np.mean(np.abs(valid_error) ** 2))),
        }

    def _resolve_crop_bounds(
        self,
        *,
        metadata: dict[str, int],
        crop_mode: str,
        crop_front: int | None,
        crop_length: int | None,
    ) -> tuple[int, int]:
        if crop_mode == "input_aligned":
            start = metadata["pad_front"]
            length = metadata["input_length"]
        elif crop_mode == "full":
            start = 0
            length = metadata["padded_length"]
        elif crop_mode == "valid":
            margin = max(self.analysis_bank.transient_length, self.synthesis_bank.transient_length)
            start = margin
            length = max(0, metadata["padded_length"] - 2 * margin)
        elif crop_mode == "custom":
            if crop_front is None:
                raise ValueError("crop_front must be provided when crop_mode='custom'.")
            start = crop_front
            length = metadata["input_length"] if crop_length is None else crop_length
        else:
            raise ValueError("Unsupported crop_mode.")

        if crop_mode != "custom" and crop_front is not None:
            start = crop_front
        if crop_length is not None:
            length = crop_length
        if start < 0 or length < 0:
            raise ValueError("crop bounds must be non-negative.")
        stop = min(metadata["padded_length"], start + length)
        return start, stop

    def _resolve_reference(
        self,
        x: np.ndarray,
        *,
        metadata: dict[str, int],
        crop_mode: str,
        crop_front: int | None,
        crop_length: int | None,
    ) -> np.ndarray:
        padded = np.pad(
            x,
            [(0, 0)] * (x.ndim - 1) + [(metadata["pad_front"], metadata["pad_back"])],
        )
        start, stop = self._resolve_crop_bounds(
            metadata=metadata,
            crop_mode=crop_mode,
            crop_front=crop_front,
            crop_length=crop_length,
        )
        return padded[..., start:stop]

    def _resolve_valid_region_bounds(
        self,
        *,
        output_length: int,
        valid_region_mode: str,
        valid_margin: int | None,
    ) -> tuple[int, int]:
        if valid_region_mode == "none":
            return 0, output_length
        if valid_region_mode == "transient":
            margin = max(self.analysis_bank.transient_length, self.synthesis_bank.transient_length)
        elif valid_region_mode == "analysis":
            margin = self.analysis_bank.transient_length
        elif valid_region_mode == "synthesis":
            margin = self.synthesis_bank.transient_length
        elif valid_region_mode == "custom":
            if valid_margin is None:
                raise ValueError("valid_margin must be provided when valid_region_mode='custom'.")
            margin = valid_margin
        else:
            raise ValueError("Unsupported valid_region_mode.")

        if valid_margin is not None and valid_region_mode != "custom":
            margin = valid_margin
        margin = max(0, margin)
        if output_length > 2 * margin:
            return margin, output_length - margin
        return 0, output_length


class PrototypePairDesigner:
    """PR を満たすよう解析原型に対する合成原型を設計する。"""

    def __init__(self, n_band: int, decimation: int) -> None:
        self.n_band = n_band
        self.decimation = decimation
        self.designer = DFTModulatedFilterDesigner(n_band=n_band, decimation=decimation)

    def build_cascade_matrix(
        self,
        analysis_prototype: PrototypeFilter,
        *,
        impulse_length: int | None = None,
        n_phase_inputs: int | None = None,
    ) -> tuple[np.ndarray, int, int]:
        """解析バンクと基底合成 FIR の cascade 行列を構築する。"""
        self._validate_prototype(analysis_prototype)
        impulse_length = analysis_prototype.prototype_length if impulse_length is None else impulse_length
        n_phase_inputs = self.decimation if n_phase_inputs is None else n_phase_inputs
        if impulse_length <= 0:
            raise ValueError("impulse_length must be positive.")
        if n_phase_inputs <= 0:
            raise ValueError("n_phase_inputs must be positive.")

        analysis = PRDFTAnalysisBank(prototype=analysis_prototype)
        basis_input = np.zeros(impulse_length, dtype=np.complex64)
        basis_input[0] = 1.0
        basis_subbands = analysis.analysis(basis_input)
        output_length = analysis_prototype.prototype_length + (basis_subbands.shape[-1] - 1) * self.decimation
        matrix = np.zeros((n_phase_inputs * output_length, analysis_prototype.prototype_length), dtype=np.complex64)

        subbands_by_phase = []
        for phase_idx in range(n_phase_inputs):
            impulse = np.zeros(impulse_length, dtype=np.complex64)
            impulse[phase_idx] = 1.0
            subbands_by_phase.append(analysis.analysis(impulse))

        for tap_idx in range(analysis_prototype.prototype_length):
            basis = np.zeros(analysis_prototype.prototype_length, dtype=np.complex64)
            basis[tap_idx] = 1.0
            basis_prototype = PrototypeFilter(
                basis,
                n_band=analysis_prototype.n_band,
                decimation=analysis_prototype.decimation,
            )
            synthesis = PRDFTSynthesisBank(prototype=basis_prototype, delay_compensation=0)
            column_blocks = []
            for subbands in subbands_by_phase:
                response = synthesis.synthesis(subbands, length=output_length)
                column_blocks.append(response)
            matrix[:, tap_idx] = np.concatenate(column_blocks, axis=0)
        return matrix, output_length, n_phase_inputs

    def design_synthesis_prototype(
        self,
        analysis_prototype: PrototypeFilter,
        *,
        delay_samples: int,
        impulse_length: int | None = None,
        cascade_matrix: tuple[np.ndarray, int, int] | None = None,
        regularization: float = 0.0,
    ) -> PrototypeFilter:
        """cascade 行列に対する最小二乗で合成原型を設計する。"""
        self._validate_prototype(analysis_prototype)
        matrix, output_length, n_phase_inputs = (
            self.build_cascade_matrix(analysis_prototype, impulse_length=impulse_length)
            if cascade_matrix is None
            else cascade_matrix
        )
        if delay_samples < 0 or delay_samples + n_phase_inputs - 1 >= output_length:
            raise ValueError("delay_samples is out of range for the cascade response length.")
        if regularization < 0.0:
            raise ValueError("regularization must be non-negative.")

        target = np.zeros(matrix.shape[0], dtype=np.complex64)
        for phase_idx in range(n_phase_inputs):
            target[phase_idx * output_length + delay_samples + phase_idx] = 1.0
        if regularization == 0.0:
            coeffs, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        else:
            # 正則化は cascade 行列が悪条件な場合に係数発散を抑えるために加える。
            lhs = matrix.conj().T @ matrix + regularization * np.eye(matrix.shape[1], dtype=np.complex64)
            rhs = matrix.conj().T @ target
            coeffs = np.linalg.solve(lhs, rhs)
        return PrototypeFilter(
            coeffs,
            n_band=analysis_prototype.n_band,
            decimation=analysis_prototype.decimation,
        )

    def evaluate_pair_residual(
        self,
        analysis_prototype: PrototypeFilter,
        synthesis_prototype: PrototypeFilter,
        *,
        delay_samples: int,
        impulse_length: int | None = None,
        cascade_matrix: tuple[np.ndarray, int, int] | None = None,
    ) -> dict[str, float]:
        """設計済み解析・合成原型の cascade 残差を評価する。"""
        self._validate_prototype(analysis_prototype)
        self._validate_prototype(synthesis_prototype)
        matrix, output_length, n_phase_inputs = (
            self.build_cascade_matrix(analysis_prototype, impulse_length=impulse_length)
            if cascade_matrix is None
            else cascade_matrix
        )
        target = np.zeros(matrix.shape[0], dtype=np.complex64)
        for phase_idx in range(n_phase_inputs):
            target[phase_idx * output_length + delay_samples + phase_idx] = 1.0
        response = matrix @ synthesis_prototype.coefficients
        error = response - target
        return {
            "max_abs_error": float(np.max(np.abs(error))),
            "rms_error": float(np.sqrt(np.mean(np.abs(error) ** 2))),
        }

    def _validate_prototype(self, prototype: PrototypeFilter) -> None:
        if prototype.n_band != self.n_band or prototype.decimation != self.decimation:
            raise ValueError("prototype does not match the configured n_band/decimation.")
