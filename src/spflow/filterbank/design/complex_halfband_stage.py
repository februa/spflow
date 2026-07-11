"""spflow.filterbank.design.complex_halfband_stage を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from ..complex_halfband_stage import ComplexFIRHalfbandStage, ComplexFIRHalfbandStageFilters
from ..halfband_stage_candidates import OrthonormalQMFStageCandidate


@dataclass(frozen=True)
class ResolvedHalfbandStageParameters:
    """完全再構成を満たした位相・遅延条件を保持する。"""

    analysis_phase: int
    synthesis_phase: int
    delay_compensation: int
    reconstruction_max_abs_error: float


def design_daubechies_qmf_lowpass(order: int) -> np.ndarray:
    """長さ `2 * order` の直交 Daubechies 低域係数を返す。"""

    if order <= 0:
        raise ValueError("order must be positive.")

    polynomial = np.array([comb(order - 1 + k, k) for k in range(order)], dtype=float)
    # Daubechies 多項式の根から最小位相半分を選び、直交 QMF の低域係数を構成する。
    y_roots = np.roots(polynomial[::-1])

    z_roots = []
    for root in y_roots:
        pair = np.roots([1.0, 4.0 * root - 2.0, 1.0])
        z_roots.append(min(pair, key=abs))

    taps = np.array([1.0], dtype=complex)
    for _ in range(order):
        taps = np.convolve(taps, np.array([1.0, 1.0], dtype=complex))
    for root in z_roots:
        taps = np.convolve(taps, np.array([1.0, -root], dtype=complex))

    taps = np.real_if_close(taps, tol=1000)
    if np.iscomplexobj(taps):
        imag_peak = float(np.max(np.abs(np.imag(taps))))
        if imag_peak > 1e-4:
            raise RuntimeError(f"Daubechies tap synthesis left a residual imaginary part: {imag_peak}")
        taps = np.real(taps)
    taps = np.asarray(taps, dtype=np.float32)
    taps *= np.sqrt(2.0) / np.sum(taps)
    return taps


def qmf_analysis_high_from_low(analysis_low: np.ndarray) -> np.ndarray:
    """直交 QMF 条件から high 側解析フィルタを導く。"""

    low = np.asarray(analysis_low, dtype=np.float32)
    if low.ndim != 1 or low.size == 0:
        raise ValueError("analysis_low must be a non-empty one-dimensional array.")
    n = np.arange(low.size, dtype=np.int64)
    return ((-1.0) ** n) * low[::-1]


def resolve_qmf_stage_parameters(
    analysis_low: np.ndarray,
    *,
    max_delay_compensation: int | None = None,
    tolerance: float = 1e-9,
) -> ResolvedHalfbandStageParameters:
    """与えた lowpass 係数に対して PR を満たす位相・遅延条件を探索する。"""

    low = np.asarray(analysis_low, dtype=np.float32)
    high = qmf_analysis_high_from_low(low)
    synthesis_low = low[::-1]
    synthesis_high = high[::-1]
    max_delay = max_delay_compensation if max_delay_compensation is not None else 2 * low.size + 2

    signals = _reference_signals()
    best: ResolvedHalfbandStageParameters | None = None
    best_error = np.inf

    for analysis_phase in (0, 1):
        for synthesis_phase in (0, 1):
            for delay_compensation in range(max_delay + 1):
                filters = ComplexFIRHalfbandStageFilters(
                    analysis_low=low,
                    analysis_high=high,
                    synthesis_low=synthesis_low,
                    synthesis_high=synthesis_high,
                    analysis_phase=analysis_phase,
                    synthesis_phase=synthesis_phase,
                    delay_compensation=delay_compensation,
                )
                stage = ComplexFIRHalfbandStage(filters)
                error = 0.0
                valid = True
                for signal in signals:
                    low_band, high_band = stage.analysis(signal)
                    reconstructed = stage.synthesis(low_band, high_band, length=signal.shape[-1])
                    if reconstructed.shape != signal.shape:
                        valid = False
                        break
                    error = max(error, float(np.max(np.abs(reconstructed - signal))))
                if not valid:
                    continue
                if error < best_error:
                    best_error = error
                    best = ResolvedHalfbandStageParameters(
                        analysis_phase=analysis_phase,
                        synthesis_phase=synthesis_phase,
                        delay_compensation=delay_compensation,
                        reconstruction_max_abs_error=error,
                    )
                if error <= tolerance:
                    # 直前で同じ候補から生成したため、この時点の best は必ず非 None である。
                    if best is None:
                        raise RuntimeError("resolved stage parameters unexpectedly missing.")
                    return best

    if best is None:
        raise RuntimeError("Failed to resolve valid stage parameters for the supplied QMF taps.")
    return best


def make_daubechies_qmf_candidate(order: int) -> OrthonormalQMFStageCandidate:
    """Daubechies 係数から QMF stage 候補を組み立てる。"""

    low = design_daubechies_qmf_lowpass(order)
    params = resolve_qmf_stage_parameters(low)
    return OrthonormalQMFStageCandidate(
        name=f"daubechies_qmf_order{order}_taps{low.size}",
        analysis_low=low,
        analysis_phase=params.analysis_phase,
        synthesis_phase=params.synthesis_phase,
        delay_compensation=params.delay_compensation,
    )


def make_daubechies_qmf_candidates(orders: list[int] | tuple[int, ...]) -> dict[str, OrthonormalQMFStageCandidate]:
    """複数 order の Daubechies QMF 候補をまとめて返す。"""

    candidates = [make_daubechies_qmf_candidate(int(order)) for order in orders]
    return {candidate.name: candidate for candidate in candidates}


def _reference_signals() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal(257) + 1j * rng.standard_normal(257),
        rng.standard_normal(514) + 1j * rng.standard_normal(514),
    )
