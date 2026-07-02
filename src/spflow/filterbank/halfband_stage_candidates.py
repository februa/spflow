"""spflow.filterbank.halfband_stage_candidates を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .complex_halfband_stage import ComplexFIRHalfbandStage, ComplexFIRHalfbandStageFilters


@dataclass(frozen=True)
class OrthonormalQMFStageCandidate:
    """直交 QMF halfband stage の係数候補を表す。"""

    name: str
    analysis_low: np.ndarray
    analysis_phase: int
    synthesis_phase: int
    delay_compensation: int

    @property
    def analysis_high(self) -> np.ndarray:
        n = np.arange(self.analysis_low.size)
        return ((-1.0) ** n) * self.analysis_low[::-1]

    @property
    def synthesis_low(self) -> np.ndarray:
        return self.analysis_low[::-1]

    @property
    def synthesis_high(self) -> np.ndarray:
        return self.analysis_high[::-1]

    def make_stage_filters(self) -> ComplexFIRHalfbandStageFilters:
        return ComplexFIRHalfbandStageFilters(
            analysis_low=self.analysis_low,
            analysis_high=self.analysis_high,
            synthesis_low=self.synthesis_low,
            synthesis_high=self.synthesis_high,
            analysis_phase=self.analysis_phase,
            synthesis_phase=self.synthesis_phase,
            delay_compensation=self.delay_compensation,
        )

    def make_stage(self) -> ComplexFIRHalfbandStage:
        return ComplexFIRHalfbandStage(self.make_stage_filters())

    def response_metrics(self, fft_size: int = 65536) -> dict[str, float]:
        if fft_size <= 0:
            raise ValueError("fft_size must be positive.")

        n = np.arange(self.analysis_low.size, dtype=np.float32)
        omega = np.linspace(0.0, np.pi, fft_size // 2 + 1)
        low = np.abs(np.sum(self.analysis_low[np.newaxis, :] * np.exp(-1j * omega[:, np.newaxis] * n[np.newaxis, :]), axis=1))
        high = np.abs(np.sum(self.analysis_high[np.newaxis, :] * np.exp(-1j * omega[:, np.newaxis] * n[np.newaxis, :]), axis=1))

        low_pass = low[omega <= 0.25 * np.pi]
        low_stop = low[omega >= 0.75 * np.pi]
        high_pass = high[omega >= 0.75 * np.pi]
        high_stop = high[omega <= 0.25 * np.pi]
        eps = np.finfo(np.float32).tiny

        return {
            "low_passband_ripple_db": _ripple_db(low_pass),
            "high_passband_ripple_db": _ripple_db(high_pass),
            "low_stopband_attenuation_db": float(
                -20.0 * np.log10(max(float(np.max(low_stop)) / max(float(np.max(low_pass)), eps), eps))
            ),
            "high_stopband_attenuation_db": float(
                -20.0 * np.log10(max(float(np.max(high_stop)) / max(float(np.max(high_pass)), eps), eps))
            ),
            "power_complementarity_error": float(np.max(np.abs(low**2 + high**2 - 2.0))),
        }


KNOWN_QMF_CANDIDATE_ALIASES = {
    "haar2": "haar_qmf_taps2",
    "db2_len4": "daubechies_qmf_order2_taps4",
    "db3_len6": "daubechies_qmf_order3_taps6",
    "db4_len8": "daubechies_qmf_order4_taps8",
}


def make_known_qmf_candidates() -> dict[str, OrthonormalQMFStageCandidate]:
    """既知の QMF 候補一覧を返す。"""

    return {
        "haar_qmf_taps2": OrthonormalQMFStageCandidate(
            name="haar_qmf_taps2",
            analysis_low=np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0),
            analysis_phase=0,
            synthesis_phase=0,
            delay_compensation=1,
        ),
        "daubechies_qmf_order2_taps4": OrthonormalQMFStageCandidate(
            name="daubechies_qmf_order2_taps4",
            analysis_low=np.array(
                [
                    (1.0 + np.sqrt(3.0)) / (4.0 * np.sqrt(2.0)),
                    (3.0 + np.sqrt(3.0)) / (4.0 * np.sqrt(2.0)),
                    (3.0 - np.sqrt(3.0)) / (4.0 * np.sqrt(2.0)),
                    (1.0 - np.sqrt(3.0)) / (4.0 * np.sqrt(2.0)),
                ],
                dtype=np.float32,
            ),
            analysis_phase=0,
            synthesis_phase=0,
            delay_compensation=3,
        ),
        "daubechies_qmf_order3_taps6": OrthonormalQMFStageCandidate(
            name="daubechies_qmf_order3_taps6",
            analysis_low=np.array(
                [
                    0.3326705529500826,
                    0.8068915093110928,
                    0.4598775021184915,
                    -0.1350110200102546,
                    -0.08544127388224149,
                    0.03522629188570953,
                ],
                dtype=np.float32,
            ),
            analysis_phase=0,
            synthesis_phase=0,
            delay_compensation=5,
        ),
        "daubechies_qmf_order4_taps8": OrthonormalQMFStageCandidate(
            name="daubechies_qmf_order4_taps8",
            analysis_low=np.array(
                [
                    0.2303778133088964,
                    0.7148465705529154,
                    0.6308807679298587,
                    -0.027983769416859854,
                    -0.18703481171888114,
                    0.030841381835560764,
                    0.0328830116668852,
                    -0.010597401785069032,
                ],
                dtype=np.float32,
            ),
            analysis_phase=1,
            synthesis_phase=0,
            delay_compensation=6,
        ),
    }


def resolve_known_qmf_candidate_name(name: str) -> str:
    """別名を含めて既知候補の正式名へ正規化する。"""

    candidates = make_known_qmf_candidates()
    if name in candidates:
        return name
    if name in KNOWN_QMF_CANDIDATE_ALIASES:
        return KNOWN_QMF_CANDIDATE_ALIASES[name]
    raise ValueError(f"Unknown candidate_name: {name}")


def get_known_qmf_candidate(name: str) -> OrthonormalQMFStageCandidate:
    """候補名から QMF stage 候補を返す。"""

    candidates = make_known_qmf_candidates()
    return candidates[resolve_known_qmf_candidate_name(name)]


def _ripple_db(magnitude: np.ndarray) -> float:
    mag = np.asarray(magnitude, dtype=np.float32)
    if mag.size == 0:
        raise ValueError("magnitude must not be empty.")
    eps = np.finfo(np.float32).tiny
    return float(20.0 * np.log10(max(float(np.max(mag)), eps) / max(float(np.min(mag)), eps)))
