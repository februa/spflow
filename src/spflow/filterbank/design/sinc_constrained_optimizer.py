"""spflow.filterbank.design.sinc_constrained_optimizer を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..halfband_stage_candidates import OrthonormalQMFStageCandidate
from .complex_halfband_stage import ResolvedHalfbandStageParameters, resolve_qmf_stage_parameters
from .sinc_target import build_halfband_power_target


@dataclass(frozen=True)
class ConstrainedSincQMFOptimizerConfig:
    """sinc 目標 constrained 最適化の設定を保持する。"""

    num_taps: int
    cutoff: float = 0.25
    window: str = "blackman"
    fft_size: int = 8192
    passband_edge_scale: float = 0.22
    stopband_edge_scale: float = 0.28
    transition_weight: float = 0.2
    positivity_floor: float = 1e-3
    initial_step: float = 0.05
    max_passes: int = 60
    reduction_if_improved: float = 0.95
    reduction_if_stalled: float = 0.7


@dataclass(frozen=True)
class ConstrainedSincQMFDiagnostics:
    """最適化結果の診断量を保持する。"""

    config: ConstrainedSincQMFOptimizerConfig
    weighted_power_rms_error: float
    min_halfband_power_value: float
    analysis_phase: int
    synthesis_phase: int
    delay_compensation: int
    stage_pr_max_abs_error: float
    stage_pr_rms_error: float
    low_stopband_attenuation_db: float
    high_stopband_attenuation_db: float
    max_passband_ripple_db: float
    power_complementarity_error: float
    odd_lag_coefficients: np.ndarray
    analysis_low: np.ndarray


def make_constrained_sinc_qmf_candidate(
    config: ConstrainedSincQMFOptimizerConfig,
) -> tuple[OrthonormalQMFStageCandidate, ConstrainedSincQMFDiagnostics]:
    """sinc 目標に近い constrained QMF 候補を構成する。"""

    if config.num_taps <= 0 or config.num_taps % 2 != 0:
        raise ValueError("num_taps must be a positive even integer.")

    omega, target, basis, weights = _build_odd_lag_power_basis(config)
    odd_lag_coefficients = _initialize_odd_lag_coefficients(target, basis, weights, config.positivity_floor)
    odd_lag_coefficients = _coordinate_descent_optimize(
        odd_lag_coefficients,
        target=target,
        basis=basis,
        weights=weights,
        config=config,
    )

    power_response = _build_halfband_power_response(odd_lag_coefficients, basis)
    analysis_low = _spectral_factorize_halfband_power(odd_lag_coefficients, config.num_taps)
    params = resolve_qmf_stage_parameters(analysis_low, tolerance=1e-6)
    candidate = OrthonormalQMFStageCandidate(
        name=f"sinc_target_constrained_{config.window}_taps{config.num_taps}",
        analysis_low=analysis_low,
        analysis_phase=params.analysis_phase,
        synthesis_phase=params.synthesis_phase,
        delay_compensation=params.delay_compensation,
    )
    diagnostics = _evaluate_candidate(
        candidate,
        params=params,
        config=config,
        odd_lag_coefficients=odd_lag_coefficients,
        power_response=power_response,
    )
    return candidate, diagnostics


def _build_odd_lag_power_basis(
    config: ConstrainedSincQMFOptimizerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omega, target = build_halfband_power_target(
        config.num_taps,
        cutoff=config.cutoff,
        window=config.window,
        fft_size=config.fft_size,
    )
    order = config.num_taps // 2
    basis = np.stack([2.0 * np.cos((2 * idx + 1) * omega) for idx in range(order)], axis=1)

    weights = np.ones_like(omega)
    transition = (omega >= config.passband_edge_scale * np.pi) & (omega <= config.stopband_edge_scale * np.pi)
    weights[transition] = config.transition_weight
    return omega, target, basis, weights


def _initialize_odd_lag_coefficients(
    target: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    positivity_floor: float,
) -> np.ndarray:
    weighted_basis = basis * weights[:, np.newaxis]
    weighted_target = (target - 1.0) * weights
    coeffs, *_ = np.linalg.lstsq(weighted_basis, weighted_target, rcond=None)

    scale = 1.0
    while np.min(_build_halfband_power_response(scale * coeffs, basis)) <= positivity_floor and scale > 1e-8:
        scale *= 0.95
    return scale * coeffs


def _coordinate_descent_optimize(
    initial_coeffs: np.ndarray,
    *,
    target: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    config: ConstrainedSincQMFOptimizerConfig,
) -> np.ndarray:
    coeffs = np.asarray(initial_coeffs, dtype=np.float32).copy()
    best = _weighted_power_objective(
        coeffs,
        target=target,
        basis=basis,
        weights=weights,
        positivity_floor=config.positivity_floor,
    )
    step = config.initial_step

    for _ in range(config.max_passes):
        improved = False
        for idx in range(coeffs.size):
            baseline = coeffs[idx]
            local_best = best
            local_value = baseline
            for delta in (step, -step, 0.5 * step, -0.5 * step, 0.25 * step, -0.25 * step):
                trial = coeffs.copy()
                trial[idx] = baseline + delta
                score = _weighted_power_objective(
                    trial,
                    target=target,
                    basis=basis,
                    weights=weights,
                    positivity_floor=config.positivity_floor,
                )
                if score < local_best:
                    local_best = score
                    local_value = trial[idx]
            if local_best < best:
                coeffs[idx] = local_value
                best = local_best
                improved = True

        step *= config.reduction_if_improved if improved else config.reduction_if_stalled
    return coeffs


def _weighted_power_objective(
    coeffs: np.ndarray,
    *,
    target: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    positivity_floor: float,
) -> float:
    power_response = _build_halfband_power_response(coeffs, basis)
    if np.min(power_response) <= positivity_floor:
        return np.inf
    residual = (power_response - target) * weights
    return float(np.sqrt(np.mean(residual**2)))


def _build_halfband_power_response(coeffs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return 1.0 + basis @ np.asarray(coeffs, dtype=np.float32)


def _spectral_factorize_halfband_power(coeffs: np.ndarray, num_taps: int) -> np.ndarray:
    max_lag = num_taps - 1
    autocorr = np.zeros(2 * max_lag + 1, dtype=np.float32)
    autocorr[max_lag] = 1.0
    for idx, value in enumerate(np.asarray(coeffs, dtype=np.float32)):
        lag = 2 * idx + 1
        autocorr[max_lag + lag] = value
        autocorr[max_lag - lag] = value

    roots = np.roots(autocorr)
    inside_or_smallest = sorted(roots, key=lambda root: abs(root))[:max_lag]
    poly = np.real_if_close(np.poly(inside_or_smallest), tol=1000)
    if np.iscomplexobj(poly):
        imag_peak = float(np.max(np.abs(np.imag(poly))))
        if imag_peak > 1e-4:
            raise RuntimeError(f"Spectral factorization left a residual imaginary part: {imag_peak}")
        poly = np.real(poly)
    taps = np.asarray(poly, dtype=np.float32)
    taps *= np.sqrt(2.0) / np.sum(taps)
    return taps


def _evaluate_candidate(
    candidate: OrthonormalQMFStageCandidate,
    *,
    params: ResolvedHalfbandStageParameters,
    config: ConstrainedSincQMFOptimizerConfig,
    odd_lag_coefficients: np.ndarray,
    power_response: np.ndarray,
) -> ConstrainedSincQMFDiagnostics:
    stage = candidate.make_stage()
    metrics = candidate.response_metrics()

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    low, high = stage.analysis(signal)
    reconstructed = stage.synthesis(low, high, length=signal.shape[-1])
    error = reconstructed - signal

    omega, target, basis, weights = _build_odd_lag_power_basis(config)
    del omega
    weighted_power_rms_error = _weighted_power_objective(
        odd_lag_coefficients,
        target=target,
        basis=basis,
        weights=weights,
        positivity_floor=config.positivity_floor,
    )

    return ConstrainedSincQMFDiagnostics(
        config=config,
        weighted_power_rms_error=weighted_power_rms_error,
        min_halfband_power_value=float(np.min(power_response)),
        analysis_phase=params.analysis_phase,
        synthesis_phase=params.synthesis_phase,
        delay_compensation=params.delay_compensation,
        stage_pr_max_abs_error=float(np.max(np.abs(error))),
        stage_pr_rms_error=float(np.sqrt(np.mean(np.abs(error) ** 2))),
        low_stopband_attenuation_db=metrics["low_stopband_attenuation_db"],
        high_stopband_attenuation_db=metrics["high_stopband_attenuation_db"],
        max_passband_ripple_db=max(metrics["low_passband_ripple_db"], metrics["high_passband_ripple_db"]),
        power_complementarity_error=metrics["power_complementarity_error"],
        odd_lag_coefficients=odd_lag_coefficients.copy(),
        analysis_low=candidate.analysis_low.copy(),
    )
