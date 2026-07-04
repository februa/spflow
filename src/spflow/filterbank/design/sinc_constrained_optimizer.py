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

    # 半帯域電力応答 P(ω) = 1 + Σ_k a_k 2cos((2k+1)ω) を再構成し、
    # その自己相関列から最小位相側のスペクトル因子を取り出して解析ローパス係数に戻す。
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
    # basis shape: [fft_size, order]。
    # 線形位相 halfband FIR の電力応答は奇数ラグの余弦級数で書けるため、
    # 係数最適化を FIR 係数そのものではなく odd-lag の自己相関係数に落として凸に近い形で扱う。
    basis = np.stack([2.0 * np.cos((2 * idx + 1) * omega) for idx in range(order)], axis=1)

    weights = np.ones_like(omega)
    transition = (omega >= config.passband_edge_scale * np.pi) & (omega <= config.stopband_edge_scale * np.pi)
    # 遷移帯域は理想 sinc と有限長 FIR の食い違いが最も大きく、そこを厳密一致させると
    # 通過域リップルや阻止域減衰が悪化しやすいため、重みを下げて両側帯域の品質を優先する。
    weights[transition] = config.transition_weight
    return omega, target, basis, weights


def _initialize_odd_lag_coefficients(
    target: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    positivity_floor: float,
) -> np.ndarray:
    # weighted_basis shape: [n_freq, order]、weighted_target shape: [n_freq]。
    # P(ω) - 1 を odd-lag 基底へ最小二乗射影し、制約付き探索の初期値を sinc 目標に近づける。
    weighted_basis = basis * weights[:, np.newaxis]
    weighted_target = (target - 1.0) * weights
    coeffs, *_ = np.linalg.lstsq(weighted_basis, weighted_target, rcond=None)

    scale = 1.0
    # P(ω) が 0 以下になるとスペクトル因子が実 FIR として存在しなくなるため、
    # 自己相関列が正値を保つ範囲まで係数全体を縮めて安全な初期点に戻す。
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
            # 1 係数ずつ座標降下し、粗い刻みから細かい刻みへ順に試す。
            # 電力応答は非線形制約付きなので勾配法よりも失敗時の巻き戻しが明確な探索を優先する。
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

        # 改善した周回では探索半径を少しだけ縮め、停滞した周回では大きく縮めることで、
        # 早い段階では広く探索し、終盤では positivity 制約を壊さない微調整へ移る。
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
        # 数値誤差で 0 に接近した自己相関列はスペクトル因子化で不安定になるため、
        # ここでは候補を即座に棄却して実装を安全側へ倒す。
        return np.inf
    residual = (power_response - target) * weights
    return float(np.sqrt(np.mean(residual**2)))


def _build_halfband_power_response(coeffs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    # basis shape: [n_freq, order]、coeffs shape: [order]。
    # 各周波数点で odd-lag 余弦基底を線形結合し、半帯域フィルタの電力応答 P(ω) を評価する。
    return 1.0 + basis @ np.asarray(coeffs, dtype=np.float32)


def _spectral_factorize_halfband_power(coeffs: np.ndarray, num_taps: int) -> np.ndarray:
    max_lag = num_taps - 1
    autocorr = np.zeros(2 * max_lag + 1, dtype=np.float32)
    autocorr[max_lag] = 1.0
    for idx, value in enumerate(np.asarray(coeffs, dtype=np.float32)):
        lag = 2 * idx + 1
        # autocorr shape: [2 * max_lag + 1]。中心が 0 ラグで、左右へ奇数ラグ係数を対称配置する。
        # halfband 電力応答のフーリエ級数係数を自己相関列へ戻すことで、
        # 実係数 FIR のスペクトル因子化問題へ変換する。
        autocorr[max_lag + lag] = value
        autocorr[max_lag - lag] = value

    roots = np.roots(autocorr)
    # 単位円内の根だけを採用すると最小位相側のスペクトル因子になり、
    # 有限長の解析ローパスを安定に一意決定しやすい。
    inside_or_smallest = sorted(roots, key=lambda root: abs(root))[:max_lag]
    poly = np.real_if_close(np.poly(inside_or_smallest), tol=1000)
    if np.iscomplexobj(poly):
        imag_peak = float(np.max(np.abs(np.imag(poly))))
        if imag_peak > 1e-4:
            raise RuntimeError(f"Spectral factorization left a residual imaginary part: {imag_peak}")
        poly = np.real(poly)
    taps = np.asarray(poly, dtype=np.float32)
    # QMF 正規化ではローパス DC 利得を √2 に合わせる必要があるため、
    # 係数和が √2 になるようにスケーリングしてエネルギ整合を取る。
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
    # 複素白色雑音を使うことで全帯域をほぼ一様に励振し、
    # 特定周波数だけでは見落としやすい PR 誤差や位相ずれを時間領域でまとめて観測する。
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
