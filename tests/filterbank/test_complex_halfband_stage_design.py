"""complex halfband stage design に関する回帰試験。"""

import numpy as np

from spflow.filterbank.design.complex_halfband_stage import (
    design_daubechies_qmf_lowpass,
    make_daubechies_qmf_candidate,
    qmf_analysis_high_from_low,
    resolve_qmf_stage_parameters,
)
from spflow.filterbank.halfband_stage_candidates import make_known_qmf_candidates


def test_daubechies_generator_matches_known_small_orders():
    """Daubechies 係数生成器について 既知の低次結果と一致する を確認する。"""
    known = make_known_qmf_candidates()

    np.testing.assert_allclose(design_daubechies_qmf_lowpass(2), known["daubechies_qmf_order2_taps4"].analysis_low, atol=1e-6)
    np.testing.assert_allclose(design_daubechies_qmf_lowpass(3), known["daubechies_qmf_order3_taps6"].analysis_low, atol=1e-6)
    np.testing.assert_allclose(design_daubechies_qmf_lowpass(4), known["daubechies_qmf_order4_taps8"].analysis_low, atol=1e-6)


def test_resolved_parameters_reconstruct_for_high_order_daubechies_candidate():
    """解決された stage パラメータについて 高次 Daubechies 候補で再構成条件を満たす を確認する。"""
    candidate = make_daubechies_qmf_candidate(24)
    stage = candidate.make_stage()

    rng = np.random.default_rng(90)
    x = rng.standard_normal(513) + 1j * rng.standard_normal(513)
    low, high = stage.analysis(x)
    reconstructed = stage.synthesis(low, high, length=x.shape[-1])

    np.testing.assert_allclose(reconstructed, x, atol=1e-5)


def test_order_24_candidate_meets_formal_80db_stopband_target():
    """24 次候補が正式な 80 dB stopband 目標を満たすことを確認する。"""
    candidate = make_daubechies_qmf_candidate(24)
    metrics = candidate.response_metrics()

    assert metrics["low_stopband_attenuation_db"] >= 80.0
    assert metrics["high_stopband_attenuation_db"] >= 80.0


def test_resolve_qmf_stage_parameters_returns_a_low_error_solution():
    """QMF stage パラメータ解決が低誤差の解を返すことを確認する。"""
    low = design_daubechies_qmf_lowpass(6)
    params = resolve_qmf_stage_parameters(low)

    assert params.reconstruction_max_abs_error <= 1e-5
    assert params.analysis_phase in (0, 1)
    assert params.synthesis_phase in (0, 1)
    assert params.delay_compensation >= 0


def test_qmf_highpass_has_expected_alternating_reversal():
    """QMF highpassについて 期待通りの交互反転になる を確認する。"""
    low = np.array([1.0, 2.0, 3.0, 4.0])
    high = qmf_analysis_high_from_low(low)

    np.testing.assert_array_equal(high, np.array([4.0, -3.0, 2.0, -1.0]))
