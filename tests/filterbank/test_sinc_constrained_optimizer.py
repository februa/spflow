"""sinc constrained optimizer に関する回帰試験。"""

import numpy as np

from spflow.filterbank.design.sinc_constrained_optimizer import (
    ConstrainedSincQMFOptimizerConfig,
    make_constrained_sinc_qmf_candidate,
)


def test_constrained_sinc_optimizer_returns_low_pr_error_candidate():
    """constrained sinc 最適化器について 低 PR 誤差の候補を返す を確認する。"""
    candidate, diagnostics = make_constrained_sinc_qmf_candidate(
        ConstrainedSincQMFOptimizerConfig(num_taps=16, max_passes=20, fft_size=4096)
    )

    assert candidate.analysis_low.size == 16
    assert diagnostics.stage_pr_max_abs_error < 1e-2
    assert diagnostics.stage_pr_rms_error < 5e-3
    assert diagnostics.power_complementarity_error < 1e-2


def test_constrained_sinc_optimizer_candidate_reconstructs_signal_better_than_naive_sinc():
    """constrained sinc 最適化器について 候補が単純な sinc より良く信号を再構成する を確認する。"""
    candidate, _ = make_constrained_sinc_qmf_candidate(
        ConstrainedSincQMFOptimizerConfig(num_taps=24, max_passes=20, fft_size=4096)
    )
    stage = candidate.make_stage()

    rng = np.random.default_rng(123)
    x = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    low, high = stage.analysis(x)
    reconstructed = stage.synthesis(low, high, length=x.shape[-1])

    assert np.max(np.abs(reconstructed - x)) < 2e-2
