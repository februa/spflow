"""sinc target design に関する回帰試験。"""

import numpy as np

from spflow.filterbank.design.complex_halfband_stage import make_daubechies_qmf_candidate
from spflow.filterbank.design.sinc_target import (
    build_halfband_power_target,
    design_windowed_sinc_halfband_target,
    evaluate_candidate_against_sinc_target,
)


def test_windowed_sinc_halfband_target_is_normalized_to_sqrt2_sum():
    """windowed sinc halfband targetが総和 sqrt(2) に正規化されていることを確認する。"""
    taps = design_windowed_sinc_halfband_target(48, window="blackman")

    np.testing.assert_allclose(np.sum(taps), np.sqrt(2.0), atol=1e-6)


def test_halfband_power_target_is_complementary_on_frequency_grid():
    """halfband power targetが周波数グリッド上で相補的であることを確認する。"""
    _, target = build_halfband_power_target(48, window="blackman", fft_size=4096)

    np.testing.assert_allclose(target + target[::-1], 2.0, atol=1e-5)


def test_higher_order_daubechies_candidate_is_closer_to_sinc_target_than_order4():
    """高次 Daubechies 候補について is closer to sinc target than order4 を確認する。"""
    order4 = make_daubechies_qmf_candidate(4)
    order24 = make_daubechies_qmf_candidate(24)

    metrics4 = evaluate_candidate_against_sinc_target(order4)
    metrics24 = evaluate_candidate_against_sinc_target(order24)

    assert metrics24.fullband_rms_error < metrics4.fullband_rms_error
    assert metrics24.stopband_rms_error < metrics4.stopband_rms_error
