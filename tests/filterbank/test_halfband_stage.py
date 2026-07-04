"""halfband stage に関する回帰試験。"""

import numpy as np

from spflow.filterbank.halfband_stage import ParaunitaryHalfbandStagePrototype


def test_paraunitary_halfband_stage_prototype_reconstructs_complex_signal():
    """試作 paraunitary halfband stageが複素信号を再構成することを確認する。"""
    rng = np.random.default_rng(50)
    x = rng.standard_normal((4, 66)) + 1j * rng.standard_normal((4, 66))
    stage = ParaunitaryHalfbandStagePrototype()

    low, high = stage.analysis(x)
    reconstructed = stage.synthesis(low, high)

    np.testing.assert_allclose(reconstructed, x, atol=1e-6)


def test_paraunitary_halfband_stage_prototype_is_power_complementary():
    """試作 paraunitary halfband stageがパワー相補条件を満たすことを確認する。"""
    stage = ParaunitaryHalfbandStagePrototype()
    metrics = stage.response_metrics()

    assert metrics["power_complementarity_error"] <= 1e-6


def test_paraunitary_halfband_stage_prototype_is_not_selective_enough_for_formal_stage():
    """試作 paraunitary halfband stageが正式 stage に必要な選択度へ届かないことを確認する。"""
    stage = ParaunitaryHalfbandStagePrototype()
    metrics = stage.response_metrics()

    assert metrics["low_stopband_attenuation_db"] < 80.0
    assert metrics["high_stopband_attenuation_db"] < 80.0
