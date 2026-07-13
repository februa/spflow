"""長大ULAの正式S2a/T2a広帯域endfire評価を回帰検証する。"""

from functools import lru_cache

import numpy as np

from evaluations.beamforming.formal_s2a_t2a_endfire import (
    AZIMUTH_DEG,
    CovarianceStage,
    METHOD_IDS,
    WeightStage,
    estimate_covariance_stage,
    design_weight_stage,
    generate_scenario_signals,
    realize_residual_fir,
    theoretical_grating_azimuths,
)


@lru_cache(maxsize=1)
def _zero_endfire_stages() -> tuple[CovarianceStage, WeightStage]:
    """4096 snapshotの正式0 deg条件を同一test process内で一度だけ設計する。"""
    covariance = estimate_covariance_stage(generate_scenario_signals(0.0))
    return covariance, design_weight_stage(covariance)


def test_occupied_band_has_no_predicted_grating_lobe() -> None:
    """40--88 Hzでは6.25 m間隔の両endfireに空間aliasがないことを確認する。"""
    for frequency_hz in (40.0, 64.0, 88.0):
        assert theoretical_grating_azimuths(frequency_hz, 0.0) == ()
        assert theoretical_grating_azimuths(frequency_hz, 180.0) == ()


def test_t2a_preserves_endfire_music_while_s2a_fails_before_fir() -> None:
    """tapに依存しない共分散段でT2aだけが0 deg MUSIC peakを保持する。"""
    _, weights = _zero_endfire_stages()
    source_index = int(np.argmin(np.abs(AZIMUTH_DEG - 0.0)))
    s2a_index = METHOD_IDS.index("S2a")
    t2a_index = METHOD_IDS.index("T2a")

    assert float(weights.music_peak_deg[s2a_index, 0, source_index]) >= 20.0
    assert float(weights.music_peak_deg[t2a_index, 0, source_index]) == 0.0
    assert int(weights.signal_counts[t2a_index, 0, source_index]) > 1


def test_t2a_finite_residual_fir_reaches_high_energy_containment() -> None:
    """T2aの1024 tap残差FIRが有限でtarget beam energyの99%以上を含む。"""
    covariance, weights = _zero_endfire_stages()
    realization = realize_residual_fir("T2a", 1024, covariance, weights)
    source_index = int(np.argmin(np.abs(AZIMUTH_DEG - 0.0)))

    assert bool(np.all(np.isfinite(realization.coefficients)))
    assert float(realization.energy_containment[source_index]) >= 0.99
