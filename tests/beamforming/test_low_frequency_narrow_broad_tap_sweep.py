"""低周波tap sweepの事前グレーティング判定を検証する。"""

import numpy as np

from evaluations.beamforming.low_frequency_narrow_broad_tap_sweep import (
    predict_grating_lobe_azimuths,
)


def test_predicts_150_hz_grating_lobe_from_array_spacing() -> None:
    """150 Hz・6 m間隔・150 deg信号の理論偽像を約36.8 degに求める。"""
    predicted = predict_grating_lobe_azimuths(
        150.0, np.asarray([150.0], dtype=np.float64)
    )

    np.testing.assert_allclose(predicted, np.asarray([36.808618], dtype=np.float64), atol=1.0e-6)


def test_64_hz_has_no_visible_grating_lobe() -> None:
    """半波長条件内の64 Hzでは可視領域にグレーティングローブを作らない。"""
    predicted = predict_grating_lobe_azimuths(
        150.0, np.asarray([64.0], dtype=np.float64)
    )

    assert predicted.size == 0
