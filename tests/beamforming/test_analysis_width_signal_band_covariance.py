"""粗いFFT binでsteering etaとMVDRの成立条件が分かれることを確認する。"""

import numpy as np

from evaluations.beamforming.analysis_width_long_array_mvdr import SCAN_AZIMUTHS_DEG, _delays, _positions
from evaluations.beamforming.analysis_width_signal_band_covariance import (
    SIGNALS,
    SOURCE_AZIMUTH_DEG,
    _evaluate_condition,
    _nearest_fft_bin_center,
)


def test_256_hz_width_maps_low_frequency_band_to_dc_bin() -> None:
    """40--60 Hzを256 Hz幅で扱う場合、最寄りbinがDCになることを固定する。"""

    assert _nearest_fft_bin_center(SIGNALS[0], 256.0) == 0.0


def test_dc_bin_keeps_direction_specific_eta_but_cannot_form_mvdr_scan() -> None:
    """候補依存etaが残ってもDC steeringのMVDRは全方位を区別できない。"""

    positions = _positions()
    scan_delays = _delays(positions, SCAN_AZIMUTHS_DEG)
    true_delay = _delays(positions, np.asarray([SOURCE_AZIMUTH_DEG], dtype=np.float32))[0]
    result = _evaluate_condition(SIGNALS[0], 256.0, scan_delays, true_delay)
    target_plus_noise = result["scenes"]["target_plus_noise"]

    assert target_plus_noise["steering_eta_metrics"]["peak_error_deg"] == 0.0
    assert target_plus_noise["steering_eta_metrics"]["source_far_peak_margin"] > 0.7
    assert target_plus_noise["mvdr_metrics"]["source_far_peak_margin"] == 0.0
    assert target_plus_noise["mvdr_metrics"]["peak_error_deg"] == SOURCE_AZIMUTH_DEG
