"""長大ULA分析幅評価の決定論的成立条件を確認する。"""

import numpy as np

from evaluations.beamforming.analysis_width_long_array_mvdr import (
    APERTURE_DELAY_S,
    FS_HZ,
    SCAN_AZIMUTHS_DEG,
    _curve_metrics,
    _delays,
    _direction_covariance,
    _eta,
    _positions,
    _steering,
)


def test_direction_time_cut_preserves_endfire_eta_at_256_hz_width() -> None:
    """物理遅延を圧縮しない時間切り出しが粗い分析幅でもendfireを保持する。"""

    positions = _positions()
    scan_delays = _delays(positions, SCAN_AZIMUTHS_DEG)
    true_delay = _delays(positions, np.asarray([0.0], dtype=np.float32))[0]
    scan_steering = _steering(scan_delays, 256.0)
    true_steering = _steering(true_delay[None, :], 256.0)[0]
    covariance = _direction_covariance(
        true_delay,
        scan_delays,
        true_steering,
        delta_f_hz=256.0,
        time_cut=True,
        tone=False,
        scene="target_plus_noise",
    )

    metrics = _curve_metrics(_eta(covariance, scan_steering), 0.0)
    assert metrics["peak_error_deg"] == 0.0
    assert metrics["source_far_peak_margin"] > 0.9


def test_same_time_endfire_fails_narrowband_model_at_256_hz_width() -> None:
    """同一時間blockではΔfτが大きいendfireの単一steering近似が破綻する。"""

    positions = _positions()
    scan_delays = _delays(positions, SCAN_AZIMUTHS_DEG)
    true_delay = _delays(positions, np.asarray([0.0], dtype=np.float32))[0]
    scan_steering = _steering(scan_delays, 256.0)
    true_steering = _steering(true_delay[None, :], 256.0)[0]
    covariance = _direction_covariance(
        true_delay,
        scan_delays,
        true_steering,
        delta_f_hz=256.0,
        time_cut=False,
        tone=False,
        scene="target_plus_noise",
    )

    metrics = _curve_metrics(_eta(covariance, scan_steering), 0.0)
    assert metrics["peak_error_deg"] > 2.0
    assert metrics["source_far_peak_margin"] < 0.0


def test_one_hz_window_and_delay_aperture_do_not_fit_one_second() -> None:
    """1秒FFTへ262.5 ms遅延をscaleせず加えると1秒scheduleを超える。"""

    nfft = int(FS_HZ / 1.0)
    assert nfft / FS_HZ + APERTURE_DELAY_S > 1.0
