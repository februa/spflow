"""平面波tone生成のlevel、遅延符号、FFT位相を検証する。"""

from __future__ import annotations

import numpy as np

from spflow import (
    integrate_one_sided_band_rms_power,
    one_sided_rfft_bin_rms_power,
    rms_amplitude_to_level_db,
    synthesize_plane_wave_tone,
)


def test_plane_wave_tone_matches_requested_rms_level_and_fft_phase() -> None:
    """整数bin toneで各channelのRMS levelと相対到達遅延位相が一致することを確認する。"""
    sample_count = 1024
    sampling_frequency_hz = 8192.0
    tone_bin_index = 128
    tone_frequency_hz = tone_bin_index * sampling_frequency_hz / sample_count
    sensor_positions_m = np.array([[0.0, 0.0, 0.0], [0.375, 0.0, 0.0]])
    generated = synthesize_plane_wave_tone(
        sensor_positions_m,
        np.array([1.0, 0.0, 0.0]),
        sound_speed_m_per_s=1500.0,
        sampling_frequency_hz=sampling_frequency_hz,
        sample_count=sample_count,
        frequency_hz=tone_frequency_hz,
        level_db_re_rms=-12.0,
    )

    spectrum = np.fft.rfft(generated.signal, axis=1)
    bin_power = one_sided_rfft_bin_rms_power(
        spectrum,
        sample_count=sample_count,
        frequency_axis=1,
    )
    tone_mask = np.zeros(spectrum.shape[1], dtype=np.bool_)
    tone_mask[tone_bin_index] = True
    tone_power = integrate_one_sided_band_rms_power(
        bin_power,
        tone_mask,
        frequency_axis=1,
    )
    observed_level_db = rms_amplitude_to_level_db(np.sqrt(tone_power))
    observed_phase_rad = np.angle(spectrum[:, tone_bin_index])
    expected_phase_rad = -2.0 * np.pi * tone_frequency_hz * generated.relative_delay_s
    phase_error_rad = (observed_phase_rad - expected_phase_rad + np.pi) % (2.0 * np.pi) - np.pi

    # source方向へ0.375 m近い第2センサは0.25 ms早着するため、delayは負になる。
    np.testing.assert_allclose(generated.relative_delay_s, np.array([0.0, -0.00025]))
    np.testing.assert_allclose(observed_level_db, np.full(2, -12.0), atol=1.0e-12)
    np.testing.assert_allclose(phase_error_rad, np.zeros(2), atol=1.0e-12)
