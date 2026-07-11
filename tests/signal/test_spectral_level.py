"""RMS level と one-sided spectrum の共通変換規約を検証する。"""

from __future__ import annotations

import numpy as np

from spflow.spectral_level import (
    integrate_one_sided_band_rms_power,
    noise_asd_level_db_to_sample_rms,
    one_sided_rfft_bin_rms_power,
    rms_amplitude_to_level_db,
    tone_rms_level_db_to_peak_amplitude,
)


def test_tone_level_and_rfft_bin_power_preserve_rms_level() -> None:
    """整数 bin tone の入力 RMS level が one-sided bin 積分後も保存されることを確認する。"""
    sample_count = 1024
    sampling_frequency_hz = 8192.0
    tone_frequency_hz = 1024.0
    expected_level_db = -6.0
    time_s = np.arange(sample_count, dtype=np.float64) / sampling_frequency_hz
    peak_amplitude = tone_rms_level_db_to_peak_amplitude(expected_level_db)
    signal = peak_amplitude * np.cos(2.0 * np.pi * tone_frequency_hz * time_s)

    spectrum = np.fft.rfft(signal)
    bin_power = one_sided_rfft_bin_rms_power(spectrum, sample_count=sample_count)
    tone_index = int(round(tone_frequency_hz * sample_count / sampling_frequency_hz))
    mask = np.zeros(bin_power.size, dtype=np.bool_)
    mask[tone_index] = True
    band_power = integrate_one_sided_band_rms_power(bin_power, mask)
    observed_level = rms_amplitude_to_level_db(np.sqrt(band_power))

    assert np.isclose(float(observed_level), expected_level_db, atol=1.0e-12)


def test_one_sided_power_sum_matches_time_domain_mean_square_for_odd_length() -> None:
    """Nyquist bin がない奇数 FFT でも Parseval の平均二乗値と一致することを確認する。"""
    rng = np.random.default_rng(20260711)
    signal = rng.standard_normal(255)
    spectrum = np.fft.rfft(signal)

    bin_power = one_sided_rfft_bin_rms_power(spectrum, sample_count=signal.size)

    assert np.isclose(float(np.sum(bin_power)), float(np.mean(signal**2)), rtol=1.0e-12)


def test_noise_asd_conversion_integrates_one_sided_bandwidth() -> None:
    """0 dB re RMS/sqrt(Hz) が sqrt(fs/2) の sample RMS になることを確認する。"""
    observed = noise_asd_level_db_to_sample_rms(0.0, sampling_frequency_hz=8000.0)

    assert np.isclose(observed, np.sqrt(4000.0))


def test_rms_amplitude_floor_is_explicit_display_contract() -> None:
    """0 amplitude が指定 floor へ写像され、未指定時だけ -inf になることを確認する。"""
    floored = rms_amplitude_to_level_db(np.array([0.0, 1.0]), floor_db=-80.0)
    unfloored = rms_amplitude_to_level_db(np.array([0.0, 1.0]))

    np.testing.assert_allclose(floored, np.array([-80.0, 0.0]))
    assert np.isneginf(unfloored[0])
    assert unfloored[1] == 0.0
