"""外部 scene_renderer 評価の RMS level 換算を確認する。"""

from __future__ import annotations

import numpy as np

from examples.beamforming.evaluate_external_scene_renderer_fixed_delay_diff_mvdr import (
    db20_noise_density_to_sample_rms_amplitude,
    db20_rms_to_tone_peak_amplitude,
    tone_rms_level_db_from_fft_bin,
)


def test_db20_rms_to_tone_peak_amplitude_uses_sqrt2() -> None:
    """SL の RMS 指定を scene_renderer の正弦波ピーク振幅へ変換する。"""
    np.testing.assert_allclose(db20_rms_to_tone_peak_amplitude(0.0), np.sqrt(2.0))
    np.testing.assert_allclose(
        db20_rms_to_tone_peak_amplitude(-6.0),
        np.sqrt(2.0) * (10.0 ** (-6.0 / 20.0)),
    )


def test_tone_rms_level_db_from_fft_bin_matches_requested_rms_level() -> None:
    """FFT bin 値から `10*log10(2*(abs(result/N_FFT)**2))` で RMS level を戻す。

    非 DC の整数 bin に tone を置くことで、窓漏れを含まない基準条件にする。
    `A_peak=sqrt(2)*A_rms` の実正弦波では、正周波数 rfft bin の振幅が
    `N_FFT*A_peak/2` になるため、指定 RMS level と一致しなければならない。
    """
    n_fft = 1024
    tone_bin = 7
    requested_level_db = -3.0
    peak_amplitude = db20_rms_to_tone_peak_amplitude(requested_level_db)
    sample_index = np.arange(n_fft, dtype=np.float64)
    waveform = peak_amplitude * np.cos(2.0 * np.pi * tone_bin * sample_index / float(n_fft))

    spectrum = np.fft.rfft(waveform)
    observed_level_db = tone_rms_level_db_from_fft_bin(
        np.asarray([spectrum[tone_bin]], dtype=np.complex128),
        n_fft=n_fft,
    )

    np.testing.assert_allclose(observed_level_db, np.asarray([requested_level_db]), atol=1.0e-12)


def test_noise_level_db20_converts_to_white_noise_sample_rms() -> None:
    """NL を `10^(NL/20)*sqrt(fs/2)` で sample RMS へ変換する。"""
    fs_hz = 32768.0
    noise_level_db = -32.0
    expected = (10.0 ** (noise_level_db / 20.0)) * np.sqrt(fs_hz / 2.0)

    observed = db20_noise_density_to_sample_rms_amplitude(noise_level_db, fs_hz=fs_hz)

    np.testing.assert_allclose(observed, expected, atol=1.0e-12)


def test_white_noise_frequency_spectrum_matches_requested_noise_level() -> None:
    """白色雑音の片側周波数スペクトル平均が指定 NL に一致することを確認する。

    `Amp_NL = 10^(NL/20)*sqrt(fs/2)` を時間波形の標準偏差として与えると、
    非 DC / 非 Nyquist の rfft bin では
    `E[2*abs(X)^2/(N_FFT*fs)] = 10^(NL/10)` になる。
    ここでは多数 channel の bin power を平均し、周波数スペクトル上の NL を確認する。
    """
    fs_hz = 32768.0
    n_fft = 4096
    n_ch = 512
    noise_level_db = -32.0
    rng = np.random.default_rng(20260708)
    noise_sample_rms = db20_noise_density_to_sample_rms_amplitude(
        noise_level_db,
        fs_hz=fs_hz,
    )
    noise = noise_sample_rms * rng.standard_normal((n_ch, n_fft))

    spectrum = np.fft.rfft(noise, axis=1)
    # spectrum[:, 1:-1] shape: [n_ch, n_positive_bin_without_dc_nyquist]。
    # 実数 white noise の片側 ASD power は 2*|X|^2/(N_FFT*fs) で推定する。
    one_sided_density_power = 2.0 * (np.abs(spectrum[:, 1:-1]) ** 2) / (float(n_fft) * fs_hz)
    observed_level_db = 10.0 * np.log10(float(np.mean(one_sided_density_power)))

    assert abs(observed_level_db - noise_level_db) < 0.1
