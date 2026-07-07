"""外部 scene_renderer 評価の RMS level 換算を確認する。"""

from __future__ import annotations

import numpy as np

from examples.beamforming.evaluate_external_scene_renderer_fixed_delay_diff_mvdr import (
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
