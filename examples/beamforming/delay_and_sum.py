"""平面波 tone の生成規約を delay-and-sum 前後で検証する最小例。"""

from __future__ import annotations

import numpy as np

from spflow import (
    integrate_one_sided_band_rms_power,
    one_sided_rfft_bin_rms_power,
    rms_amplitude_to_level_db,
    steering_from_relative_delay,
    synthesize_plane_wave_tone,
)


def wrap_phase_rad(phase_rad: np.ndarray) -> np.ndarray:
    """位相差を `[-π, π)` へ折り返す。

    Args:
        phase_rad: 位相。shape は任意、単位は rad。

    Returns:
        折り返した位相。shape は入力と同じ、単位は rad。
    """
    return (phase_rad + np.pi) % (2.0 * np.pi) - np.pi


def main() -> None:
    """生成toneのlevel・channel位相とdelay-and-sum後levelを確認する。

    入力と出力:
        8 channel 線状アレイへ単一平面波 tone を生成し、各 channel のFFT level、
        相対位相誤差、整相加算後のtone levelを標準出力へ表示する。

    単位と境界条件:
        位置はm、時間はs、周波数はHz、位相はrad、levelは`dB re input RMS`。
        FFT leakageを信号生成誤差と混同しないよう、toneは整数binに配置する。
        この例はBL、方式比較、parameter sweep、採否判定を扱わない。
    """
    sound_speed_m_per_s = 1500.0
    sampling_frequency_hz = 12000.0
    sample_count = 4096
    tone_bin_index = 512
    tone_frequency_hz = tone_bin_index * sampling_frequency_hz / sample_count
    tone_level_db_re_input_rms = -6.0
    source_azimuth_deg = 65.0
    channel_count = 8
    sensor_spacing_m = 0.25

    # sensor_positions_m shape: [n_channel, 3]。axis=0 は channel、axis=1 は x/y/z [m]。
    sensor_positions_m = np.zeros((channel_count, 3), dtype=np.float64)
    sensor_positions_m[:, 0] = np.arange(channel_count, dtype=np.float64) * sensor_spacing_m
    source_azimuth_rad = np.deg2rad(source_azimuth_deg)
    arrival_direction = np.array(
        [np.cos(source_azimuth_rad), np.sin(source_azimuth_rad), 0.0],
        dtype=np.float64,
    )
    generated = synthesize_plane_wave_tone(
        sensor_positions_m,
        arrival_direction,
        sound_speed_m_per_s=sound_speed_m_per_s,
        sampling_frequency_hz=sampling_frequency_hz,
        sample_count=sample_count,
        frequency_hz=tone_frequency_hz,
        level_db_re_rms=tone_level_db_re_input_rms,
    )

    # spectrum shape: [n_channel, n_frequency]。axis=0 は channel、axis=1 は rFFT bin。
    spectrum = np.fft.rfft(generated.signal, axis=1)
    bin_rms_power = one_sided_rfft_bin_rms_power(
        spectrum,
        sample_count=sample_count,
        frequency_axis=1,
    )
    tone_mask = np.zeros(spectrum.shape[1], dtype=np.bool_)
    tone_mask[tone_bin_index] = True
    channel_tone_power = integrate_one_sided_band_rms_power(
        bin_rms_power,
        tone_mask,
        frequency_axis=1,
    )
    channel_tone_level_db = rms_amplitude_to_level_db(np.sqrt(channel_tone_power))

    observed_phase_rad = np.angle(spectrum[:, tone_bin_index])
    expected_phase_rad = -2.0 * np.pi * tone_frequency_hz * generated.relative_delay_s
    phase_error_rad = wrap_phase_rad(observed_phase_rad - expected_phase_rad)

    # steering shape: [n_channel, n_frequency=1]。
    # w=a/N とし、w^H X により channel 軸を整相加算する。
    steering = steering_from_relative_delay(
        generated.relative_delay_s,
        np.array([tone_frequency_hz], dtype=np.float64),
    )[:, 0]
    weights = steering / float(channel_count)
    delay_and_sum_tone_bin = np.vdot(weights, spectrum[:, tone_bin_index])
    delay_and_sum_bin_power = 2.0 * np.abs(delay_and_sum_tone_bin / sample_count) ** 2
    delay_and_sum_level_db = float(rms_amplitude_to_level_db(np.sqrt(delay_and_sum_bin_power)))

    max_channel_level_error_db = float(
        np.max(np.abs(channel_tone_level_db - tone_level_db_re_input_rms))
    )
    max_channel_phase_error_rad = float(np.max(np.abs(phase_error_rad)))
    print(f"expected_channel_level_db={tone_level_db_re_input_rms:.12f}")
    print(f"max_channel_level_error_db={max_channel_level_error_db:.3e}")
    print(f"max_channel_phase_error_rad={max_channel_phase_error_rad:.3e}")
    print(f"delay_and_sum_level_db={delay_and_sum_level_db:.12f}")


if __name__ == "__main__":
    main()
