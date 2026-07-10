"""外部 scene_renderer 評価の RMS level 換算を確認する。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

# PNG 生成関数の検証では描画結果のファイル出力だけを確認する。
# CI や Codex 環境では Tcl/Tk が利用できないことがあるため、GUI backend を使わない。
matplotlib.use("Agg", force=True)

from examples.beamforming.evaluate_external_scene_renderer_fixed_delay_diff_mvdr import (
    ExternalLevelNormalizationCheck,
    _prepare_clean_tone_level_for_display,
    db20_noise_density_to_sample_rms_amplitude,
    db20_rms_to_tone_peak_amplitude,
    tone_rms_level_db_from_fft_bin,
    write_beam_pattern_definition_example_png,
    write_beam_response_definition_example_png,
    write_fixed_beamformed_spectrum_check_png,
    write_level_normalization_check_png,
    write_rendered_input_spectrum_check_png,
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
    n_fft = 32768
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


def test_prepare_clean_tone_level_for_display_hides_numeric_residue() -> None:
    """単一 tone 確認図で -150 dB 級の丸め残差を副信号として表示しない。

    scene_renderer の complex64 tone では、主 tone 以外に浮動小数点丸め由来の
    極小成分が等間隔 bin に現れる場合がある。SL 確認図は入力レベルの確認が目的なので、
    source peak から -120 dB 未満の成分は表示対象から外す。
    """
    frequency_hz = np.arange(0.0, 8193.0, 1.0, dtype=np.float64)
    clean_level_db = np.full(frequency_hz.shape, -3000.0, dtype=np.float64)
    source_bin = 1024
    clean_level_db[source_bin] = 0.0
    clean_level_db[3072] = -150.0
    clean_level_db[5120] = -155.0

    display_level_db, display_floor_db, max_non_source_level_db, false_peak_count = (
        _prepare_clean_tone_level_for_display(
            frequency_hz=frequency_hz,
            clean_level_db=clean_level_db,
            source_frequencies_hz=(float(source_bin),),
            source_levels_db20=(0.0,),
        )
    )

    assert display_floor_db == -120.0
    assert max_non_source_level_db == -150.0
    assert false_peak_count == 0
    assert np.isfinite(display_level_db[source_bin])
    assert bool(np.all(np.isnan(display_level_db[[3072, 5120]])))


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


def test_write_rendered_input_spectrum_check_png_creates_pre_beamforming_plot() -> None:
    """source+noise 合成後、整相前の周波数スペクトル PNG を保存する。

    clean/noise 分離図だけでは実際に scene_renderer へ入る合成波形が見えないため、
    source と channel 無相関雑音を足した後の spectrum を別 PNG として確認する。
    """
    fs_hz = 32768.0
    n_fft = 32768
    n_ch = 4
    tone_bin = 32
    source_level_db = 0.0
    noise_level_db = -32.0
    sample_index = np.arange(n_fft, dtype=np.float64)
    clean_mono = db20_rms_to_tone_peak_amplitude(source_level_db) * np.cos(
        2.0 * np.pi * tone_bin * sample_index / float(n_fft)
    )
    clean = np.tile(clean_mono[np.newaxis, :], (n_ch, 1))
    rng = np.random.default_rng(20260708)
    noise = db20_noise_density_to_sample_rms_amplitude(
        noise_level_db, fs_hz=fs_hz
    ) * rng.standard_normal((n_ch, n_fft))
    output_path = Path(
        "artifacts/beamforming/fixed_delay_diff_mvdr/level_normalization_test/"
        "external_rendered_input_spectrum_check.png"
    )

    write_rendered_input_spectrum_check_png(
        output_path=output_path,
        arrays={"rendered_signal": clean + noise},
        check=ExternalLevelNormalizationCheck(
            source_frequencies_hz=(float(tone_bin),),
            source_levels_db20=(source_level_db,),
            noise_level_db20=noise_level_db,
            fs_hz=fs_hz,
        ),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_write_fixed_beamformed_spectrum_check_png_creates_bl_fl_fraz_plot() -> None:
    """整相後の BL/FL/FRAZ を 1 枚の PNG として保存する。

    評価者が整相前の入力 spectrum と混同しないよう、整相後図には
    signal 周波数近傍の BL、signal 方位近傍ビームの FL、FRAZ を同時に表示する。
    """
    frequency_hz = np.arange(1.0, 129.0, dtype=np.float64)
    azimuth_deg = np.linspace(0.0, 180.0, 13, dtype=np.float64)
    source_frequency_hz = 32.0
    source_azimuth_deg = 60.0
    azimuth_width_deg = 18.0
    frequency_width_hz = 4.0
    # fraz_level_db shape: [n_beam, n_frequency]。axis=0 は方位、axis=1 は周波数を表す。
    fraz_level_db = -70.0 + 70.0 * np.exp(
        -(((azimuth_deg[:, np.newaxis] - source_azimuth_deg) / azimuth_width_deg) ** 2)
        - ((frequency_hz[np.newaxis, :] - source_frequency_hz) / frequency_width_hz) ** 2
    )
    output_path = Path(
        "artifacts/beamforming/fixed_delay_diff_mvdr/level_normalization_test/"
        "external_fixed_beamformed_spectrum_check.png"
    )

    write_fixed_beamformed_spectrum_check_png(
        output_path=output_path,
        arrays={
            "beamformed_fixed_frequency_hz": frequency_hz,
            "azimuth_deg": azimuth_deg,
            "beamformed_fixed_fraz_level_db": fraz_level_db,
        },
        check=ExternalLevelNormalizationCheck(
            source_frequencies_hz=(source_frequency_hz,),
            source_levels_db20=(0.0,),
            noise_level_db20=-32.0,
            fs_hz=32768.0,
            source_azimuths_deg=(source_azimuth_deg,),
        ),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_write_beam_response_and_pattern_definition_examples_create_pngs() -> None:
    """beam response と beam pattern を別定義の PNG として保存する。

    beam response は source 方位を固定して待ち受け beam 軸を走査する。
    beam pattern は待ち受け重みを固定して入力方位を細かく掃引するため、
    beam response と同じ x 軸定義にしてはいけない。
    """
    frequency_hz = np.arange(1.0, 129.0, dtype=np.float64)
    azimuth_deg = np.linspace(0.0, 180.0, 13, dtype=np.float64)
    source_frequency_hz = 32.0
    source_azimuth_deg = 60.0
    # fraz_level_db shape: [n_beam, n_frequency]。
    # beam response では、この中から source 周波数最近傍の beam 軸を取り出す。
    fraz_level_db = -70.0 + 70.0 * np.exp(
        -(((azimuth_deg[:, np.newaxis] - source_azimuth_deg) / 18.0) ** 2)
        - ((frequency_hz[np.newaxis, :] - source_frequency_hz) / 4.0) ** 2
    )
    pattern_input_azimuth_deg = np.linspace(0.0, 180.0, 721, dtype=np.float64)
    pattern_level_db = -50.0 + 50.0 * np.exp(
        -(((pattern_input_azimuth_deg - source_azimuth_deg) / 12.0) ** 2)
    )
    arrays = {
        "beamformed_fixed_frequency_hz": frequency_hz,
        "azimuth_deg": azimuth_deg,
        "beamformed_fixed_fraz_level_db": fraz_level_db,
        "beam_response_level_db": fraz_level_db[
            :, int(np.argmin(np.abs(frequency_hz - source_frequency_hz)))
        ],
        "beam_response_frequency_hz": np.asarray([source_frequency_hz], dtype=np.float64),
        "beam_pattern_input_azimuth_deg": pattern_input_azimuth_deg,
        "beam_pattern_level_db": pattern_level_db,
        "beam_pattern_steering_azimuth_deg": np.asarray([source_azimuth_deg], dtype=np.float64),
        "beam_pattern_source_frequency_hz": np.asarray([source_frequency_hz], dtype=np.float64),
        "beam_pattern_noise_floor_db": np.asarray([-32.0], dtype=np.float64),
    }
    check = ExternalLevelNormalizationCheck(
        source_frequencies_hz=(source_frequency_hz,),
        source_levels_db20=(0.0,),
        noise_level_db20=-32.0,
        fs_hz=32768.0,
        source_azimuths_deg=(source_azimuth_deg,),
    )
    response_path = Path(
        "artifacts/beamforming/fixed_delay_diff_mvdr/level_normalization_test/"
        "external_beam_response_definition_example.png"
    )
    pattern_path = Path(
        "artifacts/beamforming/fixed_delay_diff_mvdr/level_normalization_test/"
        "external_beam_pattern_definition_example.png"
    )

    write_beam_response_definition_example_png(
        output_path=response_path,
        arrays=arrays,
        check=check,
    )
    write_beam_pattern_definition_example_png(
        output_path=pattern_path,
        arrays=arrays,
        check=check,
    )

    assert response_path.exists()
    assert response_path.stat().st_size > 0
    assert pattern_path.exists()
    assert pattern_path.stat().st_size > 0


def test_write_level_normalization_check_png_creates_frequency_spectrum_plot() -> None:
    """SL/NL 確認方法を PNG として保存できることを確認する。

    source は整数 bin の 0 dB RMS tone、noise は `sqrt(fs/2)` 換算済み white noise にする。
    この条件なら、図上で tone peak と noise ASD の期待線が直接比較できる。
    """
    fs_hz = 32768.0
    n_fft = 32768
    n_ch = 4
    tone_bin = 32
    source_frequency_hz = float(tone_bin)
    source_level_db = 0.0
    noise_level_db = -32.0
    sample_index = np.arange(n_fft, dtype=np.float64)
    clean_mono = db20_rms_to_tone_peak_amplitude(source_level_db) * np.cos(
        2.0 * np.pi * tone_bin * sample_index / float(n_fft)
    )
    clean = np.tile(clean_mono[np.newaxis, :], (n_ch, 1))
    rng = np.random.default_rng(20260708)
    noise_sample_rms = db20_noise_density_to_sample_rms_amplitude(
        noise_level_db,
        fs_hz=fs_hz,
    )
    noise = noise_sample_rms * rng.standard_normal((n_ch, n_fft))
    arrays = {
        "clean_signal": clean,
        "noise_signal": noise,
    }
    output_path = Path(
        "artifacts/beamforming/fixed_delay_diff_mvdr/level_normalization_test/"
        "external_level_normalization_check.png"
    )

    write_level_normalization_check_png(
        output_path=output_path,
        arrays=arrays,
        check=ExternalLevelNormalizationCheck(
            source_frequencies_hz=(source_frequency_hz,),
            source_levels_db20=(source_level_db,),
            noise_level_db20=noise_level_db,
            fs_hz=fs_hz,
        ),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
