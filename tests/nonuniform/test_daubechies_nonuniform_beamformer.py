"""daubechies nonuniform beamformer に関する回帰試験。"""

import numpy as np

from spflow.filterbank.daubechies_nonuniform_beamformer import (
    DaubechiesNonuniformBeamformer,
    make_reference_dense_sparse_array_design,
)


def _direction_from_azimuth_deg(angle_deg: float) -> np.ndarray:
    """方位角から水平面内の方向余弦を作る。"""
    theta = np.deg2rad(angle_deg)
    return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)


def _positions_to_side_array_3d(beamformer: DaubechiesNonuniformBeamformer) -> np.ndarray:
    """1 次元直線アレイを y 軸上の 3 次元座標へ展開する。"""
    return beamformer.array_design.positions_3d(axis=1)


def _build_single_beam_frequency_dependent_steering(
    beamformer: DaubechiesNonuniformBeamformer,
    *,
    source_angle_deg: float,
    sound_speed: float,
) -> dict[str, np.ndarray]:
    """target 方位 1 本に一致した帯域別 steering 辞書を作る。"""
    positions_3d = _positions_to_side_array_3d(beamformer)
    direction = _direction_from_azimuth_deg(source_angle_deg)
    delays_s = positions_3d @ direction / sound_speed

    steering: dict[str, np.ndarray] = {}
    for band_index, spec in enumerate(beamformer.band_specs):
        valid_size = int(round(spec.nominal_sample_rate_hz / spec.target_resolution_hz))
        positive_bin_count = valid_size // 2 + 1
        frequencies_hz = spec.f_low_hz + np.arange(positive_bin_count, dtype=np.float32) * (
            spec.nominal_sample_rate_hz / valid_size
        )
        phase = -1j * 2.0 * np.pi * delays_s[:, np.newaxis, np.newaxis] * frequencies_hz[np.newaxis, np.newaxis, :]
        band_steering = np.exp(phase).astype(np.complex64)
        band_steering *= beamformer.array_design.shading_table[:, band_index][:, np.newaxis, np.newaxis]
        steering[spec.band_id] = band_steering
    return steering


def _make_db20_tone_scene(
    beamformer: DaubechiesNonuniformBeamformer,
    *,
    frequency_hz: float,
    source_angle_deg: float,
    level_db20: float,
    sound_speed: float,
    n_sample: int,
) -> np.ndarray:
    """指定方位から到来する dB20 基準の多チャネルトーンを作る。"""
    positions_3d = _positions_to_side_array_3d(beamformer)
    direction = _direction_from_azimuth_deg(source_angle_deg)
    delays_s = positions_3d @ direction / sound_speed
    peak_amplitude = float(np.sqrt(2.0) * 10.0 ** (level_db20 / 20.0))
    time_axis = np.arange(n_sample, dtype=np.float32) / np.float32(beamformer.fs_hz)

    multichannel = np.zeros((beamformer.array_design.n_ch, n_sample), dtype=np.float32)
    for channel_index in range(beamformer.array_design.n_ch):
        multichannel[channel_index] = peak_amplitude * np.cos(
            2.0 * np.pi * frequency_hz * (time_axis - delays_s[channel_index])
        )
    return multichannel


def _tone_level_db20_rms(signal: np.ndarray, *, frequency_hz: float, fs_hz: float) -> float:
    """実波形の指定周波数成分を dB20 RMS で評価する。"""
    time_axis = np.arange(signal.shape[-1], dtype=np.float32) / np.float32(fs_hz)
    reference = np.exp(-1j * 2.0 * np.pi * frequency_hz * time_axis).astype(np.complex64)
    coefficient = np.vdot(reference, np.asarray(signal, dtype=np.complex64)) / signal.shape[-1]
    peak_amplitude = 2.0 * np.abs(coefficient)
    rms_amplitude = peak_amplitude / np.sqrt(2.0)
    return float(20.0 * np.log10(max(rms_amplitude, np.finfo(np.float32).tiny)))


def test_reference_dense_sparse_array_design_matches_documented_active_channel_counts_per_band():
    """基準の中央密・端疎アレイ設計について 文書化された帯域別 active channel 数と一致する を確認する。"""
    design = make_reference_dense_sparse_array_design()

    assert design.n_ch == 32
    assert design.active_channel_counts_per_band().tolist() == [32, 32, 24, 20, 16, 12, 8, 4]


def test_daubechies_nonuniform_beamformer_reconstructs_identical_broadside_channels():
    """Daubechies 非均一ビームフォーマが同一 broadside 入力を正しく再構成することを確認する。"""
    beamformer = DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8")

    rng = np.random.default_rng(700)
    x = rng.standard_normal(32768) + 1j * rng.standard_normal(32768)
    multichannel = np.repeat(x[np.newaxis, :], beamformer.array_design.n_ch, axis=0)

    y = beamformer.beamform_analytic(multichannel)

    assert y.shape == (1, x.shape[-1])
    np.testing.assert_allclose(y[0], x, atol=1e-5)


def test_daubechies_nonuniform_mvdr_beamformer_reconstructs_identical_broadside_channels():
    """Daubechies 非均一 MVDR ビームフォーマが同一 broadside 入力を正しく再構成することを確認する。"""
    beamformer = DaubechiesNonuniformBeamformer(
        candidate_name="daubechies_qmf_order4_taps8",
        beamformer_mode="mvdr",
        integration_time=0.0,
        weight_update_period=0.0,
        diag_load=1e-3,
    )

    rng = np.random.default_rng(704)
    x = rng.standard_normal(32768) + 1j * rng.standard_normal(32768)
    multichannel = np.repeat(x[np.newaxis, :], beamformer.array_design.n_ch, axis=0)

    y = beamformer.beamform_analytic(multichannel)

    assert y.shape == (1, x.shape[-1])
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y[0], x, atol=1e-5)


def test_daubechies_nonuniform_cbf_formal_one_sided_ols_recovers_target_matched_zero_db20_level():
    """Daubechies 非均一 CBF について正式 one-side OLS 出力が target 一致時に 0 dB20 をほぼ保つことを確認する。"""
    source_angle_deg = 20.0
    frequency_hz = 1536.0
    sound_speed = 1500.0

    base = DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8")
    steering = _build_single_beam_frequency_dependent_steering(
        base,
        source_angle_deg=source_angle_deg,
        sound_speed=sound_speed,
    )
    beamformer = DaubechiesNonuniformBeamformer(
        candidate_name="daubechies_qmf_order4_taps8",
        steering=steering,
        beamformer_mode="cbf",
        output_path_mode="leaf_independent_one_sided",
    )
    x = _make_db20_tone_scene(
        beamformer,
        frequency_hz=frequency_hz,
        source_angle_deg=source_angle_deg,
        level_db20=0.0,
        sound_speed=sound_speed,
        n_sample=16384,
    )

    y = beamformer.beamform_real(x)[0]
    output_level_db20 = _tone_level_db20_rms(y, frequency_hz=frequency_hz, fs_hz=beamformer.fs_hz)

    assert abs(output_level_db20) <= 0.1


def test_daubechies_nonuniform_mvdr_formal_one_sided_ols_recovers_target_matched_zero_db20_level():
    """Daubechies 非均一 MVDR について正式 one-side OLS 出力が target 一致時に 0 dB20 近傍へ戻ることを確認する。"""
    source_angle_deg = 20.0
    frequency_hz = 1536.0
    sound_speed = 1500.0

    base = DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8")
    steering = _build_single_beam_frequency_dependent_steering(
        base,
        source_angle_deg=source_angle_deg,
        sound_speed=sound_speed,
    )
    beamformer = DaubechiesNonuniformBeamformer(
        candidate_name="daubechies_qmf_order4_taps8",
        steering=steering,
        beamformer_mode="mvdr",
        output_path_mode="leaf_independent_one_sided",
        integration_time=0.25,
        weight_update_period=0.05,
        diag_load=1e-3,
    )
    x = _make_db20_tone_scene(
        beamformer,
        frequency_hz=frequency_hz,
        source_angle_deg=source_angle_deg,
        level_db20=0.0,
        sound_speed=sound_speed,
        n_sample=16384,
    )

    y = beamformer.beamform_real(x)[0]
    output_level_db20 = _tone_level_db20_rms(y, frequency_hz=frequency_hz, fs_hz=beamformer.fs_hz)

    assert abs(output_level_db20) <= 2.0


def test_daubechies_nonuniform_beamformer_preserves_formal_leaf_metadata():
    """Daubechies 非均一ビームフォーマがformal leaf metadata を保持することを確認する。"""
    beamformer = DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8")

    rng = np.random.default_rng(702)
    x = rng.standard_normal((beamformer.array_design.n_ch, 4096)) + 1j * rng.standard_normal((beamformer.array_design.n_ch, 4096))

    analyzed = beamformer.analyze_analytic(x)
    beamformed = beamformer.beamform_analysis_result(analyzed)

    for analyzed_packet, beamformed_packet in zip(analyzed.packets, beamformed.packets, strict=True):
        assert beamformed_packet.band_id == analyzed_packet.band_id
        assert beamformed_packet.time_origin_at_root_rate == analyzed_packet.time_origin_at_root_rate
        assert beamformed_packet.delay_samples_at_root_rate == analyzed_packet.delay_samples_at_root_rate
        assert beamformed_packet.complex_samples.shape[0] == 1
        assert beamformed_packet.complex_samples.shape[-1] == analyzed_packet.complex_samples.shape[-1]


def test_daubechies_nonuniform_mvdr_beamformer_preserves_formal_leaf_metadata():
    """Daubechies 非均一 MVDR ビームフォーマがformal leaf metadata を保持することを確認する。"""
    beamformer = DaubechiesNonuniformBeamformer(
        candidate_name="daubechies_qmf_order4_taps8",
        beamformer_mode="mvdr",
        integration_time=0.0,
        weight_update_period=0.0,
        diag_load=1e-3,
    )

    rng = np.random.default_rng(705)
    x = rng.standard_normal((beamformer.array_design.n_ch, 4096)) + 1j * rng.standard_normal((beamformer.array_design.n_ch, 4096))

    analyzed = beamformer.analyze_analytic(x)
    beamformed = beamformer.beamform_analysis_result(analyzed)

    for analyzed_packet, beamformed_packet in zip(analyzed.packets, beamformed.packets, strict=True):
        assert beamformed_packet.band_id == analyzed_packet.band_id
        assert beamformed_packet.time_origin_at_root_rate == analyzed_packet.time_origin_at_root_rate
        assert beamformed_packet.delay_samples_at_root_rate == analyzed_packet.delay_samples_at_root_rate
        assert beamformed_packet.complex_samples.shape[0] == 1
        assert beamformed_packet.complex_samples.shape[-1] == analyzed_packet.complex_samples.shape[-1]
        assert np.all(np.isfinite(beamformed_packet.complex_samples))


def test_daubechies_nonuniform_beamformer_real_path_matches_identical_broadside_input():
    """Daubechies 非均一ビームフォーマがreal path が同一 broadside 入力と一致することを確認する。"""
    beamformer = DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8")

    rng = np.random.default_rng(703)
    x = rng.standard_normal(4096)
    y = beamformer.beamform_real(x)

    assert y.shape == (1, x.shape[-1])
    np.testing.assert_allclose(y[0], x, atol=1e-5)

