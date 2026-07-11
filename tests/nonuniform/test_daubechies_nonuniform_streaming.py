"""daubechies nonuniform streaming に関する回帰試験。"""

# 非均一分割は葉ごとのレート差と内部状態の持ち越しが複雑なため、
# 木構造・streaming・ビームフォーミング統合時の安全側挙動を回帰試験で固定する。

import numpy as np

from spflow.filterbank.daubechies_nonuniform_beamformer import DaubechiesNonuniformBeamformer
from spflow.filterbank.daubechies_nonuniform_streaming import DaubechiesNonuniformBeamformerStreaming


def _chunk_signal(x: np.ndarray, chunk_sizes: list[int]) -> list[np.ndarray]:
    """`_chunk_signal` を実行する。"""
    chunks = []
    start = 0
    for size in chunk_sizes:
        stop = min(start + size, x.shape[-1])
        if stop > start:
            chunks.append(x[..., start:stop])
        start = stop
    if start < x.shape[-1]:
        chunks.append(x[..., start:])
    return chunks


def _steering_for_angle(
    beamformer: DaubechiesNonuniformBeamformer,
    angle_deg: float,
    sound_speed: float = 340.0,
) -> np.ndarray:
    """`_steering_for_angle` を実行する。"""
    positions_m = beamformer.array_design.channel_positions_m
    theta = np.deg2rad(angle_deg)
    steering = np.zeros((beamformer.array_design.n_ch, 1, len(beamformer.band_specs)), dtype=np.complex64)
    for band_idx, spec in enumerate(beamformer.band_specs):
        delay_s = positions_m * np.sin(theta) / sound_speed
        steering[:, 0, band_idx] = np.exp(-1j * 2.0 * np.pi * spec.center_frequency_hz * delay_s)
    return steering


def _make_single_band_interferer_scene(
    beamformer: DaubechiesNonuniformBeamformer,
    *,
    band_idx: int,
    n_sample: int,
    target_angle_deg: float,
    interferer_angle_deg: float,
    interferer_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """`_make_single_band_interferer_scene` を実行する。"""
    fs = beamformer.fs_hz
    n = np.arange(n_sample, dtype=np.float32)
    t = n / fs
    spec = beamformer.band_specs[band_idx]
    target_steering = _steering_for_angle(beamformer, target_angle_deg)
    interferer_steering = _steering_for_angle(beamformer, interferer_angle_deg)

    rng = np.random.default_rng(710)
    block_size = 32
    n_block = int(np.ceil(n_sample / block_size))
    target_envelope = np.repeat(
        (rng.standard_normal(n_block) + 1j * rng.standard_normal(n_block)) / np.sqrt(2.0),
        block_size,
    )[:n_sample]
    interferer_envelope = np.repeat(
        (rng.standard_normal(n_block) + 1j * rng.standard_normal(n_block)) / np.sqrt(2.0),
        block_size,
    )[:n_sample]
    smooth = np.ones(9, dtype=np.float32) / 9.0
    target_envelope = np.convolve(target_envelope, smooth, mode="same")
    interferer_envelope = np.convolve(interferer_envelope, smooth, mode="same")

    frequency_hz = spec.center_frequency_hz
    target_source = target_envelope * np.exp(1j * 2.0 * np.pi * frequency_hz * t)
    interferer_source = interferer_gain * interferer_envelope * np.exp(1j * 2.0 * np.pi * (frequency_hz + 4.0) * t)

    multichannel = np.zeros((beamformer.array_design.n_ch, n_sample), dtype=np.complex64)
    multichannel += target_steering[:, :, band_idx] @ target_source[np.newaxis, :]
    multichannel += interferer_steering[:, :, band_idx] @ interferer_source[np.newaxis, :]
    return multichannel, target_source


def _make_narrowband_tone_scene(
    beamformer: DaubechiesNonuniformBeamformer,
    *,
    frequency_hz: float,
    source_angle_deg: float,
    n_sample: int,
) -> np.ndarray:
    """`_make_narrowband_tone_scene` を実行する。"""
    positions_m = beamformer.array_design.channel_positions_m
    theta = np.deg2rad(source_angle_deg)
    delay_s = positions_m * np.sin(theta) / 340.0
    steering = np.exp(-1j * 2.0 * np.pi * frequency_hz * delay_s)
    n = np.arange(n_sample, dtype=np.float32)
    tone = np.exp(1j * 2.0 * np.pi * frequency_hz * n / beamformer.fs_hz)
    return steering[:, np.newaxis] @ tone[np.newaxis, :]


def _fft_bin_magnitude(y: np.ndarray, frequency_hz: float, fs_hz: float) -> float:
    """`_fft_bin_magnitude` を実行する。"""
    n_sample = int(y.shape[-1])
    fft_bin = int(round(frequency_hz * n_sample / fs_hz))
    return float(np.abs(np.fft.fft(y)[fft_bin]) / n_sample)


def _stream_output(
    streaming: DaubechiesNonuniformBeamformerStreaming,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`_stream_output` を実行する。"""
    emitted = []
    boundary_indices = []
    produced = 0
    for chunk in _chunk_signal(x, [17, 131, 509, 23, 2047, 61, 997, 37, 4093, 113]):
        y_chunk = streaming.process_analytic(chunk)
        if y_chunk.shape[-1] > 0:
            emitted.append(y_chunk)
            produced += y_chunk.shape[-1]
            boundary_indices.append(produced - 1)
    y_tail = streaming.flush()
    if y_tail.shape[-1] > 0:
        emitted.append(y_tail)
        produced += y_tail.shape[-1]
        boundary_indices.append(produced - 1)

    reconstructed = np.concatenate(emitted, axis=-1)
    boundary_indices = np.asarray(boundary_indices[:-1], dtype=np.int64)
    boundary_indices = boundary_indices[
        (boundary_indices >= 0) & (boundary_indices < reconstructed.shape[-1] - 1)
    ]
    return reconstructed, boundary_indices


def test_daubechies_nonuniform_streaming_matches_offline_and_preserves_continuity():
    """Daubechies 非均一 streaming 経路がoffline 実装と一致し、境界連続性も保つことを確認する。"""
    beamformer = DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8")
    streaming = DaubechiesNonuniformBeamformerStreaming(beamformer=beamformer)

    rng = np.random.default_rng(701)
    x = rng.standard_normal(65536) + 1j * rng.standard_normal(65536)
    multichannel = np.repeat(x[np.newaxis, :], beamformer.array_design.n_ch, axis=0)

    offline = beamformer.beamform_analytic(multichannel)
    reconstructed, boundary_indices = _stream_output(streaming, multichannel)

    np.testing.assert_allclose(reconstructed, offline, atol=3e-5)

    jump_abs_error = np.abs(np.diff(reconstructed[0] - offline[0]))
    assert float(np.max(jump_abs_error)) <= 2e-5
    if boundary_indices.size > 0:
        assert float(np.max(jump_abs_error[boundary_indices])) <= 2e-5


def test_daubechies_nonuniform_mvdr_streaming_matches_offline_and_preserves_continuity():
    """Daubechies 非均一 MVDR streaming 経路がoffline 実装と一致し、境界連続性も保つことを確認する。"""
    beamformer = DaubechiesNonuniformBeamformer(
        candidate_name="daubechies_qmf_order4_taps8",
        beamformer_mode="mvdr",
        integration_time=0.0,
        weight_update_period=0.0,
        diag_load=1e-3,
    )
    streaming = DaubechiesNonuniformBeamformerStreaming(beamformer=beamformer)

    rng = np.random.default_rng(706)
    x = rng.standard_normal(65536) + 1j * rng.standard_normal(65536)
    multichannel = np.repeat(x[np.newaxis, :], beamformer.array_design.n_ch, axis=0)

    offline = beamformer.beamform_analytic(multichannel)
    reconstructed, boundary_indices = _stream_output(streaming, multichannel)

    assert np.all(np.isfinite(reconstructed))
    np.testing.assert_allclose(reconstructed, offline, atol=3e-5)

    jump_abs_error = np.abs(np.diff(reconstructed[0] - offline[0]))
    assert float(np.max(jump_abs_error)) <= 2e-5
    if boundary_indices.size > 0:
        assert float(np.max(jump_abs_error[boundary_indices])) <= 2e-5


def test_daubechies_nonuniform_phase_b_streaming_matches_offline():
    """Daubechies 非均一 Phase B 経路について streaming 実装が offline 実装と一致する を確認する。"""
    beamformer = DaubechiesNonuniformBeamformer(
        candidate_name="daubechies_qmf_order4_taps8",
        output_path_mode="leaf_independent_one_sided",
    )
    streaming = DaubechiesNonuniformBeamformerStreaming(beamformer=beamformer)

    rng = np.random.default_rng(708)
    x = rng.standard_normal(65536) + 1j * rng.standard_normal(65536)
    multichannel = np.repeat(x[np.newaxis, :], beamformer.array_design.n_ch, axis=0)

    offline = beamformer.beamform_analytic(multichannel)
    reconstructed, boundary_indices = _stream_output(streaming, multichannel)

    assert np.all(np.isfinite(reconstructed))
    np.testing.assert_allclose(reconstructed, offline, atol=3e-5)

    jump_abs_error = np.abs(np.diff(reconstructed[0] - offline[0]))
    assert float(np.max(jump_abs_error)) <= 2e-5
    if boundary_indices.size > 0:
        assert float(np.max(jump_abs_error[boundary_indices])) <= 2e-5


def test_daubechies_nonuniform_phase_b_reconstructed_fft_scan_peaks_at_steering_angle():
    """Daubechies 非均一 Phase B 経路について 再構成後 FFT scan のピークが steering 角に立つ を確認する。"""
    steer_angle_deg = 10.0
    frequency_hz = 1536.0
    scan_angles_deg = np.arange(-30.0, 31.0, 5.0)
    steering = _steering_for_angle(
        DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8"),
        angle_deg=steer_angle_deg,
    )
    beamformer = DaubechiesNonuniformBeamformer(
        candidate_name="daubechies_qmf_order4_taps8",
        steering=steering,
        output_path_mode="leaf_independent_one_sided",
    )

    magnitudes = []
    for source_angle_deg in scan_angles_deg:
        multichannel = _make_narrowband_tone_scene(
            beamformer,
            frequency_hz=frequency_hz,
            source_angle_deg=float(source_angle_deg),
            n_sample=16384,
        )
        reconstructed = beamformer.beamform_analytic(multichannel)[0]
        magnitudes.append(_fft_bin_magnitude(reconstructed, frequency_hz, beamformer.fs_hz))

    peak_angle_deg = float(scan_angles_deg[int(np.argmax(magnitudes))])
    assert peak_angle_deg == steer_angle_deg
    assert max(magnitudes) >= min(magnitudes) + 0.25


def test_daubechies_nonuniform_phase_b_streaming_reconstructed_waveform_has_no_boundary_jump():
    """Daubechies 非均一 Phase B 経路について streaming 再構成波形に境界ジャンプがない を確認する。"""
    steer_angle_deg = 10.0
    frequency_hz = 1536.0
    steering = _steering_for_angle(
        DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8"),
        angle_deg=steer_angle_deg,
    )
    beamformer = DaubechiesNonuniformBeamformer(
        candidate_name="daubechies_qmf_order4_taps8",
        steering=steering,
        output_path_mode="leaf_independent_one_sided",
    )
    streaming = DaubechiesNonuniformBeamformerStreaming(beamformer=beamformer)

    multichannel = _make_narrowband_tone_scene(
        beamformer,
        frequency_hz=frequency_hz,
        source_angle_deg=steer_angle_deg,
        n_sample=16384,
    )
    reconstructed, boundary_indices = _stream_output(streaming, multichannel)
    reconstructed = reconstructed[0, :16384]

    assert boundary_indices.size > 0
    jump_abs = np.abs(np.diff(reconstructed))
    boundary_jump_abs = jump_abs[boundary_indices]
    median_jump_abs = float(np.median(jump_abs))

    assert np.all(np.isfinite(reconstructed))
    assert float(np.max(boundary_jump_abs)) <= 1.05 * median_jump_abs


def test_daubechies_nonuniform_phase_b_mvdr_reconstructed_fft_scan_peaks_at_steering_angle_across_frequencies():
    """Daubechies 非均一 Phase B 経路について mvdr 再構成後 FFT scan のピークが steering 角に立つ across frequencies を確認する。"""
    steer_angle_deg = 10.0
    scan_angles_deg = np.arange(-30.0, 31.0, 10.0)
    test_frequencies_hz = [192.0, 1536.0, 12288.0]
    steering = _steering_for_angle(
        DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8"),
        angle_deg=steer_angle_deg,
    )

    for frequency_hz in test_frequencies_hz:
        beamformer = DaubechiesNonuniformBeamformer(
            candidate_name="daubechies_qmf_order4_taps8",
            steering=steering,
            beamformer_mode="mvdr",
            integration_time=0.0,
            weight_update_period=0.0,
            diag_load=1e-3,
            output_path_mode="leaf_independent_one_sided",
        )

        magnitudes = []
        for source_angle_deg in scan_angles_deg:
            multichannel = _make_narrowband_tone_scene(
                beamformer,
                frequency_hz=frequency_hz,
                source_angle_deg=float(source_angle_deg),
                n_sample=16384,
            )
            reconstructed = beamformer.beamform_analytic(multichannel)[0]
            magnitudes.append(_fft_bin_magnitude(reconstructed, frequency_hz, beamformer.fs_hz))

        magnitudes_array = np.asarray(magnitudes, dtype=np.float32)
        peak_angle_deg = float(scan_angles_deg[int(np.argmax(magnitudes_array))])
        sorted_magnitudes = np.sort(magnitudes_array)

        assert abs(peak_angle_deg - steer_angle_deg) <= 10.0
        assert float(sorted_magnitudes[-1] - sorted_magnitudes[-2]) >= 1e-2


def test_daubechies_nonuniform_phase_b_mvdr_streaming_waveform_stays_continuous_across_frequency_and_angle():
    """Daubechies 非均一 Phase B 経路について MVDR streaming 波形が周波数・角度をまたいでも連続である を確認する。"""
    steer_angle_deg = 10.0
    test_cases = [
        (192.0, 0.0),
        (1536.0, 10.0),
        (12288.0, 20.0),
    ]
    steering = _steering_for_angle(
        DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8"),
        angle_deg=steer_angle_deg,
    )

    for frequency_hz, source_angle_deg in test_cases:
        beamformer = DaubechiesNonuniformBeamformer(
            candidate_name="daubechies_qmf_order4_taps8",
            steering=steering,
            beamformer_mode="mvdr",
            integration_time=0.0,
            weight_update_period=0.0,
            diag_load=1e-3,
            output_path_mode="leaf_independent_one_sided",
        )
        streaming = DaubechiesNonuniformBeamformerStreaming(beamformer=beamformer)

        multichannel = _make_narrowband_tone_scene(
            beamformer,
            frequency_hz=frequency_hz,
            source_angle_deg=source_angle_deg,
            n_sample=16384,
        )
        reconstructed, boundary_indices = _stream_output(streaming, multichannel)
        reconstructed = reconstructed[0, :16384]

        assert boundary_indices.size > 0
        assert np.all(np.isfinite(reconstructed))

        jump_abs = np.abs(np.diff(reconstructed))
        boundary_jump_abs = jump_abs[boundary_indices]
        jump_p95 = float(np.percentile(jump_abs, 95.0))

        assert float(np.max(boundary_jump_abs)) <= jump_p95 + 2e-5


def test_daubechies_nonuniform_phase_b_mvdr_improves_over_cbf_across_frequency_and_angle_cases():
    """Daubechies 非均一 Phase B 経路について MVDR が周波数・角度ケース全体で CBF より改善する を確認する。"""
    interferer_cases = [
        (1, 0.0, -20.0),
        (4, 10.0, -25.0),
        (7, 20.0, -10.0),
    ]

    for band_idx, target_angle_deg, interferer_angle_deg in interferer_cases:
        steering = _steering_for_angle(
            DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8"),
            angle_deg=target_angle_deg,
        )
        cbf = DaubechiesNonuniformBeamformer(
            candidate_name="daubechies_qmf_order4_taps8",
            steering=steering,
            beamformer_mode="cbf",
            output_path_mode="leaf_independent_one_sided",
        )
        mvdr = DaubechiesNonuniformBeamformer(
            candidate_name="daubechies_qmf_order4_taps8",
            steering=steering,
            beamformer_mode="mvdr",
            integration_time=1.0,
            weight_update_period=0.0,
            diag_load=1e-3,
            output_path_mode="leaf_independent_one_sided",
        )

        multichannel, target_reference = _make_single_band_interferer_scene(
            mvdr,
            band_idx=band_idx,
            n_sample=32768,
            target_angle_deg=target_angle_deg,
            interferer_angle_deg=interferer_angle_deg,
            interferer_gain=1.5,
        )

        cbf_output = cbf.beamform_analytic(multichannel)[0]
        mvdr_output = mvdr.beamform_analytic(multichannel)[0]

        start = target_reference.shape[-1] // 2
        cbf_error = float(np.sqrt(np.mean(np.abs(cbf_output[start:] - target_reference[start:]) ** 2)))
        mvdr_error = float(np.sqrt(np.mean(np.abs(mvdr_output[start:] - target_reference[start:]) ** 2)))

        assert mvdr_error < cbf_error


def test_daubechies_nonuniform_mvdr_streaming_matches_offline_under_interferer_condition():
    """Daubechies 非均一 MVDR streaming 経路が妨害波あり条件でも offline 実装と一致することを確認する。"""
    steering = _steering_for_angle(
        DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8"),
        angle_deg=10.0,
    )
    cbf = DaubechiesNonuniformBeamformer(
        candidate_name="daubechies_qmf_order4_taps8",
        steering=steering,
        beamformer_mode="cbf",
    )
    mvdr = DaubechiesNonuniformBeamformer(
        candidate_name="daubechies_qmf_order4_taps8",
        steering=steering,
        beamformer_mode="mvdr",
        integration_time=1.0,
        weight_update_period=0.0,
        diag_load=1e-3,
    )
    streaming = DaubechiesNonuniformBeamformerStreaming(
        beamformer=DaubechiesNonuniformBeamformer(
            candidate_name="daubechies_qmf_order4_taps8",
            steering=steering,
            beamformer_mode="mvdr",
            integration_time=1.0,
            weight_update_period=0.0,
            diag_load=1e-3,
        )
    )

    multichannel, target_reference = _make_single_band_interferer_scene(
        mvdr,
        band_idx=4,
        n_sample=40960,
        target_angle_deg=10.0,
        interferer_angle_deg=-25.0,
        interferer_gain=1.5,
    )

    cbf_output = cbf.beamform_analytic(multichannel)[0]
    offline = mvdr.beamform_analytic(multichannel)
    reconstructed, boundary_indices = _stream_output(streaming, multichannel)

    start = target_reference.shape[-1] // 2
    cbf_error = float(np.sqrt(np.mean(np.abs(cbf_output[start:] - target_reference[start:]) ** 2)))
    mvdr_error = float(np.sqrt(np.mean(np.abs(offline[0, start:] - target_reference[start:]) ** 2)))

    assert mvdr_error < cbf_error
    # offline と streaming は complex64 の積和順序が異なるため、長時間の干渉条件では丸め誤差が累積する。
    # 最大観測差 3.14e-5 を包含する 4e-5 を全体一致限界とし、ブロック境界の不連続は下で別に拘束する。
    np.testing.assert_allclose(reconstructed, offline, atol=4e-5)

    jump_abs_error = np.abs(np.diff(reconstructed[0] - offline[0]))
    # 隣接差は両サンプルの complex64 丸め差を含むため、観測最大値 2.30e-5 を包含する 2.5e-5 とする。
    # chunk 境界だけは状態引き継ぎの不連続を検出する目的で、従来の 2e-5 を維持する。
    assert float(np.max(jump_abs_error)) <= 2.5e-5
    if boundary_indices.size > 0:
        assert float(np.max(jump_abs_error[boundary_indices])) <= 2e-5
