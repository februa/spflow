"""mvdr streaming pipeline に関する回帰試験。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "scene_renderer"))

from scene_renderer import (
    AcousticSource,
    ConstantEnvelope,
    FreeField,
    LinearArray,
    Receiver,
    Scene,
    SceneRenderer,
    SourceComponent,
    StaticPose,
    ToneSpectrum,
)

from spflow import (
    MVDROverlapSaveBeamformer,
    PolyphaseDFTFilterBank,
    apply_beamformer_bands,
    beam_response_rms_db,
    design_cbf_weights,
    design_mvdr_weights,
    forgetting_factor_from_integration_time,
    integrate_band_covariances,
    integration_blocks_from_integration_time,
    make_directions,
    recommended_integration_time_for_independent_samples,
)


def _signal_peak_amplitude(level_db20: float) -> float:
    """`_signal_peak_amplitude` を実行する。"""
    return float(np.sqrt(2.0) * (10.0 ** (level_db20 / 20.0)))


def _make_source(receiver: Receiver, bearing_deg: float, freq: float, level_db20: float, elevation_deg: float = 0.0):
    """`_make_source` を実行する。"""
    component = SourceComponent(
        spectrum=ToneSpectrum(freq),
        envelope=ConstantEnvelope(),
        amplitude=_signal_peak_amplitude(level_db20),
    )
    source = AcousticSource.from_relative_bearing(
        bearing_deg=bearing_deg,
        distance=1000.0,
        receiver_pose=receiver.trajectory.pose(0.0),
        components=[component],
        elevation_deg=elevation_deg,
    )
    return source, component


def _render_scene(
    *,
    fs,
    freq,
    n_samples,
    n_ch,
    spacing_m,
    sound_speed,
    target_deg,
    interferer_deg,
    signal_level_db20,
    interferer_level_db20,
    target_el_deg=0.0,
    interferer_el_deg=0.0,
    include_target=True,
    include_interferer=True,
):
    """`_render_scene` を実行する。"""
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=n_ch, spacing=spacing_m),
    )
    target, target_component = _make_source(receiver, target_deg, freq, signal_level_db20, elevation_deg=target_el_deg)
    sources = []
    if include_target:
        sources.append(target)
    interferer = None
    if include_interferer:
        interferer, _ = _make_source(receiver, interferer_deg, freq, interferer_level_db20, elevation_deg=interferer_el_deg)
        sources.append(interferer)
    scene = Scene(sources=sources, ambient_fields=[], environment=FreeField(c=sound_speed))
    axis_t = np.arange(n_samples, dtype=float) / fs
    rendered = SceneRenderer().render(scene, receiver, axis_t)
    x = np.asarray(np.real(rendered), dtype=np.float32)
    reference = target_component.amplitude_value * np.cos(2.0 * np.pi * freq * axis_t)
    return x, reference, receiver, target, interferer, scene.environment


def _direction_from_source(receiver: Receiver, source: AcousticSource) -> np.ndarray:
    """`_direction_from_source` を実行する。"""
    receiver_pose = receiver.trajectory.pose(0.0)
    source_pos = source.trajectory.position(0.0)
    direction_world = source_pos - receiver_pose.position_world
    direction_world = direction_world / np.linalg.norm(direction_world)
    return receiver_pose.world_vector_to_array(direction_world)


def _steering_from_dir3d(receiver: Receiver, environment: FreeField, fft_size: int, fs: float, dir3d: np.ndarray) -> np.ndarray:
    """`_steering_from_dir3d` を実行する。"""
    tau = receiver.array.positions() @ dir3d / environment.c
    freqs = np.fft.fftfreq(fft_size, d=1.0 / fs)
    steering = np.exp(-1j * 2.0 * np.pi * freqs[np.newaxis, :, np.newaxis] * tau[:, np.newaxis, :])
    return np.moveaxis(steering, -1, 1)


def _make_target_beam_steering(receiver: Receiver, source: AcousticSource, environment: FreeField, fft_size: int, fs: float):
    """`_make_target_beam_steering` を実行する。"""
    direction = _direction_from_source(receiver, source)
    az_deg = float(np.rad2deg(np.arctan2(direction[1], direction[0])))
    el_deg = float(np.rad2deg(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
    dir3d, axis_az, axis_el = make_directions(
        az_min_deg=az_deg,
        az_max_deg=az_deg,
        el_min_deg=el_deg,
        el_max_deg=el_deg,
        n_beam_az_real=1,
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side='right side',
        el_preset_deg=[el_deg],
    )
    steering = _steering_from_dir3d(receiver, environment, fft_size, fs, dir3d)
    return steering, axis_az, axis_el


def _compute_mvdr_weights(X_cov: np.ndarray, steering_target: np.ndarray, fft_size: int, integration_time: float, rate: float) -> np.ndarray:
    """`_compute_mvdr_weights` を実行する。"""
    alpha = forgetting_factor_from_integration_time(integration_time, rate)
    n_blocks = integration_blocks_from_integration_time(integration_time, rate)
    rxx = integrate_band_covariances(
        X_cov,
        forgetting_factor=alpha,
        normalization=fft_size,
        n_blocks=n_blocks,
    )
    return np.stack(
        [design_mvdr_weights(rxx[band], steering_target[:, :, band], diag_load=1e-3) for band in range(X_cov.shape[1])],
        axis=-1,
    )


def test_scene_renderer_polyphase_mvdr_improves_target_reconstruction_over_cbf():
    """scene renderer の polyphase MVDR 経路について CBF より target 再構成を改善する を確認する。"""
    fs = 16000.0
    freq = 1000.0
    n_samples = 40000
    n_ch = 4
    spacing_m = 0.04
    target_deg = 20.0
    interferer_deg = -30.0
    sound_speed = 343.0
    signal_level_db20 = 0.0
    interferer_level_db20 = 0.0
    fft_size = 32
    rate = fs / fft_size
    integration_time = recommended_integration_time_for_independent_samples(n_ch, rate)

    x, reference, receiver, target, interferer, environment = _render_scene(
        fs=fs,
        freq=freq,
        n_samples=n_samples,
        n_ch=n_ch,
        spacing_m=spacing_m,
        sound_speed=sound_speed,
        target_deg=target_deg,
        interferer_deg=interferer_deg,
        signal_level_db20=signal_level_db20,
        interferer_level_db20=interferer_level_db20,
        include_target=True,
        include_interferer=True,
    )

    fb = PolyphaseDFTFilterBank(fft_size=fft_size)
    X = fb.analysis(x)
    steering_target, axis_az, axis_el = _make_target_beam_steering(receiver, target, environment, fft_size, fs)
    steering_interferer, _, _ = _make_target_beam_steering(receiver, interferer, environment, fft_size, fs)
    cbf_weights = design_cbf_weights(steering_target)
    mvdr_weights = _compute_mvdr_weights(X, steering_target, fft_size, integration_time, rate)

    Y_cbf = apply_beamformer_bands(X, cbf_weights)[0]
    Y_mvdr = apply_beamformer_bands(X, mvdr_weights)[0]
    y_cbf = np.real(fb.synthesis(Y_cbf, length=x.shape[-1]))
    y_mvdr = np.real(fb.synthesis(Y_mvdr, length=x.shape[-1]))
    reanalyzed_mvdr = fb.analysis(y_mvdr)

    target_bin = int(round(freq / (fs / fft_size))) % fft_size
    cbf_target_response = cbf_weights[:, :, target_bin].conj().T @ steering_target[:, :, target_bin]
    mvdr_target_response = mvdr_weights[:, :, target_bin].conj().T @ steering_target[:, :, target_bin]
    cbf_interferer_response = cbf_weights[:, :, target_bin].conj().T @ steering_interferer[:, :, target_bin]
    mvdr_interferer_response = mvdr_weights[:, :, target_bin].conj().T @ steering_interferer[:, :, target_bin]

    cbf_rms_error = np.sqrt(np.mean((y_cbf - reference) ** 2))
    mvdr_rms_error = np.sqrt(np.mean((y_mvdr - reference) ** 2))

    np.testing.assert_allclose(axis_az, np.array([20.0]), atol=1e-6)
    np.testing.assert_allclose(axis_el, np.array([0.0]), atol=1e-6)
    np.testing.assert_allclose(beam_response_rms_db(cbf_target_response[0, 0]), 0.0, atol=1e-2)
    np.testing.assert_allclose(beam_response_rms_db(mvdr_target_response[0, 0]), 0.0, atol=1e-2)
    assert beam_response_rms_db(mvdr_interferer_response[0, 0]) < beam_response_rms_db(cbf_interferer_response[0, 0]) - 3.0
    assert mvdr_rms_error < cbf_rms_error
    np.testing.assert_allclose(reanalyzed_mvdr, Y_mvdr, atol=1e-5)


def test_scene_renderer_polyphase_mvdr_with_interferer_only_covariance_avoids_self_nulling():
    """scene renderer の polyphase MVDR 経路について 妨害波のみの共分散で self nulling を避ける を確認する。"""
    fs = 32768.0
    freq = 1000.0
    n_samples = 65536
    n_ch = 32
    spacing_m = 343.0 / fs
    target_deg = 20.0
    interferer_deg = -30.0
    sound_speed = 343.0
    signal_level_db20 = 0.0
    interferer_level_db20 = 0.0
    fft_size = 64
    rate = fs / fft_size
    integration_time = recommended_integration_time_for_independent_samples(n_ch, rate)

    x_mix, reference, receiver, target, interferer, environment = _render_scene(
        fs=fs,
        freq=freq,
        n_samples=n_samples,
        n_ch=n_ch,
        spacing_m=spacing_m,
        sound_speed=sound_speed,
        target_deg=target_deg,
        interferer_deg=interferer_deg,
        signal_level_db20=signal_level_db20,
        interferer_level_db20=interferer_level_db20,
        include_target=True,
        include_interferer=True,
    )
    x_int, _, _, _, _, _ = _render_scene(
        fs=fs,
        freq=freq,
        n_samples=n_samples,
        n_ch=n_ch,
        spacing_m=spacing_m,
        sound_speed=sound_speed,
        target_deg=target_deg,
        interferer_deg=interferer_deg,
        signal_level_db20=signal_level_db20,
        interferer_level_db20=interferer_level_db20,
        include_target=False,
        include_interferer=True,
    )

    fb = PolyphaseDFTFilterBank(fft_size=fft_size)
    X_mix = fb.analysis(x_mix)
    X_int = fb.analysis(x_int)
    steering_target, _, _ = _make_target_beam_steering(receiver, target, environment, fft_size, fs)

    mvdr_mixture = _compute_mvdr_weights(X_mix, steering_target, fft_size, integration_time, rate)
    mvdr_interferer_only = _compute_mvdr_weights(X_int, steering_target, fft_size, integration_time, rate)

    y_mix_cov = np.real(fb.synthesis(apply_beamformer_bands(X_mix, mvdr_mixture)[0], length=x_mix.shape[-1]))
    y_int_cov = np.real(fb.synthesis(apply_beamformer_bands(X_mix, mvdr_interferer_only)[0], length=x_mix.shape[-1]))

    err_mix_cov = np.sqrt(np.mean((y_mix_cov - reference) ** 2))
    err_int_cov = np.sqrt(np.mean((y_int_cov - reference) ** 2))

    assert err_int_cov < err_mix_cov * 0.8


def _collect_overlap_save_output(records: list[tuple[int, np.ndarray]], n_beam: int, n_band: int) -> np.ndarray:
    """`_collect_overlap_save_output` を実行する。"""
    per_band: list[list[np.ndarray]] = [[] for _ in range(n_band)]
    for band_idx, valid in records:
        per_band[band_idx].append(valid)

    pieces = []
    for band_idx in range(n_band):
        if per_band[band_idx]:
            pieces.append(np.concatenate(per_band[band_idx], axis=-1))
        else:
            pieces.append(np.zeros((n_beam, 0), dtype=np.complex64))
    return np.stack(pieces, axis=1)


def test_scene_renderer_polyphase_mvdr_overlap_save_recovers_continuous_time_waveform():
    """scene renderer の polyphase MVDR 経路について overlap-save で連続時間波形を復元する を確認する。"""
    fs = 16000.0
    freq = 1000.0
    n_samples = 40000
    n_ch = 4
    spacing_m = 0.04
    target_deg = 60.0
    sound_speed = 343.0
    signal_level_db20 = 0.0
    noise_level_db20 = -100.0
    fft_size = 32
    subband_frame_size = 2048
    subband_valid_size = 1024
    subband_chunk_size = 257
    rate = fs / fft_size
    integration_time = recommended_integration_time_for_independent_samples(n_ch, rate)

    x_clean, reference, receiver, target, _, environment = _render_scene(
        fs=fs,
        freq=freq,
        n_samples=n_samples,
        n_ch=n_ch,
        spacing_m=spacing_m,
        sound_speed=sound_speed,
        target_deg=target_deg,
        interferer_deg=-30.0,
        signal_level_db20=signal_level_db20,
        interferer_level_db20=0.0,
        include_target=True,
        include_interferer=False,
    )
    rng = np.random.default_rng(1234)
    noise_std = 10.0 ** (noise_level_db20 / 20.0)
    x_mix = x_clean + noise_std * rng.standard_normal(x_clean.shape)
    x_cov = noise_std * rng.standard_normal(x_clean.shape)

    fb = PolyphaseDFTFilterBank(fft_size=fft_size)
    X_mix = fb.analysis(x_mix)
    X_cov = fb.analysis(x_cov)
    steering_target, _, _ = _make_target_beam_steering(receiver, target, environment, fft_size, fs)
    mvdr_weights = _compute_mvdr_weights(X_cov, steering_target, fft_size, integration_time, rate)

    y_direct = np.real(fb.synthesis(apply_beamformer_bands(X_mix, mvdr_weights)[0], length=x_mix.shape[-1]))

    beamformer = MVDROverlapSaveBeamformer(
        mvdr_weights,
        frame_size=subband_frame_size,
        valid_size=subband_valid_size,
    )
    records: list[tuple[int, np.ndarray]] = []
    for start in range(0, X_mix.shape[-1], subband_chunk_size):
        records.extend(beamformer.process(X_mix[:, :, start : start + subband_chunk_size]))
    records.extend(beamformer.flush())

    Y_os = _collect_overlap_save_output(records, n_beam=1, n_band=fb.n_bands)[0, :, : X_mix.shape[-1]]
    y_os = np.real(fb.synthesis(Y_os, length=x_mix.shape[-1]))

    jump_abs_error = np.abs(np.diff(y_os - y_direct))
    boundary_indices = np.arange(subband_valid_size, y_os.size, subband_valid_size) - 1
    boundary_indices = boundary_indices[(boundary_indices >= 0) & (boundary_indices < jump_abs_error.size)]

    assert np.sqrt(np.mean((y_os - reference) ** 2)) <= 3e-2 * np.sqrt(2.0)
    np.testing.assert_allclose(y_os, y_direct, atol=1e-5)
    assert float(np.max(jump_abs_error)) <= 2e-6
    if boundary_indices.size > 0:
        assert float(np.max(jump_abs_error[boundary_indices])) <= 2e-6
