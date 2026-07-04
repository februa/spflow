"""cbf streaming fullband に関する回帰試験。"""

# ここでは steering の向き、共分散推定、重み適用後の再構成が噛み合うことを
# 決定論的な入力で固定し、ビームフォーミング変更時の退行を早期に検知する。

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

from spflow import CBFOverlapSaveBeamformer, FullDFTFilterBank, apply_beamformer_bands, beam_response_rms_db, design_cbf_weights


def _signal_peak_amplitude(level_db20: float) -> float:
    """`_signal_peak_amplitude` を実行する。"""
    return float(np.sqrt(2.0) * (10.0 ** (level_db20 / 20.0)))


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


def _render_target_scene(*, fs, freq, n_samples, n_ch, spacing_m, bearing_deg, sound_speed, signal_level_db20):
    """`_render_target_scene` を実行する。"""
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=n_ch, spacing=spacing_m),
    )
    component = SourceComponent(
        spectrum=ToneSpectrum(freq),
        envelope=ConstantEnvelope(),
        amplitude=_signal_peak_amplitude(signal_level_db20),
    )
    source = AcousticSource.from_relative_bearing(
        bearing_deg=bearing_deg,
        distance=1000.0,
        receiver_pose=receiver.trajectory.pose(0.0),
        components=[component],
        elevation_deg=0.0,
    )
    scene = Scene(sources=[source], ambient_fields=[], environment=FreeField(c=sound_speed))
    axis_t = np.arange(n_samples, dtype=float) / fs
    rendered = SceneRenderer().render(scene, receiver, axis_t)
    x = np.asarray(np.real(rendered), dtype=np.float32)
    reference = component.amplitude_value * np.cos(2.0 * np.pi * freq * axis_t)
    return x, reference, receiver, source, scene.environment


def _make_steering_from_scene(receiver: Receiver, source: AcousticSource, environment: FreeField, fft_size: int, fs: float) -> np.ndarray:
    """`_make_steering_from_scene` を実行する。"""
    receiver_pose = receiver.trajectory.pose(0.0)
    source_pos = source.trajectory.position(0.0)
    direction_world = source_pos - receiver_pose.position_world
    direction_world = direction_world / np.linalg.norm(direction_world)
    direction_array = receiver_pose.world_vector_to_array(direction_world)
    tau = receiver.array.positions() @ direction_array / environment.c
    freqs = np.fft.fftfreq(fft_size, d=1.0 / fs)
    steering = np.exp(-1j * 2.0 * np.pi * freqs[np.newaxis, :] * tau[:, np.newaxis])
    return steering[:, np.newaxis, :]


def test_scene_renderer_streaming_cbf_overlap_save_matches_fullband_cbf_and_recovers_time_waveform():
    """scene renderer の streaming CBF overlap-save 経路について fullband CBF と一致し時間波形も復元する を確認する。"""
    fs = 16000.0
    freq = 1000.0
    n_samples = 20000
    n_ch = 4
    spacing_m = 0.04
    target_deg = 20.0
    sound_speed = 343.0
    signal_level_db20 = 0.0
    noise_level_db20 = -60.0
    fb_fft_size = 32
    fb_hop_size = 16
    subband_frame_size = 2048
    subband_valid_size = 1024
    subband_chunk_size = 257
    target_bin = int(round(freq / (fs / fb_fft_size)))
    noise_std = 10.0 ** (noise_level_db20 / 20.0)

    x_clean, reference, receiver, source, environment = _render_target_scene(
        fs=fs,
        freq=freq,
        n_samples=n_samples,
        n_ch=n_ch,
        spacing_m=spacing_m,
        bearing_deg=target_deg,
        sound_speed=sound_speed,
        signal_level_db20=signal_level_db20,
    )
    rng = np.random.default_rng(1234)
    x = x_clean + noise_std * rng.standard_normal(x_clean.shape)

    fb = FullDFTFilterBank(fft_size=fb_fft_size, hop_size=fb_hop_size)
    X = fb.analysis(x)
    steering = _make_steering_from_scene(receiver, source, environment, fft_size=fb_fft_size, fs=fs)
    weights = design_cbf_weights(steering)
    expected = apply_beamformer_bands(X, weights)
    beamformer = CBFOverlapSaveBeamformer(
        steering,
        frame_size=subband_frame_size,
        valid_size=subband_valid_size,
    )

    records: list[tuple[int, np.ndarray]] = []
    for start in range(0, X.shape[-1], subband_chunk_size):
        records.extend(beamformer.process(X[:, :, start : start + subband_chunk_size]))
    records.extend(beamformer.flush())

    Y = _collect_overlap_save_output(records, n_beam=1, n_band=fb.n_bands)
    Y = Y[:, :, : X.shape[-1]]
    y = fb.synthesis(Y[0], length=x.shape[-1])

    target_response = weights[:, :, target_bin].conj().T @ steering[:, :, target_bin]
    target_response_db = beam_response_rms_db(target_response[0, 0])
    time_error = np.real(y) - reference

    np.testing.assert_allclose(target_response_db, 0.0, atol=1e-2)
    np.testing.assert_allclose(Y[0], expected[0], atol=1e-5)
    assert np.all(np.isfinite(Y))
    assert np.all(np.isfinite(y))
    assert np.sqrt(np.mean(time_error**2)) <= 3e-2 * np.sqrt(2.0)
