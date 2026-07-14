"""fullband CBF のビーム応答と再構成波形を評価するサンプル。"""

# scene_renderer で合成した観測波面を使い、設計した steering や重みが
# 単体式だけでなく波形再構成・指向性評価の流れ全体で破綻しないかを確認する例である。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'vendor' / 'scene_renderer'))

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

from spflow import FrameBuffer, FullDFTFilterBank, design_cbf_coefficients
from spflow.beamforming import apply_beamformer


def signal_peak_amplitude(level_db20: float) -> float:
    """振幅レベル dB20 を正弦波のピーク振幅へ変換する。"""
    return float(np.sqrt(2.0) * (10.0 ** (level_db20 / 20.0)))


def render_target_scene(fs, freq, n_samples, n_ch, spacing_m, bearing_deg, sound_speed, signal_level_db20):
    """単一 target シーンをレンダリングし、観測信号と参照信号を返す。"""
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=n_ch, spacing=spacing_m),
    )
    component = SourceComponent(
        spectrum=ToneSpectrum(freq),
        envelope=ConstantEnvelope(),
        amplitude=signal_peak_amplitude(signal_level_db20),
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


def make_steering(receiver, source, environment, fft_size, fs):
    """target 方位に向けた steering ベクトルを構成する。"""
    receiver_pose = receiver.trajectory.pose(0.0)
    source_pos = source.trajectory.position(0.0)
    direction_world = source_pos - receiver_pose.position_world
    direction_world = direction_world / np.linalg.norm(direction_world)
    direction_array = receiver_pose.world_vector_to_array(direction_world)
    tau = receiver.array.positions() @ direction_array / environment.c
    freqs = np.fft.fftfreq(fft_size, d=1.0 / fs)
    return np.exp(-1j * 2.0 * np.pi * freqs[np.newaxis, :] * tau[:, np.newaxis])[:, np.newaxis, :]


def apply_fixed(x, weights, fb, chunk_size, pos_band, neg_band):
    """固定重みビームフォーマを subband 信号へ適用して時間波形へ戻す。"""
    buffer = FrameBuffer(frame_size=fb.fft_size, hop_size=fb.hop_size, axis=-1)
    frames = []
    for start in range(0, x.shape[-1], chunk_size):
        for frame in buffer.process(x[:, start : start + chunk_size]):
            spec = np.fft.fft(frame * fb.prototype, axis=-1)
            yb = np.zeros((spec.shape[1],), dtype=np.complex64)
            yb[pos_band] = apply_beamformer(spec[:, pos_band][:, None], weights[:, :, pos_band])[0, 0]
            yb[neg_band] = apply_beamformer(spec[:, neg_band][:, None], weights[:, :, neg_band])[0, 0]
            frames.append(yb)
    Y = np.stack(frames, axis=-1)
    y = fb.synthesis(Y, length=x.shape[-1])
    return y, Y


def main() -> None:
    """fullband CBF のビーム応答と再構成波形を表示する。"""
    fs = 16000.0
    freq = 1000.0
    n_samples = 4096
    n_ch = 4
    spacing_m = 0.04
    target_deg = 20.0
    sound_speed = 343.0
    signal_level_db20 = 0.0
    noise_level_db20 = -60.0
    fft_size = 256
    hop_size = 128
    chunk_size = 64
    pos_band = int(round(freq / (fs / fft_size)))
    neg_band = (-pos_band) % fft_size
    noise_std = 10.0 ** (noise_level_db20 / 20.0)

    x_clean, reference, receiver, source, environment = render_target_scene(
        fs, freq, n_samples, n_ch, spacing_m, target_deg, sound_speed, signal_level_db20
    )
    rng = np.random.default_rng(1234)
    x = x_clean + noise_std * rng.standard_normal(x_clean.shape)

    fb = FullDFTFilterBank(fft_size=fft_size, hop_size=hop_size)
    steering = make_steering(receiver, source, environment, fft_size, fs)
    weights = np.zeros((steering.shape[0], steering.shape[1], steering.shape[2]), dtype=np.complex64)
    weights[:, :, pos_band] = design_cbf_coefficients(steering[:, :, pos_band])
    weights[:, :, neg_band] = design_cbf_coefficients(steering[:, :, neg_band])

    y, Y = apply_fixed(x, weights, fb, chunk_size, pos_band, neg_band)
    rean = fb.analysis(y)
    resp_pos = weights[:, :, pos_band].T @ steering[:, :, pos_band]
    resp_neg = weights[:, :, neg_band].T @ steering[:, :, neg_band]

    err = y - reference
    print(f'positive_target_response_db={20 * np.log10(np.abs(resp_pos[0, 0])):.12f}')
    print(f'negative_target_response_db={20 * np.log10(np.abs(resp_neg[0, 0])):.12f}')
    print(f'max_subband_reanalysis_error={np.max(np.abs(rean - Y)):.12e}')
    print(f'rms_subband_reanalysis_error={np.sqrt(np.mean(np.abs(rean - Y)**2)):.12e}')
    print(f'rms_time_error_to_reference={np.sqrt(np.mean(err**2)):.12e}')
    print(f'max_time_error_to_reference={np.max(np.abs(err)):.12e}')


if __name__ == '__main__':
    main()
