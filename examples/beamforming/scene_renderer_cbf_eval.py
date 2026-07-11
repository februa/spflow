"""単一周波数の CBF 応答と再構成波形を評価するサンプル。"""

# scene_renderer で合成した観測波面を使い、設計した steering や重みが
# 単体式だけでなく波形再構成・指向性評価の流れ全体で破綻しないかを確認する例である。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from scene_renderer import (  # noqa: E402
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

from spflow import (  # noqa: E402
    DFT_FilterBank,
    FrameBuffer,
    design_cbf_weights,
    relative_arrival_delay,
    steering_from_relative_delay,
    tone_rms_level_db_to_peak_amplitude,
    unit_direction_from_positions,
)
from spflow.beamforming import apply_beamformer  # noqa: E402


def render_target_scene(
    fs, freq, n_samples, n_ch, spacing_m, bearing_deg, sound_speed, signal_level_db20
):
    """単一 target シーンをレンダリングし、観測信号と参照信号を返す。"""
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=n_ch, spacing=spacing_m),
    )
    component = SourceComponent(
        spectrum=ToneSpectrum(freq),
        envelope=ConstantEnvelope(),
        amplitude=tone_rms_level_db_to_peak_amplitude(signal_level_db20),
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
    direction_world = unit_direction_from_positions(receiver_pose.position_world, source_pos)
    direction_array = receiver_pose.world_vector_to_array(direction_world)
    tau = relative_arrival_delay(
        receiver.array.positions(),
        direction_array,
        sound_speed_m_per_s=environment.c,
    )
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / fs)
    return steering_from_relative_delay(tau, freqs)[:, np.newaxis, :]


def apply_fixed(x, weights, fb, chunk_size, active_band):
    """固定重みビームフォーマを subband 信号へ適用して時間波形へ戻す。"""
    buffer = FrameBuffer(frame_size=fb.fft_size, hop_size=fb.hop_size, axis=-1)
    frames = []
    for start in range(0, x.shape[-1], chunk_size):
        for frame in buffer.process(x[:, start : start + chunk_size]):
            spec = np.fft.rfft(frame * fb.prototype, axis=-1)
            yb = np.zeros((spec.shape[1],), dtype=np.complex64)
            yb[active_band] = apply_beamformer(
                spec[:, active_band][:, None], weights[:, :, active_band]
            )[0, 0]
            frames.append(yb)
    Y = np.stack(frames, axis=-1)
    y = fb.synthesis(Y, length=x.shape[-1])
    return y, Y


def main() -> None:
    """単一周波数 CBF のビーム応答と再構成波形を表示する。"""
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
    target_bin = int(round(freq / (fs / fft_size)))
    noise_std = 10.0 ** (noise_level_db20 / 20.0)

    x_clean, reference, receiver, source, environment = render_target_scene(
        fs, freq, n_samples, n_ch, spacing_m, target_deg, sound_speed, signal_level_db20
    )
    rng = np.random.default_rng(1234)
    x = x_clean + noise_std * rng.standard_normal(x_clean.shape)

    fb = DFT_FilterBank(fft_size=fft_size, hop_size=hop_size)
    steering = make_steering(receiver, source, environment, fft_size, fs)
    weights = np.zeros(
        (steering.shape[0], steering.shape[1], steering.shape[2]), dtype=np.complex64
    )
    weights[:, :, target_bin] = design_cbf_weights(steering[:, :, target_bin])

    y, Y = apply_fixed(x, weights, fb, chunk_size, target_bin)
    rean = fb.analysis(y)
    resp = weights[:, :, target_bin].conj().T @ steering[:, :, target_bin]
    resp_db = 20 * np.log10(np.abs(resp[0, 0]))

    angles = np.linspace(-90.0, 90.0, 181)
    responses = []
    for angle in angles:
        x_scan, ref_scan, _, _, _ = render_target_scene(
            fs, freq, n_samples, n_ch, spacing_m, float(angle), sound_speed, signal_level_db20
        )
        y_scan, _ = apply_fixed(x_scan, weights, fb, chunk_size, target_bin)
        responses.append(20 * np.log10(np.sqrt(np.mean(y_scan**2)) / np.sqrt(np.mean(ref_scan**2))))
    responses = np.asarray(responses)

    err = y - reference
    print(f"target_response_db={resp_db:.12f}")
    print(f"target_scan_level_db={responses[np.argmin(np.abs(angles - target_deg))]:.12f}")
    print(f"peak_angle_deg={angles[int(np.argmax(responses))]:.3f}")
    print(f"peak_level_db={np.max(responses):.12f}")
    print(f"max_subband_reanalysis_error={np.max(np.abs(rean - Y)):.12e}")
    print(f"rms_subband_reanalysis_error={np.sqrt(np.mean(np.abs(rean - Y) ** 2)):.12e}")
    print(f"rms_time_error_to_reference={np.sqrt(np.mean(err**2)):.12e}")
    print(f"max_time_error_to_reference={np.max(np.abs(err)):.12e}")


if __name__ == "__main__":
    main()
