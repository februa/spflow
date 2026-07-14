"""polyphase CBF の周波数 sweep 結果を集計するサンプル。"""

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
    Environment,
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
    CBFOverlapSaveBeamformer,
    PolyphaseDFTFilterBank,
    apply_beamformer_bands,
    beam_response_rms_db,
    design_cbf_coefficients,
)

FREQUENCIES = [10.0, 100.0, 490.0, 500.0, 510.0, 990.0, 1000.0, 1010.0, 3990.0, 4000.0, 4010.0, 7500.0, 7900.0]


def signal_peak_amplitude(level_db20: float) -> float:
    """振幅レベル dB20 を正弦波のピーク振幅へ変換する。"""
    return float(np.sqrt(2.0) * (10.0 ** (level_db20 / 20.0)))


def collect_overlap_save_output(records: list[tuple[int, np.ndarray]], n_beam: int, n_band: int) -> np.ndarray:
    """overlap-save で分割された出力を帯域別の連続波形へ束ねる。"""
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


def render_target_scene(*, fs, freq, n_samples, n_ch, spacing_m, bearing_deg, sound_speed, signal_level_db20):
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


def make_steering(receiver: Receiver, source: AcousticSource, environment: Environment, fft_size: int, fs: float) -> np.ndarray:
    """target 方位に向けた steering ベクトルを構成する。"""
    receiver_pose = receiver.trajectory.pose(0.0)
    source_pos = source.trajectory.position(0.0)
    direction_world = source_pos - receiver_pose.position_world
    direction_world = direction_world / np.linalg.norm(direction_world)
    direction_array = receiver_pose.world_vector_to_array(direction_world)
    tau = receiver.array.positions() @ direction_array / environment.c
    freqs = np.fft.fftfreq(fft_size, d=1.0 / fs)
    steering = np.exp(-1j * 2.0 * np.pi * freqs[np.newaxis, :] * tau[:, np.newaxis])
    return steering[:, np.newaxis, :]


def evaluate_frequency(freq: float) -> dict[str, float]:
    """指定周波数条件でビームフォーマの応答指標を評価する。"""
    fs = 16000.0
    n_samples = 40000
    n_ch = 4
    spacing_m = 0.04
    target_deg = 20.0
    sound_speed = 343.0
    signal_level_db20 = 0.0
    noise_level_db20 = -60.0
    fb_fft_size = 32
    subband_frame_size = 2048
    subband_valid_size = 1024
    subband_chunk_size = 257
    noise_std = 10.0 ** (noise_level_db20 / 20.0)

    x_clean, reference, receiver, source, environment = render_target_scene(
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

    fb = PolyphaseDFTFilterBank(fft_size=fb_fft_size)
    X = fb.analysis(x)
    steering = make_steering(receiver, source, environment, fft_size=fb_fft_size, fs=fs)
    weights = design_cbf_coefficients(steering)
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

    Y = collect_overlap_save_output(records, n_beam=1, n_band=fb.n_bands)
    Y = Y[:, :, : X.shape[-1]]
    y = fb.synthesis(Y[0], length=x.shape[-1])
    reanalyzed = fb.analysis(y)

    nearest_bin = int(round(freq / (fs / fb_fft_size))) % fb_fft_size
    target_response = weights[:, :, nearest_bin].T @ steering[:, :, nearest_bin]
    time_error = np.real(y) - reference

    return {
        'freq_hz': freq,
        'nearest_bin': nearest_bin,
        'target_response_db': beam_response_rms_db(target_response[0, 0]),
        'max_subband_diff_to_direct': float(np.max(np.abs(Y[0] - expected[0]))),
        'rms_subband_diff_to_direct': float(np.sqrt(np.mean(np.abs(Y[0] - expected[0]) ** 2))),
        'max_subband_reanalysis_error': float(np.max(np.abs(reanalyzed - Y[0]))),
        'rms_subband_reanalysis_error': float(np.sqrt(np.mean(np.abs(reanalyzed - Y[0]) ** 2))),
        'rms_time_error_to_reference': float(np.sqrt(np.mean(time_error ** 2))),
        'max_time_error_to_reference': float(np.max(np.abs(time_error))),
    }


def main() -> None:
    """polyphase CBF の周波数 sweep 結果を表形式で表示する。"""
    rows = [evaluate_frequency(freq) for freq in FREQUENCIES]

    header = (
        '| freq [Hz] | nearest_bin | target_response_db | '
        'max_subband_diff_to_direct | max_subband_reanalysis_error | '
        'rms_time_error_to_reference | max_time_error_to_reference |'
    )
    sep = '|---:|---:|---:|---:|---:|---:|---:|'
    print(header)
    print(sep)
    for row in rows:
        print(
            f"| {row['freq_hz']:.0f} | {row['nearest_bin']} | {row['target_response_db']:.12f} | "
            f"{row['max_subband_diff_to_direct']:.3e} | {row['max_subband_reanalysis_error']:.3e} | "
            f"{row['rms_time_error_to_reference']:.3e} | {row['max_time_error_to_reference']:.3e} |"
        )


if __name__ == '__main__':
    main()
