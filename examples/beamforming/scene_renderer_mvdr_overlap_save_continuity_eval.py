"""MVDR overlap-save 出力の連続性を確認するサンプル。"""

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
    MVDROverlapSaveBeamformer,
    PolyphaseDFTFilterBank,
    apply_beamformer_bands,
    beam_response_rms_db,
    design_mvdr_coefficients,
    forgetting_factor_from_integration_time,
    integrate_band_covariances,
    integration_blocks_from_integration_time,
    make_directions,
    recommended_integration_time_for_independent_samples,
)


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


def make_source(receiver: Receiver, bearing_deg: float, freq: float, level_db20: float, elevation_deg: float = 0.0):
    """指定方位・周波数条件の単一トーン音源を生成する。"""
    component = SourceComponent(
        spectrum=ToneSpectrum(freq),
        envelope=ConstantEnvelope(),
        amplitude=signal_peak_amplitude(level_db20),
    )
    source = AcousticSource.from_relative_bearing(
        bearing_deg=bearing_deg,
        distance=1000.0,
        receiver_pose=receiver.trajectory.pose(0.0),
        components=[component],
        elevation_deg=elevation_deg,
    )
    return source, component


def render_target_scene(*, fs, freq, n_samples, n_ch, spacing_m, bearing_deg, sound_speed, signal_level_db20):
    """単一 target シーンをレンダリングし、観測信号と参照信号を返す。"""
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=n_ch, spacing=spacing_m),
    )
    source, component = make_source(receiver, bearing_deg, freq, signal_level_db20)
    scene = Scene(sources=[source], ambient_fields=[], environment=FreeField(c=sound_speed))
    axis_t = np.arange(n_samples, dtype=float) / fs
    rendered = SceneRenderer().render(scene, receiver, axis_t)
    x_clean = np.asarray(np.real(rendered), dtype=np.float32)
    reference = component.amplitude_value * np.cos(2.0 * np.pi * freq * axis_t)
    return x_clean, reference, receiver, source, scene.environment


def direction_from_source(receiver: Receiver, source: AcousticSource) -> np.ndarray:
    """音源位置から受信機座標系での到来方向ベクトルを求める。"""
    receiver_pose = receiver.trajectory.pose(0.0)
    source_pos = source.trajectory.position(0.0)
    direction_world = source_pos - receiver_pose.position_world
    direction_world = direction_world / np.linalg.norm(direction_world)
    return receiver_pose.world_vector_to_array(direction_world)


def steering_from_dir3d(receiver: Receiver, environment: Environment, fft_size: int, fs: float, dir3d: np.ndarray) -> np.ndarray:
    """到来方向ベクトルから周波数依存 steering ベクトルを構成する。"""
    tau = receiver.array.positions() @ dir3d / environment.c
    freqs = np.fft.fftfreq(fft_size, d=1.0 / fs)
    steering = np.exp(-1j * 2.0 * np.pi * freqs[np.newaxis, :, np.newaxis] * tau[:, np.newaxis, :])
    return np.moveaxis(steering, -1, 1)


def make_target_beam_steering(receiver: Receiver, source: AcousticSource, environment: Environment, fft_size: int, fs: float):
    """target 方位に対応する steering ベクトルと走査軸を構成する。"""
    direction = direction_from_source(receiver, source)
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
    steering = steering_from_dir3d(receiver, environment, fft_size, fs, dir3d)
    return steering, axis_az, axis_el


def main() -> None:
    """MVDR overlap-save 出力の境界連続性評価結果を表示する。"""
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
    noise_std = 10.0 ** (noise_level_db20 / 20.0)
    x_mix = x_clean + noise_std * rng.standard_normal(x_clean.shape)
    x_cov = noise_std * rng.standard_normal(x_clean.shape)

    fb = PolyphaseDFTFilterBank(fft_size=fft_size)
    X_mix = fb.analysis(x_mix)
    X_cov = fb.analysis(x_cov)
    steering, axis_az, axis_el = make_target_beam_steering(receiver, source, environment, fft_size, fs)

    rate = fs / fft_size
    integration_time = recommended_integration_time_for_independent_samples(n_ch, rate)
    alpha = forgetting_factor_from_integration_time(integration_time, rate)
    n_blocks = integration_blocks_from_integration_time(integration_time, rate)
    rxx = integrate_band_covariances(
        X_cov,
        forgetting_factor=alpha,
        normalization=fft_size,
        n_blocks=n_blocks,
    )
    mvdr_weights = np.stack(
        [design_mvdr_coefficients(rxx[band], steering[:, :, band], diag_load=1e-3) for band in range(fb.n_bands)],
        axis=-1,
    )

    Y_direct = apply_beamformer_bands(X_mix, mvdr_weights)[0]
    y_direct = np.real(fb.synthesis(Y_direct, length=x_mix.shape[-1]))

    beamformer = MVDROverlapSaveBeamformer(
        mvdr_weights,
        frame_size=subband_frame_size,
        valid_size=subband_valid_size,
    )
    records: list[tuple[int, np.ndarray]] = []
    for start in range(0, X_mix.shape[-1], subband_chunk_size):
        records.extend(beamformer.process(X_mix[:, :, start : start + subband_chunk_size]))
    records.extend(beamformer.flush())
    Y_os = collect_overlap_save_output(records, n_beam=1, n_band=fb.n_bands)[0, :, : X_mix.shape[-1]]
    y_os = np.real(fb.synthesis(Y_os, length=x_mix.shape[-1]))

    target_bin = int(round(freq / (fs / fft_size))) % fft_size
    target_response = mvdr_weights[:, :, target_bin].T @ steering[:, :, target_bin]

    direct_jump_abs = np.abs(np.diff(y_direct))
    os_jump_abs = np.abs(np.diff(y_os))
    ref_jump_abs = np.abs(np.diff(reference))
    jump_abs_error = np.abs(np.diff(y_os - y_direct))
    jump_abs_error_to_ref = np.abs(np.diff(y_os - reference))

    boundary_period = subband_valid_size
    boundary_indices = np.arange(boundary_period, y_os.size, boundary_period) - 1
    boundary_indices = boundary_indices[(boundary_indices >= 0) & (boundary_indices < jump_abs_error.size)]

    print(f'target_axis_az_deg={axis_az[0]:.6f}')
    print(f'target_axis_el_deg={axis_el[0]:.6f}')
    print(f'target_response_db={beam_response_rms_db(target_response[0,0]):.12f}')
    print(f'integration_time_s={integration_time:.6f}')
    print(f'integration_blocks={n_blocks}')
    print(f'rms_time_error_direct_to_ref={np.sqrt(np.mean((y_direct-reference)**2)):.12e}')
    print(f'rms_time_error_os_to_ref={np.sqrt(np.mean((y_os-reference)**2)):.12e}')
    print(f'max_time_diff_os_vs_direct={np.max(np.abs(y_os-y_direct)):.12e}')
    print(f'rms_time_diff_os_vs_direct={np.sqrt(np.mean((y_os-y_direct)**2)):.12e}')
    print(f'max_jump_abs_direct={np.max(direct_jump_abs):.12e}')
    print(f'max_jump_abs_os={np.max(os_jump_abs):.12e}')
    print(f'max_jump_abs_ref={np.max(ref_jump_abs):.12e}')
    print(f'max_jump_abs_error_os_vs_direct={np.max(jump_abs_error):.12e}')
    print(f'rms_jump_abs_error_os_vs_direct={np.sqrt(np.mean(jump_abs_error**2)):.12e}')
    print(f'max_jump_abs_error_os_to_ref={np.max(jump_abs_error_to_ref):.12e}')
    print(f'rms_jump_abs_error_os_to_ref={np.sqrt(np.mean(jump_abs_error_to_ref**2)):.12e}')
    if boundary_indices.size > 0:
        print(f'max_boundary_jump_abs_error_os_vs_direct={np.max(jump_abs_error[boundary_indices]):.12e}')
        print(f'rms_boundary_jump_abs_error_os_vs_direct={np.sqrt(np.mean(jump_abs_error[boundary_indices]**2)):.12e}')
    else:
        print('max_boundary_jump_abs_error_os_vs_direct=nan')
        print('rms_boundary_jump_abs_error_os_vs_direct=nan')


if __name__ == '__main__':
    main()
