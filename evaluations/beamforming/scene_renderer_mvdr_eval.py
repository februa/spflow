"""単一条件のMVDRビーム応答と再構成波形を評価する。"""

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

from spflow import (
    PolyphaseDFTFilterBank,
    apply_beamformer_bands,
    beam_response_rms_db,
    design_cbf_coefficients,
    design_mvdr_coefficients,
    forgetting_factor_from_integration_time,
    integrate_band_covariances,
    integration_blocks_from_integration_time,
    make_directions,
    recommended_integration_time_for_independent_samples,
)


def signal_peak_amplitude(level_db20):
    """振幅レベル dB20 を正弦波のピーク振幅へ変換する。"""
    return float(np.sqrt(2.0) * 10.0 ** (level_db20 / 20.0))


def make_source(receiver, bearing_deg, freq, level_db20, elevation_deg=0.0):
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


def render_scene(
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
    """指定条件のシーンをレンダリングし、観測信号と参照情報を返す。"""
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=n_ch, spacing=spacing_m),
    )
    target, target_component = make_source(receiver, target_deg, freq, signal_level_db20, elevation_deg=target_el_deg)
    sources = []
    if include_target:
        sources.append(target)
    interferer = None
    if include_interferer:
        interferer, _ = make_source(receiver, interferer_deg, freq, interferer_level_db20, elevation_deg=interferer_el_deg)
        sources.append(interferer)
    scene = Scene(sources=sources, ambient_fields=[], environment=FreeField(c=sound_speed))
    axis_t = np.arange(n_samples, dtype=float) / fs
    rendered = SceneRenderer().render(scene, receiver, axis_t)
    mixture = np.asarray(np.real(rendered), dtype=np.float32)
    reference = signal_peak_amplitude(signal_level_db20) * np.cos(2.0 * np.pi * freq * axis_t)
    return mixture, reference, receiver, target, interferer, scene.environment


def direction_from_source(receiver, source):
    """音源位置から受信機座標系での到来方向ベクトルを求める。"""
    receiver_pose = receiver.trajectory.pose(0.0)
    source_pos = source.trajectory.position(0.0)
    direction_world = source_pos - receiver_pose.position_world
    direction_world = direction_world / np.linalg.norm(direction_world)
    return receiver_pose.world_vector_to_array(direction_world)


def steering_from_dir3d(receiver, environment, fft_size, fs, dir3d):
    """到来方向ベクトルから周波数依存 steering ベクトルを構成する。"""
    tau = receiver.array.positions() @ dir3d / environment.c
    freqs = np.fft.fftfreq(fft_size, d=1.0 / fs)
    steering = np.exp(-1j * 2.0 * np.pi * freqs[np.newaxis, :, np.newaxis] * tau[:, np.newaxis, :])
    return np.moveaxis(steering, -1, 1)


def make_target_beam_steering(receiver, source, environment, fft_size, fs, array_side='right side'):
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
        array_side=array_side,
        el_preset_deg=[el_deg],
    )
    steering = steering_from_dir3d(receiver, environment, fft_size, fs, dir3d)
    return steering, dir3d, axis_az, axis_el


def main() -> None:
    """単一条件の MVDR 応答と再構成波形を表示する。"""
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
    target_el_deg = 0.0
    interferer_el_deg = 0.0

    x_mix, reference, receiver, target, interferer, environment = render_scene(
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
        target_el_deg=target_el_deg,
        interferer_el_deg=interferer_el_deg,
        include_target=True,
        include_interferer=True,
    )
    x_cov, _, _, _, _, _ = render_scene(
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
        target_el_deg=target_el_deg,
        interferer_el_deg=interferer_el_deg,
        include_target=False,
        include_interferer=True,
    )

    fb = PolyphaseDFTFilterBank(fft_size=fft_size)
    X_mix = fb.analysis(x_mix)
    X_cov = fb.analysis(x_cov)
    steering_target, dir3d_target, axis_az, axis_el = make_target_beam_steering(receiver, target, environment, fft_size, fs)
    steering_interferer, _, _, _ = make_target_beam_steering(receiver, interferer, environment, fft_size, fs)
    cbf_weights = design_cbf_coefficients(steering_target)

    forgetting_factor = forgetting_factor_from_integration_time(integration_time, rate)
    n_blocks = integration_blocks_from_integration_time(integration_time, rate)
    rxx = integrate_band_covariances(
        X_cov,
        forgetting_factor=forgetting_factor,
        normalization=fft_size,
        n_blocks=n_blocks,
    )

    mvdr_weights = np.stack(
        [design_mvdr_coefficients(rxx[band], steering_target[:, :, band], diag_load=1e-3) for band in range(X_mix.shape[1])],
        axis=-1,
    )

    Y_cbf = apply_beamformer_bands(X_mix, cbf_weights)[0]
    Y_mvdr = apply_beamformer_bands(X_mix, mvdr_weights)[0]
    y_cbf = np.real(fb.synthesis(Y_cbf, length=x_mix.shape[-1]))
    y_mvdr = np.real(fb.synthesis(Y_mvdr, length=x_mix.shape[-1]))
    reanalyzed_mvdr = fb.analysis(y_mvdr)

    target_bin = int(round(freq / (fs / fft_size))) % fft_size
    mvdr_target_response = mvdr_weights[:, :, target_bin].T @ steering_target[:, :, target_bin]
    cbf_target_response = cbf_weights[:, :, target_bin].T @ steering_target[:, :, target_bin]
    mvdr_interferer_response = mvdr_weights[:, :, target_bin].T @ steering_interferer[:, :, target_bin]
    cbf_interferer_response = cbf_weights[:, :, target_bin].T @ steering_interferer[:, :, target_bin]

    cbf_err = y_cbf - reference
    mvdr_err = y_mvdr - reference

    print('covariance_source=interferer-only')
    print(f'integration_time_s={integration_time:.6f}')
    print(f'integration_rate_hz={rate:.6f}')
    print(f'integration_blocks={n_blocks}')
    print(f'forgetting_factor={forgetting_factor:.12f}')
    print(f'target_axis_az_deg={axis_az[0]:.12f}')
    print(f'target_axis_el_deg={axis_el[0]:.12f}')
    print(f'target_dir3d_x={dir3d_target[0, 0]:.12f}')
    print(f'target_dir3d_y={dir3d_target[1, 0]:.12f}')
    print(f'target_dir3d_z={dir3d_target[2, 0]:.12f}')
    print(f'target_bin={target_bin}')
    print(f'cbf_target_response_db={beam_response_rms_db(cbf_target_response[0, 0]):.12f}')
    print(f'mvdr_target_response_db={beam_response_rms_db(mvdr_target_response[0, 0]):.12f}')
    print(f'cbf_interferer_response_db={beam_response_rms_db(cbf_interferer_response[0, 0]):.12f}')
    print(f'mvdr_interferer_response_db={beam_response_rms_db(mvdr_interferer_response[0, 0]):.12f}')
    print(f'cbf_rms_time_error_to_target_reference={np.sqrt(np.mean(cbf_err ** 2)):.12e}')
    print(f'mvdr_rms_time_error_to_target_reference={np.sqrt(np.mean(mvdr_err ** 2)):.12e}')
    print(f'cbf_max_time_error_to_target_reference={np.max(np.abs(cbf_err)):.12e}')
    print(f'mvdr_max_time_error_to_target_reference={np.max(np.abs(mvdr_err)):.12e}')
    print(f'max_subband_reanalysis_error={np.max(np.abs(reanalyzed_mvdr - Y_mvdr)):.12e}')
    print(f'rms_subband_reanalysis_error={np.sqrt(np.mean(np.abs(reanalyzed_mvdr - Y_mvdr) ** 2)):.12e}')


if __name__ == '__main__':
    main()
