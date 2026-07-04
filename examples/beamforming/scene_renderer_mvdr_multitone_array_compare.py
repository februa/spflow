"""中央密・端疎アレイと一様アレイの MVDR 性能を比較するサンプル。"""

# scene_renderer で合成した観測波面を使い、設計した steering や重みが
# 単体式だけでなく波形再構成・指向性評価の流れ全体で破綻しないかを確認する例である。

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'vendor' / 'scene_renderer'))

from scene_renderer import (
    AcousticSource,
    ArrayGeometry,
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
    BandwiseArrayDesign,
    PolyphaseDFTFilterBank,
    apply_beamformer_bands,
    beam_response_rms_db,
    design_cbf_weights_with_channel_window,
    design_mvdr_weights_with_channel_window,
    forgetting_factor_from_integration_time,
    integrate_band_covariances,
    integration_blocks_from_integration_time,
    make_directions,
    recommended_integration_time_for_independent_samples,
)

FREQS = [100.0, 2000.0, 8000.0]


@dataclass(frozen=True)
class ExplicitArrayGeometry(ArrayGeometry):
    """明示的な 3 次元センサ座標を返す ArrayGeometry 実装。"""
    positions_xyz_m: np.ndarray

    def positions(self) -> np.ndarray:
        """保持している 3 次元センサ座標を `(n_ch, 3)` で返す。"""
        pos = np.asarray(self.positions_xyz_m, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError('positions_xyz_m must have shape (n_ch, 3).')
        return pos.copy()


def signal_peak_amplitude(level_db20: float) -> float:
    """振幅レベル dB20 を正弦波のピーク振幅へ変換する。"""
    return float(np.sqrt(2.0) * (10.0 ** (level_db20 / 20.0)))


def tone_amplitude(signal: np.ndarray, fs: float, freq: float) -> complex:
    """信号中の指定トーン成分の複素振幅を推定する。"""
    axis_t = np.arange(signal.size, dtype=float) / fs
    basis = np.exp(-1j * 2.0 * np.pi * freq * axis_t)
    return 2.0 * np.vdot(basis, signal) / signal.size


def rms_db_from_amplitude(amplitude: complex) -> float:
    """複素振幅を RMS 基準の dB20 値へ変換する。"""
    return float(20.0 * np.log10(max(np.abs(amplitude), 1e-15) / np.sqrt(2.0)))


def make_multitone_source(
    receiver: Receiver,
    bearing_deg: float,
    freqs_hz: list[float],
    level_db20: float,
    elevation_deg: float = 0.0,
):
    """複数トーンを持つ音源オブジェクトを生成する。"""
    components = [
        SourceComponent(
            spectrum=ToneSpectrum(freq),
            envelope=ConstantEnvelope(),
            amplitude=signal_peak_amplitude(level_db20),
        )
        for freq in freqs_hz
    ]
    return AcousticSource.from_relative_bearing(
        bearing_deg=bearing_deg,
        distance=1000.0,
        receiver_pose=receiver.trajectory.pose(0.0),
        components=components,
        elevation_deg=elevation_deg,
    )


def make_array_geometry(n_ch: int, spacing_m: float, array_design: BandwiseArrayDesign | None) -> ArrayGeometry:
    """評価条件に応じたアレイ形状オブジェクトを構成する。"""
    if array_design is None:
        return LinearArray(n_ch=n_ch, spacing=spacing_m)
    return ExplicitArrayGeometry(array_design.positions_3d(axis=0))


def render_multitone_scene(
    *,
    fs: float,
    n_samples: int,
    freqs_hz: list[float],
    n_ch: int,
    spacing_m: float,
    sound_speed: float,
    target_deg: float,
    interferer_deg: float,
    signal_level_db20: float,
    interferer_level_db20: float,
    array_design: BandwiseArrayDesign | None,
    include_target: bool = True,
    include_interferer: bool = True,
):
    """多周波トーンを含むシーンをレンダリングして観測信号を返す。"""
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=make_array_geometry(n_ch, spacing_m, array_design),
    )
    sources = []
    target = make_multitone_source(receiver, target_deg, freqs_hz, signal_level_db20)
    interferer = make_multitone_source(receiver, interferer_deg, freqs_hz, interferer_level_db20)
    if include_target:
        sources.append(target)
    if include_interferer:
        sources.append(interferer)

    scene = Scene(sources=sources, ambient_fields=[], environment=FreeField(c=sound_speed))
    axis_t = np.arange(n_samples, dtype=float) / fs
    rendered = SceneRenderer().render(scene, receiver, axis_t)
    x = np.asarray(np.real(rendered), dtype=np.float32)
    reference = np.sum(
        [signal_peak_amplitude(signal_level_db20) * np.cos(2.0 * np.pi * freq * axis_t) for freq in freqs_hz],
        axis=0,
    )
    return x, reference, receiver, target, interferer, scene.environment


def direction_from_source(receiver: Receiver, source: AcousticSource) -> np.ndarray:
    """音源位置から受信機座標系での到来方向ベクトルを求める。"""
    receiver_pose = receiver.trajectory.pose(0.0)
    source_pos = source.trajectory.position(0.0)
    direction_world = source_pos - receiver_pose.position_world
    direction_world = direction_world / np.linalg.norm(direction_world)
    return receiver_pose.world_vector_to_array(direction_world)


def steering_from_dir3d(receiver: Receiver, environment: FreeField, fft_size: int, fs: float, dir3d: np.ndarray) -> np.ndarray:
    """到来方向ベクトルから周波数依存 steering ベクトルを構成する。"""
    tau = receiver.array.positions() @ dir3d / environment.c
    freqs = np.fft.fftfreq(fft_size, d=1.0 / fs)
    steering = np.exp(-1j * 2.0 * np.pi * freqs[np.newaxis, :, np.newaxis] * tau[:, np.newaxis, :])
    return np.moveaxis(steering, -1, 1)


def make_beam_steering(receiver: Receiver, source: AcousticSource, environment: FreeField, fft_size: int, fs: float) -> np.ndarray:
    """指定方位に対応する steering ベクトルと走査軸を構成する。"""
    direction = direction_from_source(receiver, source)
    az_deg = float(np.rad2deg(np.arctan2(direction[1], direction[0])))
    el_deg = float(np.rad2deg(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
    dir3d, _, _ = make_directions(
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
    return steering_from_dir3d(receiver, environment, fft_size, fs, dir3d)


def build_mvdr_weights(
    X_cov: np.ndarray,
    steering_target: np.ndarray,
    shading_table: np.ndarray,
    fft_size: int,
    fs: float,
    integration_time: float,
) -> tuple[np.ndarray, float, int]:
    """共分散行列と steering から帯域別 MVDR 重みを設計する。"""
    rate = fs / fft_size
    alpha = forgetting_factor_from_integration_time(integration_time, rate)
    requested_blocks = integration_blocks_from_integration_time(integration_time, rate)
    n_blocks = min(requested_blocks, X_cov.shape[-1])
    effective_time = n_blocks / rate
    rxx = integrate_band_covariances(
        X_cov,
        forgetting_factor=alpha,
        normalization=fft_size,
        n_blocks=n_blocks,
    )
    weights = design_mvdr_weights_with_channel_window(
        rxx,
        steering_target,
        shading_table,
        diag_load=1e-3,
    )
    return weights, effective_time, n_blocks


def evaluate_case(name: str, array_design: BandwiseArrayDesign | None, n_ch: int, spacing_m: float) -> None:
    """1 つのアレイ構成条件で CBF と MVDR の性能を比較評価する。"""
    fs = 32768.0
    n_samples = 65536
    fft_size = 256
    sound_speed = 343.0
    target_deg = 20.0
    interferer_deg = -30.0
    signal_level_db20 = 0.0
    interferer_level_db20 = 0.0

    x_mix, reference, receiver, target, _, environment = render_multitone_scene(
        fs=fs,
        n_samples=n_samples,
        freqs_hz=FREQS,
        n_ch=n_ch,
        spacing_m=spacing_m,
        sound_speed=sound_speed,
        target_deg=target_deg,
        interferer_deg=interferer_deg,
        signal_level_db20=signal_level_db20,
        interferer_level_db20=interferer_level_db20,
        array_design=array_design,
        include_target=True,
        include_interferer=True,
    )
    x_int, _, _, _, _, _ = render_multitone_scene(
        fs=fs,
        n_samples=n_samples,
        freqs_hz=FREQS,
        n_ch=n_ch,
        spacing_m=spacing_m,
        sound_speed=sound_speed,
        target_deg=target_deg,
        interferer_deg=interferer_deg,
        signal_level_db20=signal_level_db20,
        interferer_level_db20=interferer_level_db20,
        array_design=array_design,
        include_target=False,
        include_interferer=True,
    )

    fb = PolyphaseDFTFilterBank(fft_size=fft_size)
    X_mix = fb.analysis(x_mix)
    X_int = fb.analysis(x_int)
    steering_target = make_beam_steering(receiver, target, environment, fft_size, fs)

    if array_design is None:
        shading_table = np.ones((n_ch, fft_size), dtype=np.float32)
        active_count = lambda band: n_ch
        active_aperture = lambda band: (n_ch - 1) * spacing_m
        min_spacing = lambda band: spacing_m
    else:
        shading_table = array_design.shading_table
        active_count = array_design.active_count
        active_aperture = array_design.active_aperture_m
        min_spacing = array_design.minimum_spacing_m

    cbf_weights = design_cbf_weights_with_channel_window(steering_target, shading_table)
    requested_time = recommended_integration_time_for_independent_samples(n_ch, fs / fft_size)
    mvdr_weights, effective_time, n_blocks = build_mvdr_weights(
        X_int,
        steering_target,
        shading_table,
        fft_size,
        fs,
        requested_time,
    )

    y_cbf = np.real(fb.synthesis(apply_beamformer_bands(X_mix, cbf_weights)[0], length=x_mix.shape[-1]))
    y_mvdr = np.real(fb.synthesis(apply_beamformer_bands(X_mix, mvdr_weights)[0], length=x_mix.shape[-1]))

    print(f'## {name}')
    print(f'requested_integration_time_s={requested_time:.6f}')
    print(f'effective_integration_time_s={effective_time:.6f}')
    print(f'integration_blocks={n_blocks}')
    print(f'total_cbf_rms_err={np.sqrt(np.mean((y_cbf - reference) ** 2)):.6e}')
    print(f'total_mvdr_rms_err={np.sqrt(np.mean((y_mvdr - reference) ** 2)):.6e}')
    print('| freq [Hz] | bin | active_ch | aperture [m] | min_d [m] | cbf_out [dBrms] | mvdr_out [dBrms] | cbf_target_resp [dB] | mvdr_target_resp [dB] |')
    print('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for freq in FREQS:
        band = int(round(freq / (fs / fft_size))) % fft_size
        cbf_out_amp = tone_amplitude(y_cbf, fs, freq)
        mvdr_out_amp = tone_amplitude(y_mvdr, fs, freq)
        cbf_target_resp = cbf_weights[:, :, band].conj().T @ steering_target[:, :, band]
        mvdr_target_resp = mvdr_weights[:, :, band].conj().T @ steering_target[:, :, band]
        print(
            f'| {freq:.0f} | {band} | {active_channel_count(band)} | {active_aperture(band):.6f} | {min_spacing(band):.6f} | '
            f'{rms_db_from_amplitude(cbf_out_amp):.3f} | {rms_db_from_amplitude(mvdr_out_amp):.3f} | '
            f'{beam_response_rms_db(cbf_target_resp[0,0]):.3f} | {beam_response_rms_db(mvdr_target_resp[0,0]):.3f} |'
        )
    print('')


def main() -> None:
    """アレイ構成ごとの多周波 MVDR 比較結果を表形式で表示する。"""
    small_n_ch = 32
    small_spacing_m = 0.04
    large_n_ch = 256
    large_spacing_m = 5.0
    large_design = BandwiseArrayDesign.from_nested_sparse_linear_frequency_progressive(
        n_dense_ch=64,
        dense_spacing_m=0.01,
        n_outer_pairs=(large_n_ch - 64) // 2,
        outer_spacing_m=large_spacing_m,
        fs=32768.0,
        n_band=256,
        sound_speed=343.0,
        aperture_wavelengths=100.0,
        min_active_ch=8,
    )

    evaluate_case('Small Array Full', None, small_n_ch, small_spacing_m)
    evaluate_case('Large Nested Sparse', large_design, large_n_ch, large_spacing_m)


if __name__ == '__main__':
    main()
