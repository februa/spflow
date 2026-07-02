"""MVDR の安定性を周波数・アレイ条件で sweep するサンプル。"""

from __future__ import annotations

import argparse
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


DEFAULT_FREQS = [100.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0, 10000.0]


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


def design_spacing(fs: float, sound_speed: float) -> float:
    """目標周波数に対して所望開口となるセンサ間隔を設計する。"""
    return float(sound_speed / fs)


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


def make_array_geometry(*, n_ch: int, spacing_m: float, array_design: BandwiseArrayDesign | None) -> ArrayGeometry:
    """評価条件に応じたアレイ形状オブジェクトを構成する。"""
    if array_design is None:
        return LinearArray(n_ch=n_ch, spacing=spacing_m)
    return ExplicitArrayGeometry(array_design.positions_3d(axis=0))


def render_scene(
    *,
    fs: float,
    freq: float,
    n_samples: int,
    n_ch: int,
    spacing_m: float,
    sound_speed: float,
    target_deg: float,
    signal_level_db20: float,
    target_el_deg: float = 0.0,
    interferer_deg: float | None = None,
    interferer_level_db20: float | None = None,
    interferer_el_deg: float = 0.0,
    include_target: bool = True,
    include_interferer: bool = True,
    array_design: BandwiseArrayDesign | None = None,
):
    """指定条件のシーンをレンダリングし、観測信号と参照情報を返す。"""
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=make_array_geometry(n_ch=n_ch, spacing_m=spacing_m, array_design=array_design),
    )
    target, target_component = make_source(receiver, target_deg, freq, signal_level_db20, elevation_deg=target_el_deg)
    sources = []
    interferer = None
    if include_target:
        sources.append(target)
    if include_interferer and interferer_deg is not None and interferer_level_db20 is not None:
        interferer, _ = make_source(
            receiver,
            interferer_deg,
            freq,
            interferer_level_db20,
            elevation_deg=interferer_el_deg,
        )
        sources.append(interferer)

    scene = Scene(sources=sources, ambient_fields=[], environment=FreeField(c=sound_speed))
    axis_t = np.arange(n_samples, dtype=float) / fs
    rendered = SceneRenderer().render(scene, receiver, axis_t)
    x = np.asarray(np.real(rendered), dtype=np.float32)
    reference = target_component.amplitude_value * np.cos(2.0 * np.pi * freq * axis_t)
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


def make_beam_steering(receiver: Receiver, source: AcousticSource, environment: FreeField, fft_size: int, fs: float):
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


def build_band_covariance(
    *,
    x_cov: np.ndarray,
    fft_size: int,
    integration_time: float,
    fs: float,
) -> tuple[np.ndarray, float, int]:
    """MVDR 設計に使う帯域共分散行列を構成する。"""
    fb = PolyphaseDFTFilterBank(fft_size=fft_size)
    X_cov = fb.analysis(x_cov)
    rate = fs / fft_size
    forgetting_factor = forgetting_factor_from_integration_time(integration_time, rate)
    n_blocks = integration_blocks_from_integration_time(integration_time, rate)
    if n_blocks > X_cov.shape[-1]:
        raise ValueError('integration_time requires more blocks than available in the simulated signal.')
    rxx = integrate_band_covariances(
        X_cov,
        forgetting_factor=forgetting_factor,
        normalization=fft_size,
        n_blocks=n_blocks,
    )
    return rxx, forgetting_factor, n_blocks


def build_array_design(
    *,
    n_ch: int,
    spacing_m: float,
    fft_size: int,
    fs: float,
    sound_speed: float,
    selector_mode: str,
    aperture_wavelengths: float,
    min_active_ch: int,
    dense_spacing_m: float | None,
    n_dense_ch: int | None,
) -> BandwiseArrayDesign:
    """帯域ごとの使用チャネルを持つアレイ設計を構成する。"""
    if selector_mode == 'full':
        return BandwiseArrayDesign.from_uniform_linear_centered_rectangular(
            n_ch=n_ch,
            spacing_m=spacing_m,
            n_band=fft_size,
            active_counts=np.full(fft_size, n_ch, dtype=np.int64),
        )
    if selector_mode == 'progressive':
        return BandwiseArrayDesign.from_uniform_linear_frequency_progressive_rectangular(
            n_ch=n_ch,
            spacing_m=spacing_m,
            fs=fs,
            n_band=fft_size,
            sound_speed=sound_speed,
            aperture_wavelengths=aperture_wavelengths,
            min_active_ch=min_active_ch,
            force_odd_counts=False,
        )
    if selector_mode == 'nested-progressive':
        if dense_spacing_m is None:
            dense_spacing_m = spacing_m / 4.0
        if n_dense_ch is None:
            n_dense_ch = max(8, n_ch // 2)
        n_outer_pairs = max(0, (n_ch - n_dense_ch) // 2)
        if n_dense_ch + 2 * n_outer_pairs != n_ch:
            raise ValueError('n_ch must equal n_dense_ch + 2*n_outer_pairs for nested-progressive mode.')
        return BandwiseArrayDesign.from_nested_sparse_linear_frequency_progressive(
            n_dense_ch=n_dense_ch,
            dense_spacing_m=dense_spacing_m,
            n_outer_pairs=n_outer_pairs,
            outer_spacing_m=spacing_m,
            fs=fs,
            n_band=fft_size,
            sound_speed=sound_speed,
            aperture_wavelengths=aperture_wavelengths,
            min_active_ch=min_active_ch,
        )
    raise ValueError('selector_mode must be full, progressive, or nested-progressive.')


def evaluate_frequency(
    *,
    fs: float,
    fft_size: int,
    freq: float,
    n_samples: int,
    n_ch: int,
    spacing_m: float,
    sound_speed: float,
    target_deg: float,
    signal_level_db20: float,
    integration_time: float | None,
    diag_load: float,
    interferer_deg: float | None,
    interferer_level_db20: float | None,
    covariance_source: str,
    selector_mode: str,
    aperture_wavelengths: float,
    min_active_ch: int,
    dense_spacing_m: float | None,
    n_dense_ch: int | None,
) -> dict[str, float | str]:
    """指定周波数条件でビームフォーマの応答指標を評価する。"""
    if freq >= fs / 2.0:
        return {
            'status': 'out_of_band',
            'freq_hz': freq,
            'fft_size': fft_size,
            'spacing_m': spacing_m,
        }

    array_design = build_array_design(
        n_ch=n_ch,
        spacing_m=spacing_m,
        fft_size=fft_size,
        fs=fs,
        sound_speed=sound_speed,
        selector_mode=selector_mode,
        aperture_wavelengths=aperture_wavelengths,
        min_active_ch=min_active_ch,
        dense_spacing_m=dense_spacing_m,
        n_dense_ch=n_dense_ch,
    )

    x_mix, reference, receiver, target, interferer, environment = render_scene(
        fs=fs,
        freq=freq,
        n_samples=n_samples,
        n_ch=n_ch,
        spacing_m=spacing_m,
        sound_speed=sound_speed,
        target_deg=target_deg,
        signal_level_db20=signal_level_db20,
        interferer_deg=interferer_deg,
        interferer_level_db20=interferer_level_db20,
        include_target=True,
        include_interferer=interferer_deg is not None and interferer_level_db20 is not None,
        array_design=array_design,
    )

    if covariance_source == 'mixture':
        x_cov = x_mix
    elif covariance_source == 'interferer-only':
        x_cov, _, _, _, _, _ = render_scene(
            fs=fs,
            freq=freq,
            n_samples=n_samples,
            n_ch=n_ch,
            spacing_m=spacing_m,
            sound_speed=sound_speed,
            target_deg=target_deg,
            signal_level_db20=signal_level_db20,
            interferer_deg=interferer_deg,
            interferer_level_db20=interferer_level_db20,
            include_target=False,
            include_interferer=True,
            array_design=array_design,
        )
    else:
        raise ValueError('covariance_source must be mixture or interferer-only.')

    fb = PolyphaseDFTFilterBank(fft_size=fft_size)
    X_mix = fb.analysis(x_mix)
    steering_target = make_beam_steering(receiver, target, environment, fft_size, fs)
    cbf_weights = design_cbf_weights_with_channel_window(steering_target, array_design.shading_table)

    rate = fs / fft_size
    effective_integration_time = (
        recommended_integration_time_for_independent_samples(n_ch, rate)
        if integration_time is None
        else integration_time
    )
    rxx, forgetting_factor, n_blocks = build_band_covariance(
        x_cov=x_cov,
        fft_size=fft_size,
        integration_time=effective_integration_time,
        fs=fs,
    )

    mvdr_weights = design_mvdr_weights_with_channel_window(
        rxx,
        steering_target,
        array_design.shading_table,
        diag_load=diag_load,
    )

    Y_cbf = apply_beamformer_bands(X_mix, cbf_weights)[0]
    Y_mvdr = apply_beamformer_bands(X_mix, mvdr_weights)[0]
    y_cbf = np.real(fb.synthesis(Y_cbf, length=x_mix.shape[-1]))
    y_mvdr = np.real(fb.synthesis(Y_mvdr, length=x_mix.shape[-1]))
    reanalyzed_mvdr = fb.analysis(y_mvdr)

    target_bin = int(round(freq / (fs / fft_size))) % fft_size
    cbf_target_response = cbf_weights[:, :, target_bin].conj().T @ steering_target[:, :, target_bin]
    mvdr_target_response = mvdr_weights[:, :, target_bin].conj().T @ steering_target[:, :, target_bin]

    cbf_interferer_db = float('nan')
    mvdr_interferer_db = float('nan')
    if interferer is not None:
        steering_interferer = make_beam_steering(receiver, interferer, environment, fft_size, fs)
        cbf_interferer_response = cbf_weights[:, :, target_bin].conj().T @ steering_interferer[:, :, target_bin]
        mvdr_interferer_response = mvdr_weights[:, :, target_bin].conj().T @ steering_interferer[:, :, target_bin]
        cbf_interferer_db = beam_response_rms_db(cbf_interferer_response[0, 0])
        mvdr_interferer_db = beam_response_rms_db(mvdr_interferer_response[0, 0])

    cbf_err = float(np.sqrt(np.mean((y_cbf - reference) ** 2)))
    mvdr_err = float(np.sqrt(np.mean((y_mvdr - reference) ** 2)))

    return {
        'status': 'ok',
        'freq_hz': freq,
        'fft_size': fft_size,
        'spacing_m': spacing_m,
        'target_bin': target_bin,
        'integration_time_s': effective_integration_time,
        'integration_blocks': float(n_blocks),
        'forgetting_factor': forgetting_factor,
        'covariance_source': covariance_source,
        'selector_mode': selector_mode,
        'active_channels_at_target_bin': float(array_design.active_channel_count(target_bin)),
        'active_aperture_m': array_design.active_aperture_m(target_bin),
        'active_min_spacing_m': array_design.minimum_spacing_m(target_bin),
        'active_alias_limit_hz': array_design.spatial_alias_limit_hz(target_bin, sound_speed),
        'cbf_target_db': beam_response_rms_db(cbf_target_response[0, 0]),
        'mvdr_target_db': beam_response_rms_db(mvdr_target_response[0, 0]),
        'cbf_interferer_db': cbf_interferer_db,
        'mvdr_interferer_db': mvdr_interferer_db,
        'cbf_rms_err': cbf_err,
        'mvdr_rms_err': mvdr_err,
        'mvdr_improves_target_err': float(mvdr_err < cbf_err),
        'max_reanalysis_err': float(np.max(np.abs(reanalyzed_mvdr - Y_mvdr))),
    }


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--fs', type=float, default=32768.0)
    parser.add_argument('--fft-sizes', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--freqs', type=float, nargs='+', default=DEFAULT_FREQS)
    parser.add_argument('--spacing-m', type=float, default=None)
    parser.add_argument('--n-ch', type=int, default=32)
    parser.add_argument('--n-samples', type=int, default=65536)
    parser.add_argument('--sound-speed', type=float, default=343.0)
    parser.add_argument('--target-deg', type=float, default=20.0)
    parser.add_argument('--signal-level-db20', type=float, default=0.0)
    parser.add_argument('--integration-time', type=float, default=None)
    parser.add_argument('--diag-load', type=float, default=1e-3)
    parser.add_argument('--interferer-deg', type=float, default=-30.0)
    parser.add_argument('--interferer-level-db20', type=float, default=0.0)
    parser.add_argument('--no-interferer', action='store_true')
    parser.add_argument('--covariance-source', choices=['mixture', 'interferer-only'], default='mixture')
    parser.add_argument('--selector-mode', choices=['full', 'progressive', 'nested-progressive'], default='nested-progressive')
    parser.add_argument('--aperture-wavelengths', type=float, default=4.0)
    parser.add_argument('--min-active-ch', type=int, default=4)
    parser.add_argument('--dense-spacing-m', type=float, default=None)
    parser.add_argument('--n-dense-ch', type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """MVDR 安定性 sweep の結果を表形式で表示する。"""
    args = parse_args()
    spacing_m = design_spacing(args.fs, args.sound_speed) if args.spacing_m is None else args.spacing_m
    interferer_deg = None if args.no_interferer else args.interferer_deg
    interferer_level_db20 = None if args.no_interferer else args.interferer_level_db20

    print(
        f'# fs={args.fs:.1f} Hz, n_ch={args.n_ch}, spacing_m={spacing_m:.9f}, '
        f'covariance_source={args.covariance_source}, selector_mode={args.selector_mode}, '
        f'aperture_wavelengths={args.aperture_wavelengths:.3f}, min_active_ch={args.min_active_ch}, '
        f'dense_spacing_m={args.dense_spacing_m}, n_dense_ch={args.n_dense_ch}'
    )
    print(
        '| fft_size | freq [Hz] | active_ch | active_aperture [m] | active_min_d [m] | alias_limit [Hz] | cbf_target_db | mvdr_target_db | '
        'cbf_interferer_db | mvdr_interferer_db | cbf_rms_err | mvdr_rms_err | improves | max_reanalysis_err | status |'
    )
    print('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---|')

    for fft_size in args.fft_sizes:
        for freq in args.freqs:
            row = evaluate_frequency(
                fs=args.fs,
                fft_size=fft_size,
                freq=freq,
                n_samples=args.n_samples,
                n_ch=args.n_ch,
                spacing_m=spacing_m,
                sound_speed=args.sound_speed,
                target_deg=args.target_deg,
                signal_level_db20=args.signal_level_db20,
                integration_time=args.integration_time,
                diag_load=args.diag_load,
                interferer_deg=interferer_deg,
                interferer_level_db20=interferer_level_db20,
                covariance_source=args.covariance_source,
                selector_mode=args.selector_mode,
                aperture_wavelengths=args.aperture_wavelengths,
                min_active_ch=args.min_active_ch,
                dense_spacing_m=args.dense_spacing_m,
                n_dense_ch=args.n_dense_ch,
            )
            if row['status'] != 'ok':
                print(
                    f"| {fft_size} | {freq:.0f} | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | out_of_band |"
                )
                continue
            print(
                f"| {row['fft_size']} | {row['freq_hz']:.0f} | {int(row['active_channels_at_target_bin'])} | {row['active_aperture_m']:.6f} | {row['active_min_spacing_m']:.6f} | {row['active_alias_limit_hz']:.3f} | "
                f"{row['cbf_target_db']:.6f} | {row['mvdr_target_db']:.6f} | "
                f"{row['cbf_interferer_db']:.6f} | {row['mvdr_interferer_db']:.6f} | "
                f"{row['cbf_rms_err']:.3e} | {row['mvdr_rms_err']:.3e} | "
                f"{'yes' if row['mvdr_improves_target_err'] else 'no'} | {row['max_reanalysis_err']:.3e} | ok |"
            )


if __name__ == '__main__':
    main()
