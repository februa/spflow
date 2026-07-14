"""非均一フィルタバンクのstreaming、完全再構成、ビーム形成結果を評価する。"""

# 非均一木構造では分割仕様と streaming 状態の組み合わせで挙動が大きく変わるため、
# 実運用に近い入出力条件を一式そろえて可視化・書き出しできる例として管理する。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

plt: Any
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 実行環境依存
    plt = None


from evaluations.nonuniform._array_input_support import load_array_design
from evaluations.nonuniform._matlab_fig_export import flush_matlab_figure_exports, stage_matlab_figure
from spflow.beamforming.directions import make_directions
from spflow.filterbank.causal_analytic_frontend import CausalAnalyticFrontendStreamer
from spflow.filterbank.daubechies_nonuniform_beamformer import DaubechiesNonuniformBeamformer
from spflow.filterbank.daubechies_nonuniform_streaming import DaubechiesNonuniformBeamformerStreaming
from spflow.filterbank.formal_nonuniform_streaming import (
    FormalNonuniformTreeStreamingAnalyzer,
    FormalNonuniformTreeStreamingSynthesizer,
)
from spflow.filterbank.formal_nonuniform_tree import FormalNonuniformTreeFilterBank


class NumpyEncoder(json.JSONEncoder):
    """numpy 型を summary JSON へ落とす encoder。"""

    def default(self, o: object) -> object:
        """numpy 配列と numpy scalar を JSON 化可能な Python 型へ変換する。"""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return super().default(o)




def configure_matplotlib_japanese() -> None:
    """matplotlib で日本語表示しやすいフォント設定を入れる。"""
    assert plt is not None
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'Yu Gothic',
        'Yu Gothic UI',
        'Meiryo',
        'MS Gothic',
        'IPAexGothic',
        'Noto Sans CJK JP',
        'DejaVu Sans',
    ]
    plt.rcParams['axes.unicode_minus'] = False

def require_matplotlib() -> None:
    """matplotlib 未導入環境を明示的に弾く。"""
    if plt is None:
        raise SystemExit('matplotlib is required to run this example.')
    configure_matplotlib_japanese()


def parse_float_list(text: str) -> list[float]:
    """カンマ区切り実数列を解析する。"""
    values = [item.strip() for item in text.split(',') if item.strip()]
    if not values:
        raise ValueError('at least one value must be provided.')
    return [float(item) for item in values]


def amplitude_from_db20(level_db20: float) -> float:
    """dB20 をトーンのピーク振幅へ変換する。"""
    return float(np.sqrt(2.0) * 10.0 ** (level_db20 / 20.0))


def direction_from_az_el(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """方位角・俯仰角から方向余弦を作る。"""
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    cos_el = np.cos(el)
    return np.array([
        np.cos(az) * cos_el,
        np.sin(az) * cos_el,
        np.sin(el),
    ], dtype=np.float32)


def positions_to_3d(array_design) -> np.ndarray:
    """1 次元配置を右舷左舷軸(y)上の 3 次元座標へ拡張する。"""
    return array_design.positions_3d(axis=1)


def build_sources(args: argparse.Namespace) -> list[dict]:
    """CLI 引数から音源設定列を構成する。"""
    freqs = parse_float_list(args.source_freqs_hz)
    levels = parse_float_list(args.source_levels_db20)
    azimuths = parse_float_list(args.source_azimuths_deg)
    elevations = parse_float_list(args.source_elevations_deg)
    phases = parse_float_list(args.source_phases_deg)
    n_source = len(freqs)

    def expand(values: list[float], name: str) -> list[float]:
        if len(values) == n_source:
            return values
        if len(values) == 1:
            return values * n_source
        raise ValueError(f'{name} must have length 1 or match source_freqs_hz.')

    levels = expand(levels, 'source_levels_db20')
    azimuths = expand(azimuths, 'source_azimuths_deg')
    elevations = expand(elevations, 'source_elevations_deg')
    phases = expand(phases, 'source_phases_deg')
    return [
        {
            'frequency_hz': freqs[index],
            'level_db20': levels[index],
            'azimuth_deg': azimuths[index],
            'elevation_deg': elevations[index],
            'phase_deg': phases[index],
        }
        for index in range(n_source)
    ]


def build_beam_grid(args: argparse.Namespace, target_source: dict) -> dict:
    """ビーム走査軸と表示対象俯仰を決める。"""
    elevation_grid = parse_float_list(args.elevation_grid_deg)
    directions, axis_az, axis_el = make_directions(
        az_min_deg=float(args.az_min_deg),
        az_max_deg=float(args.az_max_deg),
        el_min_deg=float(min(elevation_grid)),
        el_max_deg=float(max(elevation_grid)),
        n_beam_az_real=int(args.n_beam_az_real),
        n_beam_az_virtual=int(args.n_beam_az_virtual),
        n_beam_el=len(elevation_grid),
        array_side=str(args.array_side),
        el_preset_deg=elevation_grid,
    )
    if str(args.array_side) == 'left side':
        axis_az_signed = -np.abs(axis_az)
    else:
        axis_az_signed = axis_az.copy()
    display_el = target_source['elevation_deg'] if args.display_elevation_deg is None else float(args.display_elevation_deg)
    display_el_index = int(np.argmin(np.abs(axis_el - display_el)))
    return {
        'directions': directions,
        'axis_az': axis_az,
        'axis_az_signed': axis_az_signed,
        'axis_el': axis_el,
        'display_el_index': display_el_index,
        'display_el_deg': float(axis_el[display_el_index]),
    }


def build_leaf_frequency_dependent_steering(array_design, band_specs, beam_grid: dict, sound_speed: float) -> dict[str, np.ndarray]:
    """leaf 内 short-FFT 正側ビンごとの steering を band_id 辞書で作る。"""
    positions_3d = positions_to_3d(array_design)
    delays_s = positions_3d @ beam_grid['directions'] / sound_speed
    steering_by_band: dict[str, np.ndarray] = {}

    for band_index, spec in enumerate(band_specs):
        short_fft_size = int(round(spec.nominal_sample_rate_hz / spec.target_resolution_hz))
        positive_bin_count = short_fft_size // 2 + 1
        absolute_freqs_hz = spec.f_low_hz + np.arange(positive_bin_count, dtype=np.float32) * (spec.nominal_sample_rate_hz / short_fft_size)
        phase = -1j * 2.0 * np.pi * delays_s[:, :, np.newaxis] * absolute_freqs_hz[np.newaxis, np.newaxis, :]
        steering = np.exp(phase).astype(np.complex64)
        channel_window = array_design.shading_table[:, band_index].astype(np.float32)
        steering_by_band[spec.band_id] = steering * channel_window[:, np.newaxis, np.newaxis]

    return steering_by_band


def generate_scene(positions_3d_m: np.ndarray, sources: list[dict], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """多音源 + 白色雑音の多チャネル観測信号を作る。"""
    fs_hz = float(args.fs_hz)
    n_sample = int(round(fs_hz * float(args.duration_s)))
    axis_t = np.arange(n_sample, dtype=np.float32) / np.float32(fs_hz)
    mixture = np.zeros((positions_3d_m.shape[0], n_sample), dtype=np.float32)
    references = np.zeros((len(sources), n_sample), dtype=np.float32)

    for source_index, source in enumerate(sources):
        direction = direction_from_az_el(source['azimuth_deg'], source['elevation_deg'])
        delays_s = positions_3d_m @ direction / float(args.sound_speed)
        amplitude = amplitude_from_db20(source['level_db20'])
        phase_rad = np.deg2rad(source['phase_deg'])
        references[source_index] = amplitude * np.cos(2.0 * np.pi * source['frequency_hz'] * axis_t + phase_rad)
        for channel_index in range(positions_3d_m.shape[0]):
            mixture[channel_index] += amplitude * np.cos(
                2.0 * np.pi * source['frequency_hz'] * (axis_t - delays_s[channel_index]) + phase_rad
            )

    rng = np.random.default_rng(int(args.random_seed))
    noise_std = float(10.0 ** (float(args.noise_level_db20) / 20.0))
    noise = noise_std * rng.standard_normal(mixture.shape, dtype=np.float32)
    return (mixture + noise).astype(np.float32), references.astype(np.float32), noise.astype(np.float32)


def concat_or_empty(chunks: list[np.ndarray], prefix_shape: tuple[int, ...], dtype) -> np.ndarray:
    """空列を許容して配列を連結する。"""
    if not chunks:
        return np.zeros(prefix_shape + (0,), dtype=dtype)
    return np.concatenate(chunks, axis=-1)


def crop_analytic(analytic: np.ndarray, delay_samples: int, length: int) -> np.ndarray:
    """front-end 遅延分を落として元長へ切り出す。"""
    return np.asarray(analytic[..., delay_samples:delay_samples + length], dtype=np.complex64)


def adjust_boundary_indices(boundary_indices: list[int], delay_samples: int, length: int) -> np.ndarray:
    """analytic 出力境界を real 出力側の境界へ写像する。"""
    shifted = np.asarray(boundary_indices[:-1], dtype=np.int64) - int(delay_samples)
    return shifted[(shifted >= 0) & (shifted < max(0, length - 1))]


def max_boundary_jump_abs(signal: np.ndarray, boundary_indices: np.ndarray) -> float:
    """指定境界での隣接サンプル差分最大値を返す。"""
    if signal.shape[-1] < 2 or boundary_indices.size == 0:
        return 0.0
    diff_abs = np.abs(np.diff(signal, axis=-1))[..., boundary_indices]
    return float(np.max(diff_abs)) if diff_abs.size > 0 else 0.0


def stream_filterbank(filterbank: FormalNonuniformTreeFilterBank, x_real: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """非均一フィルタバンク単体の streaming 完全再構成を実行する。"""
    frontend_streamer = CausalAnalyticFrontendStreamer(filterbank.frontend)
    analyzer = FormalNonuniformTreeStreamingAnalyzer(filterbank)
    synthesizer = FormalNonuniformTreeStreamingSynthesizer(filterbank)
    chunks = []
    boundary_indices = []
    produced = 0

    def consume_analytic(analytic_chunk: np.ndarray) -> None:
        nonlocal produced
        if analytic_chunk.shape[-1] == 0:
            return
        for block in analyzer.process_analytic(analytic_chunk):
            root_chunk = synthesizer.process_block(block)
            if root_chunk.shape[-1] == 0:
                continue
            chunks.append(root_chunk)
            produced += int(root_chunk.shape[-1])
            boundary_indices.append(produced - 1)

    for start in range(0, x_real.shape[-1], chunk_size):
        consume_analytic(frontend_streamer.process(x_real[..., start:start + chunk_size]).samples)
    consume_analytic(frontend_streamer.flush().samples)
    for block in analyzer.flush():
        root_chunk = synthesizer.process_block(block)
        if root_chunk.shape[-1] == 0:
            continue
        chunks.append(root_chunk)
        produced += int(root_chunk.shape[-1])
        boundary_indices.append(produced - 1)

    analytic_output = concat_or_empty(chunks, x_real.shape[:-1], np.complex64)
    real_output = filterbank.frontend.recover_real(analytic_output, length=x_real.shape[-1])
    real_boundaries = adjust_boundary_indices(boundary_indices, filterbank.frontend.delay_samples, real_output.shape[-1])
    return crop_analytic(analytic_output, filterbank.frontend.delay_samples, x_real.shape[-1]), real_output, real_boundaries

def stream_beamformer(beamformer: DaubechiesNonuniformBeamformer, x_real: np.ndarray, chunk_size: int, offline_real: np.ndarray) -> dict:
    """real 入力を front-end + nonuniform beamformer streaming で処理する。"""
    frontend_streamer = CausalAnalyticFrontendStreamer(beamformer.filterbank.frontend)
    streaming = DaubechiesNonuniformBeamformerStreaming(beamformer=beamformer)
    chunks = []
    boundary_indices = []
    produced = 0

    def consume_analytic(analytic_chunk: np.ndarray) -> None:
        nonlocal produced
        if analytic_chunk.shape[-1] == 0:
            return
        root_chunk = streaming.process_analytic(analytic_chunk)
        if root_chunk.shape[-1] == 0:
            return
        chunks.append(root_chunk)
        produced += int(root_chunk.shape[-1])
        boundary_indices.append(produced - 1)

    for start in range(0, x_real.shape[-1], chunk_size):
        consume_analytic(frontend_streamer.process(x_real[..., start:start + chunk_size]).samples)
    consume_analytic(frontend_streamer.flush().samples)
    root_tail = streaming.flush()
    if root_tail.shape[-1] > 0:
        chunks.append(root_tail)
        produced += int(root_tail.shape[-1])
        boundary_indices.append(produced - 1)

    analytic_output = concat_or_empty(chunks, offline_real.shape[:-1], np.complex64)
    real_output = beamformer.filterbank.frontend.recover_real(analytic_output, length=x_real.shape[-1])
    real_boundaries = adjust_boundary_indices(boundary_indices, beamformer.filterbank.frontend.delay_samples, real_output.shape[-1])
    return {
        'analytic_output': crop_analytic(analytic_output, beamformer.filterbank.frontend.delay_samples, x_real.shape[-1]),
        'real_output': real_output,
        'boundary_indices': real_boundaries,
        'max_boundary_jump_abs': max_boundary_jump_abs(real_output - offline_real, real_boundaries),
        'max_abs_error_to_offline': float(np.max(np.abs(real_output - offline_real))),
        'rms_error_to_offline': float(np.sqrt(np.mean((real_output - offline_real) ** 2))),
        'streaming_object': streaming,
    }




def select_target_beam_index(beam_grid: dict, target_source: dict) -> int:
    """表示俯仰上で target に最も近いビーム index を返す。"""
    azimuth_index = int(np.argmin(np.abs(beam_grid['axis_az_signed'] - float(target_source['azimuth_deg']))))
    return azimuth_index * beam_grid['axis_el'].size + int(beam_grid['display_el_index'])


def find_leaf_band_index(beamformer: DaubechiesNonuniformBeamformer, frequency_hz: float) -> int:
    """指定周波数を含む leaf 帯域 index を返す。"""
    for band_index, spec in enumerate(beamformer.band_specs):
        if spec.f_low_hz <= frequency_hz < spec.f_high_hz:
            return band_index
    if np.isclose(frequency_hz, beamformer.band_specs[-1].f_high_hz):
        return len(beamformer.band_specs) - 1
    raise ValueError('display_frequency_hz is outside the nonuniform leaf bands.')


def compute_classic_beam_response(
    beamformer: DaubechiesNonuniformBeamformer,
    stream_result: dict,
    beam_grid: dict,
    target_source: dict,
    display_frequency_hz: float,
) -> tuple[np.ndarray, int, float]:
    """ターゲットビーム 1 本の正規化ビーム応答を返す。"""
    target_beam_index = select_target_beam_index(beam_grid, target_source)
    band_index = find_leaf_band_index(beamformer, display_frequency_hz)
    spec = beamformer.band_specs[band_index]
    processor = stream_result['streaming_object']._leaf_processors[spec.band_id]
    bin_width_hz = spec.nominal_sample_rate_hz / processor.output_fft_size
    bin_index = int(np.clip(round((display_frequency_hz - spec.f_low_hz) / bin_width_hz), 0, processor.output_inner_product_bin_count - 1))
    weights_bin = processor._current_weights_output_positive[:, target_beam_index, bin_index]
    steering_bin = processor._steering_short_positive[:, :, min(bin_index, processor._steering_short_positive.shape[-1] - 1)]
    response = np.sum(np.conjugate(weights_bin)[:, np.newaxis] * steering_bin, axis=0)
    response_grid = reshape_beams(response, beam_grid['axis_az_signed'].size, beam_grid['axis_el'].size)
    response_az = response_grid[:, beam_grid['display_el_index']]
    response_abs = np.abs(response_az)
    response_db = 20.0 * np.log10(np.maximum(response_abs / max(float(np.max(response_abs)), np.finfo(np.float32).tiny), np.finfo(np.float32).tiny))
    peak_azimuth_deg = float(beam_grid['axis_az_signed'][int(np.argmax(response_abs))])
    return np.asarray(response_db, dtype=np.float32), bin_index, peak_azimuth_deg

def amplitude_db(values: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    """振幅を dB へ変換する。"""
    eps = np.finfo(np.float32).tiny
    levels = 20.0 * np.log10(np.maximum(np.abs(np.asarray(values)), eps))
    return np.maximum(levels, floor_db)


def rfft_levels(real_signals: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """実波形の one-sided RMS スペクトルレベルを返す。"""
    signals = np.asarray(real_signals, dtype=np.float32)
    n_sample = max(1, signals.shape[-1])
    spectrum = np.fft.rfft(signals, axis=-1) / np.float32(n_sample)
    if spectrum.shape[-1] > 1:
        if n_sample % 2 == 0 and spectrum.shape[-1] > 2:
            spectrum[..., 1:-1] *= np.float32(np.sqrt(2.0))
        else:
            spectrum[..., 1:] *= np.float32(np.sqrt(2.0))
    freqs = np.fft.rfftfreq(signals.shape[-1], d=1.0 / fs_hz).astype(np.float32)
    levels = amplitude_db(spectrum)
    return freqs, np.asarray(levels, dtype=np.float32)


def compress_time_levels(analytic_signals: np.ndarray, fs_hz: float, max_bins: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """BTR 用に時間方向を間引いたレベル行列を返す。"""
    envelopes = np.abs(np.asarray(analytic_signals, dtype=np.complex64))
    n_sample = envelopes.shape[-1]
    step = max(1, int(np.ceil(n_sample / max_bins)))
    trimmed = max(step, (n_sample // step) * step)
    envelopes = envelopes[..., :trimmed]
    reshaped = envelopes.reshape(envelopes.shape[:-1] + (trimmed // step, step))
    levels = amplitude_db(np.mean(reshaped, axis=-1))
    times_s = (np.arange(levels.shape[-1], dtype=np.float32) * step) / np.float32(fs_hz)
    return np.asarray(levels, dtype=np.float32), times_s.astype(np.float32)


def reshape_beams(matrix: np.ndarray, n_az: int, n_el: int) -> np.ndarray:
    """beam 次元を `(n_az, n_el, ...)` へ戻す。"""
    return np.asarray(matrix).reshape((n_az, n_el) + np.asarray(matrix).shape[1:])


def save_figure(fig, base_path: Path, matlab_spec: dict | None = None) -> None:
    """図を PNG と MATLAB 互換 .fig で保存する。"""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    if matlab_spec is not None:
        stage_matlab_figure(base_path, matlab_spec)
    plt.close(fig)



def add_caption(fig, caption: str) -> None:
    """図下部へ caption を付ける。"""
    fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9)


def plot_input_spectrum(x_real: np.ndarray, fs_hz: float, output_dir: Path, max_channels: int = 5) -> None:
    """入力信号の周波数スペクトルを保存する。"""
    display_count = min(max_channels, x_real.shape[0])
    freqs, levels = rfft_levels(x_real[:display_count], fs_hz)
    fig, axes = plt.subplots(display_count, 1, figsize=(10, 2.4 * display_count), sharex=True)
    if display_count == 1:
        axes = [axes]

    matlab_axes = []
    for channel_index, axis in enumerate(axes):
        axis.plot(freqs, levels[channel_index], linewidth=1.0, label=f'ch{channel_index}')
        axis.set_ylabel(f'ch{channel_index} [dB20 RMS]')
        axis.grid(True, alpha=0.3)
        axis.legend(loc='upper right', fontsize=8)
        matlab_axes.append({
            'kind': 'line',
            'index': channel_index + 1,
            'xlabel': 'Frequency [Hz]' if channel_index == display_count - 1 else '',
            'ylabel': f'ch{channel_index} [dB20 RMS]',
            'title': '',
            'grid': True,
            'legend_location': 'northeast',
            'lines': [{
                'x': freqs,
                'y': levels[channel_index],
                'label': f'ch{channel_index}',
                'line_width': 1.0,
                'line_style': '-',
            }],
        })

    axes[-1].set_xlabel('Frequency [Hz]')
    figure_title = '入力信号の周波数スペクトル'
    caption = f'表示チャネル数={display_count} / 総チャネル数={x_real.shape[0]}'
    fig.suptitle(figure_title)
    add_caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.96))
    save_figure(
        fig,
        output_dir / 'input_spectrum',
        matlab_spec={
            'nrows': display_count,
            'ncols': 1,
            'suptitle': figure_title,
            'caption': caption,
            'axes': matlab_axes,
        },
    )


def plot_subband_spectrum(analysis_result, output_dir: Path, max_channels: int = 5) -> None:
    """サブバンド入力スペクトルを絶対周波数軸で保存する。"""
    display_count = min(max_channels, analysis_result.packets[0].complex_samples.shape[0])
    fig, axes = plt.subplots(display_count, 1, figsize=(11, 2.6 * display_count), sharex=True)
    if display_count == 1:
        axes = [axes]

    matlab_axes = []
    for channel_index, axis in enumerate(axes):
        matlab_lines = []
        for packet in analysis_result.packets:
            samples = np.asarray(packet.complex_samples[channel_index], dtype=np.complex64)
            n_fft = min(4096, max(256, 1 << int(np.ceil(np.log2(max(2, samples.shape[-1]))))))
            spectrum = np.fft.fft(samples, n=n_fft)[: n_fft // 2 + 1]
            abs_freq = packet.f_low_hz + np.arange(n_fft // 2 + 1, dtype=np.float32) * (packet.sample_rate_hz / n_fft)
            levels = amplitude_db(spectrum / max(1, samples.shape[-1]))
            axis.plot(abs_freq, levels, linewidth=0.9, label=packet.band_id)
            matlab_lines.append({
                'x': abs_freq,
                'y': levels,
                'label': packet.band_id,
                'line_width': 0.9,
                'line_style': '-',
            })
        axis.set_ylabel(f'ch{channel_index} [dB20 RMS]')
        axis.grid(True, alpha=0.3)
        axis.legend(loc='upper right', fontsize=7, ncol=2)
        matlab_axes.append({
            'kind': 'line',
            'index': channel_index + 1,
            'xlabel': 'Absolute Frequency [Hz]' if channel_index == display_count - 1 else '',
            'ylabel': f'ch{channel_index} [dB20 RMS]',
            'title': '',
            'grid': True,
            'legend_location': 'northeast',
            'lines': matlab_lines,
        })

    axes[-1].set_xlabel('Absolute Frequency [Hz]')
    figure_title = 'サブバンド入力時の信号スペクトル'
    caption = '各 leaf packet の複素 baseband スペクトルを絶対周波数へ写像して表示した。'
    fig.suptitle(figure_title)
    add_caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.96))
    save_figure(
        fig,
        output_dir / 'subband_input_spectrum',
        matlab_spec={
            'nrows': display_count,
            'ncols': 1,
            'suptitle': figure_title,
            'caption': caption,
            'axes': matlab_axes,
        },
    )


def prepare_method_plot_data(
    name: str,
    stream_result: dict,
    beam_grid: dict,
    fs_hz: float,
    display_frequency_hz: float,
    beamformer: DaubechiesNonuniformBeamformer,
    target_source: dict,
) -> dict:
    """方式別の走査応答・FRAZ・BTR 描画行列を作る。"""
    n_az = beam_grid['axis_az_signed'].size
    n_el = beam_grid['axis_el'].size
    freqs, fraz_levels_all = rfft_levels(stream_result['real_output'], fs_hz)
    fraz_levels = reshape_beams(fraz_levels_all, n_az, n_el)[:, beam_grid['display_el_index'], :]
    display_bin = int(np.argmin(np.abs(freqs - display_frequency_hz)))
    output_beam_levels = fraz_levels[:, display_bin]
    peak_beam_index = int(np.argmax(output_beam_levels))
    peak_azimuth_deg = float(beam_grid['axis_az_signed'][peak_beam_index])
    beam_scan_levels = np.asarray(output_beam_levels, dtype=np.float32)

    classical_beam_levels, beam_response_bin, beam_response_peak_azimuth_deg = compute_classic_beam_response(
        beamformer,
        stream_result,
        beam_grid,
        target_source,
        display_frequency_hz,
    )
    btr_levels_all, btr_times_s = compress_time_levels(stream_result['analytic_output'], fs_hz)
    btr_levels = reshape_beams(btr_levels_all, n_az, n_el)[:, beam_grid['display_el_index'], :].T
    btr_peak_indices = np.argmax(btr_levels, axis=1)
    btr_peak_azimuths_deg = beam_grid['axis_az_signed'][btr_peak_indices]
    btr_relative_levels = np.asarray(btr_levels - np.max(btr_levels, axis=1, keepdims=True), dtype=np.float32)
    return {
        'name': name,
        'freqs': freqs,
        'fraz_levels': fraz_levels,
        'beam_scan_levels': beam_scan_levels,
        'classical_beam_levels': classical_beam_levels,
        'display_bin': display_bin,
        'beam_response_bin': beam_response_bin,
        'beam_response_peak_azimuth_deg': beam_response_peak_azimuth_deg,
        'output_beam_levels': output_beam_levels,
        'output_real': stream_result['real_output'],
        'output_analytic': stream_result['analytic_output'],
        'boundary_indices': stream_result['boundary_indices'],
        'peak_beam_index': peak_beam_index,
        'peak_azimuth_deg': peak_azimuth_deg,
        'target_azimuth_deg': float(target_source['azimuth_deg']),
        'btr_levels': btr_levels,
        'btr_relative_levels': btr_relative_levels,
        'btr_peak_azimuths_deg': np.asarray(btr_peak_azimuths_deg, dtype=np.float32),
        'btr_times_s': btr_times_s,
        'max_boundary_jump_abs': stream_result['max_boundary_jump_abs'],
    }


def plot_beam_response(cbf_data: dict, mvdr_data: dict, beam_grid: dict, output_dir: Path) -> None:
    """CBF/MVDR の絶対走査応答を重ねて保存する。"""
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(beam_grid['axis_az_signed'], cbf_data['beam_scan_levels'], label='CBF', linewidth=1.5)
    axis.plot(beam_grid['axis_az_signed'], mvdr_data['beam_scan_levels'], label='MVDR', linewidth=1.5)
    axis.axvline(cbf_data['target_azimuth_deg'], color='black', linestyle=':', linewidth=1.0, label='Target azimuth')
    axis.set_xlabel('Azimuth [deg]')
    axis.set_ylabel('RMS Level [dB20]')
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure_title = '走査応答'
    axis.set_title(figure_title)
    caption = (
        f"表示俯仰={beam_grid['display_el_deg']:.2f} deg, "
        f"表示 bin={cbf_data['display_bin']}, 周波数={cbf_data['freqs'][cbf_data['display_bin']]:.2f} Hz, "
        f"target az={cbf_data['target_azimuth_deg']:.2f} deg, "
        f"CBF peak={cbf_data['peak_azimuth_deg']:.2f} deg, "
        f"MVDR peak={mvdr_data['peak_azimuth_deg']:.2f} deg, "
        f"one-sided RMS level 表示"
    )
    add_caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.96))
    y_min = float(min(np.min(cbf_data['beam_scan_levels']), np.min(mvdr_data['beam_scan_levels'])) - 3.0)
    y_max = float(max(np.max(cbf_data['beam_scan_levels']), np.max(mvdr_data['beam_scan_levels'])) + 3.0)
    save_figure(
        fig,
        output_dir / 'beam_response_overlay',
        matlab_spec={
            'nrows': 1,
            'ncols': 1,
            'suptitle': '',
            'caption': caption,
            'axes': [{
                'kind': 'line',
                'index': 1,
                'xlabel': 'Azimuth [deg]',
                'ylabel': 'RMS Level [dB20]',
                'title': figure_title,
                'grid': True,
                'legend_location': 'northeast',
                'lines': [
                    {
                        'x': beam_grid['axis_az_signed'],
                        'y': cbf_data['beam_scan_levels'],
                        'label': 'CBF',
                        'line_width': 1.5,
                        'line_style': '-',
                    },
                    {
                        'x': beam_grid['axis_az_signed'],
                        'y': mvdr_data['beam_scan_levels'],
                        'label': 'MVDR',
                        'line_width': 1.5,
                        'line_style': '-',
                    },
                    {
                        'x': np.array([cbf_data['target_azimuth_deg'], cbf_data['target_azimuth_deg']], dtype=np.float32),
                        'y': np.array([y_min, y_max], dtype=np.float32),
                        'label': 'Target azimuth',
                        'line_width': 1.0,
                        'line_style': ':',
                    },
                ],
            }],
        },
    )


def plot_fraz(method_data: dict, beam_grid: dict, output_dir: Path) -> None:
    """FRAZ を保存する。"""
    fig, axis = plt.subplots(figsize=(10, 5.5))
    image_data = method_data['fraz_levels'].T
    x_lim = [beam_grid['axis_az_signed'][0], beam_grid['axis_az_signed'][-1]]
    y_lim = [method_data['freqs'][0], method_data['freqs'][-1]]
    image = axis.imshow(image_data, aspect='auto', origin='lower', extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], cmap='viridis')
    axis.set_xlabel('Azimuth [deg]')
    axis.set_ylabel('Frequency [Hz]')
    figure_title = f"{method_data['name']} FRAZ"
    axis.set_title(figure_title)
    axis.axvline(method_data['peak_azimuth_deg'], color='white', linestyle='--', linewidth=1.0, label='Peak azimuth')
    axis.legend(loc='upper right')
    fig.colorbar(image, ax=axis, label='Level [dB]')
    caption = f"表示俯仰={beam_grid['display_el_deg']:.2f} deg, peak azimuth={method_data['peak_azimuth_deg']:.2f} deg"
    add_caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.96))
    save_figure(
        fig,
        output_dir / f"{method_data['name'].lower()}_fraz",
        matlab_spec={
            'nrows': 1,
            'ncols': 1,
            'suptitle': '',
            'caption': caption,
            'axes': [{
                'kind': 'image',
                'index': 1,
                'xlabel': 'Azimuth [deg]',
                'ylabel': 'Frequency [Hz]',
                'title': figure_title,
                'grid': False,
                'legend_location': 'northeast',
                'image': image_data,
                'x_lim': x_lim,
                'y_lim': y_lim,
                'colorbar_label': 'Level [dB]',
                'colormap': 'parula',
                'vlines': [{
                    'x': method_data['peak_azimuth_deg'],
                    'label': 'Peak azimuth',
                    'line_width': 1.0,
                    'line_style': '--',
                    'color_rgb': [1.0, 1.0, 1.0],
                }],
            }],
        },
    )


def plot_btr(method_data: dict, beam_grid: dict, output_dir: Path) -> None:
    """BTR を相対レベル表示で保存する。"""
    fig, axis = plt.subplots(figsize=(10, 5.5))
    image_data = method_data['btr_relative_levels']
    x_lim = [beam_grid['axis_az_signed'][0], beam_grid['axis_az_signed'][-1]]
    y_lim = [method_data['btr_times_s'][0], method_data['btr_times_s'][-1]]
    image = axis.imshow(image_data, aspect='auto', origin='lower', extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], cmap='viridis', vmin=-12.0, vmax=0.0)
    axis.set_xlabel('Azimuth [deg]')
    axis.set_ylabel('Time [s]')
    figure_title = f"{method_data['name']} BTR"
    axis.set_title(figure_title)
    axis.plot(method_data['btr_peak_azimuths_deg'], method_data['btr_times_s'], color='white', linestyle='--', linewidth=1.0, label='Peak track')
    axis.axvline(method_data['target_azimuth_deg'], color='black', linestyle=':', linewidth=1.0, label='Target azimuth')
    axis.legend(loc='upper right')
    fig.colorbar(image, ax=axis, label='Relative Level [dB]')
    caption = (
        f"表示俯仰={beam_grid['display_el_deg']:.2f} deg, "
        f"各時刻で最大ビームを 0 dB に正規化, "
        f"mean peak azimuth={float(np.mean(method_data['btr_peak_azimuths_deg'])):.2f} deg"
    )
    add_caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.96))
    save_figure(
        fig,
        output_dir / f"{method_data['name'].lower()}_btr",
        matlab_spec={
            'nrows': 1,
            'ncols': 1,
            'suptitle': '',
            'caption': caption,
            'axes': [{
                'kind': 'image',
                'index': 1,
                'xlabel': 'Azimuth [deg]',
                'ylabel': 'Time [s]',
                'title': figure_title,
                'grid': False,
                'legend_location': 'northeast',
                'image': image_data,
                'x_lim': x_lim,
                'y_lim': y_lim,
                'colorbar_label': 'Relative Level [dB]',
                'colormap': 'parula',
                'vlines': [
                    {
                        'x': method_data['target_azimuth_deg'],
                        'label': 'Target azimuth',
                        'line_width': 1.0,
                        'line_style': ':',
                        'color_rgb': [0.0, 0.0, 0.0],
                    },
                ],
                'polylines': [
                    {
                        'x': method_data['btr_peak_azimuths_deg'],
                        'y': method_data['btr_times_s'],
                        'label': 'Peak track',
                        'line_width': 1.0,
                        'line_style': '--',
                        'color_rgb': [1.0, 1.0, 1.0],
                    },
                ],
            }],
        },
    )


def plot_peak_waveform(method_data: dict, reference_waveform: np.ndarray, fs_hz: float, output_dir: Path) -> None:
    """ピーク方位ビームの再構成時間波形を保存する。"""
    fig, axis = plt.subplots(figsize=(11, 4.5))
    time_axis = np.arange(method_data['output_real'].shape[-1], dtype=np.float32) / np.float32(fs_hz)
    output_waveform = method_data['output_real'][method_data['peak_beam_index']]
    axis.plot(time_axis, output_waveform, linewidth=1.0, label=method_data['name'])
    axis.plot(time_axis, reference_waveform, linewidth=0.8, linestyle='--', label='Target reference')
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Amplitude')
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure_title = f"{method_data['name']} peak-direction waveform"
    axis.set_title(figure_title)
    caption = f"peak azimuth={method_data['peak_azimuth_deg']:.2f} deg, jump_abs={method_data['max_boundary_jump_abs']:.6e}"
    add_caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.96))
    save_figure(
        fig,
        output_dir / f"{method_data['name'].lower()}_peak_waveform",
        matlab_spec={
            'nrows': 1,
            'ncols': 1,
            'suptitle': '',
            'caption': caption,
            'axes': [{
                'kind': 'line',
                'index': 1,
                'xlabel': 'Time [s]',
                'ylabel': 'Amplitude',
                'title': figure_title,
                'grid': True,
                'legend_location': 'northeast',
                'lines': [
                    {
                        'x': time_axis,
                        'y': output_waveform,
                        'label': method_data['name'],
                        'line_width': 1.0,
                        'line_style': '-',
                    },
                    {
                        'x': time_axis,
                        'y': reference_waveform,
                        'label': 'Target reference',
                        'line_width': 0.8,
                        'line_style': '--',
                    },
                ],
            }],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """CLI 引数定義を返す。"""
    parser = argparse.ArgumentParser(description='非均一フィルタバンクの streaming・完全再構成・CBF/MVDR 結果を可視化する。')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts' / 'nonuniform_demo')
    parser.add_argument('--fs-hz', type=float, default=32768.0)
    parser.add_argument('--duration-s', type=float, default=1.0)
    parser.add_argument('--sound-speed', type=float, default=343.0)
    parser.add_argument('--chunk-size', type=int, default=257)
    parser.add_argument('--random-seed', type=int, default=1234)
    parser.add_argument('--noise-level-db20', type=float, default=-40.0)
    parser.add_argument('--source-freqs-hz', default='1536,4096')
    parser.add_argument('--source-levels-db20', default='0,-3')
    parser.add_argument('--source-azimuths-deg', default='20,55')
    parser.add_argument('--source-elevations-deg', default='0,0')
    parser.add_argument('--source-phases-deg', default='0,90')
    parser.add_argument('--target-source-index', type=int, default=0)
    parser.add_argument('--array-side', choices=['right side', 'left side', 'forward'], default='right side')
    parser.add_argument('--az-min-deg', type=float, default=0.0)
    parser.add_argument('--az-max-deg', type=float, default=180.0)
    parser.add_argument('--n-beam-az-real', type=int, default=61)
    parser.add_argument('--n-beam-az-virtual', type=int, default=0)
    parser.add_argument('--elevation-grid-deg', default='0')
    parser.add_argument('--display-elevation-deg', default=None)
    parser.add_argument('--display-frequency-hz', default=None)
    parser.add_argument('--candidate-name', default='daubechies_qmf_order4_taps8')
    parser.add_argument('--mvdr-integration-time', type=float, default=0.25)
    parser.add_argument('--mvdr-weight-update-period', type=float, default=0.05)
    parser.add_argument('--mvdr-diag-load', type=float, default=1e-3)
    parser.add_argument('--channel-positions-npy', default=None)
    parser.add_argument('--shading-table-npy', default=None)
    parser.add_argument('--n-dense-ch', type=int, default=24)
    parser.add_argument('--dense-spacing-m', type=float, default=0.01)
    parser.add_argument('--n-outer-pairs', type=int, default=4)
    parser.add_argument('--outer-spacing-m', type=float, default=0.04)
    parser.add_argument('--aperture-wavelengths', type=float, default=4.0)
    parser.add_argument('--min-active-ch', type=int, default=4)
    return parser


def main() -> None:
    """デモ全体を実行し、図と summary を保存する。"""
    require_matplotlib()
    args = build_parser().parse_args()
    sources = build_sources(args)
    if not 0 <= int(args.target_source_index) < len(sources):
        raise ValueError('target_source_index is out of range.')
    target_source = sources[int(args.target_source_index)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    array_design = load_array_design(
        channel_positions_path=args.channel_positions_npy,
        shading_table_path=args.shading_table_npy,
        fs_hz=float(args.fs_hz),
        sound_speed=float(args.sound_speed),
        n_dense_ch=int(args.n_dense_ch),
        dense_spacing_m=float(args.dense_spacing_m),
        n_outer_pairs=int(args.n_outer_pairs),
        outer_spacing_m=float(args.outer_spacing_m),
        aperture_wavelengths=float(args.aperture_wavelengths),
        min_active_ch=int(args.min_active_ch),
    )
    positions_3d_m = positions_to_3d(array_design)
    beam_grid = build_beam_grid(args, target_source)
    filterbank = FormalNonuniformTreeFilterBank.default_for_fs(float(args.fs_hz), candidate_name=str(args.candidate_name))
    steering = build_leaf_frequency_dependent_steering(array_design, filterbank.band_specs, beam_grid, float(args.sound_speed))
    x_real, references, noise = generate_scene(positions_3d_m, sources, args)

    offline_pr = filterbank.synthesize(filterbank.analyze_real(x_real))
    _, streaming_pr, pr_boundary = stream_filterbank(filterbank, x_real, int(args.chunk_size))
    analysis_result = filterbank.analyze_real(x_real)

    cbf = DaubechiesNonuniformBeamformer(
        fs_hz=float(args.fs_hz), candidate_name=str(args.candidate_name), array_design=array_design,
        beamformer_mode='cbf', output_path_mode='leaf_independent_one_sided', steering=steering,
    )
    mvdr = DaubechiesNonuniformBeamformer(
        fs_hz=float(args.fs_hz), candidate_name=str(args.candidate_name), array_design=array_design,
        beamformer_mode='mvdr', output_path_mode='leaf_independent_one_sided', steering=steering,
        integration_time=float(args.mvdr_integration_time), weight_update_period=float(args.mvdr_weight_update_period),
        diag_load=float(args.mvdr_diag_load),
    )
    cbf_offline = cbf.beamform_real(x_real)
    mvdr_offline = mvdr.beamform_real(x_real)
    cbf_stream = stream_beamformer(cbf, x_real, int(args.chunk_size), cbf_offline)
    mvdr_stream = stream_beamformer(mvdr, x_real, int(args.chunk_size), mvdr_offline)

    display_frequency_hz = target_source['frequency_hz'] if args.display_frequency_hz is None else float(args.display_frequency_hz)
    cbf_plot = prepare_method_plot_data('CBF', cbf_stream, beam_grid, float(args.fs_hz), display_frequency_hz, cbf, target_source)
    mvdr_plot = prepare_method_plot_data('MVDR', mvdr_stream, beam_grid, float(args.fs_hz), display_frequency_hz, mvdr, target_source)

    plot_input_spectrum(x_real, float(args.fs_hz), output_dir)
    plot_subband_spectrum(analysis_result, output_dir)
    plot_beam_response(cbf_plot, mvdr_plot, beam_grid, output_dir)
    plot_fraz(cbf_plot, beam_grid, output_dir)
    plot_fraz(mvdr_plot, beam_grid, output_dir)
    plot_btr(cbf_plot, beam_grid, output_dir)
    plot_btr(mvdr_plot, beam_grid, output_dir)
    plot_peak_waveform(cbf_plot, references[int(args.target_source_index)], float(args.fs_hz), output_dir)
    plot_peak_waveform(mvdr_plot, references[int(args.target_source_index)], float(args.fs_hz), output_dir)
    flush_matlab_figure_exports(strict=True)

    summary = {
        'sources': sources,
        'array': {
            'n_ch': array_design.n_ch,
            'n_band': array_design.n_band,
            'active_channel_counts_per_band': array_design.active_channel_counts_per_band(),
        },
        'display_frequency_hz': display_frequency_hz,
        'display_elevation_deg': beam_grid['display_el_deg'],
        'pr_metrics': {
            'offline_max_abs_error': float(np.max(np.abs(offline_pr - x_real))),
            'streaming_max_abs_error': float(np.max(np.abs(streaming_pr - x_real))),
            'streaming_rms_error': float(np.sqrt(np.mean((streaming_pr - x_real) ** 2))),
            'streaming_max_boundary_jump_abs': max_boundary_jump_abs(streaming_pr - offline_pr, pr_boundary),
        },
        'cbf_streaming': {
            'max_abs_error_to_offline': cbf_stream['max_abs_error_to_offline'],
            'rms_error_to_offline': cbf_stream['rms_error_to_offline'],
            'max_boundary_jump_abs': cbf_stream['max_boundary_jump_abs'],
            'beam_scan_peak_azimuth_deg': cbf_plot['peak_azimuth_deg'],
        },
        'mvdr_streaming': {
            'max_abs_error_to_offline': mvdr_stream['max_abs_error_to_offline'],
            'rms_error_to_offline': mvdr_stream['rms_error_to_offline'],
            'max_boundary_jump_abs': mvdr_stream['max_boundary_jump_abs'],
            'beam_scan_peak_azimuth_deg': mvdr_plot['peak_azimuth_deg'],
        },
        'noise_rms': float(np.sqrt(np.mean(noise ** 2))),
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, cls=NumpyEncoder), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False, cls=NumpyEncoder))


if __name__ == '__main__':
    main()
