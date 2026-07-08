"""外部アレイ係数と scene_renderer 入力による fixed-delay diff-MVDR 評価。"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 実行環境依存
    plt = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from scene_renderer import (  # noqa: E402
    AcousticSource,
    ConstantEnvelope,
    FreeField,
    Receiver,
    Scene,
    SceneRenderer,
    SourceComponent,
    StaticPose,
    ToneSpectrum,
)
from scene_renderer.receiver import ArrayGeometry  # noqa: E402

from examples.beamforming.evaluate_external_fixed_delay_diff_mvdr_tap_tradeoff import (  # noqa: E402
    _arrival_steering,
)
from examples.beamforming.external_fixed_delay_diff_mvdr_inputs import (  # noqa: E402
    apply_frequency_shading_to_weights,
    load_complex_shading_matlab_raw,
    load_fractional_delay_filter_bank_matlab_raw,
    load_fractional_delay_filter_bank_npz,
    load_positions_matlab_raw,
    select_shading_for_frequencies,
)
from spflow.beamforming import (  # noqa: E402
    DelayTable,
    DifferenceCorrectionFIRDesigner,
    LoadedMVDRWeightDesigner,
    design_fixed_delay_fractional_weights_from_delay_table,
    make_directions,
)
from spflow.beamforming.diagnostic_plotting import require_matplotlib  # noqa: E402
from spflow.beamforming.time_delay import FractionalDelayFilterBank  # noqa: E402

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ExternalSceneSource:
    """scene_renderer に渡す 1 source 条件を表す。

    このクラスは、source 方位、周波数、線形ピーク振幅を保持する。
    入力は dB ではなく、評価 API の呼び出し前に変換済みの振幅である。
    scene 合成以外の MVDR 設計や結果集計は責務に含めない。
    信号処理上は、狭帯域 tone source の真値条件である。
    """

    label: str
    azimuth_deg: float
    frequency_hz: float
    peak_amplitude: float
    elevation_deg: float = 0.0


@dataclass(frozen=True)
class ExternalSceneEvaluationConfig:
    """scene_renderer 入力評価の scalar 設定を保持する。"""

    fs_hz: float = 32768.0
    duration_s: float = 1.0
    sound_speed_m_s: float = 1500.0
    n_beam_az_real: int = 121
    fir_taps: int = 128
    diagonal_loading_ratio: float = 1.0e-2
    random_seed: int = 1234


@dataclass(frozen=True)
class ExternalSceneMetricRow:
    """source×method の beam peak metric を保持する。"""

    source_label: str
    source_azimuth_deg: float
    source_frequency_hz: float
    method: str
    peak_azimuth_deg: float
    peak_error_deg: float
    peak_level_db_re_input_rms: float
    peak_delta_db_re_fixed: float
    level_at_nearest_source_beam_db_re_input_rms: float
    nearest_source_beam_azimuth_deg: float
    nearest_source_beam_error_deg: float
    q_reconstruction_rms_error: float


@dataclass(frozen=True)
class ExternalLevelNormalizationCheck:
    """SL/NL 入力正規化の周波数スペクトル確認条件を保持する。

    このクラスは、scene_renderer に渡した source level と noise level の期待値を、
    出力 PNG の水平線・垂直線として描くための設定である。
    beamforming 重み設計や採否判定は責務に含めない。
    信号処理上は、入力波形の生成直後に行う input/output level consistency 確認である。
    """

    source_frequencies_hz: tuple[float, ...]
    source_levels_db20: tuple[float, ...]
    noise_level_db20: float
    fs_hz: float


class ExternalArrayGeometry(ArrayGeometry):
    """scene_renderer の `ArrayGeometry` として任意の `[n_ch, 3]` 位置を渡す。"""

    def __init__(self, positions_m: NDArray[Any]) -> None:
        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError("positions_m must have shape [n_ch, 3].")
        if not bool(np.all(np.isfinite(positions))):
            raise ValueError("positions_m contains non-finite values.")
        self._positions_m = positions

    def positions(self) -> NDArray[np.float64]:
        """センサ位置を `[n_ch, 3]`、単位 m で返す。"""
        return self._positions_m.copy()


def db20_rms_to_tone_peak_amplitude(level_db20: float) -> float:
    """RMS 基準の dB20 を正弦波ピーク振幅へ変換する。

    scene_renderer の tone amplitude は時間波形のピーク振幅である。
    SL は RMS 振幅で指定するため、`A_peak = sqrt(2) * A_rms` として渡す。
    """
    return float(np.sqrt(2.0) * (10.0 ** (float(level_db20) / 20.0)))


def db20_noise_density_to_sample_rms_amplitude(level_db20: float, *, fs_hz: float) -> float:
    """NL を時間サンプルの白色雑音 RMS 振幅へ変換する。

    Args:
        level_db20: 片側振幅スペクトル密度として指定した NL。単位は dB re input RMS/sqrt(Hz)。
        fs_hz: sampling frequency。単位は Hz。

    Returns:
        channel ごとの白色雑音 sample 標準偏差。単位は input RMS。

    Raises:
        ValueError: `fs_hz` が正でない場合。

    境界条件:
        実数 white noise では片側帯域幅が `fs/2` になる。
        そのため `Amp_NL = 10^(NL/20) * sqrt(fs/2)` を時間波形へ与える。
    """
    if float(fs_hz) <= 0.0:
        raise ValueError("fs_hz must be positive.")
    return float((10.0 ** (float(level_db20) / 20.0)) * np.sqrt(float(fs_hz) / 2.0))


def tone_rms_level_db_from_fft_bin(
    fft_bin_value: NDArray[Any],
    *,
    n_fft: int,
) -> FloatArray:
    """片側 FFT の非 DC tone bin 値を RMS 振幅 dB へ変換する。

    Args:
        fft_bin_value: `np.fft.rfft` から取り出した複素 bin 値。
            beam response の場合 shape は `[n_beam]`、channel spectrum の場合 shape は `[n_ch]`。
        n_fft: FFT 点数。時間波形 sample 数と一致する。

    Returns:
        RMS 振幅 level。shape は `fft_bin_value` と同じ、単位は `dB re input RMS`。

    Raises:
        ValueError: `n_fft` が正でない場合。

    境界条件:
        この評価では source 周波数を非 DC の単一 tone として扱う。
        DC や Nyquist bin では片側 FFT の 2 倍補正が成立しないため、この関数の対象外である。
    """
    if int(n_fft) <= 0:
        raise ValueError("n_fft must be positive.")
    values = np.asarray(fft_bin_value, dtype=np.complex128)
    normalized_power = np.abs(values / float(n_fft)) ** 2
    # real tone の非 DC 正周波数 bin は片側だけで全 power の半分を持つ。
    # RMS 確認式は 10*log10(2*(abs(result/N_FFT)**2)) であり、
    # SL=0 dB re input RMS, A_peak=sqrt(2) の tone が 0 dB になる。
    rms_power = 2.0 * normalized_power
    return np.asarray(
        10.0 * np.log10(np.maximum(rms_power, np.finfo(np.float64).tiny)),
        dtype=np.float64,
    )


def one_sided_noise_density_level_db_from_signal(
    noise_signal: NDArray[Any],
    *,
    fs_hz: float,
) -> tuple[FloatArray, FloatArray, float]:
    """白色雑音波形から片側振幅スペクトル密度 level を推定する。

    Args:
        noise_signal: 雑音波形。shape は `[n_ch, n_sample]`、単位は input amplitude。
        fs_hz: sampling frequency。単位は Hz。

    Returns:
        `(frequency_hz, density_level_db, mean_density_level_db)`。
        `frequency_hz` と `density_level_db` の shape は `[n_rfft_bin - 2]`。
        DC と Nyquist は片側 2 倍補正の対象外なので除外する。

    Raises:
        ValueError: shape または `fs_hz` が不正な場合。
    """
    samples = np.asarray(noise_signal, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 4:
        raise ValueError("noise_signal must have shape [n_ch, n_sample>=4].")
    if float(fs_hz) <= 0.0:
        raise ValueError("fs_hz must be positive.")
    n_fft = int(samples.shape[1])
    frequency_hz = np.fft.rfftfreq(n_fft, d=1.0 / float(fs_hz))
    spectrum = np.fft.rfft(samples, axis=1)
    # spectrum shape: [n_ch, n_rfft_bin]。
    # 実数 white noise の片側 ASD power は 2*|X|^2/(N_FFT*fs) で推定する。
    density_power = 2.0 * (np.abs(spectrum[:, 1:-1]) ** 2) / (float(n_fft) * float(fs_hz))
    density_level_db = np.asarray(
        10.0 * np.log10(np.maximum(np.mean(density_power, axis=0), np.finfo(np.float64).tiny)),
        dtype=np.float64,
    )
    mean_density_level_db = float(
        10.0 * np.log10(max(float(np.mean(density_power)), np.finfo(np.float64).tiny))
    )
    return (
        np.asarray(frequency_hz[1:-1], dtype=np.float64),
        density_level_db,
        mean_density_level_db,
    )


def write_level_normalization_check_png(
    *,
    output_path: Path,
    arrays: dict[str, NDArray[Any]],
    check: ExternalLevelNormalizationCheck,
) -> None:
    """SL/NL 正規化を周波数スペクトル PNG として保存する。

    Args:
        output_path: PNG 保存先。
        arrays: `evaluate_external_scene_renderer_inputs` が返した描画前配列。
            `clean_signal` と `noise_signal` を使う。
        check: 期待する SL/NL と sampling frequency。

    Returns:
        なし。

    Raises:
        ValueError: 配列 shape や source 条件数が不正な場合。
        RuntimeError: matplotlib が利用できない場合。
    """
    if len(check.source_frequencies_hz) != len(check.source_levels_db20):
        raise ValueError("source frequency and level counts must match.")
    clean = np.asarray(arrays["clean_signal"], dtype=np.float64)
    noise = np.asarray(arrays["noise_signal"], dtype=np.float64)
    if clean.ndim != 2 or noise.ndim != 2 or clean.shape != noise.shape:
        raise ValueError("clean_signal and noise_signal must have the same [n_ch, n_sample] shape.")
    if clean.shape[1] < 4:
        raise ValueError("signals must contain at least 4 samples.")

    require_matplotlib()
    assert plt is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_fft = int(clean.shape[1])
    frequency_hz = np.fft.rfftfreq(n_fft, d=1.0 / float(check.fs_hz))
    clean_spectrum = np.fft.rfft(clean, axis=1)
    # clean_power shape: [n_rfft_bin]。
    # channel 位相差に依存しない入力 tone level を見るため、channel power を平均する。
    clean_power = 2.0 * np.mean(np.abs(clean_spectrum / float(n_fft)) ** 2, axis=0)
    clean_level_db = np.asarray(
        10.0 * np.log10(np.maximum(clean_power, np.finfo(np.float64).tiny)),
        dtype=np.float64,
    )
    noise_frequency_hz, noise_level_db, mean_noise_level_db = (
        one_sided_noise_density_level_db_from_signal(noise, fs_hz=float(check.fs_hz))
    )

    figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)
    tone_axis = axes[0]
    noise_axis = axes[1]
    tone_axis.plot(frequency_hz, clean_level_db, linewidth=1.0, color="tab:blue")
    for source_index, (source_frequency_hz, source_level_db) in enumerate(
        zip(check.source_frequencies_hz, check.source_levels_db20, strict=True)
    ):
        tone_axis.axvline(float(source_frequency_hz), color="tab:orange", linewidth=0.9)
        tone_axis.axhline(float(source_level_db), color="tab:green", linewidth=0.8, linestyle="--")
        tone_axis.text(
            float(source_frequency_hz),
            float(source_level_db),
            f"S{source_index + 1}: {source_level_db:.1f} dB",
            fontsize=8,
            rotation=90,
            va="bottom",
            ha="right",
        )
    tone_axis.set_ylabel("Tone RMS level [dB re input RMS]")
    tone_axis.set_title("Input normalization check")
    tone_axis.grid(True, alpha=0.3)

    noise_axis.plot(noise_frequency_hz, noise_level_db, linewidth=0.8, color="tab:purple")
    noise_axis.axhline(
        float(check.noise_level_db20),
        color="tab:red",
        linewidth=0.9,
        linestyle="--",
    )
    noise_axis.axhline(mean_noise_level_db, color="black", linewidth=0.8, linestyle=":")
    noise_axis.set_xlabel("Frequency [Hz]")
    noise_axis.set_ylabel("Noise ASD [dB re input RMS/sqrt(Hz)]")
    noise_axis.grid(True, alpha=0.3)
    noise_axis.text(
        0.99,
        0.95,
        f"target {check.noise_level_db20:.1f} dB, mean {mean_noise_level_db:.2f} dB",
        transform=noise_axis.transAxes,
        fontsize=9,
        ha="right",
        va="top",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _render_scene(
    *,
    array_positions_m: FloatArray,
    sources: tuple[ExternalSceneSource, ...],
    noise_sample_rms_amplitude: float,
    config: ExternalSceneEvaluationConfig,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """scene_renderer で source 信号を合成し、チャネル無相関雑音を加える。"""
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=ExternalArrayGeometry(array_positions_m),
    )
    acoustic_sources: list[AcousticSource] = []
    for source in sources:
        component = SourceComponent(
            spectrum=ToneSpectrum(float(source.frequency_hz)),
            envelope=ConstantEnvelope(),
            amplitude=float(source.peak_amplitude),
        )
        acoustic_sources.append(
            AcousticSource.from_relative_bearing(
                bearing_deg=float(source.azimuth_deg),
                distance=1000.0,
                receiver_pose=receiver.trajectory.pose(0.0),
                components=[component],
                elevation_deg=float(source.elevation_deg),
            )
        )
    axis_t = np.arange(int(round(config.duration_s * config.fs_hz)), dtype=np.float64) / float(
        config.fs_hz
    )
    scene = Scene(
        sources=acoustic_sources,
        ambient_fields=[],
        environment=FreeField(c=float(config.sound_speed_m_s)),
    )
    clean = np.asarray(np.real(SceneRenderer().render(scene, receiver, axis_t)), dtype=np.float64)
    rng = np.random.default_rng(int(config.random_seed))
    # API には NL から `sqrt(fs/2)` で変換済みの sample RMS 振幅を渡す。
    # ここでは channel ごとに独立な N(0, sigma^2) を加える。
    noise = float(noise_sample_rms_amplitude) * rng.standard_normal(clean.shape)
    return np.asarray(clean + noise, dtype=np.float64), clean, noise


def _design_weights(
    *,
    array_positions_m: FloatArray,
    shading_by_channel_bin: ComplexArray,
    shading_frequency_step_hz: float,
    fractional_delay_filter_bank: FractionalDelayFilterBank,
    sources: tuple[ExternalSceneSource, ...],
    config: ExternalSceneEvaluationConfig,
) -> tuple[dict[str, ComplexArray], FloatArray, FloatArray, dict[str, FloatArray]]:
    """source 周波数上の fixed / MVDR / diff-MVDR 重みを設計する。"""
    source_frequency_values = sorted({float(source.frequency_hz) for source in sources})
    frequencies_hz = np.asarray(source_frequency_values, dtype=np.float64)
    directions, axis_azimuth_deg, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=int(config.n_beam_az_real),
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    beam_directions = directions.T.astype(np.float64)
    delay_table = DelayTable.from_geometry(
        array_pos_m=array_positions_m,
        dir_cos=beam_directions,
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
        fractional_filter_bank=fractional_delay_filter_bank,
    )
    fixed = design_fixed_delay_fractional_weights_from_delay_table(
        delay_table,
        fractional_delay_filter_bank,
        frequencies_hz,
        fs_hz=float(config.fs_hz),
        average_channels=True,
    )
    shading_by_frequency = select_shading_for_frequencies(
        shading_by_channel_bin,
        float(shading_frequency_step_hz),
        frequencies_hz,
    )
    fixed = apply_frequency_shading_to_weights(fixed, shading_by_frequency)

    steering_by_beam = np.stack(
        [
            _arrival_steering(
                array_positions_m,
                float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
                frequencies_hz,
                float(config.sound_speed_m_s),
            )
            for direction in beam_directions
        ],
        axis=1,
    )
    source_steering_by_label = {
        source.label: _arrival_steering(
            array_positions_m,
            float(source.azimuth_deg),
            frequencies_hz,
            float(config.sound_speed_m_s),
        )
        for source in sources
    }
    covariance = np.zeros(
        (frequencies_hz.size, array_positions_m.shape[0], array_positions_m.shape[0]),
        dtype=np.complex128,
    )
    for frequency_index in range(frequencies_hz.size):
        covariance[frequency_index] = 1.0e-12 * np.eye(
            array_positions_m.shape[0], dtype=np.complex128
        )
        for source in sources:
            steering = source_steering_by_label[source.label][frequency_index]
            rms_amplitude = float(source.peak_amplitude) / np.sqrt(2.0)
            # R[k] = Σ sigma_s^2 a_s a_s^H。source は統計から除外しない。
            covariance[frequency_index] += (rms_amplitude**2) * np.outer(steering, steering.conj())
    mvdr = np.zeros_like(fixed)
    diff = np.zeros_like(fixed)
    q_error = np.zeros((frequencies_hz.size, fixed.shape[1]), dtype=np.float64)
    mvdr_designer = LoadedMVDRWeightDesigner(
        diagonal_loading_ratio=float(config.diagonal_loading_ratio)
    )
    diff_designer = DifferenceCorrectionFIRDesigner(
        fir_taps=int(config.fir_taps),
        frequencies_hz=frequencies_hz,
        fs_hz=float(config.fs_hz),
    )
    for beam_index in range(fixed.shape[1]):
        protected_steering = steering_by_beam[:, beam_index, :]
        mvdr_result = mvdr_designer.compute(covariance, protected_steering, fixed[:, beam_index, :])
        diff_result = diff_designer.compute(
            fixed[:, beam_index, :],
            mvdr_result.weights,
            protected_steering,
        )
        mvdr[:, beam_index, :] = mvdr_result.weights
        diff[:, beam_index, :] = diff_result.final_weight_freq
        q_error[:, beam_index] = np.sqrt(
            np.mean(np.abs(diff_result.diagnostics.q_reconstruction_error) ** 2, axis=1)
        )
    return (
        {"fixed_baseline": fixed, "mvdr_freq_ref": mvdr, f"diff_mvdr_fir{config.fir_taps}": diff},
        frequencies_hz,
        axis_azimuth_deg.astype(np.float64),
        {"q_reconstruction_rms_error": q_error},
    )


def evaluate_external_scene_renderer_inputs(
    *,
    array_positions_m: NDArray[Any],
    shading_by_channel_bin: NDArray[Any],
    shading_frequency_step_hz: float,
    fractional_delay_filter_bank: FractionalDelayFilterBank,
    sources: tuple[ExternalSceneSource, ...],
    noise_sample_rms_amplitude: float,
    config: ExternalSceneEvaluationConfig = ExternalSceneEvaluationConfig(),
) -> tuple[list[ExternalSceneMetricRow], dict[str, NDArray[Any]]]:
    """scene_renderer 入力を使い、source 周波数 BL metric を評価する。

    Args:
        array_positions_m: 実アレイ位置。shape は `[n_ch, 3]`、単位は m。
        shading_by_channel_bin: 複素 shading。shape は `[n_ch, n_shading_bin]`。
        shading_frequency_step_hz: shading bin 間隔。単位は Hz。
        fractional_delay_filter_bank: 小数遅延 FIR バンク。
        sources: source 条件。ピーク振幅は dB から変換済みの線形値。
        noise_sample_rms_amplitude: チャネル無相関雑音の sample RMS 振幅。NL から変換済みの線形値。
        config: 評価条件。

    Returns:
        metric 行と、描画・再確認用 ndarray 群。
    """
    positions = np.asarray(array_positions_m, dtype=np.float64)
    shading = np.asarray(shading_by_channel_bin, dtype=np.complex128)
    rendered, clean, noise = _render_scene(
        array_positions_m=positions,
        sources=sources,
        noise_sample_rms_amplitude=float(noise_sample_rms_amplitude),
        config=config,
    )
    weights_by_method, frequencies_hz, axis_azimuth_deg, diagnostics = _design_weights(
        array_positions_m=positions,
        shading_by_channel_bin=shading,
        shading_frequency_step_hz=float(shading_frequency_step_hz),
        fractional_delay_filter_bank=fractional_delay_filter_bank,
        sources=sources,
        config=config,
    )
    spectrum_freqs = np.fft.rfftfreq(rendered.shape[1], d=1.0 / float(config.fs_hz))
    n_fft = int(rendered.shape[1])
    # channel_spectrum shape: [n_ch, n_rfft_bin]。
    # ここでは後段で 10*log10(2*(abs(result/N_FFT)**2)) を使って RMS level を確認するため、
    # FFT bin は `N_FFT` で正規化せずに保持する。
    channel_spectrum = np.asarray(np.fft.rfft(rendered, axis=1), dtype=np.complex128)

    rows: list[ExternalSceneMetricRow] = []
    fixed_peak_by_source: dict[str, float] = {}
    for source in sources:
        design_frequency_index = int(np.argmin(np.abs(frequencies_hz - float(source.frequency_hz))))
        spectrum_index = int(np.argmin(np.abs(spectrum_freqs - float(source.frequency_hz))))
        nearest_source_beam_index = int(
            np.argmin(np.abs(axis_azimuth_deg - float(source.azimuth_deg)))
        )
        source_spectrum = channel_spectrum[:, spectrum_index]
        for method, weights in weights_by_method.items():
            # response[beam] = w[beam]^H X[f]。axis b は beam、c は channel を表す。
            response = np.einsum(
                "bc,c->b",
                weights[design_frequency_index].conj(),
                source_spectrum,
                optimize=True,
            )
            levels_db = tone_rms_level_db_from_fft_bin(response, n_fft=n_fft)
            peak_index = int(np.argmax(levels_db))
            peak_level = float(levels_db[peak_index])
            if method == "fixed_baseline":
                fixed_peak_by_source[source.label] = peak_level
            rows.append(
                ExternalSceneMetricRow(
                    source_label=source.label,
                    source_azimuth_deg=float(source.azimuth_deg),
                    source_frequency_hz=float(source.frequency_hz),
                    method=method,
                    peak_azimuth_deg=float(axis_azimuth_deg[peak_index]),
                    peak_error_deg=abs(
                        float(axis_azimuth_deg[peak_index]) - float(source.azimuth_deg)
                    ),
                    peak_level_db_re_input_rms=peak_level,
                    peak_delta_db_re_fixed=peak_level
                    - fixed_peak_by_source.get(source.label, peak_level),
                    level_at_nearest_source_beam_db_re_input_rms=float(
                        levels_db[nearest_source_beam_index]
                    ),
                    nearest_source_beam_azimuth_deg=float(axis_azimuth_deg[nearest_source_beam_index]),
                    nearest_source_beam_error_deg=abs(
                        float(axis_azimuth_deg[nearest_source_beam_index])
                        - float(source.azimuth_deg)
                    ),
                    q_reconstruction_rms_error=(
                        float(np.max(diagnostics["q_reconstruction_rms_error"][design_frequency_index]))
                        if method.startswith("diff_mvdr_fir")
                        else 0.0
                    ),
                )
            )
    arrays: dict[str, NDArray[Any]] = {
        "rendered_signal": rendered,
        "clean_signal": clean,
        "noise_signal": noise,
        "frequency_hz": frequencies_hz,
        "azimuth_deg": axis_azimuth_deg,
    }
    return rows, arrays


def write_scene_outputs(
    rows: list[ExternalSceneMetricRow],
    arrays: dict[str, NDArray[Any]],
    output_dir: Path,
    normalization_check: ExternalLevelNormalizationCheck | None = None,
) -> None:
    """scene_renderer 評価の CSV、NPZ、Markdown report、入力正規化 PNG を保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "external_scene_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        writer.writerows([row.__dict__ for row in rows])
    np.savez_compressed(
        output_dir / "external_scene_arrays.npz",
        rendered_signal=arrays["rendered_signal"],
        clean_signal=arrays["clean_signal"],
        noise_signal=arrays["noise_signal"],
        frequency_hz=arrays["frequency_hz"],
        azimuth_deg=arrays["azimuth_deg"],
    )
    normalization_png_name: str | None = None
    if normalization_check is not None:
        normalization_png_name = "external_level_normalization_check.png"
        write_level_normalization_check_png(
            output_path=output_dir / normalization_png_name,
            arrays=arrays,
            check=normalization_check,
        )
    lines = [
        "# 外部アレイ係数 + scene_renderer 入力評価",
        "",
        "## 成果物の定義",
        "",
        "- `external_scene_summary.csv`: source×method の peak 方位・level metric。",
        (
            "- `external_scene_arrays.npz`: scene_renderer が生成した channel 信号、"
            "clean/noise 成分、評価軸。"
        ),
        "- `external_level_normalization_check.png`: SL/NL 入力正規化の周波数スペクトル確認図。",
        "- level は `dB re input RMS` 相当のシミュレーション振幅基準である。",
        "",
        "## 結果要約",
        "",
    ]
    if normalization_png_name is not None:
        lines.extend(["", "## 入力正規化確認", "", f"- `{normalization_png_name}`"])
    for row in rows:
        lines.append(
            f"- `{row.source_label}` `{row.method}`: peak {row.peak_azimuth_deg:.3f} deg, "
            f"delta {row.peak_delta_db_re_fixed:.3f} dB re fixed, "
            f"q_err {row.q_reconstruction_rms_error:.3e}"
        )
    (output_dir / "external_scene_report.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    """CLI entrypoint。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coe-pos", type=Path, required=True)
    parser.add_argument("--coe-cbfshading", type=Path, required=True)
    parser.add_argument("--shading-df-hz", type=float, default=0.5)
    parser.add_argument("--fractional-delay-npz", type=Path)
    parser.add_argument("--fractional-delay-raw", type=Path)
    parser.add_argument("--fractional-delay-taps", type=int, default=128)
    parser.add_argument("--fractional-delay-frac-min", type=float, default=-0.5)
    parser.add_argument("--fractional-delay-frac-max", type=float, default=0.5)
    parser.add_argument("--source-azimuths-deg", default="60")
    parser.add_argument("--source-frequencies-hz", default="4096")
    parser.add_argument("--source-levels-db20", default="0")
    parser.add_argument("--noise-level-db20", type=float, default=-40.0)
    parser.add_argument("--fir-taps", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/beamforming/fixed_delay_diff_mvdr/external_scene_renderer"),
    )
    args = parser.parse_args()

    positions = load_positions_matlab_raw(args.coe_pos)
    shading = load_complex_shading_matlab_raw(args.coe_cbfshading, n_ch=int(positions.shape[0]))
    if args.fractional_delay_raw is not None:
        filter_bank = load_fractional_delay_filter_bank_matlab_raw(
            args.fractional_delay_raw,
            n_tap=int(args.fractional_delay_taps),
            frac_min=float(args.fractional_delay_frac_min),
            frac_max=float(args.fractional_delay_frac_max),
        )
    elif args.fractional_delay_npz is not None:
        filter_bank = load_fractional_delay_filter_bank_npz(args.fractional_delay_npz)
    else:
        raise ValueError("Specify --fractional-delay-raw or --fractional-delay-npz.")
    azimuths = _parse_float_tuple(str(args.source_azimuths_deg))
    frequencies = _parse_float_tuple(str(args.source_frequencies_hz))
    levels = _parse_float_tuple(str(args.source_levels_db20))
    if not (len(azimuths) == len(frequencies) == len(levels)):
        raise ValueError("source azimuth/frequency/level counts must match.")
    sources = tuple(
        ExternalSceneSource(
            label=f"S{index + 1}",
            azimuth_deg=azimuths[index],
            frequency_hz=frequencies[index],
            peak_amplitude=db20_rms_to_tone_peak_amplitude(levels[index]),
        )
        for index in range(len(azimuths))
    )
    config = ExternalSceneEvaluationConfig(fir_taps=int(args.fir_taps))
    normalization_check = ExternalLevelNormalizationCheck(
        source_frequencies_hz=frequencies,
        source_levels_db20=levels,
        noise_level_db20=float(args.noise_level_db20),
        fs_hz=float(config.fs_hz),
    )
    rows, arrays = evaluate_external_scene_renderer_inputs(
        array_positions_m=positions,
        shading_by_channel_bin=shading,
        shading_frequency_step_hz=float(args.shading_df_hz),
        fractional_delay_filter_bank=filter_bank,
        sources=sources,
        noise_sample_rms_amplitude=db20_noise_density_to_sample_rms_amplitude(
            float(args.noise_level_db20),
            fs_hz=float(config.fs_hz),
        ),
        config=config,
    )
    write_scene_outputs(rows, arrays, args.output_dir, normalization_check=normalization_check)
    print(args.output_dir / "external_scene_report.md")


if __name__ == "__main__":
    main()
