"""T2a streaming評価の波形完全性、block境界、描画を担当する。

本モジュールは完成済みの入力・出力波形を受け取り、位相整列、分割／一括処理差、
per-bin RMS spectrum、診断PNGを生成する。scene生成、T2a重み設計、FIR化、
block逐次処理、MATLAB係数読込は責務に含めない。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]


class T2aWaveformScenario(Protocol):
    """波形評価が参照するscenario設定の最小契約を表す。

    入力はsampling、target tone、training、FFT、block条件であり、出力は持たない。
    scene生成、重み設計、係数読込の設定は責務に含めない。
    """

    @property
    def fs_hz(self) -> float:
        """標本化周波数をHzで返す。"""
        ...

    @property
    def target_frequency_hz(self) -> float:
        """波形完全性を評価するtarget tone周波数をHzで返す。"""
        ...

    @property
    def training_duration_s(self) -> float:
        """波形評価から除外するtraining時間を秒で返す。"""
        ...

    @property
    def analysis_fft_size(self) -> int:
        """波形評価に必要な最小完成sample数を返す。"""
        ...

    @property
    def runtime_block_size(self) -> int:
        """境界診断対象のstreaming block長をsample数で返す。"""
        ...


@dataclass(frozen=True)
class WaveformIntegrityResult:
    """target-only入力と整相出力を位相整列して得た波形完全性を保持する。

    `reference_signal`と`phase_aligned_output`は共通評価区間のshape `[n_sample]`、
    振幅単位はinput RMS基準である。`phase_delay_samples_modulo_period`はtarget toneの
    1周期を法とする位相差のsample換算であり、絶対伝搬遅延ではない。

    本結果型は完成した観測値と描画配列だけを保持し、EBAE重み設計、streaming処理、
    合否判定は責務に含めない。信号処理上はtarget-only無歪性の診断段に位置づく。
    """

    analysis_start_sample: int
    analysis_stop_sample: int
    phase_delay_samples_modulo_period: float
    rms_delta_db: float
    correlation_after_phase_alignment: float
    residual_rms_db_re_input_rms: float
    reference_signal: FloatArray
    phase_aligned_output: FloatArray


def _one_sided_rms_spectrum(signal: FloatArray, fs_hz: float) -> tuple[FloatArray, FloatArray]:
    """実時間信号の片側per-bin RMS levelを計算する。

    Args:
        signal: 実信号。shape `[n_sample]`、振幅単位はinput RMS基準。
        fs_hz: 標本化周波数。単位はHz。

    Returns:
        `(frequency_hz, level_db)`。双方shape `[n_frequency]`。levelは
        `dB re input RMS`のper-bin RMSである。

    Raises:
        ValueError: 信号が1次元でない、2 sample未満、有限でない、またはfsが正でない場合。
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("signal must be a one-dimensional array with at least two samples.")
    if fs_hz <= 0.0 or not bool(np.all(np.isfinite(values))):
        raise ValueError("fs_hz must be positive and signal must be finite.")
    spectrum = np.fft.rfft(values)
    # 実信号の片側per-bin RMS powerは内側binだけ2|X/N|^2とする。
    power = np.abs(spectrum / float(values.size)) ** 2
    if values.size % 2 == 0 and power.size > 2:
        # 偶数長では末尾がNyquistなので2倍しない。
        power[1:-1] *= 2.0
    elif values.size % 2 == 1 and power.size > 1:
        # 奇数長rFFTにはNyquist binがないため、DC以外をすべて2倍する。
        power[1:] *= 2.0
    frequency_hz = np.fft.rfftfreq(values.size, d=1.0 / fs_hz)
    level_db = 10.0 * np.log10(np.maximum(power, np.finfo(np.float64).tiny))
    return (
        np.asarray(frequency_hz, dtype=np.float64),
        np.asarray(level_db, dtype=np.float64),
    )


def calculate_target_waveform_integrity(
    reference_signal: FloatArray,
    beam_output: ComplexArray,
    valid_mask: BoolArray,
    config: T2aWaveformScenario,
) -> WaveformIntegrityResult:
    """target-only出力を入力へ位相整列し、波形完全性を計算する。

    Args:
        reference_signal: 基準channelのtarget-only入力。shape `[n_sample]`、input RMS基準。
        beam_output: target待受beamのtarget-only出力。shape `[n_sample]`、input RMS基準。
        valid_mask: beam出力の完成sample。shape `[n_sample]`、Trueだけを評価する。
        config: fs、target周波数、training区間を与えるscenario条件。

    Returns:
        位相差、RMS差、位相整列後相関、残差level、および共通評価区間の波形。

    Raises:
        ValueError: shape不一致、評価区間不足、非有限値、target周波数が無効、または
            基準信号のtarget成分が数値床以下の場合。

    境界条件:
        重み設計へ使ったtraining区間と未完成FIR履歴は除外する。単一toneでは絶対遅延を
        1周期ごとに一意化できないため、位相遅延はtarget周期を法とするsample数で返す。
    """
    reference = np.asarray(reference_signal, dtype=np.float64)
    output = np.asarray(beam_output, dtype=np.complex128)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if reference.ndim != 1 or output.ndim != 1 or valid.ndim != 1:
        raise ValueError("waveform integrity inputs must be one-dimensional.")
    if reference.shape != output.shape or reference.shape != valid.shape:
        raise ValueError("reference_signal, beam_output, and valid_mask must share shape.")
    if not 0.0 < config.target_frequency_hz < config.fs_hz / 2.0:
        raise ValueError("target_frequency_hz must lie inside the positive Nyquist band.")
    evaluation_valid = valid.copy()
    training_sample_count = int(round(config.training_duration_s * config.fs_hz))
    evaluation_valid[:training_sample_count] = False
    indices = np.flatnonzero(evaluation_valid)
    if indices.size < config.analysis_fft_size:
        raise ValueError("too few completed post-training samples for waveform integrity.")
    start = int(indices[0])
    stop = int(indices[-1]) + 1
    if not bool(np.all(evaluation_valid[start:stop])):
        # 内部欠損を一つの連続系列としてFFTすると、不連続を方式の歪みと混同するため拒否する。
        raise ValueError("waveform integrity requires one contiguous completed interval.")
    reference_segment = reference[start:stop]
    output_segment = np.real(output[start:stop])
    if not bool(np.all(np.isfinite(reference_segment))) or not bool(
        np.all(np.isfinite(output_segment))
    ):
        raise ValueError("waveform integrity interval must contain finite values.")

    sample_index = np.arange(start, stop, dtype=np.float64)
    # exact target周波数への複素射影で、FFT bin丸めに依存せず入力・出力の位相差を求める。
    carrier = np.exp(-1j * 2.0 * np.pi * config.target_frequency_hz * sample_index / config.fs_hz)
    reference_phasor = 2.0 * np.mean(reference_segment * carrier)
    output_phasor = 2.0 * np.mean(output_segment * carrier)
    if abs(reference_phasor) <= np.finfo(np.float64).eps:
        raise ValueError("reference target component is below the numerical floor.")
    phase_delta_rad = float(np.angle(output_phasor / reference_phasor))
    phase_delay_samples = (
        phase_delta_rad * config.fs_hz / (2.0 * np.pi * config.target_frequency_hz)
    )

    # 出力へ線形位相exp(-j2πfD/fs)を与え、target toneで観測した位相遅延Dを除去する。
    # 振幅応答は変えないため、整列後残差は位相遅延以外の波形変化を表す。
    output_spectrum = np.fft.rfft(output_segment)
    frequency_hz = np.fft.rfftfreq(output_segment.size, d=1.0 / config.fs_hz)
    phase_correction = np.exp(-1j * 2.0 * np.pi * frequency_hz * phase_delay_samples / config.fs_hz)
    phase_aligned = np.fft.irfft(output_spectrum * phase_correction, n=output_segment.size)
    input_rms = float(np.sqrt(np.mean(reference_segment**2)))
    output_rms = float(np.sqrt(np.mean(output_segment**2)))
    if input_rms <= np.finfo(np.float64).eps or output_rms <= np.finfo(np.float64).eps:
        raise ValueError("waveform integrity RMS must exceed the numerical floor.")
    correlation = float(np.corrcoef(reference_segment, phase_aligned)[0, 1])
    residual_rms = float(np.sqrt(np.mean((phase_aligned - reference_segment) ** 2)))
    return WaveformIntegrityResult(
        analysis_start_sample=start,
        analysis_stop_sample=stop,
        phase_delay_samples_modulo_period=phase_delay_samples,
        rms_delta_db=20.0 * np.log10(output_rms / input_rms),
        correlation_after_phase_alignment=correlation,
        residual_rms_db_re_input_rms=20.0 * np.log10(max(residual_rms, np.finfo(np.float64).tiny)),
        reference_signal=np.asarray(reference_segment, dtype=np.float64),
        phase_aligned_output=np.asarray(phase_aligned, dtype=np.float64),
    )


def calculate_streaming_reference_errors(
    streamed_output: ComplexArray,
    one_block_output: ComplexArray,
    valid_mask: BoolArray,
    block_size: int,
) -> tuple[float, float]:
    """分割streamingと一括blockの全体誤差・block境界近傍誤差を返す。

    Args:
        streamed_output: 分割streaming出力。shape `[n_sample]`、input RMS基準。
        one_block_output: 同じ係数の一括block出力。shape `[n_sample]`、input RMS基準。
        valid_mask: 完成sample。shape `[n_sample]`。
        block_size: 分割streamingのblock長。単位sample。

    Returns:
        `(全完成区間の最大絶対誤差, 各block境界前後1 sampleの最大絶対誤差)`。

    Raises:
        ValueError: 配列shape、block長、完成sample数が不正な場合。
    """
    streamed = np.asarray(streamed_output, dtype=np.complex128)
    one_block = np.asarray(one_block_output, dtype=np.complex128)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if streamed.ndim != 1 or streamed.shape != one_block.shape or streamed.shape != valid.shape:
        raise ValueError("streamed, one-block, and valid arrays must share one-dimensional shape.")
    if block_size <= 0 or not bool(np.any(valid)):
        raise ValueError(
            "block_size must be positive and valid_mask must contain completed samples."
        )
    difference = np.abs(streamed - one_block)
    overall_error = float(np.max(difference[valid]))
    boundary_mask = np.zeros(valid.shape, dtype=np.bool_)
    for boundary in range(block_size, streamed.size, block_size):
        # 境界直前・直後の2 sampleは、履歴更新漏れによる段差が最初に現れる位置である。
        boundary_mask[max(0, boundary - 1) : min(streamed.size, boundary + 1)] = True
    completed_boundary = boundary_mask & valid
    boundary_error = (
        float(np.max(difference[completed_boundary])) if bool(np.any(completed_boundary)) else 0.0
    )
    return overall_error, boundary_error


def select_diagnostic_zoom_bounds(
    valid_mask: BoolArray,
    block_size: int,
    minimum_sample: int,
) -> tuple[int, int, int]:
    """完成区間内のblock境界と拡大表示範囲を選ぶ。

    Args:
        valid_mask: 出力完成sample。shape `[n_sample]`。
        block_size: streaming block長。単位sample。
        minimum_sample: training等を除外する最小sample index。

    Returns:
        `(boundary_sample, zoom_start, zoom_stop)`。すべてsample index。

    Raises:
        ValueError: 完成したblock境界を含む表示範囲を確保できない場合。
    """
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if valid.ndim != 1 or block_size <= 0 or minimum_sample < 0:
        raise ValueError("valid_mask, block_size, and minimum_sample are invalid.")
    half_width = min(max(block_size // 2, 32), 256)
    for boundary in range(block_size, valid.size, block_size):
        start = boundary - half_width
        stop = boundary + half_width
        if start >= minimum_sample and stop <= valid.size and bool(np.all(valid[start:stop])):
            return boundary, start, stop
    raise ValueError("no completed streaming boundary is available for diagnostic zoom.")


def _draw_block_boundaries(
    axis: Any,
    start_sample: int,
    stop_sample: int,
    block_size: int,
    fs_hz: float,
) -> None:
    """時間波形axisへruntime block境界をsample時刻で描く。"""
    first_boundary = ((start_sample + block_size - 1) // block_size) * block_size
    for boundary in range(first_boundary, stop_sample, block_size):
        axis.axvline(boundary / fs_hz, color="tab:red", linestyle=":", alpha=0.75)


def write_input_waveform_diagnostics(
    output_path: Path,
    input_signal: FloatArray,
    reference_channel_index: int,
    zoom_start: int,
    zoom_stop: int,
    config: T2aWaveformScenario,
) -> None:
    """整相前mixed入力の全体波形、境界拡大波形、spectrumを保存する。

    Args:
        output_path: PNG保存先。
        input_signal: beamformer入力。shape `[n_ch,n_sample]`、input RMS基準。
        reference_channel_index: 表示する物理channel index。
        zoom_start: 拡大区間先頭。単位sample。
        zoom_stop: 拡大区間終端。単位sample、終端は含まない。
        config: fs、training時間、block長を与えるscenario条件。

    Returns:
        なし。全体波形、境界拡大、per-bin RMS spectrumをPNGへ保存する。

    Raises:
        ValueError: channelまたは拡大範囲が入力shape外の場合。
    """
    values = np.asarray(input_signal, dtype=np.float64)
    if values.ndim != 2 or not 0 <= reference_channel_index < values.shape[0]:
        raise ValueError("input_signal or reference_channel_index is invalid.")
    if not 0 <= zoom_start < zoom_stop <= values.shape[1]:
        raise ValueError("input waveform zoom range lies outside the signal.")
    waveform = values[reference_channel_index]
    time_s = np.arange(waveform.size, dtype=np.float64) / config.fs_hz
    spectrum_start = int(round(config.training_duration_s * config.fs_hz))
    frequency_hz, level_db = _one_sided_rms_spectrum(waveform[spectrum_start:], config.fs_hz)
    upper_db = float(np.max(level_db)) + 3.0
    lower_db = upper_db - 120.0

    figure, axes = plt.subplots(3, 1, figsize=(12.0, 10.0))
    axes[0].plot(time_s, waveform, linewidth=0.7)
    axes[0].set(
        title=f"Pre-beamforming mixed input: channel {reference_channel_index}",
        xlabel="Time [s]",
        ylabel="Amplitude [re input RMS]",
    )
    zoom_time_s = time_s[zoom_start:zoom_stop]
    axes[1].plot(zoom_time_s, waveform[zoom_start:zoom_stop], linewidth=1.0)
    _draw_block_boundaries(axes[1], zoom_start, zoom_stop, config.runtime_block_size, config.fs_hz)
    axes[1].set(
        title="Input waveform zoom; red dotted lines are runtime block boundaries",
        xlabel="Time [s]",
        ylabel="Amplitude [re input RMS]",
    )
    axes[2].plot(frequency_hz, np.maximum(level_db, lower_db))
    axes[2].set(
        title="Pre-beamforming mixed input spectrum after training interval",
        xlabel="Frequency [Hz]",
        ylabel="Per-bin RMS Level [dB re input RMS]",
        xlim=(0.0, config.fs_hz / 2.0),
        ylim=(lower_db, upper_db),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_output_waveform_diagnostics(
    output_path: Path,
    method_id: str,
    streamed_output: ComplexArray,
    one_block_output: ComplexArray,
    valid_mask: BoolArray,
    zoom_start: int,
    zoom_stop: int,
    config: T2aWaveformScenario,
) -> None:
    """target待受beamのmixed出力波形、境界、一括誤差、spectrumを保存する。

    Args:
        output_path: PNG保存先。
        method_id: 図へ記録する方式識別子。
        streamed_output: 分割streaming出力。shape `[n_sample]`、input RMS基準。
        one_block_output: 同じ完成係数の一括出力。shape `[n_sample]`、input RMS基準。
        valid_mask: 完成sample。shape `[n_sample]`。
        zoom_start: 境界拡大区間の先頭sample index。
        zoom_stop: 境界拡大区間の終端sample index。終端は含まない。
        config: fs、training時間、block長を与えるscenario条件。

    Returns:
        なし。PNGへ全体波形、境界拡大、一括差、per-bin RMS spectrumを保存する。

    Raises:
        ValueError: 配列shape、拡大範囲、または完成spectrum区間が不正な場合。
    """
    streamed = np.asarray(streamed_output, dtype=np.complex128)
    one_block = np.asarray(one_block_output, dtype=np.complex128)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if streamed.ndim != 1 or streamed.shape != one_block.shape or streamed.shape != valid.shape:
        raise ValueError("output diagnostic arrays must share one-dimensional shape.")
    if not 0 <= zoom_start < zoom_stop <= streamed.size:
        raise ValueError("output waveform zoom range lies outside the signal.")
    real_output = np.real(streamed)
    plot_output = np.where(valid, real_output, np.nan)
    time_s = np.arange(streamed.size, dtype=np.float64) / config.fs_hz
    spectrum_start = max(zoom_start, int(round(config.training_duration_s * config.fs_hz)))
    completed = real_output[spectrum_start:]
    if not bool(np.all(valid[spectrum_start:])):
        raise ValueError("output spectrum interval contains incomplete samples.")
    frequency_hz, level_db = _one_sided_rms_spectrum(completed, config.fs_hz)
    upper_db = float(np.max(level_db)) + 3.0
    lower_db = upper_db - 120.0
    difference = np.real(streamed - one_block)

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    axes[0, 0].plot(time_s, plot_output, linewidth=0.7)
    axes[0, 0].set(
        title=f"Post-beamforming mixed output: {method_id}, target beam",
        xlabel="Time [s]",
        ylabel="Amplitude [re input RMS]",
    )
    axes[0, 1].plot(
        time_s[zoom_start:zoom_stop], real_output[zoom_start:zoom_stop], label="streaming"
    )
    axes[0, 1].plot(
        time_s[zoom_start:zoom_stop],
        np.real(one_block[zoom_start:zoom_stop]),
        linestyle="--",
        label="one block",
    )
    _draw_block_boundaries(
        axes[0, 1], zoom_start, zoom_stop, config.runtime_block_size, config.fs_hz
    )
    axes[0, 1].set(
        title="Output zoom at runtime block boundary",
        xlabel="Time [s]",
        ylabel="Amplitude [re input RMS]",
    )
    axes[0, 1].legend()
    axes[1, 0].plot(time_s[zoom_start:zoom_stop], difference[zoom_start:zoom_stop], color="tab:red")
    _draw_block_boundaries(
        axes[1, 0], zoom_start, zoom_stop, config.runtime_block_size, config.fs_hz
    )
    maximum_error = float(np.max(np.abs(difference[valid])))
    axes[1, 0].set(
        title=f"Streaming minus one-block reference; max |error|={maximum_error:.3g}",
        xlabel="Time [s]",
        ylabel="Error [re input RMS]",
    )
    axes[1, 1].plot(frequency_hz, np.maximum(level_db, lower_db))
    axes[1, 1].set(
        title="Post-beamforming mixed output spectrum after training interval",
        xlabel="Frequency [Hz]",
        ylabel="Per-bin RMS Level [dB re input RMS]",
        xlim=(0.0, config.fs_hz / 2.0),
        ylim=(lower_db, upper_db),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_target_waveform_integrity(
    output_path: Path,
    method_id: str,
    integrity: WaveformIntegrityResult,
    config: T2aWaveformScenario,
) -> None:
    """target-only入力と位相整列後出力の波形・spectrum・残差を保存する。

    Args:
        output_path: PNG保存先。
        method_id: 図へ記録する方式識別子。
        integrity: 位相整列済み完成結果。波形shape `[n_sample]`、input RMS基準。
        config: fsとtarget tone周波数を与えるscenario条件。

    Returns:
        なし。波形、残差、per-bin RMS spectrum、数値指標をPNGへ保存する。
    """
    reference = integrity.reference_signal
    aligned = integrity.phase_aligned_output
    sample_count = reference.size
    time_s = (
        integrity.analysis_start_sample + np.arange(sample_count, dtype=np.float64)
    ) / config.fs_hz
    # target toneを少なくとも8周期表示し、過密な全区間overlayで局所歪みを隠さない。
    period_samples = max(1, int(round(config.fs_hz / config.target_frequency_hz)))
    zoom_count = min(sample_count, max(8 * period_samples, 64))
    frequency_hz, input_level_db = _one_sided_rms_spectrum(reference, config.fs_hz)
    _, output_level_db = _one_sided_rms_spectrum(aligned, config.fs_hz)
    upper_db = max(float(np.max(input_level_db)), float(np.max(output_level_db))) + 3.0
    lower_db = upper_db - 120.0
    residual = aligned - reference

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    axes[0, 0].plot(time_s[:zoom_count], reference[:zoom_count], label="input target-only")
    axes[0, 0].plot(
        time_s[:zoom_count], aligned[:zoom_count], linestyle="--", label="phase-aligned output"
    )
    axes[0, 0].set(
        title=f"Target-only waveform integrity: {method_id}",
        xlabel="Time [s]",
        ylabel="Amplitude [re input RMS]",
    )
    axes[0, 0].legend()
    axes[0, 1].plot(time_s[:zoom_count], residual[:zoom_count], color="tab:red")
    axes[0, 1].set(
        title="Phase-aligned output minus input",
        xlabel="Time [s]",
        ylabel="Residual [re input RMS]",
    )
    axes[1, 0].plot(frequency_hz, np.maximum(input_level_db, lower_db), label="input")
    axes[1, 0].plot(
        frequency_hz,
        np.maximum(output_level_db, lower_db),
        linestyle="--",
        label="phase-aligned output",
    )
    axes[1, 0].set(
        title="Target-only input/output spectrum",
        xlabel="Frequency [Hz]",
        ylabel="Per-bin RMS Level [dB re input RMS]",
        xlim=(0.0, config.fs_hz / 2.0),
        ylim=(lower_db, upper_db),
    )
    axes[1, 0].legend()
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        "\n".join(
            (
                "phase delay modulo period: "
                f"{integrity.phase_delay_samples_modulo_period:.6g} sample",
                f"output/input RMS delta: {integrity.rms_delta_db:.6g} dB",
                "correlation after phase alignment: "
                f"{integrity.correlation_after_phase_alignment:.9f}",
                f"residual RMS: {integrity.residual_rms_db_re_input_rms:.6g} dB re input RMS",
            )
        ),
        va="top",
        family="monospace",
    )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
