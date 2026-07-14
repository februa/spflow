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

from evaluations.beamforming.evaluate_external_fixed_delay_diff_mvdr_tap_tradeoff import (  # noqa: E402
    _arrival_steering,
)
from evaluations.beamforming.external_fixed_delay_diff_mvdr_inputs import (  # noqa: E402
    apply_frequency_shading_to_weights,
    load_complex_shading_matlab_raw,
    load_fractional_delay_filter_bank_matlab_raw,
    load_fractional_delay_filter_bank_npz,
    load_positions_matlab_raw,
    select_shading_for_frequencies,
)
from spflow.beamforming import (  # noqa: E402
    DelayTable,
    design_fixed_delay_fractional_weights_from_delay_table,
    make_directions,
)
from spflow.beamforming.time_delay import FractionalDelayFilterBank  # noqa: E402
from spflow.beamforming_evaluation.diagnostic_plotting import (  # noqa: E402
    centers_to_edges,
    require_matplotlib,
)

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
    source_azimuths_deg: tuple[float, ...] = ()


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


def _prepare_clean_tone_level_for_display(
    *,
    frequency_hz: NDArray[Any],
    clean_level_db: NDArray[Any],
    source_frequencies_hz: tuple[float, ...],
    source_levels_db20: tuple[float, ...],
    source_guard_bins: int = 1,
    relative_floor_db: float = -120.0,
) -> tuple[FloatArray, float, float, int]:
    """SL 確認図で表示する clean tone spectrum を作る。

    Args:
        frequency_hz: rFFT 周波数軸。shape は `[n_rfft_bin]`、単位は Hz。
        clean_level_db: clean 信号の RMS level。shape は `[n_rfft_bin]`、
            単位は `dB re input RMS`。
        source_frequencies_hz: source 周波数。要素数は source 数、単位は Hz。
        source_levels_db20: source RMS level。要素数は source 数、単位は `dB re input RMS`。
        source_guard_bins: source bin の近傍として表示判定から除外する片側 bin 数。
        relative_floor_db: 最大 source level から見た表示下限。単位は dB。

    Returns:
        `(display_level_db, display_floor_db, max_non_source_level_db, false_peak_count)`。
        `display_level_db` は表示下限未満を NaN にした `[n_rfft_bin]` 配列。

    Raises:
        ValueError: 周波数軸と level 軸の shape、source 数、または閾値が不正な場合。

    境界条件:
        scene_renderer は内部で complex64 の tone を生成するため、単一 tone でも
        -150 dB 付近に丸め残差が現れることがある。無音 bin を float tiny まで
        表示すると、この残差が複数の狭帯域信号に見えるため、SL 確認図では
        source peak から十分低い床以下を非表示にする。
    """
    freq = np.asarray(frequency_hz, dtype=np.float64)
    level = np.asarray(clean_level_db, dtype=np.float64)
    if freq.ndim != 1 or level.ndim != 1 or freq.shape != level.shape:
        raise ValueError("frequency_hz and clean_level_db must have the same [n_rfft_bin] shape.")
    if len(source_frequencies_hz) != len(source_levels_db20) or len(source_frequencies_hz) == 0:
        raise ValueError("source frequency and level counts must match and be non-empty.")
    if int(source_guard_bins) < 0:
        raise ValueError("source_guard_bins must be non-negative.")
    if float(relative_floor_db) >= 0.0:
        raise ValueError("relative_floor_db must be negative.")

    source_level_top_db = max(float(value) for value in source_levels_db20)
    display_floor_db = source_level_top_db + float(relative_floor_db)

    source_mask = np.zeros(freq.shape, dtype=np.bool_)
    guard_bins = int(source_guard_bins)
    for source_frequency_hz in source_frequencies_hz:
        source_bin = int(np.argmin(np.abs(freq - float(source_frequency_hz))))
        start_bin = max(0, source_bin - guard_bins)
        stop_bin = min(freq.size, source_bin + guard_bins + 1)
        # source bin 近傍は主信号の窓漏れや非整数 bin 条件を含むため、
        # false peak 判定から除外する。
        source_mask[start_bin:stop_bin] = True

    non_source_mask = np.logical_not(source_mask)
    if bool(np.any(non_source_mask)):
        max_non_source_level_db = float(np.max(level[non_source_mask]))
        false_peak_count = int(np.count_nonzero(level[non_source_mask] > display_floor_db))
    else:
        max_non_source_level_db = float("-inf")
        false_peak_count = 0

    display_level_db = level.copy()
    # 表示下限未満は「信号なし」として NaN にし、線で結ばない。
    # 実際に床を超える副ピークがある場合は、その bin だけ図に残る。
    display_level_db[display_level_db < display_floor_db] = np.nan
    return (
        np.asarray(display_level_db, dtype=np.float64),
        float(display_floor_db),
        max_non_source_level_db,
        false_peak_count,
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
    (
        clean_display_level_db,
        tone_display_floor_db,
        max_non_source_level_db,
        false_peak_count,
    ) = _prepare_clean_tone_level_for_display(
        frequency_hz=frequency_hz,
        clean_level_db=clean_level_db,
        source_frequencies_hz=check.source_frequencies_hz,
        source_levels_db20=check.source_levels_db20,
    )
    noise_frequency_hz, noise_level_db, mean_noise_level_db = (
        one_sided_noise_density_level_db_from_signal(noise, fs_hz=float(check.fs_hz))
    )

    figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)
    tone_axis = axes[0]
    noise_axis = axes[1]
    tone_axis.plot(frequency_hz, clean_display_level_db, linewidth=1.0, color="tab:blue")
    for source_index, (source_frequency_hz, source_level_db) in enumerate(
        zip(check.source_frequencies_hz, check.source_levels_db20, strict=True)
    ):
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
    source_caption = "; ".join(
        f"S{index + 1}: {frequency_hz:.1f} Hz, SL {level_db:.1f} dB"
        for index, (frequency_hz, level_db) in enumerate(
            zip(check.source_frequencies_hz, check.source_levels_db20, strict=True)
        )
    )
    tone_axis.set_title("Pre-beamforming input normalization check (clean/noise separated)")
    tone_axis.grid(True, alpha=0.3)
    tone_axis.set_ylim(
        tone_display_floor_db,
        max(float(value) for value in check.source_levels_db20) + 6.0,
    )
    tone_axis.text(
        0.99,
        0.95,
        (
            f"display floor {tone_display_floor_db:.1f} dB, "
            f"max non-source {max_non_source_level_db:.1f} dB, "
            f"visible false peaks {false_peak_count}"
        ),
        transform=tone_axis.transAxes,
        fontsize=9,
        ha="right",
        va="top",
    )

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
    figure.text(
        0.5,
        0.01,
        f"source frequency targets: {source_caption}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    figure.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def _one_sided_rms_level_db_from_spectrum(
    spectrum: NDArray[Any],
    *,
    n_fft: int,
) -> FloatArray:
    """rFFT の非 DC / 非 Nyquist bin を one-sided RMS level へ変換する。

    Args:
        spectrum: rFFT bin 値。shape は任意で、最後の値は未正規化 FFT 係数。
        n_fft: FFT 点数。単位は sample。

    Returns:
        `10log10(2*|X/N_FFT|^2)` で換算した level。shape は `spectrum` と同じ、
        単位は `dB re input RMS`。

    Raises:
        ValueError: `n_fft` が正でない場合。

    境界条件:
        この関数は非 DC / 非 Nyquist の片側表示用である。DC と Nyquist は 2 倍補正の
        対象外なので、呼び出し側で除外する。
    """
    if int(n_fft) <= 0:
        raise ValueError("n_fft must be positive.")
    values = np.asarray(spectrum, dtype=np.complex128)
    # 実数信号の正周波数 bin は負周波数側と対になるため、片側表示では power を 2 倍する。
    # FFT は未正規化なので、振幅へ戻すために N_FFT で割る。
    rms_power = 2.0 * (np.abs(values / float(n_fft)) ** 2)
    return np.asarray(
        10.0 * np.log10(np.maximum(rms_power, np.finfo(np.float64).tiny)),
        dtype=np.float64,
    )


def _finite_display_level_limits(
    level_db: NDArray[Any],
    *,
    top_padding_db: float = 3.0,
    minimum_span_db: float = 35.0,
    maximum_span_db: float = 90.0,
    lower_percentile: float = 1.0,
) -> tuple[float, float]:
    """有限値の level 配列から、線グラフ用の見やすい y 軸範囲を決める。

    Args:
        level_db: dB level 配列。任意 shape、単位は呼び出し側の図に従う。
        top_padding_db: 最大値の上側へ追加する余白。単位は dB。
        minimum_span_db: 少なくとも確保する y 軸幅。単位は dB。
        maximum_span_db: 深い null で図全体が潰れないようにする最大表示幅。単位は dB。
        lower_percentile: 下限候補に使う percentile。単位は percent。

    Returns:
        `(y_min, y_max)`。単位は dB。

    Raises:
        ValueError: 有限値が存在しない場合、または表示幅指定が不正な場合。

    境界条件:
        FIR や array null では -100 dB 以下の鋭い落ち込みが出る。
        その最小値まで y 軸を広げると mainlobe と sidelobe が読めなくなるため、
        下限は低 percentile と最大表示幅の大きい方で決める。
    """
    if float(top_padding_db) < 0.0:
        raise ValueError("top_padding_db must be non-negative.")
    if float(minimum_span_db) <= 0.0 or float(maximum_span_db) <= 0.0:
        raise ValueError("display span values must be positive.")
    if float(minimum_span_db) > float(maximum_span_db):
        raise ValueError("minimum_span_db must be <= maximum_span_db.")
    if not 0.0 <= float(lower_percentile) <= 100.0:
        raise ValueError("lower_percentile must be in [0, 100].")

    values = np.asarray(level_db, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("level_db does not contain finite values.")

    data_max_db = float(np.max(finite_values))
    y_max = data_max_db + float(top_padding_db)
    percentile_floor_db = float(np.percentile(finite_values, float(lower_percentile)))
    y_min = max(percentile_floor_db - 3.0, y_max - float(maximum_span_db))
    if y_max - y_min < float(minimum_span_db):
        y_min = y_max - float(minimum_span_db)
    return float(y_min), float(y_max)


def _finite_color_level_limits(
    level_db: NDArray[Any],
    *,
    dynamic_range_db: float = 80.0,
) -> tuple[float, float]:
    """FRAZ などの level 画像用 color scale を有限値から決める。

    Args:
        level_db: dB level 配列。shape は呼び出し側の図に従う。
        dynamic_range_db: 最大値から表示する下側 range。単位は dB。

    Returns:
        `(vmin, vmax)`。単位は dB。

    Raises:
        ValueError: 有限値が存在しない場合、または `dynamic_range_db` が正でない場合。

    境界条件:
        FRAZ は広帯域の雑音床と狭帯域ピークを同じ色で表示する。
        最小値まで color scale を広げるとサイドローブ差が見えなくなるため、
        最大値から一定 dB 幅を基本にする。
    """
    if float(dynamic_range_db) <= 0.0:
        raise ValueError("dynamic_range_db must be positive.")
    values = np.asarray(level_db, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("level_db does not contain finite values.")
    color_max = float(np.max(finite_values))
    color_min = max(float(np.min(finite_values)), color_max - float(dynamic_range_db))
    return float(color_min), float(color_max)


def write_rendered_input_spectrum_check_png(
    *,
    output_path: Path,
    arrays: dict[str, NDArray[Any]],
    check: ExternalLevelNormalizationCheck,
) -> None:
    """整相前の source+noise 入力スペクトルを PNG として保存する。

    Args:
        output_path: PNG 保存先。
        arrays: `rendered_signal` を含む描画前配列。`rendered_signal` shape は
            `[n_ch, n_sample]`、単位は input amplitude。
        check: source 周波数、SL、NL、sampling frequency。

    Returns:
        なし。

    Raises:
        ValueError: 入力 shape または source 条件が不正な場合。
        RuntimeError: matplotlib が利用できない場合。
    """
    if len(check.source_frequencies_hz) != len(check.source_levels_db20):
        raise ValueError("source frequency and level counts must match.")
    rendered = np.asarray(arrays["rendered_signal"], dtype=np.float64)
    if rendered.ndim != 2 or rendered.shape[0] == 0 or rendered.shape[1] < 4:
        raise ValueError("rendered_signal must have shape [n_ch, n_sample>=4].")

    require_matplotlib()
    assert plt is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_fft = int(rendered.shape[1])
    frequency_hz = np.fft.rfftfreq(n_fft, d=1.0 / float(check.fs_hz))
    rendered_spectrum = np.fft.rfft(rendered, axis=1)
    # rendered_spectrum[:, 1:-1] shape: [n_ch, n_positive_bin_without_dc_nyquist]。
    # channel ごとの位相差で source が打ち消されないよう、整相前確認では channel power を平均する。
    positive_frequency_hz = np.asarray(frequency_hz[1:-1], dtype=np.float64)
    rendered_level_db = np.asarray(
        10.0
        * np.log10(
            np.maximum(
                np.mean(
                    2.0 * (np.abs(rendered_spectrum[:, 1:-1] / float(n_fft)) ** 2),
                    axis=0,
                ),
                np.finfo(np.float64).tiny,
            )
        ),
        dtype=np.float64,
    )
    bin_width_hz = float(check.fs_hz) / float(n_fft)
    # NL は ASD 指定なので、FFT bin RMS level の期待値は NL + 10log10(Δf) になる。
    noise_bin_level_db = float(check.noise_level_db20) + 10.0 * np.log10(bin_width_hz)

    figure, axis = plt.subplots(figsize=(11.0, 4.5))
    axis.plot(positive_frequency_hz, rendered_level_db, linewidth=0.9, color="tab:blue")
    axis.axhline(noise_bin_level_db, color="tab:red", linewidth=0.9, linestyle="--")
    source_notes: list[str] = []
    for source_index, (source_frequency_hz, source_level_db) in enumerate(
        zip(check.source_frequencies_hz, check.source_levels_db20, strict=True)
    ):
        axis.axhline(float(source_level_db), color="tab:green", linewidth=0.8, linestyle=":")
        source_notes.append(
            f"S{source_index + 1}: {source_frequency_hz:.1f} Hz, {source_level_db:.1f} dB"
        )
    y_min, y_max = _finite_display_level_limits(
        rendered_level_db,
        top_padding_db=6.0,
        minimum_span_db=45.0,
        maximum_span_db=95.0,
    )
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("RMS Level [dB re input RMS]")
    axis.set_title("Pre-beamforming rendered input spectrum (signal + noise)")
    axis.grid(True, alpha=0.3)
    axis.text(
        0.99,
        0.95,
        (
            f"stage: before beamforming, noise bin target {noise_bin_level_db:.2f} dB\n"
            + "\n".join(source_notes)
        ),
        transform=axis.transAxes,
        fontsize=9,
        ha="right",
        va="top",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def _calculate_fixed_beamformed_fraz_level_db(
    *,
    array_positions_m: FloatArray,
    shading_by_channel_bin: ComplexArray,
    shading_frequency_step_hz: float,
    fractional_delay_filter_bank: FractionalDelayFilterBank,
    rendered_signal: FloatArray,
    config: ExternalSceneEvaluationConfig,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """fixed_baseline 整相後の FRAZ level を周波数領域で計算する。

    Args:
        array_positions_m: 実アレイ位置。shape は `[n_ch, 3]`、単位は m。
        shading_by_channel_bin: 複素 shading。shape は `[n_ch, n_shading_bin]`。
        shading_frequency_step_hz: shading bin 間隔。単位は Hz。
        fractional_delay_filter_bank: 小数遅延 FIR バンク。
        rendered_signal: source+noise の整相前入力。shape は `[n_ch, n_sample]`。
        config: sampling frequency、音速、ビーム数などの評価設定。

    Returns:
        `(frequency_hz, azimuth_deg, fraz_level_db)`。
        `frequency_hz` shape は `[n_positive_bin_without_dc_nyquist]`、単位は Hz。
        `azimuth_deg` shape は `[n_beam]`、単位は deg。
        `fraz_level_db` shape は `[n_beam, n_positive_bin_without_dc_nyquist]`、
        単位は `dB re input RMS`。

    Raises:
        ValueError: 入力 shape が不正な場合。
    """
    positions = np.asarray(array_positions_m, dtype=np.float64)
    shading = np.asarray(shading_by_channel_bin, dtype=np.complex128)
    rendered = np.asarray(rendered_signal, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("array_positions_m must have shape [n_ch, 3].")
    if rendered.ndim != 2 or rendered.shape[0] != positions.shape[0] or rendered.shape[1] < 4:
        raise ValueError("rendered_signal must have shape [n_ch, n_sample>=4].")

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
        array_pos_m=positions,
        dir_cos=beam_directions,
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
        fractional_filter_bank=fractional_delay_filter_bank,
    )

    n_fft = int(rendered.shape[1])
    full_frequency_hz = np.fft.rfftfreq(n_fft, d=1.0 / float(config.fs_hz))
    positive_bin_indices = np.arange(1, full_frequency_hz.size - 1, dtype=np.int64)
    max_shading_frequency_hz = float(shading.shape[1] - 1) * float(shading_frequency_step_hz)
    # fixed_baseline の整相後スペクトルは、実 shading 係数が定義されている範囲だけを描く。
    # shading 範囲外を外挿すると、実機係数を使った確認図ではなくなるため安全側で除外する。
    positive_bin_indices = positive_bin_indices[
        full_frequency_hz[positive_bin_indices] <= max_shading_frequency_hz
    ]
    if positive_bin_indices.size == 0:
        raise ValueError(
            "rendered_signal must provide positive frequency bins within shading range."
        )
    frequency_hz = np.asarray(full_frequency_hz[positive_bin_indices], dtype=np.float64)
    channel_spectrum = np.asarray(np.fft.rfft(rendered, axis=1), dtype=np.complex128)
    fraz_level_db = np.empty((axis_azimuth_deg.size, frequency_hz.size), dtype=np.float64)

    chunk_size = 256
    for start_index in range(0, positive_bin_indices.size, chunk_size):
        stop_index = min(start_index + chunk_size, positive_bin_indices.size)
        chunk_indices = positive_bin_indices[start_index:stop_index]
        chunk_frequency_hz = np.asarray(full_frequency_hz[chunk_indices], dtype=np.float64)
        fixed_weights = design_fixed_delay_fractional_weights_from_delay_table(
            delay_table,
            fractional_delay_filter_bank,
            chunk_frequency_hz,
            fs_hz=float(config.fs_hz),
            average_channels=True,
        )
        shading_by_frequency = select_shading_for_frequencies(
            shading,
            float(shading_frequency_step_hz),
            chunk_frequency_hz,
        )
        fixed_weights = apply_frequency_shading_to_weights(fixed_weights, shading_by_frequency)
        # fixed_weights shape: [n_chunk, n_beam, n_ch]。
        # channel_spectrum[:, chunk] shape は [n_ch, n_chunk]。
        # y[beam, k] = w[k, beam]^H X[:, k] により、整相後の周波数 bin を得る。
        beam_spectrum = np.einsum(
            "fbc,cf->bf",
            fixed_weights.conj(),
            channel_spectrum[:, chunk_indices],
            optimize=True,
        )
        fraz_level_db[:, start_index:stop_index] = _one_sided_rms_level_db_from_spectrum(
            beam_spectrum,
            n_fft=n_fft,
        )
    return (
        frequency_hz,
        axis_azimuth_deg.astype(np.float64),
        np.asarray(fraz_level_db, dtype=np.float64),
    )


def _calculate_beam_pattern_definition_example(
    *,
    array_positions_m: FloatArray,
    weights_by_frequency_beam_channel: ComplexArray,
    frequencies_hz: FloatArray,
    azimuth_deg: FloatArray,
    source: ExternalSceneSource,
    noise_sample_rms_amplitude: float,
    n_fft: int,
    config: ExternalSceneEvaluationConfig,
) -> dict[str, NDArray[Any]]:
    """beam pattern 定義例を計算する。

    Args:
        array_positions_m: 実アレイ位置。shape は `[n_ch, 3]`、単位は m。
        weights_by_frequency_beam_channel: 選択した方式の重み。shape は
            `[n_source_frequency, n_beam, n_ch]`。
        frequencies_hz: 重みを設計した周波数。shape は `[n_source_frequency]`、単位は Hz。
        azimuth_deg: 待ち受け方位軸。shape は `[n_beam]`、単位は deg。
        source: 定義例で使う source 条件。SL は `peak_amplitude` から復元する。
        noise_sample_rms_amplitude: NL から変換済みの時間波形 sample RMS。
        n_fft: FFT 点数。単位は sample。
        config: sampling frequency と音速を含む評価設定。

    Returns:
        beam pattern 用配列辞書。入力方位軸 shape は `[n_pattern_azimuth]`。

    Raises:
        ValueError: 入力 shape が不正な場合。

    信号処理上の位置づけ:
        beam response は source 方位を固定して待ち受けビームを並べる。一方 beam pattern は、
        1 つの待ち受け方位へ向けた重みを固定し、入力 source 方位を連続的に掃引する。
        そのため x 軸はビーム本数ではなく、任意に選んだ入力方位サンプリングである。
    """
    positions = np.asarray(array_positions_m, dtype=np.float64)
    weights = np.asarray(weights_by_frequency_beam_channel, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    waiting_azimuth = np.asarray(azimuth_deg, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("array_positions_m must have shape [n_ch, 3].")
    if weights.ndim != 3 or weights.shape[2] != positions.shape[0]:
        raise ValueError("fixed weights must have shape [n_freq, n_beam, n_ch].")
    if frequencies.ndim != 1 or frequencies.size != weights.shape[0]:
        raise ValueError("frequencies_hz must match the frequency axis of fixed weights.")
    if waiting_azimuth.ndim != 1 or waiting_azimuth.size != weights.shape[1]:
        raise ValueError("azimuth_deg must match the beam axis of fixed weights.")
    if int(n_fft) <= 0:
        raise ValueError("n_fft must be positive.")

    frequency_index = int(np.argmin(np.abs(frequencies - float(source.frequency_hz))))
    steering_beam_index = int(np.argmin(np.abs(waiting_azimuth - float(source.azimuth_deg))))
    pattern_frequency_hz = float(frequencies[frequency_index])
    steering_azimuth_deg = float(waiting_azimuth[steering_beam_index])
    pattern_input_azimuth_deg = np.linspace(0.0, 180.0, 721, dtype=np.float64)
    selected_weight = weights[frequency_index, steering_beam_index]
    source_rms_amplitude = float(source.peak_amplitude) / np.sqrt(2.0)
    source_level_db = 20.0 * np.log10(max(source_rms_amplitude, np.finfo(np.float64).tiny))

    # scene_renderer の bearing 方位を、解析 steering で使う array 方位へ変換する。
    # 右舷側 array の表示方位では displayed_az = 180 - array_az の関係になる。
    steering_by_input_azimuth = np.stack(
        [
            _arrival_steering(
                positions,
                180.0 - float(input_azimuth_deg),
                np.asarray([pattern_frequency_hz], dtype=np.float64),
                float(config.sound_speed_m_s),
            )[0]
            for input_azimuth_deg in pattern_input_azimuth_deg
        ],
        axis=0,
    )
    # response[input_az] = w(steering_az)^H a(input_az)。
    # beam pattern では重みを 1 方位に固定し、入力方位だけを掃引する。
    response = np.einsum(
        "c,ac->a", selected_weight.conj(), steering_by_input_azimuth, optimize=True
    )
    pattern_level_db = np.asarray(
        source_level_db + 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(np.float64).tiny)),
        dtype=np.float64,
    )

    bin_width_hz = float(config.fs_hz) / float(n_fft)
    noise_asd_power = (float(noise_sample_rms_amplitude) ** 2) / (float(config.fs_hz) / 2.0)
    # チャネル無相関雑音では、固定重み後の bin power は PSD*Δf*sum(|w_ch|^2)。
    # beam pattern 自体は deterministic な信号応答なので、NL は参考の期待雑音 floor として描く。
    output_noise_power = (
        noise_asd_power * bin_width_hz * float(np.sum(np.abs(selected_weight) ** 2))
    )
    noise_floor_db = 10.0 * np.log10(max(output_noise_power, np.finfo(np.float64).tiny))
    return {
        "beam_pattern_input_azimuth_deg": pattern_input_azimuth_deg,
        "beam_pattern_level_db": pattern_level_db,
        "beam_pattern_steering_azimuth_deg": np.asarray([steering_azimuth_deg], dtype=np.float64),
        "beam_pattern_source_frequency_hz": np.asarray([pattern_frequency_hz], dtype=np.float64),
        "beam_pattern_noise_floor_db": np.asarray([noise_floor_db], dtype=np.float64),
    }


def write_beam_response_definition_example_png(
    *,
    output_path: Path,
    arrays: dict[str, NDArray[Any]],
    check: ExternalLevelNormalizationCheck,
) -> None:
    """beam response の定義例 PNG を保存する。

    Args:
        output_path: PNG 保存先。
        arrays: `beamformed_fixed_frequency_hz`, `azimuth_deg`,
            `beamformed_fixed_fraz_level_db` を含む配列辞書。
        check: source 方位・周波数・SL/NL 条件。

    Returns:
        なし。

    Raises:
        ValueError: source 条件数または配列 shape が不正な場合。
        RuntimeError: matplotlib が利用できない場合。
    """
    if len(check.source_frequencies_hz) != 1 or len(check.source_azimuths_deg) != 1:
        raise ValueError("beam response definition example requires exactly one source.")
    response_frequency_hz = float(
        np.asarray(arrays["beam_response_frequency_hz"], dtype=np.float64)[0]
    )
    azimuth_deg = np.asarray(arrays["azimuth_deg"], dtype=np.float64)
    response_level_db = np.asarray(arrays["beam_response_level_db"], dtype=np.float64)
    if response_level_db.shape != azimuth_deg.shape:
        raise ValueError("beam_response_level_db must have shape [n_beam].")

    require_matplotlib()
    assert plt is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_azimuth_deg = float(check.source_azimuths_deg[0])
    peak_index = int(np.argmax(response_level_db))
    y_min, y_max = _finite_display_level_limits(response_level_db)

    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.plot(azimuth_deg, response_level_db, linewidth=1.3, color="tab:blue")
    axis.axvline(
        source_azimuth_deg, color="black", linestyle=":", linewidth=1.0, label="Signal azimuth"
    )
    axis.axvline(
        float(azimuth_deg[peak_index]),
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="Response peak",
    )
    axis.set_xlabel("Waiting beam azimuth [deg]")
    axis.set_ylabel("RMS Level [dB re input RMS]")
    axis.set_ylim(y_min, y_max)
    axis.set_title("Beam response definition: one input signal, all waiting beams")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.text(
        0.5,
        0.01,
        (
            f"source fixed at {source_azimuth_deg:.1f} deg, "
            f"fixed_baseline response frequency bin {response_frequency_hz:.1f} Hz"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    figure.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def write_beam_pattern_definition_example_png(
    *,
    output_path: Path,
    arrays: dict[str, NDArray[Any]],
    check: ExternalLevelNormalizationCheck,
) -> None:
    """beam pattern の定義例 PNG を保存する。

    Args:
        output_path: PNG 保存先。
        arrays: `beam_pattern_input_azimuth_deg`, `beam_pattern_level_db`,
            `beam_pattern_steering_azimuth_deg`, `beam_pattern_source_frequency_hz`,
            `beam_pattern_noise_floor_db` を含む配列辞書。
        check: source 方位・周波数・SL/NL 条件。

    Returns:
        なし。

    Raises:
        ValueError: source 条件数または配列 shape が不正な場合。
        RuntimeError: matplotlib が利用できない場合。
    """
    if len(check.source_frequencies_hz) != 1 or len(check.source_azimuths_deg) != 1:
        raise ValueError("beam pattern definition example requires exactly one source.")
    input_azimuth_deg = np.asarray(arrays["beam_pattern_input_azimuth_deg"], dtype=np.float64)
    pattern_level_db = np.asarray(arrays["beam_pattern_level_db"], dtype=np.float64)
    steering_azimuth_deg = float(
        np.asarray(arrays["beam_pattern_steering_azimuth_deg"], dtype=np.float64)[0]
    )
    pattern_frequency_hz = float(
        np.asarray(arrays["beam_pattern_source_frequency_hz"], dtype=np.float64)[0]
    )
    noise_floor_db = float(np.asarray(arrays["beam_pattern_noise_floor_db"], dtype=np.float64)[0])
    if input_azimuth_deg.ndim != 1 or pattern_level_db.shape != input_azimuth_deg.shape:
        raise ValueError("beam pattern arrays must have shape [n_pattern_azimuth].")

    require_matplotlib()
    assert plt is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    peak_index = int(np.argmax(pattern_level_db))
    y_min, y_max = _finite_display_level_limits(
        np.concatenate((pattern_level_db, np.asarray([noise_floor_db], dtype=np.float64)))
    )
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.plot(input_azimuth_deg, pattern_level_db, linewidth=1.3, color="tab:purple")
    axis.axvline(
        steering_azimuth_deg,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="Fixed steering azimuth",
    )
    axis.axvline(
        float(input_azimuth_deg[peak_index]),
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="Pattern peak",
    )
    axis.axhline(
        noise_floor_db,
        color="tab:gray",
        linestyle="-.",
        linewidth=0.9,
        label="Expected noise floor",
    )
    axis.set_xlabel("Input signal azimuth [deg]")
    axis.set_ylabel("RMS Level [dB re input RMS]")
    axis.set_ylim(y_min, y_max)
    axis.set_title("Beam pattern definition: fixed_baseline steering weight, swept input azimuth")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.text(
        0.5,
        0.01,
        (
            f"weight fixed at {steering_azimuth_deg:.2f} deg, "
            f"input azimuth swept independently at {pattern_frequency_hz:.1f} Hz"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    figure.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def write_fixed_beamformed_spectrum_check_png(
    *,
    output_path: Path,
    arrays: dict[str, NDArray[Any]],
    check: ExternalLevelNormalizationCheck,
) -> None:
    """fixed_baseline 整相後の BL/FL/FRAZ 確認図を保存する。

    Args:
        output_path: PNG 保存先。
        arrays: `beamformed_fixed_frequency_hz`, `azimuth_deg`,
            `beamformed_fixed_fraz_level_db` を含む描画前配列。
        check: source 周波数、方位、SL、NL、sampling frequency。

    Returns:
        なし。

    Raises:
        ValueError: 配列 shape または source 条件が不正な場合。
        RuntimeError: matplotlib が利用できない場合。
    """
    if len(check.source_frequencies_hz) != len(check.source_levels_db20):
        raise ValueError("source frequency and level counts must match.")
    if len(check.source_azimuths_deg) not in (0, len(check.source_frequencies_hz)):
        raise ValueError("source azimuth count must be zero or match source frequency count.")
    frequency_hz = np.asarray(arrays["beamformed_fixed_frequency_hz"], dtype=np.float64)
    azimuth_deg = np.asarray(arrays["azimuth_deg"], dtype=np.float64)
    fraz_level_db = np.asarray(arrays["beamformed_fixed_fraz_level_db"], dtype=np.float64)
    if frequency_hz.ndim != 1 or azimuth_deg.ndim != 1:
        raise ValueError("frequency and azimuth axes must be 1-D.")
    if fraz_level_db.shape != (azimuth_deg.size, frequency_hz.size):
        raise ValueError("beamformed_fixed_fraz_level_db must have shape [n_beam, n_frequency].")

    require_matplotlib()
    assert plt is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        3, 1, figsize=(12.0, 12.0), gridspec_kw={"height_ratios": [1.0, 1.0, 1.6]}
    )
    bl_axis = axes[0]
    fl_axis = axes[1]
    fraz_axis = axes[2]
    source_azimuths_deg = check.source_azimuths_deg
    bl_display_levels: list[FloatArray] = []
    fl_display_levels: list[FloatArray] = []

    for source_index, source_frequency_hz in enumerate(check.source_frequencies_hz):
        frequency_index = int(np.argmin(np.abs(frequency_hz - float(source_frequency_hz))))
        bl_level_db = fraz_level_db[:, frequency_index]
        bl_display_levels.append(np.asarray(bl_level_db, dtype=np.float64))
        peak_index = int(np.argmax(bl_level_db))
        bl_axis.plot(
            azimuth_deg,
            bl_level_db,
            linewidth=1.1,
            label=f"S{source_index + 1} {frequency_hz[frequency_index]:.1f} Hz",
        )
        bl_axis.axvline(
            float(azimuth_deg[peak_index]), color="tab:red", linewidth=0.8, linestyle="--"
        )
        if len(source_azimuths_deg) > 0:
            bl_axis.axvline(
                float(source_azimuths_deg[source_index]),
                color="black",
                linewidth=0.8,
                linestyle=":",
            )

    if len(source_azimuths_deg) > 0:
        for source_index, source_azimuth_deg in enumerate(source_azimuths_deg):
            beam_index = int(np.argmin(np.abs(azimuth_deg - float(source_azimuth_deg))))
            fl_level_db = np.asarray(fraz_level_db[beam_index], dtype=np.float64)
            fl_display_levels.append(fl_level_db)
            fl_axis.plot(
                frequency_hz,
                fl_level_db,
                linewidth=0.9,
                label=f"S{source_index + 1} beam {azimuth_deg[beam_index]:.2f} deg",
            )
            fl_axis.axvline(
                float(check.source_frequencies_hz[source_index]), color="tab:orange", linewidth=0.8
            )
    else:
        beam_index = int(np.argmax(np.max(fraz_level_db, axis=1)))
        fl_level_db = np.asarray(fraz_level_db[beam_index], dtype=np.float64)
        fl_display_levels.append(fl_level_db)
        fl_axis.plot(
            frequency_hz,
            fl_level_db,
            linewidth=0.9,
            label=f"peak beam {azimuth_deg[beam_index]:.2f} deg",
        )

    azimuth_edges = centers_to_edges(azimuth_deg)
    frequency_edges = centers_to_edges(frequency_hz)
    color_min, color_max = _finite_color_level_limits(fraz_level_db)
    image = fraz_axis.pcolormesh(
        azimuth_edges,
        frequency_edges,
        fraz_level_db.T,
        shading="flat",
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
    )
    for source_index, source_frequency_hz in enumerate(check.source_frequencies_hz):
        if len(source_azimuths_deg) > 0:
            fraz_axis.plot(
                [float(source_azimuths_deg[source_index])],
                [float(source_frequency_hz)],
                marker="x",
                color="white",
                markersize=6.0,
                label=f"S{source_index + 1}" if source_index == 0 else None,
            )
        fraz_axis.axhline(float(source_frequency_hz), color="white", linestyle="--", linewidth=0.7)

    bl_axis.set_title("Post-beamforming fixed_baseline BL at nearest signal frequency bins")
    bl_axis.set_xlabel("Azimuth [deg]")
    bl_axis.set_ylabel("RMS Level [dB re input RMS]")
    if len(bl_display_levels) > 0:
        bl_axis.set_ylim(_finite_display_level_limits(np.concatenate(bl_display_levels)))
    bl_axis.grid(True, alpha=0.3)
    bl_axis.legend(loc="best")

    fl_axis.set_title("Post-beamforming fixed_baseline FL at nearest source waiting beams")
    fl_axis.set_xlabel("Frequency [Hz]")
    fl_axis.set_ylabel("RMS Level [dB re input RMS]")
    if len(fl_display_levels) > 0:
        fl_axis.set_ylim(_finite_display_level_limits(np.concatenate(fl_display_levels)))
    fl_axis.grid(True, alpha=0.3)
    fl_axis.legend(loc="best")

    fraz_axis.set_title("Post-beamforming fixed_baseline FRAZ")
    fraz_axis.set_xlabel("Azimuth [deg]")
    fraz_axis.set_ylabel("Frequency [Hz]")
    fraz_axis.set_xlim(float(azimuth_edges[0]), float(azimuth_edges[-1]))
    fraz_axis.set_ylim(float(frequency_edges[0]), float(frequency_edges[-1]))
    fraz_axis.legend(loc="upper right")
    figure.colorbar(image, ax=fraz_axis, label="RMS Level [dB re input RMS]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
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
    """全 DFT bin 上で fixed / MVDR / diff-MVDR 重みを設計する。

    Args:
        array_positions_m: 実アレイ位置。shape は `[n_ch, 3]`、単位は m。
        shading_by_channel_bin: 複素 shading。shape は `[n_ch, n_shading_bin]`。
        shading_frequency_step_hz: shading bin 間隔。単位は Hz。
        fractional_delay_filter_bank: 事前計算済み小数遅延 FIR バンク。
        sources: source 条件。周波数は Hz、振幅はピーク振幅。
        config: sampling frequency、beam 数、FIR tap 数などの評価条件。

    Returns:
        `(weights_by_method, frequencies_hz, axis_azimuth_deg, diagnostics)`。
        `frequencies_hz` は `np.fft.fftfreq(fir_taps, d=1/fs)` の signed 全 bin 軸、
        `weights_by_method[method]` の shape は `[n_bin, n_beam, n_ch]` である。

    Raises:
        ValueError: source 条件または配列 shape が不正な場合。

    信号処理上の位置づけ:
        差分 FIR は source 周波数だけではなく全 DFT bin の応答 `Q[k]` を定義し、
        `ifft(Q)` で時間領域 tap へ変換する。beam 方向は Python ループで回さず、
        `[n_bin, n_beam, n_ch]` の配列として一括計算する。
    """
    if len(sources) == 0:
        raise ValueError("sources must contain at least one source.")

    n_design_bin = int(config.fir_taps)
    frequencies_hz = np.asarray(
        np.fft.fftfreq(n_design_bin, d=1.0 / float(config.fs_hz)),
        dtype=np.float64,
    )
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
    # shading table は正周波数側の係数表なので、負周波数 DFT bin は |f| の係数を対応させる。
    # 負周波数の厳密な複素共役対称性は元 table の定義に依存するため、ここでは実機係数の
    # nearest-bin 選択規約を保ち、FIR 化の全 bin 構造を優先する。
    shading_by_frequency = select_shading_for_frequencies(
        shading_by_channel_bin,
        float(shading_frequency_step_hz),
        np.abs(frequencies_hz),
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

    n_bin = int(frequencies_hz.size)
    n_ch = int(array_positions_m.shape[0])
    covariance = np.zeros((n_bin, n_ch, n_ch), dtype=np.complex128)
    covariance += 1.0e-12 * np.eye(n_ch, dtype=np.complex128)[np.newaxis, :, :]
    for source in sources:
        rms_amplitude = float(source.peak_amplitude) / np.sqrt(2.0)
        source_bin_candidates = [
            int(np.argmin(np.abs(frequencies_hz - float(source.frequency_hz)))),
            int(np.argmin(np.abs(frequencies_hz + float(source.frequency_hz)))),
        ]
        for frequency_index in sorted(set(source_bin_candidates)):
            steering = source_steering_by_label[source.label][frequency_index]
            # 実数 tone は正負両側の DFT bin に現れるため、全 bin 設計では +f と -f の
            # 最近傍 bin に source covariance を置く。これにより IFFT 前の Q[k] が
            # 実際の FFT bin 配列と同じ定義域を持つ。
            covariance[frequency_index] += (rms_amplitude**2) * np.outer(
                steering,
                steering.conj(),
            )

    eye = np.eye(n_ch, dtype=np.complex128)
    average_power = np.real(np.trace(covariance, axis1=1, axis2=2)) / float(n_ch)
    loading_power = float(config.diagonal_loading_ratio) * np.where(
        average_power > 0.0,
        average_power,
        1.0,
    )
    loaded_covariance = covariance + loading_power[:, np.newaxis, np.newaxis] * eye[np.newaxis]
    try:
        # RHS shape は `[n_bin, n_ch, n_beam]`。
        # np.linalg.solve は先頭 axis を batch として扱うため、bin ごとに全 beam の
        # `R_load[k] u[k,beam] = a[k,beam]` を一括で解く。
        solved_by_channel_beam = np.linalg.solve(
            loaded_covariance,
            np.swapaxes(steering_by_beam, 1, 2),
        )
        solved_by_beam_channel = np.swapaxes(solved_by_channel_beam, 1, 2)
        denominator = np.sum(steering_by_beam.conj() * solved_by_beam_channel, axis=2)
        desired_response = np.sum(fixed.conj() * steering_by_beam, axis=2)
        mvdr = np.asarray(
            np.conj(desired_response)[:, :, np.newaxis]
            * solved_by_beam_channel
            / denominator[:, :, np.newaxis],
            dtype=np.complex128,
        )
        fallback_mask = np.logical_or(
            np.abs(denominator) <= 1.0e-12,
            np.logical_not(np.all(np.isfinite(mvdr), axis=2)),
        )
        mvdr = np.where(fallback_mask[:, :, np.newaxis], fixed, mvdr)
    except np.linalg.LinAlgError:
        # 共分散 batch のどこかが解けない場合、target 保護を優先して全 beam を固定整相へ退避する。
        mvdr = fixed.copy()

    q_weight_freq = fixed - mvdr
    q_apply_freq = np.conj(q_weight_freq)
    # q_apply_freq shape: [n_bin, n_beam, n_ch]。
    # axis=0 の全 DFT bin を IFFT し、beam 方向は NumPy の batch としてまとめて処理する。
    q_apply_full_impulse = np.fft.ifft(q_apply_freq, axis=0)
    q_apply_taps = q_apply_full_impulse[: int(config.fir_taps), :, :]
    q_apply_padded = np.zeros_like(q_apply_freq)
    q_apply_padded[: int(config.fir_taps), :, :] = q_apply_taps
    reconstructed_q_weight_freq = np.conj(np.fft.fft(q_apply_padded, axis=0))
    diff = fixed - reconstructed_q_weight_freq
    q_error = np.asarray(
        np.sqrt(np.mean(np.abs(q_weight_freq - reconstructed_q_weight_freq) ** 2, axis=2)),
        dtype=np.float64,
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
                    nearest_source_beam_azimuth_deg=float(
                        axis_azimuth_deg[nearest_source_beam_index]
                    ),
                    nearest_source_beam_error_deg=abs(
                        float(axis_azimuth_deg[nearest_source_beam_index])
                        - float(source.azimuth_deg)
                    ),
                    q_reconstruction_rms_error=(
                        float(
                            np.max(
                                diagnostics["q_reconstruction_rms_error"][design_frequency_index]
                            )
                        )
                        if method.startswith("diff_mvdr_fir")
                        else 0.0
                    ),
                )
            )
    definition_source = sources[0]
    definition_frequency_index = int(
        np.argmin(np.abs(frequencies_hz - float(definition_source.frequency_hz)))
    )
    definition_spectrum_index = int(
        np.argmin(np.abs(spectrum_freqs - float(definition_source.frequency_hz)))
    )
    # beam response は source 方位を固定し、待ち受け beam 軸だけを走査する。
    # ここでは fixed_baseline 重みの全待ち受け方位応答を、実際の source+noise FFT bin で評価する。
    beam_response_spectrum = np.einsum(
        "bc,c->b",
        weights_by_method["fixed_baseline"][definition_frequency_index].conj(),
        channel_spectrum[:, definition_spectrum_index],
        optimize=True,
    )
    beam_response_level_db = tone_rms_level_db_from_fft_bin(
        beam_response_spectrum,
        n_fft=n_fft,
    )
    beam_pattern_arrays = _calculate_beam_pattern_definition_example(
        array_positions_m=positions,
        weights_by_frequency_beam_channel=weights_by_method["fixed_baseline"],
        frequencies_hz=frequencies_hz,
        azimuth_deg=axis_azimuth_deg,
        source=sources[0],
        noise_sample_rms_amplitude=float(noise_sample_rms_amplitude),
        n_fft=n_fft,
        config=config,
    )
    (
        beamformed_fixed_frequency_hz,
        beamformed_fixed_azimuth_deg,
        beamformed_fixed_fraz_level_db,
    ) = _calculate_fixed_beamformed_fraz_level_db(
        array_positions_m=positions,
        shading_by_channel_bin=shading,
        shading_frequency_step_hz=float(shading_frequency_step_hz),
        fractional_delay_filter_bank=fractional_delay_filter_bank,
        rendered_signal=rendered,
        config=config,
    )
    if not bool(np.allclose(beamformed_fixed_azimuth_deg, axis_azimuth_deg)):
        raise ValueError("fixed beamformed azimuth axis does not match metric azimuth axis.")
    arrays: dict[str, NDArray[Any]] = {
        "rendered_signal": rendered,
        "clean_signal": clean,
        "noise_signal": noise,
        "frequency_hz": frequencies_hz,
        "azimuth_deg": axis_azimuth_deg,
        "beamformed_fixed_frequency_hz": beamformed_fixed_frequency_hz,
        "beamformed_fixed_fraz_level_db": beamformed_fixed_fraz_level_db,
        "beam_response_level_db": beam_response_level_db,
        "beam_response_frequency_hz": np.asarray(
            [float(spectrum_freqs[definition_spectrum_index])], dtype=np.float64
        ),
        **beam_pattern_arrays,
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
        beamformed_fixed_frequency_hz=arrays["beamformed_fixed_frequency_hz"],
        beamformed_fixed_fraz_level_db=arrays["beamformed_fixed_fraz_level_db"],
        beam_response_level_db=arrays["beam_response_level_db"],
        beam_response_frequency_hz=arrays["beam_response_frequency_hz"],
        beam_pattern_input_azimuth_deg=arrays["beam_pattern_input_azimuth_deg"],
        beam_pattern_level_db=arrays["beam_pattern_level_db"],
        beam_pattern_steering_azimuth_deg=arrays["beam_pattern_steering_azimuth_deg"],
        beam_pattern_source_frequency_hz=arrays["beam_pattern_source_frequency_hz"],
        beam_pattern_noise_floor_db=arrays["beam_pattern_noise_floor_db"],
    )
    stale_clean_noise_png = output_dir / "external_level_normalization_check.png"
    if stale_clean_noise_png.exists():
        # 整相処理へ入るのは clean/noise 分離波形ではなく source+noise 合成波である。
        # 古い分離図が残ると評価対象を誤読するため、この既知の旧成果物だけ削除する。
        stale_clean_noise_png.unlink()
    rendered_input_png_name: str | None = None
    beamformed_spectrum_png_name: str | None = None
    beam_response_png_name: str | None = None
    beam_pattern_png_name: str | None = None
    if normalization_check is not None:
        rendered_input_png_name = "external_rendered_input_spectrum_check.png"
        beamformed_spectrum_png_name = "external_fixed_beamformed_spectrum_check.png"
        if (
            len(normalization_check.source_frequencies_hz) == 1
            and len(normalization_check.source_azimuths_deg) == 1
        ):
            beam_response_png_name = "external_beam_response_definition_example.png"
            beam_pattern_png_name = "external_beam_pattern_definition_example.png"
        write_rendered_input_spectrum_check_png(
            output_path=output_dir / rendered_input_png_name,
            arrays=arrays,
            check=normalization_check,
        )
        write_fixed_beamformed_spectrum_check_png(
            output_path=output_dir / beamformed_spectrum_png_name,
            arrays=arrays,
            check=normalization_check,
        )
        if beam_response_png_name is not None:
            write_beam_response_definition_example_png(
                output_path=output_dir / beam_response_png_name,
                arrays=arrays,
                check=normalization_check,
            )
        if beam_pattern_png_name is not None:
            write_beam_pattern_definition_example_png(
                output_path=output_dir / beam_pattern_png_name,
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
        (
            "- `external_rendered_input_spectrum_check.png`: "
            "整相前 source+noise 合成入力の周波数スペクトル確認図。"
        ),
        (
            "- `external_fixed_beamformed_spectrum_check.png`: "
            "fixed_baseline 整相後の BL/FL/FRAZ 確認図。"
        ),
        (
            "- `external_beam_response_definition_example.png`: "
            "beam response 定義例。source 方位を固定し、待ち受け方位ごとの応答を表示。"
        ),
        (
            "- `external_beam_pattern_definition_example.png`: "
            "beam pattern 定義例。待ち受け重みを固定し、入力方位を掃引。"
        ),
        "- level は `dB re input RMS` 相当のシミュレーション振幅基準である。",
        "",
        "## 結果要約",
        "",
    ]
    if rendered_input_png_name is not None:
        lines.append(f"- `{rendered_input_png_name}`")
    if beamformed_spectrum_png_name is not None:
        lines.append(f"- `{beamformed_spectrum_png_name}`")
    if beam_response_png_name is not None:
        lines.append(f"- `{beam_response_png_name}`")
    if beam_pattern_png_name is not None:
        lines.append(f"- `{beam_pattern_png_name}`")
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
        source_azimuths_deg=azimuths,
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
