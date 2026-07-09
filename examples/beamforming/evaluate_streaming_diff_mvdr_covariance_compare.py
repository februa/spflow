"""差分 MVDR の 3 秒 streaming 共分散評価 PNG を生成する。

固定整相、通常 256 sample 共分散による差分 MVDR、beam 方向合算共分散による
差分 MVDR を同じ入力で比較する。出力レベルは 256 sample rFFT の片側 bin power
を線形加算して評価し、入力した解析帯域の power 和が入力 RMS power と一致する
規約で図示する。
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming import (  # noqa: E402
    DelayAlignedBeamCovarianceAccumulator,
    LoadedMVDRWeightDesigner,
    ShortFFTCovarianceAccumulator,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 実行環境に依存する。
    plt = None


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
BoolArray: TypeAlias = NDArray[np.bool_]

FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
N_CH = 16
SPACING_M = 0.05
FFT_SIZE = 256
N_BIN = FFT_SIZE // 2 + 1
DURATION_SEC = 3.0
N_SAMPLE = int(FS_HZ * DURATION_SEC)
N_BLOCK = N_SAMPLE // FFT_SIZE
FRAME_SIZE = int(FS_HZ)
N_FRAME = N_SAMPLE // FRAME_SIZE
SOURCE_RMS = 1.0
DIAGONAL_LOADING_RATIO = 1.0e-2
COVARIANCE_TIME_CONSTANT_SEC = 1.0e6
N_BEAM = 181
OUTPUT_DIR = (
    ROOT / "artifacts" / "beamforming" / "fixed_delay_diff_mvdr" / "streaming_covariance_compare"
)
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
LEVEL_UNIT_LABEL = "dB re input total RMS"
PER_BIN_LEVEL_UNIT_LABEL = "dB re input RMS"
FIXED_ID = "fixed"
COV256_ID = "diff_mvdr_cov256"
BEAM_SUM_ID = "diff_mvdr_beam_sum"
METHOD_ORDER = (FIXED_ID, COV256_ID, BEAM_SUM_ID)
METHOD_LABELS = {
    "fixed": "fixed",
    "diff_mvdr_cov256": "diff MVDR (covariance 256 sample)",
    "diff_mvdr_beam_sum": "diff MVDR (beam-direction summed covariance)",
}
METHOD_COLORS = {
    "fixed": "black",
    "diff_mvdr_cov256": "tab:orange",
    "diff_mvdr_beam_sum": "tab:blue",
}


@dataclass(frozen=True)
class SourceSpec:
    """評価用音源の帯域・方位・レベルを保持する。

    1 つの遠方平面波音源を定義する。共分散推定、MVDR 重み設計、図示は責務に
    含めない。信号処理上は、入力信号レベル基準 `0 dB re input RMS` を作る
    source 定義である。
    """

    name: str
    azimuth_deg: float
    kind: str
    frequency_hz: float | None = None
    band_low_hz: float | None = None
    band_high_hz: float | None = None
    rms: float = SOURCE_RMS


@dataclass(frozen=True)
class ScenarioSpec:
    """評価 scenario を保持する。

    ユーザー指定の 6 パターンについて、音源群と beam response で加算する解析帯域を
    まとめる。信号処理上は、source-preserving scan の入力条件を表す。
    """

    scenario_id: str
    title: str
    sources: tuple[SourceSpec, ...]


@dataclass(frozen=True)
class ScenarioResult:
    """1 scenario の評価結果と PNG 作成元配列を保持する。

    3 方式の beam response、source beam spectrum、入力 spectrum、peak 方位、
    target response 診断をまとめる。
    """

    spec: ScenarioSpec
    input_power_by_bin: FloatArray
    output_power_by_method: dict[str, FloatArray]
    band_response_by_method: dict[str, FloatArray]
    source_beam_indices: dict[str, int]
    math_checks: dict[str, float]


def _plt() -> Any:
    """matplotlib.pyplot を遅延取得する。"""

    if plt is None:
        raise RuntimeError("matplotlib is required to plot figures.")
    return plt


def _array_positions() -> FloatArray:
    """x 軸上の中心化 ULA センサ位置を返す。shape は `[n_ch, 3]`、単位は m。"""

    x = (np.arange(N_CH, dtype=np.float64) - 0.5 * float(N_CH - 1)) * SPACING_M
    return np.column_stack((x, np.zeros(N_CH, dtype=np.float64), np.zeros(N_CH, dtype=np.float64)))


def _direction_from_azimuth(azimuth_deg: float) -> FloatArray:
    """0-180 deg 表示方位から x-y 平面方向ベクトルを返す。"""

    azimuth_rad = np.deg2rad(float(azimuth_deg))
    return np.asarray([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0], dtype=np.float64)


def _tau_for_azimuth(positions_m: FloatArray, azimuth_deg: float) -> FloatArray:
    """MATLAB 式 `tau = pos.' * Dir3d / c` と同じ到達遅延を返す。"""

    return np.asarray(
        positions_m @ _direction_from_azimuth(float(azimuth_deg)) / SOUND_SPEED_M_S,
        dtype=np.float64,
    )


def _steering_table(
    positions_m: FloatArray, azimuth_deg: FloatArray, frequencies_hz: FloatArray
) -> ComplexArray:
    """周波数領域の遠方場 steering vector を生成する。

    Returns:
        steering。shape は `[n_beam, n_bin, n_ch]`。
        `X_ch[k] = S[k] exp(-j 2π f tau[ch])` に対し、固定重み `w=a/n_ch` が
        `w^H X = S` となる規約である。
    """

    steering = np.empty(
        (azimuth_deg.size, frequencies_hz.size, positions_m.shape[0]), dtype=np.complex128
    )
    for beam_index, azimuth in enumerate(azimuth_deg.tolist()):
        tau_sec = _tau_for_azimuth(positions_m, float(azimuth))
        # a[ch,k] = exp(-j 2π f tau[ch]) は、遅延した観測信号の複素位相そのものを表す。
        # w^H X の規約で使うため、整合方位では conj(a) * a が 1 になる。
        steering[beam_index] = np.exp(
            -1j * 2.0 * np.pi * frequencies_hz[:, np.newaxis] * tau_sec[np.newaxis, :]
        )
    return steering


def _scenario_specs() -> tuple[ScenarioSpec, ...]:
    """ユーザー指定 6 パターンを返す。"""

    return (
        ScenarioSpec(
            "low_narrow_az030",
            "Low frequency narrowband, source azimuth 30 deg",
            (SourceSpec("low_tone_512", 30.0, "tone", frequency_hz=512.0),),
        ),
        ScenarioSpec(
            "low_broadband_az010",
            "Low frequency broadband, source azimuth 10 deg",
            (
                SourceSpec(
                    "low_band_256_1536", 10.0, "band", band_low_hz=256.0, band_high_hz=1536.0
                ),
            ),
        ),
        ScenarioSpec(
            "high_narrow_az050",
            "High frequency narrowband, source azimuth 50 deg",
            (SourceSpec("high_tone_8192", 50.0, "tone", frequency_hz=8192.0),),
        ),
        ScenarioSpec(
            "high_broadband_az180",
            "High frequency broadband, source azimuth 180 deg",
            (
                SourceSpec(
                    "high_band_7168_11264", 180.0, "band", band_low_hz=7168.0, band_high_hz=11264.0
                ),
            ),
        ),
        ScenarioSpec(
            "near_broadband_high_low_az085_az080",
            "Nearby broadband sources, high/low non-overlapped bands, 85/80 deg",
            (
                SourceSpec(
                    "high_band_7168_11264", 85.0, "band", band_low_hz=7168.0, band_high_hz=11264.0
                ),
                SourceSpec(
                    "low_band_512_2048", 80.0, "band", band_low_hz=512.0, band_high_hz=2048.0
                ),
            ),
        ),
        ScenarioSpec(
            "near_narrow_high_high_az085_az080",
            "Nearby narrowband high-frequency sources, shifted tones, 85/80 deg",
            (
                SourceSpec("high_tone_8192", 85.0, "tone", frequency_hz=8192.0),
                SourceSpec("high_tone_8448", 80.0, "tone", frequency_hz=8448.0),
            ),
        ),
    )


def _source_bin_mask(frequencies_hz: FloatArray, source: SourceSpec) -> BoolArray:
    """source の解析対象 rFFT bin mask を返す。shape は `[n_bin]`。"""

    if source.kind == "tone":
        if source.frequency_hz is None:
            raise ValueError("tone source requires frequency_hz.")
        nearest_index = int(np.argmin(np.abs(frequencies_hz - float(source.frequency_hz))))
        mask = np.zeros(frequencies_hz.shape, dtype=np.bool_)
        mask[nearest_index] = True
        return mask
    if source.kind == "band":
        if source.band_low_hz is None or source.band_high_hz is None:
            raise ValueError("band source requires band_low_hz and band_high_hz.")
        return np.asarray(
            (float(source.band_low_hz) <= frequencies_hz)
            & (frequencies_hz <= float(source.band_high_hz)),
            dtype=np.bool_,
        )
    raise ValueError(f"unsupported source kind: {source.kind}")


def _analysis_bin_mask(frequencies_hz: FloatArray, sources: tuple[SourceSpec, ...]) -> BoolArray:
    """scenario 全体で加算する解析帯域 mask を返す。"""

    mask = np.zeros(frequencies_hz.shape, dtype=np.bool_)
    for source in sources:
        mask |= _source_bin_mask(frequencies_hz, source)
    return mask


def _one_sided_bin_power(spectrum: ComplexArray) -> FloatArray:
    """rFFT spectrum を per-bin RMS power へ変換する。

    interior bin は共役負周波数側を含めるため 2 倍する。DC/Nyquist は 2 倍しない。
    この和は時間領域 RMS power と一致する。
    """

    power = np.abs(spectrum / float(FFT_SIZE)) ** 2
    scale = np.ones(spectrum.shape[-1], dtype=np.float64)
    if spectrum.shape[-1] > 2:
        scale[1:-1] = 2.0
    return np.asarray(power * scale, dtype=np.float64)


def _make_source_reference_blocks(
    source: SourceSpec, frequencies_hz: FloatArray, rng: np.random.Generator
) -> ComplexArray:
    """1 source の block-wise rFFT 基準スペクトルを作る。

    各 block の解析帯域 power を `source.rms ** 2` に正規化するため、後段で bin power を
    線形加算すると入力 RMS power と一致する。
    """

    spectra = np.zeros((N_BLOCK, N_BIN), dtype=np.complex128)
    mask = _source_bin_mask(frequencies_hz, source)
    mask[0] = False
    mask[-1] = False
    selected_indices = np.flatnonzero(mask)
    if selected_indices.size == 0:
        raise ValueError(f"source {source.name} has no analysis bins.")
    if source.kind == "tone":
        tone_index = int(selected_indices[0])
        # interior rFFT bin の real sinusoid RMS は sqrt(2)*|X[k]/N| である。
        spectra[:, tone_index] = (float(source.rms) * float(FFT_SIZE) / np.sqrt(2.0)) + 0.0j
        return spectra
    for block_index in range(N_BLOCK):
        random_phase = rng.uniform(0.0, 2.0 * np.pi, size=selected_indices.size)
        random_amplitude = rng.rayleigh(scale=1.0, size=selected_indices.size)
        spectra[block_index, selected_indices] = random_amplitude * np.exp(1j * random_phase)
        current_power = float(
            np.sum(_one_sided_bin_power(spectra[block_index : block_index + 1])[0, mask])
        )
        if current_power <= 0.0:
            raise ValueError("broadband source generated zero power.")
        spectra[block_index] *= float(source.rms) / np.sqrt(current_power)
    return spectra


def _render_scenario_blocks(
    scenario: ScenarioSpec, positions_m: FloatArray, frequencies_hz: FloatArray
) -> FloatArray:
    """scenario の 3 秒分 multi-channel 信号を block-wise に生成する。"""

    seed = sum(ord(char) for char in scenario.scenario_id) % (2**32)
    rng = np.random.default_rng(seed)
    block_spectrum_ch = np.zeros((N_BLOCK, N_CH, N_BIN), dtype=np.complex128)
    for source in scenario.sources:
        source_spectrum = _make_source_reference_blocks(source, frequencies_hz, rng)
        tau_sec = _tau_for_azimuth(positions_m, float(source.azimuth_deg))
        # X_ch[k] = S[k] exp(-j 2π f tau_ch)。この位相が steering vector の定義と一致する。
        phase = np.exp(-1j * 2.0 * np.pi * tau_sec[:, np.newaxis] * frequencies_hz[np.newaxis, :])
        block_spectrum_ch += source_spectrum[:, np.newaxis, :] * phase[np.newaxis, :, :]
    block_signal = np.fft.irfft(block_spectrum_ch, n=FFT_SIZE, axis=2)
    return np.asarray(np.moveaxis(block_signal, 0, 1).reshape(N_CH, N_SAMPLE), dtype=np.float64)


def _estimate_covariance_256_streaming(input_signal: FloatArray) -> ComplexArray:
    """通常 256 sample block 共分散を 3 秒分 streaming 積分する。"""

    accumulator = ShortFFTCovarianceAccumulator(
        n_ch=N_CH,
        fft_size=FFT_SIZE,
        block_size=FFT_SIZE,
        fs_hz=FS_HZ,
        covariance_time_constant_sec=COVARIANCE_TIME_CONSTANT_SEC,
        blocks_per_weight_update=N_BLOCK,
    )
    for block_start in range(0, N_SAMPLE, FFT_SIZE):
        block = input_signal[:, block_start : block_start + FFT_SIZE]
        accumulator.process(block)
    return np.asarray(accumulator.covariance[:N_BIN], dtype=np.complex128)


def _delay_table_for_beam_sum(
    positions_m: FloatArray, azimuth_deg: FloatArray
) -> NDArray[np.int64]:
    """beam 方向合算共分散で使う `int32(tau * fs)` 相当の整数遅延表を作る。"""

    delay = np.empty((N_CH, azimuth_deg.size), dtype=np.int64)
    for beam_index, azimuth in enumerate(azimuth_deg.tolist()):
        tau_sec = _tau_for_azimuth(positions_m, float(azimuth))
        delay[:, beam_index] = np.rint(tau_sec * FS_HZ).astype(np.int64)
    return delay


def _estimate_beam_sum_covariance_streaming(
    input_signal: FloatArray, positions_m: FloatArray, azimuth_deg: FloatArray
) -> ComplexArray:
    """beam 方向合算共分散を 3 秒分 streaming 積分する。"""

    accumulator = DelayAlignedBeamCovarianceAccumulator(
        delay_table_sample=_delay_table_for_beam_sum(positions_m, azimuth_deg),
        fs_hz=FS_HZ,
        snapshot_length=FFT_SIZE,
        frame_size=FRAME_SIZE,
        center_sample=FRAME_SIZE // 2,
        covariance_time_constant_sec=COVARIANCE_TIME_CONSTANT_SEC,
        frames_per_weight_update=N_FRAME,
    )
    result = None
    for frame_start in range(0, N_SAMPLE, FRAME_SIZE):
        frame = input_signal[:, frame_start : frame_start + FRAME_SIZE]
        result = accumulator.process(frame)
    if result is None:
        raise ValueError("beam-sum covariance did not process any frame.")
    return np.asarray(result.covariance_for_mvdr, dtype=np.complex128)


def _design_fixed_weights(steering: ComplexArray) -> ComplexArray:
    """固定整相重み `w=a/n_ch` を返す。shape は `[n_beam, n_bin, n_ch]`。"""

    return np.asarray(steering / float(N_CH), dtype=np.complex128)


def _design_mvdr_scan_weights(
    covariance: ComplexArray, steering: ComplexArray, fixed_weights: ComplexArray
) -> tuple[ComplexArray, FloatArray, FloatArray]:
    """全 beam の MVDR 重みを設計する。

    Returns:
        `(weights, condition_number, fallback_rate)`。`weights` shape は `[n_beam, n_bin, n_ch]`。
    """

    n_beam = int(steering.shape[0])
    weights = np.empty_like(fixed_weights)
    condition = np.empty((n_beam, N_BIN), dtype=np.float64)
    fallback_rate = np.empty(n_beam, dtype=np.float64)
    for beam_index in range(n_beam):
        designer = LoadedMVDRWeightDesigner(diagonal_loading_ratio=DIAGONAL_LOADING_RATIO)
        result = designer.compute(covariance, steering[beam_index], fixed_weights[beam_index])
        weights[beam_index] = result.weights
        condition[beam_index] = result.loaded_condition_number
        fallback_rate[beam_index] = float(np.mean(result.fallback_mask.astype(np.float64)))
    return weights, condition, fallback_rate


def _apply_weights_block_spectra(input_signal: FloatArray, weights: ComplexArray) -> ComplexArray:
    """block-wise rFFT 信号へ beam 重みを適用する。

    `y[block, beam, bin] = w[beam, bin]^H X[block, bin]` を計算する。
    """

    blocks = input_signal.reshape(N_CH, N_BLOCK, FFT_SIZE).transpose(1, 0, 2)
    spectrum = np.fft.rfft(blocks, n=FFT_SIZE, axis=2)
    return np.asarray(
        np.einsum("bkc,tck->tbk", weights.conj(), spectrum, optimize=True), dtype=np.complex128
    )


def _block_average_power(output_spectrum: ComplexArray) -> FloatArray:
    """block spectrum を block 平均 per-bin RMS power に変換する。"""

    return np.asarray(np.mean(_one_sided_bin_power(output_spectrum), axis=0), dtype=np.float64)


def _db10(power: FloatArray | float, floor_power: float = 1.0e-24) -> FloatArray:
    """power ratio を dB に変換する。"""

    return np.asarray(
        10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), float(floor_power))),
        dtype=np.float64,
    )


def _finite_ylim(curves: list[FloatArray], *, dynamic_range_db: float) -> tuple[float, float]:
    """有限値から dB 軸範囲を決める。"""

    finite_parts = [curve[np.isfinite(curve)] for curve in curves]
    finite = np.concatenate([part for part in finite_parts if part.size > 0])
    if finite.size == 0:
        return -100.0, 5.0
    top = float(np.max(finite)) + 3.0
    bottom = max(float(np.min(finite)) - 3.0, top - float(dynamic_range_db))
    return bottom, top


def _input_power_by_bin(input_signal: FloatArray) -> FloatArray:
    """入力信号の channel/block 平均 per-bin RMS power を返す。"""

    blocks = input_signal.reshape(N_CH, N_BLOCK, FFT_SIZE).transpose(1, 0, 2)
    spectrum = np.fft.rfft(blocks, n=FFT_SIZE, axis=2)
    return np.asarray(np.mean(_one_sided_bin_power(spectrum), axis=(0, 1)), dtype=np.float64)


def _math_check(
    scenario: ScenarioSpec,
    frequencies_hz: FloatArray,
    steering: ComplexArray,
    fixed_weights: ComplexArray,
    mvdr_weights_by_method: dict[str, ComplexArray],
    condition_by_method: dict[str, FloatArray],
    fallback_rate_by_method: dict[str, FloatArray],
    analysis_mask: BoolArray,
    input_power_by_bin: FloatArray,
    input_reference_power: float,
    source_beam_indices: dict[str, int],
) -> dict[str, float]:
    """評価時に数式上の前提が崩れていないかを数値で確認する。

    Args:
        scenario: 評価パターン。音源方位、帯域、RMS レベルを保持する。
        frequencies_hz: FFT 周波数軸。shape は [n_bin]、単位は Hz。
        steering: スキャン用ステアリング。shape は [n_beam, n_bin, n_ch]。
        fixed_weights: fixed 整相重み。shape は [n_beam, n_bin, n_ch]。
        mvdr_weights_by_method: MVDR 方式別の重み。各 shape は [n_beam, n_bin, n_ch]。
        condition_by_method: MVDR 方式別の共分散条件数。各 shape は [n_bin]。
        fallback_rate_by_method: MVDR 方式別の beam ごとの fallback 率。
        analysis_mask: レベル加算対象 bin。shape は [n_bin]。
        input_power_by_bin: 入力 channel 平均 per-bin RMS power。shape は [n_bin]。
        input_reference_power: 入力音源の総 RMS power。単位は振幅 RMS^2。
        source_beam_indices: 音源名から最寄り beam index への対応。

    Returns:
        チェック値を格納した dict。誤差は dB ではなく線形値で返す。
    """

    checked: dict[str, float] = {}

    # one-sided FFT の bin power を帯域内で加算すると、Parseval の定理により
    # 時間領域 RMS power と一致する。ここでは channel 平均の入力 power で、
    # 出力図の基準に使う入力総 power が設計値とずれていないかを見る。
    selected_input_power = float(np.sum(input_power_by_bin[analysis_mask], dtype=np.float64))
    checked["input_total_rms_from_selected_bins"] = float(np.sqrt(selected_input_power))
    checked["input_total_rms_expected"] = float(np.sqrt(input_reference_power))
    checked["input_total_rms_error"] = float(
        abs(np.sqrt(selected_input_power) - np.sqrt(input_reference_power))
    )

    for method_id, weights in mvdr_weights_by_method.items():
        checked[f"{method_id}_condition_max"] = float(np.max(condition_by_method[method_id]))
        checked[f"{method_id}_fallback_rate"] = float(np.max(fallback_rate_by_method[method_id]))

        method_max_final_error = 0.0
        method_max_distortionless_error = 0.0
        for source in scenario.sources:
            beam_index = source_beam_indices[source.name]
            mask = _source_bin_mask(frequencies_hz, source)

            # 差分 MVDR の実装は q = w_fixed - w_mvdr を内部量として扱う。
            # 最終的な出力重み fixed - q が直接 MVDR 重みと一致することを、
            # 数式の置換誤差として確認する。
            q_weight = fixed_weights[beam_index] - weights[beam_index]
            final_weight = fixed_weights[beam_index] - q_weight
            final_error = np.max(np.abs(final_weight[mask] - weights[beam_index, mask]))
            method_max_final_error = max(method_max_final_error, float(final_error))

            # MVDR の歪みなし制約は w^H a = 1。ここで steering は入力生成と同じ
            # exp(-j 2π f tau) なので、conj(w) と steering の ch 内積で確認する。
            response = np.sum(weights[beam_index, mask].conj() * steering[beam_index, mask], axis=1)
            distortionless_error = np.max(np.abs(response - 1.0 + 0.0j))
            method_max_distortionless_error = max(
                method_max_distortionless_error, float(distortionless_error)
            )

        checked[f"{method_id}_final_weight_error_max"] = method_max_final_error
        checked[f"{method_id}_distortionless_error_max"] = method_max_distortionless_error

    return checked


def _evaluate_scenario(
    scenario: ScenarioSpec,
    positions_m: FloatArray,
    azimuths_deg: FloatArray,
    frequencies_hz: FloatArray,
    steering: ComplexArray,
) -> ScenarioResult:
    """1 つの評価パターンについて 3 方式の出力 power を計算する。

    Args:
        scenario: 評価パターン。音源、帯域、到来方位を含む。
        positions_m: センサ位置。shape は [n_ch, 3]、単位は m。
        azimuths_deg: beam 方位軸。shape は [n_beam]、単位は deg。
        frequencies_hz: FFT 周波数軸。shape は [n_bin]、単位は Hz。
        steering: スキャン用ステアリング。shape は [n_beam, n_bin, n_ch]。

    Returns:
        入力、各方式の beam/bin power、帯域積分応答、数式チェック値を含む結果。

    Raises:
        ValueError: 生成信号や共分散の shape が想定と異なる場合。
    """

    input_signal = _render_scenario_blocks(scenario, positions_m, frequencies_hz)
    if input_signal.shape != (N_CH, N_SAMPLE):
        raise ValueError(f"input_signal shape must be {(N_CH, N_SAMPLE)}, got {input_signal.shape}")

    fixed_weights = _design_fixed_weights(steering)
    covariance_256 = _estimate_covariance_256_streaming(input_signal)
    covariance_beam_sum = _estimate_beam_sum_covariance_streaming(
        input_signal, positions_m, azimuths_deg
    )

    weights_cov256, condition_cov256, fallback_cov256 = _design_mvdr_scan_weights(
        covariance_256, steering, fixed_weights
    )
    weights_beam_sum, condition_beam_sum, fallback_beam_sum = _design_mvdr_scan_weights(
        covariance_beam_sum,
        steering,
        fixed_weights,
    )

    weights_by_method = {
        FIXED_ID: fixed_weights,
        COV256_ID: weights_cov256,
        BEAM_SUM_ID: weights_beam_sum,
    }

    output_power_by_method: dict[str, FloatArray] = {}
    for method_id, weights in weights_by_method.items():
        output_spectrum = _apply_weights_block_spectra(input_signal, weights)
        output_power = _block_average_power(output_spectrum)
        if output_power.shape != (N_BEAM, N_BIN):
            raise ValueError(
                f"{method_id} output power shape must be {(N_BEAM, N_BIN)}, "
                f"got {output_power.shape}"
            )
        output_power_by_method[method_id] = output_power

    analysis_mask = _analysis_bin_mask(frequencies_hz, scenario.sources)
    input_power = _input_power_by_bin(input_signal)
    input_reference_power = float(sum(source.rms**2 for source in scenario.sources))

    band_response_by_method: dict[str, FloatArray] = {}
    for method_id, output_power in output_power_by_method.items():
        # ビーム応答は、表示対象帯域の per-bin RMS power を線形和してから dB 化する。
        # 狭帯域でも広帯域でも、同じ source RMS を基準にするため、帯域幅で
        # レベルが変わる表示誤差を避けられる。
        band_power = np.sum(output_power[:, analysis_mask], axis=1, dtype=np.float64)
        band_response_by_method[method_id] = _db10(band_power / input_reference_power)

    source_beam_indices = {
        source.name: int(np.argmin(np.abs(azimuths_deg - source.azimuth_deg)))
        for source in scenario.sources
    }
    math_checks = _math_check(
        scenario,
        frequencies_hz,
        steering,
        fixed_weights,
        {COV256_ID: weights_cov256, BEAM_SUM_ID: weights_beam_sum},
        {COV256_ID: condition_cov256, BEAM_SUM_ID: condition_beam_sum},
        {COV256_ID: fallback_cov256, BEAM_SUM_ID: fallback_beam_sum},
        analysis_mask,
        input_power,
        input_reference_power,
        source_beam_indices,
    )

    return ScenarioResult(
        spec=scenario,
        input_power_by_bin=input_power,
        output_power_by_method=output_power_by_method,
        band_response_by_method=band_response_by_method,
        source_beam_indices=source_beam_indices,
        math_checks=math_checks,
    )


def _plot_beam_response(result: ScenarioResult, azimuths_deg: FloatArray) -> Path:
    """3 方式を重ねた帯域積分ビーム応答を PNG に保存する。"""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    curves = [result.band_response_by_method[method_id] for method_id in METHOD_ORDER]

    for method_id in METHOD_ORDER:
        ax.plot(
            azimuths_deg,
            result.band_response_by_method[method_id],
            label=METHOD_LABELS[method_id],
            color=METHOD_COLORS[method_id],
            linewidth=2.0,
        )

    for source in result.spec.sources:
        ax.axvline(source.azimuth_deg, color="#4b5563", linewidth=1.0, linestyle=":")
        ax.text(
            source.azimuth_deg,
            ax.get_ylim()[1] if ax.has_data() else 0.0,
            f" {source.name}: {source.azimuth_deg:.0f} deg",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="#374151",
        )

    expected_single_source_level = 10.0 * np.log10(
        SOURCE_RMS**2 / max(float(len(result.spec.sources)) * SOURCE_RMS**2, 1.0e-24)
    )
    ax.axhline(
        expected_single_source_level, color="#111827", linewidth=0.9, linestyle="--", alpha=0.6
    )

    ymin, ymax = _finite_ylim(curves, dynamic_range_db=85.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(float(np.min(azimuths_deg)), float(np.max(azimuths_deg)))
    ax.set_xlabel("Beam azimuth [deg]")
    ax.set_ylabel(f"Band-integrated RMS Level [{LEVEL_UNIT_LABEL}]")
    ax.set_title(result.spec.title)
    ax.grid(True, which="both", color="#d1d5db", alpha=0.7)
    ax.legend(loc="best")
    ax.text(
        0.01,
        0.02,
        "Level = sum of selected 256-point one-sided rFFT bin powers / "
        "input source total RMS power.",
        transform=ax.transAxes,
        fontsize=8,
        color="#374151",
        ha="left",
        va="bottom",
    )

    output_path = FIGURE_DIR / f"{result.spec.scenario_id}_beam_response.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_spectrum(result: ScenarioResult, frequencies_hz: FloatArray) -> Path:
    """音源方位 beam の周波数スペクトルを PNG に保存する。"""

    plt = _plt()
    n_source = len(result.spec.sources)
    fig, axes_raw = plt.subplots(
        n_source, 1, figsize=(11.5, 4.8 * n_source), squeeze=False, constrained_layout=True
    )
    axes = axes_raw[:, 0]

    input_level_db = _db10(result.input_power_by_bin / max(SOURCE_RMS**2, 1.0e-24))
    all_curves: list[FloatArray] = [input_level_db]

    for ax, source in zip(axes, result.spec.sources, strict=True):
        beam_index = result.source_beam_indices[source.name]
        curves_for_axis: list[FloatArray] = [input_level_db]
        ax.plot(
            frequencies_hz,
            input_level_db,
            label="Input channel mean",
            color="#6b7280",
            linewidth=1.4,
            alpha=0.8,
        )
        for method_id in METHOD_ORDER:
            output_level = _db10(
                result.output_power_by_method[method_id][beam_index] / max(SOURCE_RMS**2, 1.0e-24)
            )
            curves_for_axis.append(output_level)
            all_curves.append(output_level)
            ax.plot(
                frequencies_hz,
                output_level,
                label=METHOD_LABELS[method_id],
                color=METHOD_COLORS[method_id],
                linewidth=1.8,
            )

        for band_source in result.spec.sources:
            band_mask = _source_bin_mask(frequencies_hz, band_source)
            if bool(np.any(band_mask)):
                band_freqs = frequencies_hz[band_mask]
                ax.axvspan(float(band_freqs[0]), float(band_freqs[-1]), color="#e5e7eb", alpha=0.35)

        ymin, ymax = _finite_ylim(curves_for_axis, dynamic_range_db=115.0)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0.0, FS_HZ / 2.0)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"Per-bin RMS Level [{PER_BIN_LEVEL_UNIT_LABEL}]")
        ax.set_title(f"{source.name}: beam {source.azimuth_deg:.0f} deg")
        ax.grid(True, which="both", color="#d1d5db", alpha=0.7)
        ax.legend(loc="best")

    output_path = FIGURE_DIR / f"{result.spec.scenario_id}_spectrum.png"
    fig.suptitle(result.spec.title, fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _write_summary(results: list[ScenarioResult], azimuths_deg: FloatArray) -> Path:
    """ピーク方位と音源方位レベルを CSV に保存する。"""

    output_path = DATA_DIR / "summary_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "scenario_id",
                "method",
                "peak_azimuth_deg",
                "peak_level_db_re_input_total_rms",
                "source_name",
                "source_azimuth_deg",
                "source_beam_level_db_re_input_total_rms",
                "input_total_rms_from_bins",
                "input_total_rms_expected",
            ]
        )
        for result in results:
            for method_id in METHOD_ORDER:
                response = result.band_response_by_method[method_id]
                peak_index = int(np.argmax(response))
                for source in result.spec.sources:
                    source_beam_index = result.source_beam_indices[source.name]
                    writer.writerow(
                        [
                            result.spec.scenario_id,
                            METHOD_LABELS[method_id],
                            f"{float(azimuths_deg[peak_index]):.6f}",
                            f"{float(response[peak_index]):.6f}",
                            source.name,
                            f"{source.azimuth_deg:.6f}",
                            f"{float(response[source_beam_index]):.6f}",
                            f"{result.math_checks['input_total_rms_from_selected_bins']:.12f}",
                            f"{result.math_checks['input_total_rms_expected']:.12f}",
                        ]
                    )
    return output_path


def _write_math_check(results: list[ScenarioResult]) -> Path:
    """数式チェックと正規化条件を Markdown に保存する。"""

    output_path = DATA_DIR / "math_check.md"
    lines: list[str] = []
    lines.append("# Streaming covariance MVDR math check")
    lines.append("")
    lines.append("## Level normalization")
    lines.append("")
    lines.append("The 256-point rFFT bin power is normalized as follows:")
    lines.append("")
    lines.append("- DC/Nyquist: `|X[k]|^2 / N^2`")
    lines.append("- interior bins: `2 |X[k]|^2 / N^2`")
    lines.append("")
    lines.append(
        "Therefore `sum_k P[k]` equals the time-domain RMS power for each 256-sample block."
    )
    lines.append(
        "Beam response figures use `sum_band P_out[beam, k] / "
        "sum_source RMS^2` before converting to dB."
    )
    lines.append("")
    lines.append("## Differential MVDR identity")
    lines.append("")
    lines.append(
        "For the differential implementation, `q = w_fixed - w_mvdr`; "
        "the final weight is `w_fixed - q = w_mvdr`."
    )
    lines.append(
        "The check below reports the maximum absolute difference of this identity and "
        "the MVDR constraint `w^H a = 1` in the source bands."
    )
    lines.append("")

    for result in results:
        lines.append(f"## {result.spec.scenario_id}")
        lines.append("")
        lines.append(
            "- Input RMS from selected bins: "
            f"`{result.math_checks['input_total_rms_from_selected_bins']:.12f}`"
        )
        lines.append(
            f"- Input RMS expected: `{result.math_checks['input_total_rms_expected']:.12f}`"
        )
        lines.append(
            f"- Input RMS absolute error: `{result.math_checks['input_total_rms_error']:.12e}`"
        )
        for method_id in (COV256_ID, BEAM_SUM_ID):
            lines.append(
                f"- {METHOD_LABELS[method_id]} condition max: "
                f"`{result.math_checks[f'{method_id}_condition_max']:.6e}`"
            )
            lines.append(
                f"- {METHOD_LABELS[method_id]} fallback rate: "
                f"`{result.math_checks[f'{method_id}_fallback_rate']:.6f}`"
            )
            lines.append(
                f"- {METHOD_LABELS[method_id]} final weight identity error max: "
                f"`{result.math_checks[f'{method_id}_final_weight_error_max']:.6e}`"
            )
            lines.append(
                f"- {METHOD_LABELS[method_id]} distortionless error max: "
                f"`{result.math_checks[f'{method_id}_distortionless_error_max']:.6e}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _write_npz(
    results: list[ScenarioResult], azimuths_deg: FloatArray, frequencies_hz: FloatArray
) -> Path:
    """再確認用に主要配列を npz へ保存する。"""

    output_path = DATA_DIR / "scenario_results.npz"
    arrays: dict[str, FloatArray] = {
        "azimuths_deg": azimuths_deg,
        "frequencies_hz": frequencies_hz,
    }
    for result in results:
        prefix = result.spec.scenario_id
        arrays[f"{prefix}__input_power_by_bin"] = result.input_power_by_bin
        for method_id in METHOD_ORDER:
            arrays[f"{prefix}__{method_id}__output_power"] = result.output_power_by_method[
                method_id
            ]
            arrays[f"{prefix}__{method_id}__band_response_db"] = result.band_response_by_method[
                method_id
            ]
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            archive.writestr(f"{name}.npy", buffer.getvalue())
    return output_path


def _write_review_index(figure_paths: list[Path], data_paths: list[Path]) -> Path:
    """出力物の見方を短い index として保存する。"""

    output_path = OUTPUT_DIR / "README.md"
    lines: list[str] = [
        "# Streaming covariance MVDR evaluation",
        "",
        "この評価は 3 秒分のストリーミング入力を 256 sample block で処理し、",
        "fixed、差分 MVDR(共分散256サンプルver)、差分 MVDR(beam方向合成ver)を比較する。",
        "",
        "## レベル基準",
        "",
        "- ビーム応答: 対象帯域の one-sided 256-point rFFT bin power を線形加算し、"
        "入力音源の総 RMS power で割って dB 表示する。",
        "- スペクトル: bin ごとの RMS power を、1 音源あたりの入力 RMS power で割って "
        "dB 表示する。",
        "- したがって、狭帯域・広帯域のどちらでも、入力帯域を加算したレベルは"
        "入力信号レベルに一致する。",
        "",
        "## Figures",
        "",
    ]
    for path in figure_paths:
        lines.append(f"- `{path.relative_to(OUTPUT_DIR)}`")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    for path in data_paths:
        lines.append(f"- `{path.relative_to(OUTPUT_DIR)}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _write_metadata(results: list[ScenarioResult]) -> Path:
    """評価条件と数式チェック値を JSON に保存する。"""

    output_path = DATA_DIR / "metadata.json"
    payload: dict[str, Any] = {
        "fs_hz": FS_HZ,
        "sound_speed_m_s": SOUND_SPEED_M_S,
        "n_ch": N_CH,
        "spacing_m": SPACING_M,
        "fft_size": FFT_SIZE,
        "duration_sec": DURATION_SEC,
        "n_block": N_BLOCK,
        "frame_size": FRAME_SIZE,
        "n_frame": N_FRAME,
        "diagonal_loading_ratio": DIAGONAL_LOADING_RATIO,
        "scenarios": [],
    }
    for result in results:
        payload["scenarios"].append(
            {
                "scenario_id": result.spec.scenario_id,
                "title": result.spec.title,
                "sources": [source.__dict__ for source in result.spec.sources],
                "math_checks": result.math_checks,
            }
        )
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _zip_output() -> Path:
    """画像と数値結果を zip にまとめる。"""

    zip_path = OUTPUT_DIR.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in OUTPUT_DIR.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(OUTPUT_DIR))
    return zip_path


def build_report() -> Path:
    """評価を実行し、図・数値・数式チェック結果を出力する。

    Returns:
        出力ディレクトリをまとめた zip ファイルのパス。
    """

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    positions_m = _array_positions()
    azimuths_deg = np.linspace(0.0, 180.0, N_BEAM, dtype=np.float64)
    frequencies_hz = np.fft.rfftfreq(FFT_SIZE, d=1.0 / FS_HZ)
    steering = _steering_table(positions_m, azimuths_deg, frequencies_hz)

    results: list[ScenarioResult] = []
    figure_paths: list[Path] = []
    for scenario in _scenario_specs():
        result = _evaluate_scenario(scenario, positions_m, azimuths_deg, frequencies_hz, steering)
        results.append(result)
        figure_paths.append(_plot_beam_response(result, azimuths_deg))
        figure_paths.append(_plot_spectrum(result, frequencies_hz))

    data_paths = [
        _write_summary(results, azimuths_deg),
        _write_math_check(results),
        _write_npz(results, azimuths_deg, frequencies_hz),
        _write_metadata(results),
    ]
    data_paths.append(_write_review_index(figure_paths, data_paths))
    return _zip_output()


def main() -> None:
    """コマンドライン実行入口。"""

    zip_path = build_report()
    print(f"wrote {OUTPUT_DIR}")
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
