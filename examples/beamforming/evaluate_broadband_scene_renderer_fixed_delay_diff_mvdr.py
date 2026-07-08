"""scene_renderer 広帯域入力で固定整相+差分 MVDR の報告パックを作る。

20 deg 方向から中心 9000 Hz の帯域制限広帯域信号を入射させ、
入力スペクトル、整相後ビーム応答、整相後スペクトルを PNG と
AI レビュー用データとして保存する。
"""

from __future__ import annotations

import csv
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
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from scene_renderer import (  # noqa: E402
    AcousticSource,
    BandLimitedNoiseSpectrum,
    ConstantEnvelope,
    FreeField,
    LinearArray,
    Receiver,
    Scene,
    SceneRenderer,
    SourceComponent,
    StaticPose,
)
from spflow.beamforming import (  # noqa: E402
    DelayTable,
    DifferenceCorrectionFIRDesigner,
    design_fixed_delay_fractional_weights_from_delay_table,
    design_standard_fractional_delay_filter_bank,
    make_directions,
)
from spflow.beamforming.diagnostic_plotting import centers_to_edges, require_matplotlib  # noqa: E402

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 実行環境に依存する。
    plt = None


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
DURATION_S = 1.0
N_SAMPLE = int(round(FS_HZ * DURATION_S))
N_CH = 32
SPACING_M = 0.05
SOURCE_AZIMUTH_DEG = 20.0
SOURCE_CENTER_HZ = 9000.0
SOURCE_BAND_LOW_HZ = 8500.0
SOURCE_BAND_HIGH_HZ = 9500.0
SOURCE_LEVEL_DB = 0.0
SOURCE_NOISE_SEED = 900020
SOURCE_NOISE_FILTER_LENGTH = 513
N_BEAM = 121
DESIGN_FFT_SIZE = 1024
DIFF_FIR_TAPS = 512
DIAGONAL_LOADING_RATIO = 1.0e-2
COVARIANCE_NOISE_POWER = 1.0e-4
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "fixed_delay_diff_mvdr" / "broadband_20deg_9000hz"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
LEVEL_UNIT_LABEL = "dB re input RMS"


@dataclass(frozen=True)
class EvaluationResult:
    """広帯域 scene_renderer 評価の配列と metric を保持する。

    このクラスは、入力信号、beamforming 出力、周波数応答、診断 metric を
    Markdown/CSV/NPZ/PNG へ保存する前の中間表現としてまとめる。

    信号生成、MVDR 重み設計、図の描画は責務に含めない。
    信号処理上は、固定整相+差分 MVDR の 1 scenario 評価結果である。
    """

    frequency_hz: FloatArray
    design_frequency_hz: FloatArray
    azimuth_deg: FloatArray
    target_beam_index: int
    input_signal: FloatArray
    fixed_target_output: FloatArray
    diff_target_output: FloatArray
    input_spectrum_level_db_by_channel: FloatArray
    input_mean_spectrum_level_db: FloatArray
    fixed_output_spectrum_level_db: FloatArray
    diff_output_spectrum_level_db: FloatArray
    fixed_beam_response_center_db: FloatArray
    diff_beam_response_center_db: FloatArray
    fixed_beam_response_band_db: FloatArray
    diff_beam_response_band_db: FloatArray
    q_reconstruction_rms_error_by_beam: FloatArray
    target_response_error_abs_by_beam: FloatArray
    loaded_condition_number_max_by_beam: FloatArray
    fallback_mask_by_bin_beam: NDArray[np.bool_]


def _build_array_positions() -> FloatArray:
    """scene_renderer と spflow で共有する ULA 位置を返す。

    Returns:
        センサ位置。shape は `[n_ch, 3]`、axis=0 は CH、axis=1 は
        `[Bow, Starboard, Up]`、単位は m。
    """

    positions = np.zeros((N_CH, 3), dtype=np.float64)
    positions[:, 0] = (np.arange(N_CH, dtype=np.float64) - 0.5 * (N_CH - 1)) * SPACING_M
    return positions


def _direction_from_azimuth(azimuth_deg: float) -> FloatArray:
    """相対方位から ArrayFrame の水平面方向余弦を返す。

    Args:
        azimuth_deg: 相対方位。単位は deg。0 deg は Bow、90 deg は Starboard。

    Returns:
        方向余弦。shape は `[3]`、無次元。
    """

    azimuth_rad = np.deg2rad(float(azimuth_deg))
    return np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0], dtype=np.float64)


def _receive_steering(array_positions_m: FloatArray, direction: FloatArray, frequencies_hz: FloatArray) -> ComplexArray:
    """scene_renderer 観測信号と同じ符号規約の steering を返す。

    Args:
        array_positions_m: センサ位置。shape は `[n_ch, 3]`、単位は m。
        direction: 入射方向余弦。shape は `[3]`、無次元。
        frequencies_hz: signed DFT 周波数。shape は `[n_bin]`、単位は Hz。

    Returns:
        受信 steering。shape は `[n_bin, n_ch]`。

    Notes:
        scene_renderer の広帯域 projector は CH 間遅延
        `tau[ch] = r[ch]^T u / c` を時間領域 fractional delay で表す。
        narrowband 表現では観測位相が `exp(-j 2π f tau[ch])` になるため、
        MVDR の制約ベクトルも同じ符号で定義する。
    """

    tau_sec = (array_positions_m @ direction) / SOUND_SPEED_M_S
    phase = -1j * 2.0 * np.pi * frequencies_hz[:, np.newaxis] * tau_sec[np.newaxis, :]
    return np.asarray(np.exp(phase), dtype=np.complex128)


def _one_sided_rms_spectrum_level_db(samples: NDArray[Any]) -> tuple[FloatArray, FloatArray]:
    """実信号の one-sided FFT を per-bin RMS level に変換する。

    Args:
        samples: 実信号。shape は `[n_series, n_sample]` または `[n_sample]`。

    Returns:
        `(frequency_hz, level_db)`。
        `frequency_hz` の shape は `[n_rfft_bin]`、単位は Hz。
        `level_db` の shape は `[n_series, n_rfft_bin]`。

    境界条件:
        DC と Nyquist は one-sided の 2 倍補正を掛けない。その他の正周波数 bin は
        負周波数側の power を畳み込むため、`2 * |X/N|^2` を RMS power として使う。
    """

    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError("samples must have shape [n_series, n_sample] or [n_sample].")
    n_sample = int(arr.shape[1])
    spectrum = np.fft.rfft(arr, axis=1)
    factor = np.ones(spectrum.shape[1], dtype=np.float64)
    if n_sample % 2 == 0:
        factor[1:-1] = 2.0
    else:
        factor[1:] = 2.0
    power = factor[np.newaxis, :] * np.abs(spectrum / float(n_sample)) ** 2
    level_db = 10.0 * np.log10(np.maximum(power, np.finfo(np.float64).tiny))
    frequency_hz = np.asarray(np.fft.rfftfreq(n_sample, d=1.0 / FS_HZ), dtype=np.float64)
    return frequency_hz, np.asarray(level_db, dtype=np.float64)


def _render_broadband_scene() -> FloatArray:
    """20 deg 方向の帯域制限広帯域信号を scene_renderer で描画する。

    Returns:
        多 CH 入力信号。shape は `[n_ch, n_sample]`、単位は normalized amplitude。
    """

    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=N_CH, spacing=SPACING_M, axis=0, centered=True),
    )
    component = SourceComponent(
        spectrum=BandLimitedNoiseSpectrum(SOURCE_BAND_LOW_HZ, SOURCE_BAND_HIGH_HZ),
        envelope=ConstantEnvelope(),
        amplitude=None,
        level_db=SOURCE_LEVEL_DB,
        noise_seed=SOURCE_NOISE_SEED,
        noise_filter_length=SOURCE_NOISE_FILTER_LENGTH,
    )
    source = AcousticSource.from_relative_bearing(
        bearing_deg=SOURCE_AZIMUTH_DEG,
        distance=1000.0,
        receiver_pose=receiver.trajectory.pose(0.0),
        components=[component],
        elevation_deg=0.0,
    )
    scene = Scene(sources=[source], ambient_fields=[], environment=FreeField(c=SOUND_SPEED_M_S))
    axis_t = np.arange(N_SAMPLE, dtype=np.float64) / FS_HZ
    rendered = SceneRenderer().render(scene, receiver, axis_t)
    return np.asarray(np.real(rendered), dtype=np.float64)


def _design_fixed_and_diff_weights(
    array_positions_m: FloatArray,
    azimuth_deg: FloatArray,
    beam_directions: FloatArray,
    frequencies_hz: FloatArray,
) -> tuple[ComplexArray, ComplexArray, FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    """固定整相重みと diff-MVDR FIR512 後の最終重みを設計する。

    Args:
        array_positions_m: センサ位置。shape は `[n_ch, 3]`、単位は m。
        azimuth_deg: beam 方位軸。shape は `[n_beam]`、単位は deg。
        beam_directions: beam 方向余弦。shape は `[n_beam, 3]`。
        frequencies_hz: signed DFT 周波数。shape は `[n_bin]`、単位は Hz。

    Returns:
        `(fixed_weights, diff_weights, q_error, target_response_error, condition, fallback_mask)`。
        weight shape は `[n_bin, n_beam, n_ch]`。
    """

    fractional_filter_bank = design_standard_fractional_delay_filter_bank()
    # scene_renderer の受信位相 exp(-j 2π f r^T u/c) に対して固定整相で遅延補償するには、
    # DelayTable の補償方向へ -u を渡す。+u では反対側 beam にピークが出ることを避けるためである。
    delay_table = DelayTable.from_geometry(
        array_pos_m=array_positions_m,
        dir_cos=-beam_directions,
        fs_hz=FS_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        fractional_filter_bank=fractional_filter_bank,
    )
    fixed_weights = design_fixed_delay_fractional_weights_from_delay_table(
        delay_table,
        fractional_filter_bank,
        frequencies_hz,
        fs_hz=FS_HZ,
        average_channels=True,
    )

    beam_steering = np.stack(
        [_receive_steering(array_positions_m, direction, frequencies_hz) for direction in beam_directions],
        axis=1,
    )
    source_direction = _direction_from_azimuth(SOURCE_AZIMUTH_DEG)
    source_steering = _receive_steering(array_positions_m, source_direction, frequencies_hz)
    target_beam_index = int(np.argmin(np.abs(azimuth_deg - SOURCE_AZIMUTH_DEG)))
    protected_steering = beam_steering.copy()
    protected_steering[:, target_beam_index, :] = source_steering

    covariance = _make_broadband_covariance(source_steering, frequencies_hz)
    loaded_covariance, condition_number = _load_covariance(covariance)
    # RHS shape は `[n_bin, n_ch, n_beam]`。
    # np.linalg.solve は先頭 axis を batch として扱うため、全周波数・全 beam の
    # `R_load[k] u[k,beam] = a[k,beam]` をまとめて解く。
    solved_by_channel_beam = np.linalg.solve(loaded_covariance, np.swapaxes(protected_steering, 1, 2))
    solved_by_beam_channel = np.swapaxes(solved_by_channel_beam, 1, 2)
    denominator = np.sum(protected_steering.conj() * solved_by_beam_channel, axis=2)
    desired_response = np.sum(fixed_weights.conj() * protected_steering, axis=2)
    mvdr_weights = np.asarray(
        np.conj(desired_response)[:, :, np.newaxis]
        * solved_by_beam_channel
        / denominator[:, :, np.newaxis],
        dtype=np.complex128,
    )
    fallback_mask = np.logical_or(
        np.abs(denominator) <= 1.0e-12,
        np.logical_not(np.all(np.isfinite(mvdr_weights), axis=2)),
    )
    mvdr_weights = np.where(fallback_mask[:, :, np.newaxis], fixed_weights, mvdr_weights)

    diff_designer = DifferenceCorrectionFIRDesigner(
        fir_taps=DIFF_FIR_TAPS,
        frequencies_hz=frequencies_hz,
        fs_hz=FS_HZ,
    )
    diff_weights = np.empty_like(fixed_weights)
    q_error = np.zeros((frequencies_hz.size, azimuth_deg.size), dtype=np.float64)
    target_response_error = np.zeros((frequencies_hz.size, azimuth_deg.size), dtype=np.float64)
    for beam_index in range(azimuth_deg.size):
        diff_result = diff_designer.compute(
            fixed_weights[:, beam_index, :],
            mvdr_weights[:, beam_index, :],
            protected_steering[:, beam_index, :],
        )
        diff_weights[:, beam_index, :] = diff_result.final_weight_freq
        q_error[:, beam_index] = np.sqrt(
            np.mean(np.abs(diff_result.diagnostics.q_reconstruction_error) ** 2, axis=1)
        )
        target_response_error[:, beam_index] = np.abs(
            diff_result.diagnostics.target_response_final
            - diff_result.diagnostics.target_response_w0
        )
    return fixed_weights, diff_weights, q_error, target_response_error, condition_number, fallback_mask


def _make_broadband_covariance(source_steering: ComplexArray, frequencies_hz: FloatArray) -> ComplexArray:
    """広帯域 source を持つ周波数別空間共分散を作る。

    Args:
        source_steering: source steering。shape は `[n_bin, n_ch]`。
        frequencies_hz: signed DFT 周波数。shape は `[n_bin]`、単位は Hz。

    Returns:
        共分散。shape は `[n_bin, n_ch, n_ch]`。
    """

    n_bin, n_ch = source_steering.shape
    eye = np.eye(n_ch, dtype=np.complex128)
    covariance = COVARIANCE_NOISE_POWER * np.repeat(eye[np.newaxis, :, :], n_bin, axis=0)
    passband_mask = (SOURCE_BAND_LOW_HZ <= np.abs(frequencies_hz)) & (np.abs(frequencies_hz) <= SOURCE_BAND_HIGH_HZ)
    source_power = 10.0 ** (SOURCE_LEVEL_DB / 10.0)
    for bin_index in np.flatnonzero(passband_mask).tolist():
        steering = source_steering[int(bin_index)]
        # 広帯域 source の各通過 bin で R[k] = sigma_s^2 a[k]a[k]^H + sigma_n^2 I とする。
        covariance[int(bin_index)] += source_power * np.outer(steering, steering.conj())
    return np.asarray(covariance, dtype=np.complex128)


def _load_covariance(covariance: ComplexArray) -> tuple[ComplexArray, FloatArray]:
    """対角ローディング済み共分散と条件数を返す。

    Args:
        covariance: 共分散。shape は `[n_bin, n_ch, n_ch]`。

    Returns:
        `(loaded_covariance, condition_number)`。
    """

    n_ch = int(covariance.shape[1])
    average_power = np.real(np.trace(covariance, axis1=1, axis2=2)) / float(n_ch)
    loading_power = DIAGONAL_LOADING_RATIO * np.where(average_power > 0.0, average_power, 1.0)
    loaded = covariance + loading_power[:, np.newaxis, np.newaxis] * np.eye(n_ch, dtype=np.complex128)[np.newaxis]
    return np.asarray(loaded, dtype=np.complex128), np.asarray(np.linalg.cond(loaded), dtype=np.float64)


def _apply_weights_to_rfft(
    input_signal: FloatArray,
    weights: ComplexArray,
    design_frequency_hz: FloatArray,
) -> tuple[FloatArray, ComplexArray]:
    """rFFT 入力へ周波数別重みを掛けて beam 出力スペクトルを返す。

    Args:
        input_signal: 多 CH 実信号。shape は `[n_ch, n_sample]`。
        weights: beamforming 重み。shape は `[n_design_bin, n_beam, n_ch]`。
        design_frequency_hz: signed 設計周波数。shape は `[n_design_bin]`、単位は Hz。

    Returns:
        `(rfft_frequency_hz, output_spectrum)`。
        `output_spectrum` の shape は `[n_beam, n_rfft_bin]`。
    """

    n_sample = int(input_signal.shape[1])
    rfft_frequency_hz = np.asarray(np.fft.rfftfreq(n_sample, d=1.0 / FS_HZ), dtype=np.float64)
    input_spectrum = np.fft.rfft(input_signal, axis=1)
    design_indices = np.asarray(
        [int(np.argmin(np.abs(design_frequency_hz - float(freq)))) for freq in rfft_frequency_hz],
        dtype=np.int64,
    )
    output_spectrum = np.zeros((weights.shape[1], rfft_frequency_hz.size), dtype=np.complex128)
    selected_weights = weights[design_indices]
    for beam_index in range(weights.shape[1]):
        # selected_weights[:, beam, :] shape は `[n_rfft_bin, n_ch]`。
        # beamforming 規約 `Y[k] = w[k]^H X[k]` に合わせ、CH 軸を内積として畳み込む。
        output_spectrum[beam_index] = np.einsum(
            "fc,cf->f",
            selected_weights[:, beam_index, :].conj(),
            input_spectrum,
            optimize=True,
        )
    return rfft_frequency_hz, output_spectrum


def _spectrum_level_from_rfft(spectrum: ComplexArray, n_sample: int) -> FloatArray:
    """rFFT スペクトルを one-sided per-bin RMS level へ変換する。

    Args:
        spectrum: rFFT スペクトル。shape は `[n_series, n_rfft_bin]`。
        n_sample: 元信号長。単位は sample。

    Returns:
        レベル。shape は `[n_series, n_rfft_bin]`、単位は dB re input RMS。
    """

    factor = np.ones(spectrum.shape[1], dtype=np.float64)
    factor[1:-1] = 2.0
    power = factor[np.newaxis, :] * np.abs(spectrum / float(n_sample)) ** 2
    return np.asarray(10.0 * np.log10(np.maximum(power, np.finfo(np.float64).tiny)), dtype=np.float64)


def _band_integrated_level_db(spectrum: ComplexArray, frequency_hz: FloatArray, n_sample: int) -> FloatArray:
    """source 通過帯域を power 積分した beam level を返す。

    Args:
        spectrum: beam 出力 rFFT。shape は `[n_beam, n_rfft_bin]`。
        frequency_hz: rFFT 周波数。shape は `[n_rfft_bin]`、単位は Hz。
        n_sample: 元信号長。単位は sample。

    Returns:
        beam ごとの帯域積分 level。shape は `[n_beam]`。
    """

    passband_mask = (SOURCE_BAND_LOW_HZ <= frequency_hz) & (frequency_hz <= SOURCE_BAND_HIGH_HZ)
    factor = np.ones(frequency_hz.size, dtype=np.float64)
    factor[1:-1] = 2.0
    power = factor[np.newaxis, :] * np.abs(spectrum / float(n_sample)) ** 2
    band_power = np.sum(power[:, passband_mask], axis=1)
    return np.asarray(10.0 * np.log10(np.maximum(band_power, np.finfo(np.float64).tiny)), dtype=np.float64)


def _evaluate() -> EvaluationResult:
    """scene_renderer 広帯域入力を生成し、固定整相+diff MVDR を評価する。"""

    array_positions_m = _build_array_positions()
    directions, axis_azimuth_deg, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=N_BEAM,
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    beam_directions = directions.T.astype(np.float64)
    design_frequency_hz = np.asarray(np.fft.fftfreq(DESIGN_FFT_SIZE, d=1.0 / FS_HZ), dtype=np.float64)
    fixed_weights, diff_weights, q_error, target_response_error, condition_number, fallback_mask = (
        _design_fixed_and_diff_weights(
            array_positions_m=array_positions_m,
            azimuth_deg=axis_azimuth_deg.astype(np.float64),
            beam_directions=beam_directions,
            frequencies_hz=design_frequency_hz,
        )
    )
    input_signal = _render_broadband_scene()
    frequency_hz, input_level_db_by_channel = _one_sided_rms_spectrum_level_db(input_signal)
    fixed_frequency_hz, fixed_spectrum = _apply_weights_to_rfft(input_signal, fixed_weights, design_frequency_hz)
    diff_frequency_hz, diff_spectrum = _apply_weights_to_rfft(input_signal, diff_weights, design_frequency_hz)
    if not np.array_equal(frequency_hz, fixed_frequency_hz) or not np.array_equal(frequency_hz, diff_frequency_hz):
        raise ValueError("internal frequency axes are inconsistent.")

    fixed_level_db = _spectrum_level_from_rfft(fixed_spectrum, N_SAMPLE)
    diff_level_db = _spectrum_level_from_rfft(diff_spectrum, N_SAMPLE)
    center_index = int(np.argmin(np.abs(frequency_hz - SOURCE_CENTER_HZ)))
    target_beam_index = int(np.argmin(np.abs(axis_azimuth_deg.astype(np.float64) - SOURCE_AZIMUTH_DEG)))
    fixed_target_output = np.asarray(np.fft.irfft(fixed_spectrum[target_beam_index], n=N_SAMPLE), dtype=np.float64)
    diff_target_output = np.asarray(np.fft.irfft(diff_spectrum[target_beam_index], n=N_SAMPLE), dtype=np.float64)

    return EvaluationResult(
        frequency_hz=frequency_hz,
        design_frequency_hz=design_frequency_hz,
        azimuth_deg=axis_azimuth_deg.astype(np.float64),
        target_beam_index=target_beam_index,
        input_signal=np.asarray(input_signal, dtype=np.float64),
        fixed_target_output=fixed_target_output,
        diff_target_output=diff_target_output,
        input_spectrum_level_db_by_channel=input_level_db_by_channel,
        input_mean_spectrum_level_db=np.asarray(
            10.0
            * np.log10(
                np.maximum(
                    np.mean(np.power(10.0, input_level_db_by_channel / 10.0), axis=0),
                    np.finfo(np.float64).tiny,
                )
            ),
            dtype=np.float64,
        ),
        fixed_output_spectrum_level_db=fixed_level_db[target_beam_index],
        diff_output_spectrum_level_db=diff_level_db[target_beam_index],
        fixed_beam_response_center_db=fixed_level_db[:, center_index],
        diff_beam_response_center_db=diff_level_db[:, center_index],
        fixed_beam_response_band_db=_band_integrated_level_db(fixed_spectrum, frequency_hz, N_SAMPLE),
        diff_beam_response_band_db=_band_integrated_level_db(diff_spectrum, frequency_hz, N_SAMPLE),
        q_reconstruction_rms_error_by_beam=np.asarray(np.max(q_error, axis=0), dtype=np.float64),
        target_response_error_abs_by_beam=np.asarray(np.max(target_response_error, axis=0), dtype=np.float64),
        loaded_condition_number_max_by_beam=np.asarray(
            np.repeat(np.max(condition_number), axis_azimuth_deg.size), dtype=np.float64
        ),
        fallback_mask_by_bin_beam=fallback_mask,
    )


def _plt() -> Any:
    """matplotlib.pyplot module を返す。"""

    require_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is required.")
    return plt


def _finite_ylim(level_arrays: list[FloatArray], *, dynamic_range_db: float = 100.0) -> tuple[float, float]:
    """表示用の y 軸範囲を有限値から決める。"""

    values = np.concatenate([np.asarray(level, dtype=np.float64).ravel() for level in level_arrays])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -120.0, 0.0
    upper = float(np.max(finite) + 3.0)
    lower = max(float(np.min(finite) - 3.0), upper - float(dynamic_range_db))
    return lower, upper


def _plot_input_spectrum(result: EvaluationResult, output_path: Path) -> None:
    """入力信号の FFT 周波数スペクトルを保存する。"""

    fig, axis = _plt().subplots(figsize=(10.5, 4.8))
    axis.plot(result.frequency_hz, result.input_mean_spectrum_level_db, color="black", linewidth=1.0, label="channel mean")
    axis.axvspan(SOURCE_BAND_LOW_HZ, SOURCE_BAND_HIGH_HZ, color="tab:green", alpha=0.15, label="source passband")
    axis.axvline(SOURCE_CENTER_HZ, color="tab:green", linestyle="--", linewidth=1.0, label="center 9000 Hz")
    axis.set_xlim(0.0, FS_HZ / 2.0)
    axis.set_ylim(*_finite_ylim([result.input_mean_spectrum_level_db], dynamic_range_db=90.0))
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Per-bin RMS Level [{LEVEL_UNIT_LABEL}]")
    axis.set_title("Input signal FFT spectrum: 20 deg broadband source")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _plot_beam_response(result: EvaluationResult, output_path: Path) -> None:
    """整相後出力 FFT から beam response を保存する。"""

    fig, axes = _plt().subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    for axis, fixed, diff, title in (
        (
            axes[0],
            result.fixed_beam_response_center_db,
            result.diff_beam_response_center_db,
            "Nearest 9000 Hz FFT bin beam response",
        ),
        (
            axes[1],
            result.fixed_beam_response_band_db,
            result.diff_beam_response_band_db,
            "8500-9500 Hz band-integrated beam response",
        ),
    ):
        axis.plot(result.azimuth_deg, fixed, color="black", label="fixed_baseline")
        axis.plot(result.azimuth_deg, diff, color="tab:orange", label="diff_mvdr_fir512")
        axis.axvline(SOURCE_AZIMUTH_DEG, color="tab:green", linestyle="--", linewidth=1.0, label="source 20 deg")
        axis.set_ylim(*_finite_ylim([fixed, diff], dynamic_range_db=70.0))
        axis.set_ylabel(f"RMS Level [{LEVEL_UNIT_LABEL}]")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    axes[1].set_xlabel("Beam azimuth [deg]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _plot_output_spectrum(result: EvaluationResult, output_path: Path) -> None:
    """20 deg beam の整相後信号 FFT 周波数スペクトルを保存する。"""

    fig, axis = _plt().subplots(figsize=(10.5, 4.8))
    axis.plot(result.frequency_hz, result.fixed_output_spectrum_level_db, color="black", linewidth=1.0, label="fixed_baseline")
    axis.plot(result.frequency_hz, result.diff_output_spectrum_level_db, color="tab:orange", linewidth=1.0, label="diff_mvdr_fir512")
    axis.axvspan(SOURCE_BAND_LOW_HZ, SOURCE_BAND_HIGH_HZ, color="tab:green", alpha=0.15, label="source passband")
    axis.axvline(SOURCE_CENTER_HZ, color="tab:green", linestyle="--", linewidth=1.0, label="center 9000 Hz")
    axis.set_xlim(0.0, FS_HZ / 2.0)
    axis.set_ylim(*_finite_ylim([result.fixed_output_spectrum_level_db, result.diff_output_spectrum_level_db], dynamic_range_db=90.0))
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Per-bin RMS Level [{LEVEL_UNIT_LABEL}]")
    axis.set_title("Post-beamforming FFT spectrum: 20 deg beam")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """CSV を保存する。"""

    if not rows:
        raise ValueError("rows must not be empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(result: EvaluationResult) -> list[dict[str, object]]:
    """AI 向け summary metric 行を作る。"""

    fixed_peak_index = int(np.argmax(result.fixed_beam_response_band_db))
    diff_peak_index = int(np.argmax(result.diff_beam_response_band_db))
    target_index = int(result.target_beam_index)
    passband_mask = (SOURCE_BAND_LOW_HZ <= result.frequency_hz) & (result.frequency_hz <= SOURCE_BAND_HIGH_HZ)
    input_band_level = _power_integrated_level(result.input_mean_spectrum_level_db[passband_mask])
    rows: list[dict[str, object]] = []
    for method, beam_response, spectrum in (
        ("fixed_baseline", result.fixed_beam_response_band_db, result.fixed_output_spectrum_level_db),
        ("diff_mvdr_fir512", result.diff_beam_response_band_db, result.diff_output_spectrum_level_db),
    ):
        peak_index = fixed_peak_index if method == "fixed_baseline" else diff_peak_index
        rows.append(
            {
                "scenario": "broadband_20deg_9000hz",
                "method": method,
                "evaluation_pattern": "fixed_beam_single_source",
                "source_azimuth_deg": SOURCE_AZIMUTH_DEG,
                "source_center_hz": SOURCE_CENTER_HZ,
                "source_band_low_hz": SOURCE_BAND_LOW_HZ,
                "source_band_high_hz": SOURCE_BAND_HIGH_HZ,
                "level_unit": LEVEL_UNIT_LABEL,
                "input_band_integrated_level_db": input_band_level,
                "target_beam_azimuth_deg": float(result.azimuth_deg[target_index]),
                "band_peak_azimuth_deg": float(result.azimuth_deg[peak_index]),
                "band_peak_azimuth_error_deg": float(result.azimuth_deg[peak_index] - SOURCE_AZIMUTH_DEG),
                "target_beam_band_level_db": float(beam_response[target_index]),
                "peak_band_level_db": float(beam_response[peak_index]),
                "target_output_band_integrated_level_db": _power_integrated_level(spectrum[passband_mask]),
                "q_reconstruction_rms_error_max": float(np.max(result.q_reconstruction_rms_error_by_beam)),
                "target_response_error_abs_max": float(np.max(result.target_response_error_abs_by_beam)),
                "loaded_condition_number_max": float(np.max(result.loaded_condition_number_max_by_beam)),
                "fallback_bin_beam_count": int(np.count_nonzero(result.fallback_mask_by_bin_beam)),
            }
        )
    return rows


def _power_integrated_level(level_db: FloatArray) -> float:
    """dB per-bin level を線形 power 和へ戻して積分 level を返す。"""

    power = np.sum(np.power(10.0, np.asarray(level_db, dtype=np.float64) / 10.0))
    return float(10.0 * np.log10(max(float(power), np.finfo(np.float64).tiny)))


def _save_npz(result: EvaluationResult, path: Path) -> None:
    """AI 向け解析配列を NPZ へ保存する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frequency_hz=result.frequency_hz,
        design_frequency_hz=result.design_frequency_hz,
        azimuth_deg=result.azimuth_deg,
        target_beam_index=np.asarray([result.target_beam_index], dtype=np.int64),
        input_signal=result.input_signal,
        fixed_target_output=result.fixed_target_output,
        diff_target_output=result.diff_target_output,
        input_spectrum_level_db_by_channel=result.input_spectrum_level_db_by_channel,
        input_mean_spectrum_level_db=result.input_mean_spectrum_level_db,
        fixed_output_spectrum_level_db=result.fixed_output_spectrum_level_db,
        diff_output_spectrum_level_db=result.diff_output_spectrum_level_db,
        fixed_beam_response_center_db=result.fixed_beam_response_center_db,
        diff_beam_response_center_db=result.diff_beam_response_center_db,
        fixed_beam_response_band_db=result.fixed_beam_response_band_db,
        diff_beam_response_band_db=result.diff_beam_response_band_db,
        q_reconstruction_rms_error_by_beam=result.q_reconstruction_rms_error_by_beam,
        target_response_error_abs_by_beam=result.target_response_error_abs_by_beam,
        loaded_condition_number_max_by_beam=result.loaded_condition_number_max_by_beam,
        fallback_mask_by_bin_beam=result.fallback_mask_by_bin_beam,
    )


def _write_metadata(result: EvaluationResult, output_dir: Path) -> None:
    """package metadata JSON を保存する。"""

    metadata = {
        "scenario_id": "broadband_20deg_9000hz",
        "purpose": "scene_renderer broadband source を固定整相+diff MVDR FIR512 に入力した単一 source 評価。",
        "evaluation_pattern": "fixed_beam_single_source",
        "fs_hz": FS_HZ,
        "duration_s": DURATION_S,
        "n_sample": N_SAMPLE,
        "n_ch": N_CH,
        "spacing_m": SPACING_M,
        "sound_speed_m_s": SOUND_SPEED_M_S,
        "source": {
            "azimuth_deg": SOURCE_AZIMUTH_DEG,
            "center_hz": SOURCE_CENTER_HZ,
            "band_low_hz": SOURCE_BAND_LOW_HZ,
            "band_high_hz": SOURCE_BAND_HIGH_HZ,
            "level_db": SOURCE_LEVEL_DB,
            "noise_seed": SOURCE_NOISE_SEED,
            "noise_filter_length": SOURCE_NOISE_FILTER_LENGTH,
        },
        "methods": ["fixed_baseline", "diff_mvdr_fir512"],
        "design_fft_size": DESIGN_FFT_SIZE,
        "diff_fir_taps": DIFF_FIR_TAPS,
        "level_unit": LEVEL_UNIT_LABEL,
        "array_shapes": {
            "input_signal": "[n_ch, n_sample]",
            "input_spectrum_level_db_by_channel": "[n_ch, n_rfft_bin]",
            "beam_response_*_db": "[n_beam]",
            "output_spectrum_level_db": "[n_rfft_bin]",
            "fallback_mask_by_bin_beam": "[n_design_bin, n_beam]",
        },
        "target_beam_index": int(result.target_beam_index),
        "target_beam_azimuth_deg": float(result.azimuth_deg[result.target_beam_index]),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_review_index(output_dir: Path) -> None:
    """AI 向け review index を保存する。"""

    lines = [
        "# 固定整相 + diff MVDR FIR512 広帯域 scene_renderer 評価",
        "",
        "## Scenario",
        "",
        "- scenario_id: `broadband_20deg_9000hz`",
        "- source: 20 deg, center 9000 Hz, passband 8500-9500 Hz",
        f"- sampling: fs={FS_HZ:.1f} Hz, duration={DURATION_S:.3f} s",
        "- evaluation_pattern: `fixed_beam_single_source`",
        f"- level reference: `{LEVEL_UNIT_LABEL}`",
        "",
        "## Artifacts",
        "",
        "- `figures/input_frequency_spectrum.png`: 入力多 CH 信号を FFT した channel 平均 spectrum。",
        "- `figures/beam_response_9000hz.png`: 整相後 FFT から作った 9000 Hz 近傍 bin と 8500-9500 Hz 積分 beam response。",
        "- `figures/output_frequency_spectrum_20deg.png`: 20 deg beam の整相後信号 FFT spectrum。",
        "- `data/broadband_arrays.npz`: PNG 作成元配列。shape は `metadata.json` を参照。",
        "- `scenario_summary.csv`: method 別 peak 方位、band level、diagnostic metric。",
        "- `metadata.json`: 評価条件、単位、shape、method 定義。",
        "",
        "## Interpretation Notes",
        "",
        "- BL/beam response は fixed_baseline と diff_mvdr_fir512 を併記する。",
        "- `beam_response_9000hz.png` の下段は広帯域 passband power を積分した応答であり、単一 bin の偶然変動だけを見ないための補助である。",
        "- BTR は本評価要求に含まれないため生成していない。時間 track 連続性評価には使わない。",
    ]
    (output_dir / "review_index.md").write_text("\n".join(lines), encoding="utf-8")


def _zip_package(output_dir: Path) -> Path:
    """出力ディレクトリを zip 化する。"""

    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                package.write(path, path.relative_to(output_dir.parent))
    return zip_path


def build_report_package() -> Path:
    """評価を実行し、人間向け PNG と AI 向け report package を保存する。"""

    require_matplotlib()
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    result = _evaluate()
    _plot_input_spectrum(result, FIGURE_DIR / "input_frequency_spectrum.png")
    _plot_beam_response(result, FIGURE_DIR / "beam_response_9000hz.png")
    _plot_output_spectrum(result, FIGURE_DIR / "output_frequency_spectrum_20deg.png")
    _save_npz(result, DATA_DIR / "broadband_arrays.npz")
    _write_csv(OUTPUT_DIR / "scenario_summary.csv", _summary_rows(result))
    _write_metadata(result, OUTPUT_DIR)
    _write_review_index(OUTPUT_DIR)
    return _zip_package(OUTPUT_DIR)


def main() -> None:
    """CLI entrypoint。"""

    zip_path = build_report_package()
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "zip_path": str(zip_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
