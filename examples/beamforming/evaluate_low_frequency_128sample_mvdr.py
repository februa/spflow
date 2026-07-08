"""低周波・128 sample 共分散での MVDR 安定性レポートを作る。"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from scene_renderer_mvdr_stability_sweep import (  # noqa: E402
    build_array_design,
    evaluate_frequency,
    render_scene,
    steering_from_dir3d,
)
from spflow.beamforming import (  # noqa: E402
    design_cbf_weights_with_channel_window,
    design_mvdr_weights_with_channel_window,
    make_directions,
)
from spflow.beamforming.diagnostic_plotting import require_matplotlib  # noqa: E402

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 実行環境に依存する。
    plt = None


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
N_CH = 32
SPACING_M = 0.05
ACTIVE_APERTURE_M = SPACING_M * float(N_CH - 1)
TARGET_AZIMUTH_DEG = 20.0
INTERFERER_AZIMUTH_DEG = -30.0
SIGNAL_LEVEL_DB20 = 0.0
INTERFERER_LEVEL_DB20 = 0.0
FFT_SIZE = 128
N_SAMPLE = 128
INTEGRATION_TIME_S = float(N_SAMPLE) / FS_HZ
DIAGONAL_LOADING_RATIO = 1.0e-3
FREQUENCIES_HZ = (256.0, 512.0, 1024.0, 2048.0, 4096.0, 8960.0)
N_BEAM = 121
BROADBAND_CASES = (
    ("low_256_1024hz", 256.0, 1024.0, 2561024),
    ("high_8500_9500hz", 8500.0, 9500.0, 85009500),
)
BROADBAND_NOISE_FILTER_LENGTH = 513
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "fixed_delay_diff_mvdr" / "low_frequency_128sample_mvdr"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
LEVEL_UNIT_LABEL = "dB re input RMS"


@dataclass(frozen=True)
class FrequencyRow:
    """1 周波数・1 共分散条件の評価行を保持する。

    このクラスは、scene_renderer で生成した target + interferer 入力に対し、
    128 sample 共分散から設計した MVDR の応答指標を CSV/PNG へ渡す中間表現である。

    信号生成、MVDR 重み設計、図の描画は責務に含めない。
    信号処理上は、短時間共分散で低周波 MVDR が分離不能になるかを読むための
    scenario-by-method 行である。
    """

    covariance_source: str
    frequency_hz: float
    wavelength_m: float
    aperture_wavelength: float
    adjacent_phase_deg: float
    aperture_phase_deg: float
    cbf_interferer_db: float
    mvdr_interferer_db: float
    interferer_reduction_db: float
    cbf_rms_err: float
    mvdr_rms_err: float
    mvdr_improves_target_err: bool
    active_alias_limit_hz: float


@dataclass(frozen=True)
class BroadbandCaseResult:
    """128 sample 広帯域 beam response 評価結果を保持する。

    このクラスは、低周波または高周波の帯域制限 noise source について、
    128 sample の標本共分散から設計した MVDR と固定整相の出力を保持する。

    信号生成、重み設計、図の描画は責務に含めない。
    信号処理上は、短時間共分散で得た beam response と FFT spectrum を
    同じ dB reference で比較するための中間表現である。
    """

    scenario_id: str
    band_low_hz: float
    band_high_hz: float
    frequency_hz: FloatArray
    azimuth_deg: FloatArray
    target_beam_index: int
    input_mean_spectrum_level_db: FloatArray
    fixed_target_spectrum_level_db: FloatArray
    mvdr_target_spectrum_level_db: FloatArray
    fixed_band_response_db: FloatArray
    mvdr_band_response_db: FloatArray
    loaded_condition_number_by_bin: FloatArray



def _plt():
    """matplotlib.pyplot を遅延取得する。"""

    if plt is None:
        raise RuntimeError("matplotlib is required to plot figures.")
    return plt


def _evaluate_rows() -> list[FrequencyRow]:
    """低周波 sweep を実行して評価行を返す。

    Returns:
        評価行の list。各行は 1 周波数・1 共分散条件を表す。

    Notes:
        `mixture` は target と interferer を含む 128 sample だけから共分散を作るため、
        実運用で target が統計に混入する条件を表す。`interferer-only` は理想参照であり、
        低周波でも MVDR が動ける上限性能として併記する。
    """

    rows: list[FrequencyRow] = []
    for covariance_source in ("mixture", "interferer-only"):
        for frequency_hz in FREQUENCIES_HZ:
            raw_row = evaluate_frequency(
                fs=FS_HZ,
                fft_size=FFT_SIZE,
                freq=float(frequency_hz),
                n_samples=N_SAMPLE,
                n_ch=N_CH,
                spacing_m=SPACING_M,
                sound_speed=SOUND_SPEED_M_S,
                target_deg=TARGET_AZIMUTH_DEG,
                signal_level_db20=SIGNAL_LEVEL_DB20,
                integration_time=INTEGRATION_TIME_S,
                diag_load=DIAGONAL_LOADING_RATIO,
                interferer_deg=INTERFERER_AZIMUTH_DEG,
                interferer_level_db20=INTERFERER_LEVEL_DB20,
                covariance_source=covariance_source,
                selector_mode="full",
                aperture_wavelengths=4.0,
                min_active_ch=4,
                dense_spacing_m=None,
                n_dense_ch=None,
            )
            if raw_row["status"] != "ok":
                continue

            wavelength_m = SOUND_SPEED_M_S / float(frequency_hz)
            # ULA 隣接 CH の位相差は 2π f d cos(theta) / c。
            # 低周波ではこの値が小さく、CH 間 steering が似るため空間分離が難しくなる。
            adjacent_phase_deg = float(
                np.rad2deg(
                    2.0
                    * np.pi
                    * float(frequency_hz)
                    * SPACING_M
                    * np.cos(np.deg2rad(TARGET_AZIMUTH_DEG))
                    / SOUND_SPEED_M_S
                )
            )
            aperture_phase_deg = adjacent_phase_deg * float(N_CH - 1)
            cbf_interferer_db = float(raw_row["cbf_interferer_db"])
            mvdr_interferer_db = float(raw_row["mvdr_interferer_db"])
            rows.append(
                FrequencyRow(
                    covariance_source=covariance_source,
                    frequency_hz=float(frequency_hz),
                    wavelength_m=float(wavelength_m),
                    aperture_wavelength=float(ACTIVE_APERTURE_M / wavelength_m),
                    adjacent_phase_deg=adjacent_phase_deg,
                    aperture_phase_deg=aperture_phase_deg,
                    cbf_interferer_db=cbf_interferer_db,
                    mvdr_interferer_db=mvdr_interferer_db,
                    interferer_reduction_db=float(cbf_interferer_db - mvdr_interferer_db),
                    cbf_rms_err=float(raw_row["cbf_rms_err"]),
                    mvdr_rms_err=float(raw_row["mvdr_rms_err"]),
                    mvdr_improves_target_err=bool(raw_row["mvdr_improves_target_err"]),
                    active_alias_limit_hz=float(raw_row["active_alias_limit_hz"]),
                )
            )
    return rows


def _rows_for_source(rows: list[FrequencyRow], covariance_source: str) -> list[FrequencyRow]:
    """指定した共分散条件の行だけを周波数昇順で返す。"""

    return sorted(
        [row for row in rows if row.covariance_source == covariance_source],
        key=lambda row: row.frequency_hz,
    )


def _row_array(rows: list[FrequencyRow], field_name: str) -> FloatArray:
    """FrequencyRow の float field を NumPy 配列へ変換する。"""

    return np.asarray([float(getattr(row, field_name)) for row in rows], dtype=np.float64)


def _plot_interferer_response(rows: list[FrequencyRow], output_path: Path) -> None:
    """干渉方向応答の周波数依存を保存する。"""

    mixture_rows = _rows_for_source(rows, "mixture")
    oracle_rows = _rows_for_source(rows, "interferer-only")
    frequency_hz = _row_array(mixture_rows, "frequency_hz")

    fig, axis = _plt().subplots(figsize=(10.8, 5.3))
    axis.semilogx(frequency_hz, _row_array(mixture_rows, "cbf_interferer_db"), marker="o", color="black", label="fixed_baseline")
    axis.semilogx(frequency_hz, _row_array(mixture_rows, "mvdr_interferer_db"), marker="o", color="tab:orange", label="MVDR from mixture covariance")
    axis.semilogx(frequency_hz, _row_array(oracle_rows, "mvdr_interferer_db"), marker="o", color="tab:blue", label="MVDR from interferer-only covariance")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Interferer response [{LEVEL_UNIT_LABEL}]")
    axis.set_title("128-sample covariance: low-frequency MVDR interferer response")
    axis.text(
        0.02,
        0.05,
        "mixture covariance uses only 128 samples containing target + interferer.\n"
        "interferer-only is an oracle reference, not the operational condition.",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _plot_physical_scale(rows: list[FrequencyRow], output_path: Path) -> None:
    """波長と開口位相差を可視化する。"""

    mixture_rows = _rows_for_source(rows, "mixture")
    frequency_hz = _row_array(mixture_rows, "frequency_hz")

    fig, axis_left = _plt().subplots(figsize=(10.8, 5.3))
    axis_right = axis_left.twinx()
    line_left = axis_left.semilogx(frequency_hz, _row_array(mixture_rows, "aperture_wavelength"), marker="o", color="tab:green", label="aperture / wavelength")
    line_right = axis_right.semilogx(frequency_hz, _row_array(mixture_rows, "aperture_phase_deg"), marker="o", color="tab:red", label="phase span across aperture")
    axis_left.axhline(1.0, color="0.4", linestyle=":", linewidth=1.0)
    axis_left.set_xlabel("Frequency [Hz]")
    axis_left.set_ylabel("Active aperture / wavelength [ratio]")
    axis_right.set_ylabel("Target steering phase span [deg]")
    axis_left.set_title("Low-frequency spatial aperture with 32ch, 0.05 m spacing")
    lines = line_left + line_right
    legend_labels: list[str] = [str(line.get_label()) for line in lines]
    axis_left.legend(lines, legend_labels, loc="best")
    axis_left.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _plot_error_response(rows: list[FrequencyRow], output_path: Path) -> None:
    """target 波形誤差の周波数依存を保存する。"""

    mixture_rows = _rows_for_source(rows, "mixture")
    oracle_rows = _rows_for_source(rows, "interferer-only")
    frequency_hz = _row_array(mixture_rows, "frequency_hz")

    fig, axis = _plt().subplots(figsize=(10.8, 5.3))
    axis.loglog(frequency_hz, _row_array(mixture_rows, "cbf_rms_err"), marker="o", color="black", label="fixed_baseline")
    axis.loglog(frequency_hz, _row_array(mixture_rows, "mvdr_rms_err"), marker="o", color="tab:orange", label="MVDR from mixture covariance")
    axis.loglog(frequency_hz, _row_array(oracle_rows, "mvdr_rms_err"), marker="o", color="tab:blue", label="MVDR from interferer-only covariance")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("RMS error to target-only reference [linear]")
    axis.set_title("128-sample covariance: target waveform error")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _finite_ylim(values: list[FloatArray], *, dynamic_range_db: float) -> tuple[float, float]:
    """有限値から見やすい dB 表示範囲を返す。"""

    finite_parts = [np.asarray(value, dtype=np.float64)[np.isfinite(value)] for value in values]
    finite = np.concatenate([part for part in finite_parts if part.size > 0])
    if finite.size == 0:
        return -120.0, 5.0
    top = float(np.max(finite)) + 3.0
    bottom = max(float(np.min(finite)) - 3.0, top - float(dynamic_range_db))
    return bottom, top


def _render_broadband_scene(band_low_hz: float, band_high_hz: float, noise_seed: int) -> FloatArray:
    """20 deg 方向の帯域制限広帯域信号を 128 sample だけ描画する。

    Args:
        band_low_hz: 通過帯域下限。単位は Hz。
        band_high_hz: 通過帯域上限。単位は Hz。
        noise_seed: scene_renderer の deterministic noise seed。

    Returns:
        多 CH 入力信号。shape は `[n_ch, n_sample]`、単位は normalized amplitude。

    境界条件:
        共分散評価対象を 128 sample に固定するため、FFT spectrum も 128 点で表示する。
        周波数分解能は 256 Hz であり、低周波帯・高周波帯ともこの bin grid 上で
        power を合計して beam response を作る。
    """

    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=N_CH, spacing=SPACING_M, axis=0, centered=True),
    )
    component = SourceComponent(
        spectrum=BandLimitedNoiseSpectrum(float(band_low_hz), float(band_high_hz)),
        envelope=ConstantEnvelope(),
        amplitude=None,
        level_db=SIGNAL_LEVEL_DB20,
        noise_seed=int(noise_seed),
        noise_filter_length=BROADBAND_NOISE_FILTER_LENGTH,
    )
    source = AcousticSource.from_relative_bearing(
        bearing_deg=TARGET_AZIMUTH_DEG,
        distance=1000.0,
        receiver_pose=receiver.trajectory.pose(0.0),
        components=[component],
        elevation_deg=0.0,
    )
    scene = Scene(sources=[source], ambient_fields=[], environment=FreeField(c=SOUND_SPEED_M_S))
    axis_t = np.arange(N_SAMPLE, dtype=np.float64) / FS_HZ
    rendered = SceneRenderer().render(scene, receiver, axis_t)
    return np.asarray(np.real(rendered), dtype=np.float64)


def _make_scan_steering(receiver: Receiver, environment: FreeField) -> tuple[FloatArray, ComplexArray]:
    """beam response 用の scan steering を作る。

    Args:
        receiver: scene_renderer の受信機。アレイ位置を保持する。
        environment: 音速を保持する free-field 環境。

    Returns:
        `(azimuth_deg, steering)`。
        `azimuth_deg` の shape は `[n_beam]`、単位は deg。
        `steering` の shape は `[n_ch, n_beam, n_fft]`。
    """

    dir3d, axis_az_deg, _ = make_directions(
        az_min_deg=-90.0,
        az_max_deg=90.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=N_BEAM,
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="forward",
        el_preset_deg=[0.0],
    )
    # ULA ではセンサが x 軸上に並ぶため、水平面の +azimuth と -azimuth は
    # 同じ x 方向余弦を持つ mirror として現れる。ここでは source bearing 表示の
    # 方位軸をそのまま使い、mirror ambiguity は review_index で明示する。
    steering = steering_from_dir3d(receiver, environment, FFT_SIZE, FS_HZ, dir3d)
    return np.asarray(axis_az_deg, dtype=np.float64), np.asarray(steering, dtype=np.complex128)


def _one_block_covariance(channel_signals: FloatArray) -> ComplexArray:
    """128 sample の 1 block から周波数 bin 別共分散を推定する。

    Args:
        channel_signals: 入力信号。shape は `[n_ch, 128]`。

    Returns:
        共分散 `R[k] = X[k] X[k]^H`。shape は `[n_fft, n_ch, n_ch]`。

    Notes:
        ここでは snapshot が 1 個だけなので、共分散は rank 1 になる。
        MVDR では対角ロードを入れるが、低周波では steering 間の差も小さいため、
        128 sample 統計で安定な空間分離ができるかを直接見る条件になる。
    """

    spectrum_ch_bin = np.fft.fft(np.asarray(channel_signals, dtype=np.float64), n=FFT_SIZE, axis=1)
    # spectrum_bin_ch shape: [n_fft, n_ch]。axis=0 が周波数 bin、axis=1 が CH。
    spectrum_bin_ch = np.asarray(np.moveaxis(spectrum_ch_bin, 1, 0), dtype=np.complex128)
    return np.asarray(np.einsum("kc,kd->kcd", spectrum_bin_ch, spectrum_bin_ch.conj(), optimize=True), dtype=np.complex128)


def _loaded_condition_number(covariance: ComplexArray) -> FloatArray:
    """MVDR と同じ対角ローディング比で loaded covariance 条件数を返す。"""

    n_ch = int(covariance.shape[1])
    average_power = np.real(np.trace(covariance, axis1=1, axis2=2)) / float(n_ch)
    # 対角ロードは設計器と同じ `gamma * trace(R) / n_ch`。
    # trace が 0 の bin では 1.0 を下限にして完全特異な solve を避ける。
    loading_power = DIAGONAL_LOADING_RATIO * np.where(average_power > 0.0, average_power, 1.0)
    loaded = covariance + loading_power[:, np.newaxis, np.newaxis] * np.eye(n_ch, dtype=np.complex128)[np.newaxis]
    return np.asarray(np.linalg.cond(loaded), dtype=np.float64)


def _one_sided_power_from_spectrum(spectrum: ComplexArray, n_sample: int) -> FloatArray:
    """rFFT spectrum を one-sided per-bin RMS power に変換する。"""

    factor = np.ones(spectrum.shape[-1], dtype=np.float64)
    if int(n_sample) % 2 == 0:
        factor[1:-1] = 2.0
    else:
        factor[1:] = 2.0
    power = factor[np.newaxis, :] * np.abs(spectrum / float(n_sample)) ** 2
    return np.asarray(power, dtype=np.float64)


def _spectrum_level_db_from_complex(spectrum: ComplexArray, n_sample: int) -> FloatArray:
    """rFFT spectrum を dB re input RMS の per-bin level へ変換する。"""

    power = _one_sided_power_from_spectrum(spectrum, int(n_sample))
    return np.asarray(10.0 * np.log10(np.maximum(power, np.finfo(np.float64).tiny)), dtype=np.float64)


def _beam_output_spectrum(channel_signals: FloatArray, weights: ComplexArray) -> ComplexArray:
    """rFFT 入力へ beamforming 重みを掛けた spectrum を返す。

    Args:
        channel_signals: 入力信号。shape は `[n_ch, n_sample]`。
        weights: 周波数別 beamforming 重み。shape は `[n_ch, n_beam, n_fft]`。

    Returns:
        beam 出力 spectrum。shape は `[n_beam, n_rfft_bin]`。
    """

    input_spectrum = np.asarray(np.fft.rfft(channel_signals, n=N_SAMPLE, axis=1), dtype=np.complex128)
    n_rfft_bin = int(input_spectrum.shape[1])
    # weights[:, :, :n_rfft_bin] shape: [n_ch, n_beam, n_rfft_bin]。
    # einsum の c 軸が CH 内積で、b が beam、k が rFFT bin を表す。
    output = np.einsum("cbk,ck->bk", weights[:, :, :n_rfft_bin].conj(), input_spectrum, optimize=True)
    return np.asarray(output, dtype=np.complex128)


def _band_integrated_level(output_spectrum: ComplexArray, frequency_hz: FloatArray, band_low_hz: float, band_high_hz: float) -> FloatArray:
    """帯域内 bin power を線形加算して beam response level を返す。"""

    power = _one_sided_power_from_spectrum(output_spectrum, N_SAMPLE)
    band_mask = (float(band_low_hz) <= frequency_hz) & (frequency_hz <= float(band_high_hz))
    if not bool(np.any(band_mask)):
        raise ValueError("band does not contain any rFFT bin.")
    # 広帯域 beam response は dB の平均ではなく、各 bin の power を線形加算してから dB 化する。
    band_power = np.sum(power[:, band_mask], axis=1)
    return np.asarray(10.0 * np.log10(np.maximum(band_power, np.finfo(np.float64).tiny)), dtype=np.float64)


def _evaluate_broadband_case(scenario_id: str, band_low_hz: float, band_high_hz: float, noise_seed: int) -> BroadbandCaseResult:
    """1 つの広帯域 case で 128 sample 共分散 beam response を評価する。"""

    input_signal = _render_broadband_scene(float(band_low_hz), float(band_high_hz), int(noise_seed))
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=N_CH, spacing=SPACING_M, axis=0, centered=True),
    )
    environment = FreeField(c=SOUND_SPEED_M_S)
    azimuth_deg, steering = _make_scan_steering(receiver, environment)
    array_design = build_array_design(
        n_ch=N_CH,
        spacing_m=SPACING_M,
        fft_size=FFT_SIZE,
        fs=FS_HZ,
        sound_speed=SOUND_SPEED_M_S,
        selector_mode="full",
        aperture_wavelengths=4.0,
        min_active_ch=4,
        dense_spacing_m=None,
        n_dense_ch=None,
    )
    covariance = _one_block_covariance(input_signal)
    fixed_weights = np.asarray(design_cbf_weights_with_channel_window(steering, array_design.shading_table), dtype=np.complex128)
    mvdr_weights = np.asarray(
        design_mvdr_weights_with_channel_window(
            covariance,
            steering,
            array_design.shading_table,
            diag_load=DIAGONAL_LOADING_RATIO,
        ),
        dtype=np.complex128,
    )
    frequency_hz = np.asarray(np.fft.rfftfreq(N_SAMPLE, d=1.0 / FS_HZ), dtype=np.float64)
    input_level_by_channel = _spectrum_level_db_from_complex(
        np.asarray(np.fft.rfft(input_signal, n=N_SAMPLE, axis=1), dtype=np.complex128),
        N_SAMPLE,
    )
    fixed_output_spectrum = _beam_output_spectrum(input_signal, fixed_weights)
    mvdr_output_spectrum = _beam_output_spectrum(input_signal, mvdr_weights)
    target_beam_index = int(np.argmin(np.abs(azimuth_deg - TARGET_AZIMUTH_DEG)))
    return BroadbandCaseResult(
        scenario_id=scenario_id,
        band_low_hz=float(band_low_hz),
        band_high_hz=float(band_high_hz),
        frequency_hz=frequency_hz,
        azimuth_deg=azimuth_deg,
        target_beam_index=target_beam_index,
        input_mean_spectrum_level_db=np.asarray(np.mean(input_level_by_channel, axis=0), dtype=np.float64),
        fixed_target_spectrum_level_db=_spectrum_level_db_from_complex(fixed_output_spectrum[[target_beam_index]], N_SAMPLE)[0],
        mvdr_target_spectrum_level_db=_spectrum_level_db_from_complex(mvdr_output_spectrum[[target_beam_index]], N_SAMPLE)[0],
        fixed_band_response_db=_band_integrated_level(fixed_output_spectrum, frequency_hz, float(band_low_hz), float(band_high_hz)),
        mvdr_band_response_db=_band_integrated_level(mvdr_output_spectrum, frequency_hz, float(band_low_hz), float(band_high_hz)),
        loaded_condition_number_by_bin=_loaded_condition_number(covariance),
    )


def _evaluate_broadband_cases() -> list[BroadbandCaseResult]:
    """低周波・高周波の広帯域 case を評価する。"""

    return [
        _evaluate_broadband_case(scenario_id, band_low_hz, band_high_hz, noise_seed)
        for scenario_id, band_low_hz, band_high_hz, noise_seed in BROADBAND_CASES
    ]


def _plot_broadband_input_spectrum(result: BroadbandCaseResult, output_path: Path) -> None:
    """広帯域入力 FFT spectrum を保存する。"""

    fig, axis = _plt().subplots(figsize=(10.8, 4.8))
    axis.plot(result.frequency_hz, result.input_mean_spectrum_level_db, color="black", linewidth=1.0, label="channel mean")
    axis.axvspan(result.band_low_hz, result.band_high_hz, color="tab:green", alpha=0.15, label="source passband")
    axis.set_xlim(0.0, FS_HZ / 2.0)
    axis.set_ylim(*_finite_ylim([result.input_mean_spectrum_level_db], dynamic_range_db=90.0))
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Per-bin RMS Level [{LEVEL_UNIT_LABEL}]")
    axis.set_title(f"Input FFT spectrum: {result.scenario_id}, 128 samples")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _plot_broadband_beam_response(result: BroadbandCaseResult, output_path: Path) -> None:
    """帯域加算 beam response を絶対値と target 正規化で保存する。

    Args:
        result: 広帯域 case 評価結果。beam response の shape は `[n_beam]`。
        output_path: 保存先 PNG。

    Notes:
        上段は実際に 128 sample 入力へ重みを掛けた絶対レベルである。
        下段は fixed / MVDR それぞれの target beam level を 0 dB に揃えた相対形状で、
        MVDR の制約形状と固定整相形状を同じ基準で比較するために使う。
    """

    target_index = int(result.target_beam_index)
    # 各 method の target beam level を引くことで、正規化図では target が 0 dB になる。
    # 絶対レベル差は上段に残し、下段では beam shape の比較だけを行う。
    fixed_relative_db = result.fixed_band_response_db - float(result.fixed_band_response_db[target_index])
    mvdr_relative_db = result.mvdr_band_response_db - float(result.mvdr_band_response_db[target_index])

    fig, axes = _plt().subplots(2, 1, figsize=(10.8, 8.0), sharex=True)
    axes[0].plot(result.azimuth_deg, result.fixed_band_response_db, color="black", label="fixed_baseline")
    axes[0].plot(result.azimuth_deg, result.mvdr_band_response_db, color="tab:orange", label="MVDR from 128-sample covariance")
    axes[0].axvline(TARGET_AZIMUTH_DEG, color="tab:green", linestyle="--", linewidth=1.1, label="source 20 deg")
    axes[0].axhline(0.0, color="0.35", linestyle=":", linewidth=1.0, label="0 dB input RMS")
    axes[0].set_ylim(*_finite_ylim([result.fixed_band_response_db, result.mvdr_band_response_db], dynamic_range_db=75.0))
    axes[0].set_ylabel(f"Absolute Level [{LEVEL_UNIT_LABEL}]")
    axes[0].set_title(f"128-sample covariance beam response: {result.band_low_hz:.0f}-{result.band_high_hz:.0f} Hz")
    axes[0].text(
        0.01,
        0.04,
        "Absolute output level after applying weights to the same 128-sample broadband input.",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(result.azimuth_deg, fixed_relative_db, color="black", label="fixed_baseline")
    axes[1].plot(result.azimuth_deg, mvdr_relative_db, color="tab:orange", label="MVDR from 128-sample covariance")
    axes[1].axvline(TARGET_AZIMUTH_DEG, color="tab:green", linestyle="--", linewidth=1.1, label="source 20 deg")
    axes[1].axhline(0.0, color="0.35", linestyle=":", linewidth=1.0, label="target beam normalization")
    axes[1].set_ylim(*_finite_ylim([fixed_relative_db, mvdr_relative_db], dynamic_range_db=65.0))
    axes[1].set_xlabel("Beam azimuth [deg]")
    axes[1].set_ylabel("Relative Level [dB re each method target beam]")
    axes[1].text(
        0.01,
        0.04,
        "Normalized response: fixed and MVDR target-beam levels are both set to 0 dB.\n"
        "Band response = 10log10(sum power of 128-point FFT bins in passband).",
        transform=axes[1].transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _plot_broadband_output_spectrum(result: BroadbandCaseResult, output_path: Path) -> None:
    """target beam の出力 FFT spectrum を保存する。"""

    fig, axis = _plt().subplots(figsize=(10.8, 4.8))
    axis.plot(result.frequency_hz, result.fixed_target_spectrum_level_db, color="black", linewidth=1.0, label="fixed_baseline")
    axis.plot(result.frequency_hz, result.mvdr_target_spectrum_level_db, color="tab:orange", linewidth=1.0, label="MVDR from 128-sample covariance")
    axis.axvspan(result.band_low_hz, result.band_high_hz, color="tab:green", alpha=0.15, label="source passband")
    axis.set_xlim(0.0, FS_HZ / 2.0)
    axis.set_ylim(*_finite_ylim([result.fixed_target_spectrum_level_db, result.mvdr_target_spectrum_level_db], dynamic_range_db=90.0))
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Per-bin RMS Level [{LEVEL_UNIT_LABEL}]")
    axis.set_title(f"Target-beam output FFT spectrum: {result.scenario_id}, 128 samples")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _write_broadband_npz(results: list[BroadbandCaseResult], output_path: Path) -> None:
    """広帯域 case の PNG 作成元配列を NPZ に保存する。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        scenario_id=np.asarray([result.scenario_id for result in results]),
        frequency_hz=results[0].frequency_hz,
        azimuth_deg=results[0].azimuth_deg,
        band_low_hz=np.asarray([result.band_low_hz for result in results], dtype=np.float64),
        band_high_hz=np.asarray([result.band_high_hz for result in results], dtype=np.float64),
        input_mean_spectrum_level_db=np.stack([result.input_mean_spectrum_level_db for result in results], axis=0),
        fixed_target_spectrum_level_db=np.stack([result.fixed_target_spectrum_level_db for result in results], axis=0),
        mvdr_target_spectrum_level_db=np.stack([result.mvdr_target_spectrum_level_db for result in results], axis=0),
        fixed_band_response_db=np.stack([result.fixed_band_response_db for result in results], axis=0),
        mvdr_band_response_db=np.stack([result.mvdr_band_response_db for result in results], axis=0),
        loaded_condition_number_by_bin=np.stack([result.loaded_condition_number_by_bin for result in results], axis=0),
    )


def _write_broadband_summary(results: list[BroadbandCaseResult], output_path: Path) -> None:
    """広帯域 case の target beam level と peak 方位を CSV に保存する。"""

    rows: list[dict[str, object]] = []
    for result in results:
        for method, response in (
            ("fixed_baseline", result.fixed_band_response_db),
            ("mvdr_128sample_covariance", result.mvdr_band_response_db),
        ):
            peak_index = int(np.argmax(response))
            target_index = int(result.target_beam_index)
            rows.append(
                {
                    "scenario_id": result.scenario_id,
                    "method": method,
                    "band_low_hz": float(result.band_low_hz),
                    "band_high_hz": float(result.band_high_hz),
                    "target_beam_azimuth_deg": float(result.azimuth_deg[target_index]),
                    "target_beam_band_level_db": float(response[target_index]),
                    "peak_azimuth_deg": float(result.azimuth_deg[peak_index]),
                    "peak_band_level_db": float(response[peak_index]),
                    "loaded_condition_number_max": float(np.max(result.loaded_condition_number_by_bin)),
                }
            )
    if not rows:
        raise ValueError("broadband summary rows must not be empty.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(rows: list[FrequencyRow], output_path: Path) -> None:
    """scenario_summary.csv を保存する。"""

    fieldnames = list(FrequencyRow.__dataclass_fields__.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: getattr(row, field_name) for field_name in fieldnames})


def _write_npz(rows: list[FrequencyRow], output_path: Path) -> None:
    """PNG 作成元配列を NPZ に保存する。"""

    mixture_rows = _rows_for_source(rows, "mixture")
    oracle_rows = _rows_for_source(rows, "interferer-only")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        frequency_hz=_row_array(mixture_rows, "frequency_hz"),
        mixture_cbf_interferer_db=_row_array(mixture_rows, "cbf_interferer_db"),
        mixture_mvdr_interferer_db=_row_array(mixture_rows, "mvdr_interferer_db"),
        oracle_mvdr_interferer_db=_row_array(oracle_rows, "mvdr_interferer_db"),
        mixture_cbf_rms_err=_row_array(mixture_rows, "cbf_rms_err"),
        mixture_mvdr_rms_err=_row_array(mixture_rows, "mvdr_rms_err"),
        oracle_mvdr_rms_err=_row_array(oracle_rows, "mvdr_rms_err"),
        aperture_wavelength=_row_array(mixture_rows, "aperture_wavelength"),
        adjacent_phase_deg=_row_array(mixture_rows, "adjacent_phase_deg"),
        aperture_phase_deg=_row_array(mixture_rows, "aperture_phase_deg"),
    )


def _write_metadata(output_dir: Path) -> None:
    """評価条件と配列 shape を metadata.json に保存する。"""

    metadata = {
        "scenario_id": "low_frequency_128sample_mvdr",
        "evaluation_pattern": "fixed_beam_multi_source",
        "fs_hz": FS_HZ,
        "sound_speed_m_s": SOUND_SPEED_M_S,
        "n_ch": N_CH,
        "spacing_m": SPACING_M,
        "active_aperture_m": ACTIVE_APERTURE_M,
        "target_azimuth_deg": TARGET_AZIMUTH_DEG,
        "interferer_azimuth_deg": INTERFERER_AZIMUTH_DEG,
        "fft_size": FFT_SIZE,
        "n_sample": N_SAMPLE,
        "integration_time_s": INTEGRATION_TIME_S,
        "diagonal_loading_ratio": DIAGONAL_LOADING_RATIO,
        "frequencies_hz": list(FREQUENCIES_HZ),
        "broadband_cases": [
            {"scenario_id": scenario_id, "band_low_hz": band_low_hz, "band_high_hz": band_high_hz}
            for scenario_id, band_low_hz, band_high_hz, _ in BROADBAND_CASES
        ],
        "level_reference": LEVEL_UNIT_LABEL,
        "array_shapes": {
            "frequency_hz": "[n_freq]",
            "*_interferer_db": "[n_freq]",
            "*_rms_err": "[n_freq]",
            "aperture_wavelength": "[n_freq]",
            "broadband_input_mean_spectrum_level_db": "[n_case, n_rfft_bin]",
            "broadband_*_band_response_db": "[n_case, n_beam]",
            "broadband_*_target_spectrum_level_db": "[n_case, n_rfft_bin]",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_review_index(rows: list[FrequencyRow], output_dir: Path) -> None:
    """AI 向け review index を保存する。"""

    mixture_rows = _rows_for_source(rows, "mixture")
    low_row = mixture_rows[0]
    lines = [
        "# 低周波・128 sample 共分散 MVDR 評価",
        "",
        "## Scenario",
        "",
        "- target: 20 deg, 0 dB re input RMS",
        "- interferer: -30 deg, 0 dB re input RMS, target と同一周波数",
        f"- covariance block: `{N_SAMPLE}` sample = `{INTEGRATION_TIME_S:.9f}` s",
        f"- array: {N_CH} ch ULA, spacing {SPACING_M:.3f} m, aperture {ACTIVE_APERTURE_M:.3f} m",
        "- covariance_source `mixture`: target + interferer を含む実運用寄り条件。",
        "- covariance_source `interferer-only`: 理想参照。実運用の採否判断には直接使わない。",
        "",
        "## Artifacts",
        "",
        "- `figures/interferer_response_vs_frequency.png`: 干渉方向応答。mixture 共分散で抑圧が出ないことを見る主図。",
        "- `figures/physical_scale_vs_frequency.png`: 波長に対する開口長と target steering 位相幅。",
        "- `figures/target_error_vs_frequency.png`: target-only 参照に対する出力 RMS error。",
        "- `figures/input_frequency_spectrum_low_256_1024hz.png`: 低周波広帯域入力の 128-point FFT spectrum。",
        "- `figures/beam_response_band_integrated_low_256_1024hz.png`: 低周波広帯域の帯域加算 beam response。上段は絶対レベル、下段は各 method の target beam を 0 dB に揃えた正規化表示。",
        "- `figures/output_frequency_spectrum_low_256_1024hz.png`: 低周波広帯域の target beam 出力 spectrum。",
        "- `figures/input_frequency_spectrum_high_8500_9500hz.png`: 高周波広帯域入力の 128-point FFT spectrum。",
        "- `figures/beam_response_band_integrated_high_8500_9500hz.png`: 高周波広帯域の帯域加算 beam response。上段は絶対レベル、下段は各 method の target beam を 0 dB に揃えた正規化表示。",
        "- `figures/output_frequency_spectrum_high_8500_9500hz.png`: 高周波広帯域の target beam 出力 spectrum。",
        "- `data/low_frequency_128sample_mvdr_arrays.npz`: tone sweep 図作成元配列。",
        "- `data/broadband_128sample_mvdr_arrays.npz`: 広帯域図作成元配列。",
        "- `broadband_scenario_summary.csv`: 広帯域 case の peak 方位と target beam level。",
        "- `scenario_summary.csv`: 周波数・共分散条件別 metric。",
        "- `metadata.json`: 評価条件、単位、shape。",
        "",
        "## Interpretation Notes",
        "",
        f"- {low_row.frequency_hz:.0f} Hz では波長 {low_row.wavelength_m:.3f} m に対して開口は {low_row.aperture_wavelength:.3f} λ、隣接 CH 位相差は {low_row.adjacent_phase_deg:.3f} deg。",
        "- `mixture` 共分散では target も統計に含まれるため、128 sample だけでは干渉方向だけを安定に学習できない。",
        "- `interferer-only` が大きく抑圧できる場合でも、それは理想参照がある条件であり、運用時に同じ性能を保証しない。",
        "- 広帯域 beam response は 128-point FFT の帯域内 bin power を線形加算してから dB 化している。",
        "- 絶対レベル図では、128 sample の標本共分散に target 自身が含まれる場合の自己キャンセルも見える。正規化図は形状比較用であり、絶対出力レベルの採否判断には使わない。",
        "- 低周波広帯域は 256-1024 Hz、高周波広帯域は 8500-9500 Hz を別々の図で確認する。",
        "- 本評価の ULA は水平面で +azimuth / -azimuth の mirror ambiguity を持つため、peak 方位は target marker と mirror 側の両方を確認する。",
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

    rows = _evaluate_rows()
    broadband_results = _evaluate_broadband_cases()
    _plot_interferer_response(rows, FIGURE_DIR / "interferer_response_vs_frequency.png")
    _plot_physical_scale(rows, FIGURE_DIR / "physical_scale_vs_frequency.png")
    _plot_error_response(rows, FIGURE_DIR / "target_error_vs_frequency.png")
    for broadband_result in broadband_results:
        _plot_broadband_input_spectrum(
            broadband_result,
            FIGURE_DIR / f"input_frequency_spectrum_{broadband_result.scenario_id}.png",
        )
        _plot_broadband_beam_response(
            broadband_result,
            FIGURE_DIR / f"beam_response_band_integrated_{broadband_result.scenario_id}.png",
        )
        _plot_broadband_output_spectrum(
            broadband_result,
            FIGURE_DIR / f"output_frequency_spectrum_{broadband_result.scenario_id}.png",
        )
    _write_npz(rows, DATA_DIR / "low_frequency_128sample_mvdr_arrays.npz")
    _write_broadband_npz(broadband_results, DATA_DIR / "broadband_128sample_mvdr_arrays.npz")
    _write_csv(rows, OUTPUT_DIR / "scenario_summary.csv")
    _write_broadband_summary(broadband_results, OUTPUT_DIR / "broadband_scenario_summary.csv")
    _write_metadata(OUTPUT_DIR)
    _write_review_index(rows, OUTPUT_DIR)
    return _zip_package(OUTPUT_DIR)


def main() -> None:
    """CLI entrypoint。"""

    zip_path = build_report_package()
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "zip_path": str(zip_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

