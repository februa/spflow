"""EBAE/MVDRのS0・S1・T1・T2とFIR長依存を同一条件で評価する。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from spflow.beamforming.ebae import EbaeConfig, design_ebae_weights_band


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


OUTPUT_DIR = Path("artifacts/beamforming/ebae_mvdr_s0_s1_t1_t2_fir_sweep/review_pack")
SCENARIO_ID = "low_band_long_ula_beam_center"
FS_HZ = 8192.0
FFT_SIZE = 512
ANALYSIS_WIDTH_HZ = FS_HZ / FFT_SIZE
SOUND_SPEED_M_S = 1500.0
N_CHANNEL = 8
SPACING_M = 6.0
TARGET_AZIMUTH_DEG = 60.0
TARGET_BAND_HZ = (64.0, 128.0)
SOURCE_BAND_RMS_POWER = 1.0
NOISE_POWER_PER_BIN_RE_INPUT_RMS2 = 1.0e-2
EBAE_DIAGONAL_LOADING = 1.0
MVDR_DIAGONAL_LOADING_RATIO = 1.0e-3
AZIMUTH_DEG = np.arange(0.0, 181.0, 10.0, dtype=np.float64)
FIR_TAP_COUNTS = (16, 32, 64, 128, 256, 512)
ALGORITHM_IDS = ("ebae", "mvdr")
METHOD_IDS = ("S0", "S1", "T1", "T2")
DISPLAY_FLOOR_DB_RE_INPUT_RMS = -100.0


@dataclass(frozen=True)
class WeightDesignResult:
    """EBAE/MVDRの4方式完成重みとEBAE診断量を保持する。

    Attributes:
        weights: ``algorithm -> method -> weight``。各weight shapeは
            ``[n_fft,n_beam,n_ch]``。S0/T1は元入力座標、S1/T2は整数遅延後座標。
        integer_phase: 整数遅延後入力への位相回転。shapeは``[n_fft,n_beam,n_ch]``。
        steering: 元入力座標steering。shapeは``[n_fft,n_beam,n_ch]``。
        source_steering: source steering。shapeは``[n_fft,n_ch]``。
        source_bin_mask: source帯域mask。shapeは``[n_fft]``。
        ebae_signal_counts: ``method -> [n_bin,n_beam]``。負周波数は正側から鏡映。
        ebae_associated_beams: ``method -> [n_bin,n_beam]``。対応beam index、信号なしは-1。
    """

    weights: dict[str, dict[str, ComplexArray]]
    integer_phase: ComplexArray
    steering: ComplexArray
    source_steering: ComplexArray
    source_bin_mask: NDArray[np.bool_]
    ebae_signal_counts: dict[str, NDArray[np.int64]]
    ebae_associated_beams: dict[str, NDArray[np.int64]]


@dataclass(frozen=True)
class FirApproximationResult:
    """共通tap窓で打ち切った多channel FIRの再構成結果を保持する。

    Attributes:
        reconstructed_weights: FIR化後重み。shapeは``[n_fft,n_beam,n_ch]``。
        energy_ratio: 採用tap区間内energy比。shapeは``[n_beam]``。
    """

    reconstructed_weights: ComplexArray
    energy_ratio: FloatArray


def _positions_m() -> FloatArray:
    """中心基準ULA位置を返す。

    Returns:
        sensor位置。shapeは``[n_ch]``、単位はm。
    """
    aperture_m = SPACING_M * (N_CHANNEL - 1)
    return np.linspace(-aperture_m / 2.0, aperture_m / 2.0, N_CHANNEL, dtype=np.float64)


def _arrival_delays_s(azimuth_deg: FloatArray) -> FloatArray:
    """方位ごとの到来遅延を返す。

    Args:
        azimuth_deg: 方位。shapeは``[n_direction]``、単位はdeg。

    Returns:
        到来遅延。shapeは``[n_direction,n_ch]``、単位はs。
    """
    direction_cosine = np.cos(np.deg2rad(azimuth_deg))
    # tau=-r cos(theta)/c。0/180 degがendfire、90 degがbroadsideである。
    return np.asarray(
        -direction_cosine[:, np.newaxis] * _positions_m()[np.newaxis, :] / SOUND_SPEED_M_S,
        dtype=np.float64,
    )


def _steering(delays_s: FloatArray, frequencies_hz: FloatArray) -> ComplexArray:
    """到来遅延から未正規化steeringを返す。

    Args:
        delays_s: 到来遅延。shapeは``[n_direction,n_ch]``、単位はs。
        frequencies_hz: DFT周波数。shapeは``[n_fft]``、単位はHz。

    Returns:
        steering。shapeは``[n_fft,n_direction,n_ch]``。
    """
    # a(f,theta,ch)=exp(-j2πf tau)。axis=0は周波数、axis=1は方位、axis=2はchannel。
    return np.asarray(
        np.exp(
            -1j
            * 2.0
            * np.pi
            * frequencies_hz[:, np.newaxis, np.newaxis]
            * delays_s[np.newaxis, :, :]
        ),
        dtype=np.complex128,
    )


def _source_covariance(
    source_delay_s: FloatArray,
    source_steering: ComplexArray,
    *,
    candidate_delay_s: FloatArray | None,
    source_power: float,
) -> ComplexArray:
    """S0または候補方位別T1共分散を返す。

    Args:
        source_delay_s: source到来遅延。shapeは``[n_ch]``、単位はs。
        source_steering: 当該binのsource steering。shapeは``[n_ch]``。
        candidate_delay_s: T1の候補方位遅延。shapeは``[n_ch]``、単位はs。
            ``None``なら同一時間blockのS0共分散を作る。
        source_power: 当該binのsource power。基準はinput band RMS二乗。

    Returns:
        空間共分散。shapeは``[n_ch,n_ch]``。
    """
    if candidate_delay_s is None:
        residual_delay_s = source_delay_s
    else:
        # T1は候補方位の整数sample時刻差を切り出しで除き、残留遅延だけをbin内積分へ残す。
        quantized_candidate_s = np.rint(candidate_delay_s * FS_HZ) / FS_HZ
        residual_delay_s = source_delay_s - quantized_candidate_s
    pair_delay_s = residual_delay_s[:, np.newaxis] - residual_delay_s[np.newaxis, :]
    # 平坦な1-bin帯域積分により、pair coherenceはsinc(Δf Δtau)となる。
    coherence = np.sinc(ANALYSIS_WIDTH_HZ * pair_delay_s)
    outer = source_steering[:, np.newaxis] * source_steering.conj()[np.newaxis, :]
    return np.asarray(
        source_power * coherence * outer
        + NOISE_POWER_PER_BIN_RE_INPUT_RMS2 * np.eye(N_CHANNEL, dtype=np.complex128),
        dtype=np.complex128,
    )


def _mvdr_weight(covariance: ComplexArray, steering: ComplexArray) -> ComplexArray:
    """trace比例loading付きMVDR重みを返す。

    Args:
        covariance: 空間共分散。shapeは``[n_ch,n_ch]``。
        steering: 制約steering。shapeは``[n_ch]``。

    Returns:
        distortionless MVDR重み。shapeは``[n_ch]``。
    """
    hermitian = np.asarray(0.5 * (covariance + covariance.conj().T), dtype=np.complex128)
    average_power = float(np.real(np.trace(hermitian))) / float(N_CHANNEL)
    # trace比例loadingはunitary位相回転で不変なので、S0=S1およびT1=T2を維持する。
    loaded = hermitian + MVDR_DIAGONAL_LOADING_RATIO * average_power * np.eye(
        N_CHANNEL, dtype=np.complex128
    )
    solved = np.linalg.solve(loaded, steering)
    denominator = np.vdot(steering, solved)
    return np.asarray(solved / denominator, dtype=np.complex128)


def _ebae_weight(
    covariance: ComplexArray,
    steering_scan: ComplexArray,
    beam_index: int,
) -> tuple[ComplexArray, int, int]:
    """単一bin・単一待受beamのEBAE重みと診断量を返す。

    Args:
        covariance: 空間共分散。shapeは``[n_ch,n_ch]``。
        steering_scan: 同じchannel座標の全beam steering。shapeは``[n_ch,n_beam]``。
        beam_index: 返す待受beam index。

    Returns:
        ``(weight, Ns, associated_beam)``。weight shapeは``[n_ch]``。
        信号数0ではassociated beamを-1とする。
    """
    result = design_ebae_weights_band(
        covariance,
        steering_scan,
        snapshot_count=N_CHANNEL * N_CHANNEL,
        config=EbaeConfig(
            snapshot_rate_hz=float(N_CHANNEL * N_CHANNEL),
            integration_time_sec=1.0,
            sigmoid_slope=10.0,
            sigmoid_midpoint=0.5,
            diagonal_loading=EBAE_DIAGONAL_LOADING,
        ),
    )
    associated_beam = -1
    if result.signal_count > 0:
        associated_beam = int(result.associated_beam_indices[0])
    return (
        np.asarray(result.weights[:, beam_index], dtype=np.complex128),
        result.signal_count,
        associated_beam,
    )


def _mirror_positive_frequency_weights(positive_weights: ComplexArray) -> ComplexArray:
    """DC～Nyquist重みから正負周波数整合したfull DFT重みを作る。

    Args:
        positive_weights: shapeは``[n_rfft,n_beam,n_ch]``。

    Returns:
        正負周波数を共役鏡映したfull DFT重み。shapeは``[n_fft,n_beam,n_ch]``。
        DC/Nyquistの実数制約は課さないため、IFFT係数は複素FIRになり得る。
    """
    full = np.empty((FFT_SIZE, positive_weights.shape[1], positive_weights.shape[2]), dtype=np.complex128)
    full[: FFT_SIZE // 2 + 1] = positive_weights
    # W[-k]=conj(W[k])をDCとNyquistを除く正周波数から作り、正負source binを一致させる。
    full[FFT_SIZE // 2 + 1 :] = positive_weights[1:-1][::-1].conj()
    return full


def design_reference_weights() -> WeightDesignResult:
    """EBAE/MVDRについて正しいS0・S1・T1・T2完成重みを設計する。

    Returns:
        4方式完成重み、位相変換、steering、EBAE診断量。
    """
    positive_frequency_hz = np.fft.rfftfreq(FFT_SIZE, d=1.0 / FS_HZ)
    full_frequency_hz = np.fft.fftfreq(FFT_SIZE, d=1.0 / FS_HZ)
    beam_delay_s = _arrival_delays_s(AZIMUTH_DEG)
    source_delay_s = _arrival_delays_s(np.asarray([TARGET_AZIMUTH_DEG], dtype=np.float64))[0]
    positive_steering = _steering(beam_delay_s, positive_frequency_hz)
    positive_source_steering = _steering(
        source_delay_s[np.newaxis, :], positive_frequency_hz
    )[:, 0, :]
    full_steering = _steering(beam_delay_s, full_frequency_hz)
    full_source_steering = _steering(source_delay_s[np.newaxis, :], full_frequency_hz)[:, 0, :]
    positive_source_mask = (positive_frequency_hz >= TARGET_BAND_HZ[0]) & (
        positive_frequency_hz <= TARGET_BAND_HZ[1]
    )
    full_source_mask = (np.abs(full_frequency_hz) >= TARGET_BAND_HZ[0]) & (
        np.abs(full_frequency_hz) <= TARGET_BAND_HZ[1]
    )
    source_bin_count = int(np.count_nonzero(positive_source_mask))
    source_power_per_positive_bin = SOURCE_BAND_RMS_POWER / float(source_bin_count)

    # delay_int[beam,ch]は待受方位の物理遅延を最寄りsampleへ丸める。
    delay_int = np.rint(beam_delay_s * FS_HZ).astype(np.int64)
    # steeringがexp(-j2πf tau)なので、整数delay分を取り除く前段位相はexp(+j2πf d_int/fs)。
    # 同符号にすると遅延を二重化し、S0=S1同値性だけでは検出できてもFIR短縮が成立しない。
    positive_integer_phase = np.exp(
        1j
        * 2.0
        * np.pi
        * positive_frequency_hz[:, np.newaxis, np.newaxis]
        * delay_int[np.newaxis, :, :]
        / FS_HZ
    )
    full_integer_phase = np.exp(
        1j
        * 2.0
        * np.pi
        * full_frequency_hz[:, np.newaxis, np.newaxis]
        * delay_int[np.newaxis, :, :]
        / FS_HZ
    )

    positive_weights: dict[str, dict[str, ComplexArray]] = {
        algorithm: {
            method: np.empty(
                (positive_frequency_hz.size, AZIMUTH_DEG.size, N_CHANNEL), dtype=np.complex128
            )
            for method in METHOD_IDS
        }
        for algorithm in ALGORITHM_IDS
    }
    signal_counts = {
        method: np.zeros((positive_frequency_hz.size, AZIMUTH_DEG.size), dtype=np.int64)
        for method in METHOD_IDS
    }
    associated_beams = {
        method: np.full((positive_frequency_hz.size, AZIMUTH_DEG.size), -1, dtype=np.int64)
        for method in METHOD_IDS
    }

    for frequency_index in range(positive_frequency_hz.size):
        source_power = source_power_per_positive_bin if positive_source_mask[frequency_index] else 0.0
        s0_covariance = _source_covariance(
            source_delay_s,
            positive_source_steering[frequency_index],
            candidate_delay_s=None,
            source_power=source_power,
        )
        for beam_index in range(AZIMUTH_DEG.size):
            steering_scan = np.asarray(positive_steering[frequency_index].T, dtype=np.complex128)
            constraint = np.asarray(positive_steering[frequency_index, beam_index], dtype=np.complex128)
            phase = np.asarray(positive_integer_phase[frequency_index, beam_index], dtype=np.complex128)
            rotated_scan = np.asarray(phase[:, np.newaxis] * steering_scan, dtype=np.complex128)
            rotated_constraint = np.asarray(phase * constraint, dtype=np.complex128)
            s1_covariance = np.asarray(
                phase[:, np.newaxis] * s0_covariance * phase.conj()[np.newaxis, :],
                dtype=np.complex128,
            )
            t1_covariance = _source_covariance(
                source_delay_s,
                positive_source_steering[frequency_index],
                candidate_delay_s=beam_delay_s[beam_index],
                source_power=source_power,
            )
            t2_covariance = np.asarray(
                phase[:, np.newaxis] * t1_covariance * phase.conj()[np.newaxis, :],
                dtype=np.complex128,
            )

            positive_weights["mvdr"]["S0"][frequency_index, beam_index] = _mvdr_weight(
                s0_covariance, constraint
            )
            positive_weights["mvdr"]["S1"][frequency_index, beam_index] = _mvdr_weight(
                s1_covariance, rotated_constraint
            )
            positive_weights["mvdr"]["T1"][frequency_index, beam_index] = _mvdr_weight(
                t1_covariance, constraint
            )
            positive_weights["mvdr"]["T2"][frequency_index, beam_index] = _mvdr_weight(
                t2_covariance, rotated_constraint
            )

            for method, covariance, scan in (
                ("S0", s0_covariance, steering_scan),
                ("S1", s1_covariance, rotated_scan),
                ("T1", t1_covariance, steering_scan),
                ("T2", t2_covariance, rotated_scan),
            ):
                weight, count, associated = _ebae_weight(covariance, scan, beam_index)
                positive_weights["ebae"][method][frequency_index, beam_index] = weight
                signal_counts[method][frequency_index, beam_index] = count
                associated_beams[method][frequency_index, beam_index] = associated

    weights = {
        algorithm: {
            method: _mirror_positive_frequency_weights(positive_weights[algorithm][method])
            for method in METHOD_IDS
        }
        for algorithm in ALGORITHM_IDS
    }
    full_signal_counts = {
        method: _mirror_integer_diagnostics(signal_counts[method]) for method in METHOD_IDS
    }
    full_associated = {
        method: _mirror_integer_diagnostics(associated_beams[method]) for method in METHOD_IDS
    }
    return WeightDesignResult(
        weights=weights,
        integer_phase=full_integer_phase,
        steering=full_steering,
        source_steering=full_source_steering,
        source_bin_mask=full_source_mask,
        ebae_signal_counts=full_signal_counts,
        ebae_associated_beams=full_associated,
    )


def _mirror_integer_diagnostics(positive_values: NDArray[np.int64]) -> NDArray[np.int64]:
    """正周波数診断値をfull DFT binへ鏡映する。

    Args:
        positive_values: shapeは``[n_rfft,n_beam]``。

    Returns:
        shape``[n_fft,n_beam]``の整数診断値。
    """
    full = np.empty((FFT_SIZE, positive_values.shape[1]), dtype=np.int64)
    full[: FFT_SIZE // 2 + 1] = positive_values
    full[FFT_SIZE // 2 + 1 :] = positive_values[1:-1][::-1]
    return full


def _best_common_circular_window(impulse: ComplexArray, tap_count: int) -> tuple[int, float]:
    """全channel合算energyが最大となる共通circular tap窓を求める。

    Args:
        impulse: impulse response。shapeは``[n_fft,n_ch]``。
        tap_count: 採用tap数。単位はsample。

    Returns:
        ``(start_index, energy_ratio)``。startはcircular窓先頭index。
    """
    if not 0 < tap_count <= FFT_SIZE:
        raise ValueError("tap_count must be in [1, FFT_SIZE].")
    energy_per_time = np.sum(np.abs(impulse) ** 2, axis=1)
    total_energy = float(np.sum(energy_per_time))
    if total_energy <= 0.0:
        return 0, 1.0
    extended = np.concatenate((energy_per_time, energy_per_time[: tap_count - 1]))
    window_energy = np.convolve(extended, np.ones(tap_count), mode="valid")[:FFT_SIZE]
    start_index = int(np.argmax(window_energy))
    return start_index, float(window_energy[start_index] / total_energy)


def approximate_weights_with_fir(weights: ComplexArray, tap_count: int) -> FirApproximationResult:
    """beamごとの共通tap窓で重みをFIR近似する。

    Args:
        weights: 数式上の重み。shapeは``[n_fft,n_beam,n_ch]``。
        tap_count: FIR tap数。単位はsample。

    Returns:
        FIR再構成重みとbeam別energy比。
    """
    if weights.shape != (FFT_SIZE, AZIMUTH_DEG.size, N_CHANNEL):
        raise ValueError("weights must have shape (n_fft, n_beam, n_ch).")
    reconstructed = np.empty_like(weights)
    energy_ratio = np.empty(AZIMUTH_DEG.size, dtype=np.float64)
    for beam_index in range(AZIMUTH_DEG.size):
        # 実適用周波数応答はconj(w)。IFFT後shapeは[n_fft,n_ch]で、同じtap窓を全channelへ使う。
        impulse = np.asarray(np.fft.ifft(weights[:, beam_index, :].conj(), axis=0), dtype=np.complex128)
        start_index, energy_ratio[beam_index] = _best_common_circular_window(impulse, tap_count)
        keep_indices = (start_index + np.arange(tap_count)) % FFT_SIZE
        truncated = np.zeros_like(impulse)
        truncated[keep_indices, :] = impulse[keep_indices, :]
        reconstructed[:, beam_index, :] = np.fft.fft(truncated, axis=0).conj()
    return FirApproximationResult(reconstructed, energy_ratio)


def _original_coordinate_weights(
    method: str,
    weights: ComplexArray,
    integer_phase: ComplexArray,
) -> ComplexArray:
    """S1/T2重みを元入力座標へ戻す。

    Args:
        method: S0、S1、T1、T2。
        weights: 当該方式座標の重み。shapeは``[n_fft,n_beam,n_ch]``。
        integer_phase: 整数遅延位相。shapeは同じ。

    Returns:
        元入力座標の重み。shapeは入力と同じ。
    """
    if method in ("S1", "T2"):
        # y=v^H D xなので、元入力座標の等価weightはD^H v=conj(D)*vである。
        return np.asarray(integer_phase.conj() * weights, dtype=np.complex128)
    return weights


def _level_db_from_power(power: FloatArray) -> FloatArray:
    """線形powerを表示床付きdB re input RMSへ変換する。"""
    floor_power = 10.0 ** (DISPLAY_FLOOR_DB_RE_INPUT_RMS / 10.0)
    return np.asarray(10.0 * np.log10(np.maximum(power, floor_power)), dtype=np.float64)


def _metrics_row(
    algorithm: str,
    method: str,
    tap_count: int,
    reference_original: ComplexArray,
    approximated_original: ComplexArray,
    design: WeightDesignResult,
    energy_ratio: FloatArray,
) -> dict[str, Any]:
    """1つのalgorithm・方式・tap数について比較指標を返す。"""
    target_beam_index = int(np.argmin(np.abs(AZIMUTH_DEG - TARGET_AZIMUTH_DEG)))
    band = design.source_bin_mask
    reference_band = reference_original[band]
    approximated_band = approximated_original[band]
    relative_weight_error = float(
        np.linalg.norm(approximated_band - reference_band)
        / max(float(np.linalg.norm(reference_band)), np.finfo(np.float64).tiny)
    )
    # response[f,beam]=w[f,beam]^H a_source[f]。einsumはchannel軸を内積として畳み込む。
    reference_response = np.einsum(
        "fbc,fc->fb", reference_band.conj(), design.source_steering[band], optimize=True
    )
    approximated_response = np.einsum(
        "fbc,fc->fb", approximated_band.conj(), design.source_steering[band], optimize=True
    )
    reference_power = np.mean(np.abs(reference_response) ** 2, axis=0)
    approximated_power = np.mean(np.abs(approximated_response) ** 2, axis=0)
    reference_bl = _level_db_from_power(np.asarray(reference_power, dtype=np.float64))
    approximated_bl = _level_db_from_power(np.asarray(approximated_power, dtype=np.float64))
    target_level_delta = float(
        approximated_bl[target_beam_index] - reference_bl[target_beam_index]
    )
    peak_index = int(np.argmax(approximated_bl))
    guard_mask = np.abs(AZIMUTH_DEG - TARGET_AZIMUTH_DEG) > 10.0
    return {
        "scenario": SCENARIO_ID,
        "algorithm": algorithm,
        "method": method,
        "tap_count": tap_count,
        "coordinate": "integer_delay_residual" if method in ("S1", "T2") else "original_input",
        "relative_weight_error": relative_weight_error,
        "minimum_energy_ratio": float(np.min(energy_ratio)),
        "target_beam_energy_ratio": float(energy_ratio[target_beam_index]),
        "target_level_delta_db_re_reference": target_level_delta,
        "target_peak_error_deg": float(abs(AZIMUTH_DEG[peak_index] - TARGET_AZIMUTH_DEG)),
        "guard_outside_peak_db_re_input_rms": float(np.max(approximated_bl[guard_mask])),
        "bl_rms_delta_db_re_reference": float(
            np.sqrt(np.mean((approximated_bl - reference_bl) ** 2))
        ),
    }


def _band_bl_db(weights_original: ComplexArray, design: WeightDesignResult) -> FloatArray:
    """source帯域を積分したtarget-only BLを返す。

    Args:
        weights_original: 元入力座標の重み。shapeは``[n_fft,n_beam,n_ch]``。
        design: source steeringと帯域maskを含む完成設計。

    Returns:
        待受beam BL。shapeは``[n_beam]``、dB re input RMS。
    """
    band_weights = weights_original[design.source_bin_mask]
    # response[f,beam]はsource RMSを各正負binへ等配分する前の線形空間応答である。
    response = np.einsum(
        "fbc,fc->fb",
        band_weights.conj(),
        design.source_steering[design.source_bin_mask],
        optimize=True,
    )
    # full DFTの正負source binへ同じpowerを割り当てるため、平均powerが帯域積分応答になる。
    power = np.mean(np.abs(response) ** 2, axis=0)
    return _level_db_from_power(np.asarray(power, dtype=np.float64))


def calculate_fir_sweep() -> tuple[WeightDesignResult, tuple[dict[str, Any], ...], dict[str, FloatArray]]:
    """4方式×2アルゴリズム×FIR長の評価指標を計算する。

    Returns:
        ``(design, rows, arrays)``。arraysは同値性誤差とtap sweep描画配列を含む。
    """
    design = design_reference_weights()
    rows: list[dict[str, Any]] = []
    arrays: dict[str, FloatArray] = {
        "fir_tap_counts": np.asarray(FIR_TAP_COUNTS, dtype=np.float64),
    }
    for algorithm in ALGORITHM_IDS:
        s0_original = design.weights[algorithm]["S0"]
        s1_original = _original_coordinate_weights(
            "S1", design.weights[algorithm]["S1"], design.integer_phase
        )
        t1_original = design.weights[algorithm]["T1"]
        t2_original = _original_coordinate_weights(
            "T2", design.weights[algorithm]["T2"], design.integer_phase
        )
        arrays[f"{algorithm}_s0_s1_relative_error"] = np.asarray(
            [np.linalg.norm(s0_original - s1_original) / np.linalg.norm(s0_original)],
            dtype=np.float64,
        )
        arrays[f"{algorithm}_t1_t2_relative_error"] = np.asarray(
            [np.linalg.norm(t1_original - t2_original) / np.linalg.norm(t1_original)],
            dtype=np.float64,
        )
        for method in METHOD_IDS:
            reference_method = design.weights[algorithm][method]
            reference_original = _original_coordinate_weights(
                method, reference_method, design.integer_phase
            )
            method_errors: list[float] = []
            for tap_count in FIR_TAP_COUNTS:
                approximation = approximate_weights_with_fir(reference_method, tap_count)
                approximated_original = _original_coordinate_weights(
                    method, approximation.reconstructed_weights, design.integer_phase
                )
                row = _metrics_row(
                    algorithm,
                    method,
                    tap_count,
                    reference_original,
                    approximated_original,
                    design,
                    approximation.energy_ratio,
                )
                rows.append(row)
                method_errors.append(float(row["relative_weight_error"]))
            arrays[f"{algorithm}_{method}_relative_weight_error"] = np.asarray(
                method_errors, dtype=np.float64
            )
    return design, tuple(rows), arrays


def write_fir_sweep_report(output_dir: Path = OUTPUT_DIR) -> tuple[dict[str, Any], ...]:
    """S0/S1/T1/T2 FIR長sweepのCSV、NPZ、図、レビュー索引を保存する。

    Args:
        output_dir: review pack出力先。

    Returns:
        保存したscenario summary行。
    """
    design, rows, arrays = calculate_fir_sweep()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures" / SCENARIO_ID
    data_dir = output_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scenario_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True, sharey=True)
    for axis, algorithm in zip(axes, ALGORITHM_IDS, strict=True):
        for method in METHOD_IDS:
            axis.plot(
                FIR_TAP_COUNTS,
                arrays[f"{algorithm}_{method}_relative_weight_error"],
                marker="o",
                label=method,
            )
        axis.set(
            title=algorithm.upper(),
            xlabel="FIR tap count [sample]",
            ylabel="Relative weight reconstruction error" if algorithm == "ebae" else None,
            xscale="log",
            yscale="log",
        )
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.savefig(figure_dir / "fir_weight_error_sweep.png", dpi=160)
    plt.close(figure)

    # 32 tapは短FIR差が顕著、128 tapは元座標方式でもtarget peakを回復する代表点として選ぶ。
    selected_taps = (32, 128)
    figure, axes = plt.subplots(
        len(ALGORITHM_IDS),
        len(selected_taps),
        figsize=(14.0, 9.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for algorithm_index, algorithm in enumerate(ALGORITHM_IDS):
        for tap_index, tap_count in enumerate(selected_taps):
            axis = axes[algorithm_index, tap_index]
            for method in METHOD_IDS:
                approximation = approximate_weights_with_fir(
                    design.weights[algorithm][method], tap_count
                )
                approximated_original = _original_coordinate_weights(
                    method, approximation.reconstructed_weights, design.integer_phase
                )
                bl_db = _band_bl_db(approximated_original, design)
                arrays[f"{algorithm}_{method}_{tap_count}tap_bl_db_re_input_rms"] = bl_db
                axis.plot(AZIMUTH_DEG, bl_db, marker="o", label=method)
            axis.axvline(TARGET_AZIMUTH_DEG, color="black", linestyle="--", linewidth=1.0)
            axis.set(
                title=f"{algorithm.upper()} {tap_count} tap",
                xlabel="Waiting-beam azimuth [deg]",
                ylabel="RMS Level [dB re input RMS]" if tap_index == 0 else None,
                xlim=(0.0, 180.0),
                ylim=(DISPLAY_FLOOR_DB_RE_INPUT_RMS, 5.0),
            )
            axis.grid(True, alpha=0.25)
            axis.legend()
    figure.savefig(figure_dir / "source_frequency_bl_overlay.png", dpi=160)
    plt.close(figure)

    # BL配列を追加した後にNPZを更新し、描画直前の線形対応配列を成果物へ残す。
    np.savez(data_dir / f"{SCENARIO_ID}.npz", **arrays)  # pyright: ignore[reportArgumentType]

    review_lines = [
        "# EBAE/MVDR S0・S1・T1・T2 FIR長sweep",
        "",
        f"- scenario: `{SCENARIO_ID}`",
        "- evaluation pattern: `fixed_beam_single_source`",
        f"- array: {N_CHANNEL} ch ULA, spacing {SPACING_M:.1f} m",
        f"- fs / FFT / analysis width: {FS_HZ:.0f} Hz / {FFT_SIZE} / {ANALYSIS_WIDTH_HZ:.1f} Hz",
        f"- target: {TARGET_AZIMUTH_DEG:.1f} deg, {TARGET_BAND_HZ[0]:.0f}--{TARGET_BAND_HZ[1]:.0f} Hz",
        f"- tap counts: {', '.join(str(value) for value in FIR_TAP_COUNTS)}",
        "",
        "S1はS0共分散の整数遅延位相変換、T2はT1共分散の整数遅延位相変換である。",
        "S0/T1は元入力座標でFIR化し、S1/T2は整数delay line後の残留座標でFIR化する。",
        "full DFT重みを基準とし、全channel共通のcircular tap窓で打ち切って再構成する。",
        "",
    ]
    for algorithm in ALGORITHM_IDS:
        review_lines.extend(
            (
                f"- {algorithm.upper()} S0/S1 relative error: "
                f"{float(arrays[f'{algorithm}_s0_s1_relative_error'][0]):.3e}",
                f"- {algorithm.upper()} T1/T2 relative error: "
                f"{float(arrays[f'{algorithm}_t1_t2_relative_error'][0]):.3e}",
            )
        )
    review_lines.extend(
        (
            "",
            "本sweepは完成重みのFIR再現性を切り分ける静的評価である。BTR、係数更新境界、",
            "実時間runtime、実buffer latencyは未評価であり、方式採否には使用しない。",
            "",
            f"- figure: `figures/{SCENARIO_ID}/fir_weight_error_sweep.png`",
            f"- BL figure: `figures/{SCENARIO_ID}/source_frequency_bl_overlay.png`",
            f"- data: `data/{SCENARIO_ID}.npz`",
            "- metrics: `scenario_summary.csv`",
        )
    )
    (output_dir / "review_index.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    """既定条件で4方式×2アルゴリズムのFIR長sweepを実行する。"""
    write_fir_sweep_report()


if __name__ == "__main__":
    main()
