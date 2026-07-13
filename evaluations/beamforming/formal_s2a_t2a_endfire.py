"""64 ch長大ULAで正式S2a/T2aの共分散・完成重み・有限FIRを評価する。

実時間の帯域制限信号からS/T共分散を推定し、EBAE完成重みと有限tap残差FIRを
段階別に保存する。整数遅延buffer、有限tap残差FIR、target/noise/mixedの
block逐次処理まで同じmoduleから実行し、共分散破綻とFIR実現誤差を分離する。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from spflow.beamforming.ebae import EbaeConfig, design_ebae_weights_band
from evaluations.beamforming.stateful_delay_fir_runtime import (
    ResidualCausalFIRStage,
    StatefulIntegerDelayStage,
)


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]

OUTPUT_DIR = Path("artifacts/beamforming/formal_s2a_t2a_endfire/review_pack")
FS_HZ = 1024.0
# 16 sample分析窓は64 Hz幅であり、長大endfireの同一時刻S共分散を意図的に露呈させる。
ANALYSIS_FFT_SIZE = 16
HOP_SIZE = ANALYSIS_FFT_SIZE
# M^2個の非overlap snapshotを実信号から積分し、N/E AICのLと物理平均数を一致させる。
N_FRAME = 4096
FIR_DESIGN_NFFT = 4096
N_CHANNEL = 64
SPACING_M = 6.25
SOUND_SPEED_M_S = 1500.0
OCCUPIED_BAND_HZ = (40.0, 88.0)
TARGET_AZIMUTH_DEG = (0.0, 180.0)
# EBAEは最大M-1個のMUSIC peakを対応付けるため、候補数を64 ch以上確保する。
# 2 deg刻みはendfire peak誤差の判定分解能も10 deg刻みより十分小さい。
AZIMUTH_DEG = np.arange(0.0, 181.0, 2.0, dtype=np.float64)
# 正式比較は破綻側・中間・実用収束側を代表する3点に限定する。
# 1024/2048 tapは過去評価で512 tap以降の変化確認に使用済みであり、
# 512 tapが不合格または未収束となった条件だけを追加調査する参照点とする。
TAP_COUNTS = (32, 128, 512)
# 主sweepからは除外し、明示した追加実行条件が成立した場合だけ使う。
EXTENDED_TAP_COUNTS = (1024, 2048)
RUNTIME_SAMPLE_COUNT = 4096
RUNTIME_BLOCK_SIZE = 257
METHOD_IDS = ("S2a", "T2a")
RANDOM_SEED = 20260713
SOURCE_RMS = 1.0
NOISE_RMS_PER_CHANNEL = 0.1
AIC_SNAPSHOT_COUNT = N_CHANNEL * N_CHANNEL
LOW_INPUT_BAND_SNR_DB = 0.0


@dataclass(frozen=True)
class ScenarioSignals:
    """共分散評価へ渡す連続時間系列を保持する。

    Attributes:
        target: target-only入力。shapeは``[n_ch,n_sample]``、単位はinput RMS。
        noise: noise-only入力。shapeはtargetと同じ。
        mixed: target+noise入力。shapeはtargetと同じ。
        valid_start_sample: 伝搬遅延生成の端部を除外できる最初のsample index。

    本クラスは共分散推定、重み設計、FIR適用を行わない。
    """

    target: FloatArray
    noise: FloatArray
    mixed: FloatArray
    valid_start_sample: int


@dataclass(frozen=True)
class CovarianceStage:
    """A段の候補方位別S2a/T2a共分散を保持する。

    Attributes:
        covariance: shapeは``[method,n_beam,n_bin,n_ch,n_ch]``、単位はinput power。
        frequencies_hz: occupied bin中心。shapeは``[n_bin]``、単位はHz。
        integer_delays: 候補方位別整数遅延。shapeは``[n_beam,n_ch]``、単位はsample。
        physical_snapshot_count: 実測共分散へ平均したSTFT frame数。

    EBAE設計とFIR化は責務に含めない。
    """

    covariance: ComplexArray
    frequencies_hz: FloatArray
    integer_delays: IntArray
    physical_snapshot_count: int


@dataclass(frozen=True)
class WeightStage:
    """B段のEBAE完成周波数重みと診断量を保持する。

    Attributes:
        residual_weights: shapeは``[method,n_bin,n_beam,n_ch]``。
        equivalent_weights: 元入力座標重み。shapeはresidual_weightsと同じ。
        signal_counts: N/E AIC信号数。shapeは``[method,n_beam,n_bin]``。
        music_peak_deg: MUSIC最大方位。shapeはsignal_countsと同じ、単位はdeg。
        eigenvalues: 降順固有値。shapeは``[method,n_beam,n_bin,n_ch]``。
    """

    residual_weights: ComplexArray
    equivalent_weights: ComplexArray
    signal_counts: IntArray
    music_peak_deg: FloatArray
    eigenvalues: FloatArray


@dataclass(frozen=True)
class FirRealization:
    """D段の実buffer適用へ渡せる有限長残差FIRを保持する。

    Attributes:
        method_id: ``S2a``または``T2a``。
        tap_count: FIR長、単位はsample。
        coefficients: causal残差FIR。shapeは``[n_beam,n_ch,n_tap]``。
        integer_delays: 実buffer遅延。shapeは``[n_beam,n_ch]``、単位はsample。
        common_latency_samples: 全beam/channelを揃える共通latency、単位はsample。
        reconstructed_weights: occupied binで再構成した重み。shapeは``[n_bin,n_beam,n_ch]``。
        energy_containment: 採用tap区間のimpulse energy比。shapeは``[n_beam]``。
    """

    method_id: str
    tap_count: int
    coefficients: ComplexArray
    integer_delays: IntArray
    common_latency_samples: int
    reconstructed_weights: ComplexArray
    energy_containment: FloatArray


@dataclass(frozen=True)
class RuntimeOutput:
    """D段helperで得たbeam出力と有効区間を保持する。

    Attributes:
        data: beam出力。shapeは``[n_beam,n_sample]``。
        valid_mask: 初回整数遅延・FIR過渡を除くmask。shapeはdataと同じ。

    共分散推定、係数設計、level指標計算は責務に含めない。
    """

    data: ComplexArray
    valid_mask: NDArray[np.bool_]


def _arrival_delays_s(azimuth_deg: FloatArray) -> FloatArray:
    """中心基準ULAの到来遅延を返す。

    Args:
        azimuth_deg: 方位。shapeは``[n_direction]``、単位はdeg。

    Returns:
        相対遅延。shapeは``[n_direction,n_ch]``、単位はs。
    """
    positions_m = np.linspace(
        -SPACING_M * (N_CHANNEL - 1) / 2.0,
        SPACING_M * (N_CHANNEL - 1) / 2.0,
        N_CHANNEL,
        dtype=np.float64,
    )
    # tau=-r cos(theta)/c。0/180 degがULAの両endfireである。
    return np.asarray(
        -np.cos(np.deg2rad(azimuth_deg))[:, None] * positions_m[None, :] / SOUND_SPEED_M_S,
        dtype=np.float64,
    )


def theoretical_grating_azimuths(frequency_hz: float, steering_deg: float) -> tuple[float, ...]:
    """ULAの理論グレーティング方位を返す。

    Args:
        frequency_hz: 周波数、単位はHz。
        steering_deg: 待受方位、単位はdeg。

    Returns:
        ``d(cos(theta_g)-cos(theta_0))=m lambda``を満たす非零次数の方位。

    Raises:
        ValueError: 周波数が正でない場合。
    """
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    wavelength_m = SOUND_SPEED_M_S / frequency_hz
    steering_cosine = float(np.cos(np.deg2rad(steering_deg)))
    aliases: list[float] = []
    for order in range(-4, 5):
        if order == 0:
            continue
        cosine = steering_cosine + order * wavelength_m / SPACING_M
        if -1.0 <= cosine <= 1.0:
            aliases.append(float(np.rad2deg(np.arccos(cosine))))
    return tuple(sorted(aliases))


def generate_scenario_signals(target_azimuth_deg: float) -> ScenarioSignals:
    """実時間の帯域制限広帯域sourceと独立channel雑音を生成する。

    Args:
        target_azimuth_deg: source方位、単位はdeg。

    Returns:
        target-only、noise-only、mixedの連続系列。

    Notes:
        sourceは白色乱数をFFT領域で40--88 Hzへ厳密に制限し、各sensorの物理遅延を
        線形補間で与える。解析sinc共分散は使用しない。
    """
    rng = np.random.default_rng(RANDOM_SEED)
    maximum_delay = int(np.ceil(FS_HZ * SPACING_M * (N_CHANNEL - 1) / SOUND_SPEED_M_S)) + 4
    # 候補T切り出しでは負offsetを共通biasで因果化した後、正offset側にも余白が要る。
    # 前後2 spanずつを確保し、0/180 degの最大整数遅延でも最終frameを欠損させない。
    sample_count = (N_FRAME - 1) * HOP_SIZE + ANALYSIS_FFT_SIZE + 4 * maximum_delay
    white = rng.standard_normal(sample_count)
    spectrum = np.fft.rfft(white)
    frequency_hz = np.fft.rfftfreq(sample_count, d=1.0 / FS_HZ)
    band_mask = (frequency_hz >= OCCUPIED_BAND_HZ[0]) & (frequency_hz <= OCCUPIED_BAND_HZ[1])
    spectrum[~band_mask] = 0.0
    source = np.fft.irfft(spectrum, n=sample_count)
    source /= max(float(np.sqrt(np.mean(source**2))), np.finfo(np.float64).tiny)

    delays_sample = FS_HZ * _arrival_delays_s(np.asarray([target_azimuth_deg]))[0]
    sample_axis = np.arange(sample_count, dtype=np.float64)
    target = np.empty((N_CHANNEL, sample_count), dtype=np.float64)
    for channel_index, delay_sample in enumerate(delays_sample):
        # x_ch[n]=s[n-tau_ch fs]。端部は後段のvalid区間から除くためゼロ外挿する。
        target[channel_index] = np.interp(
            sample_axis - delay_sample, sample_axis, source, left=0.0, right=0.0
        )
    target *= SOURCE_RMS / max(float(np.sqrt(np.mean(target[:, maximum_delay:-maximum_delay] ** 2))), np.finfo(np.float64).tiny)
    noise = np.asarray(NOISE_RMS_PER_CHANNEL * rng.standard_normal(target.shape), dtype=np.float64)
    return ScenarioSignals(target, noise, target + noise, maximum_delay)


def _stft_snapshots(signal: FloatArray, start_sample: int, integer_delays: IntArray) -> ComplexArray:
    """候補方位の実整数sample切り出し後STFT snapshotを返す。

    Args:
        signal: 入力。shapeは``[n_ch,n_sample]``。
        start_sample: 共通有効区間先頭、単位はsample。
        integer_delays: channel別切り出しoffset。shapeは``[n_ch]``、単位はsample。

    Returns:
        rFFT snapshot。shapeは``[n_frame,n_bin,n_ch]``。
    """
    frame = np.empty((N_FRAME, N_CHANNEL, ANALYSIS_FFT_SIZE), dtype=np.float64)
    delay_bias = int(-np.min(integer_delays))
    for frame_index in range(N_FRAME):
        base = start_sample + delay_bias + frame_index * HOP_SIZE
        for channel_index in range(N_CHANNEL):
            begin = base + int(integer_delays[channel_index])
            frame[frame_index, channel_index] = signal[
                channel_index, begin : begin + ANALYSIS_FFT_SIZE
            ]
    # 16 sample rectangular分析窓のbin幅は64 Hz。40--88 Hzを単一64 Hz binへ積分する。
    transformed = np.fft.rfft(frame, axis=2) / np.sqrt(float(ANALYSIS_FFT_SIZE))
    return np.asarray(np.moveaxis(transformed, 2, 1), dtype=np.complex128)


def estimate_covariance_stage(signals: ScenarioSignals) -> CovarianceStage:
    """同じmixed入力から候補方位別S2a/T2a共分散を推定する。

    Args:
        signals: 連続時間系列。

    Returns:
        A段の共分散、周波数軸、整数遅延。
    """
    frequency_all = np.fft.rfftfreq(ANALYSIS_FFT_SIZE, d=1.0 / FS_HZ)
    # 64 Hz中心binは40--88 Hzの完成weightを代表し、bin幅全体のcoherence低下を含む。
    band = frequency_all == 64.0
    frequencies_hz = np.asarray(frequency_all[band], dtype=np.float64)
    integer_delays = np.rint(FS_HZ * _arrival_delays_s(AZIMUTH_DEG)).astype(np.int64)
    covariance = np.empty(
        (len(METHOD_IDS), AZIMUTH_DEG.size, frequencies_hz.size, N_CHANNEL, N_CHANNEL),
        dtype=np.complex128,
    )
    zero_delays = np.zeros(N_CHANNEL, dtype=np.int64)
    s_snapshot = _stft_snapshots(signals.mixed, signals.valid_start_sample, zero_delays)[:, band]
    # R[f]=mean_l X[l,f] X[l,f]^H。axis=0は物理STFT snapshotである。
    s_covariance = np.asarray(
        np.einsum("lfc,lfd->fcd", s_snapshot, s_snapshot.conj(), optimize=True) / N_FRAME,
        dtype=np.complex128,
    )
    for beam_index, delay in enumerate(integer_delays):
        phase = np.exp(
            1j * 2.0 * np.pi * frequencies_hz[:, None] * delay[None, :] / FS_HZ
        )
        # S2aは同一時刻S共分散を整数遅延後座標へunitary変換するだけで、coherenceは回復しない。
        covariance[0, beam_index] = phase[:, :, None] * s_covariance * phase[:, None, :].conj()
        t_snapshot = _stft_snapshots(signals.mixed, signals.valid_start_sample, delay)[:, band]
        # T2aは候補方位別の実整数sample切り出し後に共分散を直接生成する。
        covariance[1, beam_index] = np.asarray(
            np.einsum("lfc,lfd->fcd", t_snapshot, t_snapshot.conj(), optimize=True) / N_FRAME,
            dtype=np.complex128,
        )
    return CovarianceStage(covariance, frequencies_hz, integer_delays, N_FRAME)


def design_weight_stage(stage: CovarianceStage) -> WeightStage:
    """候補方位別共分散からEBAE完成重みを設計する。

    Args:
        stage: A段結果。

    Returns:
        B段の残差座標重み、等価元座標重み、AIC/MUSIC/固有値。
    """
    beam_delays_s = _arrival_delays_s(AZIMUTH_DEG)
    steering = np.exp(
        -1j * 2.0 * np.pi * stage.frequencies_hz[:, None, None] * beam_delays_s[None, :, :]
    )
    shape = (len(METHOD_IDS), stage.frequencies_hz.size, AZIMUTH_DEG.size, N_CHANNEL)
    residual_weights = np.empty(shape, dtype=np.complex128)
    equivalent_weights = np.empty(shape, dtype=np.complex128)
    signal_counts = np.empty(shape[:3], dtype=np.int64)
    music_peak_deg = np.empty(shape[:3], dtype=np.float64)
    eigenvalues = np.empty(shape, dtype=np.float64)
    config = EbaeConfig(
        snapshot_rate_hz=float(AIC_SNAPSHOT_COUNT),
        integration_time_sec=1.0,
        diagonal_loading=1.0,
    )
    for method_index, _method_id in enumerate(METHOD_IDS):
        for candidate_index, integer_delay in enumerate(stage.integer_delays):
            phase = np.exp(
                1j * 2.0 * np.pi * stage.frequencies_hz[:, None] * integer_delay[None, :] / FS_HZ
            )
            for frequency_index in range(stage.frequencies_hz.size):
                residual_scan = phase[frequency_index, :, None] * steering[frequency_index].T
                result = design_ebae_weights_band(
                    stage.covariance[method_index, candidate_index, frequency_index],
                    residual_scan,
                    snapshot_count=AIC_SNAPSHOT_COUNT,
                    config=config,
                )
                weight = np.asarray(result.weights[:, candidate_index], dtype=np.complex128)
                residual_weights[method_index, frequency_index, candidate_index] = weight
                # y=v^H D xより元入力座標の等価weightはD^H vである。
                equivalent_weights[method_index, frequency_index, candidate_index] = phase[frequency_index].conj() * weight
                signal_counts[method_index, frequency_index, candidate_index] = result.signal_count
                music_peak_deg[method_index, frequency_index, candidate_index] = float(
                    AZIMUTH_DEG[int(np.argmax(result.music_spectrum))]
                )
                eigenvalues[method_index, frequency_index, candidate_index] = result.eigenvalues
    return WeightStage(residual_weights, equivalent_weights, signal_counts, music_peak_deg, eigenvalues)


def realize_residual_fir(
    method_id: str,
    tap_count: int,
    covariance_stage: CovarianceStage,
    weight_stage: WeightStage,
) -> FirRealization:
    """完成残差重みを有限長causal FIRへ射影する。

    Args:
        method_id: ``S2a``または``T2a``。
        tap_count: FIR tap数、単位はsample。
        covariance_stage: 周波数軸と整数遅延を持つA段結果。
        weight_stage: 完成重みを持つB段結果。

    Returns:
        D段streaming helperへ直接渡せる有限FIR実現。

    Raises:
        ValueError: methodまたはtap数が不正な場合。
    """
    if method_id not in METHOD_IDS:
        raise ValueError(f"unsupported method_id: {method_id}")
    if tap_count <= 0:
        raise ValueError("tap_count must be positive.")
    method_index = METHOD_IDS.index(method_id)
    design_fft_size = max(FIR_DESIGN_NFFT, 2 ** int(np.ceil(np.log2(tap_count))))
    frequency_grid = np.fft.rfftfreq(design_fft_size, d=1.0 / FS_HZ)
    coefficients = np.empty((AZIMUTH_DEG.size, N_CHANNEL, tap_count), dtype=np.complex128)
    reconstructed = np.empty_like(weight_stage.residual_weights[method_index])
    energy_containment = np.empty(AZIMUTH_DEG.size, dtype=np.float64)
    for beam_index in range(AZIMUTH_DEG.size):
        full_response = np.zeros((frequency_grid.size, N_CHANNEL), dtype=np.complex128)
        occupied_mask = (frequency_grid >= OCCUPIED_BAND_HZ[0]) & (
            frequency_grid <= OCCUPIED_BAND_HZ[1]
        )
        occupied_indices = np.searchsorted(frequency_grid, covariance_stage.frequencies_hz)
        # 64 Hz分析binの完成weightを40--88 Hz全体へpiecewise一定に展開してFIR化する。
        full_response[occupied_mask] = weight_stage.residual_weights[
            method_index, 0, beam_index
        ].conj()
        impulse = np.fft.irfft(full_response, n=design_fft_size, axis=0)
        energy = np.sum(impulse**2, axis=1)
        extended = np.concatenate((energy, energy[: tap_count - 1]))
        window_energy = np.convolve(extended, np.ones(tap_count), mode="valid")[:design_fft_size]
        start = int(np.argmax(window_energy))
        total_energy = max(float(np.sum(energy)), np.finfo(np.float64).tiny)
        energy_containment[beam_index] = float(window_energy[start] / total_energy)
        indices = (start + np.arange(tap_count)) % design_fft_size
        coefficients[beam_index] = np.asarray(impulse[indices].T, dtype=np.complex128)
        response = np.fft.rfft(impulse[indices], n=design_fft_size, axis=0)
        reconstructed[:, beam_index] = response[occupied_indices].conj()
    # snapshot切り出しはx[n+tau]だが、因果bufferはx[n-d]なので、d+tauがbeam内で
    # 一定になるd=max(tau)-tauを使う。tau-min(tau)では符号が反転しendfireを二重化する。
    causal_integer_delays = (
        np.max(covariance_stage.integer_delays, axis=1, keepdims=True)
        - covariance_stage.integer_delays
    )
    common_latency = int(np.max(causal_integer_delays)) + tap_count - 1
    return FirRealization(
        method_id,
        tap_count,
        coefficients,
        np.asarray(causal_integer_delays, dtype=np.int64),
        common_latency,
        reconstructed,
        energy_containment,
    )


def apply_runtime_blocks(
    signal: FloatArray,
    realization: FirRealization,
    block_size: int,
) -> RuntimeOutput:
    """実整数遅延bufferと残差FIRをblock逐次適用する。

    Args:
        signal: channel入力。shapeは``[n_ch,n_sample]``、単位はinput RMS。
        realization: C段の有限FIRと因果整数遅延。
        block_size: 入力block長、単位はsample。

    Returns:
        channel和後のbeam波形と、全channelが有効な時刻mask。

    Raises:
        ValueError: signal shapeまたはblock長が不正な場合。
    """
    if signal.ndim != 2 or signal.shape[0] != N_CHANNEL:
        raise ValueError("signal must have shape (n_ch, n_sample).")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    n_beam = int(realization.coefficients.shape[0])
    if realization.integer_delays.shape != (n_beam, N_CHANNEL):
        raise ValueError("realization beam/channel shape is inconsistent.")
    output = np.empty((n_beam, signal.shape[1]), dtype=np.complex128)
    valid = np.empty(output.shape, dtype=np.bool_)
    for beam_index in range(n_beam):
        delay_stage = StatefulIntegerDelayStage(realization.integer_delays[beam_index])
        fir_stage = ResidualCausalFIRStage(realization.coefficients[beam_index])
        for start in range(0, signal.shape[1], block_size):
            stop = min(start + block_size, signal.shape[1])
            delayed = delay_stage.process(signal[:, start:stop])
            filtered = fir_stage.process(delayed.data, delayed.valid_mask)
            # 各channel FIRは完成weightの実適用応答を含むため、ここではchannel軸を加算する。
            output[beam_index, start:stop] = np.sum(filtered.data, axis=0)
            valid[beam_index, start:stop] = np.all(filtered.valid_mask, axis=0)
    return RuntimeOutput(output, valid)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """同じfieldを持つ行をUTF-8 CSVへ保存する。"""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _target_beam_realization(realization: FirRealization, target_deg: float) -> FirRealization:
    """時間領域評価用に真値方位の1 beamだけを取り出す。"""
    beam_index = int(np.argmin(np.abs(AZIMUTH_DEG - target_deg)))
    return FirRealization(
        realization.method_id,
        realization.tap_count,
        realization.coefficients[beam_index : beam_index + 1],
        realization.integer_delays[beam_index : beam_index + 1],
        realization.common_latency_samples,
        realization.reconstructed_weights[:, beam_index : beam_index + 1],
        realization.energy_containment[beam_index : beam_index + 1],
    )


def _waveform_metrics(
    target_output: ComplexArray,
    noise_output: ComplexArray,
    mixed_output: ComplexArray,
    valid_mask: NDArray[np.bool_],
    reference: FloatArray,
) -> dict[str, float | bool]:
    """共通有効区間のlevel、相関、誤差、SNR、mixed整合を返す。"""
    valid = np.asarray(valid_mask[0], dtype=np.bool_)
    y_target = np.real(target_output[0, valid])
    y_noise = np.real(noise_output[0, valid])
    y_mixed = np.real(mixed_output[0, valid])
    reference_valid = np.asarray(reference[-y_target.size :], dtype=np.float64)
    # FIRの共通shiftを除くため相互相関最大lagで比較する。gain補正は行わずlevel誤差を残す。
    correlation_sequence = np.correlate(y_target, reference_valid, mode="full")
    lag = int(np.argmax(np.abs(correlation_sequence))) - (reference_valid.size - 1)
    if lag >= 0:
        aligned_output = y_target[lag:]
        aligned_reference = reference_valid[: aligned_output.size]
    else:
        aligned_reference = reference_valid[-lag:]
        aligned_output = y_target[: aligned_reference.size]
    output_rms = float(np.sqrt(np.mean(aligned_output**2)))
    reference_rms = float(np.sqrt(np.mean(aligned_reference**2)))
    waveform_correlation = float(np.corrcoef(aligned_output, aligned_reference)[0, 1])
    rms_error = float(np.sqrt(np.mean((aligned_output - aligned_reference) ** 2)))
    target_power = float(np.mean(y_target**2))
    noise_power = float(np.mean(y_noise**2))
    mixed_consistency = float(
        np.sqrt(np.mean((y_mixed - (y_target + y_noise)) ** 2))
    )
    return {
        "valid_sample_count": float(np.count_nonzero(valid)),
        "alignment_lag_sample": float(lag),
        "target_rms_db_re_input_rms": 20.0 * np.log10(max(output_rms, np.finfo(float).tiny)),
        "waveform_correlation": waveform_correlation,
        "waveform_rms_error_re_input_rms": rms_error / max(reference_rms, np.finfo(float).tiny),
        "noise_rms_db_re_input_rms": 10.0 * np.log10(max(noise_power, np.finfo(float).tiny)),
        "output_snr_db": 10.0 * np.log10(
            max(target_power, np.finfo(float).tiny) / max(noise_power, np.finfo(float).tiny)
        ),
        "mixed_component_rms_error": mixed_consistency,
        "finite": bool(
            np.all(np.isfinite(target_output))
            and np.all(np.isfinite(noise_output))
            and np.all(np.isfinite(mixed_output))
        ),
    }


def run_evaluation(output_dir: Path = OUTPUT_DIR) -> None:
    """0/180 degのA/B/C評価と再現可能成果物を生成する。

    Args:
        output_dir: review pack出力先。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    covariance_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    fir_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    bl_records: list[tuple[int, int, int, FloatArray, FloatArray]] = []
    frequency_records: list[tuple[int, int, int, FloatArray, ComplexArray]] = []
    waveform_records: list[tuple[int, int, IntArray, ComplexArray, ComplexArray]] = []
    # NumPy stubはsavezの可変キーワードを特殊引数と誤認する版があるため、値型をAnyに限定する。
    npz_arrays: dict[str, Any] = {}
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharex=True)
    for scenario_index, target_deg in enumerate(TARGET_AZIMUTH_DEG):
        signals = generate_scenario_signals(target_deg)
        covariance_stage = estimate_covariance_stage(signals)
        weight_stage = design_weight_stage(covariance_stage)
        target_delay_s = _arrival_delays_s(np.asarray([target_deg]))[0]
        source_steering = np.exp(
            -1j * 2.0 * np.pi * covariance_stage.frequencies_hz[:, None] * target_delay_s[None, :]
        )
        for method_index, method_id in enumerate(METHOD_IDS):
            response = np.einsum(
                "fbc,fc->fb", weight_stage.equivalent_weights[method_index].conj(), source_steering,
                optimize=True,
            )
            bl_db = 10.0 * np.log10(np.maximum(np.mean(np.abs(response) ** 2, axis=0), 1.0e-12))
            axes[scenario_index, 0].plot(AZIMUTH_DEG, bl_db, marker="o", label=method_id)
            for beam_index, beam_deg in enumerate(AZIMUTH_DEG):
                counts = weight_stage.signal_counts[method_index, :, beam_index]
                peaks = weight_stage.music_peak_deg[method_index, :, beam_index]
                values = weight_stage.eigenvalues[method_index, :, beam_index]
                median_values = np.median(values, axis=0)
                trace_value = max(float(np.sum(median_values)), np.finfo(float).tiny)
                eigenvalue_power = max(
                    float(np.sum(median_values**2)), np.finfo(float).tiny
                )
                rank_one_residual = float(
                    np.sqrt(np.sum(median_values[1:] ** 2) / eigenvalue_power)
                )
                covariance_matrix = covariance_stage.covariance[
                    method_index, beam_index, 0
                ]
                diagonal = np.maximum(
                    np.real(np.diag(covariance_matrix)), np.finfo(float).tiny
                )
                first, second = np.tril_indices(N_CHANNEL, k=-1)
                coherence = np.abs(covariance_matrix[first, second]) / np.sqrt(
                    diagonal[first] * diagonal[second]
                )
                covariance_rows.append({
                    "target_deg": target_deg, "method": method_id, "beam_deg": beam_deg,
                    "signal_count_median": float(np.median(counts)),
                    "music_peak_median_deg": float(np.median(peaks)),
                    "eigenvalue_1_median": float(np.median(values[:, 0])),
                    "eigenvalue_64_median": float(np.median(values[:, -1])),
                    "condition_number_median": float(np.median(values[:, 0] / np.maximum(values[:, -1], 1.0e-15))),
                    "principal_eigenvalue_fraction": float(median_values[0] / trace_value),
                    "rank_one_residual": rank_one_residual,
                    "pair_coherence_median": float(np.median(coherence)),
                    "physical_snapshot_count": covariance_stage.physical_snapshot_count,
                    "aic_snapshot_count": AIC_SNAPSHOT_COUNT,
                })
                weight_rows.append({
                    "target_deg": target_deg, "method": method_id, "beam_deg": beam_deg,
                    "response_rms": float(np.sqrt(np.mean(np.abs(response[:, beam_index]) ** 2))),
                    "response_peak_frequency_hz": float(covariance_stage.frequencies_hz[int(np.argmax(np.abs(response[:, beam_index])))]),
                })
            for tap_count in TAP_COUNTS:
                realization = realize_residual_fir(method_id, tap_count, covariance_stage, weight_stage)
                reference = weight_stage.residual_weights[method_index]
                relative_error = np.linalg.norm(realization.reconstructed_weights - reference) / max(
                    float(np.linalg.norm(reference)), np.finfo(np.float64).tiny
                )
                energy = float(np.sum(np.abs(realization.coefficients) ** 2))
                fir_row: dict[str, Any] = {
                    "target_deg": target_deg, "method": method_id, "tap_count": tap_count,
                    "relative_weight_error": float(relative_error), "fir_energy": energy,
                    "minimum_energy_containment": float(np.min(realization.energy_containment)),
                    "target_energy_containment": float(
                        realization.energy_containment[int(np.argmin(np.abs(AZIMUTH_DEG - target_deg)))]
                    ),
                    "common_latency_samples": realization.common_latency_samples,
                    "finite": bool(np.all(np.isfinite(realization.coefficients))),
                }
                dense_frequency_hz = np.fft.rfftfreq(FIR_DESIGN_NFFT, d=1.0 / FS_HZ)
                dense_band = (dense_frequency_hz >= OCCUPIED_BAND_HZ[0]) & (
                    dense_frequency_hz <= OCCUPIED_BAND_HZ[1]
                )
                reconstructed_apply = np.fft.fft(
                    realization.coefficients, n=FIR_DESIGN_NFFT, axis=2
                )[:, :, : FIR_DESIGN_NFFT // 2 + 1]
                dense_source_steering = np.exp(
                    -1j
                    * 2.0
                    * np.pi
                    * dense_frequency_hz[dense_band, None]
                    * target_delay_s[None, :]
                )
                dense_integer_phase = np.exp(
                    -1j
                    * 2.0
                    * np.pi
                    * dense_frequency_hz[dense_band, None, None]
                    * realization.integer_delays[None, :, :]
                    / FS_HZ
                )
                # apply response shape [n_band,n_beam,n_ch]。integer bufferと残差FIRを直列合成する。
                total_apply = (
                    np.moveaxis(reconstructed_apply[:, :, dense_band], 2, 0)
                    * dense_integer_phase
                )
                target_response = np.einsum(
                    "fbc,fc->fb", total_apply, dense_source_steering, optimize=True
                )
                target_beam_index = int(np.argmin(np.abs(AZIMUTH_DEG - target_deg)))
                target_transfer = np.asarray(
                    target_response[:, target_beam_index], dtype=np.complex128
                )
                phase_rad = np.unwrap(np.angle(target_transfer))
                group_delay_sample = -FS_HZ * np.gradient(
                    phase_rad, 2.0 * np.pi * dense_frequency_hz[dense_band]
                )
                amplitude_db = 20.0 * np.log10(
                    np.maximum(np.abs(target_transfer), 1.0e-12)
                )
                interior = (
                    dense_frequency_hz[dense_band] >= OCCUPIED_BAND_HZ[0] + 2.0
                ) & (
                    dense_frequency_hz[dense_band] <= OCCUPIED_BAND_HZ[1] - 2.0
                )
                fir_row.update(
                    {
                        "target_amplitude_error_max_abs_db": float(
                            np.max(np.abs(amplitude_db))
                        ),
                        "target_amplitude_interior_max_abs_db": float(
                            np.max(np.abs(amplitude_db[interior]))
                        ),
                        "group_delay_mean_sample": float(np.mean(group_delay_sample)),
                        "group_delay_ripple_sample": float(
                            np.max(group_delay_sample) - np.min(group_delay_sample)
                        ),
                    }
                )
                fir_rows.append(fir_row)
                frequency_records.append(
                    (
                        scenario_index,
                        method_index,
                        tap_count,
                        np.asarray(dense_frequency_hz[dense_band], dtype=np.float64),
                        target_transfer,
                    )
                )
                target_bl = 10.0 * np.log10(
                    np.maximum(np.mean(np.abs(target_response) ** 2, axis=0), 1.0e-12)
                )
                noise_power = NOISE_RMS_PER_CHANNEL**2 * np.sum(
                    np.abs(realization.coefficients) ** 2, axis=(1, 2)
                )
                noise_bl = 10.0 * np.log10(np.maximum(noise_power, 1.0e-12))
                finite_peak_index = int(np.argmax(target_bl))
                fir_row.update(
                    {
                        "finite_bl_peak_deg": float(AZIMUTH_DEG[finite_peak_index]),
                        "finite_bl_peak_error_deg": float(
                            abs(AZIMUTH_DEG[finite_peak_index] - target_deg)
                        ),
                        "finite_target_level_db_re_input_rms": float(
                            target_bl[target_beam_index]
                        ),
                        "finite_noise_level_db_re_input_rms": float(
                            noise_bl[target_beam_index]
                        ),
                    }
                )
                bl_records.append(
                    (scenario_index, method_index, tap_count, target_bl, noise_bl)
                )

                runtime_realization = _target_beam_realization(realization, target_deg)
                begin = signals.valid_start_sample
                stop = min(begin + RUNTIME_SAMPLE_COUNT, signals.target.shape[1])
                target_input = signals.target[:, begin:stop]
                noise_input = signals.noise[:, begin:stop]
                mixed_input = signals.mixed[:, begin:stop]
                target_runtime = apply_runtime_blocks(
                    target_input, runtime_realization, RUNTIME_BLOCK_SIZE
                )
                noise_runtime = apply_runtime_blocks(
                    noise_input, runtime_realization, RUNTIME_BLOCK_SIZE
                )
                mixed_runtime = apply_runtime_blocks(
                    mixed_input, runtime_realization, RUNTIME_BLOCK_SIZE
                )
                monolithic = apply_runtime_blocks(
                    target_input, runtime_realization, target_input.shape[1]
                )
                runtime_metric = _waveform_metrics(
                    target_runtime.data,
                    noise_runtime.data,
                    mixed_runtime.data,
                    target_runtime.valid_mask,
                    target_input[N_CHANNEL // 2],
                )
                runtime_rows.append(
                    {
                        "target_deg": target_deg,
                        "method": method_id,
                        "tap_count": tap_count,
                        "block_size": RUNTIME_BLOCK_SIZE,
                        "common_latency_samples": runtime_realization.common_latency_samples,
                        "first_valid_sample": int(
                            np.flatnonzero(target_runtime.valid_mask[0])[0]
                        ),
                        "block_monolithic_max_abs_error": float(
                            np.max(np.abs(target_runtime.data - monolithic.data))
                        ),
                        **runtime_metric,
                    }
                )
                if tap_count == max(TAP_COUNTS):
                    valid_indices = np.flatnonzero(target_runtime.valid_mask[0])
                    waveform_records.append(
                        (
                            scenario_index,
                            method_index,
                            np.asarray(valid_indices[:512], dtype=np.int64),
                            np.asarray(target_runtime.data[0, valid_indices[:512]], dtype=np.complex128),
                            np.asarray(monolithic.data[0, valid_indices[:512]], dtype=np.complex128),
                        )
                    )
                axes[scenario_index, 1].plot(tap_count, relative_error, marker="o", label=method_id if tap_count == TAP_COUNTS[0] else None)
        axes[scenario_index, 0].axvline(target_deg, color="black", linestyle="--", linewidth=1.0)
        axes[scenario_index, 0].set_ylabel("Target-only RMS Level [dB re input RMS]")
        axes[scenario_index, 0].legend()
        axes[scenario_index, 1].set_yscale("log")
        axes[scenario_index, 1].set_ylabel("Occupied-band relative weight error")
        npz_arrays[f"target_{int(target_deg)}_covariance"] = covariance_stage.covariance
        npz_arrays[f"target_{int(target_deg)}_weights"] = weight_stage.equivalent_weights
        npz_arrays[f"target_{int(target_deg)}_signal_counts"] = weight_stage.signal_counts
    for axis in axes[-1]:
        axis.set_xlabel("Waiting azimuth [deg]" if axis is axes[-1, 0] else "Tap count")
    figure.tight_layout()
    figure.savefig(output_dir / "abc_summary.png", dpi=160)
    plt.close(figure)
    _write_rows(output_dir / "covariance_metrics.csv", covariance_rows)
    _write_rows(output_dir / "weight_metrics.csv", weight_rows)
    _write_rows(output_dir / "fir_metrics.csv", fir_rows)
    _write_rows(output_dir / "runtime_metrics.csv", runtime_rows)
    for scenario_index, method_index, indices, streamed, monolithic in waveform_records:
        prefix = (
            f"target_{int(TARGET_AZIMUTH_DEG[scenario_index])}_{METHOD_IDS[method_index]}_512tap"
        )
        npz_arrays[f"{prefix}_sample_index"] = indices
        npz_arrays[f"{prefix}_streamed"] = streamed
        npz_arrays[f"{prefix}_monolithic"] = monolithic
    np.savez_compressed(output_dir / "abc_arrays.npz", **npz_arrays)

    bl_figure, bl_axes = plt.subplots(2, 2, figsize=(14.0, 8.0), sharex=True, sharey=True)
    finite_levels = np.concatenate(
        [np.concatenate((target_bl, noise_bl)) for _, _, _, target_bl, noise_bl in bl_records]
    )
    lower = float(np.floor((np.min(finite_levels) - 5.0) / 10.0) * 10.0)
    upper = float(np.ceil((np.max(finite_levels) + 3.0) / 5.0) * 5.0)
    for scenario_index, method_index, tap_count, target_bl, noise_bl in bl_records:
        axis = bl_axes[scenario_index, method_index]
        axis.plot(AZIMUTH_DEG, target_bl, label=f"target {tap_count} tap")
        axis.plot(
            AZIMUTH_DEG,
            noise_bl,
            linestyle=":",
            label=f"noise {tap_count} tap",
        )
        axis.set_ylim(lower, upper)
        axis.grid(alpha=0.25)
        axis.set_title(
            f"{METHOD_IDS[method_index]}, target {TARGET_AZIMUTH_DEG[scenario_index]:g} deg"
        )
        axis.set_ylabel("RMS Level [dB re input RMS]")
        axis.legend(fontsize=7, ncol=2)
    for axis in bl_axes[-1]:
        axis.set_xlabel("Waiting-beam azimuth [deg]")
    bl_figure.tight_layout()
    bl_figure.savefig(output_dir / "bl_target_noise_components.png", dpi=160)
    plt.close(bl_figure)

    # 現noiseはfull-band RMSで定義されるため、占有帯域へ積分した入力SNRを明示する。
    occupied_bandwidth_hz = OCCUPIED_BAND_HZ[1] - OCCUPIED_BAND_HZ[0]
    high_input_band_noise_power = (
        NOISE_RMS_PER_CHANNEL**2 * occupied_bandwidth_hz / (FS_HZ / 2.0)
    )
    high_input_band_snr_db = 10.0 * np.log10(
        SOURCE_RMS**2 / high_input_band_noise_power
    )
    low_noise_power_scale = 10.0 ** (
        (high_input_band_snr_db - LOW_INPUT_BAND_SNR_DB) / 10.0
    )
    mixed_rows: list[dict[str, Any]] = []
    mixed_arrays: dict[str, Any] = {"azimuth_deg": AZIMUTH_DEG}
    mixed_records: dict[str, list[tuple[int, int, int, FloatArray]]] = {
        "high": [],
        "low": [],
    }
    for scenario_index, method_index, tap_count, target_bl, noise_bl in bl_records:
        target_power = 10.0 ** (target_bl / 10.0)
        high_noise_power = 10.0 ** (noise_bl / 10.0)
        for condition_id, input_snr_db, noise_power in (
            ("high", high_input_band_snr_db, high_noise_power),
            ("low", LOW_INPUT_BAND_SNR_DB, high_noise_power * low_noise_power_scale),
        ):
            mixed_bl = 10.0 * np.log10(
                np.maximum(target_power + noise_power, 1.0e-12)
            )
            mixed_records[condition_id].append(
                (scenario_index, method_index, tap_count, mixed_bl)
            )
            peak_index = int(np.argmax(mixed_bl))
            mixed_rows.append(
                {
                    "condition": condition_id,
                    "input_band_snr_db": float(input_snr_db),
                    "target_deg": TARGET_AZIMUTH_DEG[scenario_index],
                    "method": METHOD_IDS[method_index],
                    "tap_count": tap_count,
                    "mixed_peak_deg": float(AZIMUTH_DEG[peak_index]),
                    "mixed_peak_error_deg": float(
                        abs(AZIMUTH_DEG[peak_index] - TARGET_AZIMUTH_DEG[scenario_index])
                    ),
                    "mixed_target_level_db_re_input_rms": float(
                        mixed_bl[
                            int(
                                np.argmin(
                                    np.abs(
                                        AZIMUTH_DEG - TARGET_AZIMUTH_DEG[scenario_index]
                                    )
                                )
                            )
                        ]
                    ),
                }
            )
            key = (
                f"{condition_id}_target_{int(TARGET_AZIMUTH_DEG[scenario_index])}_"
                f"{METHOD_IDS[method_index]}_{tap_count}tap_db_re_input_rms"
            )
            mixed_arrays[key] = mixed_bl
    _write_rows(output_dir / "mixed_bl_metrics.csv", mixed_rows)
    np.savez_compressed(output_dir / "mixed_bl_arrays.npz", **mixed_arrays)

    for condition_id, title, input_snr_db in (
        ("high", "High-SNR mixed BL", high_input_band_snr_db),
        ("low", "Low-SNR mixed BL", LOW_INPUT_BAND_SNR_DB),
    ):
        mixed_figure, mixed_axes = plt.subplots(
            2, 2, figsize=(14.0, 8.0), sharex=True, sharey=True
        )
        condition_levels = np.concatenate(
            [record[3] for record in mixed_records[condition_id]]
        )
        mixed_lower = float(
            np.floor((np.min(condition_levels) - 5.0) / 10.0) * 10.0
        )
        mixed_upper = float(
            np.ceil((np.max(condition_levels) + 3.0) / 5.0) * 5.0
        )
        for scenario_index, method_index, tap_count, mixed_bl in mixed_records[condition_id]:
            axis = mixed_axes[scenario_index, method_index]
            axis.plot(AZIMUTH_DEG, mixed_bl, label=f"{tap_count} tap")
        for scenario_index in range(len(TARGET_AZIMUTH_DEG)):
            for method_index in range(len(METHOD_IDS)):
                axis = mixed_axes[scenario_index, method_index]
                axis.axvline(
                    TARGET_AZIMUTH_DEG[scenario_index],
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                )
                axis.set(
                    title=(
                        f"{METHOD_IDS[method_index]}, target "
                        f"{TARGET_AZIMUTH_DEG[scenario_index]:g} deg"
                    ),
                    ylabel="Mixed RMS Level [dB re input RMS]",
                    ylim=(mixed_lower, mixed_upper),
                )
                axis.grid(alpha=0.25)
                axis.legend(fontsize=8)
        for axis in mixed_axes[-1]:
            axis.set_xlabel("Waiting-beam azimuth [deg]")
        # BL単独でも狭帯域試験と誤読されないよう、信号種別と占有帯域をタイトルへ残す。
        mixed_figure.suptitle(
            f"{title}: broadband {OCCUPIED_BAND_HZ[0]:g}--"
            f"{OCCUPIED_BAND_HZ[1]:g} Hz, input band SNR {input_snr_db:.2f} dB"
        )
        mixed_figure.tight_layout()
        mixed_figure.savefig(
            output_dir / f"bl_broadband_mixed_{condition_id}_snr.png",
            dpi=160,
            facecolor="white",
        )
        plt.close(mixed_figure)

    frequency_figure, frequency_axes = plt.subplots(
        2, 2, figsize=(14.0, 8.0), sharex=True, sharey=True
    )
    for scenario_index, method_index, tap_count, frequency_hz, transfer in frequency_records:
        frequency_axes[scenario_index, method_index].plot(
            frequency_hz,
            20.0 * np.log10(np.maximum(np.abs(transfer), 1.0e-12)),
            label=f"{tap_count} tap",
        )
    for scenario_index in range(len(TARGET_AZIMUTH_DEG)):
        for method_index in range(len(METHOD_IDS)):
            axis = frequency_axes[scenario_index, method_index]
            axis.set_title(
                f"{METHOD_IDS[method_index]}, target {TARGET_AZIMUTH_DEG[scenario_index]:g} deg"
            )
            axis.set_ylabel("Amplitude [dB re distortionless response]")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    for axis in frequency_axes[-1]:
        axis.set_xlabel("Frequency [Hz]")
    frequency_figure.tight_layout()
    frequency_figure.savefig(output_dir / "frequency_response.png", dpi=160)
    plt.close(frequency_figure)

    waveform_figure, waveform_axes = plt.subplots(
        2, 2, figsize=(14.0, 7.0), sharex=True, sharey=True
    )
    for scenario_index, method_index, indices, streamed, monolithic in waveform_records:
        axis = waveform_axes[scenario_index, method_index]
        axis.plot(indices, np.real(monolithic), label="monolithic", linewidth=1.5)
        axis.plot(indices, np.real(streamed), linestyle="--", label="block streaming")
        axis.set_title(
            f"{METHOD_IDS[method_index]}, target {TARGET_AZIMUTH_DEG[scenario_index]:g} deg, 512 tap"
        )
        axis.set_ylabel("Output [input RMS amplitude]")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in waveform_axes[-1]:
        axis.set_xlabel("Global sample index")
    waveform_figure.tight_layout()
    waveform_figure.savefig(output_dir / "waveform_block_boundary.png", dpi=160)
    plt.close(waveform_figure)

    (output_dir / "review_index.md").write_text(
        "# 正式S2a/T2a広帯域endfire評価\n\n"
        "64 ch、6.25 m間隔、393.75 m開口、40--88 Hz実時間広帯域信号、"
        "64 Hz分析幅で0/180 degを評価した。共分散は4096個の非overlap snapshot、"
        "N/E AICはL=M^2=4096を使用する。\n\n"
        "S2a/T2aとも実整数delay buffer、32/128/512 tap残差FIR、257 sample blockを通す。"
        "target-only、noise-only、target+noiseを分離し、noise-only BLを物理noise floorとして描く。\n\n"
        "主表示は約30.28 dB入力帯域SNRのmixed BLとし、同じ完成weightへ0 dB相当noiseを"
        "与えた固定weight stressを別図にする。低SNRでの重み再推定試験とは区別する。\n\n"
        "合否観点はMUSIC peak誤差2 deg以下、有限BL peak誤差2 deg以下、target level誤差0.5 dB以下、"
        "帯域内端2 Hzを除く振幅誤差1 dB以下、波形相関0.99以上、energy包含率0.98以上、"
        "block/monolithic最大差1e-6以下とする。AIC信号数は真値1との一致を独立判定する。\n\n"
        "- `covariance_metrics.csv`: 固有値、AIC、MUSIC、rank/coherence\n"
        "- `weight_metrics.csv`: 完成周波数重み応答\n"
        "- `fir_metrics.csv`: FIR energy、振幅、群遅延、有限BL\n"
        "- `runtime_metrics.csv`: target/noise/mixed、波形、SNR、block境界\n"
        "- `abc_arrays.npz`: 図と監査用配列\n"
        "- `abc_summary.png`: A--C概要\n"
        "- `bl_target_noise_components.png`: target/noise分離BL\n"
        "- `bl_broadband_mixed_high_snr.png`: 40--88 Hz広帯域・高SNRの主表示mixed BL\n"
        "- `bl_broadband_mixed_low_snr.png`: 40--88 Hz広帯域・0 dB入力帯域SNRのmixed BL\n"
        "- `frequency_response.png`: 帯域内振幅応答\n"
        "- `waveform_block_boundary.png`: 一括/streaming境界比較\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_evaluation()
