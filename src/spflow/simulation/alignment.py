"""整相方式の重み設計と有限FIR実現を再現可能に比較する支援部品。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow.beamforming.ebae import EbaeConfig, design_ebae_weights_band
from spflow.simulation.numerics import SimulationPrecision

FloatArray = NDArray[np.floating[Any]]
ComplexArray = NDArray[np.complexfloating[Any, Any]]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

ALIGNMENT_ALGORITHM_IDS = ("ebae", "mvdr")
ALIGNMENT_METHOD_IDS = ("S1", "S2a", "T1", "T2a")


@dataclass(frozen=True)
class AlignmentSimulationConfig:
    """整相重み設計に必要な物理条件と離散化条件を保持する。

    センサ位置はULA軸上の座標 ``[n_ch]``、方位は ``[n_beam]``、単位はそれぞれ
    mとdegである。図表、artifact、合否閾値、parameter sweepは責務に含めない。
    """

    fs_hz: float
    fft_size: int
    sound_speed_m_per_s: float
    sensor_positions_m: FloatArray
    beam_azimuth_deg: FloatArray
    target_azimuth_deg: float
    target_band_hz: tuple[float, float]
    analysis_width_hz: float
    source_band_rms_power: float
    noise_power_per_bin_re_input_rms2: float
    ebae_diagonal_loading: float = 1.0
    mvdr_diagonal_loading_ratio: float = 1.0e-3
    precision: SimulationPrecision = SimulationPrecision.DOUBLE

    def __post_init__(self) -> None:
        """shape、単位上の範囲、DFT鏡映条件を構築時に検証する。"""
        if not isinstance(self.precision, SimulationPrecision):
            raise ValueError("precision must be a SimulationPrecision value.")
        positions = np.asarray(self.sensor_positions_m, dtype=self.precision.real_dtype)
        azimuths = np.asarray(self.beam_azimuth_deg, dtype=self.precision.real_dtype)
        if positions.ndim != 1 or positions.size == 0 or not bool(np.all(np.isfinite(positions))):
            raise ValueError("sensor_positions_m must be a finite non-empty 1-D array.")
        if azimuths.ndim != 1 or azimuths.size == 0 or not bool(np.all(np.isfinite(azimuths))):
            raise ValueError("beam_azimuth_deg must be a finite non-empty 1-D array.")
        if self.fs_hz <= 0.0 or self.sound_speed_m_per_s <= 0.0:
            raise ValueError("fs_hz and sound_speed_m_per_s must be positive.")
        if self.fft_size <= 0 or self.fft_size % 2 != 0:
            raise ValueError("fft_size must be a positive even integer.")
        band_low_hz, band_high_hz = self.target_band_hz
        if not 0.0 <= band_low_hz <= band_high_hz <= self.fs_hz / 2.0:
            raise ValueError("target_band_hz must be ordered inside [0, fs_hz / 2].")
        if self.analysis_width_hz < 0.0 or self.source_band_rms_power < 0.0:
            raise ValueError("analysis width and source power must be non-negative.")
        if self.noise_power_per_bin_re_input_rms2 <= 0.0:
            raise ValueError("noise power must be positive.")
        if self.ebae_diagonal_loading < 0.0 or self.mvdr_diagonal_loading_ratio < 0.0:
            raise ValueError("diagonal loading values must be non-negative.")
        # 呼出側の配列変更で設計条件が暗黙に変わらないよう、設定自身が専用copyを保持する。
        positions = positions.copy()
        azimuths = azimuths.copy()
        positions.setflags(write=False)
        azimuths.setflags(write=False)
        object.__setattr__(self, "sensor_positions_m", positions)
        object.__setattr__(self, "beam_azimuth_deg", azimuths)


@dataclass(frozen=True)
class AlignmentWeightDesign:
    """EBAE/MVDRの完成重みと座標変換・診断量を保持する。

    重み、整数遅延位相、steeringは shape ``[n_fft,n_beam,n_ch]``、source
    steeringは ``[n_fft,n_ch]``、source maskは ``[n_fft]`` である。評価指標や
    可視化は責務に含めない。
    """

    config: AlignmentSimulationConfig
    weights: dict[str, dict[str, ComplexArray]]
    integer_phase: ComplexArray
    steering: ComplexArray
    source_steering: ComplexArray
    source_bin_mask: BoolArray
    ebae_signal_counts: dict[str, IntArray]
    ebae_associated_beams: dict[str, IntArray]


@dataclass(frozen=True)
class FrequencyWeightFirApproximation:
    """周波数重みを共通tap窓で有限FIR化した結果を保持する。

    再構成重みは ``[n_fft,n_beam,n_ch]``、energy比と窓先頭は ``[n_beam]``。
    窓先頭の単位はsampleである。
    """

    reconstructed_weights: ComplexArray
    energy_ratio: FloatArray
    window_start_samples: IntArray


def calculate_ula_arrival_delays_s(
    sensor_positions_m: FloatArray,
    azimuth_deg: FloatArray,
    sound_speed_m_per_s: float,
) -> FloatArray:
    """ULA軸位置と方位から基準点に対する到来遅延を計算する。

    Args:
        sensor_positions_m: ULA軸位置。shape ``[n_ch]``、単位m。
        azimuth_deg: 方位。shape ``[n_direction]``、単位deg。0/180 degがendfire。
        sound_speed_m_per_s: 伝搬速度、単位m/s。

    Returns:
        到来遅延。shape ``[n_direction,n_ch]``、単位s。

    Raises:
        ValueError: 入力が1次元でない、非有限、空、または音速が正でない場合。
    """
    input_dtype = np.result_type(sensor_positions_m, azimuth_deg)
    real_dtype = (
        np.dtype(np.float32) if input_dtype == np.dtype(np.float32) else np.dtype(np.float64)
    )
    positions = np.asarray(sensor_positions_m, dtype=real_dtype)
    azimuths = np.asarray(azimuth_deg, dtype=real_dtype)
    if positions.ndim != 1 or positions.size == 0 or azimuths.ndim != 1 or azimuths.size == 0:
        raise ValueError("positions and azimuths must be non-empty 1-D arrays.")
    if not bool(np.all(np.isfinite(positions))) or not bool(np.all(np.isfinite(azimuths))):
        raise ValueError("positions and azimuths must be finite.")
    if sound_speed_m_per_s <= 0.0:
        raise ValueError("sound_speed_m_per_s must be positive.")
    # tau=-r cos(theta)/c。axis=0は方位、axis=1はchannelを表す。
    return np.asarray(
        -np.cos(np.deg2rad(azimuths))[:, None] * positions[None, :] / sound_speed_m_per_s,
        dtype=real_dtype,
    )


def _steering(delays_s: FloatArray, frequencies_hz: FloatArray) -> ComplexArray:
    # a(f,theta,ch)=exp(-j2πf tau)。axisは周波数、方位、channelの順である。
    complex_dtype = (
        np.dtype(np.complex64)
        if delays_s.dtype == np.dtype(np.float32)
        else np.dtype(np.complex128)
    )
    return np.asarray(
        np.exp(-1j * 2.0 * np.pi * frequencies_hz[:, None, None] * delays_s[None, :, :]),
        dtype=complex_dtype,
    )


def _source_covariance(
    config: AlignmentSimulationConfig,
    source_delay_s: FloatArray,
    source_steering: ComplexArray,
    candidate_delay_s: FloatArray | None,
    source_power: float,
) -> ComplexArray:
    residual_delay_s = source_delay_s
    if candidate_delay_s is not None:
        # T共分散では候補方位の整数sample時刻差を切出しで除き、残留遅延だけを積分する。
        quantized_candidate_s = np.rint(candidate_delay_s * config.fs_hz) / config.fs_hz
        residual_delay_s = source_delay_s - quantized_candidate_s
    pair_delay_s = residual_delay_s[:, None] - residual_delay_s[None, :]
    # 平坦なbin内積分のpair coherenceは sinc(Δf Δtau) に対応する。
    coherence = np.sinc(config.analysis_width_hz * pair_delay_s)
    outer = source_steering[:, None] * source_steering.conj()[None, :]
    return np.asarray(
        source_power * coherence * outer
        + config.noise_power_per_bin_re_input_rms2
        * np.eye(source_delay_s.size, dtype=config.precision.complex_dtype),
        dtype=config.precision.complex_dtype,
    )


def _mvdr_weight(
    config: AlignmentSimulationConfig, covariance: ComplexArray, steering: ComplexArray
) -> ComplexArray:
    hermitian = np.asarray(
        0.5 * (covariance + covariance.conj().T), dtype=config.precision.complex_dtype
    )
    n_channel = steering.size
    average_power = float(np.real(np.trace(hermitian))) / float(n_channel)
    # trace比例loadingはunitary位相回転で不変であり、座標変換前後の同値性を保つ。
    loaded = hermitian + config.mvdr_diagonal_loading_ratio * average_power * np.eye(
        n_channel, dtype=config.precision.complex_dtype
    )
    solved = np.linalg.solve(loaded, steering)
    return np.asarray(solved / np.vdot(steering, solved), dtype=config.precision.complex_dtype)


def _ebae_weight(
    config: AlignmentSimulationConfig,
    covariance: ComplexArray,
    steering_scan: ComplexArray,
    beam_index: int,
) -> tuple[ComplexArray, int, int]:
    n_channel = covariance.shape[0]
    result = design_ebae_weights_band(
        covariance,
        steering_scan,
        snapshot_count=n_channel * n_channel,
        config=EbaeConfig(
            snapshot_rate_hz=float(n_channel * n_channel),
            integration_time_sec=1.0,
            sigmoid_slope=10.0,
            sigmoid_midpoint=0.5,
            diagonal_loading=config.ebae_diagonal_loading,
        ),
    )
    associated = int(result.associated_beam_indices[0]) if result.signal_count > 0 else -1
    return (
        np.asarray(result.weights[:, beam_index], dtype=config.precision.complex_dtype),
        result.signal_count,
        associated,
    )


def _mirror(values: ComplexArray, fft_size: int) -> ComplexArray:
    full = np.empty((fft_size, values.shape[1], values.shape[2]), dtype=values.dtype)
    full[: fft_size // 2 + 1] = values
    # DC/Nyquistを除いてW[-k]=conj(W[k])とし、full DFTの正負周波数を対応させる。
    full[fft_size // 2 + 1 :] = values[1:-1][::-1].conj()
    return full


def _mirror_int(values: IntArray, fft_size: int) -> IntArray:
    full = np.empty((fft_size, values.shape[1]), dtype=np.int64)
    full[: fft_size // 2 + 1] = values
    full[fft_size // 2 + 1 :] = values[1:-1][::-1]
    return full


def design_alignment_weights(config: AlignmentSimulationConfig) -> AlignmentWeightDesign:
    """明示された一つの条件からEBAE/MVDRのS1・S2a・T1・T2a重みを設計する。

    Args:
        config: sampling、ULA位置、方位、source帯域、noise/loadingを含む設計条件。

    Returns:
        full DFT完成重みと座標変換、steering、EBAE診断量。

    Raises:
        ValueError: source帯域に正周波数DFT binが一つも含まれない場合。
        numpy.linalg.LinAlgError: loading後共分散を解けない場合。
    """
    n_fft = config.fft_size
    n_beam = config.beam_azimuth_deg.size
    n_channel = config.sensor_positions_m.size
    positive_hz = np.asarray(
        np.fft.rfftfreq(n_fft, d=1.0 / config.fs_hz), dtype=config.precision.real_dtype
    )
    full_hz = np.asarray(
        np.fft.fftfreq(n_fft, d=1.0 / config.fs_hz), dtype=config.precision.real_dtype
    )
    beam_delays_s = calculate_ula_arrival_delays_s(
        config.sensor_positions_m, config.beam_azimuth_deg, config.sound_speed_m_per_s
    )
    source_delays_s = calculate_ula_arrival_delays_s(
        config.sensor_positions_m,
        np.asarray([config.target_azimuth_deg], dtype=config.precision.real_dtype),
        config.sound_speed_m_per_s,
    )[0]
    positive_steering = _steering(beam_delays_s, positive_hz)
    positive_source = _steering(source_delays_s[None, :], positive_hz)[:, 0, :]
    full_steering = _steering(beam_delays_s, full_hz)
    full_source = _steering(source_delays_s[None, :], full_hz)[:, 0, :]
    positive_mask = (positive_hz >= config.target_band_hz[0]) & (
        positive_hz <= config.target_band_hz[1]
    )
    source_bin_count = int(np.count_nonzero(positive_mask))
    if source_bin_count == 0:
        raise ValueError("target_band_hz must contain at least one non-negative DFT bin.")
    full_mask = (np.abs(full_hz) >= config.target_band_hz[0]) & (
        np.abs(full_hz) <= config.target_band_hz[1]
    )
    power_per_bin = config.source_band_rms_power / float(source_bin_count)

    # 待受方位の物理遅延を最寄りsampleへ丸め、残差座標への位相回転Dを作る。
    integer_delays = np.rint(beam_delays_s * config.fs_hz).astype(np.int64)
    positive_phase = np.exp(
        1j * 2.0 * np.pi * positive_hz[:, None, None] * integer_delays[None, :, :] / config.fs_hz
    )
    full_phase = np.exp(
        1j * 2.0 * np.pi * full_hz[:, None, None] * integer_delays[None, :, :] / config.fs_hz
    )
    positive_weights = {
        algorithm: {
            method: np.empty(
                (positive_hz.size, n_beam, n_channel), dtype=config.precision.complex_dtype
            )
            for method in ALIGNMENT_METHOD_IDS
        }
        for algorithm in ALIGNMENT_ALGORITHM_IDS
    }
    counts = {
        method: np.zeros((positive_hz.size, n_beam), dtype=np.int64)
        for method in ALIGNMENT_METHOD_IDS
    }
    associated = {
        method: np.full((positive_hz.size, n_beam), -1, dtype=np.int64)
        for method in ALIGNMENT_METHOD_IDS
    }
    for frequency_index in range(positive_hz.size):
        source_power = power_per_bin if positive_mask[frequency_index] else 0.0
        s_covariance = _source_covariance(
            config, source_delays_s, positive_source[frequency_index], None, source_power
        )
        steering_scan = np.asarray(
            positive_steering[frequency_index].T, dtype=config.precision.complex_dtype
        )
        for beam_index in range(n_beam):
            constraint = positive_steering[frequency_index, beam_index]
            phase = positive_phase[frequency_index, beam_index]
            rotated_scan = np.asarray(
                phase[:, None] * steering_scan, dtype=config.precision.complex_dtype
            )
            rotated_constraint = np.asarray(
                phase * constraint, dtype=config.precision.complex_dtype
            )
            if source_power == 0.0:
                # 白色雑音だけのbinでは両方式がCBFへ帰着するため、同値な完成重みを直接置く。
                original_cbf = np.asarray(
                    constraint / np.vdot(constraint, constraint),
                    dtype=config.precision.complex_dtype,
                )
                residual_cbf = np.asarray(
                    rotated_constraint / np.vdot(rotated_constraint, rotated_constraint),
                    dtype=config.precision.complex_dtype,
                )
                for algorithm in ALIGNMENT_ALGORITHM_IDS:
                    for method in ("S1", "T1"):
                        positive_weights[algorithm][method][frequency_index, beam_index] = (
                            original_cbf
                        )
                    for method in ("S2a", "T2a"):
                        positive_weights[algorithm][method][frequency_index, beam_index] = (
                            residual_cbf
                        )
                continue
            s_residual = np.asarray(
                phase[:, None] * s_covariance * phase.conj()[None, :],
                dtype=config.precision.complex_dtype,
            )
            t_original = _source_covariance(
                config,
                source_delays_s,
                positive_source[frequency_index],
                beam_delays_s[beam_index],
                source_power,
            )
            t_residual = np.asarray(
                phase[:, None] * t_original * phase.conj()[None, :],
                dtype=config.precision.complex_dtype,
            )
            for method, covariance, scan, method_constraint in (
                ("S1", s_covariance, steering_scan, constraint),
                ("S2a", s_residual, rotated_scan, rotated_constraint),
                ("T1", t_original, steering_scan, constraint),
                ("T2a", t_residual, rotated_scan, rotated_constraint),
            ):
                positive_weights["mvdr"][method][frequency_index, beam_index] = _mvdr_weight(
                    config, covariance, method_constraint
                )
                weight, count, beam = _ebae_weight(config, covariance, scan, beam_index)
                positive_weights["ebae"][method][frequency_index, beam_index] = weight
                counts[method][frequency_index, beam_index] = count
                associated[method][frequency_index, beam_index] = beam
    weights = {
        algorithm: {
            method: _mirror(positive_weights[algorithm][method], n_fft)
            for method in ALIGNMENT_METHOD_IDS
        }
        for algorithm in ALIGNMENT_ALGORITHM_IDS
    }
    return AlignmentWeightDesign(
        config=config,
        weights=weights,
        integer_phase=np.asarray(full_phase, dtype=config.precision.complex_dtype),
        steering=full_steering,
        source_steering=full_source,
        source_bin_mask=np.asarray(full_mask, dtype=np.bool_),
        ebae_signal_counts={
            method: _mirror_int(counts[method], n_fft) for method in ALIGNMENT_METHOD_IDS
        },
        ebae_associated_beams={
            method: _mirror_int(associated[method], n_fft) for method in ALIGNMENT_METHOD_IDS
        },
    )


def approximate_frequency_weights_with_fir(
    weights: ComplexArray, tap_count: int
) -> FrequencyWeightFirApproximation:
    """beam内の全channelで共有するcircular窓により周波数重みをFIR近似する。

    Args:
        weights: full DFT重み。shape ``[n_fft,n_beam,n_ch]``。
        tap_count: 採用tap数、単位sample。

    Returns:
        再構成重み、beam別energy比、窓先頭sample。

    Raises:
        ValueError: weightsが3次元でない、軸が空、またはtap数が範囲外の場合。
    """
    checked = np.asarray(weights)
    if checked.ndim != 3 or 0 in checked.shape:
        raise ValueError("weights must have non-empty shape (n_fft, n_beam, n_ch).")
    if checked.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("weights dtype must be complex64 or complex128.")
    n_fft, n_beam, _ = checked.shape
    if not 0 < tap_count <= n_fft:
        raise ValueError("tap_count must be in [1, n_fft].")
    reconstructed = np.empty_like(checked)
    real_dtype = (
        np.dtype(np.float32) if checked.dtype == np.dtype(np.complex64) else np.dtype(np.float64)
    )
    energy_ratio = np.empty(n_beam, dtype=real_dtype)
    starts = np.empty(n_beam, dtype=np.int64)
    for beam_index in range(n_beam):
        # 実適用応答conj(w)をIFFTし、axis=0の時間で全channel合算energyを最大化する。
        impulse = np.asarray(
            np.fft.ifft(checked[:, beam_index, :].conj(), axis=0), dtype=checked.dtype
        )
        energy = np.sum(np.abs(impulse) ** 2, axis=1)
        total = float(np.sum(energy))
        extended = np.concatenate((energy, energy[: tap_count - 1]))
        window_energy = np.convolve(extended, np.ones(tap_count), mode="valid")[:n_fft]
        start = int(np.argmax(window_energy)) if total > 0.0 else 0
        starts[beam_index] = start
        energy_ratio[beam_index] = float(window_energy[start] / total) if total > 0.0 else 1.0
        keep = (start + np.arange(tap_count)) % n_fft
        truncated = np.zeros_like(impulse)
        truncated[keep, :] = impulse[keep, :]
        reconstructed[:, beam_index, :] = np.fft.fft(truncated, axis=0).conj()
    return FrequencyWeightFirApproximation(reconstructed, energy_ratio, starts)


def to_original_input_coordinates(
    method: str, weights: ComplexArray, integer_phase: ComplexArray
) -> ComplexArray:
    """残差座標のS2a/T2a重みを元入力座標の等価重みへ変換する。

    Args:
        method: ``S1``、``S2a``、``T1``、``T2a``のいずれか。
        weights: 当該方式座標の重み。shape ``[n_fft,n_beam,n_ch]``。
        integer_phase: 整数遅延位相D。weightsと同じshape。

    Returns:
        元入力座標の重み。shapeは入力と同じ。

    Raises:
        ValueError: methodまたはshapeが不正な場合。
    """
    if method not in ALIGNMENT_METHOD_IDS:
        raise ValueError(f"unknown alignment method: {method}")
    checked = np.asarray(weights)
    phase = np.asarray(integer_phase)
    if checked.shape != phase.shape or checked.ndim != 3:
        raise ValueError("weights and integer_phase must have the same 3-D shape.")
    if method in ("S2a", "T2a"):
        # y=v^H D xなので、元入力座標の等価weightはD^H v=conj(D)*vである。
        return np.asarray(phase.conj() * checked, dtype=checked.dtype)
    return checked.copy()


def calculate_source_beam_level_db(
    weights_original: ComplexArray,
    design: AlignmentWeightDesign,
    *,
    floor_db_re_input_rms: float = -100.0,
) -> FloatArray:
    """設計source帯域を積分したtarget-only beam levelを計算する。

    Args:
        weights_original: 元入力座標の重み。shape ``[n_fft,n_beam,n_ch]``。
        design: source steering、帯域mask、shape契約を含む完成設計。
        floor_db_re_input_rms: 対数表示のpower床、単位dB re input RMS。

    Returns:
        待受beam別level。shape ``[n_beam]``、単位dB re input RMS。

    Raises:
        ValueError: 重みshapeが設計shapeと一致しない、または表示床が非有限の場合。
    """
    checked = np.asarray(weights_original)
    if checked.shape != design.steering.shape:
        raise ValueError("weights_original shape must match design steering shape.")
    if not np.isfinite(floor_db_re_input_rms):
        raise ValueError("floor_db_re_input_rms must be finite.")
    band_weights = checked[design.source_bin_mask]
    # response[f,beam]=w[f,beam]^H a_source[f]。channel軸だけを内積として畳み込む。
    response = np.einsum(
        "fbc,fc->fb",
        band_weights.conj(),
        design.source_steering[design.source_bin_mask],
        optimize=True,
    )
    # 正負source binに等powerを置く契約では、bin平均powerが帯域積分応答に対応する。
    power = np.mean(np.abs(response) ** 2, axis=0)
    floor_power = 10.0 ** (floor_db_re_input_rms / 10.0)
    real_dtype = (
        np.dtype(np.float32) if checked.dtype == np.dtype(np.complex64) else np.dtype(np.float64)
    )
    return np.asarray(10.0 * np.log10(np.maximum(power, floor_power)), dtype=real_dtype)
