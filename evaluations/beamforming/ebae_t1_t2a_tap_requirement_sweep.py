"""T共分散の方位推定とT1/T2aの有限長FIR実現誤差を分離して評価する。"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evaluations.beamforming import ebae_mvdr_s1_s2a_t1_t2a_fir_sweep as reference
from spflow.simulation import (
    AlignmentSimulationConfig,
    AlignmentWeightDesign,
    FrequencyWeightFirApproximation,
    approximate_frequency_weights_with_fir,
    calculate_ula_arrival_delays_s,
    design_alignment_weights,
    to_original_input_coordinates,
)

FloatArray = NDArray[np.floating[Any]]
ComplexArray = NDArray[np.complexfloating[Any, Any]]

OUTPUT_DIR = Path("artifacts/beamforming/ebae_t1_t2a_tap_requirement_sweep/review_pack")
TAP_COUNTS = (8, 16, 32, 64, 128, 256, 512)
METHOD_IDS = ("T1", "T2a")

# 合否値は、方位gridの離散化とFIR打切り誤差を方式差から分離できる水準に固定する。
PEAK_ERROR_LIMIT_DEG = 10.0
ENERGY_RATIO_MIN = 0.99
WEIGHT_ERROR_MAX = 0.05
DISTORTIONLESS_LEVEL_ERROR_MAX_DB = 0.2
PHASE_RMS_ERROR_MAX_DEG = 5.0
GROUP_DELAY_RMS_ERROR_MAX_SAMPLE = 0.5
WAVEFORM_CORRELATION_MIN = 0.995
WAVEFORM_NORMALIZED_RMS_ERROR_MAX = 0.10
MAINLOBE_WIDTH_RELATIVE_ERROR_MAX = 0.20
SIDELOBE_DEGRADATION_MAX_DB = 2.0
NULL_DEGRADATION_MAX_DB = 3.0


@dataclass(frozen=True)
class TapScenario:
    """T1/T2a tap評価の物理条件を保持する。

    このクラスはアレイ開口、方位、bin内分析幅、使用帯域を定義する。
    共分散計算やFIR変換そのものは責務に含めない。

    Attributes:
        scenario_id: 成果物内で一意な条件名。
        aperture_m: ULAの端点間開口、単位はm。
        target_azimuth_deg: 到来方位、単位はdeg。0 degがendfire、90 degがbroadside。
        analysis_width_hz: T共分散のbin内積分幅、単位はHz。0はbin中心tone。
        occupied_band_hz: 信号が占有するDFT帯域端、単位はHz。
        spectrum_kind: ``bin_center_tone``、``flat_broadband``、``operational_taper``の別。
    """

    scenario_id: str
    aperture_m: float
    target_azimuth_deg: float
    analysis_width_hz: float
    occupied_band_hz: tuple[float, float]
    spectrum_kind: str


SCENARIOS = (
    TapScenario("small_broadside_tone", 7.5, 90.0, 0.0, (256.0, 256.0), "bin_center_tone"),
    TapScenario("large_broadside_tone", 42.0, 90.0, 0.0, (256.0, 256.0), "bin_center_tone"),
    TapScenario("large_endfire_tone", 42.0, 0.0, 0.0, (256.0, 256.0), "bin_center_tone"),
    TapScenario("medium_oblique_flat", 21.0, 60.0, 16.0, (128.0, 384.0), "flat_broadband"),
    TapScenario("large_endfire_flat", 42.0, 0.0, 64.0, (128.0, 384.0), "flat_broadband"),
    TapScenario("large_oblique_operational", 42.0, 60.0, 32.0, (64.0, 512.0), "operational_taper"),
)


def _simulation_config(scenario: TapScenario) -> AlignmentSimulationConfig:
    """一つのtap評価条件を不変シミュレーション設定へ変換する。

    Args:
        scenario: 適用する物理条件。

    Returns:
        共分散・符号・共役・FFT規約を公開部品と共有する明示設定。
    """
    return replace(
        reference.DEFAULT_ALIGNMENT_CONFIG,
        sensor_positions_m=np.linspace(
            -scenario.aperture_m / 2.0,
            scenario.aperture_m / 2.0,
            reference.N_CHANNEL,
            dtype=np.float64,
        ),
        target_azimuth_deg=scenario.target_azimuth_deg,
        analysis_width_hz=scenario.analysis_width_hz,
        target_band_hz=scenario.occupied_band_hz,
    )


def _response(weights: ComplexArray, design: AlignmentWeightDesign) -> ComplexArray:
    """信号占有binにおける全beamの複素応答 ``w^H a`` を返す。

    Args:
        weights: 元入力座標重み。shapeは``[n_fft,n_beam,n_ch]``。
        design: source steeringと占有bin maskを持つ設計結果。

    Returns:
        複素応答。shapeは``[n_occupied_bin,n_beam]``。
    """
    band = design.source_bin_mask
    return np.asarray(
        np.einsum("fbc,fc->fb", weights[band].conj(), design.source_steering[band], optimize=True),
        dtype=np.complex128,
    )


def _spectrum_amplitude(scenario: TapScenario, n_bin: int) -> FloatArray:
    """波形評価に使う決定論的な信号振幅を返す。

    Args:
        scenario: spectrum種別を含む条件。
        n_bin: 正負を含む占有bin数。

    Returns:
        非負振幅。shapeは``[n_bin]``、線形値。
    """
    if n_bin <= 0:
        raise ValueError("n_bin must be positive.")
    if scenario.spectrum_kind in ("bin_center_tone", "flat_broadband"):
        return np.ones(n_bin, dtype=np.float64)
    if scenario.spectrum_kind == "operational_taper":
        # 帯域端を滑らかにするHann形状で、実運用信号に近い端点減衰を与える。
        return np.asarray(0.1 + 0.9 * np.hanning(n_bin), dtype=np.float64)
    raise ValueError(f"unsupported spectrum_kind: {scenario.spectrum_kind}")


def _bl_features(
    response: ComplexArray, beam_azimuth_deg: FloatArray
) -> tuple[float, float, float, float]:
    """BLからpeak方位、-3 dB幅、guard外peak、null床を返す。

    Args:
        response: 複素応答。shapeは``[n_bin,n_beam]``。

    Returns:
        ``(peak_deg, width_deg, sidelobe_db, null_db)``。
        dB値は当該BL peak基準である。
    """
    power = np.mean(np.abs(response) ** 2, axis=0)
    peak_index = int(np.argmax(power))
    peak_power = max(float(power[peak_index]), np.finfo(np.float64).tiny)
    raw_relative_db = 10.0 * np.log10(np.maximum(power / peak_power, np.finfo(np.float64).tiny))
    # 深いnullの数値床は微小な丸め誤差で数十dB変化するため、形状比較範囲を-60 dBまでに限定する。
    relative_db = np.maximum(raw_relative_db, -60.0)
    main_mask = relative_db >= -3.0
    main_indices = np.flatnonzero(main_mask)
    width_deg = float(beam_azimuth_deg[main_indices[-1]] - beam_azimuth_deg[main_indices[0]])
    guard = np.abs(beam_azimuth_deg - beam_azimuth_deg[peak_index]) > 20.0
    if not bool(np.any(guard)):
        raise ValueError("azimuth grid must contain guard-outside beams.")
    return (
        float(beam_azimuth_deg[peak_index]),
        width_deg,
        float(np.max(relative_db[guard])),
        float(np.min(relative_db[guard])),
    )


def _error_metrics(
    scenario: TapScenario,
    method: str,
    tap_count: int,
    design: AlignmentWeightDesign,
    approximation: FrequencyWeightFirApproximation,
) -> dict[str, Any]:
    """完成重みを基準に有限長FIR実現誤差と合否を計算する。"""
    config = design.config
    target_index = int(np.argmin(np.abs(config.beam_azimuth_deg - scenario.target_azimuth_deg)))
    full = to_original_input_coordinates(
        method, design.weights["ebae"][method], design.integer_phase
    )
    finite = to_original_input_coordinates(
        method, approximation.reconstructed_weights, design.integer_phase
    )
    band = design.source_bin_mask
    full_response = _response(full, design)
    finite_response = _response(finite, design)
    full_target = full_response[:, target_index]
    finite_target = finite_response[:, target_index]
    # 待受方位ごとに別FIRを実装する契約なので、重み誤差も対象beamのchannel重みだけで評価する。
    full_target_weights = full[band, target_index, :]
    finite_target_weights = finite[band, target_index, :]
    weight_denominator = max(float(np.linalg.norm(full_target_weights)), np.finfo(np.float64).tiny)
    weight_error = float(
        np.linalg.norm(finite_target_weights - full_target_weights) / weight_denominator
    )

    ratio = finite_target / np.where(np.abs(full_target) > 1.0e-12, full_target, 1.0 + 0.0j)
    level_error_db = float(np.max(np.abs(20.0 * np.log10(np.maximum(np.abs(ratio), 1.0e-12)))))
    phase_error_rad = np.unwrap(np.angle(ratio))
    phase_rms_deg = float(np.rad2deg(np.sqrt(np.mean(phase_error_rad**2))))
    occupied_frequencies_hz = np.fft.fftfreq(config.fft_size, d=1.0 / config.fs_hz)[band]
    if occupied_frequencies_hz.size >= 2 and float(np.ptp(occupied_frequencies_hz)) > 0.0:
        order = np.argsort(occupied_frequencies_hz)
        phase_sorted = np.unwrap(np.angle(ratio[order]))
        frequency_sorted = occupied_frequencies_hz[order]
        group_delay_s = -np.gradient(phase_sorted, 2.0 * np.pi * frequency_sorted)
        group_delay_rms_sample = float(
            config.fs_hz * np.sqrt(np.mean((group_delay_s - np.mean(group_delay_s)) ** 2))
        )
    else:
        # 単一toneでは群遅延を観測できないため、位相誤差だけを判定し群遅延誤差は0とする。
        group_delay_rms_sample = 0.0

    amplitude = _spectrum_amplitude(scenario, int(np.count_nonzero(band)))
    reference_spectrum = np.zeros(config.fft_size, dtype=np.complex128)
    finite_spectrum = np.zeros(config.fft_size, dtype=np.complex128)
    reference_spectrum[band] = amplitude * full_target
    finite_spectrum[band] = amplitude * finite_target
    reference_waveform = np.real(np.fft.ifft(reference_spectrum))
    finite_waveform = np.real(np.fft.ifft(finite_spectrum))
    waveform_denominator = max(float(np.linalg.norm(reference_waveform)), np.finfo(np.float64).tiny)
    normalized_rms_error = float(
        np.linalg.norm(finite_waveform - reference_waveform) / waveform_denominator
    )
    correlation = float(np.corrcoef(reference_waveform, finite_waveform)[0, 1])

    full_peak, full_width, full_sidelobe, full_null = _bl_features(
        full_response, config.beam_azimuth_deg
    )
    finite_peak, finite_width, finite_sidelobe, finite_null = _bl_features(
        finite_response, config.beam_azimuth_deg
    )
    width_relative_error = abs(finite_width - full_width) / max(full_width, 10.0)
    peak_error_deg = abs(finite_peak - scenario.target_azimuth_deg)
    fir_pass = bool(
        approximation.energy_ratio[target_index] >= ENERGY_RATIO_MIN
        and weight_error <= WEIGHT_ERROR_MAX
        and level_error_db <= DISTORTIONLESS_LEVEL_ERROR_MAX_DB
        and phase_rms_deg <= PHASE_RMS_ERROR_MAX_DEG
        and group_delay_rms_sample <= GROUP_DELAY_RMS_ERROR_MAX_SAMPLE
        and correlation >= WAVEFORM_CORRELATION_MIN
        and normalized_rms_error <= WAVEFORM_NORMALIZED_RMS_ERROR_MAX
        and peak_error_deg <= PEAK_ERROR_LIMIT_DEG
        and width_relative_error <= MAINLOBE_WIDTH_RELATIVE_ERROR_MAX
        and finite_sidelobe - full_sidelobe <= SIDELOBE_DEGRADATION_MAX_DB
        and finite_null - full_null <= NULL_DEGRADATION_MAX_DB
    )
    return {
        "scenario": scenario.scenario_id,
        "algorithm": "ebae",
        "method": method,
        "tap_count": tap_count,
        "aperture_m": scenario.aperture_m,
        "target_azimuth_deg": scenario.target_azimuth_deg,
        "analysis_width_hz": scenario.analysis_width_hz,
        "occupied_band_low_hz": scenario.occupied_band_hz[0],
        "occupied_band_high_hz": scenario.occupied_band_hz[1],
        "spectrum_kind": scenario.spectrum_kind,
        "common_window_start_sample": int(approximation.window_start_samples[target_index]),
        "target_energy_ratio": float(approximation.energy_ratio[target_index]),
        "relative_weight_error": weight_error,
        "distortionless_level_error_db": level_error_db,
        "phase_rms_error_deg": phase_rms_deg,
        "group_delay_rms_error_sample": group_delay_rms_sample,
        "waveform_correlation": correlation,
        "waveform_normalized_rms_error": normalized_rms_error,
        "full_peak_deg": full_peak,
        "finite_peak_deg": finite_peak,
        "peak_error_deg": peak_error_deg,
        "full_mainlobe_width_deg": full_width,
        "finite_mainlobe_width_deg": finite_width,
        "mainlobe_width_relative_error": width_relative_error,
        "sidelobe_degradation_db": finite_sidelobe - full_sidelobe,
        "null_degradation_db": finite_null - full_null,
        "fir_realization_pass": fir_pass,
    }


def calculate_tap_requirement_sweep() -> tuple[
    tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]
]:
    """全条件でT共分散成立性とT1/T2aの最短tap数を計算する。

    Returns:
        ``(detail_rows, minimum_rows)``。前者は全tap、後者は条件・方式別の最短合格tapを持つ。
    """
    detail_rows: list[dict[str, Any]] = []
    minimum_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        config = _simulation_config(scenario)
        design = design_alignment_weights(config)
        target_index = int(np.argmin(np.abs(config.beam_azimuth_deg - scenario.target_azimuth_deg)))
        delays_s = calculate_ula_arrival_delays_s(
            config.sensor_positions_m,
            np.asarray([scenario.target_azimuth_deg], dtype=np.float64),
            config.sound_speed_m_per_s,
        )[0]
        geometric_delay_span_sample = config.fs_hz * float(np.ptp(delays_s))
        residual_s = delays_s - np.rint(delays_s * config.fs_hz) / config.fs_hz
        residual_delay_span_sample = config.fs_hz * float(np.ptp(residual_s))
        for method in METHOD_IDS:
            full = to_original_input_coordinates(
                method, design.weights["ebae"][method], design.integer_phase
            )
            full_response = _response(full, design)
            full_peak, _, _, _ = _bl_features(full_response, config.beam_azimuth_deg)
            signal_counts = design.ebae_signal_counts[method][design.source_bin_mask, target_index]
            associated = design.ebae_associated_beams[method][design.source_bin_mask, target_index]
            direction_pass = bool(
                abs(full_peak - scenario.target_azimuth_deg) <= PEAK_ERROR_LIMIT_DEG
                and bool(np.all(signal_counts == 1))
                and bool(
                    np.all(
                        np.abs(config.beam_azimuth_deg[associated] - scenario.target_azimuth_deg)
                        <= PEAK_ERROR_LIMIT_DEG
                    )
                )
            )
            method_rows: list[dict[str, Any]] = []
            for tap_count in TAP_COUNTS:
                approximation = approximate_frequency_weights_with_fir(
                    design.weights["ebae"][method], tap_count
                )
                row = _error_metrics(scenario, method, tap_count, design, approximation)
                row["direction_estimation_pass"] = direction_pass
                row["overall_pass"] = bool(direction_pass and row["fir_realization_pass"])
                row["geometric_delay_span_sample"] = geometric_delay_span_sample
                row["residual_delay_span_sample"] = residual_delay_span_sample
                method_rows.append(row)
                detail_rows.append(row)
            passing = [int(row["tap_count"]) for row in method_rows if bool(row["overall_pass"])]
            minimum_rows.append(
                {
                    "scenario": scenario.scenario_id,
                    "method": method,
                    "direction_estimation_pass": direction_pass,
                    "minimum_passing_tap": min(passing) if passing else "",
                    "geometric_delay_span_sample": geometric_delay_span_sample,
                    "residual_delay_span_sample": residual_delay_span_sample,
                    "analysis_width_hz": scenario.analysis_width_hz,
                    "occupied_band_width_hz": scenario.occupied_band_hz[1]
                    - scenario.occupied_band_hz[0],
                }
            )
    return tuple(detail_rows), tuple(minimum_rows)


def write_review_pack(output_dir: Path = OUTPUT_DIR) -> None:
    """再現可能なCSVと合否基準をreview packへ保存する。

    Args:
        output_dir: 出力先directory。
    """
    detail_rows, minimum_rows = calculate_tap_requirement_sweep()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("tap_sweep.csv", detail_rows), ("minimum_taps.csv", minimum_rows)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    criteria = (
        "# T1/T2a tap requirement evaluation\n\n"
        "A（T共分散の方位推定）とB（有限長FIR実現）を別々に判定する。"
        "幾何遅延幅はT1の有力な予測値であり、無条件の数学的下限とは扱わない。\n\n"
        f"- peak error <= {PEAK_ERROR_LIMIT_DEG} deg\n"
        f"- energy containment >= {ENERGY_RATIO_MIN}\n"
        f"- weight error <= {WEIGHT_ERROR_MAX}\n"
        f"- |w^H a| level error <= {DISTORTIONLESS_LEVEL_ERROR_MAX_DB} dB\n"
        f"- phase RMS <= {PHASE_RMS_ERROR_MAX_DEG} deg\n"
        f"- group-delay RMS <= {GROUP_DELAY_RMS_ERROR_MAX_SAMPLE} sample\n"
        f"- waveform correlation >= {WAVEFORM_CORRELATION_MIN}\n"
        f"- waveform normalized RMS error <= {WAVEFORM_NORMALIZED_RMS_ERROR_MAX}\n"
        f"- mainlobe width relative error <= {MAINLOBE_WIDTH_RELATIVE_ERROR_MAX}\n"
        f"- sidelobe degradation <= {SIDELOBE_DEGRADATION_MAX_DB} dB\n"
        f"- null degradation <= {NULL_DEGRADATION_MAX_DB} dB\n"
    )
    (output_dir / "README.md").write_text(criteria, encoding="utf-8")


if __name__ == "__main__":
    write_review_pack()
