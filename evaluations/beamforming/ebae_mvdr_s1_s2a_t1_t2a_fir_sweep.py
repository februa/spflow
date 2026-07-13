"""EBAE/MVDRのS1・S2a・T1・T2aとFIR長依存を同一条件で評価する。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from spflow.simulation import (
    ALIGNMENT_ALGORITHM_IDS,
    ALIGNMENT_METHOD_IDS,
    AlignmentSimulationConfig,
    AlignmentWeightDesign,
    FrequencyWeightFirApproximation,
    approximate_frequency_weights_with_fir,
    calculate_source_beam_level_db,
    design_alignment_weights,
    to_original_input_coordinates,
)

FloatArray = NDArray[np.floating[Any]]
ComplexArray = NDArray[np.complexfloating[Any, Any]]


OUTPUT_DIR = Path("artifacts/beamforming/ebae_mvdr_s1_s2a_t1_t2a_fir_sweep/review_pack")
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
METHOD_IDS = ("S1", "S2a", "T1", "T2a")
DISPLAY_FLOOR_DB_RE_INPUT_RMS = -100.0

# scenario ID、tap sweep、表示床は評価条件であり、公開シミュレーション部品へ含めない。
DEFAULT_ALIGNMENT_CONFIG = AlignmentSimulationConfig(
    fs_hz=FS_HZ,
    fft_size=FFT_SIZE,
    sound_speed_m_per_s=SOUND_SPEED_M_S,
    sensor_positions_m=np.linspace(
        -SPACING_M * (N_CHANNEL - 1) / 2.0,
        SPACING_M * (N_CHANNEL - 1) / 2.0,
        N_CHANNEL,
        dtype=np.float64,
    ),
    beam_azimuth_deg=AZIMUTH_DEG,
    target_azimuth_deg=TARGET_AZIMUTH_DEG,
    target_band_hz=TARGET_BAND_HZ,
    analysis_width_hz=ANALYSIS_WIDTH_HZ,
    source_band_rms_power=SOURCE_BAND_RMS_POWER,
    noise_power_per_bin_re_input_rms2=NOISE_POWER_PER_BIN_RE_INPUT_RMS2,
    ebae_diagonal_loading=EBAE_DIAGONAL_LOADING,
    mvdr_diagonal_loading_ratio=MVDR_DIAGONAL_LOADING_RATIO,
)

if ALGORITHM_IDS != ALIGNMENT_ALGORITHM_IDS or METHOD_IDS != ALIGNMENT_METHOD_IDS:
    raise RuntimeError("evaluation identifiers must match the public simulation contract.")


WeightDesignResult = AlignmentWeightDesign
FirApproximationResult = FrequencyWeightFirApproximation


def design_reference_weights() -> WeightDesignResult:
    """既定評価条件から完成重みを設計する。"""
    return design_alignment_weights(DEFAULT_ALIGNMENT_CONFIG)


def approximate_weights_with_fir(weights: ComplexArray, tap_count: int) -> FirApproximationResult:
    """公開シミュレーション部品で周波数重みを有限FIR化する。"""
    return approximate_frequency_weights_with_fir(weights, tap_count)


def _original_coordinate_weights(
    method: str, weights: ComplexArray, integer_phase: ComplexArray
) -> ComplexArray:
    """公開シミュレーション部品で元入力座標へ変換する。"""
    return to_original_input_coordinates(method, weights, integer_phase)


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
    target_level_delta = float(approximated_bl[target_beam_index] - reference_bl[target_beam_index])
    peak_index = int(np.argmax(approximated_bl))
    guard_mask = np.abs(AZIMUTH_DEG - TARGET_AZIMUTH_DEG) > 10.0
    return {
        "scenario": SCENARIO_ID,
        "algorithm": algorithm,
        "method": method,
        "tap_count": tap_count,
        "coordinate": "integer_delay_residual" if method in ("S2a", "T2a") else "original_input",
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
    return np.asarray(
        calculate_source_beam_level_db(
            weights_original,
            design,
            floor_db_re_input_rms=DISPLAY_FLOOR_DB_RE_INPUT_RMS,
        ),
        dtype=np.float64,
    )


def calculate_fir_sweep() -> tuple[
    WeightDesignResult, tuple[dict[str, Any], ...], dict[str, FloatArray]
]:
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
        s1_original = design.weights[algorithm]["S1"]
        s2_original = _original_coordinate_weights(
            "S2a", design.weights[algorithm]["S2a"], design.integer_phase
        )
        t1_original = design.weights[algorithm]["T1"]
        t2_original = _original_coordinate_weights(
            "T2a", design.weights[algorithm]["T2a"], design.integer_phase
        )
        arrays[f"{algorithm}_s1_s2a_relative_error"] = np.asarray(
            [np.linalg.norm(s1_original - s2_original) / np.linalg.norm(s1_original)],
            dtype=np.float64,
        )
        arrays[f"{algorithm}_t1_t2a_relative_error"] = np.asarray(
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
    """S1/S2a/T1/T2a FIR長sweepのCSV、NPZ、図、レビュー索引を保存する。

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
        "# EBAE/MVDR S1・S2a・T1・T2a FIR長sweep",
        "",
        f"- scenario: `{SCENARIO_ID}`",
        "- evaluation pattern: `fixed_beam_single_source`",
        f"- array: {N_CHANNEL} ch ULA, spacing {SPACING_M:.1f} m",
        f"- fs / FFT / analysis width: {FS_HZ:.0f} Hz / {FFT_SIZE} / {ANALYSIS_WIDTH_HZ:.1f} Hz",
        f"- target: {TARGET_AZIMUTH_DEG:.1f} deg, "
        f"{TARGET_BAND_HZ[0]:.0f}--{TARGET_BAND_HZ[1]:.0f} Hz",
        f"- tap counts: {', '.join(str(value) for value in FIR_TAP_COUNTS)}",
        "",
        "S2aはS1共分散の整数遅延位相変換、T2aはT1共分散の整数遅延位相変換である。",
        "S1/T1は元入力座標でFIR化し、S2a/T2aは整数delay line後の残留座標でFIR化する。",
        "full DFT重みを基準とし、全channel共通のcircular tap窓で打ち切って再構成する。",
        "",
    ]
    for algorithm in ALGORITHM_IDS:
        review_lines.extend(
            (
                f"- {algorithm.upper()} S1/S2a relative error: "
                f"{float(arrays[f'{algorithm}_s1_s2a_relative_error'][0]):.3e}",
                f"- {algorithm.upper()} T1/T2a relative error: "
                f"{float(arrays[f'{algorithm}_t1_t2a_relative_error'][0]):.3e}",
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
