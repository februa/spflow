"""低周波の狭帯域・広帯域について、各bin独立共分散でFIR tap依存を評価する。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from evaluations.beamforming import ebae_mvdr_s1_s2a_t1_t2a_fir_sweep as engine


OUTPUT_DIR = Path("artifacts/beamforming/low_frequency_narrow_broad_tap_sweep/review_pack")
TAP_COUNTS = (32, 128, 256, 512, 1024)
LONG_ARRAY_N_CHANNEL = 64
LONG_ARRAY_SPACING_M = 6.25
BAND_CASES = {
    "narrow_64Hz": (64.0, 64.0),
    "narrow_150Hz": (150.0, 150.0),
    "broad_64_256Hz": (64.0, 256.0),
}
FloatArray = NDArray[np.float64]


def predict_grating_lobe_azimuths(
    source_azimuth_deg: float, frequencies_hz: FloatArray
) -> FloatArray:
    """ULA配置だけから可視領域内のグレーティングローブ方位を予測する。

    Args:
        source_azimuth_deg: 信号方位。単位はdeg、範囲は0--180 deg。
        frequencies_hz: 評価周波数。shapeは``[n_frequency]``、単位はHz。

    Returns:
        理論グレーティングローブ方位。shapeは``[n_grating]``、単位はdeg。

    Raises:
        ValueError: 方位または周波数が物理範囲外の場合。
    """
    if not 0.0 <= source_azimuth_deg <= 180.0:
        raise ValueError("source_azimuth_deg must be in [0, 180].")
    if frequencies_hz.ndim != 1 or bool(np.any(frequencies_hz <= 0.0)):
        raise ValueError("frequencies_hz must be a positive one-dimensional array.")
    source_cosine = float(np.cos(np.deg2rad(source_azimuth_deg)))
    predicted: list[float] = []
    for frequency_hz in frequencies_hz:
        wavelength_m = engine.SOUND_SPEED_M_S / float(frequency_hz)
        # d(cos(theta_g)-cos(theta_0))=m lambdaを満たす方位は、センサ間位相が2πmだけ
        # 異なるためサンプル済みアレイから識別できない。m=±1から可視領域だけを列挙する。
        for order in (-1, 1):
            candidate_cosine = source_cosine + order * wavelength_m / LONG_ARRAY_SPACING_M
            if -1.0 <= candidate_cosine <= 1.0:
                predicted.append(float(np.rad2deg(np.arccos(candidate_cosine))))
    if not predicted:
        return np.empty(0, dtype=np.float64)
    return np.asarray(sorted(set(np.round(predicted, decimals=9))), dtype=np.float64)


def _peak_class(peak_azimuth_deg: float, grating_azimuths_deg: FloatArray) -> str:
    """最大ピークを信号近傍、理論グレーティング、その他に分類する。"""
    # 待受beam間隔10 degのため、信号方位の隣接beamまでは主ローブ近傍として区別する。
    if abs(peak_azimuth_deg - 150.0) <= 10.0:
        return "target_mainlobe_neighborhood"
    if grating_azimuths_deg.size > 0 and float(
        np.min(np.abs(grating_azimuths_deg - peak_azimuth_deg))
    ) <= 5.0:
        return "predicted_grating_lobe"
    return "fir_or_other_artifact"


def _calculate_case(
    case_id: str, band_hz: tuple[float, float]
) -> tuple[list[dict[str, Any]], dict[str, dict[int, FloatArray]]]:
    """1帯域の全方式・全tapについてBLと数値指標を計算する。

    Args:
        case_id: 成果物上の条件識別子。
        band_hz: 正周波数の占有帯域端。単位はHz。

    Returns:
        指標行とBL曲線。各BLのshapeは``[n_beam]``、単位はdB re input RMS。
    """
    engine.FFT_SIZE = 4096
    # 共分散検討で使用した64ch・6.25 m間隔・393.75 m開口の長大ULAへ合わせる。
    # fsは150 Hzを2 Hz刻みのbin中心に保ち、off-bin誤差を混ぜないため8192 Hzを維持する。
    engine.FS_HZ = 8192.0
    engine.N_CHANNEL = LONG_ARRAY_N_CHANNEL
    engine.SPACING_M = LONG_ARRAY_SPACING_M
    # 各DFT binを独立したrank-1信号共分散として扱い、共分散内の周波数積分は行わない。
    engine.ANALYSIS_WIDTH_HZ = 0.0
    engine.TARGET_AZIMUTH_DEG = 150.0
    engine.TARGET_BAND_HZ = band_hz
    engine.SOURCE_BAND_RMS_POWER = 1.0
    engine.NOISE_POWER_PER_BIN_RE_INPUT_RMS2 = 1.0e-2
    engine.SCENARIO_ID = case_id

    design = engine.design_reference_weights()
    positive_frequency_hz = np.fft.rfftfreq(engine.FFT_SIZE, d=1.0 / engine.FS_HZ)
    occupied_frequency_hz = positive_frequency_hz[
        (positive_frequency_hz >= band_hz[0]) & (positive_frequency_hz <= band_hz[1])
    ]
    grating_azimuths_deg = predict_grating_lobe_azimuths(150.0, occupied_frequency_hz)
    target_index = int(np.argmin(np.abs(engine.AZIMUTH_DEG - engine.TARGET_AZIMUTH_DEG)))
    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[int, FloatArray]] = {}
    for algorithm in engine.ALGORITHM_IDS:
        for method in engine.METHOD_IDS:
            key = f"{algorithm}_{method}"
            curves[key] = {}
            reference = engine._original_coordinate_weights(
                method, design.weights[algorithm][method], design.integer_phase
            )
            reference_bl = engine._band_bl_db(reference, design)
            for tap_count in TAP_COUNTS:
                approximation = engine.approximate_weights_with_fir(
                    design.weights[algorithm][method], tap_count
                )
                reconstructed = engine._original_coordinate_weights(
                    method, approximation.reconstructed_weights, design.integer_phase
                )
                bl_db = engine._band_bl_db(reconstructed, design)
                curves[key][tap_count] = bl_db
                peak_index = int(np.argmax(bl_db))
                rows.append(
                    {
                        "case": case_id,
                        "band_low_hz": band_hz[0],
                        "band_high_hz": band_hz[1],
                        "covariance_frequency_handling": "independent_fft_bins_no_band_integration",
                        "algorithm": algorithm,
                        "method": method,
                        "tap_count": tap_count,
                        "target_level_db_re_input_rms": float(bl_db[target_index]),
                        "target_level_error_db": float(
                            bl_db[target_index] - reference_bl[target_index]
                        ),
                        "peak_azimuth_deg": float(engine.AZIMUTH_DEG[peak_index]),
                        "peak_error_deg": float(
                            abs(engine.AZIMUTH_DEG[peak_index] - engine.TARGET_AZIMUTH_DEG)
                        ),
                        "peak_class": _peak_class(
                            float(engine.AZIMUTH_DEG[peak_index]), grating_azimuths_deg
                        ),
                        "predicted_grating_low_deg": (
                            float(np.min(grating_azimuths_deg))
                            if grating_azimuths_deg.size > 0
                            else ""
                        ),
                        "predicted_grating_high_deg": (
                            float(np.max(grating_azimuths_deg))
                            if grating_azimuths_deg.size > 0
                            else ""
                        ),
                        "bl_rms_error_db": float(
                            np.sqrt(np.mean((bl_db - reference_bl) ** 2))
                        ),
                        "target_energy_ratio": float(approximation.energy_ratio[target_index]),
                    }
                )
    return rows, curves


def _write_bl_figure(
    case_id: str,
    band_hz: tuple[float, float],
    curves: dict[str, dict[int, FloatArray]],
    output_dir: Path,
) -> None:
    """同一表示条件のtarget-only BLを保存する。"""
    figure, axes = plt.subplots(
        2, len(TAP_COUNTS), figsize=(22.0, 8.0), sharex=True, sharey=True, constrained_layout=True
    )
    positive_frequency_hz = np.fft.rfftfreq(engine.FFT_SIZE, d=1.0 / engine.FS_HZ)
    occupied_frequency_hz = positive_frequency_hz[
        (positive_frequency_hz >= band_hz[0]) & (positive_frequency_hz <= band_hz[1])
    ]
    grating_azimuths_deg = predict_grating_lobe_azimuths(150.0, occupied_frequency_hz)
    for row_index, algorithm in enumerate(engine.ALGORITHM_IDS):
        for column_index, tap_count in enumerate(TAP_COUNTS):
            axis = axes[row_index, column_index]
            for method in engine.METHOD_IDS:
                # BL shapeは[n_beam]。全方式で同じ待受方位axisとdB基準を使う。
                axis.plot(
                    engine.AZIMUTH_DEG,
                    curves[f"{algorithm}_{method}"][tap_count],
                    marker="o",
                    label=method,
                )
            axis.axvline(150.0, color="black", linestyle="--", linewidth=1.0)
            if grating_azimuths_deg.size == 1:
                axis.axvline(
                    float(grating_azimuths_deg[0]),
                    color="darkorange",
                    linestyle=":",
                    linewidth=1.5,
                    label="predicted grating",
                )
            elif grating_azimuths_deg.size > 1:
                axis.axvspan(
                    float(np.min(grating_azimuths_deg)),
                    float(np.max(grating_azimuths_deg)),
                    color="darkorange",
                    alpha=0.12,
                    label="predicted grating range",
                )
            axis.set(
                title=f"{algorithm.upper()}, {tap_count} taps",
                xlabel="Waiting-beam azimuth [deg]",
                xlim=(0.0, 180.0),
                ylim=(-60.0, 5.0),
            )
            if column_index == 0:
                axis.set_ylabel("Band-integrated RMS level [dB re input RMS]")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
    figure.suptitle(f"{case_id}: independent covariance per FFT bin, source azimuth 150 deg")
    figure.savefig(output_dir / f"{case_id}_bl_tap_sweep.png", dpi=160)
    plt.close(figure)


def write_review_pack(output_dir: Path = OUTPUT_DIR) -> tuple[dict[str, Any], ...]:
    """CSVとBL図を保存する。

    Args:
        output_dir: 成果物の出力先。

    Returns:
        全条件の指標行。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for case_id, band_hz in BAND_CASES.items():
        rows, curves = _calculate_case(case_id, band_hz)
        all_rows.extend(rows)
        _write_bl_figure(case_id, band_hz, curves, output_dir)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    return tuple(all_rows)


def main() -> None:
    """既定の低周波tap sweepを実行する。"""
    write_review_pack()


if __name__ == "__main__":
    main()
