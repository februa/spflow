"""全整相方式の単一信号level・SNR・tap直交試験を実行する。"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evaluations.beamforming import ebae_mvdr_s1_s2a_t1_t2a_fir_sweep as engine
from spflow.simulation import (
    AlignmentSimulationConfig,
    approximate_frequency_weights_with_fir,
    design_alignment_weights,
    to_original_input_coordinates,
)

OUTPUT_DIR = Path("artifacts/beamforming/alignment_single_source_validation_matrix/review_pack")
AZIMUTHS_DEG = (0.0, 90.0, 150.0)
SIGNAL_LEVELS_DB_RE_INPUT_RMS = (-20.0, 0.0, 20.0)
INPUT_SNRS_DB = (-10.0, 0.0, 10.0, 20.0)
TAP_COUNTS = (32, 128, 256)


@dataclass(frozen=True)
class BandCase:
    """単一信号試験の分析幅と占有帯域を定義する。

    Attributes:
        case_id: 成果物上の識別子。
        analysis_width_hz: 共分散bin内分析幅、単位はHz。0はbin中心tone。
        occupied_band_hz: 正周波数の占有帯域端、単位はHz。
    """

    case_id: str
    analysis_width_hz: float
    occupied_band_hz: tuple[float, float]


BAND_CASES = (
    BandCase("narrow_bin_center", 0.0, (96.0, 96.0)),
    BandCase("broadband_flat", 16.0, (64.0, 128.0)),
)


def _simulation_config(
    band_case: BandCase,
    azimuth_deg: float,
    signal_level_db: float,
    input_snr_db: float,
) -> tuple[AlignmentSimulationConfig, str]:
    """1試験条件の不変シミュレーション設定と成果物識別子を返す。"""
    signal_power = 10.0 ** (signal_level_db / 10.0)
    noise_band_power = signal_power / (10.0 ** (input_snr_db / 10.0))
    positive_frequency_hz = np.fft.rfftfreq(engine.FFT_SIZE, d=1.0 / engine.FS_HZ)
    occupied = (positive_frequency_hz >= band_case.occupied_band_hz[0]) & (
        positive_frequency_hz <= band_case.occupied_band_hz[1]
    )
    occupied_count = int(np.count_nonzero(occupied))
    if occupied_count <= 0:
        raise ValueError("occupied band must contain at least one positive-frequency bin.")
    scenario_id = f"{band_case.case_id}_az{azimuth_deg:g}_sl{signal_level_db:g}_snr{input_snr_db:g}"
    # 雑音値はbinごとのchannel powerなので、帯域積分値を占有bin数へ等配分する。
    config = replace(
        engine.DEFAULT_ALIGNMENT_CONFIG,
        analysis_width_hz=band_case.analysis_width_hz,
        target_band_hz=band_case.occupied_band_hz,
        target_azimuth_deg=azimuth_deg,
        source_band_rms_power=signal_power,
        noise_power_per_bin_re_input_rms2=noise_band_power / float(occupied_count),
    )
    return config, scenario_id


def _method_rows(
    band_case: BandCase,
    azimuth_deg: float,
    signal_level_db: float,
    input_snr_db: float,
) -> list[dict[str, Any]]:
    """1物理条件について全方式・全tapの成分power指標を返す。"""
    config, scenario_id = _simulation_config(band_case, azimuth_deg, signal_level_db, input_snr_db)
    design = design_alignment_weights(config)
    target_index = int(np.argmin(np.abs(config.beam_azimuth_deg - azimuth_deg)))
    source_power = 10.0 ** (signal_level_db / 10.0)
    noise_band_power = source_power / (10.0 ** (input_snr_db / 10.0))
    rows: list[dict[str, Any]] = []
    for algorithm in engine.ALGORITHM_IDS:
        for direct_method in engine.METHOD_IDS:
            for tap_count in TAP_COUNTS:
                approximation = approximate_frequency_weights_with_fir(
                    design.weights[algorithm][direct_method], tap_count
                )
                original = to_original_input_coordinates(
                    direct_method, approximation.reconstructed_weights, design.integer_phase
                )
                band_weights = original[design.source_bin_mask, target_index, :]
                band_steering = design.source_steering[design.source_bin_mask]
                response = np.einsum("fc,fc->f", band_weights.conj(), band_steering)
                target_power = source_power * float(np.mean(np.abs(response) ** 2))
                # channel間無相関かつ帯域内総powerがnoise_band_powerになる条件の理論出力。
                noise_power = noise_band_power * float(
                    np.mean(np.sum(np.abs(band_weights) ** 2, axis=1))
                )
                output_snr_db = 10.0 * np.log10(
                    max(target_power, np.finfo(np.float64).tiny)
                    / max(noise_power, np.finfo(np.float64).tiny)
                )
                method_id = direct_method
                if direct_method == "S2a":
                    paired_method = "S2b"
                elif direct_method == "T2a":
                    paired_method = "T2b"
                else:
                    paired_method = ""
                rows.append(
                    {
                        "scenario": scenario_id,
                        "evaluation_pattern": "fixed_beam_single_source",
                        "band_case": band_case.case_id,
                        "analysis_width_hz": band_case.analysis_width_hz,
                        "occupied_band_low_hz": band_case.occupied_band_hz[0],
                        "occupied_band_high_hz": band_case.occupied_band_hz[1],
                        "source_azimuth_deg": azimuth_deg,
                        "signal_level_db_re_input_rms": signal_level_db,
                        "noise_band_level_db_re_input_rms": signal_level_db - input_snr_db,
                        "input_snr_db": input_snr_db,
                        "algorithm": algorithm,
                        "method": method_id,
                        "equivalent_difference_branch_method": paired_method,
                        "tap_count": tap_count,
                        "target_level_db_re_input_rms": 10.0
                        * np.log10(max(target_power, np.finfo(np.float64).tiny)),
                        "target_level_error_db": 10.0
                        * np.log10(max(target_power / source_power, np.finfo(np.float64).tiny)),
                        "noise_output_db_re_input_rms": 10.0
                        * np.log10(max(noise_power, np.finfo(np.float64).tiny)),
                        "output_snr_db": output_snr_db,
                        "snr_gain_db": output_snr_db - input_snr_db,
                        "target_energy_ratio": float(approximation.energy_ratio[target_index]),
                        "finite": bool(
                            np.isfinite(target_power)
                            and np.isfinite(noise_power)
                            and np.isfinite(output_snr_db)
                        ),
                    }
                )
    return rows


def calculate_validation_matrix() -> tuple[dict[str, Any], ...]:
    """指定された帯域・方位・level・SNR・tap条件を評価する。"""
    rows: list[dict[str, Any]] = []
    # 全SNR・帯域・方位は0 dB信号で方式性能を評価する。
    for band_case in BAND_CASES:
        for azimuth_deg in AZIMUTHS_DEG:
            for input_snr_db in INPUT_SNRS_DB:
                rows.extend(_method_rows(band_case, azimuth_deg, 0.0, input_snr_db))
    # 絶対level依存性はSNR=0 dBのまま-20/+20 dBへ平行移動して確認する。
    for band_case in BAND_CASES:
        for azimuth_deg in AZIMUTHS_DEG:
            for signal_level_db in (-20.0, 20.0):
                rows.extend(_method_rows(band_case, azimuth_deg, signal_level_db, 0.0))
    return tuple(rows)


def write_review_pack(output_dir: Path = OUTPUT_DIR) -> None:
    """単一信号直交試験のCSVと日本語索引を保存する。"""
    rows = calculate_validation_matrix()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scenario_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True, sharey=True)
    method_labels = {
        "S1": "S1",
        "S2a": "S2a / S2b",
        "T1": "T1",
        "T2a": "T2a / T2b",
    }
    for axis, band_case in zip(axes, BAND_CASES, strict=True):
        for method, label in method_labels.items():
            values = []
            for tap_count in TAP_COUNTS:
                selected = [
                    abs(float(row["target_level_error_db"]))
                    for row in rows
                    if row["band_case"] == band_case.case_id
                    and row["method"] == method
                    and int(row["tap_count"]) == tap_count
                ]
                values.append(max(selected))
            axis.plot(TAP_COUNTS, values, marker="o", label=label)
        title = (
            "Narrowband bin-center tone" if band_case.analysis_width_hz == 0.0 else "Flat broadband"
        )
        axis.set(title=title, xlabel="FIR tap count", xticks=TAP_COUNTS)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Maximum target-level error [dB re configured signal level]")
    figure.savefig(figure_dir / "tap_target_level_error.png", dpi=160)
    plt.close(figure)
    (output_dir / "review_index.md").write_text(
        "# 単一信号・全整相方式の直交試験\n\n"
        "狭帯域/広帯域、0/90/150 deg、入力SNR -10/0/10/20 dB、"
        "信号と雑音の同時level shift、32/128/256 tapを評価する。\n\n"
        "絶対levelは帯域積分RMSの dB re input RMS、SNRは帯域積分power比である。\n"
        "方式表記はS1、S2a/S2b、T1、T2a/T2bとする。"
        "b方式は共通FIR射影下でa方式と同値である。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_review_pack()
