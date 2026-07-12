"""強信号の近傍に置いた弱信号のS/T方位推定可視性を評価する。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from spflow.beamforming.ebae import EbaeConfig, design_ebae_weights_band


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
OUTPUT_DIR = Path("artifacts/beamforming/alignment_weak_source_visibility_sweep/review_pack")
FS_HZ = 8192.0
SOUND_SPEED_M_S = 1500.0
N_CHANNEL = 8
APERTURE_M = 42.0
CENTER_FREQUENCY_HZ = 96.0
AZIMUTH_DEG = np.arange(0.0, 181.0, 2.0, dtype=np.float64)
STRONG_AZIMUTHS_DEG = (0.0, 90.0, 150.0)
SEPARATIONS_DEG = (2.0, 5.0, 10.0, 20.0)
WEAK_LEVEL_DELTAS_DB = (-6.0, -20.0, -40.0)
ANALYSIS_WIDTHS_HZ = (0.0, 16.0)
NOISE_POWER = 1.0e-2


@dataclass(frozen=True)
class Source:
    """単一sourceの方位と帯域積分powerを保持する。"""

    azimuth_deg: float
    power: float


def _delays(azimuth_deg: FloatArray) -> FloatArray:
    """ULAの相対到来遅延を返す。shapeは``[n_direction,n_ch]``、単位はs。"""
    positions_m = np.linspace(-APERTURE_M / 2.0, APERTURE_M / 2.0, N_CHANNEL)
    return np.asarray(
        -np.cos(np.deg2rad(azimuth_deg))[:, None] * positions_m[None, :] / SOUND_SPEED_M_S,
        dtype=np.float64,
    )


def _steering(delays_s: FloatArray) -> ComplexArray:
    """中心周波数steeringを返す。shapeは``[n_ch,n_direction]``。"""
    return np.asarray(
        np.exp(-1j * 2.0 * np.pi * CENTER_FREQUENCY_HZ * delays_s).T,
        dtype=np.complex128,
    )


def _covariance(
    sources: tuple[Source, ...],
    scan_delays_s: FloatArray,
    steering: ComplexArray,
    analysis_width_hz: float,
    candidate_index: int | None,
) -> ComplexArray:
    """複数sourceのS共分散または候補方位別T共分散を返す。"""
    covariance = NOISE_POWER * np.eye(N_CHANNEL, dtype=np.complex128)
    candidate_delay_s = None
    if candidate_index is not None:
        candidate_delay_s = np.rint(scan_delays_s[candidate_index] * FS_HZ) / FS_HZ
    for source in sources:
        source_index = int(np.argmin(np.abs(AZIMUTH_DEG - source.azimuth_deg)))
        residual = scan_delays_s[source_index]
        if candidate_delay_s is not None:
            residual = residual - candidate_delay_s
        pair_delay = residual[:, None] - residual[None, :]
        coherence = np.sinc(analysis_width_hz * pair_delay)
        vector = steering[:, source_index]
        covariance += source.power * coherence * vector[:, None] * vector.conj()[None, :]
    return np.asarray(covariance, dtype=np.complex128)


def _curves(
    sources: tuple[Source, ...], analysis_width_hz: float
) -> tuple[dict[str, FloatArray], dict[str, int]]:
    """S/TのEBAE MUSICとMVDR Capon curveを返す。"""
    delays = _delays(AZIMUTH_DEG)
    steering = _steering(delays)
    curves = {key: np.empty(AZIMUTH_DEG.size) for key in ("ebae_S", "ebae_T", "mvdr_S", "mvdr_T")}
    counts: dict[str, int] = {}
    s_covariance = _covariance(sources, delays, steering, analysis_width_hz, None)
    config = EbaeConfig(
        snapshot_rate_hz=float(N_CHANNEL * N_CHANNEL),
        integration_time_sec=1.0,
        sigmoid_slope=10.0,
        sigmoid_midpoint=0.5,
        diagonal_loading=1.0,
    )
    s_result = design_ebae_weights_band(
        s_covariance, steering, snapshot_count=N_CHANNEL * N_CHANNEL, config=config
    )
    curves["ebae_S"] = np.asarray(s_result.music_spectrum, dtype=np.float64)
    counts["ebae_S"] = s_result.signal_count
    for candidate in range(AZIMUTH_DEG.size):
        covariance = (
            s_covariance
            if candidate == 0
            else _covariance(sources, delays, steering, analysis_width_hz, candidate)
        )
        hermitian = 0.5 * (covariance + covariance.conj().T)
        loading = 1.0e-3 * float(np.real(np.trace(hermitian))) / N_CHANNEL
        inverse = np.linalg.inv(hermitian + loading * np.eye(N_CHANNEL))
        a = steering[:, candidate]
        curves["mvdr_S"][candidate] = 1.0 / max(
            float(np.real(np.vdot(a, np.linalg.solve(s_covariance, a)))),
            np.finfo(np.float64).tiny,
        )
        curves["mvdr_T"][candidate] = 1.0 / max(
            float(np.real(np.vdot(a, inverse @ a))), np.finfo(np.float64).tiny
        )
        t_result = design_ebae_weights_band(
            covariance, steering, snapshot_count=N_CHANNEL * N_CHANNEL, config=config
        )
        curves["ebae_T"][candidate] = float(t_result.music_spectrum[candidate])
        # Tの信号数は最後に列挙したsource方位へ整合したcandidateで記録する。
        # 1信号条件でも同じ規約を使えるため、source数sweepの境界で特別分岐を持たない。
        if candidate == int(np.argmin(np.abs(AZIMUTH_DEG - sources[-1].azimuth_deg))):
            counts["ebae_T"] = t_result.signal_count
    return curves, counts


def _visibility(
    curve: FloatArray, weak_azimuth_deg: float, strong_azimuth_deg: float
) -> tuple[float, float, bool]:
    """強信号とは別の弱信号peakについて誤差、prominence、可視性を返す。"""
    nearest = int(np.argmin(np.abs(AZIMUTH_DEG - weak_azimuth_deg)))
    strong_index = int(np.argmin(np.abs(AZIMUTH_DEG - strong_azimuth_deg)))
    candidates = [
        index
        for index in range(max(0, nearest - 1), min(curve.size, nearest + 2))
        if index != strong_index
    ]
    local = max(candidates, key=lambda index: float(curve[index]))
    normalized_db = 10.0 * np.log10(
        np.maximum(curve / max(float(np.max(curve)), np.finfo(np.float64).tiny), 1.0e-12)
    )
    between = normalized_db[min(strong_index, local) : max(strong_index, local) + 1]
    prominence = float(normalized_db[local] - np.min(between))
    error = float(abs(AZIMUTH_DEG[local] - weak_azimuth_deg))
    is_local_peak = bool(
        0 < local < curve.size - 1
        and curve[local] >= curve[local - 1]
        and curve[local] >= curve[local + 1]
    )
    return error, prominence, bool(error <= 2.0 and prominence >= 3.0 and is_local_peak)


def calculate_visibility_sweep() -> tuple[dict[str, Any], ...]:
    """強弱level差・方位間隔・分析幅の全条件を評価する。"""
    rows: list[dict[str, Any]] = []
    for strong_azimuth in STRONG_AZIMUTHS_DEG:
        sign = 1.0 if strong_azimuth < 90.0 else -1.0
        for separation in SEPARATIONS_DEG:
            weak_azimuth = strong_azimuth + sign * separation
            for weak_delta_db in WEAK_LEVEL_DELTAS_DB:
                sources = (
                    Source(strong_azimuth, 1.0),
                    Source(weak_azimuth, 10.0 ** (weak_delta_db / 10.0)),
                )
                for analysis_width_hz in ANALYSIS_WIDTHS_HZ:
                    curves, counts = _curves(sources, analysis_width_hz)
                    for key, curve in curves.items():
                        error, prominence, visible = _visibility(
                            curve, weak_azimuth, strong_azimuth
                        )
                        algorithm, method = key.split("_")
                        rows.append(
                            {
                                "evaluation_pattern": "fixed_beam_multi_source",
                                "algorithm": algorithm,
                                "covariance_method": method,
                                "analysis_width_hz": analysis_width_hz,
                                "strong_azimuth_deg": strong_azimuth,
                                "weak_azimuth_deg": weak_azimuth,
                                "separation_deg": separation,
                                "weak_level_delta_db_re_strong": weak_delta_db,
                                "weak_peak_error_deg": error,
                                "weak_peak_prominence_db": prominence,
                                "weak_source_visible": visible,
                                "detected_source_count": counts.get(key, -1),
                            }
                        )
    return tuple(rows)


def calculate_source_count_sweep() -> tuple[dict[str, Any], ...]:
    """1～3信号についてEBAEの推定信号数をS/T共分散で確認する。"""
    layouts = ((90.0,), (80.0, 100.0), (70.0, 90.0, 110.0))
    rows: list[dict[str, Any]] = []
    for azimuths in layouts:
        sources = tuple(Source(value, 1.0) for value in azimuths)
        for analysis_width_hz in ANALYSIS_WIDTHS_HZ:
            _, counts = _curves(sources, analysis_width_hz)
            for method in ("S", "T"):
                detected = counts[f"ebae_{method}"]
                rows.append(
                    {
                        "analysis_width_hz": analysis_width_hz,
                        "covariance_method": method,
                        "expected_source_count": len(sources),
                        "detected_source_count": detected,
                        "source_azimuths_deg": ";".join(str(value) for value in azimuths),
                        "count_matches": detected == len(sources),
                    }
                )
    return tuple(rows)


def write_review_pack(output_dir: Path = OUTPUT_DIR) -> None:
    """弱信号可視性sweepのCSVと日本語索引を保存する。"""
    rows = calculate_visibility_sweep()
    source_count_rows = calculate_source_count_sweep()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scenario_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "source_count_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_count_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_count_rows)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), constrained_layout=True)
    image_handle = None
    title_by_key = {
        "ebae_S": "EBAE, S covariance family",
        "ebae_T": "EBAE, T covariance family",
        "mvdr_S": "MVDR, S covariance family",
        "mvdr_T": "MVDR, T covariance family",
    }
    for axis, key in zip(axes.reshape(-1), ("ebae_S", "ebae_T", "mvdr_S", "mvdr_T"), strict=True):
        algorithm, method = key.split("_")
        rates = np.empty((len(WEAK_LEVEL_DELTAS_DB), len(SEPARATIONS_DEG)))
        for level_index, level_delta in enumerate(WEAK_LEVEL_DELTAS_DB):
            for separation_index, separation in enumerate(SEPARATIONS_DEG):
                selected = [
                    bool(row["weak_source_visible"])
                    for row in rows
                    if row["algorithm"] == algorithm
                    and row["covariance_method"] == method
                    and float(row["weak_level_delta_db_re_strong"]) == level_delta
                    and float(row["separation_deg"]) == separation
                ]
                rates[level_index, separation_index] = np.mean(selected)
        image_handle = axis.imshow(rates, origin="lower", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set(
            title=title_by_key[key],
            xlabel="Azimuth separation [deg]",
            ylabel="Weak-source level [dB re strong]",
            xticks=np.arange(len(SEPARATIONS_DEG)),
            xticklabels=[str(value) for value in SEPARATIONS_DEG],
            yticks=np.arange(len(WEAK_LEVEL_DELTAS_DB)),
            yticklabels=[str(value) for value in WEAK_LEVEL_DELTAS_DB],
        )
    if image_handle is None:
        raise RuntimeError("visibility heatmap requires at least one method panel.")
    figure.colorbar(image_handle, ax=axes, label="Weak-source visibility rate")
    figure.savefig(figure_dir / "weak_source_visibility_heatmap.png", dpi=160)
    plt.close(figure)
    (output_dir / "review_index.md").write_text(
        "# 強弱近接信号の可視性sweep\n\n"
        "弱信号peak誤差2 deg以下かつprominence 3 dB以上を可視とする。\n\n"
        "S covariance familyはS1/S2a/S2b、T covariance familyはT1/T2a/T2bに共通する共分散構成を表す。"
        "本図はFIR実現座標を適用する前の方位推定結果である。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_review_pack()
