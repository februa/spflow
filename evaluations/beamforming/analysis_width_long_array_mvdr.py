"""長大ULAで分析幅がsteering powerとMVDRへ与える影響を分離評価する。"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming_evaluation.diagnostic_plotting import require_matplotlib  # noqa: E402

OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "analysis_width_long_array_mvdr"
FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
N_CHANNEL = 64
SPACING_M = 6.25
APERTURE_M = SPACING_M * (N_CHANNEL - 1)
APERTURE_DELAY_S = APERTURE_M / SOUND_SPEED_M_S
DELTA_F_HZ = (1.0, 4.0, 16.0, 64.0, 256.0)
SOURCE_AZIMUTHS_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)
SCAN_AZIMUTHS_DEG = np.linspace(0.0, 180.0, 721, dtype=np.float32)
DIAGONAL_LOADING = 1.0e-3
NOISE_POWER = 1.0e-2


def _positions() -> NDArray[np.float64]:
    """中心対称な64ch等間隔ULA位置`[n_ch]`をm単位で返す。"""

    return np.linspace(-APERTURE_M / 2.0, APERTURE_M / 2.0, N_CHANNEL, dtype=np.float64)


def _delays(positions_m: NDArray[np.float64], azimuths_deg: NDArray[np.float32]) -> NDArray[np.float64]:
    """ULA方位ごとの相対到来遅延`[n_direction,n_ch]`を秒単位で返す。"""

    # 0/180度をendfire、90度をbroadsideとする既存方位規約に合わせる。
    return np.cos(np.deg2rad(azimuths_deg.astype(np.float64)))[:, None] * positions_m[None, :] / SOUND_SPEED_M_S


def _nearest_positive_bin_center(delta_f_hz: float) -> float:
    """100 Hzに最も近い正のFFT bin中心周波数を返す。"""

    return max(delta_f_hz, round(100.0 / delta_f_hz) * delta_f_hz)


def _steering(delays_s: NDArray[np.float64], frequency_hz: float) -> NDArray[np.complex128]:
    """中心周波数steering`[n_direction,n_ch]`を生成する。"""

    # a(theta,f)=exp(-j 2 pi f tau)として、R=a a^Hならu^H R uが正解方位で最大になる。
    return np.exp(-1j * 2.0 * np.pi * frequency_hz * delays_s).astype(np.complex128)


def _direction_covariance(
    true_delay_s: NDArray[np.float64],
    candidate_delays_s: NDArray[np.float64],
    true_steering: NDArray[np.complex128],
    *,
    delta_f_hz: float,
    time_cut: bool,
    tone: bool,
    scene: str,
) -> NDArray[np.complex128]:
    """選択binの方位別共分散`[n_direction,n_ch,n_ch]`を解析式で返す。

    平坦な1 bin広帯域信号のpair coherenceは、矩形周波数積分から
    `sinc(delta_f * residual_delay_ij)`となる。時間切り出しありでは候補遅延を
    true遅延から引き、同一時間blockでは候補によらずtrue遅延を残す。
    """

    n_direction = int(candidate_delays_s.shape[0])
    identity = np.eye(N_CHANNEL, dtype=np.complex128)[None, :, :]
    if scene == "noise_only":
        return np.broadcast_to(NOISE_POWER * identity, (n_direction, N_CHANNEL, N_CHANNEL)).copy()

    # 実装と同じ整数sample中心を使い、物理遅延自体はscaleせず1/fsへ丸める。
    quantized_candidate_delays_s = np.rint(candidate_delays_s * FS_HZ) / FS_HZ
    residual_delay = (
        true_delay_s[None, :] - quantized_candidate_delays_s
        if time_cut
        else np.broadcast_to(true_delay_s[None, :], candidate_delays_s.shape)
    )
    pair_residual = residual_delay[:, :, None] - residual_delay[:, None, :]
    coherence = np.ones(pair_residual.shape, dtype=np.float64) if tone else np.sinc(delta_f_hz * pair_residual)
    source_outer = true_steering[:, None] * true_steering.conj()[None, :]
    covariance = coherence * source_outer[None, :, :]
    if scene == "target_plus_noise":
        covariance = covariance + NOISE_POWER * identity
    return np.asarray(covariance, dtype=np.complex128)


def _eta(covariance: NDArray[np.complex128], steering: NDArray[np.complex128]) -> NDArray[np.float64]:
    """方位別共分散と同じ候補steeringからeta`[n_direction]`を計算する。"""

    normalized = steering / np.linalg.norm(steering, axis=1, keepdims=True)
    numerator = np.real(np.einsum("di,dij,dj->d", normalized.conj(), covariance, normalized, optimize=True))
    trace = np.real(np.trace(covariance, axis1=1, axis2=2))
    return np.asarray(numerator / np.maximum(trace, np.finfo(np.float64).tiny), dtype=np.float64)


def _correlation_median(covariance: NDArray[np.complex128]) -> NDArray[np.float64]:
    """全channel pair絶対正規化相関median`[n_direction]`を返す。"""

    first, second = np.tril_indices(N_CHANNEL, k=-1)
    diagonal = np.real(np.diagonal(covariance, axis1=1, axis2=2))
    denominator = np.sqrt(diagonal[:, first] * diagonal[:, second])
    pair = np.abs(covariance[:, first, second]) / np.maximum(denominator, np.finfo(np.float64).tiny)
    return np.asarray(np.median(pair, axis=1), dtype=np.float64)


def _curve_metrics(curve: NDArray[np.float64], source_azimuth_deg: float) -> dict[str, float]:
    """scan curveのpeak誤差、margin、半高幅を返す。"""

    error = np.abs(SCAN_AZIMUTHS_DEG.astype(np.float64) - source_azimuth_deg)
    source_index = int(np.argmin(error))
    far = error >= 20.0
    peak_index = int(np.argmax(curve))
    source_value = float(curve[source_index])
    far_mean = float(np.mean(curve[far]))
    far_max = float(np.max(curve[far]))
    far_indices = np.flatnonzero(far)
    far_peak_index = int(far_indices[int(np.argmax(curve[far]))])
    half = float(np.min(curve) + 0.5 * (np.max(curve) - np.min(curve)))
    left = source_index
    right = source_index
    while left > 0 and curve[left - 1] >= half:
        left -= 1
    while right + 1 < curve.size and curve[right + 1] >= half:
        right += 1
    return {
        "source_value": source_value,
        "far_mean": far_mean,
        "far_maximum": far_max,
        "far_peak_azimuth_deg": float(SCAN_AZIMUTHS_DEG[far_peak_index]),
        "source_far_mean_margin": source_value - far_mean,
        "source_far_peak_margin": source_value - far_max,
        "peak_azimuth_deg": float(SCAN_AZIMUTHS_DEG[peak_index]),
        "peak_error_deg": float(abs(float(SCAN_AZIMUTHS_DEG[peak_index]) - source_azimuth_deg)),
        "half_prominence_width_deg": float(SCAN_AZIMUTHS_DEG[right] - SCAN_AZIMUTHS_DEG[left]),
    }


def _mvdr_capon(
    covariance: NDArray[np.complex128],
    scan_steering: NDArray[np.complex128],
    source_steering: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """固定loadingのCapon spectrumとMVDR品質を返す。"""

    hermitian = 0.5 * (covariance + covariance.conj().T)
    trace = float(np.real(np.trace(hermitian)))
    loading = DIAGONAL_LOADING * trace / N_CHANNEL
    loaded = hermitian + loading * np.eye(N_CHANNEL, dtype=np.complex128)
    inverse = np.linalg.inv(loaded)
    denominator = np.real(np.einsum("di,ij,dj->d", scan_steering.conj(), inverse, scan_steering, optimize=True))
    capon = 1.0 / np.maximum(denominator, np.finfo(np.float64).tiny)
    capon /= np.max(capon)
    solved = inverse @ source_steering
    weight = solved / np.vdot(source_steering, solved)
    response = np.vdot(weight, source_steering)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    hermitian_error = np.linalg.norm(covariance - covariance.conj().T) / max(
        float(np.linalg.norm(covariance)), np.finfo(np.float64).tiny
    )
    return np.asarray(capon, dtype=np.float64), {
        "diagonal_loading_ratio": DIAGONAL_LOADING,
        "diagonal_loading_absolute": loading,
        "distortionless_response_error_db": float(20.0 * np.log10(max(abs(response), np.finfo(np.float64).tiny))),
        "hermitian_relative_error": float(hermitian_error),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "trace": trace,
        "condition_number_loaded": float(np.linalg.cond(loaded)),
    }


def _plot_overlays(records: dict[str, Any]) -> None:
    """代表方位のdelta_f重ね描きとheatmapを保存する。"""

    plt = require_matplotlib()
    for source in (0.0, 90.0, 180.0):
        figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
        for row, mode in enumerate(("same_time", "time_cut")):
            for delta_f in DELTA_F_HZ:
                key = f"df_{delta_f:g}_source_{source:g}_{mode}_broadband_target_plus_noise"
                record = records[key]
                axes[row, 0].plot(SCAN_AZIMUTHS_DEG, record["eta_curve"], label=f"df={delta_f:g} Hz")
                axes[row, 1].plot(SCAN_AZIMUTHS_DEG, record["mvdr_curve"], label=f"df={delta_f:g} Hz")
            for column, metric_title in enumerate(("Steering power eta", "MVDR/Capon spectrum")):
                axis = axes[row, column]
                axis.axvline(source, color="black", linestyle="--", linewidth=1.0)
                axis.set(
                    title=f"{mode}: {metric_title}",
                    xlabel="Azimuth [deg]",
                    ylabel="Normalized ratio",
                    xlim=(0.0, 180.0),
                )
                axis.grid(True, alpha=0.25)
                axis.legend(fontsize=8)
        figure.suptitle(f"Long ULA analysis-width comparison: source {source:g} deg")
        figure.savefig(OUTPUT_DIR / f"overlay_source_{source:g}_deg.png", dpi=160)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
        for delta_f in DELTA_F_HZ:
            key = f"df_{delta_f:g}_source_{source:g}_time_cut_tone_target_plus_noise"
            record = records[key]
            axes[0].plot(SCAN_AZIMUTHS_DEG, record["eta_curve"], label=f"df={delta_f:g} Hz")
            axes[1].plot(SCAN_AZIMUTHS_DEG, record["mvdr_curve"], label=f"df={delta_f:g} Hz")
        for axis, title in zip(axes, ("Tone steering power eta", "Tone MVDR/Capon spectrum"), strict=True):
            axis.axvline(source, color="black", linestyle="--", linewidth=1.0)
            axis.set(title=title, xlabel="Azimuth [deg]", ylabel="Normalized ratio", xlim=(0.0, 180.0))
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
        figure.suptitle(f"Bin-centered tone comparison: source {source:g} deg")
        figure.savefig(OUTPUT_DIR / f"overlay_tone_source_{source:g}_deg.png", dpi=160)
        plt.close(figure)

    for mode_name in ("same_time", "time_cut"):
        for metric_name, title in (
            ("eta_source_far_peak_margin", "Steering eta source-to-far-peak margin"),
            ("mvdr_source_far_peak_margin", "MVDR source-to-far-peak margin"),
        ):
            table = np.empty((len(DELTA_F_HZ), len(SOURCE_AZIMUTHS_DEG)), dtype=np.float64)
            for row, delta_f in enumerate(DELTA_F_HZ):
                for column, source in enumerate(SOURCE_AZIMUTHS_DEG):
                    key = f"df_{delta_f:g}_source_{source:g}_{mode_name}_broadband_target_plus_noise"
                    table[row, column] = records[key][metric_name]
            figure, axis = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
            image = axis.imshow(
                table,
                origin="lower",
                aspect="auto",
                vmin=float(np.min(table)),
                vmax=float(np.max(table)),
            )
            axis.set(
                title=f"{mode_name}: {title}",
                xlabel="Source azimuth [deg]",
                ylabel="Analysis width delta_f [Hz]",
                xticks=np.arange(len(SOURCE_AZIMUTHS_DEG)),
                xticklabels=[f"{value:g}" for value in SOURCE_AZIMUTHS_DEG],
                yticks=np.arange(len(DELTA_F_HZ)),
                yticklabels=[f"{value:g}" for value in DELTA_F_HZ],
            )
            figure.colorbar(image, ax=axis, label="Margin [ratio]")
            figure.savefig(OUTPUT_DIR / f"heatmap_{mode_name}_{metric_name}.png", dpi=160)
            plt.close(figure)


def main() -> None:
    """分析幅・方位・時間切り出し・信号種別をsweepして成果物を生成する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    positions = _positions()
    scan_delays = _delays(positions, SCAN_AZIMUTHS_DEG)
    records: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    tracemalloc.start()
    evaluation_start = time.perf_counter()

    for delta_f in DELTA_F_HZ:
        nfft = int(round(FS_HZ / delta_f))
        center_frequency = _nearest_positive_bin_center(delta_f)
        scan_steering = _steering(scan_delays, center_frequency)
        schedule_fits = (nfft / FS_HZ + APERTURE_DELAY_S) <= 1.0
        for source_azimuth in SOURCE_AZIMUTHS_DEG:
            true_delay = _delays(positions, np.asarray([source_azimuth], dtype=np.float32))[0]
            true_steering = _steering(true_delay[None, :], center_frequency)[0]
            for time_cut in (False, True):
                mode_name = "time_cut" if time_cut else "same_time"
                for tone in (False, True):
                    signal_name = "tone" if tone else "broadband"
                    for scene in ("target_only", "noise_only", "target_plus_noise"):
                        start = time.perf_counter()
                        covariance_by_direction = _direction_covariance(
                            true_delay,
                            scan_delays,
                            true_steering,
                            delta_f_hz=delta_f,
                            time_cut=time_cut,
                            tone=tone,
                            scene=scene,
                        )
                        eta_curve = _eta(covariance_by_direction, scan_steering)
                        correlation_curve = _correlation_median(covariance_by_direction)
                        eta_metrics = _curve_metrics(eta_curve, source_azimuth)
                        source_index = int(np.argmin(np.abs(SCAN_AZIMUTHS_DEG - source_azimuth)))
                        selected_covariance = covariance_by_direction[source_index]
                        mvdr_curve, covariance_quality = _mvdr_capon(
                            selected_covariance,
                            scan_steering,
                            true_steering,
                        )
                        mvdr_metrics = _curve_metrics(mvdr_curve, source_azimuth)
                        elapsed = time.perf_counter() - start
                        key = (
                            f"df_{delta_f:g}_source_{source_azimuth:g}_{mode_name}_{signal_name}_{scene}"
                        )
                        record = {
                            "delta_f_hz": delta_f,
                            "nfft": nfft,
                            "block_duration_s": nfft / FS_HZ,
                            "center_frequency_hz": center_frequency,
                            "tone_bin_center_aligned": True,
                            "source_azimuth_deg": source_azimuth,
                            "time_cut": time_cut,
                            "signal_type": signal_name,
                            "scene": scene,
                            "schedule_fits_one_second": schedule_fits,
                            "schedule_failure_reason": None if schedule_fits else "GEOMETRY_WINDOW_DOES_NOT_FIT_ONE_SECOND",
                            "delta_f_tau_aperture": delta_f * APERTURE_DELAY_S,
                            "maximum_bin_edge_residual_phase_rad": np.pi * delta_f * APERTURE_DELAY_S,
                            "correlation_median_source": float(correlation_curve[source_index]),
                            "eta_source_far_peak_margin": eta_metrics["source_far_peak_margin"],
                            "mvdr_source_far_peak_margin": mvdr_metrics["source_far_peak_margin"],
                            "eta_metrics": eta_metrics,
                            "mvdr_metrics": mvdr_metrics,
                            "covariance_quality": covariance_quality,
                            "processing_seconds": elapsed,
                            "covariance_bytes_selected_bin": int(covariance_by_direction.nbytes),
                            "eta_curve": eta_curve.tolist(),
                            "mvdr_curve": mvdr_curve.tolist(),
                        }
                        records[key] = record
                        if signal_name == "broadband" and scene == "target_plus_noise":
                            summary_rows.append({name: value for name, value in record.items() if not name.endswith("_curve")})

    total_elapsed = time.perf_counter() - evaluation_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    failure_boundaries: list[dict[str, Any]] = []
    for mode_name in ("same_time", "time_cut"):
        for source_azimuth in SOURCE_AZIMUTHS_DEG:
            reference_key = f"df_1_source_{source_azimuth:g}_{mode_name}_broadband_target_plus_noise"
            reference = records[reference_key]
            reference_eta_width = max(float(reference["eta_metrics"]["half_prominence_width_deg"]), 0.25)
            reference_mvdr_width = max(float(reference["mvdr_metrics"]["half_prominence_width_deg"]), 0.25)
            first_eta_failure: float | None = None
            first_mvdr_failure: float | None = None
            for delta_f in DELTA_F_HZ:
                key = f"df_{delta_f:g}_source_{source_azimuth:g}_{mode_name}_broadband_target_plus_noise"
                record = records[key]
                eta_difference = np.asarray(record["eta_curve"], dtype=np.float64) - np.asarray(
                    reference["eta_curve"], dtype=np.float64
                )
                mvdr_difference = np.asarray(record["mvdr_curve"], dtype=np.float64) - np.asarray(
                    reference["mvdr_curve"], dtype=np.float64
                )
                record["eta_curve_rmse_re_one_hz"] = float(np.sqrt(np.mean(eta_difference**2)))
                record["mvdr_curve_rmse_re_one_hz"] = float(np.sqrt(np.mean(mvdr_difference**2)))
                eta_failure = (
                    float(record["eta_metrics"]["peak_error_deg"]) > 2.0
                    or float(record["eta_metrics"]["source_far_peak_margin"]) <= 0.0
                    or float(record["eta_metrics"]["half_prominence_width_deg"]) > 2.0 * reference_eta_width
                )
                mvdr_failure = (
                    float(record["mvdr_metrics"]["peak_error_deg"]) > 2.0
                    or float(record["mvdr_metrics"]["source_far_peak_margin"]) <= 0.0
                    or float(record["mvdr_metrics"]["half_prominence_width_deg"]) > 2.0 * reference_mvdr_width
                    or abs(float(record["covariance_quality"]["distortionless_response_error_db"])) > 0.5
                )
                record["eta_failure"] = eta_failure
                record["mvdr_failure"] = mvdr_failure
                if eta_failure and first_eta_failure is None:
                    first_eta_failure = delta_f
                if mvdr_failure and first_mvdr_failure is None:
                    first_mvdr_failure = delta_f
            failure_boundaries.append(
                {
                    "mode": mode_name,
                    "source_azimuth_deg": source_azimuth,
                    "first_eta_failure_delta_f_hz": first_eta_failure,
                    "first_mvdr_failure_delta_f_hz": first_mvdr_failure,
                    "one_hz_schedule_geometry_failure": True,
                }
            )

    _plot_overlays(records)
    payload = {
        "array": {
            "n_channel": N_CHANNEL,
            "spacing_m": SPACING_M,
            "aperture_m": APERTURE_M,
            "aperture_delay_s": APERTURE_DELAY_S,
            "sound_speed_m_s": SOUND_SPEED_M_S,
        },
        "sample_rate_hz": FS_HZ,
        "delta_f_hz": list(DELTA_F_HZ),
        "source_azimuths_deg": list(SOURCE_AZIMUTHS_DEG),
        "diagonal_loading_ratio": DIAGONAL_LOADING,
        "noise_power_re_target_power": NOISE_POWER,
        "evaluation_seconds_total": total_elapsed,
        "tracemalloc_peak_bytes": peak_memory,
        "summary_rows": summary_rows,
        "failure_boundaries": failure_boundaries,
        "records": records,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
