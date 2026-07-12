"""方位別時間切り出し共分散を実信号帯域と粗いFFT binの組合せで評価する。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming.diagnostic_plotting import require_matplotlib  # noqa: E402
from evaluations.beamforming.analysis_width_long_array_mvdr import (  # noqa: E402
    DIAGONAL_LOADING,
    FS_HZ,
    N_CHANNEL,
    SCAN_AZIMUTHS_DEG,
    SOUND_SPEED_M_S,
    _correlation_median,
    _curve_metrics,
    _delays,
    _eta,
    _mvdr_capon,
    _positions,
    _steering,
)


OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "analysis_width_signal_band_covariance"
DELTA_F_HZ = (16.0, 64.0, 256.0)
SOURCE_AZIMUTH_DEG = 60.0
NOISE_ASD_DB_RE_TARGET_RMS_PER_SQRT_HZ = -32.0
FREQUENCY_QUADRATURE_COUNT = 65


@dataclass(frozen=True)
class SignalCondition:
    """物理周波数を維持して粗いFFT binへ入力する信号条件を表す。"""

    name: str
    display_name: str
    band_hz: tuple[float, float] | None = None
    tone_hz: float | None = None


SIGNALS = (
    SignalCondition("broadband_40_60_hz", "Broadband 40-60 Hz", band_hz=(40.0, 60.0)),
    SignalCondition("broadband_80_120_hz", "Broadband 80-120 Hz", band_hz=(80.0, 120.0)),
    SignalCondition("broadband_160_240_hz", "Broadband 160-240 Hz", band_hz=(160.0, 240.0)),
    SignalCondition("tone_100_hz", "Tone 100 Hz", tone_hz=100.0),
)


def _nearest_fft_bin_center(signal: SignalCondition, delta_f_hz: float) -> float:
    """信号中心に最も近い非負rFFT bin中心を返す。"""

    if signal.tone_hz is not None:
        physical_center_hz = signal.tone_hz
    elif signal.band_hz is not None:
        physical_center_hz = 0.5 * (signal.band_hz[0] + signal.band_hz[1])
    else:
        raise ValueError("signal condition requires band_hz or tone_hz.")
    return round(physical_center_hz / delta_f_hz) * delta_f_hz


def _target_covariance(
    signal: SignalCondition,
    delta_f_hz: float,
    bin_center_hz: float,
    true_delay_s: NDArray[np.float64],
    candidate_delays_s: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], float]:
    """window leakageを含む方位別target共分散`[n_direction,n_ch,n_ch]`を返す。"""

    # 実装と同じ整数sample中心を使い、物理遅延をwindow内へscaleしない。
    quantized_candidate_delay_s = np.rint(candidate_delays_s * FS_HZ) / FS_HZ
    residual_delay_s = true_delay_s[None, :] - quantized_candidate_delay_s
    bin_steering = _steering(true_delay_s[None, :], bin_center_hz)[0]
    covariance = np.empty(
        (candidate_delays_s.shape[0], N_CHANNEL, N_CHANNEL),
        dtype=np.complex128,
    )

    if signal.tone_hz is not None:
        frequency_offset_hz = signal.tone_hz - bin_center_hz
        window_amplitude = float(np.sinc(frequency_offset_hz / delta_f_hz))
        for direction_start in range(0, candidate_delays_s.shape[0], 16):
            direction_stop = min(direction_start + 16, candidate_delays_s.shape[0])
            residual = residual_delay_s[direction_start:direction_stop]
            vector = (
                window_amplitude
                * bin_steering[None, :]
                * np.exp(-1j * 2.0 * np.pi * frequency_offset_hz * residual)
            )
            covariance[direction_start:direction_stop] = np.einsum(
                "di,dj->dij", vector, vector.conj(), optimize=True
            )
        return covariance, window_amplitude**2

    if signal.band_hz is None:
        raise RuntimeError("broadband condition requires band_hz.")
    low_hz, high_hz = signal.band_hz
    frequencies_hz = np.linspace(low_hz, high_hz, FREQUENCY_QUADRATURE_COUNT, dtype=np.float64)
    frequency_offset_hz = frequencies_hz - bin_center_hz
    # 入力帯域積分powerを1とし、矩形windowのbin応答sinc^2を物理周波数上で積分する。
    spectral_density = np.full(frequencies_hz.shape, 1.0 / (high_hz - low_hz), dtype=np.float64)
    window_power = np.sinc(frequency_offset_hz / delta_f_hz) ** 2
    integration_weight = spectral_density * window_power
    captured_power = float(np.trapezoid(integration_weight, frequencies_hz))
    for direction_start in range(0, candidate_delays_s.shape[0], 8):
        direction_stop = min(direction_start + 8, candidate_delays_s.shape[0])
        residual = residual_delay_s[direction_start:direction_stop]
        # vector shapeは`[n_frequency,n_direction_chunk,n_ch]`。
        vector = bin_steering[None, None, :] * np.exp(
            -1j
            * 2.0
            * np.pi
            * frequency_offset_hz[:, None, None]
            * residual[None, :, :]
        )
        outer = np.einsum("fdi,fdj->fdij", vector, vector.conj(), optimize=True)
        covariance[direction_start:direction_stop] = np.trapezoid(
            integration_weight[:, None, None, None] * outer,
            frequencies_hz,
            axis=0,
        )
    return covariance, captured_power


def _evaluate_condition(
    signal: SignalCondition,
    delta_f_hz: float,
    scan_delays_s: NDArray[np.float64],
    true_delay_s: NDArray[np.float64],
) -> dict[str, Any]:
    """一つの信号・分析幅について3 sceneとMVDRを評価する。"""

    bin_center_hz = _nearest_fft_bin_center(signal, delta_f_hz)
    scan_steering = _steering(scan_delays_s, bin_center_hz)
    true_bin_steering = _steering(true_delay_s[None, :], bin_center_hz)[0]
    target_covariance, captured_power = _target_covariance(
        signal,
        delta_f_hz,
        bin_center_hz,
        true_delay_s,
        scan_delays_s,
    )
    # NL=-32 dB re target RMS/sqrt(Hz)をbin幅で積分した空間白色雑音power。
    noise_power = 10.0 ** (NOISE_ASD_DB_RE_TARGET_RMS_PER_SQRT_HZ / 10.0) * delta_f_hz
    identity = np.eye(N_CHANNEL, dtype=np.complex128)[None, :, :]
    scene_covariances = {
        "target_only": target_covariance,
        "noise_only": np.broadcast_to(noise_power * identity, target_covariance.shape).copy(),
        "target_plus_noise": target_covariance + noise_power * identity,
    }
    source_index = int(np.argmin(np.abs(SCAN_AZIMUTHS_DEG - SOURCE_AZIMUTH_DEG)))
    scenes: dict[str, Any] = {}
    curves: dict[str, dict[str, NDArray[np.float64]]] = {}
    for scene_name, covariance in scene_covariances.items():
        eta_curve = _eta(covariance, scan_steering)
        correlation_curve = _correlation_median(covariance)
        mvdr_curve, covariance_quality = _mvdr_capon(
            covariance[source_index],
            scan_steering,
            true_bin_steering,
        )
        curves[scene_name] = {
            "correlation_median": correlation_curve,
            "steering_eta": eta_curve,
            "mvdr_capon": mvdr_curve,
        }
        scenes[scene_name] = {
            "correlation_median_metrics": _curve_metrics(correlation_curve, SOURCE_AZIMUTH_DEG),
            "steering_eta_metrics": _curve_metrics(eta_curve, SOURCE_AZIMUTH_DEG),
            "mvdr_metrics": _curve_metrics(mvdr_curve, SOURCE_AZIMUTH_DEG),
            "covariance_quality": covariance_quality,
        }
    return {
        "delta_f_hz": delta_f_hz,
        "nfft": int(round(FS_HZ / delta_f_hz)),
        "bin_center_hz": bin_center_hz,
        "bin_is_dc": bool(bin_center_hz == 0.0),
        "captured_target_power_re_input_band_rms": captured_power,
        "noise_power_re_target_rms_squared": noise_power,
        "scenes": scenes,
        "curves": curves,
    }


def _plot_signal(signal: SignalCondition, results: dict[str, Any]) -> None:
    """target+noiseについて分析幅ごとに3指標を同一axisへ重ねて保存する。

    Args:
        signal: 評価する信号帯域またはtone条件。
        results: 分析幅ごとの評価結果。各curveのshapeは`[n_direction]`。

    Returns:
        なし。比較図を成果物directoryへ保存する。

    Notes:
        target-only、noise-onlyは数値診断へ保持するが、この方式比較図へ重ねない。
        分析幅ごとの成立性を読む目的では、scene差よりmedian、eta、MVDRの差を
        同じaxisへ重ねる方が直接比較できるためである。
    """

    plt = require_matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    metric_names = ("correlation_median", "steering_eta", "mvdr_capon")
    metric_titles = ("All-pair correlation median", "Steering power eta", "MVDR/Capon spectrum")
    for axis, delta_f_hz in zip(axes, DELTA_F_HZ, strict=True):
        record = results[f"df_{delta_f_hz:g}"]
        for metric_name, metric_title in zip(metric_names, metric_titles, strict=True):
            axis.plot(
                SCAN_AZIMUTHS_DEG,
                record["curves"]["target_plus_noise"][metric_name],
                label=metric_title,
            )
        axis.axvline(SOURCE_AZIMUTH_DEG, color="black", linestyle="--", linewidth=1.0)
        axis.set(
            title=f"df={delta_f_hz:g} Hz, bin={record['bin_center_hz']:g} Hz",
            xlabel="Azimuth [deg]",
            ylabel="Ratio",
            xlim=(0.0, 180.0),
            ylim=(0.0, 1.05),
        )
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"{signal.display_name}, target+noise, source azimuth 60 deg")
    figure.savefig(OUTPUT_DIR / f"{signal.name}_comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """3分析幅×4信号条件の方位別時間切り出し共分散を評価する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    positions = _positions()
    scan_delays = _delays(positions, SCAN_AZIMUTHS_DEG)
    true_delay = _delays(positions, np.asarray([SOURCE_AZIMUTH_DEG], dtype=np.float32))[0]
    payload_results: dict[str, Any] = {}
    start = time.perf_counter()
    for signal in SIGNALS:
        signal_results: dict[str, Any] = {}
        for delta_f_hz in DELTA_F_HZ:
            record = _evaluate_condition(signal, delta_f_hz, scan_delays, true_delay)
            signal_results[f"df_{delta_f_hz:g}"] = record
        _plot_signal(signal, signal_results)
        # JSONへ巨大curveを重複保存せず、描画再現用だけNPZへ固定shapeで分離する。
        np.savez_compressed(  # pyright: ignore[reportArgumentType]
            OUTPUT_DIR / f"{signal.name}_curves.npz",
            azimuth_deg=SCAN_AZIMUTHS_DEG,
            **{
                f"df_{delta_f_hz:g}_{scene_name}_{metric_name}": signal_results[f"df_{delta_f_hz:g}"]["curves"][
                    scene_name
                ][metric_name]
                for delta_f_hz in DELTA_F_HZ
                for scene_name in ("target_only", "noise_only", "target_plus_noise")
                for metric_name in ("correlation_median", "steering_eta", "mvdr_capon")
            },
        )
        for record in signal_results.values():
            del record["curves"]
        payload_results[signal.name] = signal_results
    payload = {
        "array": {"n_channel": N_CHANNEL, "spacing_m": 6.25, "aperture_m": 393.75},
        "source_azimuth_deg": SOURCE_AZIMUTH_DEG,
        "sample_rate_hz": FS_HZ,
        "delta_f_hz": list(DELTA_F_HZ),
        "noise_asd_db_re_target_rms_per_sqrt_hz": NOISE_ASD_DB_RE_TARGET_RMS_PER_SQRT_HZ,
        "diagonal_loading_ratio": DIAGONAL_LOADING,
        "evaluation_seconds": time.perf_counter() - start,
        "results": payload_results,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
