"""ローカル運用アレイでsteering選択共分散を校正し、MVDRまで段階評価する。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from scene_renderer import (  # noqa: E402
    AcousticSource,
    AmbientField,
    BandLimitedNoiseSpectrum,
    ConstantEnvelope,
    FreeField,
    Receiver,
    Scene,
    SceneRenderer,
    SourceComponent,
    StaticPose,
)

from evaluations.beamforming.method3_sparse_64ch_correlation import CoordinateArray  # noqa: E402
from evaluations.beamforming.steering_power_threshold_calibration import (  # noqa: E402
    calculate_steering_power_calibration_signature,
    calibrate_steering_power_thresholds,
)
from spflow.beamforming import (  # noqa: E402
    OperationalShadingDefinition,
    OperationalSparseArrayDefinition,
    SelectedFrequencyDirectionCovarianceAccumulator,
    build_two_second_covariance_snapshot_schedule,
    design_mvdr_coefficients,
    relative_arrival_delay,
    steering_from_relative_delay,
)
from spflow.beamforming_evaluation.diagnostic_plotting import require_matplotlib  # noqa: E402

ARRAY_PATH = ROOT / "artifacts" / "beamforming" / "operational_fractional_delay_performance_test" / "array.json"
SHADING_PATH = ROOT / "artifacts" / "beamforming" / "operational_fractional_delay_performance_test" / "shading.json"
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "operational_steering_covariance_mvdr"
EVALUATION_FREQUENCY_HZ = (256.0, 1024.0, 10000.0)
NFFT = 128
N_BEAM_PER_HALF = 159
CALIBRATION_DURATION_S = 10
EVALUATION_DURATION_S = 10
TARGET_AZIMUTH_DEG = 40.0
INTERFERER_AZIMUTH_DEG = 75.0
NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ = -32.0
DIAGONAL_LOADING = 1.0e-2
MAXIMUM_ACCEPTED_PEAK_ERROR_DEG = 3.0
MAXIMUM_ACCEPTED_OUTSIDE_PEAK_DB = 0.0
MAXIMUM_ACCEPTED_WHITE_NOISE_GAIN_DB = 0.0


def _directions(azimuth_deg: NDArray[np.float32]) -> NDArray[np.float64]:
    """水平方位degを単位方向vector`[n_direction,3]`へ変換する。"""

    rad = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    return np.stack((np.cos(rad), np.sin(rad), np.zeros_like(rad)), axis=1)


def _render_scene(
    positions_m: NDArray[np.float32],
    *,
    fs_hz: float,
    sound_speed_m_s: float,
    frequency_hz: float,
    source_azimuths_deg: tuple[float, ...],
    noise: bool,
    seed: int,
    duration_s: int,
) -> NDArray[np.float32]:
    """scene_rendererで校正・評価用toneと空間白色雑音を生成する。"""

    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=CoordinateArray(positions_m),
    )
    sources = []
    for source_index, azimuth_deg in enumerate(source_azimuths_deg):
        # 同一toneを複数方位へ置くとsource間が完全coherentとなり、通常MVDRでは
        # 空間信号部分を分離できない。中心binの周囲±96 Hzに独立な狭帯域雑音を置き、
        # 同じbinを共有しつつsnapshot間のsource係数を独立化する。
        component = SourceComponent(
            spectrum=BandLimitedNoiseSpectrum(
                max(1.0, frequency_hz - 96.0),
                min(fs_hz / 2.0, frequency_hz + 96.0),
            ),
            envelope=ConstantEnvelope(),
            amplitude=None,
            level_db=0.0,
            noise_seed=seed + source_index * 100,
            noise_filter_length=513,
        )
        sources.append(
            AcousticSource.from_relative_bearing(
                bearing_deg=azimuth_deg,
                distance=1000.0,
                receiver_pose=receiver.trajectory.pose(0.0),
                components=[component],
                elevation_deg=0.0,
            )
        )
    ambient_fields = []
    if noise:
        ambient_fields.append(
            AmbientField.from_asd_level_db(
                BandLimitedNoiseSpectrum(0.0, fs_hz / 2.0),
                NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ,
                covariance=np.eye(positions_m.shape[0], dtype=np.float32),
                noise_seed=seed + 9000,
                noise_filter_length=513,
            )
        )
    scene = Scene(
        sources=sources,
        ambient_fields=ambient_fields,
        environment=FreeField(c=sound_speed_m_s),
    )
    time_s = np.arange(int(round(fs_hz * duration_s)), dtype=np.float64) / fs_hz
    return np.asarray(np.real(SceneRenderer().render(scene, receiver, time_s)), dtype=np.float32)


def _method2_covariance(
    signal: NDArray[np.float32],
    *,
    frequency_bin_index: int,
) -> NDArray[np.complex64]:
    """同一時間軸の非重複NFFT blockから対象binの方式2参照共分散を求める。"""

    n_ch, n_sample = signal.shape
    n_block = n_sample // NFFT
    # blocks shapeは`[n_ch,n_block,NFFT]`。axis=2をrFFTし対象binだけを選ぶ。
    blocks = signal[:, : n_block * NFFT].reshape(n_ch, n_block, NFFT)
    spectrum = np.asarray(np.fft.rfft(blocks, n=NFFT, axis=2)[:, :, frequency_bin_index], dtype=np.complex64)
    return np.asarray(np.einsum("ib,jb->ij", spectrum, spectrum.conj(), optimize=True) / n_block, dtype=np.complex64)


def _run_direction_accumulator(
    signal: NDArray[np.float32],
    accumulator: SelectedFrequencyDirectionCovarianceAccumulator,
    *,
    fs_hz: float,
) -> tuple[list[NDArray[np.float32]], Any]:
    """1秒ずつ処理し、2秒完成ごとのetaと最終結果を返す。"""

    eta_snapshots: list[NDArray[np.float32]] = []
    for second in range(signal.shape[1] // int(round(fs_hz))):
        accumulator.process_one_second(
            signal[:, second * int(round(fs_hz)) : (second + 1) * int(round(fs_hz))]
        )
        if (second + 1) % 2 == 0:
            eta_snapshots.append(accumulator.completed_result().eta)
    return eta_snapshots, accumulator.completed_result()


def _covariance_health(covariance: NDArray[np.complex64]) -> dict[str, float]:
    """MVDR前にHermitian性、PSD、trace、condition numberを数値化する。"""

    hermitian = np.asarray(0.5 * (covariance + covariance.conj().T), dtype=np.complex64)
    norm = float(np.linalg.norm(hermitian))
    eigenvalues = np.linalg.eigvalsh(hermitian)
    return {
        "hermitian_relative_error": float(np.linalg.norm(covariance - covariance.conj().T)) / max(norm, 1.0e-20),
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "trace": float(np.real(np.trace(hermitian))),
        "condition_number": float(np.linalg.cond(hermitian)),
    }


def _beam_metrics(
    weight: NDArray[np.complex64],
    steering_scan: NDArray[np.complex64],
    scan_azimuth_deg: NDArray[np.float32],
) -> tuple[dict[str, float], NDArray[np.float32]]:
    """weightのBL応答、target保存、interferer抑圧、外側peakを返す。"""

    response = np.abs(np.einsum("i,id->d", weight, steering_scan, optimize=True))
    target_index = int(np.argmin(np.abs(scan_azimuth_deg - TARGET_AZIMUTH_DEG)))
    interferer_index = int(np.argmin(np.abs(scan_azimuth_deg - INTERFERER_AZIMUTH_DEG)))
    normalized_db = 20.0 * np.log10(np.maximum(response / max(float(response[target_index]), 1.0e-20), 1.0e-12))
    outside = np.abs(scan_azimuth_deg - TARGET_AZIMUTH_DEG) > 5.0
    peak_index = int(np.argmax(response))
    return (
        {
            "distortionless_magnitude": float(response[target_index]),
            "peak_azimuth_deg": float(scan_azimuth_deg[peak_index]),
            "peak_error_deg": float(abs(float(scan_azimuth_deg[peak_index]) - TARGET_AZIMUTH_DEG)),
            "interferer_response_db_re_target": float(normalized_db[interferer_index]),
            "outside_peak_db_re_target": float(np.max(normalized_db[outside])),
            "white_noise_gain_db10_re_input_channel": float(10.0 * np.log10(np.sum(np.abs(weight) ** 2))),
        },
        np.asarray(normalized_db, dtype=np.float32),
    )


def _write_bl_overlay(
    path: Path,
    scan_azimuth_deg: NDArray[np.float32],
    curves: dict[str, NDArray[np.float32]],
    *,
    frequency_hz: float,
) -> None:
    """CBF・参照MVDR・steering選択MVDRのBLを同一表示条件で保存する。"""

    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    for label, curve in curves.items():
        axis.plot(scan_azimuth_deg, curve, label=label)
    axis.axvline(TARGET_AZIMUTH_DEG, color="tab:green", linestyle="--", linewidth=1.0, label="target")
    axis.axvline(INTERFERER_AZIMUTH_DEG, color="tab:red", linestyle=":", linewidth=1.0, label="interferer")
    axis.set(xlabel="Azimuth [deg]", ylabel="Level [dB re target response]", ylim=(-80.0, 5.0))
    axis.set_title(f"Operational array BL comparison at {frequency_hz:.0f} Hz")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower left")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    """代表3周波数で校正、共分散品質、MVDR BL比較を順に実行する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    array_definition = OperationalSparseArrayDefinition.load_json(ARRAY_PATH)
    shading_definition = OperationalShadingDefinition.load_json(SHADING_PATH)
    fs_hz = float(array_definition.fs_hz)
    sound_speed_m_s = float(array_definition.sound_speed_m_s)
    positions_all = np.asarray(array_definition.positions_m, dtype=np.float32)
    # BL走査軸も保持共分散の159方位に合わせ、317保持との誤認を防ぐ。
    scan_azimuth_deg = np.linspace(0.0, 180.0, 159, dtype=np.float32)
    records: list[dict[str, Any]] = []

    for frequency_hz in EVALUATION_FREQUENCY_HZ:
        full_weights = np.asarray(shading_definition.coefficients_for_frequency(frequency_hz), dtype=np.float32)
        active_indices = np.flatnonzero(full_weights > 0.0)
        positions = positions_all[active_indices]
        shading = full_weights[active_indices]
        schedule = build_two_second_covariance_snapshot_schedule(
            positions,
            fs_hz=fs_hz,
            sound_speed_m_s=sound_speed_m_s,
            snapshot_length_samples=NFFT,
            beams_per_half=N_BEAM_PER_HALF,
        )
        direction_vectors = _directions(schedule.global_direction_azimuth_deg)
        delay_s = relative_arrival_delay(positions, direction_vectors, sound_speed_m_per_s=sound_speed_m_s)
        # steering_from_relative_delayのshapeは`[n_ch,n_direction,n_bin]`。
        # 単一周波数bin axis=2を選び`[n_ch,n_direction]`へ縮退する。
        steering = np.asarray(
            steering_from_relative_delay(delay_s, np.array([frequency_hz], dtype=np.float32))[:, :, 0],
            dtype=np.complex64,
        )
        frequency_bin_index = int(round(frequency_hz * NFFT / fs_hz))

        noise_signal = _render_scene(
            positions,
            fs_hz=fs_hz,
            sound_speed_m_s=sound_speed_m_s,
            frequency_hz=frequency_hz,
            source_azimuths_deg=(),
            noise=True,
            seed=11000 + frequency_bin_index,
            duration_s=CALIBRATION_DURATION_S,
        )
        noise_accumulator = SelectedFrequencyDirectionCovarianceAccumulator(
            schedule,
            steering,
            shading,
            frequency_bin_index=frequency_bin_index,
            integration_time_seconds=10.0,
        )
        noise_eta, noise_result = _run_direction_accumulator(noise_signal, noise_accumulator, fs_hz=fs_hz)
        noise_reference_covariance = _method2_covariance(noise_signal, frequency_bin_index=frequency_bin_index)
        del noise_signal, noise_accumulator

        target_calibration_signal = _render_scene(
            positions,
            fs_hz=fs_hz,
            sound_speed_m_s=sound_speed_m_s,
            frequency_hz=frequency_hz,
            source_azimuths_deg=(TARGET_AZIMUTH_DEG,),
            noise=True,
            seed=12000 + frequency_bin_index,
            duration_s=CALIBRATION_DURATION_S,
        )
        target_accumulator = SelectedFrequencyDirectionCovarianceAccumulator(
            schedule,
            steering,
            shading,
            frequency_bin_index=frequency_bin_index,
            integration_time_seconds=10.0,
        )
        target_eta, _ = _run_direction_accumulator(target_calibration_signal, target_accumulator, fs_hz=fs_hz)
        del target_calibration_signal, target_accumulator
        source_mask = np.abs(schedule.global_direction_azimuth_deg - TARGET_AZIMUTH_DEG) <= 2.0
        signature = calculate_steering_power_calibration_signature(
            sensor_positions_m=positions,
            frequency_hz=np.array([frequency_hz], dtype=np.float32),
            direction_azimuth_deg=schedule.global_direction_azimuth_deg,
            channel_weight_table=shading[:, np.newaxis],
            sound_speed_m_s=sound_speed_m_s,
            snapshot_length_samples=NFFT,
            integration_time_seconds=10.0,
        )
        calibration = calibrate_steering_power_thresholds(
            np.stack(noise_eta)[:, :, np.newaxis],
            np.stack(target_eta)[:, source_mask, np.newaxis],
            effective_channel_count=np.array([noise_result.effective_channel_count], dtype=np.float32),
            active_channel_count=np.array([active_indices.size], dtype=np.int32),
            configuration_signature=signature,
        )

        evaluation_signal = _render_scene(
            positions,
            fs_hz=fs_hz,
            sound_speed_m_s=sound_speed_m_s,
            frequency_hz=frequency_hz,
            source_azimuths_deg=(TARGET_AZIMUTH_DEG, INTERFERER_AZIMUTH_DEG),
            noise=True,
            seed=13000 + frequency_bin_index,
            duration_s=EVALUATION_DURATION_S,
        )
        evaluation_accumulator = SelectedFrequencyDirectionCovarianceAccumulator(
            schedule,
            steering,
            shading,
            frequency_bin_index=frequency_bin_index,
            integration_time_seconds=10.0,
        )
        evaluation_eta, evaluation_result = _run_direction_accumulator(
            evaluation_signal,
            evaluation_accumulator,
            fs_hz=fs_hz,
        )
        method2_covariance = _method2_covariance(evaluation_signal, frequency_bin_index=frequency_bin_index)
        del evaluation_signal, evaluation_accumulator

        gamma_off = float(calibration.gamma_off[0])
        gamma_on = float(calibration.gamma_on[0])
        previous_eta = evaluation_eta[-2] if len(evaluation_eta) >= 2 else evaluation_eta[-1]
        weight = np.clip((previous_eta - gamma_off) / (gamma_on - gamma_off), 0.0, 1.0).astype(np.float32)
        weight_sum = float(np.sum(weight))
        if weight_sum <= 0.0:
            raise RuntimeError("steering-selected covariance has zero direction weight.")
        selected_covariance = np.asarray(
            np.einsum("d,dij->ij", weight, evaluation_result.direction_covariance, optimize=True) / weight_sum,
            dtype=np.complex64,
        )
        target_direction = int(np.argmin(np.abs(schedule.global_direction_azimuth_deg - TARGET_AZIMUTH_DEG)))
        target_steering = steering[:, target_direction]
        cbf_weight = np.asarray(np.conj(shading * target_steering / np.sum(shading)), dtype=np.complex64)
        method2_mvdr = np.asarray(
            design_mvdr_coefficients(method2_covariance, target_steering, diag_load=DIAGONAL_LOADING)[:, 0],
            dtype=np.complex64,
        )
        selected_mvdr = np.asarray(
            design_mvdr_coefficients(selected_covariance, target_steering, diag_load=DIAGONAL_LOADING)[:, 0],
            dtype=np.complex64,
        )
        oracle_noise_mvdr = np.asarray(
            design_mvdr_coefficients(noise_reference_covariance, target_steering, diag_load=DIAGONAL_LOADING)[:, 0],
            dtype=np.complex64,
        )
        steering_scan = np.asarray(
            steering_from_relative_delay(
                relative_arrival_delay(
                    positions,
                    _directions(scan_azimuth_deg),
                    sound_speed_m_per_s=sound_speed_m_s,
                ),
                np.array([frequency_hz], dtype=np.float32),
            )[:, :, 0],
            dtype=np.complex64,
        )
        curves: dict[str, NDArray[np.float32]] = {}
        beam_results: dict[str, dict[str, float]] = {}
        for name, beam_weight in (
            ("CBF", cbf_weight),
            ("method2_MVDR", method2_mvdr),
            ("steering_selected_MVDR", selected_mvdr),
            ("noise_reference_MVDR", oracle_noise_mvdr),
        ):
            metrics, curve = _beam_metrics(beam_weight, steering_scan, scan_azimuth_deg)
            beam_results[name] = metrics
            curves[name] = curve
        selected_metrics = beam_results["steering_selected_MVDR"]
        # 共分散がHermitian/PSDでもMVDRの下流応答が安全とは限らない。peak誤差、
        # mainlobe外peak、白色雑音利得の全条件を満たさない場合は方式2へfallbackする。
        selection_accepted = bool(
            selected_metrics["peak_error_deg"] <= MAXIMUM_ACCEPTED_PEAK_ERROR_DEG
            and selected_metrics["outside_peak_db_re_target"] <= MAXIMUM_ACCEPTED_OUTSIDE_PEAK_DB
            and selected_metrics["white_noise_gain_db10_re_input_channel"]
            <= MAXIMUM_ACCEPTED_WHITE_NOISE_GAIN_DB
        )
        adopted_source = "steering_selected" if selection_accepted else "method2_fallback"
        adopted_metrics = (
            selected_metrics if selection_accepted else beam_results["method2_MVDR"]
        )
        _write_bl_overlay(
            OUTPUT_DIR / f"bl_{int(round(frequency_hz))}_hz.png",
            scan_azimuth_deg,
            curves,
            frequency_hz=frequency_hz,
        )
        far_from_sources = (
            (np.abs(schedule.global_direction_azimuth_deg - TARGET_AZIMUTH_DEG) >= 20.0)
            & (np.abs(schedule.global_direction_azimuth_deg - INTERFERER_AZIMUTH_DEG) >= 20.0)
        )
        records.append(
            {
                "frequency_hz": frequency_hz,
                "frequency_bin_index": frequency_bin_index,
                "n_active_ch": int(active_indices.size),
                "effective_channel_count": noise_result.effective_channel_count,
                "noise_eta_reference": noise_result.noise_eta_reference,
                "noise_eta_mean": float(calibration.noise_mean[0]),
                "gamma_off": gamma_off,
                "gamma_on": gamma_on,
                "roc_auc": float(calibration.roc_auc[0]),
                "calibration_false_positive_rate": float(calibration.calibrated_false_positive_rate[0]),
                "calibration_detection_rate": float(calibration.calibrated_detection_rate[0]),
                "evaluation_far_soft_weight_mean": float(np.mean(weight[far_from_sources])),
                "weight_sum": weight_sum,
                "effective_direction_count": float(weight_sum**2 / np.sum(weight**2)),
                "selected_covariance_health": _covariance_health(selected_covariance),
                "method2_covariance_health": _covariance_health(method2_covariance),
                "beam_metrics": beam_results,
                "adoption": {
                    "source": adopted_source,
                    "steering_selected_accepted": selection_accepted,
                    "fallback_reason": None if selection_accepted else "MVDR_RESPONSE_CONDITION_FAILED",
                    "criteria": {
                        "maximum_peak_error_deg": MAXIMUM_ACCEPTED_PEAK_ERROR_DEG,
                        "maximum_outside_peak_db_re_target": MAXIMUM_ACCEPTED_OUTSIDE_PEAK_DB,
                        "maximum_white_noise_gain_db10_re_input_channel": MAXIMUM_ACCEPTED_WHITE_NOISE_GAIN_DB,
                    },
                    "adopted_beam_metrics": adopted_metrics,
                },
                "configuration_signature": signature,
            }
        )

    payload = {
        "array_path": str(ARRAY_PATH.resolve()),
        "shading_path": str(SHADING_PATH.resolve()),
        "n_ch_total": int(array_definition.n_ch),
        "aperture_m": float(array_definition.aperture_m),
        "sample_rate_hz": fs_hz,
        "sound_speed_m_s": sound_speed_m_s,
        "calibration_duration_s": CALIBRATION_DURATION_S,
        "evaluation_duration_s": EVALUATION_DURATION_S,
        "target_azimuth_deg": TARGET_AZIMUTH_DEG,
        "interferer_azimuth_deg": INTERFERER_AZIMUTH_DEG,
        "noise_level_db_re_input_rms_per_sqrt_hz": NOISE_LEVEL_DB_RE_INPUT_RMS_PER_SQRT_HZ,
        "diagonal_loading_trace_ratio": DIAGONAL_LOADING,
        "records": records,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
