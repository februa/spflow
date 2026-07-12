"""粗い分析幅で整数遅延と方位別共分散のMVDR成立性を比較する。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming.diagnostic_plotting import require_matplotlib  # noqa: E402

OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "coarse_covariance_integer_delay_mvdr"
FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
N_CHANNEL = 64
SPACING_M = 6.25
ANALYSIS_WIDTH_HZ = 64.0
CENTER_FREQUENCY_HZ = 128.0
TARGET_AZIMUTH_DEG = 60.0
INTERFERER_AZIMUTH_DEG = 110.0
NOISE_POWER_RE_TARGET = 1.0e-2
DIAGONAL_LOADING_RATIO = 1.0e-3
SCAN_AZIMUTH_DEG = np.linspace(0.0, 180.0, 721, dtype=np.float64)
DIRECT_METHOD_ID = "T1"
INTEGER_DELAY_METHOD_ID = "T2"
METHOD_IDS = (DIRECT_METHOD_ID, INTEGER_DELAY_METHOD_ID)
SWEEP_ANALYSIS_WIDTHS_HZ = (16.0, 32.0, 64.0, 128.0)
SWEEP_TARGET_AZIMUTHS_DEG = (15.0, 30.0, 60.0, 120.0, 150.0, 165.0)


def sensor_positions_m() -> NDArray[np.float64]:
    """中心対称ULAのx座標を返す。

    Returns:
        センサ位置。shapeは`[n_ch]`、単位はm。
    """

    aperture_m = SPACING_M * (N_CHANNEL - 1)
    return np.linspace(-aperture_m / 2.0, aperture_m / 2.0, N_CHANNEL, dtype=np.float64)


def arrival_delays_s(
    positions_m: NDArray[np.float64], azimuth_deg: NDArray[np.float64]
) -> NDArray[np.float64]:
    """方位ごとの相対到来遅延を返す。

    Args:
        positions_m: ULA位置。shapeは`[n_ch]`、単位はm。
        azimuth_deg: 方位。shapeは`[n_direction]`、単位はdeg。

    Returns:
        相対到来遅延。shapeは`[n_direction,n_ch]`、単位はs。
    """

    # 0/180 degがendfire、90 degがbroadsideとなる既存方位規約に合わせる。
    return (
        np.cos(np.deg2rad(azimuth_deg))[:, np.newaxis]
        * positions_m[np.newaxis, :]
        / SOUND_SPEED_M_S
    )


def steering(
    delays_s: NDArray[np.float64], *, frequency_hz: float = CENTER_FREQUENCY_HZ
) -> NDArray[np.complex128]:
    """中心周波数のsteeringを返す。

    Args:
        delays_s: 相対遅延。shapeは`[...,n_ch]`、単位はs。
        frequency_hz: 周波数。単位はHz。

    Returns:
        複素steering。shapeは入力と同じ。
    """

    # a=exp(-j 2πfτ)とし、R=a a^Hの位相規約を全方式で共通化する。
    return np.asarray(
        np.exp(-1j * 2.0 * np.pi * frequency_hz * delays_s),
        dtype=np.complex128,
    )


def _source_covariance(
    coherence_residual_s: NDArray[np.float64],
    output_steering: NDArray[np.complex128],
    *,
    analysis_width_hz: float = ANALYSIS_WIDTH_HZ,
) -> NDArray[np.complex128]:
    """平坦1-bin広帯域信号の解析共分散を返す。

    Args:
        coherence_residual_s: 窓内に残る遅延。shapeは`[n_ch]`、単位はs。
        output_steering: 共分散を表す座標のsteering。shapeは`[n_ch]`。
        analysis_width_hz: 1-bin広帯域信号の帯域幅。単位はHz。

    Returns:
        信号共分散。shapeは`[n_ch,n_ch]`。
    """

    # 平坦な幅Δfの矩形周波数積分は、pair間残留遅延に
    # sinc(Δf(τ_i-τ_j))のcoherence低下を与える。
    pair_residual_s = coherence_residual_s[:, np.newaxis] - coherence_residual_s[np.newaxis, :]
    coherence = np.sinc(analysis_width_hz * pair_residual_s)
    source_outer = output_steering[:, np.newaxis] * output_steering.conj()[np.newaxis, :]
    return np.asarray(coherence * source_outer, dtype=np.complex128)


def method_covariances() -> dict[str, NDArray[np.complex128]]:
    """方位別時間切り出し共分散の2つの適用位相基準を返す。

    Returns:
        method IDから共分散`[n_ch,n_ch]`への対応。
    """

    positions = sensor_positions_m()
    true_delay = arrival_delays_s(
        positions, np.asarray([TARGET_AZIMUTH_DEG], dtype=np.float64)
    )[0]
    physical_steering = steering(true_delay)
    # 整数sample整相は物理遅延を1/fs格子へ丸め、中心周波数で
    # その位相を取り除く。共通遅延は共分散に影響しない。
    quantized_delay = np.rint(true_delay * FS_HZ) / FS_HZ
    integer_alignment = np.exp(1j * 2.0 * np.pi * CENTER_FREQUENCY_HZ * quantized_delay)
    residual_delay = true_delay - quantized_delay
    # target方位の時間切り出しは整数sample丸め残差だけを
    # coherenceに残すが、切り出し時刻差の位相補正後は元入力と同じ位相基準になる。
    direct_covariance = _source_covariance(residual_delay, physical_steering)
    # 整数遅延前段方式では、同じ完成共分散のchannel位相を、
    # 実際に信号へ与える整数遅延分だけ合わせ直す。共分散は再推定しない。
    integer_delay_covariance = np.asarray(
        integer_alignment[:, np.newaxis]
        * direct_covariance
        * integer_alignment.conj()[np.newaxis, :],
        dtype=np.complex128,
    )
    return {
        DIRECT_METHOD_ID: direct_covariance,
        INTEGER_DELAY_METHOD_ID: integer_delay_covariance,
    }


def _mvdr_weight(
    covariance: NDArray[np.complex128], constraint: NDArray[np.complex128]
) -> tuple[NDArray[np.complex128], dict[str, float]]:
    """対角loading付きMVDR重みと共分散品質を返す。"""

    hermitian = np.asarray(0.5 * (covariance + covariance.conj().T), dtype=np.complex128)
    trace = float(np.real(np.trace(hermitian)))
    loading = DIAGONAL_LOADING_RATIO * trace / N_CHANNEL
    loaded = hermitian + loading * np.eye(N_CHANNEL, dtype=np.complex128)
    # inv(R)を明示生成せずsolveし、悪条件時の数値誤差を抑える。
    solved = np.linalg.solve(loaded, constraint)
    denominator = np.vdot(constraint, solved)
    weight = np.asarray(solved / denominator, dtype=np.complex128)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    return weight, {
        "hermitian_relative_error": float(
            np.linalg.norm(covariance - covariance.conj().T)
            / max(float(np.linalg.norm(covariance)), np.finfo(np.float64).tiny)
        ),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "loaded_condition_number": float(np.linalg.cond(loaded)),
        "weight_norm": float(np.linalg.norm(weight)),
        "distortionless_error": float(abs(np.vdot(weight, constraint) - 1.0)),
    }


def evaluate_methods() -> tuple[list[dict[str, Any]], dict[str, NDArray[np.float64]]]:
    """方位別共分散を使う2整相方式の指標とbeam patternを返す。"""

    positions = sensor_positions_m()
    scan_delay = arrival_delays_s(positions, SCAN_AZIMUTH_DEG)
    scan_physical = steering(scan_delay)
    target_delay = arrival_delays_s(
        positions, np.asarray([TARGET_AZIMUTH_DEG], dtype=np.float64)
    )[0]
    interferer_delay = arrival_delays_s(
        positions, np.asarray([INTERFERER_AZIMUTH_DEG], dtype=np.float64)
    )[0]
    target_physical = steering(target_delay)
    interferer_physical = steering(interferer_delay)
    quantized_delay = np.rint(target_delay * FS_HZ) / FS_HZ
    integer_alignment = np.exp(1j * 2.0 * np.pi * CENTER_FREQUENCY_HZ * quantized_delay)
    covariances = method_covariances()
    rows: list[dict[str, Any]] = []
    patterns: dict[str, NDArray[np.float64]] = {}
    complex_outputs: dict[str, complex] = {}

    for method_id in METHOD_IDS:
        integer_delay_applied = method_id == INTEGER_DELAY_METHOD_ID
        constraint = (
            np.asarray(integer_alignment * target_physical, dtype=np.complex128)
            if integer_delay_applied
            else target_physical
        )
        scan = (
            np.asarray(scan_physical * integer_alignment[np.newaxis, :], dtype=np.complex128)
            if integer_delay_applied
            else scan_physical
        )
        interferer = (
            np.asarray(integer_alignment * interferer_physical, dtype=np.complex128)
            if integer_delay_applied
            else interferer_physical
        )
        weight, health = _mvdr_weight(
            covariances[method_id]
            + NOISE_POWER_RE_TARGET * np.eye(N_CHANNEL, dtype=np.complex128),
            constraint,
        )
        # beam patternは1つの保護weightを固定し、入力方位axis=0をsweepする。
        response = np.asarray(np.abs(scan.conj() @ weight), dtype=np.float64)
        response_db = 20.0 * np.log10(np.maximum(response, 1.0e-12))
        patterns[method_id] = response_db
        target_output = np.vdot(weight, constraint)
        interferer_output = np.vdot(weight, interferer)
        # np.vdotはNumPy complex scalarを返すため、後段のPython complex型辞書へ明示変換する。
        complex_outputs[method_id] = complex(target_output)
        outside = np.abs(SCAN_AZIMUTH_DEG - TARGET_AZIMUTH_DEG) >= 10.0
        peak_index = int(np.argmax(response))
        rows.append(
            {
                "method": method_id,
                "evaluation_pattern": "slc_same_frequency_interference",
                "analysis_width_hz": ANALYSIS_WIDTH_HZ,
                "center_frequency_hz": CENTER_FREQUENCY_HZ,
                "target_azimuth_deg": TARGET_AZIMUTH_DEG,
                "interferer_azimuth_deg": INTERFERER_AZIMUTH_DEG,
                "target_level_db_re_input_rms": float(
                    20.0 * np.log10(max(abs(target_output), np.finfo(np.float64).tiny))
                ),
                "peak_azimuth_deg": float(SCAN_AZIMUTH_DEG[peak_index]),
                "peak_error_deg": float(abs(SCAN_AZIMUTH_DEG[peak_index] - TARGET_AZIMUTH_DEG)),
                "guard_outside_peak_db_re_target": float(np.max(response_db[outside])),
                "interferer_leakage_db_re_target": float(
                    20.0 * np.log10(max(abs(interferer_output), np.finfo(np.float64).tiny))
                ),
                "noise_power_re_input_channel": float(
                    NOISE_POWER_RE_TARGET * np.sum(np.abs(weight) ** 2)
                ),
                **health,
            }
        )

    # 両方式は同じ完成共分散と整数遅延分の位相補正だけが異なるため、
    # 整合した信号とweightを組み合わせればtarget複素出力は一致する。
    equivalence_error = abs(
        complex_outputs[DIRECT_METHOD_ID] - complex_outputs[INTEGER_DELAY_METHOD_ID]
    )
    for row in rows:
        row["direct_integer_delay_target_complex_error"] = float(equivalence_error)
    return rows, patterns


def _pattern_metrics(
    weight: NDArray[np.complex128],
    scan_steering: NDArray[np.complex128],
    target_azimuth_deg: float,
) -> dict[str, float]:
    """固定weight beam patternのpeak誤差とguard外peakを返す。

    Args:
        weight: MVDR重み。shapeは`[n_ch]`。
        scan_steering: 入力方位sweepのsteering。shapeは`[n_direction,n_ch]`。
        target_azimuth_deg: 保護方位。単位はdeg。

    Returns:
        peak方位誤差とguard外peak。角度はdeg、levelは`dB re target response`。
    """

    response = np.asarray(np.abs(scan_steering.conj() @ weight), dtype=np.float64)
    response_db = 20.0 * np.log10(np.maximum(response, 1.0e-12))
    peak_index = int(np.argmax(response))
    outside = np.abs(SCAN_AZIMUTH_DEG - target_azimuth_deg) >= 10.0
    return {
        "peak_azimuth_deg": float(SCAN_AZIMUTH_DEG[peak_index]),
        "peak_error_deg": float(abs(SCAN_AZIMUTH_DEG[peak_index] - target_azimuth_deg)),
        "guard_outside_peak_db_re_target": float(np.max(response_db[outside])),
    }


def evaluate_failure_condition_sweep() -> list[dict[str, Any]]:
    """通常共分散が破綻する複数条件で2整相方式を評価する。

    Returns:
        分析幅・target方位ごとの評価行。角度はdeg、周波数はHz。

    Notes:
        同一時間block共分散は整相候補ではなく、試験条件が実際に
        破綻条件であることを確認する参照値としてのみ使う。
    """

    positions = sensor_positions_m()
    scan_delay = arrival_delays_s(positions, SCAN_AZIMUTH_DEG)
    scan_physical = steering(scan_delay)
    identity = np.eye(N_CHANNEL, dtype=np.complex128)
    rows: list[dict[str, Any]] = []
    for analysis_width_hz in SWEEP_ANALYSIS_WIDTHS_HZ:
        for target_azimuth_deg in SWEEP_TARGET_AZIMUTHS_DEG:
            target_delay = arrival_delays_s(
                positions, np.asarray([target_azimuth_deg], dtype=np.float64)
            )[0]
            target_steering = steering(target_delay)
            quantized_delay = np.rint(target_delay * FS_HZ) / FS_HZ
            residual_delay = target_delay - quantized_delay
            integer_alignment = np.exp(
                1j * 2.0 * np.pi * CENTER_FREQUENCY_HZ * quantized_delay
            )

            # 参照は同一時間blockの全到来遅延が粗いbin内に残る条件である。
            reference_covariance = _source_covariance(
                target_delay,
                target_steering,
                analysis_width_hz=analysis_width_hz,
            )
            reference_weight, _ = _mvdr_weight(
                reference_covariance + NOISE_POWER_RE_TARGET * identity,
                target_steering,
            )
            reference_metrics = _pattern_metrics(
                reference_weight, scan_physical, target_azimuth_deg
            )

            # 直接MVDRは切り出し時刻差の位相補正後の共分散を
            # 元の多channel入力と同じ位相基準で使う。
            direct_covariance = _source_covariance(
                residual_delay,
                target_steering,
                analysis_width_hz=analysis_width_hz,
            )
            direct_weight, _ = _mvdr_weight(
                direct_covariance + NOISE_POWER_RE_TARGET * identity,
                target_steering,
            )
            direct_metrics = _pattern_metrics(
                direct_weight, scan_physical, target_azimuth_deg
            )

            # 整数遅延前段方式は、同じ完成共分散とsteeringに
            # 整数遅延分のchannel位相を与え、遅延後入力と整合させる。
            integer_covariance = np.asarray(
                integer_alignment[:, np.newaxis]
                * direct_covariance
                * integer_alignment.conj()[np.newaxis, :],
                dtype=np.complex128,
            )
            integer_constraint = np.asarray(
                integer_alignment * target_steering, dtype=np.complex128
            )
            integer_scan = np.asarray(
                scan_physical * integer_alignment[np.newaxis, :], dtype=np.complex128
            )
            integer_weight, _ = _mvdr_weight(
                integer_covariance + NOISE_POWER_RE_TARGET * identity,
                integer_constraint,
            )
            integer_metrics = _pattern_metrics(
                integer_weight, integer_scan, target_azimuth_deg
            )
            reference_failed = bool(
                reference_metrics["peak_error_deg"] > 2.0
                or reference_metrics["guard_outside_peak_db_re_target"] >= 0.0
            )
            direct_passed = bool(
                direct_metrics["peak_error_deg"] <= 2.0
                and direct_metrics["guard_outside_peak_db_re_target"] < 0.0
            )
            integer_passed = bool(
                integer_metrics["peak_error_deg"] <= 2.0
                and integer_metrics["guard_outside_peak_db_re_target"] < 0.0
            )
            rows.append(
                {
                    "analysis_width_hz": analysis_width_hz,
                    "center_frequency_hz": CENTER_FREQUENCY_HZ,
                    "target_azimuth_deg": target_azimuth_deg,
                    "same_time_reference_failed": reference_failed,
                    "same_time_reference_peak_error_deg": reference_metrics["peak_error_deg"],
                    "same_time_reference_guard_outside_peak_db_re_target": reference_metrics[
                        "guard_outside_peak_db_re_target"
                    ],
                    "direct_mvdr_passed": direct_passed,
                    "direct_mvdr_peak_error_deg": direct_metrics["peak_error_deg"],
                    "direct_mvdr_guard_outside_peak_db_re_target": direct_metrics[
                        "guard_outside_peak_db_re_target"
                    ],
                    "integer_delay_mvdr_passed": integer_passed,
                    "integer_delay_mvdr_peak_error_deg": integer_metrics["peak_error_deg"],
                    "integer_delay_mvdr_guard_outside_peak_db_re_target": integer_metrics[
                        "guard_outside_peak_db_re_target"
                    ],
                    "direct_integer_delay_pattern_max_abs_error_db": float(
                        max(
                            abs(
                                direct_metrics["guard_outside_peak_db_re_target"]
                                - integer_metrics["guard_outside_peak_db_re_target"]
                            ),
                            abs(
                                direct_metrics["peak_error_deg"]
                                - integer_metrics["peak_error_deg"]
                            ),
                        )
                    ),
                }
            )
    return rows


def _write_artifacts(
    rows: list[dict[str, Any]], patterns: dict[str, NDArray[np.float64]]
) -> None:
    """比較CSV、NPZ、図、日本語review indexを保存する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "scenario_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    # np.savezの可変長keyword型stubがndarray値をallow_pickleと誤推論しないよう、
    # 成果物境界の辞書だけをAnyとして、配列keyを保持する。
    plot_arrays: dict[str, Any] = {"azimuth_deg": SCAN_AZIMUTH_DEG}
    for method_id in METHOD_IDS:
        plot_arrays[f"{method_id.lower()}_beam_pattern_db_re_target"] = patterns[method_id]
    np.savez(OUTPUT_DIR / "plot_data.npz", **plot_arrays)
    plt = require_matplotlib()
    figure, axis = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    for method_id in METHOD_IDS:
        axis.plot(SCAN_AZIMUTH_DEG, patterns[method_id], label=method_id)
    axis.axvline(TARGET_AZIMUTH_DEG, color="tab:green", linestyle="--", label="target")
    axis.axvline(INTERFERER_AZIMUTH_DEG, color="tab:red", linestyle=":", label="interferer")
    axis.set(
        title="Frozen-weight beam pattern: coarse covariance comparison",
        xlabel="Input azimuth [deg]",
        ylabel="Response [dB re target response]",
        xlim=(0.0, 180.0),
        ylim=(-80.0, 5.0),
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(OUTPUT_DIR / "beam_pattern_overlay.png", dpi=160)
    plt.close(figure)
    payload = {
        "scenario": "representative_coarse_bin",
        "analysis_width_hz": ANALYSIS_WIDTH_HZ,
        "center_frequency_hz": CENTER_FREQUENCY_HZ,
        "rows": rows,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review = "\n".join(
        (
            "# 粗い分析幅の整相方式 代表条件評価",
            "",
            f"- 分析幅: `{ANALYSIS_WIDTH_HZ:g} Hz`、中心周波数: `{CENTER_FREQUENCY_HZ:g} Hz`。",
            (
                f"- target: `{TARGET_AZIMUTH_DEG:g} deg`、"
                f"interferer: `{INTERFERER_AZIMUTH_DEG:g} deg`。"
            ),
            "- 図は保護weightを固定したbeam patternであり、横軸は入力方位。BLではない。",
            "- levelは`dB re target response`、target出力は`dB re input RMS`。",
            "- `scenario_summary.csv`が数値根拠、`plot_data.npz`が図の元配列。",
            (
                "- この代表条件は直接適用と整数遅延前段の位相整合を検証するものであり、"
                "運用採否やBL複合scoreの根拠には使用しない。"
            ),
            "",
        )
    )
    (OUTPUT_DIR / "review_index.md").write_text(review, encoding="utf-8")


def main() -> None:
    """方位別共分散を使う2整相方式の評価と成果物生成を実行する。"""

    rows, patterns = evaluate_methods()
    _write_artifacts(rows, patterns)
    sweep_rows = evaluate_failure_condition_sweep()
    with (OUTPUT_DIR / "failure_condition_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)


if __name__ == "__main__":
    main()
