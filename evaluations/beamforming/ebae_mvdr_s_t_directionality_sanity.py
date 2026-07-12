"""狭帯域toneと1-bin広帯域でS/T共分散の方位推定差を評価する。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from spflow.beamforming.ebae import EbaeConfig, design_ebae_weights_band


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


OUTPUT_DIR = Path("artifacts/beamforming/ebae_mvdr_s_t_directionality_sanity/review_pack")
SCENARIO_ID = "long_ula_tone_and_flat_bin_endfire"
FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
N_CHANNEL = 64
SPACING_M = 6.25
TARGET_AZIMUTH_DEG = 0.0
CENTER_FREQUENCY_HZ = 100.0
ANALYSIS_WIDTH_HZ = 64.0
NOISE_POWER_RE_SOURCE = 1.0e-2
MVDR_DIAGONAL_LOADING_RATIO = 1.0e-3
AZIMUTH_DEG = np.arange(0.0, 180.0 + 0.5, 0.5, dtype=np.float64)
FAR_GUARD_DEG = 20.0
DISPLAY_FLOOR_DB_RE_PEAK = -100.0
SIGNAL_TYPE_IDS = ("bin_center_tone", "flat_one_bin_broadband")
ALGORITHM_IDS = ("ebae_music", "mvdr_capon")
METHOD_IDS = ("S", "T")


@dataclass(frozen=True)
class DirectionalityResult:
    """S/T共分散のEBAE MUSIC・MVDR Capon方位推定結果を保持する。

    Attributes:
        curves_db: ``signal_type -> algorithm -> method -> curve``。各curve shapeは
            ``[n_beam]``、levelはdB re curve peak。
        rows: 方式別の方位peak、幅、margin、EBAE信号数を含むCSV行。
    """

    curves_db: dict[str, dict[str, dict[str, FloatArray]]]
    rows: tuple[dict[str, Any], ...]


def _positions_m() -> FloatArray:
    """中心基準の64ch ULA位置を返す。

    Returns:
        sensor位置。shapeは``[n_ch]``、単位はm。
    """
    aperture_m = SPACING_M * (N_CHANNEL - 1)
    return np.linspace(-aperture_m / 2.0, aperture_m / 2.0, N_CHANNEL, dtype=np.float64)


def _arrival_delays_s(azimuth_deg: FloatArray) -> FloatArray:
    """方位ごとの到来遅延を返す。

    Args:
        azimuth_deg: 方位。shapeは``[n_direction]``、単位はdeg。

    Returns:
        到来遅延。shapeは``[n_direction,n_ch]``、単位はs。
    """
    direction_cosine = np.cos(np.deg2rad(azimuth_deg))
    # tau=-r cos(theta)/c。sourceとscanで同じ符号規約を使う。
    return np.asarray(
        -direction_cosine[:, np.newaxis] * _positions_m()[np.newaxis, :] / SOUND_SPEED_M_S,
        dtype=np.float64,
    )


def _steering(delays_s: FloatArray) -> ComplexArray:
    """中心周波数の未正規化steeringを返す。

    Args:
        delays_s: 到来遅延。shapeは``[n_direction,n_ch]``、単位はs。

    Returns:
        steering。shapeは``[n_ch,n_direction]``。
    """
    return np.asarray(
        np.exp(-1j * 2.0 * np.pi * CENTER_FREQUENCY_HZ * delays_s).T,
        dtype=np.complex128,
    )


def _covariance(
    true_delay_s: FloatArray,
    source_steering: ComplexArray,
    *,
    candidate_delay_s: FloatArray | None,
    tone: bool,
) -> ComplexArray:
    """Sまたは候補方位別T共分散を返す。

    Args:
        true_delay_s: source到来遅延。shapeは``[n_ch]``、単位はs。
        source_steering: source steering。shapeは``[n_ch]``。
        candidate_delay_s: Tの候補方位遅延。shapeは``[n_ch]``。Sでは``None``。
        tone: bin中心toneならTrue、平坦1-bin広帯域ならFalse。

    Returns:
        空間共分散。shapeは``[n_ch,n_ch]``。
    """
    if candidate_delay_s is None:
        residual_delay_s = true_delay_s
    else:
        # Tは候補方位のchannel別切り出し時刻を整数sampleへ量子化し、残留遅延を使う。
        quantized_candidate_s = np.rint(candidate_delay_s * FS_HZ) / FS_HZ
        residual_delay_s = true_delay_s - quantized_candidate_s
    pair_residual_s = residual_delay_s[:, np.newaxis] - residual_delay_s[np.newaxis, :]
    # bin中心toneは全pair coherence=1。平坦1-bin広帯域だけsinc(ΔfΔtau)低下を持つ。
    coherence = (
        np.ones(pair_residual_s.shape, dtype=np.float64)
        if tone
        else np.sinc(ANALYSIS_WIDTH_HZ * pair_residual_s)
    )
    source_outer = source_steering[:, np.newaxis] * source_steering.conj()[np.newaxis, :]
    return np.asarray(
        coherence * source_outer
        + NOISE_POWER_RE_SOURCE * np.eye(N_CHANNEL, dtype=np.complex128),
        dtype=np.complex128,
    )


def _loaded_inverse(covariance: ComplexArray) -> ComplexArray:
    """trace比例loading後の共分散逆行列を返す。"""
    hermitian = np.asarray(0.5 * (covariance + covariance.conj().T), dtype=np.complex128)
    average_power = float(np.real(np.trace(hermitian))) / float(N_CHANNEL)
    loaded = hermitian + MVDR_DIAGONAL_LOADING_RATIO * average_power * np.eye(
        N_CHANNEL, dtype=np.complex128
    )
    return np.asarray(np.linalg.inv(loaded), dtype=np.complex128)


def _mvdr_capon_s_curve(covariance: ComplexArray, steering: ComplexArray) -> FloatArray:
    """共通S共分散からCapon scanを計算する。"""
    inverse = _loaded_inverse(covariance)
    denominator = np.real(
        np.einsum("cd,ce,ed->d", steering.conj(), inverse, steering, optimize=True)
    )
    return np.asarray(1.0 / np.maximum(denominator, np.finfo(np.float64).tiny), dtype=np.float64)


def _mvdr_capon_t_curve(
    true_delay_s: FloatArray,
    source_steering: ComplexArray,
    scan_delays_s: FloatArray,
    steering: ComplexArray,
    *,
    tone: bool,
) -> FloatArray:
    """候補方位別T共分散からCapon scanを計算する。"""
    curve = np.empty(AZIMUTH_DEG.size, dtype=np.float64)
    for beam_index in range(AZIMUTH_DEG.size):
        covariance = _covariance(
            true_delay_s,
            source_steering,
            candidate_delay_s=scan_delays_s[beam_index],
            tone=tone,
        )
        inverse = _loaded_inverse(covariance)
        constraint = steering[:, beam_index]
        denominator = float(np.real(np.vdot(constraint, inverse @ constraint)))
        curve[beam_index] = 1.0 / max(denominator, np.finfo(np.float64).tiny)
    return curve


def _ebae_result(covariance: ComplexArray, steering: ComplexArray):
    """N/E AICとMUSICを含む単一共分散EBAE結果を返す。"""
    return design_ebae_weights_band(
        covariance,
        steering,
        snapshot_count=N_CHANNEL * N_CHANNEL,
        config=EbaeConfig(
            snapshot_rate_hz=float(N_CHANNEL * N_CHANNEL),
            integration_time_sec=1.0,
            sigmoid_slope=10.0,
            sigmoid_midpoint=0.5,
            diagonal_loading=1.0,
        ),
    )


def _ebae_music_t_curve(
    true_delay_s: FloatArray,
    source_steering: ComplexArray,
    scan_delays_s: FloatArray,
    steering: ComplexArray,
    *,
    tone: bool,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """候補方位別T共分散からcandidate一致MUSIC値と信号数を返す。"""
    curve = np.empty(AZIMUTH_DEG.size, dtype=np.float64)
    signal_counts = np.empty(AZIMUTH_DEG.size, dtype=np.int64)
    for beam_index in range(AZIMUTH_DEG.size):
        covariance = _covariance(
            true_delay_s,
            source_steering,
            candidate_delay_s=scan_delays_s[beam_index],
            tone=tone,
        )
        result = _ebae_result(covariance, steering)
        # 候補方位別共分散の評価値は、その共分散を作った同じcandidate beamのMUSIC値とする。
        curve[beam_index] = float(result.music_spectrum[beam_index])
        signal_counts[beam_index] = result.signal_count
    return curve, signal_counts


def _normalize_curve_db(curve: FloatArray) -> FloatArray:
    """非負scan curveをpeak基準dBへ変換する。"""
    finite = np.asarray(curve, dtype=np.float64)
    if bool(np.any(np.isposinf(finite))):
        # 理想toneで雑音部分空間と厳密直交する点は+infとなるため、その点だけ0 dBにする。
        normalized = np.zeros_like(finite)
        normalized[np.isposinf(finite)] = 1.0
    else:
        peak = float(np.max(finite))
        normalized = finite / max(peak, np.finfo(np.float64).tiny)
    floor_power = 10.0 ** (DISPLAY_FLOOR_DB_RE_PEAK / 10.0)
    return np.asarray(10.0 * np.log10(np.maximum(normalized, floor_power)), dtype=np.float64)


def _curve_metrics(
    signal_type: str,
    algorithm: str,
    method: str,
    curve_db: FloatArray,
    *,
    signal_count: int,
) -> dict[str, Any]:
    """方位scanのpeak誤差、遠方margin、半高幅を返す。"""
    source_index = int(np.argmin(np.abs(AZIMUTH_DEG - TARGET_AZIMUTH_DEG)))
    peak_index = int(np.argmax(curve_db))
    far_mask = np.abs(AZIMUTH_DEG - TARGET_AZIMUTH_DEG) >= FAR_GUARD_DEG
    half_level_db = -3.0
    left = source_index
    right = source_index
    while left > 0 and curve_db[left - 1] >= half_level_db:
        left -= 1
    while right + 1 < curve_db.size and curve_db[right + 1] >= half_level_db:
        right += 1
    return {
        "scenario": SCENARIO_ID,
        "signal_type": signal_type,
        "algorithm": algorithm,
        "method": method,
        "source_azimuth_deg": TARGET_AZIMUTH_DEG,
        "peak_azimuth_deg": float(AZIMUTH_DEG[peak_index]),
        "peak_error_deg": float(abs(AZIMUTH_DEG[peak_index] - TARGET_AZIMUTH_DEG)),
        "source_to_far_peak_margin_db": float(curve_db[source_index] - np.max(curve_db[far_mask])),
        "three_db_width_deg": float(AZIMUTH_DEG[right] - AZIMUTH_DEG[left]),
        "source_count_at_source_candidate": signal_count,
    }


def calculate_directionality_sanity() -> DirectionalityResult:
    """狭帯域toneと1-bin広帯域のS/T方位scanを計算する。

    Returns:
        EBAE MUSIC、MVDR CaponのS/T curveと方式別指標。
    """
    scan_delays_s = _arrival_delays_s(AZIMUTH_DEG)
    source_index = int(np.argmin(np.abs(AZIMUTH_DEG - TARGET_AZIMUTH_DEG)))
    true_delay_s = scan_delays_s[source_index]
    steering = _steering(scan_delays_s)
    source_steering = steering[:, source_index]
    curves: dict[str, dict[str, dict[str, FloatArray]]] = {}
    rows: list[dict[str, Any]] = []
    for signal_type in SIGNAL_TYPE_IDS:
        tone = signal_type == "bin_center_tone"
        s_covariance = _covariance(
            true_delay_s,
            source_steering,
            candidate_delay_s=None,
            tone=tone,
        )
        ebae_s = _ebae_result(s_covariance, steering)
        ebae_t_curve, ebae_t_counts = _ebae_music_t_curve(
            true_delay_s,
            source_steering,
            scan_delays_s,
            steering,
            tone=tone,
        )
        linear_curves = {
            "ebae_music": {
                "S": np.asarray(ebae_s.music_spectrum, dtype=np.float64),
                "T": ebae_t_curve,
            },
            "mvdr_capon": {
                "S": _mvdr_capon_s_curve(s_covariance, steering),
                "T": _mvdr_capon_t_curve(
                    true_delay_s,
                    source_steering,
                    scan_delays_s,
                    steering,
                    tone=tone,
                ),
            },
        }
        curves[signal_type] = {
            algorithm: {
                method: _normalize_curve_db(linear_curves[algorithm][method])
                for method in METHOD_IDS
            }
            for algorithm in ALGORITHM_IDS
        }
        for algorithm in ALGORITHM_IDS:
            for method in METHOD_IDS:
                signal_count = -1
                if algorithm == "ebae_music":
                    signal_count = (
                        ebae_s.signal_count if method == "S" else int(ebae_t_counts[source_index])
                    )
                rows.append(
                    _curve_metrics(
                        signal_type,
                        algorithm,
                        method,
                        curves[signal_type][algorithm][method],
                        signal_count=signal_count,
                    )
                )
    return DirectionalityResult(curves, tuple(rows))


def write_directionality_report(output_dir: Path = OUTPUT_DIR) -> DirectionalityResult:
    """S/T方位推定sanityのCSV、NPZ、PNG、Markdownを保存する。

    Args:
        output_dir: review pack出力先。

    Returns:
        保存した方位推定結果。
    """
    result = calculate_directionality_sanity()
    figure_dir = output_dir / "figures" / SCENARIO_ID
    data_dir = output_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scenario_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.rows[0].keys()))
        writer.writeheader()
        writer.writerows(result.rows)

    arrays: dict[str, FloatArray] = {"azimuth_deg": AZIMUTH_DEG}
    for signal_type in SIGNAL_TYPE_IDS:
        for algorithm in ALGORITHM_IDS:
            for method in METHOD_IDS:
                arrays[f"{signal_type}_{algorithm}_{method}_db_re_peak"] = result.curves_db[
                    signal_type
                ][algorithm][method]
    np.savez(data_dir / f"{SCENARIO_ID}.npz", **arrays)  # pyright: ignore[reportArgumentType]

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True, sharex=True)
    for row_index, algorithm in enumerate(ALGORITHM_IDS):
        for column_index, signal_type in enumerate(SIGNAL_TYPE_IDS):
            axis = axes[row_index, column_index]
            for method in METHOD_IDS:
                axis.plot(
                    AZIMUTH_DEG,
                    result.curves_db[signal_type][algorithm][method],
                    label=method,
                )
            axis.axvline(TARGET_AZIMUTH_DEG, color="black", linestyle="--", linewidth=1.0)
            axis.set(
                title=f"{algorithm} / {signal_type}",
                xlabel="Candidate azimuth [deg]",
                ylabel="Spatial spectrum [dB re peak]" if column_index == 0 else None,
                xlim=(0.0, 180.0),
                ylim=(DISPLAY_FLOOR_DB_RE_PEAK, 5.0),
            )
            axis.grid(True, alpha=0.25)
            axis.legend()
    figure.savefig(figure_dir / "source_frequency_bl_overlay.png", dpi=160)
    plt.close(figure)

    lines = (
        "# EBAE/MVDR S/T方位推定sanity",
        "",
        f"- scenario: `{SCENARIO_ID}`",
        f"- source: {TARGET_AZIMUTH_DEG:.1f} deg, center {CENTER_FREQUENCY_HZ:.1f} Hz",
        f"- flat broadband analysis width: {ANALYSIS_WIDTH_HZ:.1f} Hz",
        f"- array: {N_CHANNEL} ch, spacing {SPACING_M:.2f} m",
        "- S: 同一時間block共分散",
        "- T: 候補方位別時間切り出し共分散",
        "",
        "toneではS/Tとも正方位を推定し、平坦1-bin広帯域ではSが破綻してTが正方位を維持するかを確認する。",
        "本評価は完成共分散の方位選択性を対象とし、FIR tap長は含めない。",
        "",
        f"- figure: `figures/{SCENARIO_ID}/source_frequency_bl_overlay.png`",
        f"- data: `data/{SCENARIO_ID}.npz`",
        "- metrics: `scenario_summary.csv`",
    )
    (output_dir / "review_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """既定条件でS/T方位推定sanity成果物を生成する。"""
    write_directionality_report()


if __name__ == "__main__":
    main()
