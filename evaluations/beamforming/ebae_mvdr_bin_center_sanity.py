"""bin中心・beam直上の単一信号でEBAEとMVDRの基本応答を比較する。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from spflow.beamforming.ebae import EbaeConfig, design_ebae_weights_band
from spflow.beamforming.mvdr_weight_designer import design_mvdr_weights


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


OUTPUT_DIR = Path("artifacts/beamforming/ebae_mvdr_bin_center_sanity/review_pack")
SCENARIO_ID = "single_source_bin_center_beam_center"
SOUND_SPEED_M_S = 1500.0
SAMPLE_RATE_HZ = 8000.0
FFT_SIZE = 256
TARGET_BIN_INDEX = 32
TARGET_FREQUENCY_HZ = SAMPLE_RATE_HZ * TARGET_BIN_INDEX / FFT_SIZE
TARGET_AZIMUTH_DEG = 60.0
N_CHANNEL = 8
NOISE_POWER_RE_INPUT_RMS2_PER_CHANNEL = 0.1
SOURCE_POWER_RE_INPUT_RMS2 = 1.0
DISPLAY_FLOOR_DB_RE_INPUT_RMS = -100.0
GUARD_HALF_WIDTH_DEG = 10.0


@dataclass(frozen=True)
class EbaeMvdrSanityResult:
    """EBAEとMVDRの単一条件比較結果を保持する。

    入力はbin中心・beam直上の単一sourceと空間白色雑音であり、出力は両方式の
    target-only、noise-only、mixed BLと主要比較指標である。FIR化、S1/S2/T1/T2、
    周波数sweep、BTRは責務に含めない。

    Attributes:
        azimuth_deg: 待受beam方位。shapeは``[n_beam]``、単位はdeg。
        ebae_target_bl_db: EBAE target-only BL。shapeは``[n_beam]``、dB re input RMS。
        mvdr_target_bl_db: MVDR target-only BL。shapeは``[n_beam]``、dB re input RMS。
        ebae_noise_bl_db: EBAE noise-only BL。shapeは``[n_beam]``、dB re input RMS。
        mvdr_noise_bl_db: MVDR noise-only BL。shapeは``[n_beam]``、dB re input RMS。
        ebae_mixed_bl_db: EBAE mixed BL。shapeは``[n_beam]``、dB re input RMS。
        mvdr_mixed_bl_db: MVDR mixed BL。shapeは``[n_beam]``、dB re input RMS。
        summary_rows: CSVへ保存する方式別指標。
        signal_count: EBAEのN/E AIC推定信号数。
        associated_azimuth_deg: EBAEが対応付けたMUSIC方位。単位はdeg。
    """

    azimuth_deg: FloatArray
    ebae_target_bl_db: FloatArray
    mvdr_target_bl_db: FloatArray
    ebae_noise_bl_db: FloatArray
    mvdr_noise_bl_db: FloatArray
    ebae_mixed_bl_db: FloatArray
    mvdr_mixed_bl_db: FloatArray
    summary_rows: tuple[dict[str, Any], ...]
    signal_count: int
    associated_azimuth_deg: float


def _steering_matrix(azimuth_deg: FloatArray) -> ComplexArray:
    """半波長間隔ULAの未正規化steeringを返す。

    Args:
        azimuth_deg: 方位。shapeは``[n_beam]``、単位はdeg。0/180 degがendfire、
            90 degがbroadsideである。

    Returns:
        未正規化steering。shapeは``[n_ch,n_beam]``。
    """
    wavelength_m = SOUND_SPEED_M_S / TARGET_FREQUENCY_HZ
    spacing_m = wavelength_m / 2.0
    positions_m = np.arange(N_CHANNEL, dtype=np.float64) * spacing_m
    direction_cosine = np.cos(np.deg2rad(azimuth_deg))
    # tau[ch,beam]=position[ch]*cos(theta)/c は基準sensorに対する到来遅延で、単位はs。
    delays_s = positions_m[:, np.newaxis] * direction_cosine[np.newaxis, :] / SOUND_SPEED_M_S
    # a=exp(-j2πf tau)により、bin中心周波数でのchannel間位相を表す。
    return np.asarray(
        np.exp(-1j * 2.0 * np.pi * TARGET_FREQUENCY_HZ * delays_s),
        dtype=np.complex128,
    )


def _amplitude_level_db(amplitude: FloatArray) -> FloatArray:
    """線形RMS振幅を``dB re input RMS``へ変換する。

    Args:
        amplitude: 線形RMS振幅。任意の1次元shape。

    Returns:
        表示床を適用した同じshapeのlevel。単位はdB re input RMS。
    """
    floor_amplitude = 10.0 ** (DISPLAY_FLOOR_DB_RE_INPUT_RMS / 20.0)
    return np.asarray(20.0 * np.log10(np.maximum(amplitude, floor_amplitude)), dtype=np.float64)


def _power_level_db(power: FloatArray) -> FloatArray:
    """線形RMS powerを``dB re input RMS``へ変換する。

    Args:
        power: 非負power。任意の1次元shape、基準はinput RMS二乗。

    Returns:
        表示床を適用した同じshapeのlevel。単位はdB re input RMS。
    """
    floor_power = 10.0 ** (DISPLAY_FLOOR_DB_RE_INPUT_RMS / 10.0)
    return np.asarray(10.0 * np.log10(np.maximum(power, floor_power)), dtype=np.float64)


def _method_metrics(
    method: str,
    weights: ComplexArray,
    source_steering: ComplexArray,
    azimuth_deg: FloatArray,
) -> tuple[dict[str, Any], FloatArray, FloatArray, FloatArray]:
    """単一方式の成分別BLと指標を計算する。

    Args:
        method: 方式ID。
        weights: beam重み。shapeは``[n_ch,n_beam]``。
        source_steering: source steering。shapeは``[n_ch]``。
        azimuth_deg: 待受beam方位。shapeは``[n_beam]``、単位はdeg。

    Returns:
        ``(summary, target_bl, noise_bl, mixed_bl)``。各BL shapeは``[n_beam]``、
        levelはdB re input RMS。
    """
    # response[beam]=w(theta_b)^H a(theta_source)。source RMS=1なので振幅が出力RMSに等しい。
    response = np.asarray(weights.conj().T @ source_steering, dtype=np.complex128)
    target_power = SOURCE_POWER_RE_INPUT_RMS2 * np.abs(response) ** 2
    # 空間白色雑音ではw^H(sigma^2 I)w=sigma^2 sum_ch|w_ch|^2となる。
    noise_power = NOISE_POWER_RE_INPUT_RMS2_PER_CHANNEL * np.sum(np.abs(weights) ** 2, axis=0)
    mixed_power = target_power + noise_power
    target_bl_db = _power_level_db(np.asarray(target_power, dtype=np.float64))
    noise_bl_db = _power_level_db(np.asarray(noise_power, dtype=np.float64))
    mixed_bl_db = _power_level_db(np.asarray(mixed_power, dtype=np.float64))

    target_index = int(np.argmin(np.abs(azimuth_deg - TARGET_AZIMUTH_DEG)))
    peak_index = int(np.argmax(target_bl_db))
    guard_mask = np.abs(azimuth_deg - TARGET_AZIMUTH_DEG) > GUARD_HALF_WIDTH_DEG
    summary: dict[str, Any] = {
        "scenario": SCENARIO_ID,
        "method": method,
        "evaluation_pattern": "fixed_beam_single_source",
        "target_frequency_hz": TARGET_FREQUENCY_HZ,
        "target_azimuth_deg": TARGET_AZIMUTH_DEG,
        "target_level_db_re_input_rms": float(target_bl_db[target_index]),
        "target_peak_azimuth_deg": float(azimuth_deg[peak_index]),
        "target_peak_error_deg": float(abs(azimuth_deg[peak_index] - TARGET_AZIMUTH_DEG)),
        "guard_outside_peak_db_re_input_rms": float(np.max(target_bl_db[guard_mask])),
        "noise_target_beam_db_re_input_rms": float(noise_bl_db[target_index]),
        "mixed_target_beam_db_re_input_rms": float(mixed_bl_db[target_index]),
        "distortionless_error": float(abs(response[target_index] - 1.0)),
        "weight_norm_target_beam": float(np.linalg.norm(weights[:, target_index])),
    }
    return summary, target_bl_db, noise_bl_db, mixed_bl_db


def calculate_ebae_mvdr_bin_center_sanity() -> EbaeMvdrSanityResult:
    """bin中心・beam直上の単一信号でEBAEとMVDRを比較する。

    Returns:
        成分別BL、N/E AIC結果、MUSIC対応方位、方式別指標。

    Raises:
        RuntimeError: EBAEが信号数1またはtarget方位を推定できない場合。
    """
    azimuth_deg = np.arange(0.0, 181.0, 1.0, dtype=np.float64)
    steering = _steering_matrix(azimuth_deg)
    target_index = int(np.argmin(np.abs(azimuth_deg - TARGET_AZIMUTH_DEG)))
    source_steering = steering[:, target_index]
    # R=a a^H+sigma_n^2 I。sourceはbin中心・beam直上なので狭帯域steeringと厳密に一致する。
    covariance = np.asarray(
        SOURCE_POWER_RE_INPUT_RMS2 * np.outer(source_steering, source_steering.conj())
        + NOISE_POWER_RE_INPUT_RMS2_PER_CHANNEL * np.eye(N_CHANNEL, dtype=np.complex128),
        dtype=np.complex128,
    )

    # N/E AIC契約L=rate*T=M^2を、rate=64 snapshot/s、T=1 sで満たす。
    ebae_result = design_ebae_weights_band(
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
    if ebae_result.signal_count != 1:
        raise RuntimeError(f"Expected one signal, detected {ebae_result.signal_count}.")
    associated_index = int(ebae_result.associated_beam_indices[0])
    if associated_index != target_index:
        raise RuntimeError(
            f"Expected associated beam {target_index}, detected {associated_index}."
        )

    # MVDRは同じ完成共分散を使い、追加loadingなしで解析共分散の基準解を作る。
    # 共分散にはsigma_n^2 Iが含まれ正定値なので、数値安定化loadingは不要である。
    mvdr_weights = np.asarray(
        design_mvdr_weights(covariance, steering, diag_load=0.0),
        dtype=np.complex128,
    )
    # EBAE公開結果を本評価のcomplex128規約へ揃え、方式間の丸め差を比較へ混ぜない。
    ebae_weights = np.asarray(ebae_result.weights, dtype=np.complex128)
    ebae_summary, ebae_target, ebae_noise, ebae_mixed = _method_metrics(
        "ebae_dl1", ebae_weights, source_steering, azimuth_deg
    )
    mvdr_summary, mvdr_target, mvdr_noise, mvdr_mixed = _method_metrics(
        "mvdr", mvdr_weights, source_steering, azimuth_deg
    )

    comparison_mask = (ebae_target > DISPLAY_FLOOR_DB_RE_INPUT_RMS) & (
        mvdr_target > DISPLAY_FLOOR_DB_RE_INPUT_RMS
    )
    target_bl_delta = ebae_target[comparison_mask] - mvdr_target[comparison_mask]
    ebae_summary.update(
        {
            "source_count_detected": ebae_result.signal_count,
            "music_associated_azimuth_deg": float(azimuth_deg[associated_index]),
            "fallback_required": ebae_result.used_fallback,
            "target_bl_rms_delta_db_re_mvdr": float(
                np.sqrt(np.mean(target_bl_delta * target_bl_delta))
            ),
            "target_bl_max_abs_delta_db_re_mvdr": float(np.max(np.abs(target_bl_delta))),
        }
    )
    mvdr_summary.update(
        {
            "source_count_detected": "not_applicable",
            "music_associated_azimuth_deg": "not_applicable",
            "fallback_required": False,
            "target_bl_rms_delta_db_re_mvdr": 0.0,
            "target_bl_max_abs_delta_db_re_mvdr": 0.0,
        }
    )
    return EbaeMvdrSanityResult(
        azimuth_deg=azimuth_deg,
        ebae_target_bl_db=ebae_target,
        mvdr_target_bl_db=mvdr_target,
        ebae_noise_bl_db=ebae_noise,
        mvdr_noise_bl_db=mvdr_noise,
        ebae_mixed_bl_db=ebae_mixed,
        mvdr_mixed_bl_db=mvdr_mixed,
        summary_rows=(ebae_summary, mvdr_summary),
        signal_count=ebae_result.signal_count,
        associated_azimuth_deg=float(azimuth_deg[associated_index]),
    )


def write_ebae_mvdr_bin_center_sanity_report(output_dir: Path = OUTPUT_DIR) -> EbaeMvdrSanityResult:
    """EBAE/MVDR sanity結果をCSV、NPZ、PNG、Markdownへ保存する。

    Args:
        output_dir: review pack出力先。

    Returns:
        保存に使用した比較結果。

    Raises:
        OSError: 出力directoryまたは成果物を書き込めない場合。
    """
    result = calculate_ebae_mvdr_bin_center_sanity()
    figure_dir = output_dir / "figures" / SCENARIO_ID
    data_dir = output_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "scenario_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(result.summary_rows[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.summary_rows)

    np.savez(
        data_dir / f"{SCENARIO_ID}.npz",
        azimuth_deg=result.azimuth_deg,
        ebae_target_bl_db_re_input_rms=result.ebae_target_bl_db,
        mvdr_target_bl_db_re_input_rms=result.mvdr_target_bl_db,
        ebae_noise_bl_db_re_input_rms=result.ebae_noise_bl_db,
        mvdr_noise_bl_db_re_input_rms=result.mvdr_noise_bl_db,
        ebae_mixed_bl_db_re_input_rms=result.ebae_mixed_bl_db,
        mvdr_mixed_bl_db_re_input_rms=result.mvdr_mixed_bl_db,
    )

    figure, axis = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    axis.plot(result.azimuth_deg, result.ebae_target_bl_db, label="EBAE DL=1")
    axis.plot(result.azimuth_deg, result.mvdr_target_bl_db, label="MVDR")
    axis.axvline(TARGET_AZIMUTH_DEG, color="black", linestyle="--", linewidth=1.0, label="source")
    axis.set(
        title="Bin-center, beam-center single-source target-only BL",
        xlabel="Waiting-beam azimuth [deg]",
        ylabel="RMS Level [dB re input RMS]",
        xlim=(0.0, 180.0),
        ylim=(DISPLAY_FLOOR_DB_RE_INPUT_RMS, 5.0),
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(figure_dir / "bl_overlay.png", dpi=160)
    figure.savefig(figure_dir / "source_frequency_bl_overlay.png", dpi=160)
    plt.close(figure)

    ebae_row = result.summary_rows[0]
    review_lines = (
        "# EBAE/MVDR bin中心・beam直上 sanity check",
        "",
        f"- scenario: `{SCENARIO_ID}`",
        "- evaluation pattern: `fixed_beam_single_source`",
        f"- target: {TARGET_AZIMUTH_DEG:.1f} deg, {TARGET_FREQUENCY_HZ:.1f} Hz, 0 dB re input RMS",
        f"- array: {N_CHANNEL} ch ULA, target周波数の半波長間隔",
        f"- noise: {10.0 * np.log10(NOISE_POWER_RE_INPUT_RMS2_PER_CHANNEL):.1f} dB re input RMS^2/channel",
        f"- EBAE: DL=1, sigm_a=10, sigm_b=0.5, detected Ns={result.signal_count}",
        f"- MUSIC対応方位: {result.associated_azimuth_deg:.1f} deg",
        f"- target BL RMS差: {float(ebae_row['target_bl_rms_delta_db_re_mvdr']):.3f} dB re MVDR",
        f"- target BL最大絶対差: {float(ebae_row['target_bl_max_abs_delta_db_re_mvdr']):.3f} dB re MVDR",
        "",
        "この成果物はEBAEの基本動作確認専用であり、S1/S2/T1/T2、FIR長、FRAZ、BTR、",
        "streaming成立性、方式採否の根拠には使用しない。単一binのためFRAZとBTRは未評価である。",
        "EBAEはtarget応答とpeak方位をMVDRと一致させる一方、DL=1のロバスト化により",
        "非target抑圧がMVDRより浅くなることを許容する。",
        "",
        f"- figure: `figures/{SCENARIO_ID}/bl_overlay.png`",
        f"- data: `data/{SCENARIO_ID}.npz`",
        "- metrics: `scenario_summary.csv`",
    )
    (output_dir / "review_index.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """既定出力先へEBAE/MVDR sanity review packを生成する。"""
    write_ebae_mvdr_bin_center_sanity_report()


if __name__ == "__main__":
    main()
