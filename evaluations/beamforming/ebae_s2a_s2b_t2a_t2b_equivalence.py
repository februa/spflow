"""S2a/S2bおよびT2a/T2bの有限長FIR実現同値性を評価する。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evaluations.beamforming import ebae_mvdr_s1_s2a_t1_t2a_fir_sweep as reference


ComplexArray = NDArray[np.complex128]
OUTPUT_DIR = Path("artifacts/beamforming/ebae_s2a_s2b_t2a_t2b_equivalence/review_pack")
DIRECT_METHOD_IDS = ("S2a", "T2a")
BRANCH_METHOD_IDS = {"S2a": "S2b", "T2a": "T2b"}
TAP_COUNTS = (16, 32, 64, 128, 256, 512)


def _truncate_with_declared_window(
    weights: ComplexArray,
    tap_count: int,
    window_start_samples: NDArray[np.int64],
) -> ComplexArray:
    """宣言済み共通窓で多channel重みを有限長FIRへ射影する。

    Args:
        weights: 周波数重み。shapeは``[n_fft,n_beam,n_ch]``。
        tap_count: FIR tap数。単位はsample。
        window_start_samples: beam別の共通窓先頭。shapeは``[n_beam]``、単位はsample。

    Returns:
        FIR再構成重み。shapeは入力と同じ。

    Raises:
        ValueError: shapeまたはtap範囲が評価契約と異なる場合。

    Notes:
        同値性には主枝と差分枝で同じ線形射影を使う必要がある。枝ごとに最大energy窓を
        独立選択すると、窓演算が異なるため有限長での厳密同値性は保証されない。
    """
    expected_shape = (reference.FFT_SIZE, reference.AZIMUTH_DEG.size, reference.N_CHANNEL)
    if weights.shape != expected_shape:
        raise ValueError(f"weights must have shape {expected_shape}.")
    if window_start_samples.shape != (reference.AZIMUTH_DEG.size,):
        raise ValueError("window_start_samples must have shape (n_beam,).")
    if not 0 < tap_count <= reference.FFT_SIZE:
        raise ValueError("tap_count must be in [1, n_fft].")

    reconstructed = np.empty_like(weights)
    for beam_index, raw_start in enumerate(window_start_samples.tolist()):
        start = int(raw_start)
        # H=conj(w)をIFFTしてFIR係数へ移し、全channelへ同一のtap支持を適用する。
        impulse = np.asarray(
            np.fft.ifft(weights[:, beam_index, :].conj(), axis=0), dtype=np.complex128
        )
        keep_indices = (start + np.arange(tap_count)) % reference.FFT_SIZE
        truncated = np.zeros_like(impulse)
        truncated[keep_indices, :] = impulse[keep_indices, :]
        reconstructed[:, beam_index, :] = np.fft.fft(truncated, axis=0).conj()
    return reconstructed


def _fixed_residual_weights(design: reference.WeightDesignResult) -> ComplexArray:
    """整数遅延後座標の固定CBF主枝重みを返す。

    Args:
        design: 整数遅延位相と元座標steeringを含む完成設計。

    Returns:
        固定主枝重み。shapeは``[n_fft,n_beam,n_ch]``。
    """
    rotated_steering = design.integer_phase * design.steering
    # 未正規化steeringのnorm二乗はactive channel数Mなので、a_D^H f=1となるCBF重みにする。
    return np.asarray(rotated_steering / float(reference.N_CHANNEL), dtype=np.complex128)


def calculate_equivalence_rows() -> tuple[dict[str, Any], ...]:
    """EBAE/MVDR、S/T、tap数ごとのa/b同値誤差を計算する。

    Returns:
        評価行。各行は複素重み、``w^H a``、BL、決定論的波形の最大誤差を含む。
    """
    design = reference.design_reference_weights()
    fixed_residual = _fixed_residual_weights(design)
    rows: list[dict[str, Any]] = []
    source_spectrum = np.zeros(reference.FFT_SIZE, dtype=np.complex128)
    source_spectrum[design.source_bin_mask] = 1.0 + 0.0j

    for algorithm in reference.ALGORITHM_IDS:
        for direct_method in DIRECT_METHOD_IDS:
            branch_method = BRANCH_METHOD_IDS[direct_method]
            adaptive_residual = design.weights[algorithm][direct_method]
            # q=f-vと定義するため、固定主枝－差分枝は厳密にvへ戻る。
            difference_residual = np.asarray(
                fixed_residual - adaptive_residual, dtype=np.complex128
            )
            for tap_count in TAP_COUNTS:
                direct = reference.approximate_weights_with_fir(adaptive_residual, tap_count)
                starts = direct.window_start_samples
                fixed_fir = _truncate_with_declared_window(fixed_residual, tap_count, starts)
                difference_fir = _truncate_with_declared_window(
                    difference_residual, tap_count, starts
                )
                branch_combined = np.asarray(fixed_fir - difference_fir, dtype=np.complex128)
                direct_fir = direct.reconstructed_weights

                weight_error = float(np.max(np.abs(branch_combined - direct_fir)))
                direct_original = reference._original_coordinate_weights(
                    direct_method, direct_fir, design.integer_phase
                )
                branch_original = reference._original_coordinate_weights(
                    direct_method, branch_combined, design.integer_phase
                )
                direct_response = np.einsum(
                    "fbc,fc->fb",
                    direct_original.conj(),
                    design.source_steering,
                    optimize=True,
                )
                branch_response = np.einsum(
                    "fbc,fc->fb",
                    branch_original.conj(),
                    design.source_steering,
                    optimize=True,
                )
                response_error = float(np.max(np.abs(branch_response - direct_response)))
                direct_bl = np.mean(np.abs(direct_response[design.source_bin_mask]) ** 2, axis=0)
                branch_bl = np.mean(np.abs(branch_response[design.source_bin_mask]) ** 2, axis=0)
                bl_error = float(np.max(np.abs(branch_bl - direct_bl)))

                target_index = int(
                    np.argmin(np.abs(reference.AZIMUTH_DEG - reference.TARGET_AZIMUTH_DEG))
                )
                direct_output_spectrum = source_spectrum * direct_response[:, target_index]
                branch_output_spectrum = source_spectrum * branch_response[:, target_index]
                direct_waveform = np.fft.ifft(direct_output_spectrum)
                branch_waveform = np.fft.ifft(branch_output_spectrum)
                waveform_error = float(np.max(np.abs(branch_waveform - direct_waveform)))
                rows.append(
                    {
                        "algorithm": algorithm,
                        "covariance_family": direct_method[0],
                        "direct_method": direct_method,
                        "difference_branch_method": branch_method,
                        "tap_count": tap_count,
                        "maximum_complex_weight_error": weight_error,
                        "maximum_w_h_a_error": response_error,
                        "maximum_bl_power_error": bl_error,
                        "maximum_waveform_error": waveform_error,
                        "equivalent": bool(
                            weight_error < 1.0e-12
                            and response_error < 1.0e-12
                            and bl_error < 1.0e-12
                            and waveform_error < 1.0e-12
                        ),
                    }
                )
    return tuple(rows)


def write_review_pack(output_dir: Path = OUTPUT_DIR) -> None:
    """a/b同値性CSVと評価条件を保存する。

    Args:
        output_dir: review pack出力先。
    """
    rows = calculate_equivalence_rows()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "equivalence_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "README.md").write_text(
        "# S2a/S2b and T2a/T2b equivalence\n\n"
        "同一完成重み、同一整数遅延、同一tap支持を使い、"
        "複素重み、w^H a、BL power、波形の誤差が1e-12未満であることを合格条件とする。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_review_pack()
