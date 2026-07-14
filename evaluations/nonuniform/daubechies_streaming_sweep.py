"""非均一Daubechies streamingビームフォーマの誤差をsweepする。"""

# 非均一木構造では分割仕様と streaming 状態の組み合わせで挙動が大きく変わるため、
# 実運用に近い入出力条件を一式そろえて可視化・書き出しできる例として管理する。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming.cbf import design_cbf_coefficients
from spflow.beamforming.mvdr_filter import beam_response_rms_db
from spflow.filterbank.daubechies_nonuniform_beamformer import DaubechiesNonuniformBeamformer
from spflow.filterbank.daubechies_nonuniform_streaming import (
    DaubechiesNonuniformBeamformerStreaming,
)


def _chunk_signal(x: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    """信号を streaming 検証用の固定長チャンク列へ分割する。"""
    return [x[..., start : start + chunk_size] for start in range(0, x.shape[-1], chunk_size)]


def _matched_peak_response_db(beamformer: DaubechiesNonuniformBeamformer, freq_hz: float) -> float:
    """target 方位に対するビーム応答ピークを dB で評価する。"""
    spec = next(
        spec
        for spec in beamformer.band_specs
        if spec.f_low_hz <= freq_hz < spec.f_high_hz
        or (np.isclose(freq_hz, spec.f_high_hz) and np.isclose(spec.f_high_hz, beamformer.root_band_hz))
    )
    config = beamformer.leaf_configs[spec.band_id]
    steering = np.asarray(config.steering, dtype=np.complex64)
    used = np.asarray(config.used_channels, dtype=np.int64)
    reduced = steering[used, :, 0] if steering.ndim == 3 else steering[used, :]
    weights = design_cbf_coefficients(reduced)
    response = weights[:, 0].conj() @ reduced[:, 0]
    return beam_response_rms_db(response)


def _run_case(*, freq_hz: float, chunk_size: int) -> dict[str, float]:
    """1 つの sweep 条件で streaming 指標を計算する。"""
    fs_hz = 32768.0
    n_sample = 8192
    beamformer = DaubechiesNonuniformBeamformer(candidate_name="daubechies_qmf_order4_taps8")
    streaming = DaubechiesNonuniformBeamformerStreaming(beamformer=beamformer)

    axis_n = np.arange(n_sample, dtype=np.float32)
    x = np.exp(1j * 2.0 * np.pi * freq_hz * axis_n / fs_hz)
    multichannel = np.repeat(x[np.newaxis, :], beamformer.array_design.n_ch, axis=0)
    offline = beamformer.beamform_analytic(multichannel)[0]

    emitted = []
    boundary_indices = []
    produced = 0
    for chunk in _chunk_signal(multichannel, chunk_size):
        y_chunk = streaming.process_analytic(chunk)
        if y_chunk.shape[-1] > 0:
            emitted.append(y_chunk)
            produced += y_chunk.shape[-1]
            boundary_indices.append(produced - 1)
    y_tail = streaming.flush()
    if y_tail.shape[-1] > 0:
        emitted.append(y_tail)
        produced += y_tail.shape[-1]
        boundary_indices.append(produced - 1)

    reconstructed = np.concatenate(emitted, axis=-1)[0]
    error = reconstructed - offline
    jump_abs_error = np.abs(np.diff(error))
    boundary_indices = np.asarray(boundary_indices[:-1], dtype=np.int64)
    boundary_indices = boundary_indices[
        (boundary_indices >= 0) & (boundary_indices < jump_abs_error.size)
    ]
    if boundary_indices.size > 0:
        max_boundary_jump_abs_error = float(np.max(jump_abs_error[boundary_indices]))
    else:
        max_boundary_jump_abs_error = float("nan")

    return {
        "freq_hz": float(freq_hz),
        "chunk_size": float(chunk_size),
        "peak_response_db": _matched_peak_response_db(beamformer, freq_hz),
        "max_abs_error": float(np.max(np.abs(error))),
        "rms_error": float(np.sqrt(np.mean(np.abs(error) ** 2))),
        "max_jump_abs_error": float(np.max(jump_abs_error)) if jump_abs_error.size > 0 else 0.0,
        "max_boundary_jump_abs_error": max_boundary_jump_abs_error,
    }


def main() -> None:
    """非均一 Daubechies streaming 誤差の sweep 結果を CSV 形式で表示する。"""
    freqs_hz = [64.0, 192.0, 384.0, 768.0, 1536.0, 3072.0, 6144.0, 12288.0]
    chunk_sizes = [16, 32, 64]

    print(
        "freq_hz,chunk_size,peak_response_db,max_abs_error,rms_error,max_jump_abs_error,max_boundary_jump_abs_error"
    )
    for freq_hz in freqs_hz:
        for chunk_size in chunk_sizes:
            result = _run_case(freq_hz=freq_hz, chunk_size=chunk_size)
            print(
                f"{result['freq_hz']:.1f},{int(result['chunk_size'])},"
                f"{result['peak_response_db']:.6f},{result['max_abs_error']:.6e},"
                f"{result['rms_error']:.6e},{result['max_jump_abs_error']:.6e},"
                f"{result['max_boundary_jump_abs_error']:.6e}"
            )


if __name__ == "__main__":
    main()
