"""Complex halfband stage 候補を評価して最良係数を保存するサンプル。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.filterbank.complex_halfband_stage import (
    ComplexFIRHalfbandStageStreamingAnalyzer,
    ComplexFIRHalfbandStageStreamingSynthesizer,
)
from spflow.filterbank.design.complex_halfband_stage import make_daubechies_qmf_candidate

ARTIFACT_ROOT = ROOT / "artifacts" / "complex_halfband_stage_design"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ComplexHalfbandStageCandidateResult:
    """Complex halfband stage 候補 1 件の評価結果を保持する。"""
    family: str
    name: str
    order: int
    analysis_length: int
    analysis_phase: int
    synthesis_phase: int
    delay_compensation: int
    low_passband_ripple_db: float
    high_passband_ripple_db: float
    low_stopband_attenuation_db: float
    high_stopband_attenuation_db: float
    power_complementarity_error: float
    pr_max_abs_error: float
    pr_rms_error: float
    streaming_analysis_max_abs_error: float
    streaming_synthesis_max_abs_error: float
    exact_pr: bool
    meets_frequency_target: bool
    meets_formal_target: bool
    score: float


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="Design, evaluate, and save ComplexPRHalfbandStage coefficients.")
    parser.add_argument("--family", choices=["daubechies_qmf"], default="daubechies_qmf")
    parser.add_argument(
        "--order-list",
        type=int,
        nargs="*",
        default=[2, 3, 4, 6, 8, 10, 12, 16, 20, 22, 24, 26, 28, 30],
    )
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fft-size", type=int, default=65536)
    parser.add_argument("--stream-chunk-size-list", type=int, nargs="*", default=[17, 64, 255])
    parser.add_argument("--artifact-name", type=str, default="best_complex_halfband_stage")
    return parser.parse_args()


def make_signal(length: int, seed: int) -> np.ndarray:
    """評価用の複素乱数信号を生成する。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(length) + 1j * rng.standard_normal(length)


def split_chunks(x: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    """1 次元信号を streaming 評価用のチャンク列へ分割する。"""
    return [x[start : start + chunk_size] for start in range(0, x.shape[-1], chunk_size)]


def compute_pr_metrics(stage, signal: np.ndarray) -> tuple[float, float]:
    """offline 解析合成の完全再構成誤差を集計する。"""
    low, high = stage.analysis(signal)
    reconstructed = stage.synthesis(low, high, length=signal.shape[-1])
    error = reconstructed - signal
    return float(np.max(np.abs(error))), float(np.sqrt(np.mean(np.abs(error) ** 2)))


def compute_streaming_metrics(stage, signal: np.ndarray, chunk_sizes: list[int]) -> tuple[float, float]:
    """streaming 解析合成の oracle 一致誤差を集計する。"""
    low_offline, high_offline = stage.analysis(signal)
    reconstructed_offline = stage.synthesis(low_offline, high_offline)

    analysis_error = 0.0
    synthesis_error = 0.0
    for chunk_size in chunk_sizes:
        analyzer = ComplexFIRHalfbandStageStreamingAnalyzer(stage)
        synth = ComplexFIRHalfbandStageStreamingSynthesizer(stage)
        low_chunks: list[np.ndarray] = []
        high_chunks: list[np.ndarray] = []
        reconstructed_chunks: list[np.ndarray] = []

        for chunk in split_chunks(signal, chunk_size):
            low_chunk, high_chunk = analyzer.process(chunk)
            low_chunks.append(low_chunk)
            high_chunks.append(high_chunk)
            reconstructed_chunks.append(synth.process(low_chunk, high_chunk))

        tail_low, tail_high = analyzer.flush()
        low_chunks.append(tail_low)
        high_chunks.append(tail_high)
        reconstructed_chunks.append(synth.process(tail_low, tail_high))
        reconstructed_chunks.append(synth.flush())

        low_stream = np.concatenate(low_chunks)
        high_stream = np.concatenate(high_chunks)
        reconstructed_stream = np.concatenate(reconstructed_chunks)

        analysis_error = max(
            analysis_error,
            float(np.max(np.abs(low_stream - low_offline))),
            float(np.max(np.abs(high_stream - high_offline))),
        )
        synthesis_error = max(
            synthesis_error,
            float(np.max(np.abs(reconstructed_stream - reconstructed_offline))),
        )

    return analysis_error, synthesis_error


def compute_score(
    *,
    exact_pr: bool,
    meets_frequency_target: bool,
    analysis_length: int,
    min_stopband_attenuation_db: float,
    max_passband_ripple_db: float,
    pr_max_abs_error: float,
) -> float:
    """候補比較用の総合スコアを計算する。"""
    return (
        (1_000_000.0 if meets_frequency_target else 0.0)
        + (100_000.0 if exact_pr else 0.0)
        - float(analysis_length)
        - 1.0e10 * pr_max_abs_error
        + 0.1 * min_stopband_attenuation_db
        - max_passband_ripple_db
    )


def evaluate_candidate(
    *,
    order: int,
    n_samples: int,
    seed: int,
    fft_size: int,
    stream_chunk_sizes: list[int],
) -> tuple[ComplexHalfbandStageCandidateResult, object]:
    """候補 1 件を評価し、指標と候補オブジェクトを返す。"""
    candidate = make_daubechies_qmf_candidate(order)
    stage = candidate.make_stage()
    metrics = candidate.response_metrics(fft_size=fft_size)
    signal = make_signal(n_samples, seed)
    pr_max_abs_error, pr_rms_error = compute_pr_metrics(stage, signal)
    streaming_analysis_error, streaming_synthesis_error = compute_streaming_metrics(stage, signal, stream_chunk_sizes)

    max_ripple = max(metrics["low_passband_ripple_db"], metrics["high_passband_ripple_db"])
    min_stopband = min(metrics["low_stopband_attenuation_db"], metrics["high_stopband_attenuation_db"])
    exact_pr = pr_max_abs_error <= 1e-10 and pr_rms_error <= 1e-12
    meets_frequency_target = (
        streaming_analysis_error <= 1e-10
        and streaming_synthesis_error <= 1e-10
        and max_ripple <= 0.1
        and min_stopband >= 80.0
    )
    meets_formal_target = exact_pr and meets_frequency_target
    score = compute_score(
        exact_pr=exact_pr,
        meets_frequency_target=meets_frequency_target,
        analysis_length=candidate.analysis_low.size,
        min_stopband_attenuation_db=min_stopband,
        max_passband_ripple_db=max_ripple,
        pr_max_abs_error=pr_max_abs_error,
    )

    return (
        ComplexHalfbandStageCandidateResult(
            family="daubechies_qmf",
            name=candidate.name,
            order=order,
            analysis_length=int(candidate.analysis_low.size),
            analysis_phase=candidate.analysis_phase,
            synthesis_phase=candidate.synthesis_phase,
            delay_compensation=candidate.delay_compensation,
            low_passband_ripple_db=metrics["low_passband_ripple_db"],
            high_passband_ripple_db=metrics["high_passband_ripple_db"],
            low_stopband_attenuation_db=metrics["low_stopband_attenuation_db"],
            high_stopband_attenuation_db=metrics["high_stopband_attenuation_db"],
            power_complementarity_error=metrics["power_complementarity_error"],
            pr_max_abs_error=pr_max_abs_error,
            pr_rms_error=pr_rms_error,
            streaming_analysis_max_abs_error=streaming_analysis_error,
            streaming_synthesis_max_abs_error=streaming_synthesis_error,
            exact_pr=exact_pr,
            meets_frequency_target=meets_frequency_target,
            meets_formal_target=meets_formal_target,
            score=score,
        ),
        candidate,
    )


def save_best_candidate(best_row: ComplexHalfbandStageCandidateResult, candidate, artifact_name: str) -> tuple[Path, Path]:
    """最良候補の係数と評価結果を artifact に保存する。"""
    artifact_dir = ARTIFACT_ROOT / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    npz_path = artifact_dir / "stage_filters.npz"
    json_path = artifact_dir / "stage_filters.json"

    stage_filters = candidate.make_stage_filters()
    np.savez(
        npz_path,
        family=best_row.family,
        name=best_row.name,
        order=best_row.order,
        analysis_low=stage_filters.analysis_low,
        analysis_high=stage_filters.analysis_high,
        synthesis_low=stage_filters.synthesis_low,
        synthesis_high=stage_filters.synthesis_high,
        analysis_phase=best_row.analysis_phase,
        synthesis_phase=best_row.synthesis_phase,
        delay_compensation=best_row.delay_compensation,
    )
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(best_row), f, indent=2)
    return npz_path, json_path


def main() -> None:
    """Complex halfband stage 候補を評価し、最良候補を artifact に保存する。"""
    args = parse_args()
    rows: list[ComplexHalfbandStageCandidateResult] = []
    candidate_by_name = {}

    for order in args.order_list:
        row, candidate = evaluate_candidate(
            order=int(order),
            n_samples=args.n_samples,
            seed=args.seed,
            fft_size=args.fft_size,
            stream_chunk_sizes=[int(size) for size in args.stream_chunk_size_list],
        )
        rows.append(row)
        candidate_by_name[row.name] = candidate

    rows.sort(key=lambda item: item.score, reverse=True)
    best = rows[0]
    best_candidate = candidate_by_name[best.name]
    npz_path, json_path = save_best_candidate(best, best_candidate, args.artifact_name)

    print("ComplexPRHalfbandStage coefficient design / evaluation")
    print(f"family={args.family}")
    print(f"order_list={args.order_list}")
    print(f"reference_n_samples={args.n_samples}, reference_seed={args.seed}")
    print(f"stream_chunk_size_list={args.stream_chunk_size_list}")
    print()
    print("Selection policy")
    print("- Prefer candidates that satisfy the 80 dB / 0.1 dB frequency target with streaming consistency.")
    print("- Within that set, prefer lower PR error and shorter filters.")
    print()
    print("Top candidates")
    print(
        "| rank | name | len | phases(a/s) | delay | exact_pr | freq80 | formal | stopband_att_db | ripple_db | "
        "stream_a_err | stream_s_err | score |"
    )
    print("|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(rows[:10], start=1):
        min_stop = min(row.low_stopband_attenuation_db, row.high_stopband_attenuation_db)
        max_ripple = max(row.low_passband_ripple_db, row.high_passband_ripple_db)
        print(
            f"| {idx} | {row.name} | {row.analysis_length} | {row.analysis_phase}/{row.synthesis_phase} | "
            f"{row.delay_compensation} | {row.exact_pr} | {row.meets_frequency_target} | {row.meets_formal_target} | {min_stop:.3f} | "
            f"{max_ripple:.6f} | {row.streaming_analysis_max_abs_error:.3e} | "
            f"{row.streaming_synthesis_max_abs_error:.3e} | {row.score:.3f} |"
        )

    print()
    print("Selected stage filters")
    print(f"name={best.name}")
    print(f"order={best.order}")
    print(f"analysis_length={best.analysis_length}")
    print(f"analysis_phase={best.analysis_phase}")
    print(f"synthesis_phase={best.synthesis_phase}")
    print(f"delay_compensation={best.delay_compensation}")
    print(f"exact_pr={best.exact_pr}")
    print(f"meets_frequency_target={best.meets_frequency_target}")
    print(f"meets_formal_target={best.meets_formal_target}")
    print(f"low_stopband_attenuation_db={best.low_stopband_attenuation_db:.6f}")
    print(f"high_stopband_attenuation_db={best.high_stopband_attenuation_db:.6f}")
    print(f"low_passband_ripple_db={best.low_passband_ripple_db:.6f}")
    print(f"high_passband_ripple_db={best.high_passband_ripple_db:.6f}")
    print(f"streaming_analysis_max_abs_error={best.streaming_analysis_max_abs_error:.6e}")
    print(f"streaming_synthesis_max_abs_error={best.streaming_synthesis_max_abs_error:.6e}")
    print(f"artifact_npz={npz_path}")
    print(f"artifact_json={json_path}")


if __name__ == "__main__":
    main()
