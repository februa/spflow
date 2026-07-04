"""sinc 目標応答に対する候補誤差を評価するサンプル。"""

# フィルタ設計式の妥当性を stopband・リップル・完全再構成誤差の観点で比較し、
# どの設計パラメータが実用上のトレードオフを支配するかを観察するための評価例である。

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.filterbank.design.complex_halfband_stage import make_daubechies_qmf_candidate
from spflow.filterbank.design.sinc_target import evaluate_candidate_against_sinc_target

ARTIFACT_ROOT = ROOT / "artifacts" / "complex_halfband_stage_design"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SincTargetCandidateRow:
    """sinc 目標応答に対する候補評価結果 1 件を保持する。"""
    name: str
    order: int
    taps: int
    target_window: str
    cutoff: float
    fullband_rms_error: float
    passband_rms_error: float
    stopband_rms_error: float
    transition_rms_error: float


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="Evaluate paraunitary candidates against a sinc-derived target response.")
    parser.add_argument("--order-list", type=int, nargs="*", default=[4, 8, 12, 16, 20, 24, 28, 30])
    parser.add_argument("--window", choices=["hann", "hamming", "blackman"], default="blackman")
    parser.add_argument("--cutoff", type=float, default=0.25)
    parser.add_argument("--fft-size", type=int, default=65536)
    parser.add_argument("--artifact-name", type=str, default="sinc_target_candidate_ranking")
    return parser.parse_args()


def main() -> None:
    """sinc 目標応答に対する候補誤差を評価して結果を表示する。"""
    args = parse_args()
    rows: list[SincTargetCandidateRow] = []

    for order in args.order_list:
        candidate = make_daubechies_qmf_candidate(int(order))
        metrics = evaluate_candidate_against_sinc_target(
            candidate,
            cutoff=float(args.cutoff),
            window=args.window,
            fft_size=int(args.fft_size),
        )
        rows.append(
            SincTargetCandidateRow(
                name=candidate.name,
                order=int(order),
                taps=int(candidate.analysis_low.size),
                target_window=metrics.target_window,
                cutoff=metrics.cutoff,
                fullband_rms_error=metrics.fullband_rms_error,
                passband_rms_error=metrics.passband_rms_error,
                stopband_rms_error=metrics.stopband_rms_error,
                transition_rms_error=metrics.transition_rms_error,
            )
        )

    rows.sort(key=lambda item: item.fullband_rms_error)

    artifact_path = ARTIFACT_ROOT / f"{args.artifact_name}.json"
    artifact_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")

    print("ComplexPRHalfbandStage sinc-target candidate evaluation")
    print(f"order_list={args.order_list}")
    print(f"target_window={args.window}")
    print(f"cutoff={args.cutoff}")
    print(f"fft_size={args.fft_size}")
    print(f"artifact_json={artifact_path}")
    print()
    print("| rank | name | taps | fullband_rms | passband_rms | stopband_rms | transition_rms |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(rows, start=1):
        print(
            f"| {idx} | {row.name} | {row.taps} | {row.fullband_rms_error:.6e} | "
            f"{row.passband_rms_error:.6e} | {row.stopband_rms_error:.6e} | {row.transition_rms_error:.6e} |"
        )


if __name__ == "__main__":
    main()
