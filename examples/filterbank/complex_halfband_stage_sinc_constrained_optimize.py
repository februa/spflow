"""sinc 目標 constrained 最適化を実行して候補を保存するサンプル。"""

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

from spflow.filterbank.design.sinc_constrained_optimizer import (
    ConstrainedSincQMFOptimizerConfig,
    make_constrained_sinc_qmf_candidate,
)
from spflow.filterbank.design.sinc_target import evaluate_candidate_against_sinc_target

ARTIFACT_ROOT = ROOT / "artifacts" / "complex_halfband_stage_design"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ConstrainedSincOptimizerRow:
    """constrained sinc 最適化結果 1 件を保持する。"""
    name: str
    taps: int
    target_window: str
    weighted_power_rms_error: float
    sinc_fullband_rms_error: float
    sinc_stopband_rms_error: float
    stage_pr_max_abs_error: float
    stage_pr_rms_error: float
    power_complementarity_error: float
    low_stopband_attenuation_db: float
    high_stopband_attenuation_db: float
    max_passband_ripple_db: float
    analysis_phase: int
    synthesis_phase: int
    delay_compensation: int


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="Run the prototype constrained optimizer for sinc-target halfband stages.")
    parser.add_argument("--tap-list", type=int, nargs="*", default=[16, 24, 32, 48])
    parser.add_argument("--window", choices=["hann", "hamming", "blackman"], default="blackman")
    parser.add_argument("--cutoff", type=float, default=0.25)
    parser.add_argument("--fft-size", type=int, default=8192)
    parser.add_argument("--max-passes", type=int, default=60)
    parser.add_argument("--artifact-name", type=str, default="sinc_constrained_optimizer_results")
    return parser.parse_args()


def main() -> None:
    """sinc 目標 constrained 最適化を実行し、候補と診断量を保存する。"""
    args = parse_args()
    rows: list[ConstrainedSincOptimizerRow] = []

    for taps in args.tap_list:
        candidate, diagnostics = make_constrained_sinc_qmf_candidate(
            ConstrainedSincQMFOptimizerConfig(
                num_taps=int(taps),
                cutoff=float(args.cutoff),
                window=args.window,
                fft_size=int(args.fft_size),
                max_passes=int(args.max_passes),
            )
        )
        sinc_metrics = evaluate_candidate_against_sinc_target(
            candidate,
            cutoff=float(args.cutoff),
            window=args.window,
            fft_size=65536,
        )
        rows.append(
            ConstrainedSincOptimizerRow(
                name=candidate.name,
                taps=int(taps),
                target_window=args.window,
                weighted_power_rms_error=diagnostics.weighted_power_rms_error,
                sinc_fullband_rms_error=sinc_metrics.fullband_rms_error,
                sinc_stopband_rms_error=sinc_metrics.stopband_rms_error,
                stage_pr_max_abs_error=diagnostics.stage_pr_max_abs_error,
                stage_pr_rms_error=diagnostics.stage_pr_rms_error,
                power_complementarity_error=diagnostics.power_complementarity_error,
                low_stopband_attenuation_db=diagnostics.low_stopband_attenuation_db,
                high_stopband_attenuation_db=diagnostics.high_stopband_attenuation_db,
                max_passband_ripple_db=diagnostics.max_passband_ripple_db,
                analysis_phase=diagnostics.analysis_phase,
                synthesis_phase=diagnostics.synthesis_phase,
                delay_compensation=diagnostics.delay_compensation,
            )
        )

    rows.sort(key=lambda item: (item.weighted_power_rms_error, item.stage_pr_rms_error))
    artifact_path = ARTIFACT_ROOT / f"{args.artifact_name}.json"
    artifact_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")

    print("ComplexPRHalfbandStage sinc constrained optimizer")
    print(f"tap_list={args.tap_list}")
    print(f"target_window={args.window}")
    print(f"cutoff={args.cutoff}")
    print(f"fft_size={args.fft_size}")
    print(f"max_passes={args.max_passes}")
    print(f"artifact_json={artifact_path}")
    print()
    print("| rank | name | taps | weighted_power_rms | sinc_fullband_rms | stopband_att_db | pr_max | pr_rms | pce |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(rows, start=1):
        min_stop = min(row.low_stopband_attenuation_db, row.high_stopband_attenuation_db)
        print(
            f"| {idx} | {row.name} | {row.taps} | {row.weighted_power_rms_error:.6e} | "
            f"{row.sinc_fullband_rms_error:.6e} | {min_stop:.3f} | {row.stage_pr_max_abs_error:.6e} | "
            f"{row.stage_pr_rms_error:.6e} | {row.power_complementarity_error:.6e} |"
        )


if __name__ == "__main__":
    main()
