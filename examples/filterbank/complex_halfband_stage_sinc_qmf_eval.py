"""窓付き sinc QMF 候補を評価するサンプル。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.filterbank.design.complex_halfband_stage import resolve_qmf_stage_parameters
from spflow.filterbank.halfband_stage_candidates import OrthonormalQMFStageCandidate

ARTIFACT_ROOT = ROOT / "artifacts" / "complex_halfband_stage_design"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SincQMFExperimentResult:
    """sinc 系 QMF 候補 1 件の評価結果を保持する。"""
    name: str
    window: str
    taps: int
    analysis_phase: int
    synthesis_phase: int
    delay_compensation: int
    low_stopband_attenuation_db: float
    high_stopband_attenuation_db: float
    max_passband_ripple_db: float
    power_complementarity_error: float
    pr_max_abs_error: float
    pr_rms_error: float


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="Evaluate naive windowed-sinc QMF stage candidates.")
    parser.add_argument("--window-list", nargs="*", default=["hann", "hamming", "blackman"])
    parser.add_argument("--tap-list", type=int, nargs="*", default=[16, 24, 32, 48, 64, 96])
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-name", type=str, default="sinc_qmf_experiment")
    return parser.parse_args()


def design_windowed_sinc_lowpass(num_taps: int, *, cutoff: float = 0.25, window: str = "hamming") -> np.ndarray:
    """窓付き sinc から lowpass 係数を設計する。"""
    if num_taps <= 0:
        raise ValueError("num_taps must be positive.")
    n = np.arange(num_taps, dtype=np.float32)
    midpoint = 0.5 * (num_taps - 1)
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * (n - midpoint))
    taps *= select_window(window, num_taps)
    taps *= np.sqrt(2.0) / np.sum(taps)
    return taps


def select_window(name: str, length: int) -> np.ndarray:
    """名前に応じた窓関数を返す。"""
    if name == "hann":
        return np.hanning(length)
    if name == "hamming":
        return np.hamming(length)
    if name == "blackman":
        return np.blackman(length)
    raise ValueError(f"Unsupported window: {name}")


def make_signal(length: int, seed: int) -> np.ndarray:
    """評価用の複素乱数信号を生成する。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(length) + 1j * rng.standard_normal(length)


def evaluate_candidate(*, window: str, taps: int, signal: np.ndarray) -> SincQMFExperimentResult:
    """候補 1 件を評価し、指標と候補オブジェクトを返す。"""
    analysis_low = design_windowed_sinc_lowpass(taps, window=window)
    params = resolve_qmf_stage_parameters(analysis_low)
    candidate = OrthonormalQMFStageCandidate(
        name=f"sinc_{window}_taps{taps}",
        analysis_low=analysis_low,
        analysis_phase=params.analysis_phase,
        synthesis_phase=params.synthesis_phase,
        delay_compensation=params.delay_compensation,
    )

    stage = candidate.make_stage()
    metrics = candidate.response_metrics()
    low, high = stage.analysis(signal)
    reconstructed = stage.synthesis(low, high, length=signal.shape[-1])
    error = reconstructed - signal

    return SincQMFExperimentResult(
        name=candidate.name,
        window=window,
        taps=taps,
        analysis_phase=params.analysis_phase,
        synthesis_phase=params.synthesis_phase,
        delay_compensation=params.delay_compensation,
        low_stopband_attenuation_db=metrics["low_stopband_attenuation_db"],
        high_stopband_attenuation_db=metrics["high_stopband_attenuation_db"],
        max_passband_ripple_db=max(metrics["low_passband_ripple_db"], metrics["high_passband_ripple_db"]),
        power_complementarity_error=metrics["power_complementarity_error"],
        pr_max_abs_error=float(np.max(np.abs(error))),
        pr_rms_error=float(np.sqrt(np.mean(np.abs(error) ** 2))),
    )


def main() -> None:
    """窓付き sinc QMF 候補を評価し、結果を表示する。"""
    args = parse_args()
    signal = make_signal(args.n_samples, args.seed)
    rows = [
        evaluate_candidate(window=window, taps=taps, signal=signal)
        for window in args.window_list
        for taps in args.tap_list
    ]
    rows.sort(
        key=lambda item: min(item.low_stopband_attenuation_db, item.high_stopband_attenuation_db),
        reverse=True,
    )

    artifact_path = ARTIFACT_ROOT / f"{args.artifact_name}.json"
    artifact_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")

    print("Naive windowed-sinc QMF experiment")
    print(f"window_list={args.window_list}")
    print(f"tap_list={args.tap_list}")
    print(f"reference_n_samples={args.n_samples}, reference_seed={args.seed}")
    print(f"artifact_json={artifact_path}")
    print()
    print("| rank | name | stopband_att_db | ripple_db | pce | pr_max | pr_rms | phases(a/s) | delay |")
    print("|---:|---|---:|---:|---:|---:|---:|---|---:|")
    for idx, row in enumerate(rows[:10], start=1):
        min_stop = min(row.low_stopband_attenuation_db, row.high_stopband_attenuation_db)
        print(
            f"| {idx} | {row.name} | {min_stop:.3f} | {row.max_passband_ripple_db:.6f} | "
            f"{row.power_complementarity_error:.6e} | {row.pr_max_abs_error:.6e} | {row.pr_rms_error:.6e} | "
            f"{row.analysis_phase}/{row.synthesis_phase} | {row.delay_compensation} |"
        )


if __name__ == "__main__":
    main()
