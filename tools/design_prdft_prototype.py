"""PRDFT prototype候補を探索して最良係数を保存するCLI。"""

# フィルタ設計式の妥当性を stopband・リップル・完全再構成誤差の観点で比較し、
# どの設計パラメータが実用上のトレードオフを支配するかを観察するための評価例である。

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from spflow import (
    FiniteLengthPRChecker,
    PRChecker,
    PRDFTAnalysisBank,
    PRDFTSynthesisBank,
    PolyphasePRDFTAnalysisBank,
    PolyphasePRDFTSynthesisBank,
    PolyphasePRPairDesigner,
    PrototypeFilter,
    PrototypePairDesigner,
)

ARTIFACT_ROOT = ROOT / 'artifacts' / 'prototype_design'
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PrototypeCandidateResult:
    """PRDFT prototype 候補 1 件の評価結果を保持する。"""
    structure: str
    name: str
    cutoff: float | None
    delay: int
    delay_unit: str
    synthesis_prototype_length: int
    regularization: float
    crop_mode: str
    valid_region_mode: str
    valid_margin: int | None
    pad_front: int | None
    pad_back: int | None
    n_eval_cases: int
    passband_peak: float
    stopband_peak: float
    stopband_attenuation_db: float
    pair_max_abs_error: float
    pair_rms_error: float
    pr_max_abs_error: float
    pr_rms_error: float
    valid_pr_max_abs_error: float
    valid_pr_rms_error: float
    exact_pr: bool
    score: float


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description='Design, optimize, and evaluate paired PRDFT prototypes.')
    parser.add_argument('--structure', choices=['explicit_modulation', 'polyphase_pr'], default='polyphase_pr')
    parser.add_argument('--n-band', type=int, default=32)
    parser.add_argument('--decimation', type=int, default=32)
    parser.add_argument('--prototype-length', type=int, default=256)
    parser.add_argument('--n-samples', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-length-list', type=int, nargs='*', default=None)
    parser.add_argument('--eval-seed-list', type=int, nargs='*', default=None)
    parser.add_argument('--cutoff-scale-list', type=float, nargs='*', default=[0.70, 0.80, 0.90, 1.00, 1.10, 1.20])
    parser.add_argument('--delay-start', type=int, default=0)
    parser.add_argument('--delay-stop', type=int, default=256)
    parser.add_argument('--delay-step', type=int, default=32)
    parser.add_argument('--delay-block-start', type=int, default=0)
    parser.add_argument('--delay-block-stop', type=int, default=16)
    parser.add_argument('--delay-block-step', type=int, default=1)
    parser.add_argument('--regularization-list', type=float, nargs='*', default=[0.0, 1e-8, 1e-6, 1e-4, 1e-2])
    parser.add_argument('--synthesis-prototype-length-list', type=int, nargs='*', default=[256, 384, 512])
    parser.add_argument('--crop-mode', choices=['input_aligned', 'full', 'valid', 'custom'], default='input_aligned')
    parser.add_argument('--crop-front', type=int, default=None)
    parser.add_argument('--crop-length', type=int, default=None)
    parser.add_argument('--valid-region-mode', choices=['transient', 'analysis', 'synthesis', 'none', 'custom'], default='transient')
    parser.add_argument('--valid-margin', type=int, default=None)
    parser.add_argument('--pad-front', type=int, default=None)
    parser.add_argument('--pad-back', type=int, default=None)
    parser.add_argument('--score-mode', choices=['worst_valid', 'worst_full', 'balanced'], default='worst_valid')
    parser.add_argument('--artifact-name', type=str, default='optimal_pr_prototype')
    return parser.parse_args()


def build_candidate_specs(args: argparse.Namespace) -> list[tuple[str, PrototypeFilter]]:
    """探索対象となる prototype 候補一覧を構成する。"""
    base_cutoff = 1.0 / args.decimation
    specs: list[tuple[str, PrototypeFilter]] = [
        (
            'block_dft_baseline',
            PrototypeFilter.block_dft_baseline(
                n_band=args.n_band,
                decimation=args.decimation,
                prototype_length=args.prototype_length,
            ),
        )
    ]
    for scale in args.cutoff_scale_list:
        cutoff = base_cutoff * scale
        specs.append(
            (
                f'windowed_sinc@{float(cutoff):.8f}',
                PrototypeFilter.windowed_sinc(
                    n_band=args.n_band,
                    decimation=args.decimation,
                    prototype_length=args.prototype_length,
                    cutoff=float(cutoff),
                ),
            )
        )
    return specs


def build_eval_cases(args: argparse.Namespace) -> list[tuple[int, int]]:
    """信号長と seed の評価ケース一覧を構成する。"""
    lengths = args.eval_length_list if args.eval_length_list is not None else [args.n_samples]
    seeds = args.eval_seed_list if args.eval_seed_list is not None else [args.seed]
    cases = [(int(length), int(seed)) for length in lengths for seed in seeds]
    if not cases:
        raise ValueError('At least one evaluation case is required.')
    return cases


def make_signal(length: int, seed: int) -> np.ndarray:
    """評価用の複素乱数信号を生成する。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(length) + 1j * rng.standard_normal(length)


def aggregate_finite_metrics(
    checker: FiniteLengthPRChecker,
    eval_cases: list[tuple[int, int]],
    args: argparse.Namespace,
) -> dict[str, float]:
    """複数の有限長評価ケースから最悪誤差を集約する。"""
    metrics_list = []
    for length, seed in eval_cases:
        metrics_list.append(
            checker.check(
                make_signal(length, seed),
                pad_front=args.pad_front,
                pad_back=args.pad_back,
                crop_mode=args.crop_mode,
                crop_front=args.crop_front,
                crop_length=args.crop_length,
                valid_region_mode=args.valid_region_mode,
                valid_margin=args.valid_margin,
            )
        )
    return {
        'max_abs_error': max(item['max_abs_error'] for item in metrics_list),
        'rms_error': max(item['rms_error'] for item in metrics_list),
        'valid_max_abs_error': max(item['valid_max_abs_error'] for item in metrics_list),
        'valid_rms_error': max(item['valid_rms_error'] for item in metrics_list),
        'n_eval_cases': float(len(metrics_list)),
    }


def compute_score(
    *,
    exact_pr: bool,
    stopband_attenuation_db: float,
    pair_rms_error: float,
    pr_rms_error: float,
    valid_pr_rms_error: float,
    score_mode: str,
) -> float:
    """候補比較用の総合スコアを計算する。"""
    base = (-1e6 if not exact_pr else 0.0) + stopband_attenuation_db
    pair_term = 20.0 * np.log10(max(pair_rms_error, 1e-15))
    full_term = 20.0 * np.log10(max(pr_rms_error, 1e-15))
    valid_term = 20.0 * np.log10(max(valid_pr_rms_error, 1e-15))
    if score_mode == 'worst_full':
        return base - pair_term - full_term
    if score_mode == 'balanced':
        return base - pair_term - 0.5 * (full_term + valid_term)
    return base - pair_term - valid_term


def evaluate_candidate(
    *,
    structure: str,
    name: str,
    analysis_prototype: PrototypeFilter,
    synthesis_prototype: PrototypeFilter,
    delay: int,
    delay_unit: str,
    regularization: float,
    eval_cases: list[tuple[int, int]],
    args: argparse.Namespace,
    pair_designer: PrototypePairDesigner | PolyphasePRPairDesigner,
    pair_context: tuple[np.ndarray, int, int] | np.ndarray,
) -> PrototypeCandidateResult:
    """候補 1 件を評価し、指標と候補オブジェクトを返す。"""
    if structure == 'explicit_modulation':
        if not isinstance(pair_designer, PrototypePairDesigner) or not isinstance(pair_context, tuple):
            raise TypeError('explicit_modulation requires PrototypePairDesigner and cascade context.')
        analysis = PRDFTAnalysisBank(prototype=analysis_prototype)
        synthesis = PRDFTSynthesisBank(prototype=synthesis_prototype, delay_compensation=delay)
        pair_error = pair_designer.evaluate_pair_residual(
            analysis_prototype,
            synthesis_prototype,
            delay_samples=delay,
            cascade_matrix=pair_context,
        )
    else:
        if not isinstance(pair_designer, PolyphasePRPairDesigner) or not isinstance(pair_context, np.ndarray):
            raise TypeError('polyphase structure requires PolyphasePRPairDesigner and branch context.')
        analysis = PolyphasePRDFTAnalysisBank(prototype=analysis_prototype)
        synthesis = PolyphasePRDFTSynthesisBank(
            prototype=synthesis_prototype,
            delay_compensation=delay * analysis_prototype.decimation,
        )
        pair_error = pair_designer.evaluate_pair_residual(
            analysis_prototype,
            synthesis_prototype,
            delay_blocks=delay,
            branch_matrices=pair_context,
        )

    checker = PRChecker(analysis, synthesis)
    finite_checker = FiniteLengthPRChecker(analysis, synthesis)
    reference_signal = make_signal(args.n_samples, args.seed)
    pr = checker.check_perfect_reconstruction(reference_signal, length=reference_signal.shape[-1])
    finite_pr = aggregate_finite_metrics(finite_checker, eval_cases, args)
    response = checker.evaluate_prototype_response(analysis_prototype)
    exact_pr = pr['max_abs_error'] <= 1e-10 and pr['rms_error'] <= 1e-12
    score = compute_score(
        exact_pr=exact_pr,
        stopband_attenuation_db=response['stopband_attenuation_db'],
        pair_rms_error=pair_error['rms_error'],
        pr_rms_error=finite_pr['rms_error'],
        valid_pr_rms_error=finite_pr['valid_rms_error'],
        score_mode=args.score_mode,
    )

    cutoff = None
    if name.startswith('windowed_sinc@'):
        cutoff = float(name.split('@', 1)[1])

    return PrototypeCandidateResult(
        structure=structure,
        name=name,
        cutoff=cutoff,
        delay=delay,
        delay_unit=delay_unit,
        synthesis_prototype_length=synthesis_prototype.prototype_length,
        regularization=regularization,
        crop_mode=args.crop_mode,
        valid_region_mode=args.valid_region_mode,
        valid_margin=args.valid_margin,
        pad_front=args.pad_front,
        pad_back=args.pad_back,
        n_eval_cases=int(finite_pr['n_eval_cases']),
        passband_peak=response['passband_peak'],
        stopband_peak=response['stopband_peak'],
        stopband_attenuation_db=response['stopband_attenuation_db'],
        pair_max_abs_error=pair_error['max_abs_error'],
        pair_rms_error=pair_error['rms_error'],
        pr_max_abs_error=finite_pr['max_abs_error'],
        pr_rms_error=finite_pr['rms_error'],
        valid_pr_max_abs_error=finite_pr['valid_max_abs_error'],
        valid_pr_rms_error=finite_pr['valid_rms_error'],
        exact_pr=exact_pr,
        score=float(score),
    )


def optimize_prototype(args: argparse.Namespace) -> tuple[list[PrototypeCandidateResult], PrototypeFilter, PrototypeFilter]:
    """prototype 候補を探索し、最良の解析・合成係数を選定する。"""
    candidate_specs = build_candidate_specs(args)
    eval_cases = build_eval_cases(args)

    rows: list[PrototypeCandidateResult] = []
    best_analysis = candidate_specs[0][1]
    best_synthesis = candidate_specs[0][1]
    best_score = -np.inf

    pair_designer: PrototypePairDesigner | PolyphasePRPairDesigner
    if args.structure == 'explicit_modulation':
        pair_designer = PrototypePairDesigner(args.n_band, args.decimation)
        delay_grid = list(range(args.delay_start, args.delay_stop + 1, args.delay_step))
        synthesis_lengths = [args.prototype_length]
        delay_unit = 'sample'
    else:
        pair_designer = PolyphasePRPairDesigner(args.n_band, args.decimation)
        delay_grid = list(range(args.delay_block_start, args.delay_block_stop + 1, args.delay_block_step))
        synthesis_lengths = [length for length in args.synthesis_prototype_length_list if length % args.decimation == 0]
        if not synthesis_lengths:
            raise ValueError('synthesis-prototype-length-list must contain at least one multiple of decimation.')
        delay_unit = 'block'

    for name, analysis_prototype in candidate_specs:
        for synthesis_length in synthesis_lengths:
            if args.structure == 'explicit_modulation':
                if not isinstance(pair_designer, PrototypePairDesigner):
                    raise TypeError('explicit_modulation requires PrototypePairDesigner.')
                pair_context: tuple[np.ndarray, int, int] | np.ndarray = pair_designer.build_cascade_matrix(analysis_prototype)
                if not isinstance(pair_context, tuple):
                    raise TypeError('explicit modulation cascade context must be a tuple.')
                max_delay = pair_context[1] - pair_context[2]
            else:
                if not isinstance(pair_designer, PolyphasePRPairDesigner):
                    raise TypeError('polyphase structure requires PolyphasePRPairDesigner.')
                pair_context = pair_designer.build_branch_matrices(
                    analysis_prototype,
                    synthesis_prototype_length=synthesis_length,
                )
                max_delay = pair_context.shape[1] - 1
            valid_delays = [delay for delay in delay_grid if 0 <= delay <= max_delay]
            for delay in valid_delays:
                for regularization in args.regularization_list:
                    if args.structure == 'explicit_modulation':
                        if not isinstance(pair_designer, PrototypePairDesigner) or not isinstance(pair_context, tuple):
                            raise TypeError('explicit_modulation design context is invalid.')
                        explicit_designer: PrototypePairDesigner = pair_designer
                        synthesis_prototype = PrototypePairDesigner.design_synthesis_prototype(
                            explicit_designer,
                            analysis_prototype,
                            delay_samples=delay,
                            cascade_matrix=pair_context,
                            regularization=regularization,
                        )
                    else:
                        if not isinstance(pair_designer, PolyphasePRPairDesigner) or not isinstance(pair_context, np.ndarray):
                            raise TypeError('polyphase design context is invalid.')
                        synthesis_prototype = pair_designer.design_synthesis_prototype(
                            analysis_prototype,
                            delay_blocks=delay,
                            synthesis_prototype_length=synthesis_length,
                            branch_matrices=pair_context,
                            regularization=regularization,
                        )
                    row = evaluate_candidate(
                        structure=args.structure,
                        name=name,
                        analysis_prototype=analysis_prototype,
                        synthesis_prototype=synthesis_prototype,
                        delay=delay,
                        delay_unit=delay_unit,
                        regularization=regularization,
                        eval_cases=eval_cases,
                        args=args,
                        pair_designer=pair_designer,
                        pair_context=pair_context,
                    )
                    rows.append(row)
                    if row.score > best_score:
                        best_score = row.score
                        best_analysis = analysis_prototype
                        best_synthesis = synthesis_prototype

    rows.sort(key=lambda item: item.score, reverse=True)
    return rows, best_analysis, best_synthesis


def save_best_prototype(
    analysis_prototype: PrototypeFilter,
    synthesis_prototype: PrototypeFilter,
    best_row: PrototypeCandidateResult,
    artifact_name: str,
) -> tuple[Path, Path]:
    """最良 prototype の係数と評価結果を artifact に保存する。"""
    artifact_dir = ARTIFACT_ROOT / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    npz_path = artifact_dir / 'prototype_pair.npz'
    json_path = artifact_dir / 'prototype_pair.json'
    np.savez(
        npz_path,
        analysis_coefficients=analysis_prototype.coefficients,
        synthesis_coefficients=synthesis_prototype.coefficients,
        n_band=analysis_prototype.n_band,
        decimation=analysis_prototype.decimation,
        analysis_prototype_length=analysis_prototype.prototype_length,
        synthesis_prototype_length=synthesis_prototype.prototype_length,
        delay=best_row.delay,
        delay_unit=best_row.delay_unit,
        regularization=best_row.regularization,
        structure=best_row.structure,
        crop_mode=best_row.crop_mode,
        valid_region_mode=best_row.valid_region_mode,
        valid_margin=-1 if best_row.valid_margin is None else best_row.valid_margin,
        pad_front=-1 if best_row.pad_front is None else best_row.pad_front,
        pad_back=-1 if best_row.pad_back is None else best_row.pad_back,
        n_eval_cases=best_row.n_eval_cases,
    )
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(asdict(best_row), f, indent=2)
    return npz_path, json_path


def main() -> None:
    """PRDFT prototype 候補を探索し、最良係数を artifact に保存する。"""
    args = parse_args()
    rows, best_analysis, best_synthesis = optimize_prototype(args)
    best = rows[0]
    npz_path, json_path = save_best_prototype(best_analysis, best_synthesis, best, args.artifact_name)

    eval_cases = build_eval_cases(args)
    print('Prototype pair design / optimization / evaluation')
    print(
        f'structure={args.structure}, n_band={args.n_band}, decimation={args.decimation}, '
        f'prototype_length={args.prototype_length}, reference_n_samples={args.n_samples}, reference_seed={args.seed}'
    )
    print(f'eval_cases={eval_cases}')
    print(f'cutoff_scale_list={args.cutoff_scale_list}')
    if args.structure == 'explicit_modulation':
        print(f'delay_grid_samples={list(range(args.delay_start, args.delay_stop + 1, args.delay_step))}')
    else:
        print(f'delay_grid_blocks={list(range(args.delay_block_start, args.delay_block_stop + 1, args.delay_block_step))}')
        print(f'synthesis_prototype_length_list={args.synthesis_prototype_length_list}')
    print(f'regularization_list={args.regularization_list}')
    print(
        f'crop_mode={args.crop_mode}, crop_front={args.crop_front}, crop_length={args.crop_length}, '
        f'valid_region_mode={args.valid_region_mode}, valid_margin={args.valid_margin}, '
        f'pad_front={args.pad_front}, pad_back={args.pad_back}, score_mode={args.score_mode}'
    )
    print()
    print('Cause analysis')
    if args.structure == 'explicit_modulation':
        print('- This mode evaluates the earlier explicit modulation scaffold.')
        print('- A tiny multi-phase pair residual still does not guarantee finite-length PR.')
    else:
        print('- This mode evaluates the main-direction polyphase + DFT/IDFT structure.')
        print('- The pair designer enforces branch-wise polyphase convolution targets with a shared block delay.')
        print('- FiniteLengthPRChecker now separates crop_mode and valid_region_mode so the optimization target matches the intended use.')
    print()
    print('Top candidates')
    print('| rank | name | synth_len | delay | reg | exact_pr | stopband_att_db | pair_rms | valid_pr_rms | full_pr_rms | score |')
    print('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(rows[:10], start=1):
        print(
            f'| {idx} | {row.name} | {row.synthesis_prototype_length} | {row.delay} {row.delay_unit} | '
            f'{row.regularization:.1e} | {str(row.exact_pr)} | {row.stopband_attenuation_db:.3f} | '
            f'{row.pair_rms_error:.3e} | {row.valid_pr_rms_error:.3e} | {row.pr_rms_error:.3e} | {row.score:.3f} |'
        )

    print()
    print('Selected prototype pair')
    print(f'structure={best.structure}')
    print(f'name={best.name}')
    print(f'delay={best.delay} [{best.delay_unit}]')
    print(f'synthesis_prototype_length={best.synthesis_prototype_length}')
    print(f'regularization={best.regularization:.6e}')
    print(f'crop_mode={best.crop_mode}')
    print(f'valid_region_mode={best.valid_region_mode}')
    print(f'valid_margin={best.valid_margin}')
    print(f'n_eval_cases={best.n_eval_cases}')
    print(f'exact_pr={best.exact_pr}')
    print(f'stopband_attenuation_db={best.stopband_attenuation_db:.6f}')
    print(f'pair_max_abs_error={best.pair_max_abs_error:.6e}')
    print(f'pair_rms_error={best.pair_rms_error:.6e}')
    print(f'pr_max_abs_error={best.pr_max_abs_error:.6e}')
    print(f'pr_rms_error={best.pr_rms_error:.6e}')
    print(f'valid_pr_max_abs_error={best.valid_pr_max_abs_error:.6e}')
    print(f'valid_pr_rms_error={best.valid_pr_rms_error:.6e}')
    print(f'artifact_npz={npz_path}')
    print(f'artifact_json={json_path}')


if __name__ == '__main__':
    main()
