"""scene_renderer信号へT2a-EBAEだけを適用する逐次評価を実行する。

MATLAB独自raw係数の読込、周波数別active channel、T共分散、EBAE重み、
通常Pythonによるblock逐次処理、BL/FRAZ/FL成果物を一つのCLIから生成する。
`fixed_baseline`と`t2a_mvdr`の独立branchは生成しない。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from evaluations.beamforming.scene_renderer_t2a_streaming import (  # noqa: E402
    T2aScenarioConfig,
    run_evaluation,
    write_example_matlab_coefficients,
)


def _parse_args() -> argparse.Namespace:
    """MATLAB独自raw係数と成果物保存先をCLIから取得する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions-raw", type=Path, required=True, help="COE_POS相当raw")
    parser.add_argument("--shading-raw", type=Path, required=True, help="COE_CBFSHADING相当raw")
    parser.add_argument(
        "--shading-frequency-step-hz",
        type=float,
        required=True,
        help="shading列の周波数間隔[Hz]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/beamforming/t2a_ebae_scene_renderer_streaming/review_pack"),
    )
    parser.add_argument(
        "--write-example-coefficients",
        action="store_true",
        help="実行前に疎通確認用rawを指定2ファイルへ生成する",
    )
    return parser.parse_args()


def main() -> None:
    """MATLAB係数読込からT2a-EBAE単独評価までを実行する。

    CLI入力は位置raw、複素shading raw、shading周波数間隔[Hz]、保存先である。
    戻り値はなく、BL/FRAZ/FL、CSV、NPZ、metadataを保存先へ書き出す。

    Raises:
        ValueError: raw係数、scenario、T2a-EBAE設計または逐次処理契約が不正な場合。

    境界条件:
        EBAE内部のAIC/MUSICまたは適応重みが成立しない場合は、未完成重みを公開せず
        同じactive channelの固定整相重みへfallbackする。これは独立した
        `fixed_baseline`方式を実行することとは区別する。
    """
    args = _parse_args()
    config = T2aScenarioConfig()
    if bool(args.write_example_coefficients):
        write_example_matlab_coefficients(args.positions_raw, args.shading_raw, config)
    run_evaluation(
        args.positions_raw,
        args.shading_raw,
        args.shading_frequency_step_hz,
        args.output_dir,
        config,
        method_ids=("t2a_ebae",),
        review_title="T2a-EBAE only scene_renderer streaming review pack",
    )


if __name__ == "__main__":
    main()
