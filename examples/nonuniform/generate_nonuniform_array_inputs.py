"""非均一 example 用の受波器位置・シェーディング ndarray を生成する。"""

# 非均一木構造では分割仕様と streaming 状態の組み合わせで挙動が大きく変わるため、
# 実運用に近い入出力条件を一式そろえて可視化・書き出しできる例として管理する。

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from examples.nonuniform._array_input_support import build_default_array_design, save_array_inputs


def build_parser() -> argparse.ArgumentParser:
    """CLI 引数定義を返す。"""
    parser = argparse.ArgumentParser(
        description='非均一ビームフォーミング example 用の array ndarray を生成する。',
    )
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts' / 'nonuniform_array_inputs')
    parser.add_argument('--fs-hz', type=float, default=32768.0)
    parser.add_argument('--sound-speed', type=float, default=343.0)
    parser.add_argument('--n-dense-ch', type=int, default=24)
    parser.add_argument('--dense-spacing-m', type=float, default=0.01)
    parser.add_argument('--n-outer-pairs', type=int, default=4)
    parser.add_argument('--outer-spacing-m', type=float, default=0.04)
    parser.add_argument('--aperture-wavelengths', type=float, default=4.0)
    parser.add_argument('--min-active-ch', type=int, default=4)
    return parser


def main() -> None:
    """指定条件から ndarray 入力群を生成して保存する。"""
    args = build_parser().parse_args()
    design = build_default_array_design(
        fs_hz=args.fs_hz,
        sound_speed=args.sound_speed,
        n_dense_ch=args.n_dense_ch,
        dense_spacing_m=args.dense_spacing_m,
        n_outer_pairs=args.n_outer_pairs,
        outer_spacing_m=args.outer_spacing_m,
        aperture_wavelengths=args.aperture_wavelengths,
        min_active_ch=args.min_active_ch,
    )
    save_array_inputs(
        args.output_dir,
        design,
        fs_hz=args.fs_hz,
        sound_speed=args.sound_speed,
    )
    print(f'output_dir={args.output_dir}')
    print(f'n_ch={design.n_ch}')
    print(f'n_band={design.n_band}')
    print(f'active_channel_counts_per_band={design.active_channel_counts_per_band().tolist()}')


if __name__ == '__main__':
    main()
