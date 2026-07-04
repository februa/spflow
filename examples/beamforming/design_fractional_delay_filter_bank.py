"""小数遅延 FIR バンクを設計して保存するサンプル。"""

# 時間領域固定整相では、小数遅延 FIR をオンラインで都度設計するよりも、
# 事前設計して保存した係数群を読み出す方が、運用時の再現性と計算負荷の見通しを揃えやすい。

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow import design_fractional_delay_filter_bank


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解釈する。"""
    parser = argparse.ArgumentParser(description="小数遅延 FIR バンクを設計して .npz へ保存する。")
    parser.add_argument(
        "--n-frac-filter",
        type=int,
        default=33,
        help="小数遅延候補数。既定値は 33。",
    )
    parser.add_argument(
        "--n-tap",
        type=int,
        default=31,
        help="各 FIR のタップ長。既定値は 31。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "beamforming" / "fractional_delay_filter_bank.npz",
        help="保存先 .npz パス。",
    )
    return parser.parse_args()


def main() -> None:
    """設計した小数遅延 FIR バンクを保存する。"""
    args = parse_args()

    filter_bank = design_fractional_delay_filter_bank(
        n_frac_filter=args.n_frac_filter,
        n_tap=args.n_tap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filter_bank.save_npz(args.output)

    # 運用時はこの .npz を `FractionalDelayFilterBank.load_npz()` で読み出し、
    # `DelayTable.from_geometry(..., fractional_filter_bank=...)` へ渡す前提で使う。
    print(f"saved_fractional_delay_filter_bank={args.output}")
    print(f"n_frac_filter={filter_bank.n_frac_filter}")
    print(f"n_tap={filter_bank.n_tap}")


if __name__ == "__main__":
    main()
