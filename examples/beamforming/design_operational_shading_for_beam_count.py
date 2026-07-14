"""指定ビーム数で 3 dB down 幅に近い Kaiser-Bessel シェーディング係数を保存する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spflow.array_design import (
    OperationalFixedBeamShadingDesignConfig,
    run_operational_fixed_beam_shading_design,
)


def main() -> None:
    """ビーム数を固定した周波数別 shading 係数ファイルを作成する。"""
    parser = argparse.ArgumentParser(
        description="指定ビーム数で -3 dB 主ローブ幅がビーム間隔程度になる Kaiser-Bessel shading を設計する。",
    )
    parser.add_argument(
        "--n-beam-az-real",
        type=int,
        default=151,
        help="固定する実待受方位数。既定値は 151。",
    )
    parser.add_argument(
        "--target-overlap-margin-deg",
        type=float,
        default=0.0,
        help="-3 dB 主ローブ overlap の目標値。0 deg なら隣接 -3 dB 境界が接する条件。",
    )
    parser.add_argument(
        "--target-overlap-tolerance-deg",
        type=float,
        default=0.5,
        help="目標 overlap からの許容誤差。単位は deg。",
    )
    args = parser.parse_args()

    output_dir = Path("artifacts/beamforming/operational_shading_fixed_beam")
    output_prefix = f"operational_kaiser_bessel_shading_{int(args.n_beam_az_real):03d}beam"

    # 指定ビーム数が多すぎる場合、Kaiser-Bessel 窓では主ローブを狭くできない。
    # その場合も最も近い beta を保存し、summary の meets_three_db_width_target=false で設計限界を明示する。
    summary = run_operational_fixed_beam_shading_design(
        OperationalFixedBeamShadingDesignConfig(
            operational_array_definition_path=Path(
                "artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json"
            ),
            n_beam_az_real=int(args.n_beam_az_real),
            target_overlap_margin_deg=float(args.target_overlap_margin_deg),
            target_overlap_tolerance_deg=float(args.target_overlap_tolerance_deg),
            output_json_path=output_dir / f"{output_prefix}.json",
            output_csv_path=output_dir / f"{output_prefix}_table.csv",
            output_summary_png_path=output_dir / f"{output_prefix}_summary.png",
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
