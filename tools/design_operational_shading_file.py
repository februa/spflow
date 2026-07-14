"""運用スパースアレイ用の周波数別 Kaiser-Bessel シェーディング係数を保存するCLI。"""

from __future__ import annotations

import json
from pathlib import Path

from spflow.array_design import OperationalShadingDesignConfig, run_operational_shading_design


def main() -> None:
    """運用アレイ JSON を入力し、周波数別 shading 係数ファイルを作成する。"""
    output_dir = Path("artifacts/beamforming/operational_shading")
    summary = run_operational_shading_design(
        OperationalShadingDesignConfig(
            operational_array_definition_path=Path(
                "artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json"
            ),
            output_json_path=output_dir / "operational_kaiser_bessel_shading_fs32768.json",
            output_csv_path=output_dir / "operational_kaiser_bessel_shading_table.csv",
            output_summary_png_path=output_dir / "operational_kaiser_bessel_shading_summary.png",
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
