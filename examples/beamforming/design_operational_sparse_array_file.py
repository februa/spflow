"""運用で使用するスパースアレイ定義ファイルを作成するサンプル。"""

# アレイ配置は実行時コードへ直書きせず、このスクリプトで JSON/CSV として保存する。
# 固定整相や SLC 側は JSON を読み込み、positions_m と n_ch をファイルから取得する。

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming.operational_sparse_array import (
    OperationalSparseArrayDesignConfig,
    save_operational_sparse_array,
)


def main() -> None:
    """fs=32768 Hz / 0-10 kHz 評価用の運用スパースアレイ定義を保存する。"""
    output_dir = ROOT / "artifacts" / "beamforming" / "operational_sparse_array"
    definition = save_operational_sparse_array(
        OperationalSparseArrayDesignConfig(
            output_json_path=output_dir / "operational_sparse_array_fs32768.json",
            output_csv_path=output_dir / "operational_sparse_array_design_table.csv",
            fs_hz=32768.0,
            sound_speed_m_s=1500.0,
            maximum_frequency_hz=10000.0,
            valid_frequency_hz_min=256.0,
            target_hpbw_deg=8.0,
            required_peak_margin_db=13.0,
        )
    )
    print(json.dumps(definition.to_payload(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
