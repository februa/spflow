"""運用スパースアレイの小数遅延固定整相性能を評価する。"""

# アレイ CH 数と位置は JSON から読み込む。周波数ごとの active channel も JSON の設計表に従う。
# 小数遅延 FIR は別スクリプトで保存済みの .npz を読み込み、オンライン設計しない。

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluations.beamforming.scenarios.operational_fractional_delay_performance import (
    OperationalArrayFractionalDelayPerformanceConfig,
    run_operational_array_fractional_delay_performance_report,
)


def main() -> None:
    """運用スパースアレイ JSON を使った小数遅延固定整相評価を実行する。"""
    array_definition_path = (
        ROOT
        / "artifacts"
        / "beamforming"
        / "operational_sparse_array"
        / "operational_sparse_array_fs32768.json"
    )
    filter_bank_path = ROOT / "artifacts" / "beamforming" / "fractional_delay_filter_bank_65x63.npz"
    if not array_definition_path.exists():
        raise FileNotFoundError(
            "operational sparse array JSON not found. "
            "Run tools/design_operational_sparse_array_file.py first."
        )
    if not filter_bank_path.exists():
        raise FileNotFoundError(
            "fractional delay filter bank not found. "
            "Run tools/design_fractional_delay_filter_bank.py first."
        )

    summary = run_operational_array_fractional_delay_performance_report(
        OperationalArrayFractionalDelayPerformanceConfig(
            output_dir=ROOT / "artifacts" / "beamforming" / "operational_fractional_delay_performance",
            operational_array_definition_path=array_definition_path,
            fractional_delay_filter_bank_path=filter_bank_path,
            fs_hz=32768.0,
            sound_speed_m_s=1500.0,
            required_peak_margin_db=13.0,
            n_beam_az_real=151,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
