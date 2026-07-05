"""小数遅延固定整相の性能確認サンプル。"""

# 保存済み小数遅延 FIR バンクを読み込み、整数遅延固定整相と比較して
# スパース片舷アレイの高域 off-broadside 性能が改善することを確認する。

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming.fractional_delay_performance import (
    FractionalDelayPerformanceConfig,
    run_fractional_delay_performance_report,
)


def build_reference_sparse_positions() -> np.ndarray:
    """69 ch 片舷スパースアレイ座標を返す。"""
    positive_indices = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 27, 31, 36, 42, 49, 57, 66, 76, 87, 99, 112, 126],
        dtype=np.float64,
    )
    sensor_index = np.concatenate([-positive_indices[::-1], [0.0], positive_indices])
    positions_m = np.zeros((sensor_index.size, 3), dtype=np.float64)
    positions_m[:, 0] = 0.05 * sensor_index
    return positions_m


def main() -> None:
    """小数遅延固定整相の性能比較レポートを保存する。"""
    filter_bank_path = ROOT / "artifacts" / "beamforming" / "fractional_delay_filter_bank_65x63.npz"
    if not filter_bank_path.exists():
        raise FileNotFoundError(
            "fractional delay filter bank not found. Run examples/beamforming/design_fractional_delay_filter_bank.py "
            "with --n-frac-filter 65 --n-tap 63 first."
        )

    output_dir = ROOT / "artifacts" / "beamforming" / "fractional_delay_performance"
    summary = run_fractional_delay_performance_report(
        FractionalDelayPerformanceConfig(
            output_dir=output_dir,
            array_positions_m=build_reference_sparse_positions(),
            fractional_delay_filter_bank_path=filter_bank_path,
            fs_hz=32768.0,
            sound_speed_m_s=1500.0,
            frequency_grid_hz=(512.0, 1024.0, 2048.0, 3072.0, 4096.0, 6144.0, 8192.0, 10000.0),
            evaluation_azimuths_deg=(60.0, 90.0, 120.0),
            n_beam_az_real=151,
            comparison_specs=((10000.0, 60.0), (10000.0, 90.0)),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
