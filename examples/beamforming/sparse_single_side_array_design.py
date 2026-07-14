"""片舷スパースアレイ設計レポートを保存するサンプル。"""

# 0 Hz-10 kHz を対象に、中央密・外周疎の 1 列片舷アレイを設計し、
# CH 数、開口長、受波器間隔、exact-delay 幾何性能、現在の整数遅延制約をまとめて保存する。

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.array_design import (  # noqa: E402
    SparseSingleSideArrayDesignConfig,
    run_sparse_single_side_array_design,
)


def main() -> None:
    """片舷スパースアレイ設計レポートを生成する。"""
    output_dir = ROOT / "artifacts" / "beamforming" / "sparse_single_side_array_design"
    summary = run_sparse_single_side_array_design(
        SparseSingleSideArrayDesignConfig(
            output_dir=output_dir,
            fs_hz=32768.0,
            sound_speed_m_s=1500.0,
            dense_spacing_m=0.05,
            dense_center_positive_sensor_count=20,
            outer_positive_sensor_indices=(
                22,
                24,
                27,
                31,
                36,
                42,
                49,
                57,
                66,
                76,
                87,
                99,
                112,
                126,
            ),
            design_frequency_grid_hz=(
                0.0,
                512.0,
                1024.0,
                2048.0,
                3072.0,
                4096.0,
                6144.0,
                8192.0,
                10000.0,
            ),
            required_peak_margin_db=13.0,
            exact_delay_evaluation_azimuths_deg=(60.0, 90.0, 120.0),
            integer_delay_evaluation_azimuths_deg=(60.0, 90.0, 120.0),
            n_exact_delay_scan_azimuth=2401,
            n_integer_delay_beam_azimuth=151,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
