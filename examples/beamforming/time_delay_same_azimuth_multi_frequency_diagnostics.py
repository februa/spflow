"""スパース 1 列アレイで同一方位・異周波数の整数遅延固定整相診断を保存するサンプル。"""

# 同一方位に異なる周波数が重なると、固定整相としては方位 ridge は 1 本に見えつつ、
# FRAZ では周波数別のピークが分離して見える必要がある。その条件を確認するための
# 固定サンプルをここで提供する。

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming.time_delay_diagnostics import (
    TimeDelayDiagnosticConfig,
    TimeDelayDiagnosticSource,
    run_integer_delay_diagnostics,
)


def main() -> None:
    """同一方位・異周波数の整数遅延固定整相診断を実行する。"""
    output_dir = ROOT / "artifacts" / "beamforming" / "time_delay_same_azimuth_multi_frequency_diagnostics"
    summary = run_integer_delay_diagnostics(
        TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            fs_hz=32768.0,
            duration_s=1.0,
            sound_speed_m_s=1500.0,
            noise_level_db20=-45.0,
            random_seed=1234,
            array_n_ch=61,
            array_sensor_spacing_m=0.05,
            sparse_stride_pattern=(1, 2, 1, 3, 1, 4, 2, 5),
            n_beam_az_real=241,
            btr_block_size=1024,
            source_specs=(
                TimeDelayDiagnosticSource(azimuth_deg=58.0, frequency_hz=1024.0, level_db20=0.0, label="F1"),
                TimeDelayDiagnosticSource(azimuth_deg=58.0, frequency_hz=1536.0, level_db20=0.0, label="F2"),
                TimeDelayDiagnosticSource(azimuth_deg=58.0, frequency_hz=2304.0, level_db20=0.0, label="F3"),
            ),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
