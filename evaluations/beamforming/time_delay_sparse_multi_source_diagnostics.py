"""スパース1列アレイで複数方位・複数周波数の整数遅延固定整相診断を保存する。"""

# 単一音源だけでは、複数方位・複数周波数が同時に存在する実運用寄り条件での
# BL/FRAZ/BTR の見え方を確認できないため、ここではスパース片舷アレイを使った
# 複数音源診断シナリオを固定サンプルとして提供する。

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
    """複数方位・複数周波数の整数遅延固定整相診断を実行する。"""
    output_dir = ROOT / "artifacts" / "beamforming" / "time_delay_sparse_multi_source_diagnostics"
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
                TimeDelayDiagnosticSource(azimuth_deg=20.0, frequency_hz=1024.0, level_db20=0.0, label="A"),
                TimeDelayDiagnosticSource(azimuth_deg=58.0, frequency_hz=1536.0, level_db20=0.0, label="B"),
                TimeDelayDiagnosticSource(azimuth_deg=112.0, frequency_hz=2304.0, level_db20=0.0, label="C"),
            ),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
