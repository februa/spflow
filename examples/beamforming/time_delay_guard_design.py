"""固定整相だけの BL から周波数依存 guard 設計表を保存するサンプル。"""

# 低周波から高周波まで SLC を掛けずに固定整相の BL を測定し、
# mainlobe 外ピークが 13 dB 以上低くなる最小 guard を周波数ごとに保存する。

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming.time_delay_guard_design import TimeDelayGuardDesignConfig, run_integer_delay_guard_design


def main() -> None:
    """151 ビーム条件で周波数依存 guard 設計を実行する。"""
    output_dir = ROOT / "artifacts" / "beamforming" / "time_delay_guard_design"
    summary = run_integer_delay_guard_design(
        TimeDelayGuardDesignConfig(
            output_dir=output_dir,
            fs_hz=32768.0,
            duration_s=1.0,
            sound_speed_m_s=1500.0,
            target_azimuth_deg=20.0,
            noise_level_db20=-120.0,
            array_n_ch=61,
            array_sensor_spacing_m=0.05,
            sparse_stride_pattern=(1, 2, 1, 3, 1, 4, 2, 5),
            n_beam_az_real=151,
            frequency_start_hz=512.0,
            frequency_stop_hz=4096.0,
            n_frequency=15,
            required_peak_margin_db=13.0,
            half_power_drop_db=3.0,
            peak_search_half_width_beam=4,
            guard_safety_margin_beams=0,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
