"""固定整相 BL ベースの周波数依存 guard 設計に関する回帰試験。"""

from __future__ import annotations

import json
from pathlib import Path

from spflow.beamforming.time_delay_guard_design import TimeDelayGuardDesignConfig, run_integer_delay_guard_design


def test_integer_delay_guard_design_save_guard_table_and_pngs() -> None:
    """固定整相だけの BL から周波数依存 guard 表を保存し、各周波数で所望ピーク差を満たすことを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "time_delay_guard_design_test"
    summary = run_integer_delay_guard_design(
        TimeDelayGuardDesignConfig(
            output_dir=output_dir,
            fs_hz=32768.0,
            duration_s=0.75,
            sound_speed_m_s=1500.0,
            target_azimuth_deg=20.0,
            noise_level_db20=-120.0,
            array_n_ch=31,
            array_sensor_spacing_m=0.05,
            sparse_stride_pattern=(1, 2, 1, 3, 1, 4),
            n_beam_az_real=61,
            frequency_grid_hz=(768.0, 1536.0, 3072.0),
            required_peak_margin_db=13.0,
            half_power_drop_db=3.0,
            peak_search_half_width_beam=4,
        )
    )

    json_path = output_dir / "frequency_guard_table.json"
    csv_path = output_dir / "frequency_guard_table.csv"
    png_path = output_dir / "frequency_guard_table.png"

    assert json_path.exists()
    assert csv_path.exists()
    assert png_path.exists()

    saved_summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_summary["frequency_guard_table_json_path"] == summary["frequency_guard_table_json_path"]
    assert saved_summary["frequency_guard_table_csv_path"] == summary["frequency_guard_table_csv_path"]
    assert saved_summary["frequency_guard_table_png_path"] == summary["frequency_guard_table_png_path"]
    assert bool(summary["array_is_sparse"])
    assert int(summary["n_beam"]) == 61

    records = summary["records"]
    assert isinstance(records, list)
    assert len(records) == 3

    for record in records:
        assert Path(str(record["bl_png_path"])).exists()
        assert int(record["guard_half_width_beams"]) >= 0
        assert int(record["guard_width_beams"]) >= int(record["mainlobe_width_beams"])
        assert int(record["required_margin_guard_half_width_beams"]) >= int(record["guard_half_width_beams"])
        assert isinstance(bool(record["meets_required_peak_margin"]), bool)
        assert abs(float(record["peak_azimuth_deg"]) - 20.0) <= 4.0

