"""片舷スパースアレイ設計レポートの回帰試験。"""

from __future__ import annotations

import json
from pathlib import Path

from spflow.beamforming.sparse_single_side_array_design import (
    SparseSingleSideArrayDesignConfig,
    build_sparse_single_side_array_design,
    run_sparse_single_side_array_design,
)


def test_sparse_single_side_array_design_build_and_report() -> None:
    """0 Hz-10 kHz 設計表と評価図を保存し、幾何設計と整数遅延制約が要約へ反映されることを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "sparse_single_side_array_design_test"
    config = SparseSingleSideArrayDesignConfig(
        output_dir=output_dir,
        fs_hz=32768.0,
        sound_speed_m_s=1500.0,
        dense_spacing_m=0.05,
        dense_center_positive_sensor_count=20,
        outer_positive_sensor_indices=(22, 24, 27, 31, 36, 42, 49, 57, 66, 76, 87, 99, 112, 126),
        design_frequency_grid_hz=(0.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 10000.0),
        required_peak_margin_db=13.0,
        exact_delay_evaluation_azimuths_deg=(60.0, 90.0, 120.0),
        integer_delay_evaluation_azimuths_deg=(60.0, 90.0, 120.0),
        n_exact_delay_scan_azimuth=1801,
        n_integer_delay_beam_azimuth=151,
    )

    # この条件は、dense center 41 ch と outer sparse 28 ch で 10 kHz までの幾何性能を見つつ、
    # 現在の整数遅延実装が高域 off-broadside で制約になることも同時に検出できる代表条件として選ぶ。
    design_result = build_sparse_single_side_array_design(config)
    summary = run_sparse_single_side_array_design(config)

    assert int(design_result.n_ch) == 69
    assert abs(float(summary["array_aperture_m"]) - 12.6) <= 1e-6
    assert abs(float(summary["array_min_sensor_spacing_m"]) - 0.05) <= 1e-6

    json_path = output_dir / "design_summary.json"
    csv_path = output_dir / "design_table.csv"
    geometry_png_path = output_dir / "array_geometry.png"
    aperture_png_path = output_dir / "effective_aperture_summary.png"
    margin_png_path = output_dir / "sector_margin_summary.png"
    notes_path = output_dir / "design_notes.md"

    assert json_path.exists()
    assert csv_path.exists()
    assert geometry_png_path.exists()
    assert aperture_png_path.exists()
    assert margin_png_path.exists()
    assert notes_path.exists()

    saved_summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_summary["design_summary_json_path"] == summary["design_summary_json_path"]
    assert saved_summary["design_table_csv_path"] == summary["design_table_csv_path"]
    assert saved_summary["array_geometry_png_path"] == summary["array_geometry_png_path"]
    assert saved_summary["effective_aperture_summary_png_path"] == summary["effective_aperture_summary_png_path"]
    assert saved_summary["sector_margin_summary_png_path"] == summary["sector_margin_summary_png_path"]

    records = summary["records"]
    assert isinstance(records, list)
    assert len(records) == 7

    positive_frequency_records = [record for record in records if float(record["frequency_hz"]) > 0.0]
    assert all(bool(record["exact_delay_meets_required_peak_margin"]) for record in positive_frequency_records)

    # 8 kHz 以上の off-broadside では、現在の整数遅延だけでは 13 dB 条件を満たしにくい。
    # この差分が summary に保存されていれば、小数遅延導入前後の設計比較に再利用できる。
    highest_frequency_record = positive_frequency_records[-1]
    assert float(highest_frequency_record["exact_delay_worst_sector_peak_margin_db"]) >= 13.0
    assert float(highest_frequency_record["integer_delay_worst_sector_peak_margin_db"]) < 13.0

    for bl_png_path in summary["bl_comparison_png_paths"]:
        assert Path(str(bl_png_path)).exists()

