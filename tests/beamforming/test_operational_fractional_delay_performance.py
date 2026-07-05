"""運用スパースアレイでの小数遅延固定整相性能評価に関する回帰試験。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from spflow.beamforming.operational_fractional_delay_performance import (
    OperationalArrayFractionalDelayPerformanceConfig,
    run_operational_array_fractional_delay_performance_report,
)
from spflow.beamforming.operational_sparse_array import OperationalSparseArrayDesignConfig, save_operational_sparse_array
from spflow.beamforming.time_delay import design_fractional_delay_filter_bank


def test_operational_array_fractional_delay_performance_uses_file_active_channels() -> None:
    """運用アレイ JSON の active channel を使い、小数遅延整相が全評価周波数で 13 dB 条件を満たすことを確認する。"""
    output_root = Path.cwd() / "artifacts" / "beamforming" / "operational_fractional_delay_performance_test"
    array_json_path = output_root / "array.json"
    filter_bank_path = output_root / f"fractional_delay_filter_bank_{os.getpid()}.npz"
    output_root.mkdir(parents=True, exist_ok=True)

    # テストでは scan 点数だけ軽くし、設計式・active channel ファイル読込・小数遅延評価の接続を確認する。
    save_operational_sparse_array(
        OperationalSparseArrayDesignConfig(
            output_json_path=array_json_path,
            scan_azimuth_count=901,
        )
    )
    design_fractional_delay_filter_bank(n_frac_filter=65, n_tap=63).save_npz(filter_bank_path)

    summary = run_operational_array_fractional_delay_performance_report(
        OperationalArrayFractionalDelayPerformanceConfig(
            output_dir=output_root / "report",
            operational_array_definition_path=array_json_path,
            fractional_delay_filter_bank_path=filter_bank_path,
            frequency_grid_hz=(256.0, 512.0, 1024.0, 4096.0, 10000.0),
            evaluation_azimuths_deg=(60.0, 90.0, 120.0),
            comparison_specs=((10000.0, 60.0),),
        )
    )

    assert Path(str(summary["performance_summary_json_path"])).exists()
    assert Path(str(summary["performance_table_csv_path"])).exists()
    assert Path(str(summary["margin_summary_png_path"])).exists()
    assert bool(summary["fractional_meets_required_margin_all"])
    assert int(summary["physical_array_n_ch"]) == 125

    active_channel_count = np.asarray(summary["active_channel_count"], dtype=np.int64)
    active_aperture_m = np.asarray(summary["active_aperture_m"], dtype=np.float64)
    fractional_worst_margin_db = np.asarray(summary["fractional_worst_margin_db"], dtype=np.float64)
    assert np.all(fractional_worst_margin_db >= 13.0)
    assert active_aperture_m[0] > active_aperture_m[-1]
    assert active_channel_count.size == 5
