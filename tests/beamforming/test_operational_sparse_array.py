"""運用スパースアレイ定義ファイルの設計・読込に関する回帰試験。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spflow.beamforming.operational_sparse_array import (
    OperationalSparseArrayDefinition,
    OperationalSparseArrayDesignConfig,
    design_operational_sparse_array,
    load_operational_sparse_array,
    save_operational_sparse_array,
)


def test_operational_sparse_array_design_save_and_load() -> None:
    """300 m 級疎配置を保存し、200 Hz 未満は全 CH、200 Hz 以上は周波数別 active 開口になることを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "operational_sparse_array_test"
    json_path = output_dir / "array.json"
    csv_path = output_dir / "array.csv"
    config = OperationalSparseArrayDesignConfig(
        output_json_path=json_path,
        output_csv_path=csv_path,
        scan_azimuth_count=901,
    )

    definition = save_operational_sparse_array(config)
    loaded = load_operational_sparse_array(json_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert isinstance(loaded, OperationalSparseArrayDefinition)
    assert loaded.n_ch == definition.n_ch
    assert loaded.n_ch == int(loaded.positions_m.shape[0])
    assert 290 <= loaded.n_ch <= 320
    assert loaded.aperture_m >= 300.0

    records_by_frequency = {float(record["frequency_hz"]): record for record in loaded.records}
    assert float(records_by_frequency[10.0]["active_channel_count"]) == float(loaded.n_ch)
    assert float(records_by_frequency[128.0]["active_channel_count"]) == float(loaded.n_ch)
    assert float(records_by_frequency[200.0]["active_aperture_m"]) >= 300.0
    assert 1.0 <= float(records_by_frequency[200.0]["target_hpbw_deg"]) <= 1.5

    high_frequency_indices = loaded.active_channel_indices_for_frequency(10000.0)
    low_frequency_indices = loaded.active_channel_indices_for_frequency(10.0)
    assert high_frequency_indices.ndim == 1
    assert low_frequency_indices.ndim == 1
    assert low_frequency_indices.size == loaded.n_ch

    # 10 Hz では全 CH の 300 m 級開口を使い、10 kHz では中心密配置だけに縮める。
    # この active aperture の切替により、高域で外側疎配置の大間隔が grating lobe を作ることを避ける。
    low_active_aperture_m = np.ptp(loaded.positions_m[low_frequency_indices, 0])
    high_active_aperture_m = np.ptp(loaded.positions_m[high_frequency_indices, 0])
    assert float(low_active_aperture_m) >= 300.0
    assert float(high_active_aperture_m) < 10.0
    assert float(low_active_aperture_m) > float(high_active_aperture_m)

    # 使用側では n_ch を引数で別管理せず、ファイル内 positions_m から決める。
    bandwise_design = loaded.to_bandwise_array_design()
    assert bandwise_design.n_ch == loaded.n_ch
    np.testing.assert_allclose(bandwise_design.channel_positions_m, loaded.positions_m.astype(np.float32))


def test_operational_sparse_array_design_without_file_io() -> None:
    """設計関数単体でも shape と保存 payload が安定していることを確認する。"""
    config = OperationalSparseArrayDesignConfig(
        output_json_path=Path("unused.json"),
        scan_azimuth_count=721,
    )
    definition = design_operational_sparse_array(config)
    payload = definition.to_payload()
    restored = OperationalSparseArrayDefinition.from_payload(payload)

    assert restored.n_ch == definition.n_ch
    assert restored.design_frequencies_hz.shape == definition.design_frequencies_hz.shape
    assert len(restored.active_channel_indices_by_frequency) == restored.design_frequencies_hz.size