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
    """fs=32768 Hz 条件の疎配置を保存し、使用側が CH 数をファイルから読めることを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "operational_sparse_array_test"
    json_path = output_dir / "array.json"
    csv_path = output_dir / "array.csv"
    config = OperationalSparseArrayDesignConfig(
        output_json_path=json_path,
        output_csv_path=csv_path,
        scan_azimuth_count=901,
    )

    # 0 Hz は方位性能を定義できないため、正周波数 record だけで 13 dB 条件を確認する。
    definition = save_operational_sparse_array(config)
    loaded = load_operational_sparse_array(json_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert isinstance(loaded, OperationalSparseArrayDefinition)
    assert loaded.n_ch == definition.n_ch
    assert loaded.n_ch == int(loaded.positions_m.shape[0])
    assert loaded.aperture_m >= 40.0

    positive_records = [record for record in loaded.records if float(record["frequency_hz"]) > 0.0]
    assert len(positive_records) > 0
    assert all(bool(record["meets_required_peak_margin"]) for record in positive_records)
    assert all(float(record["worst_peak_margin_db"]) >= 13.0 for record in positive_records)

    high_frequency_indices = loaded.active_channel_indices_for_frequency(10000.0)
    low_frequency_indices = loaded.active_channel_indices_for_frequency(256.0)
    assert high_frequency_indices.ndim == 1
    assert low_frequency_indices.ndim == 1

    # 低域は少数の sparse subset で長開口を確保し、高域は中心密配置で alias を避ける。
    # そのため CH 数の大小ではなく、active aperture が低域で長いことを確認する。
    low_active_aperture_m = np.ptp(loaded.positions_m[low_frequency_indices, 0])
    high_active_aperture_m = np.ptp(loaded.positions_m[high_frequency_indices, 0])
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
