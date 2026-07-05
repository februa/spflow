"""運用アレイ向け同一方位・複数周波数診断の回帰試験。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from spflow.beamforming import (
    OperationalSameAzimuthFrequencySeparationConfig,
    run_operational_same_azimuth_frequency_separation_diagnostics,
)
from spflow.beamforming.operational_sparse_array import OperationalSparseArrayDefinition
from spflow.beamforming.time_delay import design_fractional_delay_filter_bank


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    """summary の入れ子 mapping を型検証して返す。"""
    if not isinstance(value, Mapping):
        raise AssertionError(f"{name} must be a mapping.")
    for key in value.keys():
        if not isinstance(key, str):
            raise AssertionError(f"{name} keys must be strings.")
    return value


def _require_number(value: object, name: str) -> float:
    """summary の数値 metric を Python float として取り出す。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{name} must be numeric.")
    return float(value)


def test_operational_same_azimuth_frequency_separation_saves_summary() -> None:
    """同一方位の複数周波数を固定整相後の単一 tone 射影で分けられることを確認する。"""
    positions_m = np.zeros((9, 3), dtype=np.float64)
    positions_m[:, 0] = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    active_indices = np.arange(9, dtype=np.int64)

    # このテストはアレイ設計の最適性ではなく、周波数成分分離 summary の入出力契約を見る。
    # 全周波数で同じ active subset を使い、時間領域固定整相後の target beam を評価する。
    array_definition = OperationalSparseArrayDefinition(
        schema_version=1,
        fs_hz=8192.0,
        sound_speed_m_s=1500.0,
        valid_frequency_hz_min=512.0,
        maximum_frequency_hz=2048.0,
        positions_m=positions_m,
        design_frequencies_hz=np.array([0.0, 2048.0], dtype=np.float64),
        active_channel_indices_by_frequency=(active_indices, active_indices),
        records=(),
        formula={},
    )
    output_root = Path.cwd() / "artifacts" / "beamforming" / "operational_same_azimuth_frequency_separation_test"
    output_root.mkdir(parents=True, exist_ok=True)

    array_path = output_root / "array.json"
    array_definition.save_json(array_path)

    filter_bank_path = output_root / "fractional_delay_bank.npz"
    design_fractional_delay_filter_bank(n_frac_filter=17, n_tap=31).save_npz(filter_bank_path)

    summary = run_operational_same_azimuth_frequency_separation_diagnostics(
        OperationalSameAzimuthFrequencySeparationConfig(
            output_dir=output_root / "same_azimuth_frequency",
            operational_array_definition_path=array_path,
            fractional_delay_filter_bank_path=filter_bank_path,
            processing_frequency_hz=2048.0,
            source_azimuth_deg=90.0,
            source_frequencies_hz=(512.0, 1024.0, 1536.0),
            source_levels_db20=(0.0, 0.0, 0.0),
            duration_s=0.125,
            noise_level_db20=-120.0,
            n_beam_az_real=21,
            btr_block_size=256,
        )
    )

    summary_path = output_root / "same_azimuth_frequency" / "same_azimuth_frequency_separation_summary.json"
    frequency_levels = summary["frequency_levels"]

    assert summary_path.exists()
    assert summary["level_reference"] == "dB re input RMS"
    assert summary["evaluation_pattern"] == "slc_same_azimuth_multi_frequency"
    assert int(_require_number(summary["active_channel_count"], "active_channel_count")) == 9
    assert _require_number(summary["analysis_bandwidth_hz"], "analysis_bandwidth_hz") == 8.0
    assert isinstance(frequency_levels, list)
    assert len(frequency_levels) == 3

    for index, item in enumerate(frequency_levels):
        metrics = _require_mapping(item, f"frequency_levels[{index}]")
        assert abs(_require_number(metrics["target_frequency_power_delta_db"], "target_frequency_power_delta_db")) < 1.0
        assert _require_number(metrics["frequency_bin_leakage_db"], "frequency_bin_leakage_db") < -20.0

    assert _require_number(summary["worst_frequency_bin_leakage_db"], "worst_frequency_bin_leakage_db") < -20.0
