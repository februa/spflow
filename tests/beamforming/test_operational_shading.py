"""運用スパースアレイ用 Kaiser-Bessel シェーディング設計の回帰試験。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spflow.beamforming import (
    OperationalFixedBeamShadingDesignConfig,
    OperationalShadingDefinition,
    OperationalShadingDesignConfig,
    OperationalSparseArrayDesignConfig,
    load_operational_shading,
    run_operational_fixed_beam_shading_design,
    run_operational_shading_design,
    save_operational_sparse_array,
)


def test_operational_shading_design_saves_frequency_dependent_coefficients() -> None:
    """周波数ごとの active 配置で Kaiser-Bessel 係数を設計し、隣接 -3 dB 主ローブ重なり条件を満たすことを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "operational_shading_test"
    array_json_path = output_dir / "array.json"
    shading_json_path = output_dir / "shading.json"
    shading_csv_path = output_dir / "shading.csv"

    # テストでは評価周波数を絞るが、入力アレイは本番と同じ設計式で作る。
    # これにより、周波数ごとの active channel 読込と shading 係数保存の接続を確認する。
    array_definition = save_operational_sparse_array(
        OperationalSparseArrayDesignConfig(
            output_json_path=array_json_path,
            output_csv_path=output_dir / "array.csv",
            design_frequency_grid_hz=(0.0, 256.0, 1024.0, 10000.0),
            scan_azimuth_count=721,
        )
    )

    summary = run_operational_shading_design(
        OperationalShadingDesignConfig(
            operational_array_definition_path=array_json_path,
            output_json_path=shading_json_path,
            output_csv_path=shading_csv_path,
            frequency_grid_hz=(256.0, 1024.0, 10000.0),
            candidate_kaiser_beta=(2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0),
            candidate_n_beam_az_real=(151, 181),
        )
    )
    loaded = load_operational_shading(shading_json_path)

    assert shading_json_path.exists()
    assert shading_csv_path.exists()
    assert isinstance(loaded, OperationalShadingDefinition)
    assert bool(summary["meets_all"])
    assert loaded.n_ch == array_definition.n_ch
    assert loaded.shading_coefficients_by_frequency.shape == (3, array_definition.n_ch)

    for frequency_hz, record in zip(summary["frequency_grid_hz"], summary["records"], strict=True):
        active_indices = loaded.active_channel_indices_for_frequency(float(frequency_hz))
        coefficients = loaded.coefficients_for_frequency(float(frequency_hz))
        inactive_mask = np.ones(coefficients.shape[0], dtype=bool)
        inactive_mask[active_indices] = False

        # active 外の係数を 0 にしておけば、全 CH 入力へそのまま掛けても
        # 周波数ごとの active subset だけを使う挙動になる。
        assert bool(np.all(coefficients[inactive_mask] == 0.0))
        assert np.isclose(float(np.max(coefficients[active_indices])), 1.0)
        assert float(record["minimum_three_db_overlap_margin_deg"]) >= 0.0
        assert float(record["minimum_three_db_width_deg"]) > 0.0
        assert float(record["worst_peak_margin_db"]) >= 13.0
        assert float(record["effective_channel_count"]) <= float(record["active_channel_count"])
        assert float(record["snr_loss_vs_rectangular_db"]) >= 0.0
        assert float(record["worst_first_sidelobe_level_db_re_mainlobe_peak"]) <= 0.0
        assert float(record["worst_sidelobe_95_percentile_db_re_mainlobe_peak"]) <= 0.0
        assert float(record["worst_integrated_sidelobe_level_db_re_mainlobe_peak"]) <= 0.0
        assert int(record["selected_n_beam_az_real"]) >= 151



def test_operational_fixed_beam_shading_reports_width_match_limit() -> None:
    """指定ビーム数で 3 dB 幅一致を狙い、細かすぎるビーム数では限界を summary に残すことを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "operational_fixed_beam_shading_test"
    array_json_path = output_dir / "array.json"
    shading_json_path = output_dir / "fixed_151.json"

    array_definition = save_operational_sparse_array(
        OperationalSparseArrayDesignConfig(
            output_json_path=array_json_path,
            output_csv_path=output_dir / "array.csv",
            design_frequency_grid_hz=(0.0, 256.0, 10000.0),
            scan_azimuth_count=721,
        )
    )

    summary = run_operational_fixed_beam_shading_design(
        OperationalFixedBeamShadingDesignConfig(
            operational_array_definition_path=array_json_path,
            output_json_path=shading_json_path,
            output_csv_path=output_dir / "fixed_151.csv",
            n_beam_az_real=151,
            frequency_grid_hz=(256.0, 10000.0),
            candidate_kaiser_beta=(0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0),
            target_overlap_margin_deg=0.0,
            target_overlap_tolerance_deg=0.5,
        )
    )
    loaded = load_operational_shading(shading_json_path)

    assert loaded.n_ch == array_definition.n_ch
    assert loaded.selected_n_beam_az_real_by_frequency.tolist() == [151, 151]
    assert all(int(record["selected_n_beam_az_real"]) == 151 for record in summary["records"])
    assert all(float(record["worst_peak_margin_db"]) >= 13.0 for record in summary["records"])
    assert all(float(record["effective_channel_count"]) <= float(record["active_channel_count"]) for record in summary["records"])
    assert all(float(record["worst_sidelobe_99_percentile_db_re_mainlobe_peak"]) <= 0.0 for record in summary["records"])

    # 151 本固定では margin 確保のため非ゼロ beta が必要になるが、
    # 3 dB overlap target 0 deg には届かないため、幅一致条件は未達として記録される。
    assert any(float(beta) > 0.0 for beta in summary["selected_kaiser_beta_by_frequency"])
    assert not bool(summary["meets_all"])
    assert all(not bool(record["meets_three_db_width_target"]) for record in summary["records"])
