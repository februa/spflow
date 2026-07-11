"""時間領域 SLC と MVDR / LCMV / GSC 比較診断の回帰試験。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from spflow.beamforming import (
    OperationalTimeDomainAdaptiveComparisonConfig,
    SlcConfig,
    run_operational_time_domain_adaptive_comparison,
)
from spflow.beamforming.operational_sparse_array import OperationalSparseArrayDefinition
from spflow.beamforming.time_delay import design_fractional_delay_filter_bank


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    """JSON summary の入れ子辞書を型安全に取り出す。"""
    if not isinstance(value, dict):
        raise AssertionError(f"{name} must be a mapping.")
    return value


def _require_number(value: object, name: str) -> float:
    """JSON summary の scalar 数値を float として取り出す。"""
    if not isinstance(value, int | float):
        raise AssertionError(f"{name} must be numeric.")
    return float(value)


def test_operational_time_domain_adaptive_comparison_reports_before_after_bl_metrics(tmp_path: Path) -> None:
    """SLC baseline と MVDR/LCMV/GSC の before/after BL 改善量を比較できることを確認する。

    方式比較では、固定整相前後のビーム応答重ね書きがないと改善量を判断できない。
    そのため summary には各方式の target/interferer 周波数 BL 指標と PNG パスを必須にする。
    """
    positions_m = np.zeros((9, 3), dtype=np.float64)
    positions_m[:, 0] = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    active_indices = np.arange(9, dtype=np.int64)

    # この試験の責務は適応方式の summary 契約であり、事前に生成された運用 artifact の有無ではない。
    # 全周波数で同じ 9 CH を使う小規模 ULA を一時領域へ保存し、単独実行でも同じ条件を再現できるようにする。
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
    array_path = tmp_path / "array.json"
    array_definition.save_json(array_path)

    # 17 相×31 tap は回帰試験時間を抑えつつ、小数遅延整相を整数遅延へ退化させない最小限の試験条件である。
    filter_bank_path = tmp_path / "fractional_delay_bank.npz"
    design_fractional_delay_filter_bank(n_frac_filter=17, n_tap=31).save_npz(filter_bank_path)

    summary = run_operational_time_domain_adaptive_comparison(
        config=OperationalTimeDomainAdaptiveComparisonConfig(
            output_dir=tmp_path / "comparison",
            operational_array_definition_path=array_path,
            fractional_delay_filter_bank_path=filter_bank_path,
            processing_frequency_hz=1536.0,
            target_azimuth_deg=90.0,
            interferer_azimuth_deg=60.0,
            interferer_frequency_hz=1024.0,
            target_level_db20=0.0,
            interferer_level_db20=-6.0,
            duration_s=0.25,
            n_beam_az_real=21,
            tap_len=2,
            diagonal_loading=3.0e-2,
            btr_block_size=256,
        ),
        slc_config=SlcConfig(
            guard=2,
            loading=3.0e-2,
            memory_time_sec=3.0,
            heading_scale_deg=5.0,
            min_ref=4,
            sample_per_dof=2.0,
            tap_len=1,
            eta_normal=1.0,
            eta_limited=1.0,
            enable_heading_forgetting=False,
        ),
    )

    assert summary["evaluation_pattern"] == "time_domain_adaptive_mvdr_lcmv_gsc"
    slc_baseline = _require_mapping(summary["slc_baseline"], "slc_baseline")
    assert "protected_target_bl_summary" in slc_baseline

    methods = _require_mapping(summary["adaptive_methods"], "adaptive_methods")
    assert {
        "time_domain_mvdr_real",
        "time_domain_lcmv_target_interferer_null",
        "time_domain_gsc_equivalent_lcmv",
    } <= set(methods)

    for method_name, raw_method_summary in methods.items():
        method_summary = _require_mapping(raw_method_summary, method_name)
        protected_summary = _require_mapping(method_summary["protected_target_bl_summary"], f"{method_name}.protected")
        target_metrics = _require_mapping(
            protected_summary["target_frequency_sidelobe_metrics"],
            f"{method_name}.target_frequency_sidelobe_metrics",
        )
        interferer_metrics = _require_mapping(
            protected_summary["interferer_frequency_sidelobe_metrics"],
            f"{method_name}.interferer_frequency_sidelobe_metrics",
        )
        covariance_health = _require_mapping(method_summary["covariance_health"], f"{method_name}.covariance_health")
        constraint_response = _require_mapping(method_summary["constraint_response"], f"{method_name}.constraint_response")

        assert Path(str(method_summary["target_frequency_bl_overlay_png_path"])).exists()
        assert Path(str(method_summary["interferer_frequency_bl_overlay_png_path"])).exists()
        assert _require_number(protected_summary["target_frequency_delta_at_target_db"], "target delta") <= 0.5
        assert "interferer_frequency_exact_reduction_at_interferer_db" in protected_summary
        assert "interferer_frequency_reduction_at_interferer_db" in protected_summary
        assert "guard_outside_peak_delta_db" in target_metrics
        assert "guard_outside_peak_delta_db" in interferer_metrics
        assert "first_sidelobe_reduction_db" in target_metrics
        assert "first_sidelobe_reduction_db" in interferer_metrics
        assert _require_number(covariance_health["loaded_condition_number"], "condition number") >= 1.0
        assert _require_number(constraint_response["max_target_constraint_error_db20"], "target constraint error") < -100.0
