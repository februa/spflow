"""運用アレイ向け時間領域 SLC 漏れ込み診断の回帰試験。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from evaluations.beamforming.scenarios.operational_time_domain_slc_diagnostics import (
    OperationalTimeDomainSlcDiagnosticConfig,
    run_operational_time_domain_slc_leakage_diagnostics,
)
from spflow.beamforming.operational_sparse_array import OperationalSparseArrayDefinition
from spflow.beamforming.time_delay import design_fractional_delay_filter_bank
from spflow.beamforming_evaluation import calculate_real_tone_response_rms_level_db20
from spflow.sidelobe_cancellation import SlcConfig


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    """summary の入れ子辞書を型検証して取り出す。

    Args:
        value: summary から取り出した値。
        name: エラー時に表示する項目名。

    Returns:
        文字列 key を持つ mapping。

    Raises:
        AssertionError: 値が mapping でない、または key が文字列でない場合。

    境界条件:
        診断 summary は JSON と同じ `dict[str, object]` を返すため、
        テストでも入れ子構造を確認してから metric や capacity を参照する。
    """
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


def _require_string(value: object, name: str) -> str:
    """summary の path / label を Python str として取り出す。"""
    if not isinstance(value, str):
        raise AssertionError(f"{name} must be a string.")
    return value




def test_real_tone_rms_level_uses_positive_and_negative_frequency_responses() -> None:
    """実信号 BL レベルが正負周波数応答の RMS 合成になることを確認する。

    複素 SLC 係数では `H(+f)` と `H(-f)` が非対称になり得る。
    正周波数だけで BL レベルを作ると時間波形 RMS と一致しないため、
    片側だけ応答する条件で -3.01 dB になることを固定する。
    """
    symmetric_level = calculate_real_tone_response_rms_level_db20(
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([1.0 + 0.0j], dtype=np.complex128),
        1.0,
    )
    positive_only_level = calculate_real_tone_response_rms_level_db20(
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([0.0 + 0.0j], dtype=np.complex128),
        1.0,
    )

    np.testing.assert_allclose(symmetric_level, np.array([0.0], dtype=np.float64), atol=1.0e-12)
    np.testing.assert_allclose(positive_only_level, np.array([-3.010299956639812], dtype=np.float64), atol=1.0e-12)

def test_operational_time_domain_slc_leakage_diagnostics_saves_summary() -> None:
    """時間領域 SLC 診断が固定整相後 beam output から共分散を作り、成分別 summary を保存することを確認する。"""
    positions_m = np.zeros((9, 3), dtype=np.float64)
    positions_m[:, 0] = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    active_indices = np.arange(9, dtype=np.int64)

    # このテストではアレイ設計の良否ではなく、診断関数の入出力契約を確認する。
    # そのため、全周波数で全 CH を active とする小さい一列アレイを一時 JSON として保存する。
    array_definition = OperationalSparseArrayDefinition(
        schema_version=1,
        fs_hz=8192.0,
        sound_speed_m_s=1500.0,
        valid_frequency_hz_min=512.0,
        maximum_frequency_hz=1024.0,
        positions_m=positions_m,
        design_frequencies_hz=np.array([0.0, 1024.0], dtype=np.float64),
        active_channel_indices_by_frequency=(active_indices, active_indices),
        records=(),
        formula={},
    )
    output_root = Path.cwd() / "artifacts" / "beamforming" / "operational_time_domain_slc_diagnostics_test"
    output_root.mkdir(parents=True, exist_ok=True)

    array_path = output_root / "array.json"
    array_definition.save_json(array_path)

    filter_bank_path = output_root / "fractional_delay_bank.npz"
    design_fractional_delay_filter_bank(n_frac_filter=17, n_tap=31).save_npz(filter_bank_path)

    summary = run_operational_time_domain_slc_leakage_diagnostics(
        config=OperationalTimeDomainSlcDiagnosticConfig(
            output_dir=output_root / "time_domain_slc",
            operational_array_definition_path=array_path,
            fractional_delay_filter_bank_path=filter_bank_path,
            processing_frequency_hz=1024.0,
            target_azimuth_deg=90.0,
            interferer_azimuth_deg=60.0,
            duration_s=0.125,
            n_beam_az_real=21,
            slc_analysis_block_size=256,
            # 正負周波数の desired blocking が効くことを確認するため、
            # interferer は数値上ほぼ無視できる -300 dB re input RMS にして target-only 条件へ寄せる。
            target_level_db20=0.0,
            interferer_level_db20=-300.0,
            noise_level_db20=-120.0,
        ),
        slc_config=SlcConfig(
            guard=2,
            loading=1.0e-2,
            memory_time_sec=1.0,
            heading_scale_deg=5.0,
            min_ref=4,
            sample_per_dof=2.0,
            tap_len=1,
            eta_normal=1.0,
            eta_limited=1.0,
            enable_heading_forgetting=False,
            enable_output_safety_gate=False,
        ),
    )

    summary_path = output_root / "time_domain_slc" / "time_domain_slc_leakage_summary.json"
    plot_path = Path(_require_string(summary["target_leakage_levels_png_path"], "target_leakage_levels_png_path"))
    waveform_overlay_path = Path(
        _require_string(
            summary["slc_before_after_waveform_overlay_png_path"],
            "slc_before_after_waveform_overlay_png_path",
        )
    )
    spectrum_overlay_path = Path(
        _require_string(
            summary["slc_component_spectrum_overlay_png_path"],
            "slc_component_spectrum_overlay_png_path",
        )
    )
    target_response_bl_overlay_path = Path(
        _require_string(
            summary["protected_target_response_bl_overlay_png_path"],
            "protected_target_response_bl_overlay_png_path",
        )
    )
    interferer_response_bl_overlay_path = Path(
        _require_string(
            summary["protected_target_interferer_response_bl_overlay_png_path"],
            "protected_target_interferer_response_bl_overlay_png_path",
        )
    )
    levels = _require_mapping(summary["levels"], "levels")
    protected_bl_summary = _require_mapping(summary["protected_target_bl_summary"], "protected_target_bl_summary")
    slc_process = _require_mapping(summary["slc_process"], "slc_process")
    capacity = _require_mapping(slc_process["capacity"], "slc_process.capacity")

    assert summary_path.exists()
    assert plot_path.exists()
    assert waveform_overlay_path.exists()
    assert spectrum_overlay_path.exists()
    assert target_response_bl_overlay_path.exists()
    assert interferer_response_bl_overlay_path.exists()
    assert protected_bl_summary["definition"] == "protected target beam fixed; x-axis is source azimuth, not output beam index"
    assert "interferer_frequency_reduction_at_interferer_db" in protected_bl_summary
    assert "bl_improvement_pass" in protected_bl_summary
    assert "slc_bl_improvement_pass" in summary
    target_sidelobe_metrics = _require_mapping(
        protected_bl_summary["target_frequency_sidelobe_metrics"],
        "protected_target_bl_summary.target_frequency_sidelobe_metrics",
    )
    interferer_sidelobe_metrics = _require_mapping(
        protected_bl_summary["interferer_frequency_sidelobe_metrics"],
        "protected_target_bl_summary.interferer_frequency_sidelobe_metrics",
    )
    assert _require_number(target_sidelobe_metrics["guard_outside_peak_delta_db"], "target guard outside delta") <= 1.0
    assert "first_sidelobe_reduction_db" in target_sidelobe_metrics
    assert "first_sidelobe_reduction_db" in interferer_sidelobe_metrics
    assert _require_number(interferer_sidelobe_metrics["reduction_at_marker_db"], "interferer marker reduction") > 0.0
    assert "max_guard_outside_worsening_db" in interferer_sidelobe_metrics

    # SLC 前後の BL 評価は、干渉 marker の一点だけでは合格にしない。
    # guard 外 peak、最大局所悪化、第一副極の 3 条件を同時に満たすときだけ改善ありと判定する。
    expected_bl_improvement_pass = (
        _require_number(interferer_sidelobe_metrics["guard_outside_peak_delta_db"], "interferer guard outside delta") < 0.0
        and _require_number(interferer_sidelobe_metrics["max_guard_outside_worsening_db"], "interferer max worsening") <= 0.0
        and _require_number(interferer_sidelobe_metrics["first_sidelobe_reduction_db"], "interferer first sidelobe reduction") > 0.0
    )
    assert protected_bl_summary["bl_improvement_pass"] == expected_bl_improvement_pass
    assert summary["slc_bl_improvement_pass"] == expected_bl_improvement_pass
    assert int(_require_number(summary["n_beam"], "n_beam")) == 21
    assert int(_require_number(summary["active_channel_count"], "active_channel_count")) == 9
    assert summary["level_reference"] == "dB re input RMS"
    assert "raw_interferer_reduction_db" in levels
    assert "effective_interferer_reduction_db" in levels
    assert "safety_fallback_required" in summary
    assert _require_number(levels["raw_target_power_delta_db"], "raw_target_power_delta_db") > -0.5
    assert bool(capacity["is_feasible"])
    covariance_memory = _require_mapping(slc_process["covariance_memory"], "slc_process.covariance_memory")
    condition_stats = _require_mapping(
        slc_process["block_condition_number_stats"],
        "slc_process.block_condition_number_stats",
    )
    weight_stats = _require_mapping(slc_process["block_weight_norm_stats"], "slc_process.block_weight_norm_stats")
    alpha_stats = _require_mapping(slc_process["block_alpha_stats"], "slc_process.block_alpha_stats")

    assert slc_process["condition_number_matrix"] == "R_uu + loading * mean(diag(R_uu)) I"
    assert slc_process["covariance_integration"] == "exponential_forgetting_by_block"
    assert int(_require_number(slc_process["analysis_block_size"], "slc_process.analysis_block_size")) == 256
    assert _require_number(slc_process["condition_number"], "slc_process.condition_number") >= 1.0
    assert _require_number(covariance_memory["block_time_sec"], "covariance_memory.block_time_sec") > 0.0
    assert _require_number(covariance_memory["asymptotic_effective_independent_block_count"], "effective_block_count") > 1.0
    assert int(_require_number(condition_stats["count"], "condition_stats.count")) >= 1
    assert int(_require_number(weight_stats["count"], "weight_stats.count")) >= 1
    assert int(_require_number(alpha_stats["count"], "alpha_stats.count")) >= 1
    assert _require_number(slc_process["elapsed_sec"], "slc_process.elapsed_sec") >= 0.0
    assert _require_number(slc_process["realtime_factor"], "slc_process.realtime_factor") < 1.0
