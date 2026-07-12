"""T1/T2必要tap評価の回帰試験。"""

from __future__ import annotations

from evaluations.beamforming.ebae_t1_t2_tap_requirement_sweep import (
    METHOD_IDS,
    SCENARIOS,
    TAP_COUNTS,
    calculate_tap_requirement_sweep,
)


def test_tap_sweep_separates_direction_estimation_and_fir_realization() -> None:
    """全条件に方位推定判定とFIR実現判定が独立して保存されることを確認する。"""
    detail_rows, minimum_rows = calculate_tap_requirement_sweep()

    assert len(detail_rows) == len(SCENARIOS) * len(METHOD_IDS) * len(TAP_COUNTS)
    assert len(minimum_rows) == len(SCENARIOS) * len(METHOD_IDS)
    required_fields = {
        "direction_estimation_pass",
        "fir_realization_pass",
        "common_window_start_sample",
        "target_energy_ratio",
        "relative_weight_error",
        "distortionless_level_error_db",
        "phase_rms_error_deg",
        "group_delay_rms_error_sample",
        "waveform_correlation",
        "waveform_normalized_rms_error",
        "full_mainlobe_width_deg",
        "sidelobe_degradation_db",
        "null_degradation_db",
    }
    assert required_fields <= detail_rows[0].keys()
    assert all(bool(row["direction_estimation_pass"]) for row in detail_rows)


def test_full_dft_length_is_an_exact_fir_realization_reference() -> None:
    """512 tapを打切り誤差ゼロの基準として全条件で合格させる。"""
    detail_rows, _ = calculate_tap_requirement_sweep()

    full_rows = [row for row in detail_rows if int(row["tap_count"]) == max(TAP_COUNTS)]
    assert len(full_rows) == len(SCENARIOS) * len(METHOD_IDS)
    assert all(bool(row["overall_pass"]) for row in full_rows)


def test_integer_delay_separation_never_requires_more_taps_than_direct_t1() -> None:
    """同一条件では整数遅延分離T2の最短tapが直接実現T1以下であることを確認する。"""
    _, minimum_rows = calculate_tap_requirement_sweep()
    lookup = {
        (str(row["scenario"]), str(row["method"])): int(row["minimum_passing_tap"])
        for row in minimum_rows
    }
    for scenario in SCENARIOS:
        assert lookup[(scenario.scenario_id, "T2")] <= lookup[(scenario.scenario_id, "T1")]
