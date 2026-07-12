"""強弱近接信号可視性評価の契約を確認する。"""

from evaluations.beamforming.alignment_weak_source_visibility_sweep import (
    _curves,
    _visibility,
    calculate_source_count_sweep,
    Source,
)


def test_visibility_metrics_are_finite_for_two_sources() -> None:
    """2信号条件でS/T・EBAE/MVDRの可視性指標が計算できる。"""
    curves, _ = _curves((Source(90.0, 1.0), Source(80.0, 0.1)), 16.0)
    assert set(curves) == {"ebae_S", "ebae_T", "mvdr_S", "mvdr_T"}
    for curve in curves.values():
        error, prominence, visible = _visibility(curve, 80.0, 90.0)
        assert error >= 0.0
        assert prominence >= 0.0
        assert isinstance(visible, bool)


def test_source_count_sweep_covers_one_to_three_sources() -> None:
    """信号数sweepが期待信号数1～3をS/T双方で評価する。"""
    rows = calculate_source_count_sweep()
    assert {int(row["expected_source_count"]) for row in rows} == {1, 2, 3}
    assert {str(row["covariance_method"]) for row in rows} == {"S", "T"}
