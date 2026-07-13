"""強弱近接信号可視性評価の契約を確認する。"""

import numpy as np
import pytest
from numpy.typing import NDArray

from evaluations.beamforming import alignment_weak_source_visibility_sweep as evaluation
from evaluations.beamforming.alignment_weak_source_visibility_sweep import Source


def test_visibility_metrics_are_finite_for_two_sources() -> None:
    """2信号条件でS/T・EBAE/MVDRの可視性指標が計算できる。"""
    curves, _ = evaluation._curves((Source(90.0, 1.0), Source(80.0, 0.1)), 16.0)
    assert set(curves) == {"ebae_S", "ebae_T", "mvdr_S", "mvdr_T"}
    for curve in curves.values():
        error, prominence, meets_candidate_rule = (
            evaluation._uncalibrated_visibility_observations(curve, 80.0, 90.0)
        )
        assert error >= 0.0
        assert prominence >= 0.0
        # 視覚校正前なので「可視性」ではなく、候補規則に適合したかだけを保持する。
        assert isinstance(meets_candidate_rule, bool)


def test_zero_degree_t_covariance_uses_candidate_aligned_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0 deg候補もS共分散へ退化せず、候補方位別T共分散を使用する。"""
    sources = (Source(0.0, 1.0), Source(10.0, 0.1))
    delays_s = evaluation._delays(evaluation.AZIMUTH_DEG)
    steering = evaluation._steering(delays_s)
    expected_t_covariance = evaluation._covariance(
        sources,
        delays_s,
        steering,
        16.0,
        0,
    )
    s_covariance = evaluation._covariance(
        sources,
        delays_s,
        steering,
        16.0,
        None,
    )

    candidate_indices: list[int | None] = []
    original_covariance = evaluation._covariance

    def record_candidate_index(
        source_items: tuple[Source, ...],
        scan_delays_s: NDArray[np.float64],
        scan_steering: NDArray[np.complex128],
        analysis_width_hz: float,
        candidate_index: int | None,
    ) -> NDArray[np.complex128]:
        """S/T共分散の呼出順を記録し、元の数式へそのまま委譲する。"""
        candidate_indices.append(candidate_index)
        return original_covariance(
            source_items,
            scan_delays_s,
            scan_steering,
            analysis_width_hz,
            candidate_index,
        )

    monkeypatch.setattr(evaluation, "_covariance", record_candidate_index)
    evaluation._curves(sources, 16.0)

    # 先頭はS共分散、次が0 deg候補のT共分散でなければならない。
    assert candidate_indices[:2] == [None, 0]
    assert not np.allclose(expected_t_covariance, s_covariance)


def test_source_count_sweep_covers_one_to_three_sources() -> None:
    """信号数sweepが期待信号数1～3をS/T双方で評価する。"""
    rows = evaluation.calculate_source_count_sweep()
    assert {int(row["expected_source_count"]) for row in rows} == {1, 2, 3}
    assert {str(row["covariance_method"]) for row in rows} == {"S", "T"}
