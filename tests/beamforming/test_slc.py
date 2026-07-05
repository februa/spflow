"""ビーム領域 SLC 部品に関する回帰試験。"""

from __future__ import annotations

import numpy as np
from spflow.beamforming.slc import BeamDomainSLC, BeamGuardSelector, BlockLeastSquaresSlcSolver, SlcConfig, SlcReferenceCapacityChecker




def test_slc_solver_relative_loading_is_invariant_to_input_power_scale() -> None:
    """SLC の対角 loading が入力 power スケールに依存しないことを確認する。

    共分散 `R_uu` と相互相関 `r_ud` は入力振幅を k 倍すると power として k^2 倍される。
    loading が絶対値だと入力レベルだけで解が変わるため、平均対角 power に対する比として扱う。
    """
    covariance = np.array([[2.0, 0.1], [0.1, 1.0]], dtype=np.complex128)
    cross_correlation = np.array([[0.8 + 0.2j, 0.3 - 0.1j]], dtype=np.complex128)
    solver = BlockLeastSquaresSlcSolver(loading=3.0e-2)

    base_weights = solver.solve(covariance, cross_correlation)
    scaled_weights = solver.solve(9.0 * covariance, 9.0 * cross_correlation)

    np.testing.assert_allclose(scaled_weights, base_weights, rtol=1.0e-12, atol=1.0e-12)

def test_beam_guard_selector_make_reference_beams_excludes_target_and_guard_region():
    """guard 付き target 指定から参照ビーム列を正しく作れることを確認する。"""
    selector = BeamGuardSelector(n_beam=10, guard=1)

    protected_mask = selector.make_protected_mask(np.array([3, 7], dtype=np.int64))
    reference_beams = selector.make_reference_beams(np.array([3, 7], dtype=np.int64))

    expected_mask = np.array([False, False, True, True, True, False, True, True, True, False], dtype=bool)
    np.testing.assert_array_equal(protected_mask, expected_mask)
    np.testing.assert_array_equal(reference_beams, np.array([0, 1, 5, 9], dtype=np.int64))


def test_beam_domain_slc_with_time_taps_reduces_delayed_reference_interference() -> None:
    """時間タップ付き SLC が delayed reference 成分を FIR 型に推定できることを確認する。

    target beam には reference beam の 2 sample 遅れ成分を混入させる。L=1 では同時刻の
    reference だけしか使えないが、L=3 では `u[n-2]` も自由度に含まれるため、
    FIR 型 SLC として干渉を下げられる。
    """
    n_sample = 2048
    time_axis = np.arange(n_sample, dtype=np.float64)
    desired = np.cos(2.0 * np.pi * 0.051 * time_axis)
    reference = np.cos(2.0 * np.pi * 0.173 * time_axis + 0.4)

    delayed_reference = np.zeros_like(reference)
    delayed_reference[2:] = reference[:-2]
    beam_output = np.stack([desired + 0.7 * delayed_reference, reference, np.zeros_like(reference)], axis=0)

    slc = BeamDomainSLC(
        n_beam=3,
        fs_hz=2048.0,
        block_size=n_sample,
        config=SlcConfig(
            guard=0,
            loading=1.0e-5,
            memory_time_sec=1.0,
            heading_scale_deg=5.0,
            min_ref=1,
            sample_per_dof=1.0,
            tap_len=3,
            eta_normal=1.0,
            eta_limited=1.0,
            enable_heading_forgetting=False,
            enable_output_safety_gate=False,
        ),
    )

    result = slc.process(beam_output=beam_output, target_beams=np.array([0], dtype=np.int64))

    assert result.mode == "NORMAL"
    assert result.W is not None
    assert result.W.shape == (1, result.reference_beams.size * 3)
    assert result.capacity.tap_len == 3

    # 先頭 2 サンプルは full tap が揃わないため固定整相出力を通す。
    np.testing.assert_allclose(result.Y[0, :2], beam_output[0, :2])

    before_interference = np.mean((beam_output[0, 2:] - desired[2:]) ** 2)
    after_interference = np.mean((np.asarray(np.real_if_close(result.Y[0, 2:]), dtype=np.float64) - desired[2:]) ** 2)
    assert after_interference < before_interference * 0.2


def test_beam_domain_slc_reduces_reference_correlated_interference_without_disabling():
    """reference に強く相関した干渉がある場合、SLC 後に target 出力の干渉が減ることを確認する。"""
    n_sample = 1024
    time_axis = np.arange(n_sample, dtype=np.float64)

    desired = np.cos(2.0 * np.pi * 0.07 * time_axis)
    interferer = np.cos(2.0 * np.pi * 0.19 * time_axis + 0.3)

    beam_output = np.stack(
        [
            desired + 0.8 * interferer,
            interferer,
            np.zeros_like(interferer),
        ],
        axis=0,
    ).astype(np.float64)

    slc = BeamDomainSLC(
        n_beam=3,
        fs_hz=1024.0,
        block_size=n_sample,
        config=SlcConfig(
            guard=0,
            loading=1.0e-4,
            memory_time_sec=1.0,
            heading_scale_deg=5.0,
            min_ref=1,
            sample_per_dof=1.0,
            tap_len=1,
            eta_normal=1.0,
            eta_limited=0.5,
            enable_heading_forgetting=False,
        ),
    )
    result = slc.process(beam_output=beam_output, target_beams=np.array([0], dtype=np.int64))

    capacity_checker = SlcReferenceCapacityChecker(min_ref=1, sample_per_dof=1.0, tap_len=1)
    capacity = capacity_checker.check(n_ref=result.reference_beams.size, block_size=n_sample)

    assert result.mode == "NORMAL"
    assert capacity.is_feasible

    before_interference = np.mean((beam_output[0] - desired) ** 2)
    after_interference = np.mean((np.asarray(np.real_if_close(result.Y[0]), dtype=np.float64) - desired) ** 2)
    assert after_interference < before_interference * 0.2




def test_beam_domain_slc_falls_back_when_output_power_drops_too_much() -> None:
    """target 自己消去が疑われる場合、SLC 出力ではなく固定整相出力へ戻すことを確認する。"""
    n_sample = 1024
    time_axis = np.arange(n_sample, dtype=np.float64)
    desired = np.cos(2.0 * np.pi * 0.07 * time_axis)

    # reference beam に desired と同じ信号が入る条件は、target absent が保証されない運用で起こり得る危険側条件である。
    # このとき LS-SLC は desired 自身を説明してしまうため、出力パワー急減を safety gate で検出する。
    beam_output = np.stack([desired, desired], axis=0).astype(np.float64)
    slc = BeamDomainSLC(
        n_beam=2,
        fs_hz=1024.0,
        block_size=n_sample,
        config=SlcConfig(
            guard=0,
            loading=1.0e-6,
            memory_time_sec=1.0,
            heading_scale_deg=5.0,
            min_ref=1,
            sample_per_dof=1.0,
            tap_len=1,
            eta_normal=1.0,
            eta_limited=1.0,
            enable_heading_forgetting=False,
            enable_output_safety_gate=True,
            max_output_power_drop_db=3.0,
        ),
    )

    result = slc.process(beam_output=beam_output, target_beams=np.array([0], dtype=np.int64))

    assert result.mode == "SAFETY_FALLBACK"
    assert result.eta == 0.0
    assert result.safety is not None
    assert result.safety.fallback_required
    assert "output_power_drop" in result.safety.reasons
    np.testing.assert_allclose(result.Y[0], beam_output[0])



def test_beam_domain_slc_with_desired_response_blocking_preserves_target_only_signal() -> None:
    """desired 応答 blocking を使うと、target-only 条件で自己消去しないことを確認する。"""
    n_sample = 1024
    time_axis = np.arange(n_sample, dtype=np.float64)
    desired = np.cos(2.0 * np.pi * 0.07 * time_axis)

    # reference beam 1 には target sidelobe が混入している想定にする。
    # desired_response_matrix により beam 1 の desired 応答を blocking できれば、target-only で SLC は動かない。
    beam_output = np.stack([desired, 0.5 * desired, np.zeros_like(desired)], axis=0).astype(np.float64)
    desired_response_matrix = np.array([[1.0], [0.5], [0.0]], dtype=np.complex128)
    slc = BeamDomainSLC(
        n_beam=3,
        fs_hz=1024.0,
        block_size=n_sample,
        config=SlcConfig(
            guard=0,
            loading=1.0e-4,
            memory_time_sec=1.0,
            heading_scale_deg=5.0,
            min_ref=1,
            sample_per_dof=1.0,
            tap_len=1,
            eta_normal=1.0,
            eta_limited=1.0,
            enable_heading_forgetting=False,
            enable_output_safety_gate=False,
        ),
    )

    result = slc.process(
        beam_output=beam_output,
        target_beams=np.array([0], dtype=np.int64),
        desired_response_matrix=desired_response_matrix,
    )

    assert result.mode == "NORMAL"
    assert result.reference_blocking_matrix is not None
    np.testing.assert_allclose(np.asarray(np.real_if_close(result.Y[0]), dtype=np.float64), desired, atol=1.0e-3)
