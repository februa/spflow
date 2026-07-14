"""時間領域 MVDR / LCMV / GSC 部品に関する回帰試験。"""

from __future__ import annotations

import numpy as np

from spflow.beamforming import (
    apply_time_domain_fir_beamformer,
    build_gsc_blocking_matrix,
    build_time_domain_tone_constraint_vector,
    build_time_tapped_snapshot_matrix,
    design_time_domain_gsc_coefficients,
    design_time_domain_lcmv_coefficients,
    design_time_domain_mvdr_coefficients,
    diagnose_time_domain_adaptive_weights,
    estimate_time_domain_covariance,
    evaluate_constraint_response,
)


def _linear_array_steering(n_ch: int, spatial_frequency_rad: float) -> np.ndarray:
    """単純な線形アレイ steering を作る。"""
    channel_index = np.arange(n_ch, dtype=np.float64)
    return np.exp(1j * float(spatial_frequency_rad) * channel_index).astype(np.complex128)


def test_time_domain_mvdr_preserves_target_constraint() -> None:
    """時間領域 MVDR が target tone の歪みなし制約を満たすことを確認する。

    干渉共分散だけが強い条件を作り、MVDR が target 制約 `w^H c_t = 1` を保ったまま
    干渉方向の応答を固定平均重みより下げることを見る。
    """
    tap_len = 3
    target_constraint = build_time_domain_tone_constraint_vector(
        _linear_array_steering(n_ch=4, spatial_frequency_rad=0.15),
        frequency_hz=1000.0,
        fs_hz=8000.0,
        tap_len=tap_len,
    )
    interferer_constraint = build_time_domain_tone_constraint_vector(
        _linear_array_steering(n_ch=4, spatial_frequency_rad=1.1),
        frequency_hz=1300.0,
        fs_hz=8000.0,
        tap_len=tap_len,
    )

    # 共分散は干渉 tone の rank-1 成分に白色雑音を加えたものとする。
    # MVDR はこの R に対して target 制約下の出力 power を最小化する。
    covariance = 10.0 * interferer_constraint[:, np.newaxis] @ interferer_constraint.conj()[np.newaxis, :]
    covariance += 0.1 * np.eye(interferer_constraint.size, dtype=np.complex128)
    weights = design_time_domain_mvdr_coefficients(covariance, target_constraint, diagonal_loading=1.0e-4)

    target_response = evaluate_constraint_response(weights, target_constraint[:, np.newaxis])
    interferer_response = evaluate_constraint_response(weights, interferer_constraint[:, np.newaxis])

    np.testing.assert_allclose(target_response[0, 0], 1.0 + 0.0j, atol=1.0e-8)
    assert abs(interferer_response[0, 0]) < 0.05


def test_time_domain_lcmv_can_place_explicit_interferer_null() -> None:
    """LCMV が target 保護と interferer null を同時に満たすことを確認する。"""
    tap_len = 2
    target_constraint = build_time_domain_tone_constraint_vector(
        _linear_array_steering(n_ch=3, spatial_frequency_rad=0.2),
        frequency_hz=900.0,
        fs_hz=6000.0,
        tap_len=tap_len,
    )
    interferer_constraint = build_time_domain_tone_constraint_vector(
        _linear_array_steering(n_ch=3, spatial_frequency_rad=1.4),
        frequency_hz=1500.0,
        fs_hz=6000.0,
        tap_len=tap_len,
    )
    constraints = np.stack([target_constraint, interferer_constraint], axis=1)
    desired_response = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    # 単位共分散では、LCMV は制約を満たす最小ノルム解になる。
    # ここでは null 制約の代数的な正しさだけを切り出して確認する。
    covariance = np.eye(constraints.shape[0], dtype=np.complex128)
    weights = design_time_domain_lcmv_coefficients(covariance, constraints, desired_response, diagonal_loading=0.0)
    responses = evaluate_constraint_response(weights, constraints)

    np.testing.assert_allclose(responses[:, 0], desired_response, atol=1.0e-8)


def test_time_domain_gsc_matches_lcmv_solution_for_same_constraints() -> None:
    """同じ制約と共分散では GSC 分解が LCMV と同じ重みになることを確認する。

    GSC は別方式ではなく `w = w_q - B g` という LCMV の実装分解である。
    ここで等価性を固定しておくことで、以後の評価で GSC 固有の実装誤差を切り分けられる。
    """
    tap_len = 3
    target_constraint = build_time_domain_tone_constraint_vector(
        _linear_array_steering(n_ch=4, spatial_frequency_rad=0.35),
        frequency_hz=1200.0,
        fs_hz=8000.0,
        tap_len=tap_len,
    )
    interferer_constraint = build_time_domain_tone_constraint_vector(
        _linear_array_steering(n_ch=4, spatial_frequency_rad=1.25),
        frequency_hz=1500.0,
        fs_hz=8000.0,
        tap_len=tap_len,
    )
    constraints = np.stack([target_constraint, interferer_constraint], axis=1)
    desired_response = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    rng = np.random.default_rng(1234)
    random_matrix = rng.normal(size=(constraints.shape[0], constraints.shape[0])) + 1j * rng.normal(
        size=(constraints.shape[0], constraints.shape[0])
    )
    # 正定値共分散を作り、GSC の blocked covariance が解ける条件にする。
    covariance = random_matrix @ random_matrix.conj().T + 0.5 * np.eye(constraints.shape[0], dtype=np.complex128)

    lcmv_weights = design_time_domain_lcmv_coefficients(covariance, constraints, desired_response, diagonal_loading=1.0e-3)
    gsc_weights = design_time_domain_gsc_coefficients(covariance, constraints, desired_response, diagonal_loading=1.0e-3)

    np.testing.assert_allclose(gsc_weights, lcmv_weights, atol=1.0e-8)
    blocking_matrix = build_gsc_blocking_matrix(constraints)
    np.testing.assert_allclose(constraints.conj().T @ blocking_matrix, 0.0, atol=1.0e-10)


def test_time_domain_fir_application_preserves_constrained_tone() -> None:
    """設計した FIR 重みを時間波形へ適用すると、制約 tone が歪まず出ることを確認する。"""
    fs_hz = 8000.0
    frequency_hz = 1000.0
    tap_len = 3
    n_sample = 64
    steering = _linear_array_steering(n_ch=3, spatial_frequency_rad=0.4)
    constraint = build_time_domain_tone_constraint_vector(
        steering,
        frequency_hz=frequency_hz,
        fs_hz=fs_hz,
        tap_len=tap_len,
    )
    covariance = np.eye(constraint.size, dtype=np.complex128)
    weights = design_time_domain_mvdr_coefficients(covariance, constraint, diagonal_loading=0.0)

    time_index = np.arange(n_sample, dtype=np.float64)
    base_tone = np.exp(1j * 2.0 * np.pi * frequency_hz * time_index / fs_hz)
    channel_signals = steering[:, np.newaxis] * base_tone[np.newaxis, :]
    output = apply_time_domain_fir_beamformer(channel_signals, weights, tap_len=tap_len)

    # 先頭 L-1 sample は履歴不足として 0 にする。full tap が揃った後は、w^H c = 1 により元 tone と一致する。
    np.testing.assert_allclose(output[0, : tap_len - 1], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(output[0, tap_len - 1 :], base_tone[tap_len - 1 :], atol=1.0e-8)


def test_time_domain_covariance_diagnostics_reports_loaded_condition_number() -> None:
    """時間領域適応方式の covariance health 診断を取得できることを確認する。"""
    n_sample = 128
    tap_len = 2
    channel_signals = np.vstack(
        [
            np.cos(2.0 * np.pi * 0.05 * np.arange(n_sample, dtype=np.float64)),
            np.sin(2.0 * np.pi * 0.05 * np.arange(n_sample, dtype=np.float64)),
        ]
    )
    tapped = build_time_tapped_snapshot_matrix(channel_signals, tap_len=tap_len)
    covariance = estimate_time_domain_covariance(tapped)
    constraint = np.ones((tapped.shape[0], 1), dtype=np.complex128)
    weights = design_time_domain_lcmv_coefficients(
        covariance,
        constraint,
        np.array([1.0 + 0.0j], dtype=np.complex128),
        diagonal_loading=1.0e-2,
    )
    diagnostics = diagnose_time_domain_adaptive_weights(
        covariance,
        constraint,
        weights,
        diagonal_loading=1.0e-2,
    )

    assert diagnostics.degree_of_freedom == 4
    assert diagnostics.constraint_count == 1
    assert diagnostics.output_count == 1
    assert diagnostics.loaded_condition_number >= 1.0
