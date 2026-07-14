"""mvdr beamforming に関する回帰試験。"""

# ここでは steering の向き、共分散推定、重み適用後の再構成が噛み合うことを
# 決定論的な入力で固定し、ビームフォーミング変更時の退行を早期に検知する。

import numpy as np

from spflow import (
    CovarianceEstimator,
    MVDRFilter,
    MVDROverlapSaveBeamformer,
    MVDRWeightCallback,
    MVDRWeightDesigner,
    MVDRWeightSnapshot,
    StepScheduler,
    apply_beamformer,
    design_mvdr_coefficients,
    design_mvdr_coefficients_with_channel_window,
    design_mvdr_overlap_save_filters,
    estimate_covariance,
    forgetting_factor_from_integration_time,
    integrate_band_covariances,
    integration_blocks_from_integration_time,
    recommended_integration_time_for_independent_samples,
)
from spflow.beamforming.covariance import estimate_covariance_snapshots
from spflow.beamforming.mvdr_weight_designer import design_mvdr_coefficients_bands


def _collect_overlap_save_output(
    records: list[tuple[int, np.ndarray]], n_beam: int, n_band: int
) -> np.ndarray:
    """`_collect_overlap_save_output` を実行する。"""
    per_band: list[list[np.ndarray]] = [[] for _ in range(n_band)]
    for band_idx, valid in records:
        per_band[band_idx].append(valid)

    pieces = []
    for band_idx in range(n_band):
        if per_band[band_idx]:
            pieces.append(np.concatenate(per_band[band_idx], axis=-1))
        else:
            pieces.append(np.zeros((n_beam, 0), dtype=np.complex64))
    return np.stack(pieces, axis=1)


def test_estimate_covariance_matches_manual_result():
    """共分散推定が手計算結果と一致することを確認する。"""
    X = np.array(
        [
            [1.0 + 1.0j, 2.0 - 1.0j],
            [0.5 + 0.0j, -1.0 + 2.0j],
        ]
    )

    out = estimate_covariance(X)
    expected = X @ X.conj().T / X.shape[1]

    np.testing.assert_allclose(out, expected)


def test_integration_blocks_from_integration_time_matches_ceiling_rule():
    """integration time からの積分ブロック数算出が切り上げ規則と一致することを確認する。"""
    assert integration_blocks_from_integration_time(0.25, 500.0) == 125
    assert integration_blocks_from_integration_time(0.0, 500.0) == 0


def test_recommended_integration_time_for_independent_samples_matches_rule_of_thumb():
    """独立サンプル向け推奨積分時間算出が経験則と一致することを確認する。"""
    integration_time = recommended_integration_time_for_independent_samples(32, 512.0)

    np.testing.assert_allclose(integration_time * 512.0, 64.0)


def test_forgetting_factor_from_integration_time_matches_specification():
    """integration time からの忘却係数算出が仕様式と一致することを確認する。"""
    factor = forgetting_factor_from_integration_time(2.0, 4.0)

    np.testing.assert_allclose(factor, 2.0 / (1.0 + 2.0 * 4.0))


def test_forgetting_factor_is_clamped_to_one():
    """対象機能について 忘却係数が 1 以下に丸められる を確認する。"""
    factor = forgetting_factor_from_integration_time(0.0, 16.0)

    np.testing.assert_allclose(factor, 1.0)


def test_covariance_estimator_applies_exponential_smoothing():
    """共分散推定器が指数平滑を適用することを確認する。"""
    estimator = CovarianceEstimator(smoothing=0.5)
    x1 = np.array([[1.0 + 0.0j, 0.0 + 0.0j]])
    x2 = np.array([[0.0 + 0.0j, 2.0 + 0.0j]])

    r1 = estimator.process(x1)
    r2 = estimator.process(x2)

    np.testing.assert_allclose(r1, np.array([[0.5 + 0.0j]]))
    np.testing.assert_allclose(r2, np.array([[1.25 + 0.0j]]))


def test_covariance_estimator_process_snapshot_matches_matlab_style_outer_product_update():
    """共分散推定器について `process_snapshot` が MATLAB 流の外積更新式と一致する を確認する。"""
    estimator = CovarianceEstimator(forgetting_factor=0.25)
    s1 = np.array([2.0 + 0.0j, 0.0 + 0.0j])
    s2 = np.array([0.0 + 0.0j, 4.0 + 0.0j])

    r1 = estimator.process_snapshot(s1, normalization=2.0)
    r2 = estimator.process_snapshot(s2, normalization=2.0)

    c1 = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]])
    c2 = np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 4.0 + 0.0j]])
    np.testing.assert_allclose(r1, c1)
    np.testing.assert_allclose(r2, 0.75 * c1 + 0.25 * c2)


def test_estimate_covariance_snapshots_matches_per_snapshot_outer_products():
    """snapshot 群の共分散推定がsnapshot ごとの外積和と一致することを確認する。"""
    snapshots = np.array(
        [
            [2.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 4.0 + 0.0j],
        ]
    )

    out = estimate_covariance_snapshots(snapshots, normalization=2.0)
    expected = np.stack(
        [
            np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]),
            np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 4.0 + 0.0j]]),
        ],
        axis=0,
    )

    np.testing.assert_allclose(out, expected)


def test_covariance_estimator_process_snapshots_matches_separate_estimators():
    """共分散推定器について `process_snapshots` が個別推定器の結果と一致する を確認する。"""
    snapshots = np.array(
        [
            [[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 4.0 + 0.0j]],
            [[4.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 2.0 + 0.0j]],
        ]
    )
    estimator = CovarianceEstimator(forgetting_factor=0.25)
    separate = [CovarianceEstimator(forgetting_factor=0.25) for _ in range(snapshots.shape[1])]

    out1 = estimator.process_snapshots(snapshots[0], normalization=2.0)
    out2 = estimator.process_snapshots(snapshots[1], normalization=2.0)

    expected1 = np.stack(
        [
            sep.process_snapshot(snapshots[0, idx], normalization=2.0)
            for idx, sep in enumerate(separate)
        ],
        axis=0,
    )
    expected2 = np.stack(
        [
            sep.process_snapshot(snapshots[1, idx], normalization=2.0)
            for idx, sep in enumerate(separate)
        ],
        axis=0,
    )

    np.testing.assert_allclose(out1, expected1)
    np.testing.assert_allclose(out2, expected2)


def test_integrate_band_covariances_matches_manual_bandwise_matlab_loop():
    """帯域共分散積分について 手計算の帯域別 MATLAB ループと一致する を確認する。"""
    X = np.array(
        [
            [[2.0 + 0.0j, 4.0 + 0.0j]],
            [[0.0 + 0.0j, 2.0 + 0.0j]],
        ]
    )
    X = np.concatenate([X, np.zeros_like(X)], axis=1)
    rxx = integrate_band_covariances(X, forgetting_factor=0.5, normalization=2.0, n_blocks=2)

    first = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]])
    second = np.array([[4.0 + 0.0j, 2.0 + 0.0j], [2.0 + 0.0j, 1.0 + 0.0j]])
    expected_band0 = 0.5 * first + 0.5 * second
    np.testing.assert_allclose(rxx[0], expected_band0)
    np.testing.assert_allclose(rxx[1], np.zeros((2, 2), dtype=np.complex64))


def test_covariance_estimator_can_be_configured_from_integration_time():
    """共分散推定器が integration time から設定できることを確認する。"""
    estimator = CovarianceEstimator.from_integration_time(2.0, 4.0)
    x1 = np.array([[1.0 + 0.0j, 0.0 + 0.0j]])
    x2 = np.array([[0.0 + 0.0j, 2.0 + 0.0j]])

    alpha = 2.0 / (1.0 + 2.0 * 4.0)
    r1 = estimator.process(x1)
    r2 = estimator.process(x2)

    np.testing.assert_allclose(r1, np.array([[0.5 + 0.0j]]))
    np.testing.assert_allclose(
        r2, (1.0 - alpha) * np.array([[0.5 + 0.0j]]) + alpha * np.array([[2.0 + 0.0j]])
    )


def test_design_mvdr_coefficients_is_distortionless():
    """MVDR 重み設計について 無歪み条件を満たす を確認する。"""
    steering = np.array([[1.0 + 0.0j], [1.0j]])
    Rxx = np.array(
        [
            [2.0 + 0.0j, 0.2 - 0.1j],
            [0.2 + 0.1j, 1.5 + 0.0j],
        ]
    )

    weights = design_mvdr_coefficients(Rxx, steering, diag_load=1e-3)
    response = weights.T @ steering

    np.testing.assert_allclose(response, np.ones((1, 1)), atol=1e-6)


def test_design_mvdr_overlap_save_filters_embed_conjugation_in_filter_fft():
    """overlap-save用filter FFTへ設計上必要な複素共役を埋め込むことを確認する。"""
    weights = np.array([[[1.0 + 0.0j], [1.0j]]]).transpose(0, 2, 1)
    filters = design_mvdr_overlap_save_filters(weights, frame_size=8)

    taps = np.fft.ifft(filters, axis=-1)
    np.testing.assert_allclose(taps[..., 0], weights, atol=1e-6)
    np.testing.assert_allclose(taps[..., 1:], 0.0, atol=1e-6)


def test_design_mvdr_coefficients_with_channel_window_uses_only_selected_channels():
    """channel window 付き MVDR 重み設計について 選択されたチャネルだけを使う を確認する。"""
    steering = np.array([[1.0 + 0.0j], [1.0 + 0.0j], [1.0 + 0.0j]])
    Rxx = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 2.0 + 0.0j, 0.1 + 0.0j],
            [0.0 + 0.0j, 0.1 + 0.0j, 3.0 + 0.0j],
        ]
    )
    window = np.array([0.0, 1.0, 1.0])

    weights = design_mvdr_coefficients_with_channel_window(Rxx, steering, window, diag_load=0.0)
    shaded = steering * window[:, np.newaxis]

    np.testing.assert_allclose(weights[0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(weights.T @ shaded, np.ones((1, 1)), atol=1e-6)


def test_design_mvdr_coefficients_with_channel_window_handles_bandwise_input():
    """channel window 付き MVDR 重み設計が帯域別入力を処理できることを確認する。"""
    steering = np.ones((3, 1, 2), dtype=np.complex64)
    rxx = np.stack(
        [
            np.eye(3, dtype=np.complex64),
            np.array(
                [
                    [2.0 + 0.0j, 0.2 + 0.0j, 0.0 + 0.0j],
                    [0.2 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 4.0 + 0.0j],
                ]
            ),
        ],
        axis=0,
    )
    window = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )

    weights = design_mvdr_coefficients_with_channel_window(rxx, steering, window, diag_load=0.0)

    np.testing.assert_allclose(weights[2, 0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(weights[0, 0, 1], 0.0, atol=1e-6)
    for band_idx in range(2):
        shaded = steering[:, :, band_idx] * window[:, band_idx][:, np.newaxis]
        np.testing.assert_allclose(weights[:, :, band_idx].T @ shaded, np.ones((1, 1)), atol=1e-6)


def test_mvdr_weight_designer_handles_bandwise_input():
    """MVDR 重み設計器が帯域別入力を処理できることを確認する。"""
    steering = np.stack(
        [
            np.array([[1.0 + 0.0j], [1.0 + 0.0j]]),
            np.array([[1.0 + 0.0j], [0.0 + 1.0j]]),
        ],
        axis=-1,
    )
    Rxx = np.stack(
        [
            np.eye(2, dtype=np.complex64),
            np.array([[2.0 + 0.0j, 0.0], [0.0, 1.0 + 0.0j]], dtype=np.complex64),
        ],
        axis=0,
    )

    weights = MVDRWeightDesigner(diag_load=0.0).process(Rxx, steering)

    assert weights.shape == (2, 1, 2)
    for band_idx in range(weights.shape[-1]):
        response = weights[:, :, band_idx].T @ steering[:, :, band_idx]
        np.testing.assert_allclose(response, np.ones((1, 1)), atol=1e-6)


def test_design_mvdr_coefficients_bands_matches_per_band_design():
    """帯域別 MVDR 重み設計が帯域ごとの個別設計結果と一致することを確認する。"""
    steering = np.stack(
        [
            np.array([[1.0 + 0.0j], [1.0 + 0.0j]]),
            np.array([[1.0 + 0.0j], [0.0 + 1.0j]]),
            np.array([[1.0 - 0.2j], [0.3 + 0.7j]]),
        ],
        axis=-1,
    )
    rxx = np.stack(
        [
            np.eye(2, dtype=np.complex64),
            np.array([[2.0 + 0.0j, 0.2 - 0.1j], [0.2 + 0.1j, 1.5 + 0.0j]], dtype=np.complex64),
            np.array([[1.2 + 0.0j, -0.1 + 0.05j], [-0.1 - 0.05j, 0.9 + 0.0j]], dtype=np.complex64),
        ],
        axis=0,
    )

    out = design_mvdr_coefficients_bands(rxx, steering, diag_load=1e-3)
    expected = np.stack(
        [
            design_mvdr_coefficients(rxx[idx], steering[:, :, idx], diag_load=1e-3)
            for idx in range(rxx.shape[0])
        ],
        axis=-1,
    )

    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_apply_beamformer_matches_manual_projection():
    """ビームフォーマ適用が手計算の射影結果と一致することを確認する。"""
    X = np.array(
        [
            [1.0 + 0.0j, 2.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 2.0j],
        ]
    )
    weights = np.array(
        [
            [1.0 + 0.0j, 0.5 + 0.0j],
            [1.0j, -0.5j],
        ]
    )

    out = apply_beamformer(X, weights)
    expected = np.vstack([weights[:, beam] @ X for beam in range(weights.shape[1])])

    np.testing.assert_allclose(out, expected)


def test_mvdr_filter_uses_stored_weights():
    """対象機能について MVDR filter が保持済み重みを使う を確認する。"""
    X = np.array([[1.0 + 0.0j, 2.0 + 0.0j]])
    weights = np.array([[2.0 + 0.0j]])

    filt = MVDRFilter(weights)

    np.testing.assert_allclose(filt.process(X), np.array([[2.0 + 0.0j, 4.0 + 0.0j]]))


def test_mvdr_overlap_save_beamformer_matches_pointwise_projection_for_length1_filters():
    """長さ1のMVDR filterでは、overlap-save出力が点ごとの射影と一致することを確認する。"""
    weights = np.array([[1.0 + 0.0j], [1.0j]])[:, :, np.newaxis]
    X = np.array(
        [
            np.arange(1, 17, dtype=np.float32),
            1j * np.arange(1, 17, dtype=np.float32),
        ],
        dtype=np.complex64,
    )[:, np.newaxis, :]

    beamformer = MVDROverlapSaveBeamformer(weights, frame_size=8, valid_size=4)
    records: list[tuple[int, np.ndarray]] = []
    for start in range(0, X.shape[-1], 3):
        records.extend(beamformer.process(X[:, :, start : start + 3]))
    records.extend(beamformer.flush())

    out = _collect_overlap_save_output(records, n_beam=1, n_band=1)
    expected = apply_beamformer(X[:, 0, :], weights[:, :, 0])

    np.testing.assert_allclose(out[0, 0, : X.shape[-1]], expected[0], atol=1e-6)


def test_mvdr_weight_callback_publishes_only_completed_weights():
    """MVDR係数完成までは固定CBFを使い、完成後だけ適応係数へ切り替わることを確認する。"""
    steering = np.stack(
        [
            np.array([[1.0 + 0.0j], [1.0 + 0.0j]]),
            np.array([[1.0 + 0.0j], [1.0j]]),
        ],
        axis=-1,
    )
    Rxx = np.stack(
        [
            np.eye(2, dtype=np.complex64),
            np.array([[2.0 + 0.0j, 0.0], [0.0, 1.0 + 0.0j]], dtype=np.complex64),
        ],
        axis=0,
    )
    scheduler = StepScheduler(MVDRWeightCallback(diag_load=0.0), items_per_cycle=1)
    snapshot = MVDRWeightSnapshot(covariance=Rxx, steering=steering, generation="frame-1")

    first_result = scheduler.process_result(snapshot)
    completed_result = scheduler.process_result(snapshot)

    assert first_result.updated is False
    assert first_result.generation == "frame-1"
    # 初回fallbackもh^T a=1を満たし、適応係数が未完成という理由で信号を消失させない。
    for band_idx in range(first_result.value.shape[-1]):
        fallback_response = first_result.value[:, :, band_idx].T @ steering[:, :, band_idx]
        np.testing.assert_allclose(fallback_response, np.ones((1, 1)), atol=1e-6)

    assert completed_result.updated is True
    assert completed_result.value.shape == steering.shape
    for band_idx in range(completed_result.value.shape[-1]):
        response = completed_result.value[:, :, band_idx].T @ steering[:, :, band_idx]
        np.testing.assert_allclose(response, np.ones((1, 1)), atol=1e-6)


def test_mvdr_weight_snapshot_owns_immutable_arrays():
    """入力元を変更しても、時間分割中のMVDR snapshotが変化しないことを確認する。"""
    steering = np.ones((2, 1, 2), dtype=np.complex64)
    covariance = np.repeat(np.eye(2, dtype=np.complex64)[None, :, :], 2, axis=0)
    snapshot = MVDRWeightSnapshot(
        covariance=covariance,
        steering=steering,
        generation=1,
    )

    # 呼び出し元の配列は次周期用に再利用され得るため、snapshotが所有copyを持つことを固定する。
    covariance[:] = 0.0
    steering[:] = 0.0

    np.testing.assert_array_equal(
        snapshot.covariance,
        np.repeat(np.eye(2, dtype=np.complex64)[None, :, :], 2, axis=0),
    )
    np.testing.assert_array_equal(snapshot.steering, np.ones((2, 1, 2), dtype=np.complex64))
    assert snapshot.covariance.flags.writeable is False
    assert snapshot.steering.flags.writeable is False


def test_design_mvdr_coefficients_handles_zero_trace_covariance_with_diag_load():
    """MVDR 重み設計がゼロ trace 共分散でも diagonal loading で安定に動くことを確認する。"""
    steering = np.array([[1.0 + 0.0j], [0.0 + 1.0j]])
    Rxx = np.zeros((2, 2), dtype=np.complex64)

    weights = design_mvdr_coefficients(Rxx, steering, diag_load=1e-3)
    response = weights.T @ steering

    np.testing.assert_allclose(response, np.ones((1, 1)), atol=1e-6)
