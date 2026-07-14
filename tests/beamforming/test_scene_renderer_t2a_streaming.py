"""MATLAB係数境界とFlow接続T2a逐次runtimeを検証する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluations.beamforming.scene_renderer_t2a_streaming import (
    MatlabArrayCoefficients,
    StreamingBeamBranch,
    T2aScenarioConfig,
    design_frequency_weights,
    load_matlab_array_coefficients,
    run_streaming_flow,
)


def test_frequency_weight_design_returns_three_parallel_methods() -> None:
    """同じT共分散からfixed、T2a-MVDR、T2a-EBAEの固定shape重みを返す。

    2 channelではEBAEのN/E AICに必要な`M^2=4` snapshotを短時間で満たせるため、
    方式branchの構成と診断shapeだけを小さい決定論条件で確認する。
    """
    config = T2aScenarioConfig(
        fs_hz=64.0,
        sound_speed_m_s=1500.0,
        duration_s=2.0,
        training_duration_s=1.0,
        target_frequency_hz=8.0,
        interferer_frequency_hz=16.0,
        noise_band_hz=(2.0, 24.0),
        analysis_fft_size=8,
        analysis_hop_size=8,
        residual_fir_tap_count=4,
        runtime_block_size=5,
    )
    positions = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
    coefficients = MatlabArrayCoefficients(
        positions_m=positions,
        frequency_hz=np.array([0.0, 32.0]),
        shading=np.ones((2, 2), dtype=np.complex128),
        active_channel_mask=np.ones((2, 2), dtype=np.bool_),
    )
    rng = np.random.default_rng(20260715)
    training_signal = rng.standard_normal((2, int(config.duration_s * config.fs_hz)))

    design = design_frequency_weights(
        training_signal,
        coefficients,
        np.array([45.0, 90.0], dtype=np.float64),
        config,
    )

    assert set(design.weights) == {"fixed_baseline", "t2a_mvdr", "t2a_ebae"}
    assert all(weight.shape == (5, 2, 2) for weight in design.weights.values())
    assert design.ebae_signal_count.shape == (5, 2)
    assert design.ebae_fallback_mask.shape == (5, 2)


def test_load_matlab_coefficients_preserves_frequency_switched_active_channels(
    tmp_path: Path,
) -> None:
    """MATLABのchannel-first表を読み、周波数境界で高周波側active setへ切り替える。

    500 Hzちょうどと500 Hz超を分け、高周波で低周波用の広いactive setを使わない
    `searchsorted(..., side="left")`契約を固定する。
    """
    positions_path = tmp_path / "COE_POS"
    shading_path = tmp_path / "COE_CBFSHADING"
    positions_m = np.column_stack((np.arange(4, dtype=np.float64), np.zeros(4), np.zeros(4)))
    active = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1]], dtype=np.uint8)
    shading = active.astype(np.complex128)
    np.asarray(positions_m.T, dtype="<f4").reshape(-1, order="F").tofile(positions_path)
    raw_shading = np.concatenate((shading.real, shading.imag), axis=1)
    np.asarray(raw_shading, dtype="<f4").reshape(-1, order="F").tofile(shading_path)

    coefficients = load_matlab_array_coefficients(positions_path, shading_path, 500.0)
    _, at_boundary = coefficients.table_at(500.0)
    _, above_boundary = coefficients.table_at(500.1)

    np.testing.assert_array_equal(at_boundary, np.array([True, False, True, False]))
    np.testing.assert_array_equal(above_boundary, np.array([False, True, False, True]))


def test_flow_streaming_branch_matches_single_block_processing() -> None:
    """Flowで分割した整数遅延+残差FIR出力が一括blockと一致する。

    block長7は遅延3 sampleとFIR 3 tapの境界に揃わない値を選び、履歴をblock間で
    保持しなければ一致しない条件にする。
    """
    rng = np.random.default_rng(20260715)
    signal = rng.standard_normal((3, 41))
    delays = np.array([[0, 1, 3], [2, 0, 1]], dtype=np.int64)
    coefficients = np.array(
        [
            [[1.0, 0.2, 0.0], [0.5, -0.1, 0.0], [0.25, 0.0, 0.1]],
            [[0.75, 0.0, -0.1], [0.4, 0.2, 0.0], [0.2, 0.1, 0.0]],
        ],
        dtype=np.complex128,
    )
    split = run_streaming_flow(
        signal,
        [StreamingBeamBranch("t2a_mvdr", delays, coefficients)],
        block_size=7,
    )["t2a_mvdr"]
    one = run_streaming_flow(
        signal,
        [StreamingBeamBranch("t2a_mvdr", delays, coefficients)],
        block_size=signal.shape[1],
    )["t2a_mvdr"]

    np.testing.assert_allclose(split[0], one[0], atol=0.0)
    np.testing.assert_array_equal(split[1], one[1])
