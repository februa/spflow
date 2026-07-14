"""MATLAB係数境界とT2a逐次runtimeを検証する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import evaluations.beamforming.scene_renderer_t2a_streaming as t2a_streaming
from evaluations.beamforming.scene_renderer_t2a_streaming import (
    MatlabArrayCoefficients,
    StreamingBeamBranch,
    T2aScenarioConfig,
    design_frequency_weights,
    load_matlab_array_coefficients,
    run_evaluation,
    run_streaming_beam_branches,
)
from evaluations.beamforming.scene_renderer_t2a_waveform_reporting import (
    calculate_target_waveform_integrity,
)


def test_target_waveform_integrity_removes_only_tone_phase_delay() -> None:
    """既知のtone位相差だけを除去し、RMS・相関・残差を正しく評価する。

    8 Hz toneをfs 128 Hzで生成し、出力へpi/4 rad、すなわち2 sample相当の位相差を
    与える。training後の評価区間は12周期ちょうどで、端点leakageを指標誤差へ混ぜない。
    """
    config = T2aScenarioConfig(
        fs_hz=128.0,
        duration_s=2.0,
        training_duration_s=0.5,
        target_frequency_hz=8.0,
        interferer_frequency_hz=16.0,
        noise_band_hz=(2.0, 40.0),
        analysis_fft_size=16,
        analysis_hop_size=16,
        residual_fir_tap_count=8,
        runtime_block_size=13,
    )
    sample_index = np.arange(int(config.duration_s * config.fs_hz), dtype=np.float64)
    phase_rad = np.pi / 4.0
    reference = np.cos(2.0 * np.pi * config.target_frequency_hz * sample_index / config.fs_hz)
    output = np.cos(
        2.0 * np.pi * config.target_frequency_hz * sample_index / config.fs_hz + phase_rad
    ).astype(np.complex128)

    result = calculate_target_waveform_integrity(
        reference,
        output,
        np.ones(reference.shape, dtype=np.bool_),
        config,
    )

    assert result.phase_delay_samples_modulo_period == pytest.approx(2.0, abs=1.0e-12)
    assert result.rms_delta_db == pytest.approx(0.0, abs=1.0e-12)
    assert result.correlation_after_phase_alignment == pytest.approx(1.0, abs=1.0e-12)
    assert result.residual_rms_db_re_input_rms < -250.0


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


def test_frequency_weight_design_can_run_only_t2a_ebae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EBAE単独選択時はMVDR重みを解かず、EBAE完成重みだけを公開する。

    2 channel、`M^2=4` non-overlap snapshotを満たす決定論入力を使う。
    MVDR solverを例外へ置換し、方式辞書から外すだけで裏でMVDRを計算していないことも
    固定する。EBAE内部の安全fallback用CBF計算は方式成立条件なので対象外とする。
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
    coefficients = MatlabArrayCoefficients(
        positions_m=np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64),
        frequency_hz=np.array([0.0, 32.0]),
        shading=np.ones((2, 2), dtype=np.complex128),
        active_channel_mask=np.ones((2, 2), dtype=np.bool_),
    )
    rng = np.random.default_rng(20260715)
    training_signal = rng.standard_normal((2, int(config.duration_s * config.fs_hz)))

    def fail_if_mvdr_is_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("t2a_ebae-only design must not solve t2a_mvdr weights")

    monkeypatch.setattr(t2a_streaming, "_loaded_mvdr_weight", fail_if_mvdr_is_called)
    design = design_frequency_weights(
        training_signal,
        coefficients,
        np.array([45.0, 90.0], dtype=np.float64),
        config,
        method_ids=("t2a_ebae",),
    )

    assert tuple(design.weights) == ("t2a_ebae",)
    assert design.weights["t2a_ebae"].shape == (5, 2, 2)


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


def test_streaming_beam_branch_matches_single_block_processing() -> None:
    """block分割した整数遅延+残差FIR出力が一括blockと一致する。

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

    split = run_streaming_beam_branches(
        signal,
        [StreamingBeamBranch("t2a_mvdr", delays, coefficients)],
        block_size=7,
    )["t2a_mvdr"]
    one = run_streaming_beam_branches(
        signal,
        [StreamingBeamBranch("t2a_mvdr", delays, coefficients)],
        block_size=signal.shape[1],
    )["t2a_mvdr"]

    np.testing.assert_allclose(split[0], one[0], atol=0.0)
    np.testing.assert_array_equal(split[1], one[1])


def test_streaming_beam_branches_reject_duplicate_method_ids() -> None:
    """収集先が衝突する同一方式IDを処理開始前に拒否する。

    同じ辞書keyへ二方式の出力を保存すると、後段の評価で一方を完成結果として誤認する。
    1 beam、1 channel、1 tapの最小条件を使い、信号処理前の方式境界契約だけを確認する。
    """
    delays = np.zeros((1, 1), dtype=np.int64)
    coefficients = np.ones((1, 1, 1), dtype=np.complex128)
    branches = [
        StreamingBeamBranch("t2a_mvdr", delays, coefficients),
        StreamingBeamBranch("t2a_mvdr", delays, coefficients),
    ]

    with pytest.raises(ValueError, match="method_id values must be unique"):
        run_streaming_beam_branches(np.ones((1, 4), dtype=np.float64), branches, block_size=2)


def test_run_evaluation_completes_before_review_pack_serialization(tmp_path: Path) -> None:
    """固定整相の完成評価結果をreporting境界へ渡して全成果物を生成する。

    2 channel、5 beam、128 sampleの小さい決定論scenarioを使い、MATLAB raw読込、
    scene生成、重み設計、block逐次処理、固定型評価結果、review pack保存の接続を確認する。
    適応方式の成立性は別テストで扱い、ここでは責務境界だけを見るためfixed単独とする。
    """
    positions_path = tmp_path / "COE_POS"
    shading_path = tmp_path / "COE_CBFSHADING"
    output_dir = tmp_path / "review_pack"
    positions_m = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
    shading = np.ones((2, 2), dtype=np.complex128)
    np.asarray(positions_m.T, dtype="<f4").reshape(-1, order="F").tofile(positions_path)
    raw_shading = np.concatenate((shading.real, shading.imag), axis=1)
    np.asarray(raw_shading, dtype="<f4").reshape(-1, order="F").tofile(shading_path)
    config = T2aScenarioConfig(
        fs_hz=64.0,
        duration_s=2.0,
        training_duration_s=0.5,
        target_azimuth_deg=45.0,
        target_frequency_hz=8.0,
        interferer_azimuth_deg=90.0,
        interferer_frequency_hz=16.0,
        noise_band_hz=(2.0, 24.0),
        beam_azimuth_step_deg=45.0,
        analysis_fft_size=8,
        analysis_hop_size=8,
        residual_fir_tap_count=4,
        runtime_block_size=7,
    )

    run_evaluation(
        positions_path,
        shading_path,
        shading_frequency_step_hz=32.0,
        output_dir=output_dir,
        config=config,
        method_ids=("fixed_baseline",),
        review_title="responsibility boundary test",
    )

    assert (output_dir / "scenario_summary.csv").is_file()
    assert (output_dir / "plot_arrays.npz").is_file()
    assert (output_dir / "review_index.md").is_file()
