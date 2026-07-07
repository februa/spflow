"""固定遅延+差分補正 MVDR の単体試験。"""

from __future__ import annotations

import numpy as np

from spflow.beamforming import (
    STANDARD_FRACTIONAL_DELAY_PATTERN_COUNT,
    STANDARD_FRACTIONAL_DELAY_TAP_COUNT,
    DelayTable,
    DifferenceCorrectionFIR,
    DifferenceCorrectionFIRDesigner,
    FractionalDelayAndSumBeamformer,
    LoadedMVDRWeightDesigner,
    ShortFFTCovarianceAccumulator,
    design_distortionless_fixed_weights,
    design_fixed_delay_fractional_weights_from_delay_table,
    design_standard_fractional_delay_filter_bank,
)


def _unit_steering_table(n_bin: int, n_ch: int) -> np.ndarray:
    """各 bin で有限ノルムを持つ検証用ステアリングを作る。"""
    frequencies = np.arange(n_bin, dtype=np.float64)
    channel_positions = np.arange(n_ch, dtype=np.float64)
    # phase shape: [n_bin, n_ch]。
    # bin と channel で位相が変わる表を使い、
    # 単純な全 1 steering だけでは見えない共役規約を検査する。
    phase = (2.0 * np.pi * frequencies[:, np.newaxis] * channel_positions[np.newaxis, :]) / float(
        n_bin
    )
    return np.exp(1j * phase).astype(np.complex128)


def test_short_fft_covariance_accumulator_updates_blocks_and_ready_flag() -> None:
    """128 sample 相当の block 境界と指数平均係数を確認する。

    端数 chunk を先に入れても共分散を更新せず、次 chunk と結合して full block になった
    時点でだけ FFT 統計へ反映する条件を作る。
    """
    accumulator = ShortFFTCovarianceAccumulator(
        n_ch=2,
        fft_size=8,
        block_size=4,
        fs_hz=32.0,
        covariance_time_constant_sec=2.0,
        blocks_per_weight_update=2,
    )

    first = accumulator.process(np.ones((2, 3), dtype=np.float64))
    assert first.processed_block_count == 0
    assert first.update_ready is False

    second = accumulator.process(np.ones((2, 5), dtype=np.float64))
    assert second.processed_block_count == 2
    assert second.total_block_count == 2
    assert second.update_ready is True
    assert second.covariance.shape == (8, 2, 2)

    # 共分散 R[k] は X[k]X[k]^H の指数平均なので Hermitian でなければならない。
    np.testing.assert_allclose(second.covariance, np.swapaxes(second.covariance.conj(), 1, 2))
    expected_alpha = float(np.exp(-(4.0 / 32.0) / 2.0))
    assert accumulator.alpha == expected_alpha


def test_loaded_mvdr_matches_fixed_weight_for_white_covariance() -> None:
    """白色共分散では MVDR が歪みなし固定整相重みと一致することを確認する。

    R=I では `R^{-1}a=a` なので、MVDR 解は `a/(a^H a)` になる。
    この性質は差分補正枝が不要な条件で `q=0` になるための基準である。
    """
    steering = _unit_steering_table(n_bin=8, n_ch=3)
    fixed_weight = design_distortionless_fixed_weights(steering)
    covariance = np.repeat(np.eye(3, dtype=np.complex128)[np.newaxis, :, :], 8, axis=0)
    designer = LoadedMVDRWeightDesigner(diagonal_loading_ratio=0.0)

    result = designer.compute(covariance, steering, fixed_weight)

    np.testing.assert_allclose(result.weights, fixed_weight, atol=1.0e-12)
    np.testing.assert_array_equal(result.fallback_mask, np.zeros(8, dtype=np.bool_))
    response = np.sum(result.weights.conj() * steering, axis=1)
    np.testing.assert_allclose(response, np.ones(8, dtype=np.complex128), atol=1.0e-12)


def test_loaded_mvdr_preserves_fixed_path_complex_target_response() -> None:
    """MVDR が固定主経路の複素 target 応答を保持することを確認する。

    小数遅延 FIR を含む固定主経路は、target 振幅を保っても群遅延分の位相を持つ。
    差分枝 `q = w0 - w_mvdr` が target を通さないためには、MVDR 側も
    `w_mvdr^H a = w0^H a` を満たす必要がある。
    """
    steering = _unit_steering_table(n_bin=8, n_ch=3)
    distortionless_weight = design_distortionless_fixed_weights(steering)
    desired_response = np.exp(1j * np.linspace(0.0, np.pi / 2.0, 8, dtype=np.float64))

    # response = sum(conj(w) * a) なので、固定重みへ conj(desired_response) を掛けると
    # 固定主経路の target 応答は desired_response になる。
    fixed_weight = distortionless_weight * desired_response[:, np.newaxis].conj()
    covariance = np.repeat(np.eye(3, dtype=np.complex128)[np.newaxis, :, :], 8, axis=0)
    designer = LoadedMVDRWeightDesigner(diagonal_loading_ratio=0.0)

    result = designer.compute(covariance, steering, fixed_weight)

    fixed_response = np.sum(fixed_weight.conj() * steering, axis=1)
    mvdr_response = np.sum(result.weights.conj() * steering, axis=1)
    np.testing.assert_allclose(fixed_response, desired_response, atol=1.0e-12)
    np.testing.assert_allclose(mvdr_response, desired_response, atol=1.0e-12)

    diff_result = DifferenceCorrectionFIRDesigner(
        fir_taps=8,
        frequencies_hz=np.arange(8, dtype=np.float64),
        fs_hz=8.0,
    ).compute(
        fixed_weight,
        result.weights,
        steering,
    )
    np.testing.assert_allclose(
        diff_result.diagnostics.q_blocking_response,
        np.zeros(8, dtype=np.complex128),
        atol=1.0e-12,
    )


def test_loaded_mvdr_falls_back_to_fixed_weight_for_singular_unloaded_covariance() -> None:
    """特異共分散では target 保護のため固定整相へ退避することを確認する。

    対角ローディングを意図的に 0 にし、ゼロ共分散の `solve` 失敗を発生させる。
    前回重みがない最初の異常更新では、固定整相重みを採用するのが安全側である。
    """
    steering = _unit_steering_table(n_bin=4, n_ch=2)
    fixed_weight = design_distortionless_fixed_weights(steering)
    covariance = np.zeros((4, 2, 2), dtype=np.complex128)
    designer = LoadedMVDRWeightDesigner(diagonal_loading_ratio=0.0)

    result = designer.compute(covariance, steering, fixed_weight)

    np.testing.assert_allclose(result.weights, fixed_weight)
    np.testing.assert_array_equal(result.fallback_mask, np.ones(4, dtype=np.bool_))


def test_difference_correction_designer_preserves_mvdr_weight_and_blocks_target() -> None:
    """差分 FIR 化後も `w0 - q = w_mvdr` と `q^H a = 0` が成立することを確認する。

    `fir_taps == n_bin` の full spectrum 条件では IFFT/FFT の丸め誤差だけが残る。
    ここで崩れる場合、共役規約または axis の扱いが設計式と一致していない。
    """
    n_bin = 8
    n_ch = 3
    steering = _unit_steering_table(n_bin=n_bin, n_ch=n_ch)
    fixed_weight = design_distortionless_fixed_weights(steering)

    # q_candidate は target 方向に直交する成分だけにする。
    # そのため w_mvdr = w0 - q_candidate も `w^H a = 1` を満たす。
    q_candidate = np.zeros_like(fixed_weight)
    q_candidate[:, 0] = 0.05 + 0.02j
    q_candidate[:, 1] = -q_candidate[:, 0] * steering[:, 0].conj() / steering[:, 1].conj()
    mvdr_weight = fixed_weight - q_candidate

    result = DifferenceCorrectionFIRDesigner(
        fir_taps=n_bin,
        frequencies_hz=np.arange(n_bin, dtype=np.float64),
        fs_hz=float(n_bin),
    ).compute(
        fixed_weight,
        mvdr_weight,
        steering,
    )

    np.testing.assert_allclose(result.q_weight_freq, q_candidate, atol=1.0e-12)
    np.testing.assert_allclose(result.final_weight_freq, mvdr_weight, atol=1.0e-12)
    np.testing.assert_allclose(result.diagnostics.target_response_w0, np.ones(n_bin), atol=1.0e-12)
    np.testing.assert_allclose(
        result.diagnostics.target_response_mvdr,
        np.ones(n_bin),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.diagnostics.target_response_final,
        np.ones(n_bin),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.diagnostics.q_blocking_response,
        np.zeros(n_bin),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.diagnostics.q_reconstruction_error,
        np.zeros_like(q_candidate),
        atol=1.0e-12,
    )


def test_difference_correction_designer_matches_arbitrary_physical_frequencies() -> None:
    """任意 Hz 周波数で差分 FIR の物理周波数応答が一致することを確認する。

    `np.fft.ifft` は DFT bin の周波数だけを前提にするため、768 Hz などの任意の
    評価周波数を直接渡すと物理周波数がずれる。この条件では Vandermonde 行列
    `exp(-j 2π f_k l / fs)` による設計でなければ `q` を再構成できない。
    """
    frequencies_hz = np.array([300.0, 900.0, 1700.0, 2600.0], dtype=np.float64)
    fs_hz = 8000.0
    fir_taps = 6
    n_ch = 2
    tap_index = np.arange(fir_taps, dtype=np.float64)
    response_matrix = np.exp(
        -1j * 2.0 * np.pi * frequencies_hz[:, np.newaxis] * tap_index[np.newaxis, :] / fs_hz
    )
    known_taps = np.array(
        [
            [0.25 + 0.10j, -0.05 + 0.03j, 0.02 - 0.01j, 0.01 + 0.00j, 0.0, 0.0],
            [-0.10 + 0.05j, 0.04 - 0.02j, 0.00 + 0.01j, 0.02 + 0.02j, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    q_apply = response_matrix @ known_taps.T
    q_weight = np.conj(q_apply)
    fixed_weight = np.zeros((frequencies_hz.size, n_ch), dtype=np.complex128)
    mvdr_weight = fixed_weight - q_weight
    steering = np.ones((frequencies_hz.size, n_ch), dtype=np.complex128)

    result = DifferenceCorrectionFIRDesigner(
        fir_taps=fir_taps,
        frequencies_hz=frequencies_hz,
        fs_hz=fs_hz,
    ).compute(
        fixed_weight,
        mvdr_weight,
        steering,
    )

    np.testing.assert_allclose(result.reconstructed_q_weight_freq, q_weight, atol=1.0e-12)
    np.testing.assert_allclose(
        result.diagnostics.q_reconstruction_error, np.zeros_like(q_weight), atol=1.0e-12
    )


def test_difference_correction_fir_matches_full_processing_across_chunks() -> None:
    """差分補正 FIR が chunk 分割に依存しないことを確認する。

    逐次処理では `fir_taps - 1` sample の履歴を次 chunk へ渡すため、
    一括処理と分割処理の出力が一致しなければリアルタイムルートで境界ノイズが出る。
    """
    taps = np.array(
        [
            [1.0 + 0.0j, 0.25 + 0.5j, -0.125 + 0.0j],
            [0.5 - 0.25j, -0.25 + 0.0j, 0.0 + 0.125j],
        ],
        dtype=np.complex128,
    )
    signal = np.array(
        [
            [1.0, 2.0, -1.0, 0.5, 0.25, -0.5],
            [0.0, 1.0, 0.5, -0.5, 2.0, 1.5],
        ],
        dtype=np.float64,
    )

    full_fir = DifferenceCorrectionFIR(n_ch=2, fir_taps=3)
    full_fir.update_coefficients(taps)
    full_output = full_fir.process(signal)

    chunked_fir = DifferenceCorrectionFIR(n_ch=2, fir_taps=3)
    chunked_fir.update_coefficients(taps)
    chunked_output = np.concatenate(
        [
            chunked_fir.process(signal[:, :2]),
            chunked_fir.process(signal[:, 2:5]),
            chunked_fir.process(signal[:, 5:]),
        ],
        axis=0,
    )

    np.testing.assert_allclose(chunked_output, full_output, atol=1.0e-12)

    # 1 channel 目の impulse に対し、出力先頭はその channel の FIR 係数そのものになる。
    impulse_fir = DifferenceCorrectionFIR(n_ch=2, fir_taps=3)
    impulse_fir.update_coefficients(taps)
    impulse = np.zeros((2, 3), dtype=np.complex128)
    impulse[0, 0] = 1.0 + 0.0j
    np.testing.assert_allclose(impulse_fir.process(impulse), taps[0], atol=1.0e-12)


def test_standard_fractional_delay_filter_bank_has_51_patterns_and_128_taps() -> None:
    """方式指定どおり、-0.5〜0.5 sample を 51 本の 128 tap FIR として事前計算する。

    小数遅延フィルタを実行時設計にすると係数更新コストと再現性が評価へ混ざるため、
    方式検証では標準バンクの grid と tap 数を固定する。
    """
    filter_bank = design_standard_fractional_delay_filter_bank()

    assert filter_bank.frac_grid.shape == (STANDARD_FRACTIONAL_DELAY_PATTERN_COUNT,)
    assert filter_bank.frac_filters.shape == (
        STANDARD_FRACTIONAL_DELAY_PATTERN_COUNT,
        STANDARD_FRACTIONAL_DELAY_TAP_COUNT,
    )
    np.testing.assert_allclose(filter_bank.frac_grid[0], -0.5)
    np.testing.assert_allclose(filter_bank.frac_grid[25], 0.0)
    np.testing.assert_allclose(filter_bank.frac_grid[-1], 0.5)
    np.testing.assert_allclose(np.diff(filter_bank.frac_grid), np.full(50, 0.02))


def test_fixed_delay_fractional_weights_use_selected_filter_per_channel_and_beam() -> None:
    """各 channel・各整相方位で選択済み小数遅延 FIR の実応答を w0 に反映する。

    `w0` が理想 steering だけから作られると、時間領域主経路の整数遅延・小数 FIR・
    チャネル平均と差分補正枝の基準がずれるため、選択済み FIR 応答を明示的に確認する。
    """
    filter_bank = design_standard_fractional_delay_filter_bank()
    delay_frac = np.array([[-0.5, 0.0], [0.5, 0.12]], dtype=np.float64)
    frac_filter_index = filter_bank.select_indices(delay_frac)
    delay_int = np.array([[1, 0], [2, 3]], dtype=np.int64)
    delay_table = DelayTable(
        arrival_delay_sec=np.zeros((2, 2), dtype=np.float64),
        steering_delay_sample=delay_int.astype(np.float64) + delay_frac,
        delay_int=delay_int,
        delay_frac=delay_frac,
        frac_filter_index=frac_filter_index,
    )
    frequencies_hz = np.array([0.0, 1000.0], dtype=np.float64)
    fs_hz = 8000.0

    weights = design_fixed_delay_fractional_weights_from_delay_table(
        delay_table,
        filter_bank,
        frequencies_hz,
        fs_hz=fs_hz,
        average_channels=True,
    )

    assert weights.shape == (2, 2, 2)
    np.testing.assert_array_equal(frac_filter_index, np.array([[0, 25], [50, 31]], dtype=np.int64))
    np.testing.assert_allclose(weights[0], np.full((2, 2), 0.5 + 0.0j), atol=1.0e-12)

    angular_frequency_rad = 2.0 * np.pi * frequencies_hz[1] / fs_hz
    tap_index = np.arange(filter_bank.n_tap, dtype=np.float64)
    expected_apply_response = np.zeros((2, 2), dtype=np.complex128)
    for ch_index in range(2):
        for beam_index in range(2):
            selected_taps = filter_bank.frac_filters[frac_filter_index[ch_index, beam_index]]
            fractional_response = np.sum(
                selected_taps * np.exp(-1j * angular_frequency_rad * tap_index)
            )
            integer_response = np.exp(-1j * angular_frequency_rad * delay_int[ch_index, beam_index])
            expected_apply_response[ch_index, beam_index] = (
                0.5 * integer_response * fractional_response
            )

    # weights[k, beam, ch] は w^H X 規約の w なので、実適用応答の共役・転置になる。
    np.testing.assert_allclose(weights[1], expected_apply_response.conj().T, atol=1.0e-12)


def test_fractional_delay_beamformer_applies_selected_128tap_filter_per_channel_and_beam() -> None:
    """時間領域固定整相器が beam×channel ごとに選択済み 128 tap FIR を適用する。

    impulse 入力を使うと、整数遅延位置から選択された小数遅延 FIR 係数そのものが
    channel 別整相出力に現れるため、フィルタ選択と畳み込みの対応を直接確認できる。
    """
    filter_bank = design_standard_fractional_delay_filter_bank()
    delay_frac = np.array([[-0.5, 0.0], [0.5, 0.12]], dtype=np.float64)
    frac_filter_index = filter_bank.select_indices(delay_frac)
    delay_int = np.array([[1, 0], [2, 3]], dtype=np.int64)
    delay_table = DelayTable(
        arrival_delay_sec=np.zeros((2, 2), dtype=np.float64),
        steering_delay_sample=delay_int.astype(np.float64) + delay_frac,
        delay_int=delay_int,
        delay_frac=delay_frac,
        frac_filter_index=frac_filter_index,
    )
    beamformer = FractionalDelayAndSumBeamformer(
        delay_table=delay_table,
        fractional_filter_bank=filter_bank,
        average_channels=False,
        fs_hz=8000.0,
    )
    input_signal = np.zeros((2, 140), dtype=np.complex128)
    input_signal[:, 0] = 1.0 + 0.0j

    process_result = beamformer.process(input_signal, return_steered_channels=True)
    assert isinstance(process_result, tuple)
    _beam_output, steered_channels = process_result

    for beam_index in range(2):
        for ch_index in range(2):
            expected = np.zeros(140, dtype=np.complex128)
            start = int(delay_int[ch_index, beam_index])
            selected_taps = filter_bank.frac_filters[frac_filter_index[ch_index, beam_index]]
            stop = min(start + filter_bank.n_tap, expected.size)
            # impulse が整数遅延後に FIR へ入るため、出力には選択 tap がそのまま現れる。
            expected[start:stop] = selected_taps[: stop - start]
            np.testing.assert_allclose(
                steered_channels[beam_index, ch_index],
                expected,
                atol=1.0e-12,
            )
