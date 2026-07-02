"""prdft filterbank に関する回帰試験。"""

import numpy as np

from spflow import (
    DFTModulatedFilterDesigner,
    DFT_FilterBank,
    FiniteLengthPRChecker,
    FullDFTFilterBank,
    PRChecker,
    PRDFTAnalysisBank,
    PRDFTFilterBank,
    PRDFTSynthesisBank,
    PolyphaseDFTFilterBank,
    PolyphaseDecomposition,
    PolyphasePRDFTAnalysisBank,
    PolyphasePRDFTSynthesisBank,
    PolyphasePRPairDesigner,
    PrototypeAnalysisDFTFilterBank,
    PrototypeFilter,
    PrototypePairDesigner,
    PrototypeSynthesisDFTFilterBank,
)
from spflow.filterbank.design import make_pr_prototype


def test_pr_prototype_has_constant_overlap_add_power():
    """PR prototypeについて 一定 overlap-add power を持つ を確認する。"""
    prototype = make_pr_prototype(fft_size=16)
    power = prototype[:8] ** 2 + prototype[8:] ** 2

    np.testing.assert_allclose(power, np.ones_like(power), atol=1e-6)


def test_prdft_analysis_emits_positive_frequency_subbands():
    """PRDFT 解析について 正の周波数側 subband を出力する を確認する。"""
    fb = PRDFTFilterBank(fft_size=8)
    x = np.arange(24, dtype=np.float32).reshape(2, 12)

    subbands = fb.analysis(x)

    assert subbands.shape == (2, 5, 2)


def test_prdft_synthesis_reconstructs_real_signal():
    """PRDFT 合成について 実信号を再構成する を確認する。"""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 37))
    fb = PRDFTFilterBank(fft_size=16)

    subbands = fb.analysis(x)
    reconstructed = fb.synthesis(subbands, length=x.shape[-1])

    np.testing.assert_allclose(reconstructed, x, atol=1e-5)


def test_full_dft_filter_bank_reconstructs_real_signal():
    """full DFT filter bankについて 実信号を再構成する を確認する。"""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 37))
    fb = FullDFTFilterBank(fft_size=16)

    subbands = fb.analysis(x)
    reconstructed = fb.synthesis(subbands, length=x.shape[-1])

    np.testing.assert_allclose(np.real(reconstructed), x, atol=1e-5)
    np.testing.assert_allclose(np.imag(reconstructed), 0.0, atol=1e-5)


def test_full_dft_filter_bank_reconstructs_complex_signal_when_all_bands_are_kept():
    """full DFT filter bankが複素信号を再構成することを確認する。"""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    fb = FullDFTFilterBank(fft_size=16)

    subbands = fb.analysis(x)
    reconstructed = fb.synthesis(subbands, length=x.shape[-1])

    np.testing.assert_allclose(reconstructed, x, atol=1e-5)


def test_polyphase_dft_filter_bank_reconstructs_complex_signal():
    """polyphase DFT filter bankが複素信号を再構成することを確認する。"""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    fb = PolyphaseDFTFilterBank(fft_size=16)

    subbands = fb.analysis(x)
    reconstructed = fb.synthesis(subbands, length=x.shape[-1])

    np.testing.assert_allclose(reconstructed, x, atol=1e-5)


def test_polyphase_dft_filter_bank_is_closed_for_arbitrary_complex_subbands():
    """polyphase DFT filter bankが任意の複素 subband に対して閉じていることを確認する。"""
    rng = np.random.default_rng(5)
    fb = PolyphaseDFTFilterBank(fft_size=16)
    Y = rng.standard_normal((16, 9)) + 1j * rng.standard_normal((16, 9))

    y = fb.synthesis(Y)
    reanalyzed = fb.analysis(y)

    np.testing.assert_allclose(reanalyzed, Y, atol=1e-5)


def test_polyphase_dft_filter_bank_supports_multichannel_subbands():
    """polyphase DFT filter bankが多チャネル subband をサポートすることを確認する。"""
    rng = np.random.default_rng(6)
    fb = PolyphaseDFTFilterBank(fft_size=8)
    Y = rng.standard_normal((3, 8, 7)) + 1j * rng.standard_normal((3, 8, 7))

    y = fb.synthesis(Y)
    reanalyzed = fb.analysis(y)

    np.testing.assert_allclose(reanalyzed, Y, atol=1e-5)


def test_decimating_long_block_dft_bins_returns_alias_sum_of_short_block_dfts():
    """長 block DFT ビンの decimationが短 block DFT の alias 和を返すことを確認する。"""
    rng = np.random.default_rng(20)
    short_fft_size = 8
    block_ratio = 4
    long_fft_size = short_fft_size * block_ratio
    n_long_block = 5
    x = rng.standard_normal(long_fft_size * n_long_block) + 1j * rng.standard_normal(long_fft_size * n_long_block)

    long_fb = PolyphaseDFTFilterBank(fft_size=long_fft_size)
    short_fb = PolyphaseDFTFilterBank(fft_size=short_fft_size)

    Y_long = long_fb.analysis(x)
    Y_short = short_fb.analysis(x)
    Y_short_grouped = Y_short.reshape(short_fft_size, n_long_block, block_ratio).transpose(0, 2, 1)

    # Every M-th long-FFT bin is only the alias-sum of the M short-block spectra.
    np.testing.assert_allclose(
        Y_long[::block_ratio, :],
        np.sum(Y_short_grouped, axis=1),
        atol=1e-5,
    )


def test_decimated_long_block_dft_bins_do_not_uniquely_determine_short_block_dfts():
    """decimated 長 block DFT ビンでは短 block DFT が一意に決まらないことを確認する。"""
    rng = np.random.default_rng(21)
    short_fft_size = 8
    block_ratio = 4
    long_fft_size = short_fft_size * block_ratio

    short_subbands_a = rng.standard_normal((short_fft_size, block_ratio)) + 1j * rng.standard_normal(
        (short_fft_size, block_ratio)
    )
    perturbation = rng.standard_normal((short_fft_size, block_ratio)) + 1j * rng.standard_normal(
        (short_fft_size, block_ratio)
    )
    perturbation -= np.mean(perturbation, axis=1, keepdims=True)
    short_subbands_b = short_subbands_a + perturbation

    short_fb = PolyphaseDFTFilterBank(fft_size=short_fft_size)
    long_fb = PolyphaseDFTFilterBank(fft_size=long_fft_size)

    x_a = short_fb.synthesis(short_subbands_a, length=long_fft_size)
    x_b = short_fb.synthesis(short_subbands_b, length=long_fft_size)
    decimated_long_a = long_fb.analysis(x_a)[::block_ratio, 0]
    decimated_long_b = long_fb.analysis(x_b)[::block_ratio, 0]

    np.testing.assert_allclose(decimated_long_a, decimated_long_b, atol=1e-5)
    assert not np.allclose(short_subbands_a, short_subbands_b, atol=1e-6)


def test_modulated_filter_designer_builds_fft_order_filters():
    """modulated filter designerが指定 FFT 次数のフィルタを構成することを確認する。"""
    prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    designer = DFTModulatedFilterDesigner(n_band=32, decimation=32)

    analysis_filters = designer.analysis_filters(prototype)
    synthesis_filters = designer.synthesis_filters(prototype)

    assert analysis_filters.shape == (32, 256)
    assert synthesis_filters.shape == (32, 256)
    np.testing.assert_allclose(analysis_filters[0, :32], 1.0, atol=1e-6)
    np.testing.assert_allclose(synthesis_filters[0, :32], 1.0 / 32.0, atol=1e-6)
    np.testing.assert_allclose(analysis_filters[:, 32:], 0.0, atol=1e-6)
    np.testing.assert_allclose(synthesis_filters[:, 32:], 0.0, atol=1e-6)


def test_prototype_pair_designer_recovers_exact_block_dft_baseline_pair():
    """prototype pair designerがblock DFT 基準ペアを正確に復元することを確認する。"""
    rng = np.random.default_rng(11)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    analysis_prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    pair_designer = PrototypePairDesigner(n_band=32, decimation=32)
    cascade = pair_designer.build_cascade_matrix(analysis_prototype)
    synthesis_prototype = pair_designer.design_synthesis_prototype(
        analysis_prototype,
        delay_samples=0,
        cascade_matrix=cascade,
    )
    analysis = PRDFTAnalysisBank(prototype=analysis_prototype)
    synthesis = PRDFTSynthesisBank(prototype=synthesis_prototype, delay_compensation=0)
    checker = PRChecker(analysis, synthesis)

    pair_metrics = pair_designer.evaluate_pair_residual(
        analysis_prototype,
        synthesis_prototype,
        delay_samples=0,
        cascade_matrix=cascade,
    )
    pr_metrics = checker.check_perfect_reconstruction(x, length=x.shape[-1])

    assert pair_metrics["max_abs_error"] <= 2e-5
    assert pr_metrics["max_abs_error"] <= 2e-5
    assert pr_metrics["rms_error"] <= 5e-6


def test_finite_length_pr_checker_reports_exact_baseline_reconstruction():
    """有限長 PR checkerが基準再構成を正しく報告することを確認する。"""
    rng = np.random.default_rng(13)
    x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    checker = FiniteLengthPRChecker(
        PRDFTAnalysisBank(prototype=prototype),
        PRDFTSynthesisBank(prototype=prototype, delay_compensation=0),
    )

    metrics = checker.check(x)

    assert metrics["max_abs_error"] <= 2e-5
    assert metrics["valid_rms_error"] <= 5e-6


def test_finite_length_regularized_pair_improves_windowed_sinc_valid_region_error():
    """対象機能について 有限長 regularized pair が windowed sinc の valid 領域誤差を改善する を確認する。"""
    rng = np.random.default_rng(14)
    x = rng.standard_normal(512) + 1j * rng.standard_normal(512)
    analysis_prototype = PrototypeFilter.windowed_sinc(n_band=32, decimation=32, prototype_length=256)
    pair_designer = PrototypePairDesigner(n_band=32, decimation=32)
    cascade = pair_designer.build_cascade_matrix(analysis_prototype)

    unregularized = pair_designer.design_synthesis_prototype(
        analysis_prototype,
        delay_samples=0,
        cascade_matrix=cascade,
        regularization=0.0,
    )
    regularized = pair_designer.design_synthesis_prototype(
        analysis_prototype,
        delay_samples=0,
        cascade_matrix=cascade,
        regularization=1e-4,
    )

    unregularized_checker = FiniteLengthPRChecker(
        PRDFTAnalysisBank(prototype=analysis_prototype),
        PRDFTSynthesisBank(prototype=unregularized, delay_compensation=0),
    )
    regularized_checker = FiniteLengthPRChecker(
        PRDFTAnalysisBank(prototype=analysis_prototype),
        PRDFTSynthesisBank(prototype=regularized, delay_compensation=0),
    )

    unregularized_metrics = unregularized_checker.check(x)
    regularized_metrics = regularized_checker.check(x)

    assert regularized_metrics["valid_rms_error"] < unregularized_metrics["valid_rms_error"]


def test_prototype_pair_designer_reduces_windowed_sinc_phase_residual_over_self_synthesis():
    """prototype pair designerが自己合成より windowed sinc の位相残差を減らすことを確認する。"""
    analysis_prototype = PrototypeFilter.windowed_sinc(n_band=32, decimation=32, prototype_length=256)
    pair_designer = PrototypePairDesigner(n_band=32, decimation=32)
    cascade = pair_designer.build_cascade_matrix(analysis_prototype)
    synthesis_prototype = pair_designer.design_synthesis_prototype(
        analysis_prototype,
        delay_samples=0,
        cascade_matrix=cascade,
    )

    self_residual = pair_designer.evaluate_pair_residual(
        analysis_prototype,
        analysis_prototype,
        delay_samples=0,
        cascade_matrix=cascade,
    )
    paired_residual = pair_designer.evaluate_pair_residual(
        analysis_prototype,
        synthesis_prototype,
        delay_samples=0,
        cascade_matrix=cascade,
    )

    assert paired_residual["rms_error"] < self_residual["rms_error"]


def test_explicit_modulated_prdft_bank_reconstructs_block_dft_baseline_signal():
    """明示変調 PRDFT bankについて block DFT 基準信号を再構成する を確認する。"""
    rng = np.random.default_rng(9)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    analysis = PRDFTAnalysisBank(prototype=prototype)
    synthesis = PRDFTSynthesisBank(prototype=prototype, delay_compensation=0)
    checker = PRChecker(analysis, synthesis)

    metrics = checker.check_perfect_reconstruction(x, length=x.shape[-1])

    assert metrics["max_abs_error"] <= 2e-5
    assert metrics["rms_error"] <= 5e-6


def test_explicit_modulated_prdft_bank_is_not_yet_pr_for_windowed_sinc_prototype():
    """明示変調 PRDFT bankが windowed sinc prototype ではまだ完全再構成ではないことを確認する。"""
    rng = np.random.default_rng(10)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    prototype = PrototypeFilter.windowed_sinc(n_band=32, decimation=32, prototype_length=256)
    analysis = PRDFTAnalysisBank(prototype=prototype)
    synthesis = PRDFTSynthesisBank(prototype=prototype, delay_compensation=0)
    checker = PRChecker(analysis, synthesis)

    metrics = checker.check_perfect_reconstruction(x, length=x.shape[-1])

    assert metrics["rms_error"] > 1e-3


def test_prototype_filter_block_dft_baseline_reduces_to_exact_polyphase_baseline():
    """prototype filterで block DFT 基準が正確な polyphase 基準へ還元されることを確認する。"""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    analysis = PrototypeAnalysisDFTFilterBank(prototype=prototype)
    synthesis = PrototypeSynthesisDFTFilterBank(prototype=prototype, delay_compensation=0)
    checker = PRChecker(analysis, synthesis)

    metrics = checker.check_perfect_reconstruction(x, length=x.shape[-1])

    assert metrics["max_abs_error"] <= 2e-5
    assert metrics["rms_error"] <= 5e-6


def test_polyphase_decomposition_returns_expected_shape():
    """対象機能で polyphase 分解形状が期待通りになることを確認する。"""
    prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    decomposition = PolyphaseDecomposition(decimation=32)

    poly = decomposition.decompose(prototype)

    assert poly.shape == (8, 32)
    np.testing.assert_allclose(poly[0], np.ones(32), atol=1e-6)
    np.testing.assert_allclose(poly[1:], 0.0, atol=1e-6)


def test_pr_checker_reports_subband_closure_for_block_dft_baseline():
    """PR checkerが block DFT 基準の subband closure を報告することを確認する。"""
    rng = np.random.default_rng(8)
    Y = rng.standard_normal((32, 6)) + 1j * rng.standard_normal((32, 6))
    prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    analysis = PrototypeAnalysisDFTFilterBank(prototype=prototype)
    synthesis = PrototypeSynthesisDFTFilterBank(prototype=prototype, delay_compensation=0)
    checker = PRChecker(analysis, synthesis)

    metrics = checker.check_subband_closure(Y)

    assert metrics["max_abs_error"] <= 2e-5
    assert metrics["rms_error"] <= 5e-6


def test_windowed_sinc_prototype_response_has_stopband_attenuation():
    """対象機能で windowed sinc prototype 応答に stopband 減衰があることを確認する。"""
    prototype = PrototypeFilter.windowed_sinc(n_band=32, decimation=32, prototype_length=256)

    metrics = PRChecker.evaluate_prototype_response(prototype)

    assert metrics["passband_peak"] > 0.0
    assert metrics["stopband_peak"] < metrics["passband_peak"]
    assert metrics["stopband_attenuation_db"] > 10.0


def test_sine_window_spreads_bin_center_tone_across_neighboring_bands():
    """対象機能について sine window がビン中心トーンを隣接帯域へ広げる を確認する。"""
    fs = 16000.0
    fft_size = 256
    hop_size = 128
    freq = 1000.0
    x = np.cos(2.0 * np.pi * freq * np.arange(4096) / fs)
    pos_band = int(round(freq / (fs / fft_size)))
    neg_band = (-pos_band) % fft_size

    fb = FullDFTFilterBank(fft_size=fft_size, hop_size=hop_size)
    subbands = fb.analysis(x)
    band_energy = np.mean(np.abs(subbands) ** 2, axis=-1)
    retained_ratio = (band_energy[pos_band] + band_energy[neg_band]) / np.sum(band_energy)

    assert retained_ratio < 0.9


def test_rectangular_window_keeps_bin_center_tone_in_single_conjugate_pair():
    """対象機能について 矩形窓がビン中心トーンを単一の共役帯域対に保つ を確認する。"""
    fs = 16000.0
    fft_size = 256
    hop_size = 128
    freq = 1000.0
    x = np.cos(2.0 * np.pi * freq * np.arange(4096) / fs)
    pos_band = int(round(freq / (fs / fft_size)))
    neg_band = (-pos_band) % fft_size

    fb = FullDFTFilterBank(fft_size=fft_size, hop_size=hop_size, prototype=np.ones(fft_size))
    subbands = fb.analysis(x)
    band_energy = np.mean(np.abs(subbands) ** 2, axis=-1)
    retained_ratio = (band_energy[pos_band] + band_energy[neg_band]) / np.sum(band_energy)

    processed = np.zeros_like(subbands)
    processed[pos_band, :] = subbands[pos_band, :]
    processed[neg_band, :] = subbands[neg_band, :]
    reconstructed = fb.synthesis(processed, length=x.shape[-1])

    np.testing.assert_allclose(retained_ratio, 1.0, atol=1e-6)
    np.testing.assert_allclose(np.real(reconstructed), x, atol=1e-5)
    np.testing.assert_allclose(np.imag(reconstructed), 0.0, atol=1e-5)


def test_dft_filter_bank_alias_reconstructs_signal_without_beamforming():
    """対象機能について DFT filter bank alias がビームフォーミングなしで信号を再構成する を確認する。"""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(41)
    fb = DFT_FilterBank(fft_size=12, hop_size=6)

    y = fb.synthesis(fb.analysis(x), length=x.shape[-1])

    np.testing.assert_allclose(y, x, atol=1e-5)


def test_polyphase_prdft_bank_reconstructs_block_dft_baseline_signal():
    """対象機能について polyphase prdft bank block DFT 基準信号を再構成する を確認する。"""
    rng = np.random.default_rng(15)
    x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    analysis = PolyphasePRDFTAnalysisBank(prototype=prototype)
    synthesis = PolyphasePRDFTSynthesisBank(prototype=prototype, delay_compensation=0)
    checker = PRChecker(analysis, synthesis)

    metrics = checker.check_perfect_reconstruction(x, length=x.shape[-1])

    assert metrics['max_abs_error'] <= 2e-5
    assert metrics['rms_error'] <= 5e-6


def test_polyphase_pr_pair_designer_recovers_block_dft_baseline_pair():
    """対象機能について polyphase PR pair designer が block DFT 基準ペアを復元する を確認する。"""
    rng = np.random.default_rng(16)
    x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    designer = PolyphasePRPairDesigner(n_band=32, decimation=32)
    branch_matrices = designer.build_branch_matrices(prototype)
    synthesis_prototype = designer.design_synthesis_prototype(
        prototype,
        delay_blocks=0,
        branch_matrices=branch_matrices,
    )
    analysis = PolyphasePRDFTAnalysisBank(prototype=prototype)
    synthesis = PolyphasePRDFTSynthesisBank(prototype=synthesis_prototype, delay_compensation=0)
    checker = PRChecker(analysis, synthesis)

    pair_metrics = designer.evaluate_pair_residual(
        prototype,
        synthesis_prototype,
        delay_blocks=0,
        branch_matrices=branch_matrices,
    )
    pr_metrics = checker.check_perfect_reconstruction(x, length=x.shape[-1])

    assert pair_metrics['max_abs_error'] <= 2e-5
    assert pr_metrics['max_abs_error'] <= 2e-5
    assert pr_metrics['rms_error'] <= 5e-6


def test_polyphase_pr_pair_designer_improves_windowed_sinc_valid_region_error():
    """対象機能について polyphase PR pair designer が windowed sinc の valid 領域誤差を改善する を確認する。"""
    rng = np.random.default_rng(17)
    x = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    analysis_prototype = PrototypeFilter.windowed_sinc(n_band=32, decimation=32, prototype_length=256)
    designer = PolyphasePRPairDesigner(n_band=32, decimation=32)
    synthesis_prototype = designer.design_synthesis_prototype(
        analysis_prototype,
        delay_blocks=11,
        synthesis_prototype_length=512,
    )

    self_checker = FiniteLengthPRChecker(
        PolyphasePRDFTAnalysisBank(prototype=analysis_prototype),
        PolyphasePRDFTSynthesisBank(prototype=analysis_prototype, delay_compensation=0),
    )
    paired_checker = FiniteLengthPRChecker(
        PolyphasePRDFTAnalysisBank(prototype=analysis_prototype),
        PolyphasePRDFTSynthesisBank(prototype=synthesis_prototype, delay_compensation=11 * 32),
    )

    self_metrics = self_checker.check(x)
    paired_metrics = paired_checker.check(x)

    assert paired_metrics['valid_rms_error'] < self_metrics['valid_rms_error']
    assert paired_metrics['rms_error'] < self_metrics['rms_error']


def test_finite_length_pr_checker_supports_full_crop_mode():
    """有限長 PR checkerがfull crop mode をサポートすることを確認する。"""
    rng = np.random.default_rng(18)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    checker = FiniteLengthPRChecker(
        PRDFTAnalysisBank(prototype=prototype),
        PRDFTSynthesisBank(prototype=prototype, delay_compensation=0),
    )

    metrics = checker.check(x, crop_mode='full', valid_region_mode='none')

    assert metrics['max_abs_error'] <= 2e-5
    assert metrics['rms_error'] <= 5e-6


def test_finite_length_pr_checker_supports_valid_crop_and_custom_margin():
    """有限長 PR checkerがvalid crop と custom margin をサポートすることを確認する。"""
    rng = np.random.default_rng(19)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    prototype = PrototypeFilter.block_dft_baseline(n_band=32, decimation=32, prototype_length=256)
    checker = FiniteLengthPRChecker(
        PRDFTAnalysisBank(prototype=prototype),
        PRDFTSynthesisBank(prototype=prototype, delay_compensation=0),
    )

    metrics = checker.check(
        x,
        crop_mode='valid',
        valid_region_mode='custom',
        valid_margin=0,
    )

    assert metrics['max_abs_error'] <= 2e-5
    assert metrics['valid_rms_error'] <= 5e-6
