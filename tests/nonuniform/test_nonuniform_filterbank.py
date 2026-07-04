"""nonuniform filterbank に関する回帰試験。"""

import numpy as np

from spflow.filterbank.nonuniform_tree import (
    ComplexHalfbandPRBlockStage,
    NonuniformTreeFilterBank,
)


def test_complex_halfband_pr_block_stage_reconstructs_complex_signal():
    """対象機能について complex halfband pr block stage reconstructs complex signal を確認する。"""
    rng = np.random.default_rng(30)
    x = rng.standard_normal((3, 34)) + 1j * rng.standard_normal((3, 34))
    stage = ComplexHalfbandPRBlockStage()

    low, high = stage.analysis(x)
    reconstructed = stage.synthesis(low, high)

    np.testing.assert_allclose(reconstructed, x, atol=1e-6)


def test_nonuniform_tree_default_specs_match_requested_band_plan():
    """nonuniform treeの既定仕様が要求帯域分割に一致することを確認する。"""
    fb = NonuniformTreeFilterBank.default_for_fs(32768.0)

    assert [packet_band.band_id for packet_band in fb.band_specs] == [
        "0-128Hz",
        "128-256Hz",
        "256-512Hz",
        "512-1024Hz",
        "1024-2048Hz",
        "2048-4096Hz",
        "4096-8192Hz",
        "8192-16384Hz",
    ]
    assert [packet_band.target_resolution_hz for packet_band in fb.band_specs] == [1.0, 1.0, 2.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    assert [packet_band.tree_depth for packet_band in fb.band_specs] == [7, 7, 6, 5, 4, 3, 2, 1]
    assert [packet_band.nominal_sample_rate_hz for packet_band in fb.band_specs] == [
        256.0,
        256.0,
        512.0,
        1024.0,
        2048.0,
        4096.0,
        8192.0,
        16384.0,
    ]


def test_nonuniform_tree_reconstructs_real_signal_without_beamforming():
    """nonuniform treeがビームフォーミングなしで実信号を再構成することを確認する。"""
    rng = np.random.default_rng(31)
    x = rng.standard_normal(1000)
    fb = NonuniformTreeFilterBank.default_for_fs(32768.0)

    result = fb.analyze_real(x)
    reconstructed = fb.synthesize(result)

    np.testing.assert_allclose(reconstructed, x, atol=1e-5)


def test_nonuniform_tree_reconstructs_multichannel_real_signal_without_beamforming():
    """nonuniform treeがビームフォーミングなしで多チャネル実信号を再構成することを確認する。"""
    rng = np.random.default_rng(32)
    # Use a future beamforming-like channel count; geometry is irrelevant until spatial processing starts.
    x = rng.standard_normal((19, 777))
    fb = NonuniformTreeFilterBank.default_for_fs(32768.0)

    result = fb.analyze_real(x)
    reconstructed = fb.synthesize(result)

    np.testing.assert_allclose(reconstructed, x, atol=1e-5)


def test_nonuniform_tree_leaf_packet_lengths_follow_tree_depth():
    """nonuniform treeで leaf packet 長が tree 深さに従うことを確認する。"""
    rng = np.random.default_rng(33)
    x = rng.standard_normal(1000)
    fb = NonuniformTreeFilterBank.default_for_fs(32768.0)

    result = fb.analyze_real(x)

    assert result.padded_length == 1024
    for packet in result.packets:
        expected_length = result.padded_length // (1 << packet.spec.tree_depth)
        assert packet.samples.shape[-1] == expected_length


def test_nonuniform_tree_reconstructs_analytic_complex_signal():
    """nonuniform treeが解析複素信号を再構成することを確認する。"""
    n = np.arange(1000, dtype=np.float32)
    x = np.exp(1j * 2.0 * np.pi * 321.0 * n / 32768.0)
    fb = NonuniformTreeFilterBank.default_for_fs(32768.0)

    result = fb.analyze_analytic(x)
    reconstructed = fb.synthesize(result, analytic_output=True)

    np.testing.assert_allclose(reconstructed, x, atol=1e-5)
