"""formal nonuniform tree に関する回帰試験。"""

# 非均一分割は葉ごとのレート差と内部状態の持ち越しが複雑なため、
# 木構造・streaming・ビームフォーミング統合時の安全側挙動を回帰試験で固定する。

import numpy as np

from spflow.filterbank.causal_analytic_frontend import CausalAnalyticFrontend
from spflow.filterbank.formal_nonuniform_tree import FormalNonuniformTreeFilterBank


def test_formal_nonuniform_tree_default_specs_match_requested_band_plan():
    """formal nonuniform treeの既定仕様が要求帯域分割に一致することを確認する。"""
    fb = FormalNonuniformTreeFilterBank.default_for_fs(32768.0)

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


def test_formal_nonuniform_tree_reconstructs_analytic_signal_and_propagates_metadata():
    """formal nonuniform treeが解析信号を再構成し metadata を伝搬することを確認する。"""
    rng = np.random.default_rng(800)
    x = rng.standard_normal((2, 1000)) + 1j * rng.standard_normal((2, 1000))
    fb = FormalNonuniformTreeFilterBank.default_for_fs(32768.0)

    result = fb.analyze_analytic(x)
    reconstructed = fb.synthesize(result, analytic_output=True)

    np.testing.assert_allclose(reconstructed, x, atol=1e-5)
    assert result.original_length == x.shape[-1]
    assert result.analytic_length == x.shape[-1]
    assert result.padded_length % fb.root_block_size == 0
    assert [packet.band_id for packet in result.packets] == [
        "0-128Hz",
        "128-256Hz",
        "256-512Hz",
        "512-1024Hz",
        "1024-2048Hz",
        "2048-4096Hz",
        "4096-8192Hz",
        "8192-16384Hz",
    ]

    packet_map = {packet.band_id: packet for packet in result.packets}
    assert packet_map["0-128Hz"].delay_samples_at_root_rate == packet_map["128-256Hz"].delay_samples_at_root_rate
    assert packet_map["0-128Hz"].delay_samples_at_root_rate > packet_map["8192-16384Hz"].delay_samples_at_root_rate
    assert packet_map["0-128Hz"].time_origin_at_root_rate == packet_map["128-256Hz"].time_origin_at_root_rate
    assert packet_map["0-128Hz"].time_origin_at_root_rate > packet_map["8192-16384Hz"].time_origin_at_root_rate
    assert packet_map["8192-16384Hz"].time_origin_at_root_rate == 1


def test_formal_nonuniform_tree_end_to_end_real_input_recovers_frontend_and_original_signal():
    """formal nonuniform treeについて end-to-end の実信号入力で frontend 出力と元信号を復元する を確認する。"""
    rng = np.random.default_rng(801)
    x = rng.standard_normal(999)
    frontend = CausalAnalyticFrontend.default(num_taps=31)
    fb = FormalNonuniformTreeFilterBank.default_for_fs(32768.0, frontend=frontend)

    result = fb.analyze_real(x)
    analytic_reconstructed = fb.synthesize(result, analytic_output=True)
    expected_analytic = frontend.analyze(x, pad_tail=True).samples
    real_reconstructed = fb.synthesize(result)

    np.testing.assert_allclose(analytic_reconstructed, expected_analytic, atol=1e-5)
    np.testing.assert_allclose(real_reconstructed, x, atol=1e-5)
    assert result.frontend_delay_samples_at_root_rate == frontend.delay_samples
    assert result.analytic_length == expected_analytic.shape[-1]
