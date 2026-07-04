"""formal complex pr stage に関する回帰試験。"""

# 非均一分割は葉ごとのレート差と内部状態の持ち越しが複雑なため、
# 木構造・streaming・ビームフォーミング統合時の安全側挙動を回帰試験で固定する。

import numpy as np

from spflow.filterbank.formal_complex_pr_stage import FormalBandPacket, FormalComplexPRHalfbandStage


def test_formal_complex_pr_stage_updates_band_metadata_and_reconstructs_signal():
    """formal complex PR stageが帯域 metadata を更新しつつ信号を再構成することを確認する。"""
    rng = np.random.default_rng(300)
    x = rng.standard_normal((2, 128)) + 1j * rng.standard_normal((2, 128))
    packet = FormalBandPacket(
        band_id="0-16384Hz",
        f_low_hz=0.0,
        f_high_hz=16384.0,
        sample_rate_hz=32768.0,
        time_origin_at_root_rate=0,
        delay_samples_at_root_rate=0,
        complex_samples=x,
    )
    stage = FormalComplexPRHalfbandStage.from_candidate("daubechies_qmf_order4_taps8", root_sample_rate_hz=32768.0)

    low_packet, high_packet = stage.analyze_packet(packet)
    reconstructed = stage.synthesize_packets(low_packet, high_packet, length=x.shape[-1])

    assert low_packet.band_id == "0-8192Hz"
    assert high_packet.band_id == "8192-16384Hz"
    np.testing.assert_allclose(low_packet.sample_rate_hz, 16384.0, atol=1e-6)
    np.testing.assert_allclose(high_packet.sample_rate_hz, 16384.0, atol=1e-6)
    assert low_packet.delay_samples_at_root_rate >= packet.delay_samples_at_root_rate
    assert high_packet.delay_samples_at_root_rate >= packet.delay_samples_at_root_rate
    assert low_packet.time_origin_at_root_rate == 1
    assert high_packet.time_origin_at_root_rate == 1
    assert reconstructed.time_origin_at_root_rate == packet.time_origin_at_root_rate
    np.testing.assert_allclose(reconstructed.complex_samples, x, atol=1e-5)


def test_formal_complex_pr_stage_upper_child_is_lower_edge_referenced():
    """formal complex PR stageで上側子帯域が下端基準で参照されることを確認する。"""
    fs_hz = 32768.0
    input_freq_hz = 9000.0
    n = np.arange(4096, dtype=np.float32)
    x = np.exp(1j * 2.0 * np.pi * input_freq_hz * n / fs_hz)
    packet = FormalBandPacket(
        band_id="0-16384Hz",
        f_low_hz=0.0,
        f_high_hz=16384.0,
        sample_rate_hz=fs_hz,
        time_origin_at_root_rate=0,
        delay_samples_at_root_rate=0,
        complex_samples=x,
    )
    stage = FormalComplexPRHalfbandStage.from_candidate("daubechies_qmf_order4_taps8", root_sample_rate_hz=fs_hz)

    _, high_packet = stage.analyze_packet(packet)
    steady = high_packet.complex_samples[64:]
    phase_step = np.angle(np.vdot(steady[:-1], steady[1:]))
    estimated_freq_hz = phase_step * high_packet.sample_rate_hz / (2.0 * np.pi)

    np.testing.assert_allclose(estimated_freq_hz, input_freq_hz - high_packet.f_low_hz, atol=5.0)


def test_formal_complex_pr_stage_requires_integer_root_rate_ratio():
    """formal complex PR stageで整数の root rate 比を要求することを確認する。"""
    packet = FormalBandPacket(
        band_id="0-5Hz",
        f_low_hz=0.0,
        f_high_hz=5.0,
        sample_rate_hz=10.0,
        time_origin_at_root_rate=0,
        delay_samples_at_root_rate=0,
        complex_samples=np.ones(16, dtype=np.complex64),
    )
    stage = FormalComplexPRHalfbandStage.from_candidate("haar_qmf_taps2", root_sample_rate_hz=33.0)

    try:
        stage.analyze_packet(packet)
    except ValueError as exc:
        assert "divide root_sample_rate_hz by an integer factor" in str(exc)
    else:
        raise AssertionError("Expected analyze_packet to reject non-integer root-rate ratios.")
