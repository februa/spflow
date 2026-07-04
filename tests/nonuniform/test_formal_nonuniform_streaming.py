"""formal nonuniform streaming に関する回帰試験。"""

# 非均一分割は葉ごとのレート差と内部状態の持ち越しが複雑なため、
# 木構造・streaming・ビームフォーミング統合時の安全側挙動を回帰試験で固定する。

import numpy as np

from spflow.filterbank.causal_analytic_frontend import (
    CausalAnalyticFrontend,
    CausalAnalyticFrontendStreamer,
)
from spflow.filterbank.formal_nonuniform_streaming import (
    FormalNonuniformTreeStreamingAnalyzer,
    FormalNonuniformTreeStreamingSynthesizer,
    OracleFormalNonuniformTreeStreamingSynthesizer,
)
from spflow.filterbank.formal_nonuniform_tree import FormalNonuniformTreeFilterBank


def _chunk_signal(x: np.ndarray, chunk_sizes: list[int]) -> list[np.ndarray]:
    """`_chunk_signal` を実行する。"""
    chunks = []
    start = 0
    for size in chunk_sizes:
        stop = min(start + size, x.shape[-1])
        if stop > start:
            chunks.append(x[..., start:stop])
        start = stop
    if start < x.shape[-1]:
        chunks.append(x[..., start:])
    return chunks


def test_formal_nonuniform_streaming_analysis_matches_offline_for_analytic_signal():
    """formal nonuniform streamingが解析複素信号で offline 実装と一致することを確認する。"""
    rng = np.random.default_rng(820)
    x = rng.standard_normal((2, 1000)) + 1j * rng.standard_normal((2, 1000))
    chunks = _chunk_signal(x, [13, 7, 129, 5, 257, 11, 61, 3, 401, 113])
    fb = FormalNonuniformTreeFilterBank.default_for_fs(32768.0)

    offline = fb.analyze_analytic(x)
    analyzer = FormalNonuniformTreeStreamingAnalyzer(fb)
    emitted_blocks = []
    for chunk in chunks:
        emitted_blocks.extend(analyzer.process_analytic(chunk))
    emitted_blocks.extend(analyzer.flush())
    streaming = analyzer.result()

    assert emitted_blocks
    assert streaming.original_length == offline.original_length
    assert streaming.analytic_length == offline.analytic_length
    assert streaming.padded_length == offline.padded_length
    for offline_packet, streaming_packet in zip(offline.packets, streaming.packets, strict=True):
        assert streaming_packet.band_id == offline_packet.band_id
        assert streaming_packet.delay_samples_at_root_rate == offline_packet.delay_samples_at_root_rate
        assert streaming_packet.time_origin_at_root_rate == offline_packet.time_origin_at_root_rate
        np.testing.assert_allclose(streaming_packet.complex_samples, offline_packet.complex_samples, atol=1e-6)

    per_band_chunks: dict[str, list] = {spec.band_id: [] for spec in fb.band_specs}
    for block in emitted_blocks:
        for packet in block.packets:
            per_band_chunks[packet.band_id].append(packet)
    for spec in fb.band_specs:
        packets = per_band_chunks[spec.band_id]
        if not packets:
            continue
        scale = int(round(fb.fs_hz / spec.nominal_sample_rate_hz))
        total = 0
        for idx, packet in enumerate(packets):
            if idx == 0:
                total += packet.complex_samples.shape[-1]
                continue
            assert packet.time_origin_at_root_rate == packets[0].time_origin_at_root_rate + total * scale
            total += packet.complex_samples.shape[-1]


def test_formal_nonuniform_streaming_synthesis_matches_offline_and_oracle_for_analytic_signal():
    """formal nonuniform streamingが解析複素信号で offline 実装と oracle に一致することを確認する。"""
    rng = np.random.default_rng(821)
    x = rng.standard_normal((2, 1000)) + 1j * rng.standard_normal((2, 1000))
    chunks = _chunk_signal(x, [1, 255, 17, 19, 64, 2, 300, 9, 333])
    fb = FormalNonuniformTreeFilterBank.default_for_fs(32768.0)

    offline_result = fb.analyze_analytic(x)
    offline_reconstructed = fb.synthesize(offline_result, analytic_output=True)

    analyzer = FormalNonuniformTreeStreamingAnalyzer(fb)
    synthesizer = FormalNonuniformTreeStreamingSynthesizer(fb)
    oracle = OracleFormalNonuniformTreeStreamingSynthesizer(fb)
    reconstructed_blocks = []
    oracle_blocks = []
    for chunk in chunks:
        for block in analyzer.process_analytic(chunk):
            reconstructed_blocks.append(synthesizer.process_block(block))
            oracle_blocks.append(oracle.process_block(block))
    for block in analyzer.flush():
        reconstructed_blocks.append(synthesizer.process_block(block))
        oracle_blocks.append(oracle.process_block(block))

    streaming_reconstructed = np.concatenate(reconstructed_blocks, axis=-1)[..., : x.shape[-1]]
    oracle_reconstructed = np.concatenate(oracle_blocks, axis=-1)[..., : x.shape[-1]]

    np.testing.assert_allclose(streaming_reconstructed, x, atol=1e-5)
    np.testing.assert_allclose(streaming_reconstructed, offline_reconstructed, atol=1e-6)
    np.testing.assert_allclose(streaming_reconstructed, oracle_reconstructed, atol=1e-6)


def test_formal_nonuniform_streaming_real_frontend_pipeline_matches_offline_and_oracle():
    """formal nonuniform streamingの実信号 frontend pipeline が offline 実装と oracle に一致することを確認する。"""
    rng = np.random.default_rng(822)
    x = rng.standard_normal(999)
    chunks = _chunk_signal(x, [9, 31, 7, 63, 5, 127, 11, 211, 17, 283])
    frontend = CausalAnalyticFrontend.default(num_taps=31)
    fb = FormalNonuniformTreeFilterBank.default_for_fs(32768.0, frontend=frontend)

    offline_result = fb.analyze_real(x)
    offline_analytic = fb.synthesize(offline_result, analytic_output=True)

    frontend_streamer = CausalAnalyticFrontendStreamer(frontend)
    analyzer = FormalNonuniformTreeStreamingAnalyzer(fb)
    synthesizer = FormalNonuniformTreeStreamingSynthesizer(fb)
    oracle = OracleFormalNonuniformTreeStreamingSynthesizer(fb)
    reconstructed_blocks = []
    oracle_blocks = []

    for chunk in chunks:
        analytic_chunk = frontend_streamer.process(chunk).samples
        for block in analyzer.process_analytic(analytic_chunk):
            reconstructed_blocks.append(synthesizer.process_block(block))
            oracle_blocks.append(oracle.process_block(block))

    analytic_tail = frontend_streamer.flush().samples
    for block in analyzer.process_analytic(analytic_tail):
        reconstructed_blocks.append(synthesizer.process_block(block))
        oracle_blocks.append(oracle.process_block(block))
    for block in analyzer.flush():
        reconstructed_blocks.append(synthesizer.process_block(block))
        oracle_blocks.append(oracle.process_block(block))

    streaming_analytic = np.concatenate(reconstructed_blocks, axis=-1)[..., : offline_analytic.shape[-1]]
    oracle_analytic = np.concatenate(oracle_blocks, axis=-1)[..., : offline_analytic.shape[-1]]
    streaming_real = frontend.recover_real(streaming_analytic, length=x.shape[-1])

    np.testing.assert_allclose(streaming_analytic, offline_analytic, atol=1e-5)
    np.testing.assert_allclose(streaming_analytic, oracle_analytic, atol=1e-6)
    np.testing.assert_allclose(streaming_real, x, atol=1e-5)
