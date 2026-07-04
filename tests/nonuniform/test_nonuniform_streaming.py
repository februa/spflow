"""nonuniform streaming に関する回帰試験。"""

# 非均一分割は葉ごとのレート差と内部状態の持ち越しが複雑なため、
# 木構造・streaming・ビームフォーミング統合時の安全側挙動を回帰試験で固定する。

import numpy as np

from spflow.filterbank.nonuniform_streaming import (
    NonuniformTreeStreamingAnalyzer,
    NonuniformTreeStreamingSynthesizer,
)
from spflow.filterbank.nonuniform_tree import NonuniformTreeFilterBank


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


def test_nonuniform_streaming_analysis_matches_offline_analysis_for_analytic_signal():
    """nonuniform streamingについて analysis matches offline analysis for analytic signal を確認する。"""
    rng = np.random.default_rng(40)
    x = rng.standard_normal((3, 1000)) + 1j * rng.standard_normal((3, 1000))
    chunks = _chunk_signal(x, [13, 7, 129, 5, 257, 11, 61, 3, 401, 113])
    fb = NonuniformTreeFilterBank.default_for_fs(32768.0)

    offline = fb.analyze_analytic(x)
    analyzer = NonuniformTreeStreamingAnalyzer(fb)
    emitted_blocks = []
    for chunk in chunks:
        emitted_blocks.extend(analyzer.process_analytic(chunk))
    emitted_blocks.extend(analyzer.flush())
    streaming = analyzer.result()

    assert len(emitted_blocks) == offline.padded_length // fb.root_block_size
    assert streaming.original_length == offline.original_length
    assert streaming.padded_length == offline.padded_length
    for offline_packet, streaming_packet in zip(offline.packets, streaming.packets, strict=True):
        assert offline_packet.spec == streaming_packet.spec
        np.testing.assert_allclose(streaming_packet.samples, offline_packet.samples, atol=1e-6)


def test_nonuniform_streaming_synthesis_matches_offline_synthesis_for_analytic_signal():
    """nonuniform streamingについて 解析複素信号で offline 合成と一致する を確認する。"""
    rng = np.random.default_rng(41)
    x = rng.standard_normal((2, 1000)) + 1j * rng.standard_normal((2, 1000))
    chunks = _chunk_signal(x, [1, 255, 17, 19, 64, 2, 300, 9, 333])
    fb = NonuniformTreeFilterBank.default_for_fs(32768.0)

    offline_result = fb.analyze_analytic(x)
    offline_reconstructed = fb.synthesize(offline_result, analytic_output=True)

    analyzer = NonuniformTreeStreamingAnalyzer(fb)
    synthesizer = NonuniformTreeStreamingSynthesizer(fb)
    reconstructed_blocks = []
    for chunk in chunks:
        for block in analyzer.process_analytic(chunk):
            reconstructed_blocks.append(synthesizer.process_block(block))
    for block in analyzer.flush():
        reconstructed_blocks.append(synthesizer.process_block(block))

    streaming_reconstructed = np.concatenate(reconstructed_blocks, axis=-1)[..., : x.shape[-1]]

    np.testing.assert_allclose(streaming_reconstructed, x, atol=1e-5)
    np.testing.assert_allclose(streaming_reconstructed, offline_reconstructed, atol=1e-6)


def test_nonuniform_streaming_preserves_real_signal_reconstruction_after_offline_frontend():
    """nonuniform streamingがoffline frontend 後の実信号再構成を保つことを確認する。"""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    analytic = NonuniformTreeFilterBank._analytic_signal(x)
    chunks = _chunk_signal(analytic, [31, 17, 211, 9, 255, 63, 401, 13])
    fb = NonuniformTreeFilterBank.default_for_fs(32768.0)

    analyzer = NonuniformTreeStreamingAnalyzer(fb)
    synthesizer = NonuniformTreeStreamingSynthesizer(fb)
    reconstructed_blocks = []
    for chunk in chunks:
        for block in analyzer.process_analytic(chunk):
            reconstructed_blocks.append(synthesizer.process_block(block))
    for block in analyzer.flush():
        reconstructed_blocks.append(synthesizer.process_block(block))

    reconstructed = np.concatenate(reconstructed_blocks, axis=-1)[..., : x.shape[-1]]

    np.testing.assert_allclose(np.real(reconstructed), x, atol=1e-5)
    np.testing.assert_allclose(np.imag(reconstructed), np.imag(analytic), atol=1e-5)
