"""causal analytic frontend に関する回帰試験。"""

import numpy as np

from spflow.filterbank.causal_analytic_frontend import (
    CausalAnalyticFrontend,
    CausalAnalyticFrontendStreamer,
    design_hilbert_fir,
)


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


def test_design_hilbert_fir_requires_odd_length():
    """Hilbert FIR 設計について 奇数長を要求する を確認する。"""
    taps = design_hilbert_fir(63)

    assert taps.shape == (63,)
    assert np.isclose(taps[31], 0.0)


def test_causal_analytic_frontend_streaming_matches_offline_with_tail():
    """causal analytic frontendについて tail 付き offline 実装と一致する を確認する。"""
    rng = np.random.default_rng(200)
    x = rng.standard_normal((2, 127))
    frontend = CausalAnalyticFrontend.default(num_taps=31)
    offline = frontend.analyze(x, pad_tail=True)

    streamer = CausalAnalyticFrontendStreamer(frontend)
    pieces = []
    for chunk in _chunk_signal(x, [1, 11, 3, 29, 7, 13, 2, 41]):
        pieces.append(streamer.process(chunk).samples)
    pieces.append(streamer.flush().samples)
    streaming = np.concatenate(pieces, axis=-1)

    np.testing.assert_allclose(streaming, offline.samples, atol=1e-6)
    np.testing.assert_allclose(frontend.recover_real(streaming, length=x.shape[-1]), x, atol=1e-6)


def test_causal_analytic_frontend_suppresses_negative_frequency_for_positive_tone():
    """causal analytic frontendについて 正の周波数トーンに対して負周波数成分を抑圧する を確認する。"""
    fs = 32768.0
    freq = 1000.0
    n = np.arange(4096, dtype=np.float32)
    x = np.cos(2.0 * np.pi * freq * n / fs)
    frontend = CausalAnalyticFrontend.default(num_taps=63)
    result = frontend.analyze(x, pad_tail=True)
    samples = result.samples

    spectrum = np.fft.fft(samples)
    pos_bin = int(round(freq * samples.shape[-1] / fs))
    neg_bin = (-pos_bin) % samples.shape[-1]
    suppression_db = 20.0 * np.log10(
        max(np.abs(spectrum[pos_bin]), np.finfo(np.float32).tiny)
        / max(np.abs(spectrum[neg_bin]), np.finfo(np.float32).tiny)
    )

    assert suppression_db >= 20.0
