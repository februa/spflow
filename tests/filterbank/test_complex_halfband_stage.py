"""complex halfband stage に関する回帰試験。"""

import numpy as np

from spflow.filterbank.complex_halfband_stage import (
    ComplexFIRHalfbandStage,
    ComplexFIRHalfbandStageFilters,
    ComplexFIRHalfbandStageStreamingAnalyzer,
    ComplexFIRHalfbandStageStreamingSynthesizer,
    OracleComplexFIRHalfbandStageStreamingAnalyzer,
    OracleComplexFIRHalfbandStageStreamingSynthesizer,
)
from spflow.filterbank.halfband_stage_candidates import get_known_qmf_candidate


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


def test_haar_paraunitary_halfband_stage_reconstructs_complex_signal():
    """Haar paraunitary halfband stageが複素信号を再構成することを確認する。"""
    rng = np.random.default_rng(50)
    x = rng.standard_normal((4, 129)) + 1j * rng.standard_normal((4, 129))
    stage = ComplexFIRHalfbandStage(ComplexFIRHalfbandStageFilters.haar_paraunitary())

    low, high = stage.analysis(x)
    reconstructed = stage.synthesis(low, high, length=x.shape[-1])

    np.testing.assert_allclose(reconstructed, x, atol=1e-6)


def test_haar_paraunitary_halfband_stage_streaming_analysis_matches_offline():
    """Haar paraunitary halfband stageについて streaming 解析が offline 実装と一致する を確認する。"""
    rng = np.random.default_rng(51)
    x = rng.standard_normal((2, 129)) + 1j * rng.standard_normal((2, 129))
    chunks = _chunk_signal(x, [1, 7, 3, 32, 5, 11, 48, 9])
    stage = ComplexFIRHalfbandStage(ComplexFIRHalfbandStageFilters.haar_paraunitary())

    offline_low, offline_high = stage.analysis(x)
    analyzer = ComplexFIRHalfbandStageStreamingAnalyzer(stage)

    low_chunks = []
    high_chunks = []
    for chunk in chunks:
        low_chunk, high_chunk = analyzer.process(chunk)
        low_chunks.append(low_chunk)
        high_chunks.append(high_chunk)
    tail_low, tail_high = analyzer.flush()
    low_chunks.append(tail_low)
    high_chunks.append(tail_high)

    streaming_low = np.concatenate(low_chunks, axis=-1)
    streaming_high = np.concatenate(high_chunks, axis=-1)

    np.testing.assert_allclose(streaming_low, offline_low, atol=1e-6)
    np.testing.assert_allclose(streaming_high, offline_high, atol=1e-6)


def test_haar_paraunitary_halfband_stage_streaming_synthesis_matches_offline():
    """Haar paraunitary halfband stageについて streaming 合成が offline 実装と一致する を確認する。"""
    rng = np.random.default_rng(52)
    x = rng.standard_normal((3, 129)) + 1j * rng.standard_normal((3, 129))
    stage = ComplexFIRHalfbandStage(ComplexFIRHalfbandStageFilters.haar_paraunitary())

    low, high = stage.analysis(x)
    offline_reconstructed = stage.synthesis(low, high, length=x.shape[-1])

    low_chunks = _chunk_signal(low, [2, 1, 9, 4, 7, 16, 3, 20])
    high_chunks = _chunk_signal(high, [2, 1, 9, 4, 7, 16, 3, 20])
    synthesizer = ComplexFIRHalfbandStageStreamingSynthesizer(stage)

    recon_chunks = []
    for low_chunk, high_chunk in zip(low_chunks, high_chunks, strict=True):
        recon_chunks.append(synthesizer.process(low_chunk, high_chunk))
    recon_chunks.append(synthesizer.flush())

    streaming_reconstructed = np.concatenate(recon_chunks, axis=-1)[..., : x.shape[-1]]

    np.testing.assert_allclose(streaming_reconstructed, x, atol=1e-6)
    np.testing.assert_allclose(streaming_reconstructed, offline_reconstructed, atol=1e-6)


def test_daubechies_qmf_stage_streaming_analysis_matches_oracle():
    """Daubechies QMF stageについて streaming 解析が oracle 実装と一致する を確認する。"""
    rng = np.random.default_rng(53)
    x = rng.standard_normal((2, 257)) + 1j * rng.standard_normal((2, 257))
    chunks = _chunk_signal(x, [1, 5, 2, 17, 4, 31, 11, 57, 19])
    stage = get_known_qmf_candidate("daubechies_qmf_order4_taps8").make_stage()

    analyzer = ComplexFIRHalfbandStageStreamingAnalyzer(stage)
    oracle = OracleComplexFIRHalfbandStageStreamingAnalyzer(stage)

    low_chunks = []
    high_chunks = []
    oracle_low_chunks = []
    oracle_high_chunks = []
    for chunk in chunks:
        low_chunk, high_chunk = analyzer.process(chunk)
        oracle_low_chunk, oracle_high_chunk = oracle.process(chunk)
        low_chunks.append(low_chunk)
        high_chunks.append(high_chunk)
        oracle_low_chunks.append(oracle_low_chunk)
        oracle_high_chunks.append(oracle_high_chunk)

    tail_low, tail_high = analyzer.flush()
    oracle_tail_low, oracle_tail_high = oracle.flush()
    low_chunks.append(tail_low)
    high_chunks.append(tail_high)
    oracle_low_chunks.append(oracle_tail_low)
    oracle_high_chunks.append(oracle_tail_high)

    np.testing.assert_allclose(np.concatenate(low_chunks, axis=-1), np.concatenate(oracle_low_chunks, axis=-1), atol=1e-6)
    np.testing.assert_allclose(np.concatenate(high_chunks, axis=-1), np.concatenate(oracle_high_chunks, axis=-1), atol=1e-6)


def test_daubechies_qmf_stage_streaming_synthesis_matches_oracle():
    """Daubechies QMF stageについて streaming 合成が oracle 実装と一致する を確認する。"""
    rng = np.random.default_rng(54)
    x = rng.standard_normal((2, 257)) + 1j * rng.standard_normal((2, 257))
    stage = get_known_qmf_candidate("daubechies_qmf_order4_taps8").make_stage()
    low, high = stage.analysis(x)
    low_chunks = _chunk_signal(low, [1, 3, 2, 11, 7, 29, 5, 23, 9])
    high_chunks = _chunk_signal(high, [1, 3, 2, 11, 7, 29, 5, 23, 9])

    synthesizer = ComplexFIRHalfbandStageStreamingSynthesizer(stage)
    oracle = OracleComplexFIRHalfbandStageStreamingSynthesizer(stage)

    recon_chunks = []
    oracle_chunks = []
    for low_chunk, high_chunk in zip(low_chunks, high_chunks, strict=True):
        recon_chunks.append(synthesizer.process(low_chunk, high_chunk))
        oracle_chunks.append(oracle.process(low_chunk, high_chunk))
    recon_chunks.append(synthesizer.flush())
    oracle_chunks.append(oracle.flush())

    np.testing.assert_allclose(np.concatenate(recon_chunks, axis=-1), np.concatenate(oracle_chunks, axis=-1), atol=1e-6)
