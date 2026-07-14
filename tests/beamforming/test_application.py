"""方式に依存しないbeamforming重み適用と責務別import経路を検証する。"""

from __future__ import annotations

import numpy as np

from spflow.array_design import BandwiseArrayDesign
from spflow.beamforming.application import (
    apply_beamformer,
    apply_beamformer_bands,
    apply_beamformer_filter_fft,
    apply_time_domain_fir_beamformer,
)
from spflow.beamforming.cbf import design_cbf_coefficients, design_cbf_weights
from spflow.beamforming.mvdr_filter import apply_beamformer as legacy_apply_beamformer
from spflow.beamforming.mvdr_weight_designer import design_mvdr_coefficients, design_mvdr_weights
from spflow.beamforming_evaluation import SourceSectorMask
from spflow.sidelobe_cancellation import BeamDomainSLC


def test_responsibility_packages_own_array_design_and_slc_implementations() -> None:
    """互換importではなく責務別packageが実装本体を所有することを確認する。"""

    assert BandwiseArrayDesign.__module__.startswith("spflow.array_design.")
    assert SourceSectorMask.__module__.startswith("spflow.beamforming_evaluation.")
    assert BeamDomainSLC.__module__.startswith("spflow.sidelobe_cancellation.")
    assert legacy_apply_beamformer is apply_beamformer


def test_legacy_weight_design_names_return_direct_application_coefficients() -> None:
    """旧weight設計名が新しい実適用係数と同じ配列を返すことを確認する。"""

    steering = np.array([[1.0 + 0.0j], [0.0 + 1.0j]], dtype=np.complex64)
    covariance = np.eye(2, dtype=np.complex64)

    np.testing.assert_allclose(design_cbf_weights(steering), design_cbf_coefficients(steering))
    np.testing.assert_allclose(
        design_mvdr_weights(covariance, steering, diag_load=0.0),
        design_mvdr_coefficients(covariance, steering, diag_load=0.0),
    )


def test_single_band_and_banded_application_use_same_transpose_inner_product() -> None:
    """単一帯域と帯域別適用が同じ`h^T x`規約になることを確認する。"""

    snapshots = np.array([[1.0 + 1.0j, 2.0], [3.0, 4.0 - 1.0j]], dtype=np.complex64)
    weights = np.array([[0.5 + 0.25j], [0.5 - 0.25j]], dtype=np.complex64)

    single_output = apply_beamformer(snapshots, weights)
    banded_output = apply_beamformer_bands(
        snapshots[:, np.newaxis, :],
        weights[:, :, np.newaxis],
    )

    # banded入力のaxis=1をband、axis=2をframeとして並べ替えたため、
    # 単一帯域出力のframe軸とbanded出力のframe軸を直接比較する。
    np.testing.assert_allclose(banded_output[:, 0, :], single_output, atol=1.0e-6)


def test_time_and_frequency_application_use_coefficients_without_conjugation() -> None:
    """時間FIRとfilter FFTが設計済み係数を共役せず適用することを確認する。"""

    channel_signals = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float64)
    weights = np.array([0.25 + 0.5j, 0.75 - 0.25j], dtype=np.complex128)

    time_output = apply_time_domain_fir_beamformer(
        channel_signals,
        weights,
        tap_len=1,
    )
    expected_time_output = weights[np.newaxis, :] @ channel_signals
    np.testing.assert_allclose(time_output, expected_time_output, atol=1.0e-12)

    input_spectrum = np.fft.fft(channel_signals, axis=1).astype(np.complex64)
    # 設計済み係数を時間・周波数領域の両方でそのまま掛け、適用側では共役しない。
    filter_spectrum = np.repeat(
        weights.astype(np.complex64)[:, np.newaxis, np.newaxis],
        input_spectrum.shape[1],
        axis=2,
    )
    frequency_output = apply_beamformer_filter_fft(input_spectrum, filter_spectrum)
    np.testing.assert_allclose(
        np.fft.ifft(frequency_output, axis=1),
        expected_time_output,
        atol=1.0e-5,
    )
