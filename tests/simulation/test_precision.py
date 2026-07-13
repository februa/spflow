"""シミュレーション精度が生成元から逐次処理まで伝播することを確認する。"""

from __future__ import annotations

import numpy as np
import pytest

from spflow.beamforming.ebae import EbaeConfig, design_ebae_weights_band
from spflow.simulation import (
    AlignmentSimulationConfig,
    SignalBlock,
    SimulationPrecision,
    StatefulIntegerDelay,
    VersionedCausalFIR,
    approximate_frequency_weights_with_fir,
    design_alignment_weights,
)


@pytest.mark.parametrize(
    ("precision", "real_dtype", "complex_dtype"),
    (
        (SimulationPrecision.SINGLE, np.dtype(np.float32), np.dtype(np.complex64)),
        (SimulationPrecision.DOUBLE, np.dtype(np.float64), np.dtype(np.complex128)),
    ),
)
def test_precision_propagates_from_alignment_design_to_streaming(
    precision: SimulationPrecision,
    real_dtype: np.dtype[np.float32] | np.dtype[np.float64],
    complex_dtype: np.dtype[np.complex64] | np.dtype[np.complex128],
) -> None:
    """一つの精度指定が重み、FIR係数、逐次出力へ伝播することを確認する。"""
    config = AlignmentSimulationConfig(
        fs_hz=16.0,
        fft_size=16,
        sound_speed_m_per_s=1500.0,
        sensor_positions_m=np.asarray([-1.0, 0.0, 1.0], dtype=real_dtype),
        beam_azimuth_deg=np.asarray([0.0, 90.0, 180.0], dtype=real_dtype),
        target_azimuth_deg=90.0,
        target_band_hz=(2.0, 2.0),
        analysis_width_hz=0.0,
        # source power 0では全binが白色雑音CBFへ帰着し、精度伝播だけを方式判定から分離できる。
        source_band_rms_power=0.0,
        noise_power_per_bin_re_input_rms2=0.01,
        precision=precision,
    )
    design = design_alignment_weights(config)
    assert design.weights["ebae"]["T2a"].dtype == complex_dtype
    approximation = approximate_frequency_weights_with_fir(
        design.weights["ebae"]["T2a"], tap_count=4
    )
    assert approximation.reconstructed_weights.dtype == complex_dtype
    assert approximation.energy_ratio.dtype == real_dtype

    # target beamのchannel別FIRを逐次部品へ渡すと、係数dtypeが計算dtypeとして維持される。
    target_index = 1
    impulse = np.fft.ifft(
        approximation.reconstructed_weights[:, target_index, :].conj(), axis=0
    ).T.astype(complex_dtype)
    taps = impulse[:, :4]
    signal = np.ones((3, 8), dtype=real_dtype)
    delayed = StatefulIntegerDelay(np.zeros(3, dtype=np.int64)).process(signal)
    filtered = VersionedCausalFIR(taps).process(SignalBlock(delayed.data, delayed.valid_mask))
    assert filtered.data.dtype == complex_dtype


def test_versioned_fir_rejects_precision_change_within_one_state_sequence() -> None:
    """履歴途中の暗黙な精度変更を拒否し、異なる系列の状態混在を防ぐ。"""
    taps = np.ones((1, 2), dtype=np.complex64)
    stage = VersionedCausalFIR(taps)
    data = np.ones((1, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="precision"):
        stage.process(SignalBlock(data, np.ones(data.shape, dtype=np.bool_)))
    with pytest.raises(ValueError, match="dtype"):
        stage.request_update(np.ones((1, 2), dtype=np.complex128), version=1)


def test_ebae_internal_arrays_preserve_single_precision() -> None:
    """EBAEが出力時だけでなく固有分解・MUSIC診断量でも単精度を維持する。"""
    covariance = np.eye(3, dtype=np.complex64)
    steering = np.asarray([[1.0, 1.0], [1.0, 1.0j], [1.0, -1.0]], dtype=np.complex64)
    result = design_ebae_weights_band(
        covariance,
        steering,
        snapshot_count=9,
        config=EbaeConfig(snapshot_rate_hz=9.0, integration_time_sec=1.0),
    )
    assert result.weights.dtype == np.dtype(np.complex64)
    assert result.eigenvectors.dtype == np.dtype(np.complex64)
    assert result.eigenvalues.dtype == np.dtype(np.float32)
    assert result.music_spectrum.dtype == np.dtype(np.float32)
