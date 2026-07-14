"""nonuniform leaf に関する回帰試験。"""

# 非均一分割は葉ごとのレート差と内部状態の持ち越しが複雑なため、
# 木構造・streaming・ビームフォーミング統合時の安全側挙動を回帰試験で固定する。

import numpy as np
import pytest

from spflow.beamforming.mvdr_filter import apply_beamformer_filter_fft, beam_response_rms_db
from spflow.filterbank.daubechies_nonuniform_beamformer import (
    DaubechiesNonuniformBeamformer,
    make_reference_dense_sparse_array_design,
)
from spflow.filterbank.formal_complex_pr_stage import FormalBandPacket
from spflow.filterbank.nonuniform_leaf import (
    NonuniformLeafProcessor,
    NonuniformLeafProcessorConfig,
    expand_one_sided_response,
    one_sided_bin_count,
    resample_frequency_response,
)
from spflow.filterbank.nonuniform_tree import NonuniformBandPacket, NonuniformTreeFilterBank
from spflow.frequency.overlap_save import OverlapSaveBuffer, ValidRegionExtractor


def _collect_packet_samples(packets: list[NonuniformBandPacket]) -> np.ndarray:
    """packet 列から時間サンプルを連結する。"""
    if not packets:
        return np.zeros((0, 0), dtype=np.complex64)
    return np.concatenate([packet.samples for packet in packets], axis=-1)



def _collect_formal_packet_samples(packets: list[FormalBandPacket]) -> np.ndarray:
    """formal packet 列から時間サンプルを連結する。"""
    if not packets:
        return np.zeros((0, 0), dtype=np.complex64)
    return np.concatenate([packet.complex_samples for packet in packets], axis=-1)



def _steering_vector_linear(
    positions_m: np.ndarray,
    frequency_hz: float,
    angle_deg: float,
    sound_speed: float = 340.0,
) -> np.ndarray:
    """直線アレイの narrowband steering を返す。"""
    theta = np.deg2rad(angle_deg)
    delay_s = np.asarray(positions_m, dtype=np.float32) * np.sin(theta) / sound_speed
    return np.exp(-1j * 2.0 * np.pi * frequency_hz * delay_s)[:, np.newaxis]



def test_resample_frequency_response_preserves_constant_response():
    """周波数応答の次数変換が定数応答を保つことを確認する。"""
    response = np.full((2, 3, 4), 1.5 - 0.25j, dtype=np.complex64)

    upsampled = resample_frequency_response(response, target_fft_size=16, axis=-1)
    downsampled = resample_frequency_response(response, target_fft_size=2, axis=-1)

    np.testing.assert_allclose(upsampled, 1.5 - 0.25j, atol=1e-6)
    np.testing.assert_allclose(downsampled, 1.5 - 0.25j, atol=1e-6)



def test_nonuniform_leaf_config_exposes_shared_fft_grid_and_rejects_legacy_mode():
    """nonuniform leaf 設定が shared FFT 条件だけを許可することを確認する。"""
    spec = NonuniformTreeFilterBank.default_for_fs().band_specs[0]

    config = NonuniformLeafProcessorConfig(
        spec=spec,
        used_channels=np.array([0, 1]),
        steering=np.ones((2, 1), dtype=np.complex64),
        long_fft_frame_size=4,
        long_fft_valid_size=2,
        short_fft_size=4,
        short_fft_hop_size=2,
        output_path_mode="leaf_independent_one_sided",
    )

    assert config.uses_shared_one_sided_output
    assert config.output_fft_size == 4
    assert config.output_hop_size == 2
    assert config.statistics_fft_size == 4
    assert config.statistics_hop_size == 2

    with pytest.raises(ValueError):
        NonuniformLeafProcessorConfig(
            spec=spec,
            used_channels=np.array([0, 1]),
            steering=np.ones((2, 1), dtype=np.complex64),
            long_fft_frame_size=8,
            long_fft_valid_size=4,
            short_fft_size=4,
            short_fft_hop_size=2,
            output_path_mode="leaf_independent_one_sided",
        )

    with pytest.raises(ValueError):
        NonuniformLeafProcessorConfig(
            spec=spec,
            used_channels=np.array([0, 1]),
            steering=np.ones((2, 1), dtype=np.complex64),
            long_fft_frame_size=4,
            long_fft_valid_size=2,
            short_fft_size=4,
            short_fft_hop_size=2,
            output_path_mode="full_overlap_save",
        )



def test_default_nonuniform_leaf_positive_bin_sum_is_1672():
    """既定 nonuniform leaf 構成で正側ビン合計が 1672 になることを確認する。"""
    beamformer = DaubechiesNonuniformBeamformer()
    total = sum(one_sided_bin_count(cfg.short_fft_size) for cfg in beamformer.leaf_configs.values())

    assert total == 1672



def test_nonuniform_leaf_processor_matches_direct_formal_ols_reference():
    """nonuniform leaf processor が自身の OLS filter FFT 参照実装と一致することを確認する。"""
    rng = np.random.default_rng(102)
    spec = NonuniformTreeFilterBank.default_for_fs().band_specs[0]
    x = rng.standard_normal((2, 19)) + 1j * rng.standard_normal((2, 19))

    config = NonuniformLeafProcessorConfig(
        spec=spec,
        used_channels=np.array([0, 1]),
        steering=np.ones((2, 1), dtype=np.complex64),
        long_fft_frame_size=4,
        long_fft_valid_size=2,
        short_fft_size=4,
        short_fft_hop_size=2,
        beamformer_mode="cbf",
        output_path_mode="leaf_independent_one_sided",
    )
    processor = NonuniformLeafProcessor(config)

    packets: list[NonuniformBandPacket] = []
    for start, stop in ((0, 3), (3, 8), (8, 13), (13, x.shape[-1])):
        packets.extend(processor.process(NonuniformBandPacket(spec=spec, samples=x[:, start:stop])))
    packets.extend(processor.flush())
    y_leaf = _collect_packet_samples(packets)

    reference_buffer = OverlapSaveBuffer(frame_size=4, valid_size=2, axis=-1)
    reference_extractor = ValidRegionExtractor(frame_size=4, valid_size=2, axis=-1)
    reference_outputs = []
    for start, stop in ((0, 3), (3, 8), (8, 13), (13, x.shape[-1])):
        for frame in reference_buffer.process(x[:, start:stop]):
            frame_fft = np.fft.fft(frame, n=4, axis=-1)
            filtered = apply_beamformer_filter_fft(frame_fft, processor.current_filter_fft)
            time_frame = np.fft.ifft(filtered, n=4, axis=-1)
            reference_outputs.append(reference_extractor.process(time_frame))
    for frame in reference_buffer.flush(pad=True, fill_value=0.0):
        frame_fft = np.fft.fft(frame, n=4, axis=-1)
        filtered = apply_beamformer_filter_fft(frame_fft, processor.current_filter_fft)
        time_frame = np.fft.ifft(filtered, n=4, axis=-1)
        reference_outputs.append(reference_extractor.process(time_frame))
    y_reference = np.concatenate(reference_outputs, axis=-1)

    assert processor.uses_shared_frame_fft
    assert processor.output_uses_one_sided_bins
    assert processor.output_inner_product_bin_count == 3
    np.testing.assert_allclose(y_leaf, y_reference, atol=1e-6)



def test_nonuniform_leaf_processor_mvdr_updates_weights_and_filter_fft():
    """nonuniform leaf processor で MVDR 更新が重みと OLS filter FFT に反映されることを確認する。"""
    spec = NonuniformTreeFilterBank.default_for_fs().band_specs[2]
    n = np.arange(32, dtype=np.float32)
    tone = np.exp(1j * 2.0 * np.pi * n / 4.0)
    x = np.stack([tone, 1j * tone], axis=0)

    config = NonuniformLeafProcessorConfig(
        spec=spec,
        used_channels=np.array([0, 1]),
        steering=np.ones((2, 1), dtype=np.complex64),
        long_fft_frame_size=4,
        long_fft_valid_size=2,
        short_fft_size=4,
        short_fft_hop_size=2,
        beamformer_mode="mvdr",
        integration_time=0.0,
        weight_update_period=0.0,
        diag_load=1e-6,
    )
    processor = NonuniformLeafProcessor(config)
    initial_weights = processor.current_weights_short.copy()
    initial_filter_fft = processor.current_filter_fft.copy()

    packets = processor.process(NonuniformBandPacket(spec=spec, samples=x))
    updated_weights = processor.current_weights_short

    assert packets
    assert np.max(np.abs(updated_weights - initial_weights)) > 1e-3
    assert np.max(np.abs(processor.current_covariances)) > 0.0
    assert processor.current_filter_fft.shape == (2, 1, 4)
    assert np.max(np.abs(processor.current_filter_fft - initial_filter_fft)) > 1e-3



def test_nonuniform_leaf_processor_mvdr_reduces_interferer_response_and_target_error():
    """nonuniform leaf processor で MVDR が妨害波応答と target 誤差を減らすことを確認する。"""
    band_idx = 4
    spec = NonuniformTreeFilterBank.default_for_fs().band_specs[band_idx]
    array_design = make_reference_dense_sparse_array_design()
    used = array_design.active_channel_indices(band_idx)

    fs_leaf = spec.nominal_sample_rate_hz
    n_sample = 16384
    n = np.arange(n_sample, dtype=np.float32)
    t = n / fs_leaf
    rng = np.random.default_rng(110)

    absolute_frequency_hz = spec.center_frequency_hz
    target_relative_frequency_hz = 0.5 * spec.bandwidth_hz
    interferer_relative_frequency_hz = target_relative_frequency_hz + 4.0

    block_size = 32
    n_block = int(np.ceil(n_sample / block_size))
    target_envelope = np.repeat(
        (rng.standard_normal(n_block) + 1j * rng.standard_normal(n_block)) / np.sqrt(2.0),
        block_size,
    )[:n_sample]
    interferer_envelope = np.repeat(
        (rng.standard_normal(n_block) + 1j * rng.standard_normal(n_block)) / np.sqrt(2.0),
        block_size,
    )[:n_sample]
    smooth = np.ones(9, dtype=np.float32) / 9.0
    target_envelope = np.convolve(target_envelope, smooth, mode="same")
    interferer_envelope = np.convolve(interferer_envelope, smooth, mode="same")

    target_source = target_envelope * np.exp(1j * 2.0 * np.pi * target_relative_frequency_hz * t)
    interferer_source = interferer_envelope * np.exp(1j * 2.0 * np.pi * interferer_relative_frequency_hz * t)

    full_target_steering = _steering_vector_linear(
        array_design.channel_positions_m,
        absolute_frequency_hz,
        angle_deg=10.0,
    )
    full_interferer_steering = _steering_vector_linear(
        array_design.channel_positions_m,
        absolute_frequency_hz,
        angle_deg=-25.0,
    )
    reduced_target_steering = full_target_steering[used]
    reduced_interferer_steering = full_interferer_steering[used]

    x = full_target_steering @ target_source[np.newaxis, :] + full_interferer_steering @ interferer_source[np.newaxis, :]

    cbf_processor = NonuniformLeafProcessor(
        NonuniformLeafProcessorConfig(
            spec=spec,
            used_channels=used,
            steering=full_target_steering,
            long_fft_frame_size=256,
            long_fft_valid_size=128,
            short_fft_size=256,
            short_fft_hop_size=128,
            output_path_mode="leaf_independent_one_sided",
            beamformer_mode="cbf",
        )
    )
    mvdr_processor = NonuniformLeafProcessor(
        NonuniformLeafProcessorConfig(
            spec=spec,
            used_channels=used,
            steering=full_target_steering,
            long_fft_frame_size=256,
            long_fft_valid_size=128,
            short_fft_size=256,
            short_fft_hop_size=128,
            output_path_mode="leaf_independent_one_sided",
            beamformer_mode="mvdr",
            integration_time=0.5,
            weight_update_period=0.0,
            diag_load=1e-3,
        )
    )

    cbf_packets: list[NonuniformBandPacket] = []
    mvdr_packets: list[NonuniformBandPacket] = []
    for start in range(0, x.shape[-1], 111):
        packet = NonuniformBandPacket(spec=spec, samples=x[:, start : start + 111])
        cbf_packets.extend(cbf_processor.process(packet))
        mvdr_packets.extend(mvdr_processor.process(packet))
    cbf_packets.extend(cbf_processor.flush())
    mvdr_packets.extend(mvdr_processor.flush())

    y_cbf = _collect_packet_samples(cbf_packets)[0, :n_sample]
    y_mvdr = _collect_packet_samples(mvdr_packets)[0, :n_sample]
    cbf_error = float(np.sqrt(np.mean(np.abs(y_cbf - target_source) ** 2)))
    mvdr_error = float(np.sqrt(np.mean(np.abs(y_mvdr - target_source) ** 2)))

    target_bin = int(round(target_relative_frequency_hz / fs_leaf * 256))
    weights = mvdr_processor.current_weights_short[:, 0, target_bin]
    target_response = weights @ reduced_target_steering[:, 0]
    cbf_weights = reduced_target_steering[:, 0] / reduced_target_steering.shape[0]
    cbf_interferer_response_db = beam_response_rms_db(cbf_weights.conj() @ reduced_interferer_steering[:, 0])
    mvdr_interferer_response_db = beam_response_rms_db(weights @ reduced_interferer_steering[:, 0])

    np.testing.assert_allclose(target_response, 1.0 + 0.0j, atol=1e-6)
    assert mvdr_interferer_response_db <= cbf_interferer_response_db - 1.5
    assert mvdr_error < cbf_error



def test_nonuniform_leaf_processor_formal_packets_preserve_time_origin_and_delay():
    """nonuniform leaf processor で formal packet の time origin と delay を保持することを確認する。"""
    rng = np.random.default_rng(101)
    spec = NonuniformTreeFilterBank.default_for_fs().band_specs[0]
    x = rng.standard_normal((1, 23)) + 1j * rng.standard_normal((1, 23))

    config = NonuniformLeafProcessorConfig(
        spec=spec,
        used_channels=np.array([0]),
        steering=np.ones((1, 1), dtype=np.complex64),
        long_fft_frame_size=4,
        long_fft_valid_size=2,
        short_fft_size=4,
        short_fft_hop_size=2,
        beamformer_mode="cbf",
        output_path_mode="leaf_independent_one_sided",
    )
    processor = NonuniformLeafProcessor(config)
    step = 1 << spec.tree_depth
    delay = 17
    time0 = 100

    packets = []
    first = FormalBandPacket(
        band_id=spec.band_id,
        f_low_hz=spec.f_low_hz,
        f_high_hz=spec.f_high_hz,
        sample_rate_hz=spec.nominal_sample_rate_hz,
        time_origin_at_root_rate=time0,
        delay_samples_at_root_rate=delay,
        complex_samples=x[:, :9],
    )
    second = FormalBandPacket(
        band_id=spec.band_id,
        f_low_hz=spec.f_low_hz,
        f_high_hz=spec.f_high_hz,
        sample_rate_hz=spec.nominal_sample_rate_hz,
        time_origin_at_root_rate=time0 + 9 * step,
        delay_samples_at_root_rate=delay,
        complex_samples=x[:, 9:],
    )
    packets.extend(processor.process_formal_packet(first))
    packets.extend(processor.process_formal_packet(second))
    packets.extend(processor.flush_formal())

    y = _collect_formal_packet_samples(packets)[..., : x.shape[-1]]
    np.testing.assert_allclose(y, x, atol=1e-6)
    assert packets
    assert packets[0].time_origin_at_root_rate == time0
    assert all(packet.delay_samples_at_root_rate == delay for packet in packets)
    for left, right in zip(packets[:-1], packets[1:], strict=True):
        expected = left.time_origin_at_root_rate + left.complex_samples.shape[-1] * step
        assert right.time_origin_at_root_rate == expected
