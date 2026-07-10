"""fixed-delay diff-MVDR のストリーミング block 同期試験。"""

from __future__ import annotations

import numpy as np
import pytest

from spflow.beamforming import (
    AlignedPathCombiner,
    CausalBlockFIR,
    DelayTable,
    DiffMVDRCorrectionPath,
    FixedDelayDiffMVDRStreamingProcessor,
    FractionalDelayMainPath,
    ProcessedBlock,
    StreamingBlock,
    design_standard_fractional_delay_filter_bank,
    make_directions,
)


def _streaming_block(
    *,
    data: np.ndarray,
    block_index: int,
    block_length: int,
    fs_hz: float,
    array_id: str = "array0",
) -> StreamingBlock:
    """テスト用の StreamingBlock を作る。

    data は `[n_ch, n_total]` の全体波形であり、ここから block_index に対応する
    `[n_ch, block_length]` 区間を切り出す。最後の不足 block はこのテストでは使わない。
    """
    start_sample = int(block_index) * int(block_length)
    block = data[:, start_sample : start_sample + int(block_length)]
    return StreamingBlock(
        array_id=array_id,
        block_index=int(block_index),
        start_sample=start_sample,
        length=int(block_length),
        fs_hz=float(fs_hz),
        data=block,
        valid_mask=np.ones(int(block_length), dtype=np.bool_),
    )


def test_causal_block_fir_matches_one_shot_across_impulse_boundaries() -> None:
    """履歴付き direct FIR が block 境界の impulse を欠落・重複させないことを確認する。

    impulse を `L-1`, `L`, `L+1` に置き、境界直前・境界上・境界直後の全てで
    一括処理と分割処理が一致することを検証する。
    """
    block_length = 8
    taps = np.array(
        [
            [1.0 + 0.0j, 0.25 + 0.0j, -0.125 + 0.0j],
            [0.5 + 0.0j, -0.25 + 0.0j, 0.125 + 0.0j],
        ],
        dtype=np.complex128,
    )
    signal = np.zeros((2, 3 * block_length), dtype=np.complex128)
    for sample_index in (block_length - 1, block_length, block_length + 1):
        signal[:, sample_index] = 1.0 + 0.0j

    one_shot = CausalBlockFIR(n_series=2, tap_length=3).process(signal, taps)
    streaming_fir = CausalBlockFIR(n_series=2, tap_length=3)
    streamed = np.concatenate(
        [
            streaming_fir.process(signal[:, 0:block_length], taps),
            streaming_fir.process(signal[:, block_length : 2 * block_length], taps),
            streaming_fir.process(signal[:, 2 * block_length :], taps),
        ],
        axis=1,
    )

    np.testing.assert_allclose(streamed, one_shot, atol=1.0e-12)


def test_aligned_path_combiner_rejects_latency_mismatch() -> None:
    """latency_tag が一致しない経路を暗黙に加算しないことを確認する。"""
    data = np.ones((1, 4), dtype=np.complex128)
    valid = np.ones(4, dtype=np.bool_)
    main = ProcessedBlock(
        array_id="array0",
        path_id="main",
        block_index=0,
        start_sample=0,
        length=4,
        fs_hz=32768.0,
        latency_tag="latency:a",
        coeff_version=1,
        data=data,
        valid_mask=valid,
    )
    diff = ProcessedBlock(
        array_id="array0",
        path_id="diff",
        block_index=0,
        start_sample=0,
        length=4,
        fs_hz=32768.0,
        latency_tag="latency:b",
        coeff_version=1,
        data=data,
        valid_mask=valid,
    )

    with pytest.raises(ValueError, match="latency_tag"):
        AlignedPathCombiner().add(main, diff)


def test_fractional_main_and_diff_paths_share_block_indices() -> None:
    """主経路と補正枝が同一 block の metadata を保って合成されることを確認する。"""
    fs_hz = 32768.0
    block_length = 16
    filter_bank = design_standard_fractional_delay_filter_bank()
    delay_table = DelayTable(
        arrival_delay_sec=np.zeros((1, 1), dtype=np.float64),
        steering_delay_sample=np.zeros((1, 1), dtype=np.float64),
        delay_int=np.zeros((1, 1), dtype=np.int64),
        delay_frac=np.zeros((1, 1), dtype=np.float64),
        frac_filter_index=np.full((1, 1), filter_bank.n_frac_filter // 2, dtype=np.int64),
    )
    signal = np.ones((1, block_length), dtype=np.float64)
    block = StreamingBlock(
        array_id="array0",
        block_index=3,
        start_sample=3 * block_length,
        length=block_length,
        fs_hz=fs_hz,
        data=signal,
        valid_mask=np.ones(block_length, dtype=np.bool_),
    )
    latency_tag = "fixed_delay_plus_frac_fir:G=63.5"
    main_path = FractionalDelayMainPath(
        delay_table=delay_table,
        fractional_filter_bank=filter_bank,
        fs_hz=fs_hz,
        array_id="array0",
        latency_tag=latency_tag,
        coeff_version=7,
    )
    correction_taps = np.zeros((1, 1, 1), dtype=np.complex128)
    correction_taps[0, 0, 0] = 0.125 + 0.0j
    diff_path = DiffMVDRCorrectionPath(
        correction_taps=correction_taps,
        fs_hz=fs_hz,
        array_id="array0",
        latency_tag=latency_tag,
        coeff_version=7,
    )

    result = FixedDelayDiffMVDRStreamingProcessor(
        main_path=main_path,
        correction_path=diff_path,
    ).process(block)

    assert result.block_index == block.block_index
    assert result.start_sample == block.start_sample
    assert result.length == block.length
    assert result.latency_tag == latency_tag
    assert result.coeff_version == 7
    assert result.data.shape == (1, block_length)


def test_three_second_streaming_peak_beam_waveform_has_no_block_discontinuity() -> None:
    """3秒のストリーミング最終出力でピーク方位波形が一括処理と一致することを確認する。

    4096 Hz tone を 2048 sample block に分割し、主経路とゼロ補正枝を同一
    StreamingBlock から処理する。ピーク方位の最終出力波形を一括処理と比較し、
    block 境界近傍に sample 欠落・重複・片経路 shift がないことを検証する。
    """
    fs_hz = 32768.0
    duration_s = 3.0
    n_sample = int(round(fs_hz * duration_s))
    block_length = 2048
    source_frequency_hz = 4096.0
    source_azimuth_deg = 60.0
    sound_speed_m_s = 1500.0
    array_id = "array0"
    latency_tag = "fixed_delay_plus_frac_fir:G=63.5"

    positions = np.column_stack(
        (
            np.linspace(-0.175, 0.175, 8, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
        )
    )
    directions, axis_azimuth_deg, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=19,
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    filter_bank = design_standard_fractional_delay_filter_bank()
    delay_table = DelayTable.from_geometry(
        array_pos_m=positions,
        dir_cos=directions.T.astype(np.float64),
        fs_hz=fs_hz,
        sound_speed_m_s=sound_speed_m_s,
        fractional_filter_bank=filter_bank,
    )

    source_direction = np.array(
        [
            np.cos(np.deg2rad(source_azimuth_deg)),
            np.sin(np.deg2rad(source_azimuth_deg)),
            0.0,
        ],
        dtype=np.float64,
    )
    tau_sec = positions @ source_direction / sound_speed_m_s
    time_sec = np.arange(n_sample, dtype=np.float64) / fs_hz
    signal = np.cos(
        2.0 * np.pi * source_frequency_hz * (time_sec[np.newaxis, :] - tau_sec[:, np.newaxis])
    )

    def make_processor() -> FixedDelayDiffMVDRStreamingProcessor:
        main_path = FractionalDelayMainPath(
            delay_table=delay_table,
            fractional_filter_bank=filter_bank,
            fs_hz=fs_hz,
            array_id=array_id,
            latency_tag=latency_tag,
            coeff_version=0,
        )
        correction_path = DiffMVDRCorrectionPath(
            correction_taps=np.zeros(
                (axis_azimuth_deg.size, positions.shape[0], 1), dtype=np.complex128
            ),
            fs_hz=fs_hz,
            array_id=array_id,
            latency_tag=latency_tag,
            coeff_version=0,
        )
        return FixedDelayDiffMVDRStreamingProcessor(
            main_path=main_path,
            correction_path=correction_path,
        )

    offline_block = StreamingBlock(
        array_id=array_id,
        block_index=0,
        start_sample=0,
        length=n_sample,
        fs_hz=fs_hz,
        data=signal,
        valid_mask=np.ones(n_sample, dtype=np.bool_),
    )
    offline_output = make_processor().process(offline_block).data

    streaming_processor = make_processor()
    streamed_blocks = [
        streaming_processor.process(
            _streaming_block(
                data=signal,
                block_index=block_index,
                block_length=block_length,
                fs_hz=fs_hz,
                array_id=array_id,
            )
        ).data
        for block_index in range(n_sample // block_length)
    ]
    streamed_output = np.concatenate(streamed_blocks, axis=1)

    steady_start = 512
    rms_by_beam = np.sqrt(np.mean(np.abs(offline_output[:, steady_start:]) ** 2, axis=1))
    peak_beam_index = int(np.argmax(rms_by_beam))
    # make_directions(array_side="right side") の表示方位は、ここで生成した array 方位に対して
    # displayed_az = 180 - array_az の関係になる。
    expected_display_azimuth_deg = 180.0 - source_azimuth_deg
    assert abs(float(axis_azimuth_deg[peak_beam_index]) - expected_display_azimuth_deg) <= 10.0

    peak_streamed = streamed_output[peak_beam_index]
    peak_offline = offline_output[peak_beam_index]
    np.testing.assert_allclose(peak_streamed, peak_offline, atol=1.0e-10)

    boundary_indices = np.arange(block_length, n_sample, block_length, dtype=np.int64)
    for boundary_index in boundary_indices:
        window = slice(int(boundary_index) - 2, int(boundary_index) + 3)
        np.testing.assert_allclose(peak_streamed[window], peak_offline[window], atol=1.0e-10)
