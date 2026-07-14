"""Flow、FrameBuffer、beamformerの逐次処理契約を結合して検証する。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from examples.beamforming.streaming_mvdr_weights import (
    calculate_frame_fft,
    make_environment,
    process_cycle,
)
from spflow import Flow, FrameBuffer, StepScheduler
from spflow.beamforming import (
    CBFBeamformer,
    CBFOverlapSaveBeamformer,
    MVDRWeightCallback,
    MVDRWeightSnapshot,
    apply_beamformer_bands,
)


def _make_identical_channel_chunk(values: list[float]) -> NDArray[np.float32]:
    """同相CBFで振幅保存を検証する2チャネル入力を作る。

    Args:
        values: 時間順の振幅値。単位は任意の線形振幅単位。

    Returns:
        2チャネルへ同じ値を配置した配列。shapeは`[n_channel=2, n_sample]`。
    """
    samples = np.asarray([values], dtype=np.float32)
    return np.repeat(samples, 2, axis=0)


def _process_cbf_chunk(
    chunk: NDArray[np.float32],
    frame_buffer: FrameBuffer,
    beamformer: CBFBeamformer,
) -> list[Any]:
    """1 chunkをframe化し、完成frameだけを固定CBFへ渡す。

    Args:
        chunk: shape`[n_channel=2, n_sample]`の線形振幅入力。
        frame_buffer: axis=1を時間軸とするフレームバッファ。
        beamformer: 2チャネル入力用の固定CBF。

    Returns:
        shape`[n_beam=1, frame_size]`の完成ビーム出力列。

    Raises:
        ValueError: channel数または非時間軸shapeが契約と異なる場合。
    """
    return Flow.from_value(chunk).map(frame_buffer.process).map(beamformer.process).to_list()


def test_flow_frame_buffer_cbf_publishes_only_completed_frames_and_flushes_tail() -> None:
    """未完成frameを公開せず、終端端数だけをflushで完成させることを確認する。"""
    # steering shape: [n_channel=2, n_beam=1]。
    # 同相入力に対する係数は各1/2なので、CBF後も入力振幅が保存される。
    beamformer = CBFBeamformer(np.ones((2, 1), dtype=np.complex64))
    frame_buffer = FrameBuffer(frame_size=4, hop_size=4, axis=1)

    first_outputs = _process_cbf_chunk(
        _make_identical_channel_chunk([1.0, 2.0, 3.0]),
        frame_buffer,
        beamformer,
    )
    completed_outputs = _process_cbf_chunk(
        _make_identical_channel_chunk([4.0, 5.0, 6.0]),
        frame_buffer,
        beamformer,
    )

    # 3 sample時点はframe_size=4へ届かないため、Flowには0個の完成値だけが見える。
    assert first_outputs == []
    assert len(completed_outputs) == 1
    np.testing.assert_allclose(completed_outputs[0], [[1.0, 2.0, 3.0, 4.0]])

    # 終端の2 sampleは不完全値のまま公開せず、ゼロ詰め済み完成frameとして後段へ渡す。
    final_outputs = Flow.many(frame_buffer.flush(pad=True)).map(beamformer.process).to_list()
    assert len(final_outputs) == 1
    np.testing.assert_allclose(final_outputs[0], [[5.0, 6.0, 0.0, 0.0]])


def test_flow_flattens_multiple_completed_frames_before_cbf_application() -> None:
    """1 chunkから完成した複数frameを同じCBF処理へ順番に渡すことを確認する。"""
    beamformer = CBFBeamformer(np.ones((2, 1), dtype=np.complex64))
    frame_buffer = FrameBuffer(frame_size=4, hop_size=4, axis=1)

    outputs = _process_cbf_chunk(
        _make_identical_channel_chunk([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        frame_buffer,
        beamformer,
    )

    assert len(outputs) == 2
    np.testing.assert_allclose(outputs[0], [[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(outputs[1], [[5.0, 6.0, 7.0, 8.0]])


def test_flow_preserves_overlap_save_band_record_as_one_semantic_value() -> None:
    """帯域番号と有効区間のtupleを分解せず、帯域ごとの複数出力を運ぶことを確認する。"""
    # steering shape: [n_channel=1, n_beam=1, n_band=2]。
    # 1チャネル単位係数により、ここではoverlap-saveの入出力契約だけを検証する。
    beamformer = CBFOverlapSaveBeamformer(
        np.ones((1, 1, 2), dtype=np.complex64),
        frame_size=4,
        valid_size=2,
    )
    first_sample = np.ones((1, 2, 1), dtype=np.complex64)
    second_sample = np.ones((1, 2, 1), dtype=np.complex64)

    pending = Flow.from_value(first_sample).map(beamformer.process).to_list()
    records = Flow.from_value(second_sample).map(beamformer.process).to_list()

    assert pending == []
    assert [band_index for band_index, _ in records] == [0, 1]
    assert all(valid_block.shape == (1, 2) for _, valid_block in records)


def test_flow_scheduler_mvdr_uses_safe_fallback_then_publishes_completed_update() -> None:
    """Flow接続時も未完成MVDRを公開せず、固定CBFから完成係数へ切り替えることを確認する。"""
    # steering shape: [n_ch=2, n_beam=1, n_band=2]。
    # source_snapshotは各bandでsteeringどおり到来する単位振幅sourceを表す。
    steering = np.array(
        [
            [[1.0 + 0.0j, 1.0 + 0.0j]],
            [[1.0 + 0.0j, 0.0 + 1.0j]],
        ],
        dtype=np.complex64,
    )
    covariance = np.repeat(np.eye(2, dtype=np.complex64)[None, :, :], 2, axis=0)
    snapshot = MVDRWeightSnapshot(
        covariance=covariance,
        steering=steering,
        generation="covariance-0",
    )
    scheduler = StepScheduler(MVDRWeightCallback(diag_load=0.0), items_per_cycle=1)

    first_result = Flow.from_value(snapshot).map(scheduler.process_result).to_list()[0]
    first_output = apply_beamformer_bands(steering[:, 0, :], first_result.value)
    first_updates = (
        Flow.from_value(first_result).map(lambda result: result.updated_value()).to_list()
    )

    completed_result = Flow.from_value(snapshot).map(scheduler.process_result).to_list()[0]
    completed_output = apply_beamformer_bands(steering[:, 0, :], completed_result.value)
    completed_updates = (
        Flow.from_value(completed_result).map(lambda result: result.updated_value()).to_list()
    )

    # 初回CBFと完成MVDRのどちらもh^T a=1であり、入力source振幅を保存する。
    np.testing.assert_allclose(first_output, np.ones((1, 2)), atol=1e-6)
    np.testing.assert_allclose(completed_output, np.ones((1, 2)), atol=1e-6)
    assert first_updates == []
    assert len(completed_updates) == 1


def test_streaming_mvdr_flow_applies_one_cycle_design_to_same_frame() -> None:
    """1周期で設計完了する場合、同じFFT frameへ新しい適応係数を適用することを確認する。"""
    env = make_environment(fft_length=2, items_per_cycle=None)
    signal_chunk = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    outputs = process_cycle(signal_chunk, env)

    assert env.active_coefficient_generation == 0
    assert env.coefficient_replacement_count == 1
    assert len(outputs) == 1
    frame_fft = calculate_frame_fft(signal_chunk, fft_length=2)
    np.testing.assert_allclose(
        outputs[0],
        apply_beamformer_bands(frame_fft, env.active_beamformer_coefficients),
    )
    for band_index in range(env.steering.shape[2]):
        response = (
            env.active_beamformer_coefficients[:, :, band_index].T @ env.steering[:, :, band_index]
        )
        np.testing.assert_allclose(response, np.ones((1, 1)), atol=1e-6)


def test_streaming_mvdr_flow_keeps_previous_coefficients_until_multicycle_design_finishes() -> None:
    """複数周期設計中は固定CBFを使い、完成周期のFFT frameから適応係数へ切り替える。"""
    env = make_environment(fft_length=2, items_per_cycle=1)
    initial_coefficients = env.active_beamformer_coefficients.copy()
    first_chunk = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    second_chunk = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    first_outputs = process_cycle(first_chunk, env)
    assert env.active_coefficient_generation is None
    assert env.coefficient_replacement_count == 0
    np.testing.assert_allclose(
        first_outputs[0],
        apply_beamformer_bands(
            calculate_frame_fft(first_chunk, fft_length=2),
            initial_coefficients,
        ),
    )

    second_outputs = process_cycle(second_chunk, env)
    assert env.active_coefficient_generation == 0
    assert env.coefficient_replacement_count == 1
    np.testing.assert_allclose(
        second_outputs[0],
        apply_beamformer_bands(
            calculate_frame_fft(second_chunk, fft_length=2),
            env.active_beamformer_coefficients,
        ),
    )
    # generation 0の設計中に到着したgeneration 1は、次の設計候補として最新1件だけ保持する。
    assert env.waiting_snapshot is not None
    assert env.waiting_snapshot.generation == 1


def test_streaming_mvdr_flow_processes_multiple_completed_frames_in_temporal_order() -> None:
    """1 chunkの複数frameを、frameごとの係数更新と信号適用の順序を保って処理する。"""
    env = make_environment(fft_length=2, items_per_cycle=1)
    initial_coefficients = env.active_beamformer_coefficients.copy()
    signal_chunk = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 0.25]],
        dtype=np.float32,
    )

    outputs = process_cycle(signal_chunk, env)

    # frame 0では2 band中1 bandしか設計されないため、初期CBFを適用する。
    # frame 1でgeneration 0が完成し、同じframeから完成MVDRへ切り替える。
    assert len(outputs) == 2
    assert env.active_coefficient_generation == 0
    assert env.coefficient_replacement_count == 1
    first_frame_fft = calculate_frame_fft(signal_chunk[:, :2], fft_length=2)
    second_frame_fft = calculate_frame_fft(signal_chunk[:, 2:], fft_length=2)
    np.testing.assert_allclose(
        outputs[0],
        apply_beamformer_bands(first_frame_fft, initial_coefficients),
    )
    np.testing.assert_allclose(
        outputs[1],
        apply_beamformer_bands(second_frame_fft, env.active_beamformer_coefficients),
    )
    # 2番目のframeから作ったgeneration 1は、generation 0と混ぜず次回設計まで待機する。
    assert env.waiting_snapshot is not None
    assert env.waiting_snapshot.generation == 1
