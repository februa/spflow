"""Flow、FrameBuffer、beamformerの逐次処理契約を結合して検証する。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow import Flow, FrameBuffer
from spflow.beamforming import CBFBeamformer, CBFOverlapSaveBeamformer


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
