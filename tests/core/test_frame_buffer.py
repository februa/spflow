"""frame buffer に関する回帰試験。"""

# コア部品は境界条件の取り扱いが不具合源になりやすいため、
# バッファ残量やスケジューリング順序が崩れないことを小さな入力で固定する。

import numpy as np
import pytest

from spflow import FrameBuffer


def test_frame_buffer_emits_overlapped_frames():
    """frame bufferがオーバーラップした frame を出力することを確認する。"""
    buffer = FrameBuffer(frame_size=4, hop_size=2, axis=-1)

    out1 = buffer.process(np.array([0, 1, 2]))
    out2 = buffer.process(np.array([3, 4, 5]))

    assert out1 == []
    assert len(out2) == 2
    np.testing.assert_array_equal(out2[0], np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(out2[1], np.array([2, 3, 4, 5]))


def test_frame_buffer_preserves_axis_order():
    """frame bufferが軸順序を保つことを確認する。"""
    buffer = FrameBuffer(frame_size=3, hop_size=2, axis=-1)
    x = np.arange(10).reshape(2, 5)

    frames = buffer.process(x)

    assert len(frames) == 2
    assert frames[0].shape == (2, 3)
    np.testing.assert_array_equal(frames[0], x[:, :3])
    np.testing.assert_array_equal(frames[1], x[:, 2:5])


def test_frame_buffer_flush_without_pad_drops_remainder():
    """frame bufferで pad なし flush 時に余りを破棄することを確認する。"""
    buffer = FrameBuffer(frame_size=4, hop_size=2)
    buffer.process(np.array([1, 2, 3]))

    assert buffer.flush(pad=False) == []


def test_frame_buffer_flush_with_pad_returns_single_frame_only_when_remainder_exists():
    """frame bufferで pad あり flush 時に余りがあるときだけ 1 frame を返すことを確認する。"""
    buffer = FrameBuffer(frame_size=4, hop_size=2)
    buffer.process(np.array([1, 2, 3]))

    frames = buffer.flush(pad=True, fill_value=0)

    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], np.array([1, 2, 3, 0]))
    assert buffer.flush(pad=True) == []


def test_frame_buffer_rejects_shape_mismatch_except_axis():
    """frame bufferが対象軸以外の shape 不一致を拒否することを確認する。"""
    buffer = FrameBuffer(frame_size=2, hop_size=1, axis=-1)
    buffer.process(np.zeros((2, 3)))

    with pytest.raises(ValueError, match="Input shape mismatch"):
        buffer.process(np.zeros((3, 3)))
