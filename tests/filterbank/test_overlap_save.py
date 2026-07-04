"""overlap save に関する回帰試験。"""

# フィルタバンク試験では完全再構成誤差、群遅延、設計制約の維持を同時に見たいので、
# 数値安定性を崩しやすい代表条件を回帰ケースとして明示的に残す。

import numpy as np

from spflow.frequency import OverlapSaveBuffer, ValidRegionExtractor, make_filter_fft


def test_overlap_save_buffer_prepends_previous_block_history():
    """overlap-save バッファが前ブロックの履歴を前置することを確認する。"""
    buffer = OverlapSaveBuffer(frame_size=8, valid_size=4, axis=-1)

    frames_1 = buffer.process(np.array([1, 2, 3, 4]))
    frames_2 = buffer.process(np.array([5, 6, 7, 8]))

    assert len(frames_1) == 1
    assert len(frames_2) == 1
    np.testing.assert_array_equal(frames_1[0], np.array([0, 0, 0, 0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(frames_2[0], np.array([1, 2, 3, 4, 5, 6, 7, 8]))


def test_overlap_save_identity_filter_returns_valid_region():
    """恒等 overlap-save フィルタについて valid 領域を返す を確認する。"""
    buffer = OverlapSaveBuffer(frame_size=8, valid_size=4, axis=-1)
    valid = ValidRegionExtractor(frame_size=8, valid_size=4, axis=-1)
    identity_fft = make_filter_fft(np.array([1.0 + 0.0j]), frame_size=8)

    out = []
    for chunk in (np.array([1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])):
        for frame in buffer.process(chunk):
            frame_fft = np.fft.fft(frame)
            filtered = np.fft.ifft(frame_fft * identity_fft)
            out.append(valid.process(filtered))

    flushed = buffer.flush(pad=True, fill_value=0.0)
    for frame in flushed:
        frame_fft = np.fft.fft(frame)
        filtered = np.fft.ifft(frame_fft * identity_fft)
        out.append(valid.process(filtered))

    y = np.concatenate(out)
    np.testing.assert_allclose(np.real(y[:8]), np.arange(1, 9), atol=1e-6)
