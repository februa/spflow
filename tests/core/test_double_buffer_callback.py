"""double buffer callback に関する回帰試験。"""

import numpy as np

from spflow import DoubleBufferCallback, StepScheduler


class SumCallback(DoubleBufferCallback):
    """試験用の callback 実装を表す。"""
    def signature(self, inputs):
        """`signature` を実行する。"""
        return tuple(inputs["items"])

    def make_initial_output(self, inputs):
        """`make_initial_output` で必要なオブジェクトや入力を構成する。"""
        return np.zeros(1, dtype=np.int64)

    def make_work_buffer(self, inputs):
        """`make_work_buffer` で必要なオブジェクトや入力を構成する。"""
        return np.zeros_like(self.prev)

    def make_items(self, inputs):
        """`make_items` で必要なオブジェクトや入力を構成する。"""
        return inputs["items"]

    def update_item(self, item, inputs):
        """`update_item` を実行する。"""
        self.work[0] += item


def test_double_buffer_callback_publishes_only_when_done():
    """double buffer callbackが完了時だけ publish することを確認する。"""
    callback = SumCallback()
    scheduler = StepScheduler(callback=callback, items_per_cycle=2)

    out1 = scheduler.process({"items": [1, 2, 3]})
    out2 = scheduler.process({"items": [1, 2, 3]})

    np.testing.assert_array_equal(out1, np.array([0]))
    np.testing.assert_array_equal(out2, np.array([6]))


def test_double_buffer_callback_restarts_on_signature_change():
    """double buffer callbackが signature 変更時に再始動することを確認する。"""
    callback = SumCallback()
    scheduler = StepScheduler(callback=callback, items_per_cycle=1)

    out1 = scheduler.process({"items": [1, 2]})
    out2 = scheduler.process({"items": [10, 20]})
    out3 = scheduler.process({"items": [10, 20]})

    np.testing.assert_array_equal(out1, np.array([0]))
    np.testing.assert_array_equal(out2, np.array([0]))
    np.testing.assert_array_equal(out3, np.array([30]))


def test_double_buffer_callback_publish_copies_work():
    """double buffer callbackが作業バッファをコピーして publish することを確認する。"""
    callback = SumCallback()
    scheduler = StepScheduler(callback=callback, items_per_cycle=None)

    out = scheduler.process({"items": [1, 2]})
    callback.prev[0] = 999

    np.testing.assert_array_equal(out, np.array([3]))
