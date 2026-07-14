"""double buffer callback に関する回帰試験。"""

# 逐次更新部品は境界条件の取り扱いが不具合源になりやすいため、
# バッファ残量やスケジューリング順序が崩れないことを小さな入力で固定する。

import numpy as np
from numpy.typing import NDArray

from spflow import DoubleBufferCallback, Flow, StepScheduler


class SumCallback(DoubleBufferCallback[dict[str, list[int]], int, NDArray[np.int64]]):
    """試験用の callback 実装を表す。"""

    def signature(self, inputs: dict[str, list[int]]) -> tuple[int, ...]:
        """`signature` を実行する。"""
        return tuple(inputs["items"])

    def make_initial_output(self, inputs: dict[str, list[int]]) -> NDArray[np.int64]:
        """`make_initial_output` で必要なオブジェクトや入力を構成する。"""
        return np.zeros(1, dtype=np.int64)

    def make_work_buffer(self, inputs: dict[str, list[int]]) -> NDArray[np.int64]:
        """`make_work_buffer` で必要なオブジェクトや入力を構成する。"""
        return np.zeros(1, dtype=np.int64)

    def make_items(self, inputs: dict[str, list[int]]) -> list[int]:
        """`make_items` で必要なオブジェクトや入力を構成する。"""
        return inputs["items"]

    def update_item(self, item: int, inputs: dict[str, list[int]]) -> None:
        """`update_item` を実行する。"""
        work = self.work
        if work is None:
            raise RuntimeError("work buffer was not created.")
        work[0] += item


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
    previous = callback.prev
    if previous is None:
        raise AssertionError("completed output must exist")
    previous[0] = 999

    np.testing.assert_array_equal(out, np.array([3]))


def test_flow_can_propagate_only_newly_completed_scheduler_output():
    """Flowが未完成周期を落とし、更新済み完成値だけを後段へ運べることを確認する。"""
    scheduler = StepScheduler(SumCallback(), items_per_cycle=1)
    inputs = {"items": [1, 2]}

    first_outputs = (
        Flow.from_value(inputs)
        .map(scheduler.process_result)
        .map(lambda result: result.updated_value())
        .to_list()
    )
    completed_outputs = (
        Flow.from_value(inputs)
        .map(scheduler.process_result)
        .map(lambda result: result.updated_value())
        .to_list()
    )

    # 未完成周期はNoneとなってFlowで0出力、完成周期だけが1出力になる。
    assert first_outputs == []
    assert len(completed_outputs) == 1
    np.testing.assert_array_equal(completed_outputs[0], np.array([3]))
