"""step scheduler に関する回帰試験。"""

# 分割実行部品は境界条件の取り扱いが不具合源になりやすいため、
# バッファ残量やスケジューリング順序が崩れないことを小さな入力で固定する。

from typing import TypedDict

from examples.streaming.step_scheduler_completion import (
    IncrementalSumCallback,
    SumSnapshot,
    select_newly_completed_value,
)
from spflow import Flow, StepScheduler


def test_step_scheduler_map_reduces_results():
    """step schedulerについて `map` による結果の畳み込みを確認する を確認する。"""
    out = StepScheduler.map(
        items=range(3),
        func=lambda item, inputs: item + inputs["bias"],
        inputs={"bias": 10},
        reducer=lambda results, inputs: sum(results) + inputs["bias"],
    )

    assert out == 43


class RecorderCallback:
    """試験用の callback 実装を表す。"""

    def __init__(self):
        """`__init__` を実行する。"""
        self.events = []

    def signature(self, inputs):
        """`signature` を実行する。"""
        return inputs.get("signature")

    def on_start(self, inputs):
        """`on_start` を実行する。"""
        self.events.append(("start", inputs["signature"]))
        return inputs["items"]

    def on_step(self, item, inputs):
        """`on_step` を実行する。"""
        self.events.append(("step", item))
        if item == "boom":
            raise RuntimeError("failure")

    def on_finish(self, inputs, done):
        """`on_finish` を実行する。"""
        self.events.append(("finish", done))
        return {"done": done}

    def reset_cycle(self):
        """`reset_cycle` を実行する。"""
        self.events.append(("reset", None))


def test_step_scheduler_processes_in_chunks():
    """step schedulerがチャンク単位で処理することを確認する。"""
    callback = RecorderCallback()
    scheduler = StepScheduler(callback=callback, items_per_cycle=2)

    out1 = scheduler.process({"items": [1, 2, 3], "signature": "a"})
    out2 = scheduler.process({"items": [1, 2, 3], "signature": "a"})

    assert out1 == {"done": False}
    assert out2 == {"done": True}
    assert callback.events == [
        ("start", "a"),
        ("step", 1),
        ("step", 2),
        ("finish", False),
        ("step", 3),
        ("finish", True),
        ("reset", None),
    ]


def test_step_scheduler_restarts_when_signature_changes():
    """step schedulerについて signature 変更時に再始動する を確認する。"""
    callback = RecorderCallback()
    scheduler = StepScheduler(callback=callback, items_per_cycle=1)

    scheduler.process({"items": [1, 2], "signature": "a"})
    scheduler.process({"items": [10, 20], "signature": "b"})

    assert callback.events == [
        ("start", "a"),
        ("step", 1),
        ("finish", False),
        ("reset", None),
        ("start", "b"),
        ("step", 10),
        ("finish", False),
    ]


def test_step_scheduler_resets_on_callback_error():
    """step schedulerが callback エラー時に状態をリセットすることを確認する。"""
    callback = RecorderCallback()
    scheduler = StepScheduler(callback=callback, items_per_cycle=None)

    try:
        scheduler.process({"items": ["boom"], "signature": "a"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")

    assert callback.events == [
        ("start", "a"),
        ("step", "boom"),
        ("reset", None),
    ]


class SnapshotInput(TypedDict):
    """snapshot固定試験で使う入力型。"""

    generation: str
    value: int


class SnapshotRecorderCallback:
    """周期開始時の入力snapshotだけが使われることを記録するcallback。"""

    def __init__(self) -> None:
        self.values: list[int] = []

    def signature(self, inputs: SnapshotInput) -> str:
        """明示されたgenerationを返す。"""
        return str(inputs["generation"])

    def on_start(self, inputs: SnapshotInput) -> list[int]:
        """2周期に分割するitem列を返す。"""
        return [0, 1]

    def on_step(self, item: int, inputs: SnapshotInput) -> None:
        """実際に参照したsnapshot値を記録する。"""
        self.values.append(int(inputs["value"]))

    def on_finish(self, inputs: SnapshotInput, done: bool) -> tuple[int, ...]:
        """記録済み値を固定型で返す。"""
        return tuple(self.values)


def test_step_scheduler_keeps_starting_snapshot_for_same_generation():
    """同じgenerationの後続objectを混ぜず、開始時snapshotだけで完了することを確認する。"""
    callback = SnapshotRecorderCallback()
    scheduler = StepScheduler(callback, items_per_cycle=1)

    first_result = scheduler.process_result({"generation": "a", "value": 10})
    completed_result = scheduler.process_result({"generation": "a", "value": 999})

    assert first_result.updated is False
    assert completed_result.updated is True
    assert completed_result.value == (10, 10)


def test_step_scheduler_example_separates_active_value_from_completion_event():
    """毎周期使う完成値と、新規完成時だけ流す通知を混同しないことを確認する。"""
    scheduler = StepScheduler(IncrementalSumCallback(), items_per_cycle=1)
    snapshot = SumSnapshot(values=(2, 4, 6), generation=0)

    results = [scheduler.process_result(snapshot) for _ in range(3)]
    completion_events = [
        Flow.from_value(result).map(select_newly_completed_value).to_list()
        for result in results
    ]

    # 最初の2回は未完成なので安全側の0を維持し、3回目だけ完成値12へ一括更新する。
    assert [result.value for result in results] == [0, 0, 12]
    assert completion_events == [[], [], [12]]
