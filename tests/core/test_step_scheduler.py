"""step scheduler に関する回帰試験。"""

# コア部品は境界条件の取り扱いが不具合源になりやすいため、
# バッファ残量やスケジューリング順序が崩れないことを小さな入力で固定する。

from spflow import StepScheduler


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

