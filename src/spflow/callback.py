"""spflow.callback を実装するモジュール。"""

from __future__ import annotations

from typing import Any


class DoubleBufferCallback:
    """Callback base that publishes only completed work values."""

    def __init__(self) -> None:
        self.prev: Any = None
        self.work: Any = None

    def on_start(self, inputs: Any) -> list[Any]:
        self._ensure_initialized(inputs)
        self.work = self.make_work_buffer(inputs)
        return list(self.make_items(inputs))

    def on_step(self, item: Any, inputs: Any) -> None:
        if self.work is None:
            self._ensure_initialized(inputs)
            self.work = self.make_work_buffer(inputs)
        self.update_item(item, inputs)

    def on_finish(self, inputs: Any, done: bool) -> Any:
        self._ensure_initialized(inputs)
        if done:
            self.prev = self.publish(self.work)
        return self.publish(self.prev)

    def reset_cycle(self) -> None:
        self.work = None

    def signature(self, inputs: Any) -> Any:
        return None

    def publish(self, work: Any) -> Any:
        if hasattr(work, "copy"):
            return work.copy()
        return work

    def make_initial_output(self, inputs: Any) -> Any:
        raise NotImplementedError

    def make_work_buffer(self, inputs: Any) -> Any:
        raise NotImplementedError

    def make_items(self, inputs: Any):
        raise NotImplementedError

    def update_item(self, item: Any, inputs: Any) -> None:
        raise NotImplementedError

    def _ensure_initialized(self, inputs: Any) -> None:
        if self.prev is None:
            self.prev = self.make_initial_output(inputs)
