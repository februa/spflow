"""spflow.scheduler を実装するモジュール。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class StepScheduler:
    """Scheduler for iterative item processing."""

    def __init__(self, callback, items_per_cycle: int | None = None) -> None:
        if items_per_cycle is not None and items_per_cycle <= 0:
            raise ValueError("items_per_cycle must be positive or None.")
        self.callback = callback
        self.items_per_cycle = items_per_cycle
        self._items: list[Any] = []
        self._cursor = 0
        self._active = False
        self._signature = None

    def process(self, inputs: Any) -> Any:
        signature = self.callback.signature(inputs)
        if self._active and signature != self._signature:
            self._reset_cycle()

        if not self._active:
            self._items = list(self.callback.on_start(inputs))
            self._cursor = 0
            self._active = True
            self._signature = signature

        limit = len(self._items) if self.items_per_cycle is None else min(
            self._cursor + self.items_per_cycle,
            len(self._items),
        )

        try:
            while self._cursor < limit:
                item = self._items[self._cursor]
                self.callback.on_step(item, inputs)
                self._cursor += 1
        except Exception:
            self._reset_cycle()
            raise

        done = self._cursor >= len(self._items)
        try:
            output = self.callback.on_finish(inputs, done)
        except Exception:
            self._reset_cycle()
            raise

        if done:
            self._reset_cycle()
        return output

    @staticmethod
    def map(items: Iterable[Any], func, inputs: Any, reducer):
        results = [func(item, inputs) for item in items]
        return reducer(results, inputs)

    def _reset_cycle(self) -> None:
        self._items = []
        self._cursor = 0
        self._active = False
        self._signature = None
        if hasattr(self.callback, "reset_cycle"):
            self.callback.reset_cycle()
